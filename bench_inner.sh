#!/bin/bash
# รันภายใน container (ถูกเรียกจาก run_bench.slurm)
# ทำสิ่งเดียวกับที่ปกติพิมพ์เองหลังเข้า container: ตั้ง env → build → รัน

set -uo pipefail

# script นี้ต้องรันใน container (บน host ให้ใช้: bash run_bench.slurm)
if [ ! -f /.dockerenv ] && [ -z "${FORCE_HOST:-}" ]; then
  echo "bench_inner.sh ต้องรันใน docker container"
  echo "บน host ให้ใช้:  bash run_bench.slurm"
  exit 1
fi
# ==== ค่า default (ใช้ตอนเทสด้วยมือ; Slurm script จะส่งค่ามาทับ) ====
: "${BACKENDS:=cuda sycl}"
: "${GPU_COUNTS:=1}"
: "${SIZES:=500000 1000000}"
: "${REPEATS:=1}"
: "${BUILD:=1}"
: "${JOB_ID:=manual_$(date +%Y%m%d_%H%M%S)}"
: "${OUTDIR:=results/$JOB_ID}"
: "${NODE:=$(hostname)}"
: "${PHYS_GPUS:=${NVIDIA_VISIBLE_DEVICES:-all}}"
: "${HOST_UID:=$(stat -c %u .)}"      # เจ้าของโฟลเดอร์ที่ mount เข้ามา
: "${HOST_GID:=$(stat -c %g .)}"
: "${ACPP_APPDB_DIR:=$PWD/.acpp_cache}"; export ACPP_APPDB_DIR

export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}

mkdir -p "$OUTDIR" .acpp_cache
CSV=$OUTDIR/summary.csv

# GPU ใน container ถูกเรียงเลขใหม่เป็น 0..N-1
N_ALLOC=$(nvidia-smi -L | grep -c '^GPU')

# ==== ชื่อโปรแกรม = ค่า TARGET ใน Makefile (ตัวเดียวกับที่ make run ใช้) ====
get_target() {
  make -s --no-print-directory -f "$1" \
       --eval='__print_target: ; @echo $(TARGET)' __print_target
}
CUDA_BIN=./$(get_target Makefile.cuda)
SYCL_BIN=./$(get_target Makefile.sycl)
echo "CUDA binary: $CUDA_BIN   SYCL binary: $SYCL_BIN"

if ! grep -q 'USE_SYCL' main.c; then
  echo "!! main.c ยังไม่มี #ifdef USE_SYCL: build ได้แค่ backend ที่ main.c เรียกอยู่"
fi

# ==== build: เหมือน make run แต่ยังไม่รัน ====
# ส่ง flag เพิ่มได้ เช่น CUDA_MAKE_ARGS='NVFLAGS=-O2 -arch=sm_70'
if [ "$BUILD" = "1" ]; then
  for b in $BACKENDS; do
    echo "=== build $b ==="
    case $b in
      cuda)
        make -f Makefile.cuda clean >/dev/null
        rm -f main.o                        # กันใช้ main.o ที่ compile ไว้ให้ SYCL
        make -f Makefile.cuda ${CUDA_MAKE_ARGS:+"$CUDA_MAKE_ARGS"} "${CUDA_BIN#./}" || exit 1
        ;;
      sycl)
        make -f Makefile.sycl clean >/dev/null
        rm -f main.o
        # Makefile.sycl ยังไม่ใส่ -DUSE_SYCL ให้ main.c → compile main.o ไว้ก่อน
        if grep -q 'USE_SYCL' main.c && ! grep -q 'USE_SYCL' Makefile.sycl; then
          gcc -O2 -DUSE_SYCL -c main.c -o main.o || exit 1
        fi
        make -f Makefile.sycl ${SYCL_MAKE_ARGS:+"$SYCL_MAKE_ARGS"} "${SYCL_BIN#./}" || exit 1
        ;;
    esac
  done
fi

# ==== บันทึกสภาพแวดล้อม ====
{
  echo "job_id:       $JOB_ID"
  echo "node:         $NODE"
  echo "physical gpu: $PHYS_GPUS"
  echo "backends:     $BACKENDS"
  echo "gpu_counts:   $GPU_COUNTS"
  echo "sizes:        $SIZES"
  echo "repeats:      $REPEATS"
  echo "make args:    cuda='${CUDA_MAKE_ARGS:-}' sycl='${SYCL_MAKE_ARGS:-}'"
  echo "date:         $(date '+%Y-%m-%d %H:%M:%S')"
  echo "git:          $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  echo
  nvcc --version | tail -1
  acpp --version 2>/dev/null | head -1
} > "$OUTDIR/env.txt" 2>&1

echo "backend,node,n_gpus,gpus_used,max_tasks,tasks,total_sec,gcups,wall_sec,exit_code" > "$CSV"

# ดึงค่าที่โปรแกรมพิมพ์เองจาก log
#   "using 1 / 1 GPU"  "total: 0.273 sec  (20000 tasks, ...)"  "GCUPS: 0.31"
parse_log() {
  awk '
    /^using [0-9]+ \/ [0-9]+ GPU/ { used=$2 }
    /^total: /                     { sec=$2; t=$4; gsub(/\(/,"",t); tasks=t }
    /^GCUPS:/                      { g=$2 }
    END { printf "%s,%s,%s,%s", used, tasks, sec, g }
  ' "$1"
}

for b in $BACKENDS; do
  case $b in
    cuda) BIN=$CUDA_BIN ;;
    sycl) BIN=$SYCL_BIN ;;
    *) echo "Unknown backend $b"; continue ;;
  esac

  for ngpu in $GPU_COUNTS; do
    if [ "$ngpu" -gt "$N_ALLOC" ]; then
      echo "!! skip $b n_gpus=$ngpu (only $N_ALLOC GPU)"
      continue
    fi

    # ให้โปรแกรมเห็นแค่ ngpu ตัวแรก: 0 / 0,1 / 0,1,2,3
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu - 1)))
    export N_GPU=$ngpu                   # โปรแกรมอ่านจำนวน GPU จาก env นี้
    echo "=== $b  n_gpus=$ngpu  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ==="

    # warm-up ไม่นับผล
    first_size=${SIZES%% *}
    $BIN "$first_size" > "$OUTDIR/${b}_g${ngpu}_warmup.log" 2>&1

    for size in $SIZES; do
      for r in $(seq 1 "$REPEATS"); do
        log=$OUTDIR/${b}_g${ngpu}_n${size}_r${r}.log
        echo "  N_GPU=$ngpu $BIN $size  repeat $r/$REPEATS"

        start=$(date +%s.%N)
        $BIN "$size" > "$log" 2>&1
        rc=$?
        end=$(date +%s.%N)

        wall=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e-s}')
        IFS=, read -r used tasks sec g <<< "$(parse_log "$log")"
        echo "$b,$NODE,$ngpu,$used,$size,$tasks,$sec,$g,$wall,$rc" >> "$CSV"

        # เตือนถ้าโปรแกรมใช้ GPU ไม่ตรงกับที่สั่ง หรือได้ task น้อยกว่าที่ขอ
        [ -n "$used" ] && [ "$used" != "$ngpu" ] && echo "  !! asked $ngpu GPU, program used $used"
        [ -n "$tasks" ] && [ "$tasks" -lt "$size" ] && echo "  !! asked $size tasks, input has only $tasks"
        [ "$rc" -ne 0 ] && echo "  !! exit code $rc, see $log"
      done
    done
  done
done

column -s, -t "$CSV" 2>/dev/null || cat "$CSV"

# container รันเป็น root → คืนสิทธิ์ไฟล์ให้ user เดิม
chown -R "$HOST_UID:$HOST_GID" "$OUTDIR" .acpp_cache \
  "$CUDA_BIN" "$SYCL_BIN" *.o *.a 2>/dev/null || true