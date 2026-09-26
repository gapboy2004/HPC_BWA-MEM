#!/bin/bash
#SBATCH --job-name=sw-bench
#SBATCH --nodelist=ai3            # ai3 มี 4 GPU (ai2 มีแค่ 2)
#SBATCH --gres=gpu:4              # ขอให้เท่ากับจำนวน GPU สูงสุดใน GPU_COUNTS
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# ==== ตั้งค่า ====
# เลือก backend ตอน submit ได้:  sbatch --export=ALL,BACKEND=cuda run_bench.slurm
BACKEND=${BACKEND:-sycl}
IMAGE=${IMAGE:-my-bwa-sw:latest}  # ★ ชื่อ docker image ของคุณ
GPU_COUNTS=(1 2 4)
SIZES=(500000 1000000)            # เพิ่มได้ เช่น 2000000 3000000 4000000 5000000
REPEATS=3

case "$BACKEND" in
  sycl) BIN=./sw_sycl ;;
  cuda) BIN=./main_cuda ;;
  *) echo "Unknown BACKEND: $BACKEND"; exit 1 ;;
esac

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

# ==== GPU ที่ Slurm จัดให้ (ต้องเป็น index จริงของเครื่อง) ====
# docker daemon อยู่นอก cgroup ของ Slurm จึงมองเห็น GPU ทุกตัวของเครื่อง
# ต้องส่ง index จริงให้ docker เอง ไม่อย่างนั้นอาจไปแย่ง GPU ของ job คนอื่น
PHYS_GPUS=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
IFS=',' read -r -a ALLOC_GPUS <<< "$PHYS_GPUS"
N_ALLOC=${#ALLOC_GPUS[@]}
echo "Slurm allocated $N_ALLOC GPU(s), physical ids: $PHYS_GPUS"

# ==== โฟลเดอร์ผลลัพธ์ ====
OUTDIR=results/${BACKEND}_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"
CSV=$OUTDIR/summary.csv

# cache ของ AdaptiveCpp JIT เก็บไว้นอก container
# (ไม่อย่างนั้น --rm ลบทิ้งทุกรอบ และทุกรอบจะ JIT ใหม่)
mkdir -p .acpp_cache

# ==== รันโปรแกรมใน container ด้วย GPU ที่กำหนด ====
run_in_docker() {
  local ids=$1; shift
  docker run --rm \
    --gpus "\"device=${ids}\"" \
    --user "$(id -u):$(id -g)" \
    -v "$PWD":/work -w /work \
    -e ACPP_APPDB_DIR=/work/.acpp_cache \
    "$IMAGE" "$@"
}

first_k_gpus() {
  local k=$1
  (IFS=','; echo "${ALLOC_GPUS[*]:0:$k}")
}

# ==== บันทึกสภาพแวดล้อม ====
{
  echo "job_id:     $SLURM_JOB_ID"
  echo "backend:    $BACKEND"
  echo "binary:     $BIN"
  echo "image:      $IMAGE ($(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null | cut -c8-19))"
  echo "node:       $(hostname)"
  echo "allocated:  $N_ALLOC GPU(s) [physical: $PHYS_GPUS]"
  echo "gpu_counts: ${GPU_COUNTS[*]}"
  echo "sizes:      ${SIZES[*]}"
  echo "repeats:    $REPEATS"
  echo "date:       $(date '+%Y-%m-%d %H:%M:%S')"
  echo "git:        $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo
  run_in_docker "$PHYS_GPUS" nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
} > "$OUTDIR/env.txt" 2>&1

echo "backend,node,n_gpus,gpu_ids,size,repeat,wall_sec,exit_code" > "$CSV"

for ngpu in "${GPU_COUNTS[@]}"; do
  if [ "$ngpu" -gt "$N_ALLOC" ]; then
    echo "!! skip n_gpus=$ngpu (only $N_ALLOC allocated)"
    continue
  fi

  ids=$(first_k_gpus "$ngpu")
  echo "=== n_gpus=$ngpu  (physical GPU ids: $ids) ==="

  # warm-up ต่อชุด GPU ไม่นับผล
  run_in_docker "$ids" "$BIN" "${SIZES[0]}" > "$OUTDIR/warmup_g${ngpu}.log" 2>&1

  for size in "${SIZES[@]}"; do
    for r in $(seq 1 "$REPEATS"); do
      log=$OUTDIR/g${ngpu}_n${size}_r${r}.log
      echo "Running $BIN $size  gpus=$ngpu  repeat $r/$REPEATS"

      start=$(date +%s.%N)
      run_in_docker "$ids" "$BIN" "$size" > "$log" 2>&1
      rc=$?
      end=$(date +%s.%N)

      wall=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e-s}')
      echo "$BACKEND,$(hostname),$ngpu,\"$ids\",$size,$r,$wall,$rc" >> "$CSV"

      [ "$rc" -ne 0 ] && echo "  !! exit code $rc, see $log"
    done
  done
done

echo "Done. Results in $OUTDIR"
column -s, -t "$CSV" 2>/dev/null || cat "$CSV"
