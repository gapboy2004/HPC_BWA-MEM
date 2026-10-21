#!/bin/bash
#SBATCH --job-name=sw-ncu
#SBATCH --qos=short                # <= 1 วัน
#SBATCH --gres=gpu:1               # ncu ต้องการ GPU ตัวเดียว แต่ควรได้ครองคนเดียว
##SBATCH --gres=gpu:v100:1         # ถ้าคลัสเตอร์ตั้ง GRES แยกรุ่น ให้ใช้บรรทัดนี้แทนบรรทัดบน
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G                  # profiling ใช้ MAX_TASKS น้อย เลยไม่ต้อง 60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# =============================================================================
# รัน Nsight Compute (ncu) บน V100 ผ่าน Singularity
#
#   sbatch --export=ALL,DATA_DIR=/path/to/fasta ncu_v100.sh
#
# งานนี้ "ไม่" ใช้ bench_inner.sh เพราะวัตถุประสงค์ต่างกันคนละเรื่อง:
#   bench_inner.sh  = วัดเวลา หลายขนาด หลายรอบ -> summary.csv
#   สคริปต์นี้       = profile kernel launch เดียว ให้ได้ counter ระดับ hardware
# =============================================================================

# ---------- config (override ตอน submit ได้) ----------
export BACKENDS=${BACKENDS:-"cuda sycl"}

# MAX_TASKS แค่ 1M พอ! ขนาด grid ที่ ncu เห็นถูกกำหนดโดย SW_CHUNK ไม่ใช่ MAX_TASKS
# ใช้ 1M แล้ว job เบา โหลดไฟล์เร็ว แต่ได้ grid เท่าเดิมกับที่เคยวัดบน 3070 Ti
export MAX_TASKS=${MAX_TASKS:-1000000}

# ★ สองค่านี้ต้องตรงกับรอบ 3070 Ti เป๊ะ ไม่งั้นเทียบข้ามเครื่องไม่ได้
#   (รอบนั้นได้ Grid 3,907 x Block 128 = 6.78 waves/SM)
export SW_CHUNK=${SW_CHUNK:-500000}
export SW_BLOCK=${SW_BLOCK:-128}
export N_GPU=${N_GPU:-1}

NCU_SET=${NCU_SET:-full}           # full = รวม SourceCounters (ต้องมี -lineinfo ใน binary)
BUILD=${BUILD:-1}                  # 1 = build ใหม่ใน container ก่อน profile

BIN_CUDA=${BIN_CUDA:-./sw_cuda}
BIN_SYCL=${BIN_SYCL:-./sw_sycl}
KERNEL_CUDA=${KERNEL_CUDA:-sw_batch_kernel}
KERNEL_SYCL=${KERNEL_SYCL:-regex:__acpp_sscp_kernel}

SIF=${SIF:-$HOME/hpc-bwa-mem.sif}
DATA_DIR=${DATA_DIR:-$HOME/data}

set -uo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
  RUN_ID=$SLURM_JOB_ID
else
  RUN_ID=manual_$(date +%Y%m%d_%H%M%S)
fi

SING=$(command -v singularity || command -v apptainer || true)
[ -z "$SING" ]      && { echo "ไม่พบ singularity/apptainer บน node นี้"; exit 1; }
[ ! -f "$SIF" ]     && { echo "ไม่พบ image: $SIF"; exit 1; }
[ ! -d "$DATA_DIR" ] && { echo "ไม่พบ DATA_DIR: $DATA_DIR (ส่งมาตอน submit)"; exit 1; }

OUTDIR=results/ncu_${RUN_ID}
mkdir -p "$OUTDIR" logs

NODE=$(hostname)
echo "node: $NODE   run: $RUN_ID   image: $SIF"
echo "config: MAX_TASKS=$MAX_TASKS SW_CHUNK=$SW_CHUNK SW_BLOCK=$SW_BLOCK set=$NCU_SET"
echo

# ---------- helper: รันคำสั่งใน container ----------
sing() {
  "$SING" exec --nv \
    --bind "$PWD":/workspace --pwd /workspace \
    --bind "$DATA_DIR":/data \
    --env ACPP_APPDB_DIR=/workspace/.acpp_cache \
    "$SIF" bash -lc "$1"
}

# ---------- 0) GPU ที่ได้มา ใช้คนเดียวจริงไหม ----------
# ถ้ามี process อื่นอยู่บน GPU เดียวกัน counter จะเพี้ยน ผลที่ได้จะเทียบกับ 3070 Ti ไม่ได้
echo "--- GPU ที่ Slurm จัดให้ ---"
sing 'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv'
echo

# ---------- 1) มี ncu ใน image ไหม ----------
if ! sing 'command -v ncu >/dev/null 2>&1'; then
  cat <<'MSG'
ERROR: ไม่พบ ncu ใน image

image ตระกูล nvidia/cuda:*-devel ส่วนใหญ่ไม่ได้ใส่ Nsight Compute มาให้
ต้องเติมใน Dockerfile แล้ว build + แปลงเป็น .sif ใหม่:

    RUN apt-get update && apt-get install -y --no-install-recommends \
          cuda-nsight-compute-12-4 \
     && rm -rf /var/lib/apt/lists/*

(เปลี่ยนเลขท้ายให้ตรงกับ CUDA ใน image)
MSG
  exit 1
fi
sing 'ncu --version | head -3'
echo

# ---------- 2) probe สิทธิ์ก่อน เสีย 15 วินาที ดีกว่าเสีย 30 นาที ----------
# Singularity เพิ่ม CAP_SYS_ADMIN ให้ไม่ได้ (ไม่เหมือน docker --cap-add) ถ้า host
# ยัง restrict profiling อยู่ ก็จะติดตรงนี้ และแก้เองไม่ได้ ต้องขอ admin
echo "--- probe สิทธิ์ profiling ---"
PROBE=$(sing "MAX_TASKS=200000 SW_CHUNK=100000 SW_BLOCK=$SW_BLOCK N_GPU=1 \
              ncu --metrics sm__cycles_elapsed.avg \
                  --kernel-name $KERNEL_CUDA $BIN_CUDA 2>&1" )
echo "$PROBE" | tail -20

if echo "$PROBE" | grep -q 'ERR_NVGPUCTRPERM'; then
  cat <<'MSG'

=============================================================================
ERROR: ERR_NVGPUCTRPERM -- ไม่มีสิทธิ์อ่าน GPU performance counter

Singularity/Apptainer เพิ่ม capability ให้ไม่ได้ถ้าไม่ได้รันโหมด setuid
ดังนั้นแก้ฝั่งคุณไม่ได้เลย ต้องขอ admin ของคลัสเตอร์ตั้งค่านี้บน compute node:

    # /etc/modprobe.d/nvidia-profiler.conf
    options nvidia NVreg_RestrictProfilingToAdminUsers=0
    # แล้ว update-initramfs -u && reboot

ระหว่างรอ: ncu บน 3070 Ti ที่บ้านทำได้อยู่แล้ว (docker --cap-add=SYS_ADMIN)
ถ้า admin ไม่อนุมัติ ทางสำรองคือ nsys (ไม่ต้องขอสิทธิ์) แต่ได้แค่ timeline
ไม่ได้ register/occupancy ซึ่งเป็นตัวที่ต้องการจริง
=============================================================================
MSG
  exit 2
fi
echo "สิทธิ์ผ่าน -> ไปต่อ"
echo

# ---------- 3) build ใหม่ (ต้องมี -lineinfo ถึงจะได้ SourceCounters) ----------
if [ "$BUILD" = "1" ]; then
  echo "--- build ---"
  # ★ แก้สองบรรทัดนี้ให้ตรงกับ target จริงของคุณ ถ้าชื่อไม่ตรง
  sing 'make -f Makefile.cuda clean && make -f Makefile.cuda sw_cuda' || exit 1
  sing 'make -f Makefile.sycl clean && make -f Makefile.sycl sw_sycl' || exit 1
  echo
fi

# ---------- 4) profile ----------
for be in $BACKENDS; do
  case $be in
    cuda) BIN=$BIN_CUDA; KNAME=$KERNEL_CUDA ;;
    sycl) BIN=$BIN_SYCL; KNAME=$KERNEL_SYCL ;;
    *) echo "backend ไม่รู้จัก: $be"; continue ;;
  esac

  TAG="${be}_v100_b${SW_BLOCK}_c${SW_CHUNK}"
  REP="$OUTDIR/${TAG}"

  echo "=============================================================="
  echo "profiling $be  ->  $REP.ncu-rep"
  echo "=============================================================="

  sing "MAX_TASKS=$MAX_TASKS SW_CHUNK=$SW_CHUNK SW_BLOCK=$SW_BLOCK N_GPU=$N_GPU \
        ncu --set $NCU_SET \
            --launch-count 1 \
            --kernel-name $KNAME \
            --force-overwrite -o $REP \
            $BIN" 2>&1 | tee "$OUTDIR/${TAG}.runlog.txt"

  if [ -f "$REP.ncu-rep" ]; then
    # -o ทำให้ ncu ไม่พิมพ์ details ออก stdout เลยต้อง import กลับมาเป็น txt
    sing "ncu --import $REP.ncu-rep --page details"  > "$OUTDIR/${TAG}.details.txt"
    sing "ncu --import $REP.ncu-rep --page source --print-source sass" \
         > "$OUTDIR/${TAG}.source.txt" 2>/dev/null || true
    sing "ncu --import $REP.ncu-rep --csv --page details" \
         > "$OUTDIR/${TAG}.details.csv" 2>/dev/null || true
    echo "ok: $(ls -1 $OUTDIR/${TAG}* | wc -l) ไฟล์"
  else
    echo "WARNING: ไม่ได้ไฟล์ report ของ $be -- ดู $OUTDIR/${TAG}.runlog.txt"
  fi
  echo
done

# ---------- 5) สรุปตัวเลขที่ต้องเอาไปเทียบกับ 3070 Ti ----------
echo "=============================================================="
echo "ตัวเลขที่ต้องอ่าน (เทียบกับ 3070 Ti: CUDA 39 reg/92% occ, SYCL 54 reg/70% occ)"
echo "=============================================================="
for f in "$OUTDIR"/*.details.txt; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f") ---"
  grep -E 'Registers Per Thread|Theoretical Occupancy|Achieved Occupancy|Block Size|Grid Size|Waves Per SM|Compute \(SM\) Throughput|Memory Throughput|Duration' "$f"
  echo
done

echo "Done. ผลอยู่ใน $OUTDIR"
echo "ดึงกลับเครื่อง:  rsync -av <user>@<cluster>:$PWD/$OUTDIR ./"