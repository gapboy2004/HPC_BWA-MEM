#!/bin/bash
#SBATCH --job-name=sw-bench
#SBATCH --nodelist=ai3            # ai3 มี 4 GPU (ai2 มีแค่ 2)
#SBATCH --gres=gpu:4              # ขอให้เท่ากับจำนวน GPU สูงสุดใน GPU_COUNTS
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# ==== ตั้งค่า (เปลี่ยนตอน submit ได้ด้วย --export=ALL,ชื่อ=ค่า) ====
export BACKENDS=${BACKENDS:-"cuda sycl"}     # รันทั้งสองตัว หรือเลือกตัวเดียว
export GPU_COUNTS=${GPU_COUNTS:-"1 2 4"}
export SIZES=${SIZES:-"500000 1000000"}
export REPEATS=${REPEATS:-3}
export BUILD=${BUILD:-1}                     # 1 = build ใหม่ใน container ก่อนรัน
export ARCH=${ARCH:-sm_70}                   # V100 (เครื่องตัวเองใช้ ARCH=sm_86 ฯลฯ)
export ACPP_TARGETS=${ACPP_TARGETS:-cuda:$ARCH}
IMAGE=${IMAGE:-my-sycl:v1}

set -uo pipefail

# รันผ่าน sbatch → ใช้ค่าจาก Slurm
# รันด้วย bash เองบน Ubuntu (ไม่มี Slurm) → ใช้โฟลเดอร์ปัจจุบันและ GPU ทุกตัว
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
  RUN_ID=$SLURM_JOB_ID
  # docker daemon อยู่นอกการควบคุมของ Slurm จึงเห็น GPU ทุกตัวในเครื่อง
  # ต้องส่งเฉพาะ GPU ที่ Slurm จัดให้ (index จริง) แทน all
  PHYS_GPUS=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
  if [ -z "$PHYS_GPUS" ]; then
    echo "No GPUs allocated by Slurm"; exit 1
  fi
else
  RUN_ID=manual_$(date +%Y%m%d_%H%M%S)
  PHYS_GPUS=${GPUS:-all}             # เลือก GPU เองได้ เช่น GPUS=0,1
fi
echo "GPUs for container: $PHYS_GPUS"

export OUTDIR=results/${RUN_ID}
mkdir -p "$OUTDIR" logs

# ไม่ใช้ -it เพื่อให้รันได้ทั้งใน sbatch (ไม่มี terminal) และรันเอง
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$PHYS_GPUS" \
  -e BACKENDS -e GPU_COUNTS -e SIZES -e REPEATS -e BUILD \
  -e ARCH -e ACPP_TARGETS -e OUTDIR \
  -e JOB_ID="$RUN_ID" -e NODE="$(hostname)" -e PHYS_GPUS="$PHYS_GPUS" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -e ACPP_APPDB_DIR=/workspace/.acpp_cache \
  -v "$PWD":/workspace \
  -w /workspace \
  "$IMAGE" bash bench_inner.sh

echo "Done. Results in $OUTDIR"
