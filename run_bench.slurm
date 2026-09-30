#!/bin/bash
#SBATCH --job-name=sw-bench
#SBATCH --nodelist=ai3            # ai3 มี 4 GPU (ai2 มีแค่ 2)
#SBATCH --gres=gpu:4              # ขอให้เท่ากับจำนวน GPU สูงสุดใน GPU_COUNTS
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# ==== ตั้งค่า (เปลี่ยนตอน submit ได้ด้วย --export=ALL,ชื่อ=ค่า) ====
export BACKENDS=${BACKENDS:-"cuda sycl"}     # รันทั้งสองตัว หรือเลือกตัวเดียว
export GPU_COUNTS=${GPU_COUNTS:-"1"}
# export GPU_COUNTS=${GPU_COUNTS:-"1 2 4"}
export SIZES=${SIZES:-"10000000 30000000 50000000 70000000 100000000"}
export REPEATS=${REPEATS:-3}
export BUILD=${BUILD:-1}                     # 1 = build ใหม่ใน container ก่อนรัน
# flag เพิ่มตอน build (ว่าง = build เหมือน make run ทุกอย่าง)
export CUDA_MAKE_ARGS=${CUDA_MAKE_ARGS:-}
export SYCL_MAKE_ARGS=${SYCL_MAKE_ARGS:-}

SIF=${SIF:-$HOME/hpc-bwa-mem.sif}            # singularity (cluster)
IMAGE=${IMAGE:-gapboy/hpc-bwa-mem:v1.1}               # docker (เครื่องตัวเอง)

set -uo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
  RUN_ID=$SLURM_JOB_ID
else
  RUN_ID=manual_$(date +%Y%m%d_%H%M%S)
fi

# ==== เลือก runtime: มี singularity + ไฟล์ .sif → singularity, ไม่อย่างนั้น docker ====
SING=$(command -v singularity || command -v apptainer || true)
if [ -z "${RUNTIME:-}" ]; then
  if [ -n "$SING" ] && [ -f "$SIF" ]; then RUNTIME=singularity; else RUNTIME=docker; fi
fi

export OUTDIR=results/${RUN_ID}
export JOB_ID=$RUN_ID
export NODE=$(hostname)
mkdir -p "$OUTDIR" logs
echo "runtime: $RUNTIME   node: $NODE   run: $RUN_ID"

case $RUNTIME in
  singularity)
    # singularity รันเป็น process ธรรมดาใน job → เห็นเฉพาะ GPU ที่ Slurm จัดให้
    # (CUDA_VISIBLE_DEVICES จาก Slurm ส่งต่อเข้า container อัตโนมัติ)
    export PHYS_GPUS=${CUDA_VISIBLE_DEVICES:-all}
    echo "image: $SIF   GPUs: $PHYS_GPUS"
    "$SING" exec --nv \
      --bind "$PWD":/workspace --pwd /workspace \
      "$SIF" bash bench_inner.sh
    ;;

  docker)
    # docker daemon อยู่นอกการควบคุมของ Slurm จึงเห็น GPU ทุกตัวในเครื่อง
    # ต้องส่งเฉพาะ GPU ที่ Slurm จัดให้ (index จริง) แทน all
    if [ -n "${SLURM_JOB_ID:-}" ]; then
      PHYS_GPUS=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
      [ -z "$PHYS_GPUS" ] && { echo "No GPUs allocated by Slurm"; exit 1; }
    else
      PHYS_GPUS=${GPUS:-all}           # เลือก GPU เองได้ เช่น GPUS=0,1
    fi
    echo "image: $IMAGE   GPUs: $PHYS_GPUS"
    # ไม่ใช้ -it เพื่อให้รันได้ทั้งใน sbatch (ไม่มี terminal) และรันเอง
    docker run --rm \
      --runtime=nvidia \
      -e NVIDIA_VISIBLE_DEVICES="$PHYS_GPUS" \
      -e BACKENDS -e GPU_COUNTS -e SIZES -e REPEATS -e BUILD \
      -e CUDA_MAKE_ARGS -e SYCL_MAKE_ARGS -e OUTDIR -e JOB_ID -e NODE \
      -e PHYS_GPUS="$PHYS_GPUS" \
      -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
      -e ACPP_APPDB_DIR=/workspace/.acpp_cache \
      -v "$PWD":/workspace -w /workspace \
      -v "/home/gapboy/Desktop":/data  \
      "$IMAGE" bash bench_inner.sh
    ;;

  *) echo "Unknown RUNTIME: $RUNTIME (use singularity or docker)"; exit 1 ;;
esac

echo "Done. Results in $OUTDIR"
