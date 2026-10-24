#!/bin/bash
#SBATCH --job-name=sw-bench
#SBATCH -p fullq                  # br1: dgx-03 = A100 เต็มตัว 8 ตัว (gpuq/kuresearch เป็น MIG)
#SBATCH --qos=short               # ≤ 1 วัน
#SBATCH --gres=gpu:4              # ต้อง ≥ ค่าสูงสุดใน GPU_COUNTS
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                # ไม่ใส่จะได้แค่ 512M (OOM); รวม cache ของไฟล์ input ด้วย
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# ==== ตั้งค่า (เปลี่ยนตอน submit ได้ด้วย --export=ALL,ชื่อ=ค่า) ====
export BACKENDS=${BACKENDS:-"cuda sycl"}     # รันทั้งสองตัว หรือเลือกตัวเดียว
export GPU_COUNTS=${GPU_COUNTS:-"1 2 4"}

# export SIZES=${SIZES:-"10000000 30000000 50000000 70000000 100000000"}
export SIZES=${SIZES:-"30000000"}

# block size ที่จะทดสอบ (วนทีละค่า แล้วส่งให้โปรแกรมเป็น SW_BLOCK)
# 0 = ใช้ default (128) ของ kernel, >0 = block size ใหม่
export SW_BLOCKS=${SW_BLOCKS:-"128"}

# chunk size ที่จะทดสอบ (วนทีละค่า แล้วส่งให้โปรแกรมเป็น env ชื่อ $CHUNK_VAR)
# 0 = ใช้ default ของโปรแกรม
export SW_CHUNKS=${SW_CHUNKS:-"0"}
export CHUNK_VAR=${CHUNK_VAR:-SW_CHUNK}     # แก้ให้ตรงกับชื่อที่โปรแกรมอ่านด้วย getenv()

export REPEATS=${REPEATS:-3}
export BUILD=${BUILD:-1}                     # 1 = build ใหม่ใน container ก่อนรัน
# flag เพิ่มตอน build (ว่าง = build เหมือน make run ทุกอย่าง)
export CUDA_MAKE_ARGS=${CUDA_MAKE_ARGS:-}
export SYCL_MAKE_ARGS=${SYCL_MAKE_ARGS:-}

SIF=${SIF:-$HOME/hpc-bwa-mem.sif}            # apptainer/singularity: ไฟล์ .sif หรือโฟลเดอร์ sandbox
IMAGE=${IMAGE:-gapboy/hpc-bwa-mem:v1.1}      # docker (เครื่องตัวเอง)

# โฟลเดอร์ข้อมูล input → เห็นใน container ที่ /data
# เครื่องตัวเองใช้ Desktop, บน cluster ส่ง DATA_DIR=/path/ข้อมูล มาเอง
DATA_DIR=${DATA_DIR:-/home/g6917550019/HPC_BWA-MEM}

# ==== บาง cluster ต้อง module load ก่อนถึงจะมีคำสั่ง container (br1: apptainer/apptainer) ====
# ทำก่อน set -u เพราะสคริปต์ของระบบ module มักใช้ตัวแปรที่ยังไม่ได้ตั้ง
CONTAINER_MODULE=${CONTAINER_MODULE:-apptainer/apptainer}
if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
      [ -f "$f" ] && source "$f" && break
    done
  fi
  type module >/dev/null 2>&1 && module load "$CONTAINER_MODULE" 2>/dev/null
fi

set -uo pipefail

# ==== หาโฟลเดอร์โปรเจกต์ (ที่มี bench_inner.sh) ====
# บาง cluster (เช่น br1) ตั้ง SLURM_SUBMIT_DIR เป็น /scratch/... ไม่ใช่โฟลเดอร์ที่สั่ง sbatch
# จึงลองหลายที่ และเลือกที่มี bench_inner.sh จริง
for d in "${PROJECT_DIR:-}" "$PWD" "${SLURM_SUBMIT_DIR:-}"; do
  if [ -n "$d" ] && [ -f "$d/bench_inner.sh" ]; then cd "$d"; break; fi
done
if [ ! -f bench_inner.sh ]; then
  echo "cannot find bench_inner.sh (PWD=$PWD, SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-})"
  echo "ส่ง PROJECT_DIR=/path/ไปยัง/HPC_BWA-MEM มาด้วย"
  exit 1
fi
echo "project: $PWD"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  RUN_ID=$SLURM_JOB_ID
else
  RUN_ID=manual_$(date +%Y%m%d_%H%M%S)
fi

# ==== เลือก runtime: มี singularity + ไฟล์ .sif → singularity, ไม่อย่างนั้น docker ====
SING=$(command -v apptainer || command -v singularity || true)
if [ -z "${RUNTIME:-}" ]; then
  if [ -n "$SING" ] && [ -e "$SIF" ]; then
    RUNTIME=singularity
  elif [ -e "$SIF" ]; then
    # มีไฟล์ .sif แต่ไม่มีคำสั่ง → อย่าหันไปใช้ docker เงียบ ๆ
    echo "found $SIF but no apptainer/singularity command"
    echo "try: module avail  แล้วส่ง CONTAINER_MODULE=<ชื่อ module>"
    exit 1
  else
    RUNTIME=docker
  fi
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "DATA_DIR not found: $DATA_DIR  (ส่ง DATA_DIR=/path/ข้อมูล ให้ถูก)"; exit 1
fi

export OUTDIR=results/${RUN_ID}
export JOB_ID=$RUN_ID
export NODE=$(hostname)
mkdir -p "$OUTDIR" logs
echo "runtime: $RUNTIME   node: $NODE   run: $RUN_ID   data: $DATA_DIR"

case $RUNTIME in
  singularity)
    # singularity รันเป็น process ธรรมดาใน job → เห็นเฉพาะ GPU ที่ Slurm จัดให้
    # (ตัวแปรที่ export ไว้ รวมถึง CUDA_VISIBLE_DEVICES ส่งต่อเข้า container อัตโนมัติ)
    export PHYS_GPUS=${CUDA_VISIBLE_DEVICES:-all}
    echo "image: $SIF   GPUs: $PHYS_GPUS"
    "$SING" exec --nv \
      --bind "$PWD":/workspace --pwd /workspace \
      --bind "$DATA_DIR":/data \
      "$SIF" bash bench_inner.sh
    rc=$?
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
      -e BACKENDS -e GPU_COUNTS -e SIZES -e REPEATS -e BUILD -e SW_BLOCKS -e SW_CHUNKS -e CHUNK_VAR \
      -e CUDA_MAKE_ARGS -e SYCL_MAKE_ARGS -e OUTDIR -e JOB_ID -e NODE \
      -e PHYS_GPUS="$PHYS_GPUS" \
      -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
      -e ACPP_APPDB_DIR=/workspace/.acpp_cache \
      -v "$PWD":/workspace -w /workspace \
      -v "$DATA_DIR":/data \
      "$IMAGE" bash bench_inner.sh
    rc=$?
    ;;

  *) echo "Unknown RUNTIME: $RUNTIME (use singularity or docker)"; exit 1 ;;
esac

if [ "$rc" -ne 0 ]; then
  echo "!! container exited with code $rc"
  exit "$rc"
fi
echo "Done. Results in $OUTDIR"
