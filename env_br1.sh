# ============================================================
# env_br1.sh — ตั้งค่าสำหรับรัน HPC_BWA-MEM บน cluster br1 (DGX A100)
#
# ใช้ด้วย source (ไม่ใช่ bash) เพื่อให้ module และตัวแปรติดอยู่ใน shell:
#     source ~/HPC_BWA-MEM/env_br1.sh
#
# คำสั่งลัดหลัง source:
#     bench_test [จำนวน task]     ทดสอบเร็ว 1 GPU ผ่าน srun (เห็นผลบนจอ)
#     bench_submit [จำนวน GPU]    ส่งชุดเต็มด้วย sbatch (default 4 GPU)
#     bench_status                ดูคิวและ job วันนี้ (รวม RAM ที่ใช้)
#     bench_last                  ดูตาราง summary.csv ของรอบล่าสุด
#     bench_gpu <jobid>           ดู nvidia-smi ของ job ที่กำลังรัน
# ============================================================

# ---- module (มีแค่ตัวเดียวที่ต้องใช้: CUDA/AdaptiveCpp อยู่ใน container) ----
module load apptainer/apptainer

# ---- ตำแหน่งไฟล์ ----
export PROJECT_DIR=$HOME/HPC_BWA-MEM
export SIF=$HOME/hpc-bwa-mem_sandbox       # sandbox (ไม่ต้องแตก .sif ทุกครั้ง)
export DATA_DIR=$PROJECT_DIR               # โฟลเดอร์ข้อมูล input → /data ใน container
export RUNTIME=singularity                 # cluster นี้ใช้ docker ไม่ได้

# ---- ค่า default ของ job ----
export BENCH_PARTITION=fullq               # dgx-03: A100 เต็มตัว (gpuq/kuresearch เป็น MIG)
export BENCH_CPUS=8
export BENCH_MEM=64G                       # ไม่ใส่จะได้แค่ 512M → OOM (exit 137)

# ---- คำสั่งลัด ----

# ทดสอบเร็ว 1 GPU, 1 block size, 1 รอบ, ไม่ build ใหม่ (BUILD=1 bench_test ถ้าแก้โค้ด)
bench_test() {
  local size=${1:-30000}
  ( cd "$PROJECT_DIR" && mkdir -p logs && \
    srun -p "$BENCH_PARTITION" --gres=gpu:1 --cpus-per-task="$BENCH_CPUS" \
         --mem="$BENCH_MEM" --time=00:30:00 \
         --export=ALL,GPU_COUNTS=1,SW_BLOCKS=128,REPEATS=1,SIZES="$size",BUILD="${BUILD:-0}",MAKEFLAGS=-j"$BENCH_CPUS" \
         bash run_bench.slurm )
}

# ส่งชุดเต็มด้วย sbatch: ค่าอื่น (SIZES, SW_BLOCKS, REPEATS, ...) ใช้ตามใน run_bench.slurm
# หรือกำหนดเพิ่มหน้าคำสั่ง เช่น  SIZES="10000000 30000000" bench_submit 4
bench_submit() {
  local ngpu=${1:-4}
  local counts="1"
  [ "$ngpu" -ge 2 ] && counts="1 2"
  [ "$ngpu" -ge 4 ] && counts="1 2 4"
  [ "$ngpu" -ge 8 ] && counts="1 2 4 8"
  ( cd "$PROJECT_DIR" && mkdir -p logs && \
    GPU_COUNTS="${GPU_COUNTS:-$counts}" MAKEFLAGS=-j"$BENCH_CPUS" \
    sbatch -p "$BENCH_PARTITION" --gres=gpu:"$ngpu" --cpus-per-task="$BENCH_CPUS" \
           --mem="$BENCH_MEM" --export=ALL run_bench.slurm )
}

bench_status() {
  squeue -u "$USER"
  echo
  sacct -u "$USER" -S today \
    --format=JobID,JobName%12,Partition,State,ExitCode,Elapsed,ReqMem,MaxRSS
}

bench_last() {
  local d
  d=$(ls -td "$PROJECT_DIR"/results/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "no results yet"; return 1; }
  echo "$d"
  column -s, -t "$d/summary.csv" 2>/dev/null || cat "$d/summary.csv"
}

bench_gpu() {
  [ -z "${1:-}" ] && { echo "usage: bench_gpu <jobid>"; return 1; }
  srun --jobid="$1" --overlap nvidia-smi
}

echo "HPC_BWA-MEM env ready: $(command -v apptainer || echo 'apptainer NOT found')"
echo "  project: $PROJECT_DIR"
echo "  image:   $SIF"
echo "  commands: bench_test, bench_submit, bench_status, bench_last, bench_gpu"
