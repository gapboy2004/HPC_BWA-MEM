# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GPU benchmark of BWA-MEM's seed-extension step (`ksw_extend2` semantics) with two backends from one source tree: CUDA (`bwa_sw.cu`) and SYCL via AdaptiveCpp (`bwa_sw_sycl.cpp`). It is a Linux/HPC project: building and running need Linux with `nvcc`/`acpp`, usually inside the Docker/Singularity image. The dev machine may be Windows, which can only edit the code. Many code comments are in Thai.

## Build & run

```bash
# CUDA: the default target builds only libbwa.a. Name the binary target explicitly.
make -f Makefile.cuda sw_cuda && ./sw_cuda <max_tasks>
make -f Makefile.cuda run

# SYCL
make -f Makefile.sycl                               # ACPP_TARGETS=generic (JIT, NVIDIA+AMD)
make -f Makefile.sycl ACPP_TARGETS=cuda:sm_86       # or hip:gfx90a (ahead-of-time)
make -f Makefile.sycl rebuild                       # clean + build
./sw_sycl <max_tasks>
```

The README is out of date in two places. It says the CUDA binary is `main_cuda` and that the arch is chosen with `ARCH=sm_XX`. In fact `Makefile.cuda` has `TARGET = sw_cuda` and hard-codes `-gencode` for sm_70 and sm_86 in `NVFLAGS`. To build for another arch, override the whole variable: `NVFLAGS='-O2 -arch=sm_89'`.

Make does not track flag changes. Run clean before switching arch or backend. **`main.o` is shared by both backends** (the SYCL build compiles it with `-DUSE_SYCL`), so delete `main.o` when switching backends. A stale `main.o` causes `undefined reference to sw_multi_gpu`/`sycl_multi_gpu`.

`bwa.h` includes `<cuda_runtime.h>` (for `StreamBuf`), so the SYCL build also needs the CUDA headers on `CPATH`.

### Runtime knobs (environment variables)
- `N_GPU`: number of GPUs (default 2; clamped to the available count, max 8)
- `SW_CHUNK`: tasks per device chunk (default 50000)
- `MAX_TASKS`, or argv[1]: upper limit on the number of tasks packed from the input

**Input paths are hard-coded in `main.c`** (`gzopen(...)` around line 92, currently `/home/gapboy/Desktop/real_*_full.fasta`). Commented-out alternatives point to `test_prog/*.fasta.gz` (20k sample pairs) and to `own_*_batch.fasta`. Edit these to change the input. FASTA files are gitignored.

### Benchmarks (cluster)
`run_bench.slurm` (run with `sbatch` or `bash`) picks a runtime: Singularity if `$SIF` exists, otherwise Docker (`gapboy/hpc-bwa-mem:v1.1`, built from `Dockerfile`: CUDA 12.6, LLVM 19, AdaptiveCpp). It then runs `bench_inner.sh` inside the container. That script rebuilds each backend, does a warm-up run, then sweeps `BACKENDS` × `GPU_COUNTS` × `SIZES` × `REPEATS`. It parses the program's stdout (`using X / Y GPU`, `total: ...`, `GCUPS:`) into `results/<job>/summary.csv`. If you change those printf formats, update `parse_log()` in `bench_inner.sh`. Overrides go in `--export=ALL,SIZES=...,CUDA_MAKE_ARGS=...`.

### Test data
`exam_generate` is a Python script with no extension. It writes `own_query_batch.fasta` and `own_target_batch.fasta` (synthetic read/target pairs: 150bp reads, 170bp targets, 1% substitutions), using an optional reference FASTA given as argv[1].

## Architecture

**Pipeline in `main.c`:** read paired query/target FASTA with `kseq.h` (record i of the query file pairs with record i of the target file). Skip pairs longer than `MAX_QLEN=160` / `MAX_RLEN=200`. Pack all sequences into flat host buffers with `size_t` offsets, encode them to 0–4 (A,C,G,T,N), then make one timed call to `run_multi_gpu`. That macro is set at compile time to `sw_multi_gpu` (CUDA) or `sycl_multi_gpu` (SYCL, under `-DUSE_SYCL`).

**Both backends use the same two levels and are meant to stay in lockstep:**
1. `*_multi_gpu`: splits tasks into contiguous ranges, one pthread per GPU.
2. Per-device worker (`sw_batch` for CUDA, `sw_sycl` for SYCL): loops over chunks of `SW_CHUNK` tasks. It repacks each chunk into pinned staging with 32-bit chunk-local offsets, and double-buffers H2D → kernel → D2H across `N_STREAM=2` CUDA streams (SYCL uses in-order queues).

**Kernel:** one thread per extension task. It is a port of `ksw_extend2`: an extension from seed score `h0`, a band `w` that adapts and shrinks, and z-drop. DP state is a rolling row (`eh_h`/`eh_e[MAX_QLEN+1]`) in thread-private memory, with no global DP matrix. There is no traceback and no CIGAR; `out_cigar` is just set to an empty string. Scoring constants (`MATCH`, `MISMATCH`, `O_DEL`, …, `W_DEFAULT=100`, `H0_DEFAULT=20`) are `#define`d separately in each backend file and **must match between `bwa_sw.cu`, `bwa_sw_sycl.cpp`, and the `ksw_extend2` call used for validation**. `h0_arr`/`w_arr` are passed as `nullptr` (per-task values are a TODO).

**`COUNT_CELLS` hack:** when it is defined (it is on in `bwa_sw.cu`), the kernel overwrites `out_qbeg[tid]` with the number of DP cells computed. `main.c` sums `qbeg[]` to get `total cells` and GCUPS. So `qbeg` is not a real coordinate in benchmark builds.

**Correctness reference:** `ksw.c`/`ksw.h` are BWA's CPU code. `validate.c` is a code fragment, not a standalone program. It is meant to be pasted into `main.c` after the GPU call, and compares `score`/`qend`/`rend` with `ksw_extend2` for each task. The CUDA kernel was validated this way on 611,008 tasks.

**Unused or legacy code:**
- `bwa_build.c`, `fm_index.c`, `bwa_smem.c`: a naive suffix array, BWT, FM-index, and SMEM/chaining (an earlier phase). They are compiled into `libbwa.a` but are not on the benchmark path.
- `test_sw_kernel.cpp`: an older standalone SYCL test with different scoring and BAND_W, and with traceback.
- `cuda/`: experiments, including a committed ELF binary `cuda/multi`.
- `test_prog/`: GASAL2's test program, kept for its sample FASTA data.
