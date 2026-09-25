# HPC_BWA-MEM

GPU-accelerated Smith-Waterman seed extension for BWA-MEM, with two backends built from the same source tree:

- **CUDA**: NVIDIA GPUs, single- and multi-GPU
- **SYCL** (AdaptiveCpp): cross-vendor, NVIDIA and AMD

The kernel follows BWA-MEM `ksw_extend` extension semantics (not generic local SW), uses a banded DP with band-only scratch storage, and assigns one GPU thread per extension task. `ksw.c` from BWA is included as the CPU reference.

---

## Repository layout

```
.
├── main.c              # entry point (backend selected at compile time)
├── bwa_sw.cu           # CUDA kernel + multi-GPU host code
├── bwa_sw_sycl.cpp     # SYCL kernel + multi-GPU host code
├── ksw.c / ksw.h       # CPU reference (from BWA)
├── bwa_build.c, fm_index.c, bwa_smem.c, bwa.h
├── Makefile.cuda
└── Makefile.sycl
```

---

## Requirements

| Component | Needed for |
|---|---|
| GCC | both |
| zlib (`libz-dev` / `zlib-devel`) | both |
| CUDA Toolkit (`nvcc`) | CUDA backend, and SYCL on NVIDIA |
| AdaptiveCpp (`acpp`) | SYCL backend |
| ROCm | SYCL on AMD |

---

## Quick start

```bash
# CUDA
make -f Makefile.cuda run        # build main_cuda and run it

# SYCL
make -f Makefile.sycl run         # build sw_sycl and run it
```

---

## How the backend is selected

`main.c` picks the backend with a preprocessor flag, so the source never needs to be edited:

```c
#ifdef USE_SYCL
    #define run_multi_gpu sycl_multi_gpu
#else
    #define run_multi_gpu sw_multi_gpu
#endif
```

- `Makefile.cuda` compiles without the flag → CUDA
- `Makefile.sycl` compiles with `-DUSE_SYCL` → SYCL

The two builds produce different binaries (`main_cuda`, `sw_sycl`), so both can live in the same directory.

---

## Build and run: CUDA

```bash
make -f Makefile.cuda main_cuda              # default: sm_86
make -f Makefile.cuda main_cuda ARCH=sm_70   # e.g. V100
./main_cuda
```

> Note: plain `make -f Makefile.cuda` only builds `libbwa.a`. Use the `main_cuda` or `test` target to get the executable.

| GPU | `ARCH` |
|---|---|
| Tesla V100 | `sm_70` |
| GTX 1080 Ti | `sm_61` |
| RTX 3070 Ti / RTX A4000 | `sm_86` |
| RTX 4090 | `sm_89` |

Check your GPU with:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

---

## Build and run: SYCL

```bash
make -f Makefile.sycl                                  # generic target (NVIDIA + AMD)
make -f Makefile.sycl ACPP_TARGETS=cuda:sm_86          # NVIDIA only, ahead-of-time
make -f Makefile.sycl ACPP_TARGETS=hip:gfx90a          # AMD only, ahead-of-time
./sw_sycl
```

| `ACPP_TARGETS` | Runs on | Notes |
|---|---|---|
| `generic` (default) | NVIDIA and AMD | One binary for all GPUs; kernels are JIT-compiled on first run, then cached |
| `cuda:sm_XX` | NVIDIA | No JIT at runtime |
| `hip:gfxXXX` | AMD | No JIT at runtime |

Other targets: `make -f Makefile.sycl clean`, `make -f Makefile.sycl rebuild`.

---

## Switching GPU architecture or target

Make does not track flag changes. After changing `ARCH` or `ACPP_TARGETS`, clean first:

```bash
make -f Makefile.cuda clean && make -f Makefile.cuda main_cuda ARCH=sm_70
make -f Makefile.sycl rebuild ACPP_TARGETS=cuda:sm_70
```

---


---

## Troubleshooting

- **`no kernel image is available for execution on the device`**: the binary was built for a different GPU architecture. Rebuild with the matching `ARCH` / `ACPP_TARGETS`.
- **`undefined reference to sw_multi_gpu`** when building SYCL: `main.o` is stale from a build without `-DUSE_SYCL`. Run `make -f Makefile.sycl rebuild`.
- **Nsight Compute cannot profile under WSL2**: profile on native Linux.

---
