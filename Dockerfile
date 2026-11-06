#FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# --------------------------------------------------
# Versions (override with --build-arg)
#   LLVM_VERSION : AdaptiveCpp develop supports LLVM 15–21
#   ACPP_REF     : branch, tag (e.g. v25.10.0) or commit hash
# --------------------------------------------------
ARG LLVM_VERSION=21
ARG ACPP_REF=develop
ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------
# Basic dependencies
# (Boost is no longer a core dependency since AdaptiveCpp 25.10;
#  only needed for unit tests, so libboost-all-dev is dropped)
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        python3 \
        wget \
        ca-certificates \
        lsb-release \
        software-properties-common \
        gnupg \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# LLVM ${LLVM_VERSION} (from apt.llvm.org)
# --------------------------------------------------
RUN wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh \
    && chmod +x /tmp/llvm.sh \
    && /tmp/llvm.sh ${LLVM_VERSION} \
    && apt-get update && apt-get install -y --no-install-recommends \
        llvm-${LLVM_VERSION}-dev \
        libclang-${LLVM_VERSION}-dev \
        clang-tools-${LLVM_VERSION} \
        libomp-${LLVM_VERSION}-dev \
    && rm -f /tmp/llvm.sh \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# AdaptiveCpp
# --------------------------------------------------
RUN git clone --depth 1 --branch ${ACPP_REF} \
        https://github.com/AdaptiveCpp/AdaptiveCpp /tmp/acpp \
    && cd /tmp/acpp && git log -1 --format='AdaptiveCpp commit: %H (%cd)' \
    && cmake -S /tmp/acpp -B /tmp/acpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/AdaptiveCpp \
        -DCMAKE_C_COMPILER=clang-${LLVM_VERSION} \
        -DCMAKE_CXX_COMPILER=clang++-${LLVM_VERSION} \
        -DLLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm \
        -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-${LLVM_VERSION} \
        -DWITH_CUDA_BACKEND=ON \
        -DCUDAToolkit_ROOT=/usr/local/cuda \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    && cmake --build /tmp/acpp/build -j"$(nproc)" \
    && cmake --install /tmp/acpp/build \
    && rm -rf /tmp/acpp

# --------------------------------------------------
# Environment
# --------------------------------------------------
ENV PATH=/opt/AdaptiveCpp/bin:/usr/lib/llvm-${LLVM_VERSION}/bin:$PATH \
    LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LC_ALL=C.UTF-8

# --------------------------------------------------
# Verify installation (no GPU is visible at build time,
# so only check the toolchain itself)
# --------------------------------------------------
RUN which acpp \
    && acpp --version \
    && nvcc --version \
    && clang-${LLVM_VERSION} --version
# docker run --rm -it   --runtime=nvidia   -e NVIDIA_VISIBLE_DEVICES=all   -v ~/Desktop/NV_SYCL_SW:/workspace   -w /workspace   my-sycl:v1 bash
# export CUDA_HOME=/usr/local/cuda
# export CUDA_PATH=/usr/local/cuda
# export CPATH=/usr/local/cuda/include:$CPATH
