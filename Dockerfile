FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# --------------------------------------------------
# Basic dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    cmake \
    python3 \
    libboost-all-dev \
    build-essential \
    git \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*


# --------------------------------------------------
# LLVM 19
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    wget \
    lsb-release \
    software-properties-common \
    gnupg \
    && wget -O /tmp/llvm.sh https://apt.llvm.org/llvm.sh \
    && chmod +x /tmp/llvm.sh \
    && /tmp/llvm.sh 19 \
    && apt-get update \
    && apt-get install -y \
        llvm-19 \
        llvm-19-dev \
        libclang-19-dev \
        clang-19 \
        libomp-19-dev \
    && rm -f /tmp/llvm.sh \
    && rm -rf /var/lib/apt/lists/*


# --------------------------------------------------
# AdaptiveCpp
# --------------------------------------------------
RUN git clone --depth 1 \
        https://github.com/AdaptiveCpp/AdaptiveCpp \
        /tmp/acpp \
    && mkdir -p /tmp/acpp/build \
    && cd /tmp/acpp/build \
    && cmake \
        -DCMAKE_INSTALL_PREFIX=/opt/AdaptiveCpp \
        -DLLVM_DIR=/usr/lib/llvm-19/cmake \
        -DWITH_CUDA_BACKEND=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        .. \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/acpp


# --------------------------------------------------
# Environment
# --------------------------------------------------
ENV PATH=/opt/AdaptiveCpp/bin:$PATH

# Optional: make LLVM 19 easy to find
ENV LLVM_DIR=/usr/lib/llvm-19/cmake

# CUDA paths for building (ไม่ต้อง export เองหลังเข้า container อีก)
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CPATH=/usr/local/cuda/include \
    LC_ALL=C.UTF-8

# --------------------------------------------------
# Verify installation
# --------------------------------------------------
RUN which acpp \
    && acpp --version \
    && nvcc --version \
    && clang-19 --version

# docker run --rm -it   --runtime=nvidia   -e NVIDIA_VISIBLE_DEVICES=all   -v ~/Desktop/NV_SYCL_SW:/workspace   -w /workspace   my-sycl:v1 bash
# export CUDA_HOME=/usr/local/cuda
# export CUDA_PATH=/usr/local/cuda
# export CPATH=/usr/local/cuda/include:$CPATH
