FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG CACHEBUST
RUN echo "cache bust: ${CACHEBUST}" > /tmp/cachebust

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    ca-certificates \
    clang \
    cuda-nvrtc-12-4 \
    cuda-nvrtc-dev-12-4 \
    curl \
    git \
    intel-opencl-icd \
    libffi-dev \
    libx11-dev \
    m4 \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    opam \
    pkg-config \
  && rm -rf /var/lib/apt/lists/*

ENV CUDA_PATH=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib
ENV OPAMYES=1
