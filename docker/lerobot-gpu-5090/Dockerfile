FROM ghcr.io/pytorch/pytorch-nightly:2.8.0.dev20250418-cuda12.8-cudnn9-runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git \
      libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
      speech-dispatcher libgeos-dev \
      python3.10-dev python3.10-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL="egl"

WORKDIR /lerobot
COPY . /lerobot

# Python + Conda + Nvidia deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir . && \
    python -m pip install nvidia-npp-cu12 --extra-index-url https://pypi.nvidia.com && \
    conda install -c nvidia/label/cuda-12.8.0 libnpp && \
    pip uninstall -y torchcodec && \
    pip install --pre torchcodec --index-url https://download.pytorch.org/whl/nightly/cu128
