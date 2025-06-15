FROM ghcr.io/pytorch/pytorch-nightly:2.8.0.dev20250418-cuda12.8-cudnn9-runtime

# Install only the system dependencies you need
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git \
      libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
      speech-dispatcher libgeos-dev \
      python3.10-dev python3.10-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Environment tweaks
ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL="egl"

# Copy your code and install only your package extras
WORKDIR /lerobot
COPY . /lerobot

RUN pip install --upgrade pip \
 && pip install --no-cache-dir .

# Default command (adjust as needed)
CMD ["python", "train.py"]
