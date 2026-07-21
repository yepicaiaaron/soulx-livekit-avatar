FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements_pipecat.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install FlashAttention
RUN pip install --no-cache-dir ninja
RUN pip install --no-cache-dir flash_attn==2.8.0.post2 --no-build-isolation || \
    (echo "Building FlashAttention from source..." && \
     pip install --no-cache-dir flash-attn --no-build-isolation)

# Install Pipecat and LiveKit
RUN pip install --no-cache-dir -r requirements_pipecat.txt

# Optional: FlashAttention 3 for Hopper/Blackwell (YEP-49). Long compile —
# enabled via: docker build --build-arg INSTALL_FA3=1 .
ARG INSTALL_FA3=0
RUN if [ "$INSTALL_FA3" = "1" ]; then \
      git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/fa && \
      cd /tmp/fa/hopper && MAX_JOBS=8 pip install --no-cache-dir . --no-build-isolation && \
      rm -rf /tmp/fa; \
    fi

# Copy application code
COPY webrtc_sync.py ./
COPY flash_head/ ./flash_head/
COPY examples/ ./examples/
COPY bench/ ./bench/
COPY tests/ ./tests/
COPY deploy/ ./deploy/

RUN chmod +x deploy/live_test.sh && pip install --no-cache-dir pytest

# Create directories for models (will be mounted as volumes)
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "webrtc_sync.py"]
