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

# Copy application code
COPY bot.py run_bot.sh ./
COPY flash_head/ ./flash_head/
COPY examples/ ./examples/

# Make scripts executable
RUN chmod +x run_bot.sh

# Create directories for models (will be mounted as volumes)
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "bot.py"]
