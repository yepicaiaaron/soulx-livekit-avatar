#!/bin/bash
# Setup script for SoulX-FlashHead Pipecat Bot on soulx-flashhead-node
# Run this on the target node to set up the environment

set -e

echo "=============================================="
echo "SoulX-FlashHead Pipecat Bot Setup"
echo "=============================================="

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. CUDA may not be available."
else
    echo "CUDA available:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "flashhead"; then
    echo "Creating conda environment: flashhead"
    conda create -n flashhead python=3.10 -y
fi

echo "Activating conda environment: flashhead"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flashhead

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Install SoulX-FlashHead base requirements
echo "Installing SoulX-FlashHead dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install FlashAttention
echo "Installing FlashAttention..."
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation || {
    echo "FlashAttention build failed. Attempting wheel installation..."
    pip install flash-attn --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2
}

# Install SageAttention (optional, for Pro model)
echo "Installing SageAttention (optional)..."
pip install sageattention==2.2.0 --no-build-isolation || echo "SageAttention installation failed (optional)"

# Install Pipecat and LiveKit dependencies
echo "Installing Pipecat and LiveKit dependencies..."
pip install -r requirements_pipecat.txt

# Install ffmpeg if not present
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg ffmpeg-devel
    else
        conda install -c conda-forge ffmpeg==7 -y
    fi
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Download model weights:"
echo "   huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B"
echo "   huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h"
echo ""
echo "2. Set up LiveKit credentials:"
echo "   export LIVEKIT_URL=wss://your-livekit-server:7880"
echo "   export LIVEKIT_TOKEN=your-token"
echo "   export LIVEKIT_ROOM=soulx-room"
echo ""
echo "3. Run the bot:"
echo "   ./run_bot.sh"
echo ""
