#!/usr/bin/env bash
# ==============================================================================
#  SoulX Avatar — Lightning.ai A100 One-Shot Setup Script
#
#  Run from the repo root:
#    bash setup_lightning.sh               # downloads Model_Lite (default)
#    MODEL_TYPE=pro bash setup_lightning.sh # downloads Model_Pro instead
#
#  Model options:
#    lite  — Model_Lite:  96 FPS on A100; supports 3x concurrent real-time streams.
#              Uses LTX-Video VAE. Recommended for single-GPU A100 80 GB.
#    pro   — Model_Pro:  higher visual quality; real-time (25+ FPS) on A100 80 GB.
#              Uses WAN VAE. Requires more VRAM and is slower per frame.
#
#  What it does:
#    1. Checks GPU
#    2. Installs PyTorch 2.7.1 (CUDA 12.8) + all Python deps
#    3. Installs FlashAttention 2 (pre-built wheel when possible)
#    4. Downloads the selected SoulX-FlashHead model + VAE + wav2vec2
#    5. Verifies the install
# ==============================================================================

set -euo pipefail

# ── Model selection ────────────────────────────────────────────────────────
# Precedence: MODEL_TYPE env var (set inline) > SOULX_MODEL_TYPE (Lightning.ai Secret) > lite
# Examples:
#   MODEL_TYPE=pro bash setup_lightning.sh          # explicit inline override
#   SOULX_MODEL_TYPE=pro; bash setup_lightning.sh   # via Secret already in environment
MODEL_TYPE="${MODEL_TYPE:-${SOULX_MODEL_TYPE:-lite}}"

if [[ "$MODEL_TYPE" != "lite" && "$MODEL_TYPE" != "pro" ]]; then
    echo "Unknown MODEL_TYPE='${MODEL_TYPE}'. Must be 'lite' or 'pro'."
    exit 1
fi

# Colours
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}   ✅  $*${NC}"; }
warn() { echo -e "${YELLOW}   ⚠️   $*${NC}"; }
err()  { echo -e "${RED}   ❌  $*${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   SoulX Avatar — Lightning.ai A100 Setup                ║"
echo "║   This will take 15–25 minutes on first run.            ║"
echo "╠══════════════════════════════════════════════════════════╣"
if [[ "$MODEL_TYPE" == "lite" ]]; then
echo "║   Model: Model_Lite  (96 FPS on A100 — recommended)     ║"
else
echo "║   Model: Model_Pro   (higher quality, more VRAM)        ║"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: GPU check ──────────────────────────────────────────────────────
echo "🔍  [1/5]  Checking GPU…"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    ok "GPU detected: ${GPU_NAME} (${GPU_VRAM})"
else
    warn "nvidia-smi not found — make sure you selected a GPU machine in Lightning.ai."
fi
echo ""

# ── Step 2: Python deps ────────────────────────────────────────────────────
echo "📦  [2/5]  Installing Python dependencies…"

echo "   Upgrading pip/setuptools/wheel…"
pip install --upgrade pip setuptools wheel --quiet

echo "   Installing PyTorch 2.7.1 + CUDA 12.8…"
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --quiet

echo "   Installing SoulX-FlashHead model dependencies…"
pip install -r requirements.txt --quiet

echo "   Installing Pipecat + Daily.co + OpenAI dependencies…"
pip install -r requirements_pipecat.txt --quiet

echo "   Installing HuggingFace Hub CLI (for model download)…"
pip install "huggingface_hub[cli]>=0.23.0" --quiet
ok "Python dependencies installed."
echo ""

# ── Step 3: FlashAttention ────────────────────────────────────────────────
echo "⚡  [3/5]  Installing FlashAttention 2 (requires CUDA compiler — takes ~5 min)…"
pip install ninja --quiet

# Try the pre-built wheel first (fast), fall back to source build (slow)
if pip install "flash_attn==2.8.0.post2" --no-build-isolation --quiet 2>/dev/null; then
    ok "FlashAttention 2.8.0.post2 installed (pre-built wheel)."
else
    warn "Pre-built wheel not found — building from source (this takes 5–10 minutes)…"
    pip install flash-attn --no-build-isolation --quiet
    ok "FlashAttention installed from source."
fi
echo ""

# ── Step 4: ffmpeg ────────────────────────────────────────────────────────
echo "🎬  [4/5]  Checking ffmpeg…"
if command -v ffmpeg &> /dev/null; then
    ok "ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3)"
else
    echo "   Installing ffmpeg via apt-get…"
    apt-get install -y ffmpeg --quiet 2>/dev/null && ok "ffmpeg installed." || \
        warn "apt-get failed. Run: conda install -c conda-forge ffmpeg=7 -y"
fi
echo ""

# ── Step 5: Model download ────────────────────────────────────────────────
mkdir -p models

if [[ "$MODEL_TYPE" == "lite" ]]; then
    echo "📥  [5/5]  Downloading Model_Lite + VAE_LTX (~5–8 GB)…"
    echo "   Model_Lite: 96 FPS on A100 — perfect for real-time streaming."
    echo "   This is the biggest step — grab a coffee ☕"
    echo ""
    huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
        --include "Model_Lite/**" "VAE_LTX/**" \
        --local-dir ./models/SoulX-FlashHead-1_3B
else
    echo "📥  [5/5]  Downloading Model_Pro + VAE_WAN (~8–12 GB)…"
    echo "   Model_Pro: higher visual quality; runs real-time on A100 80 GB."
    echo "   This is the biggest step — grab a coffee ☕"
    echo ""
    huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
        --include "Model_Pro/**" "VAE_WAN/**" \
        --local-dir ./models/SoulX-FlashHead-1_3B
fi

echo ""
echo "   Downloading wav2vec2-base-960h audio encoder…"
huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir ./models/wav2vec2-base-960h

ok "Models downloaded."
echo ""

# ── Verify installation ───────────────────────────────────────────────────
echo "🧪  Verifying installation…"

MODEL_TYPE_FOR_PY="$MODEL_TYPE" python - <<'PYEOF'
import sys, torch, os

# CUDA
assert torch.cuda.is_available(), "CUDA is not available — re-check your GPU machine selection in Lightning.ai"
print(f"   PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")

# flash_head module
from flash_head.inference import get_infer_params
p = get_infer_params()
print(f"   flash_head module: OK  (frame_num={p['frame_num']}, fps={p['tgt_fps']})")

# Pipecat
import pipecat
print(f"   pipecat: OK")

# Daily
import daily
print(f"   daily-python: OK")

# Model weights (spot-check)
model_type = os.environ.get("MODEL_TYPE_FOR_PY", "lite")
ckpt_base  = "./models/SoulX-FlashHead-1_3B"
if model_type == "lite":
    check_dirs = [
        os.path.join(ckpt_base, "Model_Lite"),
        os.path.join(ckpt_base, "VAE_LTX"),
    ]
else:
    check_dirs = [
        os.path.join(ckpt_base, "Model_Pro"),
        os.path.join(ckpt_base, "VAE_WAN"),
    ]
check_dirs.append("./models/wav2vec2-base-960h")
for d in check_dirs:
    assert os.path.isdir(d), f"Missing model directory: {d}"
print(f"   Model weights ({model_type}): OK")
PYEOF

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ✅  Setup complete!  Ready to launch the avatar.       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║   Run:  python gradio_room_launcher.py                   ║"
echo "║                                                          ║"
echo "║   Then expose port 7860 in the Lightning.ai UI:          ║"
echo "║     Left sidebar → Ports → Add port 7860 → Public        ║"
echo "║                                                          ║"
if [[ "$MODEL_TYPE" == "pro" ]]; then
echo "║   Model_Pro downloaded. To use it, add this secret:      ║"
echo "║     SOULX_MODEL_TYPE = pro                               ║"
echo "║   Without this secret the bot defaults to Model_Lite     ║"
echo "║   and will fail to find the Pro weights at runtime.      ║"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
