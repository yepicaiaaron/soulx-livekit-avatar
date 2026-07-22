#!/usr/bin/env bash
# One-command live-test deployment for the YEP-48/49 low-latency avatar.
#
#   ./deploy/live_test.sh            # full gated pipeline -> launches bot
#   ./deploy/live_test.sh --profile balanced
#   ./deploy/live_test.sh --skip-quality   # skip the PSNR A/B (saves ~10 min)
#
# Gates, in order — each must pass before the next runs:
#   0. environment: GPU, models, symlinks, .env
#   1. unit/smoke tests (pytest)
#   2. kernel parity + microbench (bench_kernels)
#   3. quality A/B: eager vs fused PSNR gate (bench_quality)
#   4. latency: sustains 25fps at the chosen profile (bench_latency)
#   5. launch webrtc_sync.py
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="lowlat"
SKIP_QUALITY=0
BENCH_ONLY=0
for arg in "$@"; do
  case $arg in
    --profile) ;; # value read next iteration via shift-less parse below
    --profile=*) PROFILE="${arg#*=}" ;;
    balanced|default|lowlat) PROFILE="$arg" ;;
    --skip-quality) SKIP_QUALITY=1 ;;
    --bench-only) BENCH_ONLY=1 ;;  # run gates 0-4, skip live launch (no LiveKit needed)
  esac
done

export FLASH_HEAD_PROFILE="$PROFILE"
export SOULX_MODEL_TYPE="${SOULX_MODEL_TYPE:-pro}"

echo "=== [0/5] environment checks (profile=$PROFILE, model=$SOULX_MODEL_TYPE) ==="
command -v nvidia-smi >/dev/null || { echo "FATAL: no NVIDIA driver"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; import triton; print('torch', torch.__version__, '| triton', triton.__version__)"
[ -e "models/SoulX-FlashHead-1_3B" ] || { echo "FATAL: models/SoulX-FlashHead-1_3B missing (see README symlinks)"; exit 1; }
[ -e "models/wav2vec2-base-960h" ] || { echo "FATAL: models/wav2vec2-base-960h missing"; exit 1; }
if [ "$SOULX_MODEL_TYPE" = "pro" ]; then
  [ -f "models/vae/lightvaew2_1.pth" ] || { echo "FATAL: models/vae/lightvaew2_1.pth missing (hf download lightx2v/Autoencoders lightvaew2_1.pth)"; exit 1; }
fi
if [ "$BENCH_ONLY" -eq 0 ]; then
  [ -f ".env" ] || { echo "FATAL: .env missing (copy .env.example)"; exit 1; }
  set -a; source .env; set +a
  : "${LIVEKIT_URL:?FATAL: LIVEKIT_URL not set in .env}"
  : "${LIVEKIT_API_KEY:?FATAL: LIVEKIT_API_KEY not set}"
  : "${LIVEKIT_API_SECRET:?FATAL: LIVEKIT_API_SECRET not set}"
fi

echo "=== [1/5] unit + smoke tests ==="
python3 -m pytest tests/ -q

echo "=== [2/5] kernel parity + microbench (YEP-48) ==="
python3 bench/bench_kernels.py

if [ "$SKIP_QUALITY" -eq 0 ]; then
  echo "=== [3/5] quality A/B gate: eager vs fused (PSNR) ==="
  QDIR="$(mktemp -d)"
  FLASH_HEAD_ATTN=fa2 FLASH_HEAD_FUSED_KERNELS=0 python3 bench/bench_quality.py --out "$QDIR/ref.npy"
  FLASH_HEAD_ATTN=fa2 FLASH_HEAD_FUSED_KERNELS=1 python3 bench/bench_quality.py --out "$QDIR/fused.npy"
  python3 bench/bench_quality.py --compare "$QDIR/ref.npy" "$QDIR/fused.npy"
  rm -rf "$QDIR"
else
  echo "=== [3/5] quality gate SKIPPED (--skip-quality) ==="
fi

echo "=== [4/5] latency: sustains 25fps at $PROFILE? ==="
if ! python3 bench/bench_latency.py; then
  if [ "$PROFILE" = "lowlat" ]; then
    echo "lowlat not sustained — retrying at balanced (320ms chunks)"
    export FLASH_HEAD_PROFILE="balanced"
    python3 bench/bench_latency.py
  else
    exit 1
  fi
fi

if [ "$BENCH_ONLY" -eq 1 ]; then
  echo "=== [5/5] SKIPPED (--bench-only). All gates passed at profile=$FLASH_HEAD_PROFILE ==="
  exit 0
fi

echo "=== [5/5] launching live bot (profile=$FLASH_HEAD_PROFILE) ==="
echo "Join the LiveKit room with a generated JWT to talk to the avatar."
exec python3 webrtc_sync.py
