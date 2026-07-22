#!/usr/bin/env bash
# RunPod pod bootstrap: fresh runpod/pytorch pod -> models downloaded ->
# gated benchmark suite. Run from the repo root on the pod:
#
#   bash deploy/runpod_bootstrap.sh              # bench-only (gates 0-4)
#   LIVE=1 bash deploy/runpod_bootstrap.sh       # + launch bot (needs .env)
#
# Assumes the runpod/pytorch image (torch>=2.7 + CUDA 12.8 + triton bundled).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [bootstrap] system deps ==="
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 libgl1 libglib2.0-0 > /dev/null

echo "=== [bootstrap] python deps ==="
pip install --no-cache-dir -q -r requirements.txt
pip install --no-cache-dir -q pytest "huggingface_hub[cli]" pyyaml
pip install --no-cache-dir -q flash_attn --no-build-isolation || \
  echo "WARN: flash-attn 2 unavailable; dispatch will fall back (sage/sdpa)"
if [ "${LIVE:-0}" = "1" ]; then
  pip install --no-cache-dir -q -r requirements_pipecat.txt
fi

echo "=== [bootstrap] models ==="
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p models/vae
[ -e models/SoulX-FlashHead-1_3B/Model_Pro ] || \
  hf download Soul-AILab/SoulX-FlashHead-1_3B --exclude "assets/*" \
    --local-dir models/SoulX-FlashHead-1_3B
[ -e models/wav2vec2-base-960h/config.json ] || \
  hf download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h
[ -f models/vae/lightvaew2_1.pth ] || \
  hf download lightx2v/Autoencoders lightvaew2_1.pth --local-dir models/vae
du -sh models/* || true

echo "=== [bootstrap] running gated suite ==="
export SOULX_MODEL_TYPE="${SOULX_MODEL_TYPE:-pro}"
export FLASH_HEAD_PROFILE="${FLASH_HEAD_PROFILE:-lowlat}"
if [ "${LIVE:-0}" = "1" ]; then
  exec ./deploy/live_test.sh
else
  ./deploy/live_test.sh --bench-only
fi
