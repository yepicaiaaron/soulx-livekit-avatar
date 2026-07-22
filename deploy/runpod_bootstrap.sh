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
# Filter pins that fight the image's torch build (torch ships its own nccl;
# xformers is unused by the bench path).
grep -vE '^(nvidia-nccl-cu12|xformers)' requirements.txt > /tmp/req_bench.txt
pip install --no-cache-dir -q -r /tmp/req_bench.txt
pip install --no-cache-dir -q pytest "huggingface_hub[cli]" pyyaml

# Attention backends, best-effort and never source-building (a source build
# of flash-attn stalls a fresh pod for an hour):
pip install --no-cache-dir -q sageattention || true
PYTAG=$(python3 -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
TORCHTAG=$(python3 -c 'import torch;v=torch.__version__.split("+")[0].split(".");print(f"torch{v[0]}.{v[1]}")')
FA_OK=0
for FA_VER in 2.8.2 2.8.0.post2; do
  for ABI in TRUE FALSE; do
    W="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/flash_attn-${FA_VER}+cu12${TORCHTAG}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
    if pip install --no-cache-dir -q "$W" 2>/dev/null; then FA_OK=1; break 2; fi
  done
done
[ "$FA_OK" = "1" ] || echo "WARN: no prebuilt flash-attn wheel; dispatch falls back to sage/sdpa"
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

export SOULX_MODEL_TYPE="${SOULX_MODEL_TYPE:-pro}"
export FLASH_HEAD_PROFILE="${FLASH_HEAD_PROFILE:-lowlat}"
if [ "${LIVE:-0}" = "1" ] && [ "${SKIP_GATES:-0}" = "1" ]; then
  # Demo fast path: gates already validated on this host class this session.
  echo "=== [bootstrap] SKIP_GATES=1 — launching live bot directly ==="
  python3 deploy/mint_token.py aaron > /workspace/join_url.txt 2>/dev/null || true
  exec python3 webrtc_sync.py
elif [ "${LIVE:-0}" = "1" ]; then
  echo "=== [bootstrap] running gated suite (live) ==="
  exec ./deploy/live_test.sh
else
  echo "=== [bootstrap] running gated suite ==="
  ./deploy/live_test.sh --bench-only
fi
