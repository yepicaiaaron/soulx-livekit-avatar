#!/bin/bash
# Launch script for SoulX Conversational Bot

# 1. Environment Validation
if [ -z "$LIVEKIT_URL" ]; then
    echo "Error: LIVEKIT_URL is not set"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    exit 1
fi

# 2. Paths
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$BASE_DIR

# 3. Model Configuration
export SOULX_CKPT_DIR=${SOULX_CKPT_DIR:-"./models/SoulX-FlashHead-1_3B"}
export SOULX_WAV2VEC_DIR=${SOULX_WAV2VEC_DIR:-"./models/wav2vec2-base-960h"}
export SOULX_MODEL_TYPE=${SOULX_MODEL_TYPE:-"lite"}

# 4. Activation
if [ -f "$BASE_DIR/venv/bin/activate" ]; then
    source "$BASE_DIR/venv/bin/activate"
fi

# 5. Run
echo "Starting SoulX Conversational Bot..."
python3 "$BASE_DIR/soulx_conversational_bot.py"
