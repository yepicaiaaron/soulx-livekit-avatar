Real Time Diffusion WebRTC Avatar — with Conversational Brain & Perception Engine

A fully real-time, low-latency talking head integration using [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) and [Daily.co](https://www.daily.co/) via [Pipecat](https://github.com/pipecat-ai/pipecat).

## 🤖 Models

Two models are available from [Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B):

| Model | Speed | Quality | VAE | Best for |
|-------|-------|---------|-----|----------|
| **Model_Lite** *(default)* | **96 FPS** on A100 — 3× concurrent real-time streams | Good | LTX-Video | Single A100 80 GB — recommended |
| **Model_Pro** | 10.8 FPS on RTX4090 / **25+ FPS** on A100 80 GB | Higher | WAN VAE | When visual quality is the priority |

Set `SOULX_MODEL_TYPE=lite` (default) or `SOULX_MODEL_TYPE=pro` to switch.

## 🚀 Key Features
- **Real-Time Lip-Sync:** Captures user audio from a WebRTC room and streams back a lip-synced video avatar in real time.
- **Conversational Brain:** Full speech-to-text → LLM → text-to-speech loop powered by OpenAI Whisper, GPT-4o, and OpenAI TTS — orchestrated by Pipecat.
- **Tool Calling:** GPT-4o uses function tools (`get_current_time`, `get_visual_context`, `calculate`) to give the avatar richer, context-aware responses.
- **Perception Engine:** Subscribes to participant video tracks (webcam + screen share) in the Daily.co room and analyses frames via OpenAI Vision to give the LLM real-time visual awareness of the user and their shared screen.
- **LightX2V Distilled VAE:** Model_Lite integrates the distilled LTX-Video VAE, heavily reducing decode time (~35 ms saved per chunk) and minimising VRAM footprint.
- **Hardware Optimised:** Uses `enable_flash_sdp(True)` to shave ~3 ms per denoising step over eager operations.
- **Intelligent Playback Queue:** Binds video frame delivery to the system atomic clock to guarantee mathematically accurate 25 fps WebRTC streaming without robotic skipping or "queue drift."

## 🛠️ Environment Setup & Installation

### 1. Clone this repository

```bash
git clone https://github.com/trip-nine/soulx-livekit-avatar.git
cd soulx-livekit-avatar
```

The `flash_head/` engine code is included directly in this repo — **no symlinks or separate upstream clone required.**

### 2. Download models

Models are downloaded directly from HuggingFace into `./models/`:

```bash
# Model_Lite (default — recommended for A100 80 GB):
bash setup_lightning.sh

# Model_Pro (higher quality):
MODEL_TYPE=pro bash setup_lightning.sh
```

Or download manually:

```bash
# Model_Lite
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
    --include "Model_Lite/**" "VAE_LTX/**" \
    --local-dir ./models/SoulX-FlashHead-1_3B

# Model_Pro
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
    --include "Model_Pro/**" "VAE_WAN/**" \
    --local-dir ./models/SoulX-FlashHead-1_3B

# Audio encoder (required for both models)
huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir ./models/wav2vec2-base-960h
```

### 3. Environment Variables
Create a `.env` file in the root of this repository:

```env
# Daily.co (WebRTC transport)
DAILY_ROOM_URL=https://your-domain.daily.co/your-room  # optional if DAILY_API_KEY is set
DAILY_API_KEY=your_daily_api_key                       # auto-creates a room if DAILY_ROOM_URL is empty
DAILY_TOKEN=your_daily_participant_token   # optional — leave blank for open rooms

# OpenAI (STT + LLM + TTS + Vision)
OPENAI_API_KEY=sk-...

# SoulX model — choose lite (default) or pro
SOULX_MODEL_TYPE=lite
SOULX_CKPT_DIR=./models/SoulX-FlashHead-1_3B
SOULX_WAV2VEC_DIR=./models/wav2vec2-base-960h
SOULX_COND_IMAGE=./examples/omani_character.png

# Perception engine
PERCEPTION_INTERVAL=3.0          # seconds between Vision-API analyses
```

### 4. Configuration
You must configure the chunking behaviour in `flash_head/configs/infer_params.yaml`. 
For stable streaming with standard SDPA math, use a 33-frame chunk size:
```yaml
frame_num: 33
tgt_fps: 25
sample_rate: 16000
```
*(Note: Reducing `frame_num` further without customised Triton kernels will cause PyTorch compilation `FakeTensor` shape mismatches).*

## 🏃 Running the Bot

```bash
# Install all dependencies
pip install -r requirements.txt && pip install -r requirements_pipecat.txt

# Start the conversational brain + avatar
source .env
python soulx_conversational_bot.py
```

If `DAILY_ROOM_URL` is blank and `DAILY_API_KEY` is set, the bot auto-creates a Daily room at startup and logs the generated room URL.

1. Look for `SoulX Model fully loaded and GPU is pre-warmed.` in the logs.
2. Look for `Starting SoulX Conversational Brain in Daily.co room: …` in the logs.
3. Join the room via the [Daily.co Prebuilt UI](https://www.daily.co/prebuilt/) or a custom Daily.co app — speak to the avatar and it will listen, think, and respond in real time.

You can also run the perception engine standalone for debugging:
```bash
python perception_engine.py
```

## 🧠 Pipeline Architecture

```
User speaks via WebRTC
        │
        ▼
 Daily.co Transport (audio in + Silero VAD)
        │
        ▼
 OpenAI Whisper STT
        │
        ▼
 GPT-4o LLM (tool calling)
   ├─ get_current_time
   ├─ get_visual_context ────► Perception Engine
   │                               └─ Daily.co video subscriber
   │                                   (webcam + screen-share)
   └─ calculate
        │
        ▼
 OpenAI TTS (16 kHz)
        │
        ▼
 WebRTCSyncPusher
   (GPU: audio embedding → 25 fps video frames)
        │
        ▼
 Daily.co (video + audio out) → User sees avatar speaking
```

## 🔮 Future Optimisations & Roadmap
To push the performance boundary even further (and safely lower `frame_num` to 9 for sub-360 ms latency), the following low-level updates are required:
- **Flash Attention 3 Upgrade (YEP-49):** Rewrite the core attention blocks to use TMA asynchronously on Hopper/Blackwell hardware.
- **Custom Triton Kernels (YEP-48):** Port the bare-metal Flash Norm and RoPE Triton kernels to eliminate the PyTorch eager math bottlenecks completely.
