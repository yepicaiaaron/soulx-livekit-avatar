# SoulX Daily.co Talking Head Avatar — Deployment Guide

This repository contains the integrated SoulX-FlashHead talking head avatar with:
- **Daily.co** — WebRTC transport
- **Pipecat** — real-time AI pipeline orchestration
- **OpenAI** — Whisper STT, GPT-4o LLM (with tool calling), and TTS
- **Perception Engine** — real-time visual analysis of webcam and screen-share feeds via OpenAI Vision

## Architecture Overview

```
User speaks in WebRTC session
        │
        ▼
 Daily.co Transport (audio in, with Silero VAD)
        │
        ▼
 OpenAI Whisper STT          ← speech-to-text
        │
        ▼
 OpenAI GPT-4o LLM           ← conversational brain + tool calling
   ├─ get_current_time
   ├─ get_visual_context ─────► Perception Engine
   │                               └─ Daily.co video subscriber
   │                                   (webcam + screen-share tracks)
   └─ calculate
        │
        ▼
 OpenAI TTS                  ← text-to-speech (16 kHz)
        │
        ▼
 WebRTCSyncPusher             ← SoulX-FlashHead avatar animation
        │                        (GPU inference: audio embedding → video frames)
        ▼
 Daily.co Transport (video + audio out)
        │
        ▼
User sees lip-synced avatar speaking in real time
```

## Prerequisites

1. **Daily.co** — a Daily.co room URL, and optionally a participant token for private rooms.
2. **OpenAI** — API key with access to Whisper, GPT-4o, and TTS.
3. **SoulX-FlashHead models** — download via `bash setup_lightning.sh` (see README for details).

## Deployment on Render.com

### 1. Repository Setup

Push this codebase to a GitHub repository.

### 2. Render.com Service Creation

1. Log in to [Render.com](https://dashboard.render.com/).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository.
4. **Configuration:**
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && pip install -r requirements_pipecat.txt`
   - **Start Command:** `python soulx_conversational_bot.py`
   - **Plan:** GPU instance recommended for real-time inference.

### 3. Environment Variables

| Key | Required | Value / Description |
| :--- | :---: | :--- |
| `OPENAI_API_KEY` | ✅ | OpenAI API key (Whisper STT + GPT-4o LLM + TTS). |
| `DAILY_ROOM_URL` | ✅ | Daily.co room URL (`https://your-domain.daily.co/your-room`). |
| `DAILY_TOKEN` | | *(Optional)* Daily.co participant token for private rooms. |
| `SOULX_MODEL_TYPE` | | `lite` (default) or `pro`. |
| `SOULX_CKPT_DIR` | | Path to model checkpoints (default: `./models/SoulX-FlashHead-1_3B`). |
| `SOULX_WAV2VEC_DIR` | | Path to wav2vec directory (default: `./models/wav2vec2-base-960h`). |
| `SOULX_COND_IMAGE` | | Avatar portrait image (default: `./examples/omani_character.png`). |
| `PERCEPTION_INTERVAL` | | Seconds between Vision-API analyses (default: `3.0`). |
| `ELEVENLABS_API_KEY` | | *(Legacy)* ElevenLabs key — not used by default pipeline. |
| `DEEPGRAM_API_KEY` | | *(Legacy)* Deepgram key — not used by default pipeline. |

### 4. Blueprint Deployment (Alternative)

Use the included `render.yaml` with Render Blueprints:
1. Go to **Blueprints** in the Render dashboard.
2. Connect your repository.
3. Render will detect `render.yaml` and prompt you for the required environment variables.

## Local Testing

```bash
# 1. Create and activate virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt && pip install -r requirements_pipecat.txt

# 3. Set environment variables
cp .env.example .env  # then fill in your keys

# 4. Run the conversational bot (requires CUDA GPU + SoulX models)
python soulx_conversational_bot.py

# 5. (Optional) Run the perception engine standalone for debugging
python perception_engine.py
```

Join your Daily.co room via the [Daily.co Prebuilt UI](https://www.daily.co/prebuilt/) or any Daily-compatible app to speak with the avatar in real time.

## Key Files

| File | Description |
| :--- | :--- |
| `soulx_conversational_bot.py` | **Main entry point.** Full Pipecat pipeline: Daily.co audio in → STT → LLM → TTS → SoulX avatar. |
| `perception_engine.py` | **Perception engine.** Daily.co video subscriber + OpenAI Vision analysis. |
| `webrtc_sync.py` | **Avatar pusher.** `WebRTCSyncPusher` FrameProcessor — drives SoulX-FlashHead GPU inference and publishes lip-synced video+audio to Daily.co. |
| `requirements_pipecat.txt` | Pipecat, Daily.co, and AI service dependencies. |
| `render.yaml` | Render.com deployment blueprint. |
