# Soulx LiveKit Talking Head Avatar Deployment Guide

This repository contains the integrated Soulx-FlashHead talking head avatar with LiveKit, Deepgram, OpenAI, and ElevenLabs TTS.

## Prerequisites

1.  **LiveKit Cloud or Self-Hosted:** You need a LiveKit project and its API Key/Secret.
2.  **API Keys:**
    *   **ElevenLabs:** For high-quality text-to-speech.
    *   **OpenAI:** For conversational intelligence (LLM).
    *   **Deepgram:** For rapid speech-to-text (STT).
3.  **Render.com Account:** For hosting the web service.

## Deployment on Render.com

This project is configured to be deployed as a **Render Web Service**.

### 1. Repository Setup

1.  Create a new private repository on GitHub (e.g., `yepicaiaaron/soulx-livekit-avatar`).
2.  Push this entire codebase to your new repository.

### 2. Render.com Service Creation

1.  Log in to [Render.com](https://dashboard.render.com/).
2.  Click **New +** and select **Web Service**.
3.  Connect your GitHub repository `soulx-livekit-avatar`.
4.  **Configuration:**
    *   **Name:** `soulx-livekit-avatar`
    *   **Region:** Select the one closest to you.
    *   **Branch:** `main`
    *   **Runtime:** `Python 3`
    *   **Build Command:** `pip install -r requirements.txt && pip install -r requirements_pipecat.txt`
    *   **Start Command:** `python soulx_conversational_bot.py`
    *   **Plan:** We recommend at least a **Starter** or **Pro** plan if you have heavy inference needs (though SoulX is optimized, real-time video generation benefits from resources).

### 3. Environment Variables

In the **Environment** tab of your Render service, add the following variables:

| Key | Value | Description |
| :--- | :--- | :--- |
| `ELEVENLABS_API_KEY` | `your_key` | Your ElevenLabs API key. |
| `OPENAI_API_KEY` | `your_key` | Your OpenAI API key. |
| `DEEPGRAM_API_KEY` | `your_key` | Your Deepgram API key. |
| `LIVEKIT_URL` | `wss://...` | Your LiveKit Project URL. |
| `LIVEKIT_API_KEY` | `your_key` | Your LiveKit API Key. |
| `LIVEKIT_API_SECRET` | `your_secret` | Your LiveKit API Secret. |
| `SOULX_MODEL_TYPE` | `lite` | Use `lite` for faster inference or `pro` for higher quality. |
| `SOULX_CKPT_DIR` | `./models/SoulX-FlashHead-1_3B` | Path to model checkpoints. |
| `SOULX_WAV2VEC_DIR` | `./models/wav2vec2-base-960h` | Path to wav2vec directory. |

### 4. Blueprint Deployment (Alternative)

Alternatively, you can use the included `render.yaml` to deploy via Render Blueprints:
1.  Go to **Blueprints** in the Render dashboard.
2.  Connect your repository.
3.  Render will automatically detect the `render.yaml` and prompt you for the required environment variables.

## Local Testing

1.  Create a virtual environment: `python -m venv venv && source venv/bin/activate`
2.  Install dependencies: `pip install -r requirements.txt && pip install -r requirements_pipecat.txt`
3.  Set your environment variables (e.g., in a `.env` file).
4.  Run the bot: `python soulx_conversational_bot.py`
5.  Join the corresponding room in the [LiveKit Sandbox](https://sandbox.livekit.io/) to see your avatar in action.
