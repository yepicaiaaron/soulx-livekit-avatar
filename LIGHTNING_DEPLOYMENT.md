# 🚀 SoulX Live Avatar — Lightning.ai A100 Deployment Guide

> **Goal:** Get a photorealistic AI talking-head avatar running in the cloud in under 30 minutes —
> you only need two API keys.

---

## 🧰 What You Need (just 2 things)

| Thing | Where to get it | Cost |
|-------|-----------------|------|
| **Daily.co API Key** | Sign up free at [daily.co](https://www.daily.co) → Dashboard → **Developers** → copy the API key | Free tier available |
| **OpenAI API Key** | Sign up at [platform.openai.com](https://platform.openai.com) → top-right menu → **API keys** → **+ Create new** | Pay-per-use |

Keep these two keys handy — you'll paste them in **Step 3**.

---

## Step 1 — Create a Lightning.ai Studio with an A100

1. Go to **[lightning.ai](https://lightning.ai)** and sign in (or create a free account).
2. Click the big **"New Studio"** button.
3. Under **"Select compute"**, scroll down and pick:
   ```
   NVIDIA A100 · 80 GB
   ```
4. Click **"Start Studio"**.
5. Wait about 30–60 seconds for the Studio to fully boot. You'll see a VS Code / JupyterLab-style interface.

> 💡 **What is a Studio?** It's a GPU-powered computer in the cloud — like renting a very powerful PC
> by the minute. Everything you install stays there between sessions.

---

## Step 2 — Add Your API Keys as Secrets

Your API keys must be stored as **Secrets** — this keeps them private and makes them available
to all scripts automatically.

1. In the Studio, look at the **left sidebar** (the vertical icon bar on the far left).
2. Click the **🔒 lock icon** (labelled **"Secrets"** or **"Environment"**).
3. Click **"+ Add Secret"** and add each of the following:

   | Secret Name | Secret Value |
   |-------------|--------------|
   | `DAILY_API_KEY` | Paste your Daily.co API key here |
   | `OPENAI_API_KEY` | Paste your OpenAI API key here |

4. After adding both secrets, click **"Save"** (or they may save automatically).

> 💡 These secrets become environment variables. Every terminal you open in this Studio will
> automatically have `$DAILY_API_KEY` and `$OPENAI_API_KEY` available.

---

## Step 3 — Open a Terminal

1. In the Studio, click the **Terminal** icon in the left sidebar (looks like `>_`)
   — OR press **Ctrl + `` ` ``** (backtick).
2. A terminal window opens at the bottom. You're ready to type commands.

---

## Step 4 — Clone the Repository

Copy and paste this entire block into the terminal, then press **Enter**:

```bash
git clone https://github.com/trip-nine/soulx-livekit-avatar.git
cd soulx-livekit-avatar
```

---

## Step 5 — Run the Setup Script

Copy and paste this single command:

```bash
bash setup_lightning.sh
```

**What it does (you don't need to do anything while it runs):**
- ✅ Installs PyTorch with CUDA 12.8 support
- ✅ Installs all Python dependencies (Pipecat, Daily.co, OpenAI SDK, etc.)
- ✅ Installs FlashAttention 2 (GPU speed booster)
- ✅ Downloads the **SoulX-FlashHead Model_Lite** weights (~5–8 GB) from HuggingFace
- ✅ Downloads the **wav2vec2** audio encoder from HuggingFace

⏳ **This takes about 15–25 minutes** — mostly for downloading the models.
When you see `✅ Setup complete!` you are ready.

---

## Step 6 — Start the Avatar

```bash
python gradio_room_launcher.py
```

You'll see:

```
Starting SoulX Gradio Room Launcher on port 7860…
```

Leave this running. Do **not** close the terminal.

---

## Step 7 — Expose Port 7860 (Get Your Public URL)

1. In the Studio left sidebar, click the **"Ports"** icon (looks like a plug 🔌).
2. Click **"+ Add Port"**.
3. Type `7860` and set it to **Public**.
4. Click **"Add"**.
5. Lightning.ai will show you a **public URL** like:
   ```
   https://8abc1234.lightning.ai/
   ```

**This is your shareable URL.** Anyone with this link can use the avatar — no installation needed on their end.

---

## Step 8 — Use the Avatar 🎉

1. Open the public URL in your browser.
2. Click the green **"▶ Start Session"** button.
3. The app will automatically:
   - Create a private Daily.co video room
   - Launch the AI avatar on the GPU
4. A video call window appears inside the page.
5. **Allow your microphone and camera** when the browser asks.
6. Wait about **40 seconds** for the GPU warmup — then speak to the avatar!

> 🎙️ **Tip:** Wear headphones to prevent echo.

---

## Stopping & Restarting

| Action | What to do |
|--------|------------|
| End a conversation | Click **"⏹ End Session"** in the Gradio UI |
| Start a new conversation | Click **"▶ Start Session"** again |
| Stop the server | Press `Ctrl + C` in the terminal |
| Restart the server | Run `python gradio_room_launcher.py` again |

---

## Quick Reference: All Environment Variables

The only two you **must** set are `DAILY_API_KEY` and `OPENAI_API_KEY`.
Everything else has sensible defaults.

| Secret / Env Var | Required | Default | Description |
|------------------|:--------:|---------|-------------|
| `DAILY_API_KEY` | ✅ | — | Daily.co API key (creates rooms automatically) |
| `OPENAI_API_KEY` | ✅ | — | OpenAI key for Whisper STT + GPT-4o + TTS |
| `SOULX_MODEL_TYPE` | | `lite` | `lite` (96 fps, 1 GPU) or `pro` (higher quality, 2 GPUs) |
| `SOULX_CKPT_DIR` | | `./models/SoulX-FlashHead-1_3B` | Model checkpoint directory |
| `SOULX_WAV2VEC_DIR` | | `./models/wav2vec2-base-960h` | wav2vec2 audio encoder directory |
| `SOULX_COND_IMAGE` | | `./examples/omani_character.png` | Avatar portrait image path |
| `PERCEPTION_INTERVAL` | | `3.0` | Seconds between webcam/screen analysis (OpenAI Vision) |

---

## Changing the Avatar Portrait

To use your own photo as the avatar face:

1. Upload your portrait image to the Studio (drag-drop into the file explorer).
2. In the Studio Secrets panel, add:
   ```
   SOULX_COND_IMAGE = /path/to/your/portrait.png
   ```
3. Restart the launcher: `python gradio_room_launcher.py`

> 📸 The portrait should be a clear, front-facing photo. PNG or JPG, any resolution.

---

## Troubleshooting

### "❌ DAILY_API_KEY is not set"
You haven't added the secret yet. Go back to **Step 2** and add `DAILY_API_KEY` in the Secrets panel.
Then restart the launcher: press `Ctrl+C` and run `python gradio_room_launcher.py` again.

### "❌ OPENAI_API_KEY is not set"
Same as above — add `OPENAI_API_KEY` in the Secrets panel and restart.

### The video call window is blank / not loading
Click the link **"Open in a new tab ↗"** shown below the video window.

### The avatar takes more than 2 minutes to respond
The GPU warmup is still running. Wait a bit longer — it only happens once per session.

### "CUDA not available" error during setup
You picked a CPU-only machine. Go back to Step 1 and make sure to select the **A100 · 80 GB** option.

### Models fail to download
Run just the download commands manually:
```bash
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
    --include "Model_Lite/**" "VAE_LTX/**" \
    --local-dir ./models/SoulX-FlashHead-1_3B

huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir ./models/wav2vec2-base-960h
```

---

## Architecture (for the curious)

```
User opens Gradio URL
       │
       ▼
gradio_room_launcher.py
  ├─ Calls Daily.co REST API → creates a fresh video room
  ├─ Launches soulx_conversational_bot.py (GPU process)
  └─ Embeds Daily.co Prebuilt UI in the browser page
       │
       ▼
soulx_conversational_bot.py (Pipecat pipeline)
  ├─ Daily.co WebRTC audio in  ←── User speaks
  ├─ OpenAI Whisper STT        ←── speech → text
  ├─ GPT-4o LLM                ←── think + tool calling
  ├─ OpenAI TTS                ←── text → speech (16 kHz)
  └─ SoulX-FlashHead GPU       ←── audio → 25 fps lip-synced video
       │
       ▼
Daily.co WebRTC video+audio out  ──► User sees & hears avatar
```

**Model used:** SoulX-FlashHead **Model_Lite** (1.3B params)
- Runs at **96 FPS** on a single A100 — far above real-time (25 FPS needed)
- Source: [Soul-AILab/SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead)
- Weights: [huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)
