# SoulX Real-Time WebRTC Avatar

Welcome to the **SoulX Real-Time WebRTC Avatar** project! This repository demonstrates a hyper-fast, ultra-low latency real-time conversational AI avatar pipeline. 

**Acknowledgments:** This project is built entirely on top of the incredible **SoulX-FlashHead** architecture. All credit for the base model, distillation techniques, and core inference engine goes to the researchers at Soul-AILab. You can find their original repository and paper here: [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead). Our work focuses strictly on optimizing their engine for continuous WebRTC streaming and sub-second latency.

## 🎯 Project Roadmap & Status

Our ultimate goal is to achieve a full end-to-end conversational latency (including ASR, LLM, TTS, and video rendering) of **under 600 milliseconds**. 

Here is our current progress toward building a sub-700ms diffusion video system:

- ✅ **Continuous Streaming:** Built idle/active WebRTC stream handler (100% uptime improvement).
- ✅ **JIT Pre-Warming:** Bypassed PyTorch graph compilation delays (~120,000ms latency reduction on boot).
- ✅ **Sub-1.5s Rendering:** Optimized A100 Lite model inference (~5x faster than real-time playback).
- ✅ **Dynamic Animation:** Exposed CFG scaling for live facial intensity tweaks (Zero latency cost).
- ✅ **LightX2V Autoencoder:** Integrated distilled VAE (Decode time reduced ~35ms/chunk; VRAM footprint reduced by ~9.6GB).
- ✅ **Release LightX2V Integration:** Upload VAE integration code and optimized model config files to public repo.
- ⏳ **Temporal Window Shrinking:** Reduce batch size from 32 to 8 frames (~960ms minimum audio context latency reduction).
- ⏳ **Continuous Frame Yielding:** Yield RGB frames sequentially during decode (~150ms perceived latency reduction).
- ⏳ **TensorRT Compilation:** Compile PyTorch graph to NVIDIA TensorRT (Expected latency reduction: ~150ms).
- ⏳ **Wav2Vec2 Concurrency:** Asynchronous micro-chunking for audio extraction (Expected latency reduction: ~30ms).
- ⏳ **FP8/INT8 Quantization:** Quantize model weights via `bitsandbytes` (Expected rendering speedup: ~30%).
- ⚠️ **Full Autonomous Agent (ElevenLabs Agents API):** Attempted direct WebSocket integration with ElevenLabs Agents API. Currently blocked by undocumented `1008 policy violation` errors when sending base64 `user_audio_chunk` payloads. See `feat-elevenlabs-agent` branch for isolated test scripts and integration attempts.

### 🧪 Experimental Branch: ElevenLabs Agents Integration
This branch (`feat-elevenlabs-agent`) contains experimental scripts for integrating the ElevenLabs Conversational AI Agents API natively via WebSockets.
- `elevenlabs_agent_sync.py`: Full pipeline (LiveKit Mic -> ElevenLabs Agent -> SoulX Video).
- `test_elevenlabs_only.py`: Isolated audio pipeline (LiveKit Mic -> ElevenLabs Agent -> LiveKit Speaker).
- **Status:** Failing. The ElevenLabs WebSocket API forcefully disconnects with `1008 (policy violation) Invalid message received` when transmitting 16kHz 16-bit PCM base64 audio chunks.

## 🛠 Prerequisites & Setup

### Hardware Requirements
- An **NVIDIA A100 (40GB/80GB)** or **H100** GPU is strongly recommended for real-time inference speeds.
- Linux OS (tested on Ubuntu 22.04 on Google Cloud Platform).

### Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yepicaiaaron/soulx-livekit-avatar.git
   cd soulx-livekit-avatar
   ```
2. **Install dependencies:**
   Ensure you have Python 3.10+ and CUDA 12.x installed.
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_pipecat.txt
   ```
3. **Download the Models:**
   Download the `SoulX-FlashHead-1_3B` and `wav2vec2-base-960h` models from HuggingFace and place them in the `models/` directory.

## 🏃 Running the Pipeline

### 1. Launch a Local LiveKit Server
To handle the WebRTC streams locally without browser security issues, run the open-source LiveKit server:
```bash
curl -sSL https://get.livekit.io | bash
livekit-server --dev --bind 0.0.0.0
```

### 2. Configure Environment Variables
Set your LiveKit credentials (use `devkey` and `secret` if using the local `--dev` server):
```bash
export LIVEKIT_URL=ws://127.0.0.1:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secret
```

### 3. Start the Avatar Script
Run the real-time sync script. The script will take ~90 seconds to pre-warm the GPU.
```bash
python3 webrtc_sync.py
```

### 4. Connect and Interact
Generate a connection token for your LiveKit room and use a frontend client like LiveKit Meet to join. Turn on your microphone, and the avatar will mimic your speech in real-time!

## License
The SoulX-FlashHead model weights and architecture are licensed under the Apache 2.0 License. All modifications in this repository remain open-source.