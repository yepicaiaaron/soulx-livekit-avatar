# SoulX Real-Time WebRTC Avatar

Welcome to the **SoulX Real-Time WebRTC Avatar** project! This repository demonstrates a hyper-fast, ultra-low latency real-time conversational AI avatar pipeline. 

**Acknowledgments:** This project is built entirely on top of the incredible **SoulX-FlashHead** architecture. All credit for the base model, distillation techniques, and core inference engine goes to the researchers at Soul-AILab. You can find their original repository and paper here: [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead). Our work focuses strictly on optimizing their engine for continuous WebRTC streaming and sub-second latency.

## 🎯 Project Roadmap & Status

Our ultimate goal is to achieve a full end-to-end conversational latency (including ASR, LLM, TTS, and video rendering) of **under 600 milliseconds**. 

Here is our current progress toward building a sub-700ms diffusion video system:

✅ **Continuous Streaming:** The avatar idles gracefully and responds instantly to audio without dropping connections.
✅ **WebRTC Integration:** We engineered full duplex audio-in/video-out via LiveKit and Pipecat.
✅ **JIT Pre-Warming:** We bypassed 2-minute PyTorch graph compilation delays via dummy audio passes on boot.
✅ **Sub-1.5s Rendering:** We successfully achieved ~1.0-1.5s latency on a single NVIDIA A100 using the `Lite` model.
✅ **Dynamic Animation:** We exposed CFG scaling multipliers, allowing developers to tweak facial animation intensity live.
✅ **LightX2V Autoencoder Integration:** We integrated the highly distilled `lighttaew2_1` / `lightvaew2_1` autoencoders. This reduced VAE decode time to near zero and shrank the memory footprint from ~10GB to ~0.4GB, freeing up massive VRAM bandwidth for the primary Denoise loop.
  - ⏳ *Pending Release: We will upload the VAE integration code and optimized model config files to the public repo shortly.*
✅ **Temporal Window Shrinking:** We successfully reduced the inference batch size from 32 frames (1.28s) down to 16 (0.64s) or 8 (0.32s), drastically cutting the minimum audio buffer requirement.
  - ⏳ *Pending Release: We will publish the dynamic chunk-sizing code and updated inference scripts to the public repo.*
✅ **Continuous Frame Yielding:** We refactored the PyTorch generation loop to `yield` RGB frames sequentially as they decode, effectively staggering the WebRTC push for instant playback.
  - ⏳ *Pending Release: We will push the staggered WebRTC yielding implementation to the public repo.*

⏳ **TensorRT Compilation:** We are currently working to compile the PyTorch graph to NVIDIA TensorRT for aggressive layer fusion and precision calibration (Expected latency drop: ~150ms).
⏳ **Wav2Vec2 Concurrency:** We are moving the audio feature extraction into an asynchronous 20ms micro-chunking thread so embeddings are instantly ready for the video model.
⏳ **FP8/INT8 Quantization:** We plan to quantize the model weights via `bitsandbytes` to eliminate VRAM bandwidth bottlenecks on Hopper architecture.
⏳ **Full Autonomous Agent:** We will pipe ASR (Deepgram) -> LLM (OpenAI) -> TTS (ElevenLabs) directly into the video transport.

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