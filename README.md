# SoulX Real-Time WebRTC Avatar

Welcome to the **SoulX Real-Time WebRTC Avatar** project! This repository demonstrates how to build a hyper-fast, ultra-low latency real-time conversational AI avatar pipeline using the **SoulX-FlashHead** model and **LiveKit** WebRTC.

By combining the power of PyTorch `Torchinductor` JIT compilation, background asynchronous processing, and LiveKit's WebRTC transport, we have achieved a fully continuous, perfectly synced audio-to-video pipeline with a latency of just **~1.0 - 1.5 seconds** on a single NVIDIA A100 GPU!

## 🚀 Features
- **Real-Time Lip Sync:** Generates perfectly synced video frames directly from incoming WebRTC audio (e.g., your microphone or an LLM/TTS stream).
- **Ultra-Low Latency:** Renders a 1.28s audio chunk in just ~0.25s using the SoulX `Lite` model.
- **Continuous Streaming:** The avatar idles gracefully and responds instantly to audio without dropping connections or timing out.
- **Dynamic Animation Intensity:** Customize the responsiveness of the avatar's facial expressions on the fly by tweaking the audio embedding multiplier.

## 🛠 Prerequisites & Setup

### Hardware Requirements
- We recommend an **NVIDIA A100 (40GB/80GB)** or **H100** GPU for real-time inference speeds.
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
To handle the WebRTC streams locally, download and run the open-source LiveKit server:
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
Run the real-time sync script. The script will take ~90 seconds to pre-warm the GPU (Torchinductor graph compilation).
```bash
python3 webrtc_sync.py
```

### 4. Connect and Interact
Generate a connection token for the LiveKit room (`aarons-private-soulx-room-20260315-1341`) and use a frontend client like LiveKit Meet to join. Turn on your microphone, and the avatar will mimic your speech in real-time!

## 🔮 Future Improvements & Roadmap
To push latency down to sub-500ms and improve visual quality, we are actively exploring the following optimizations:

- **TensorRT Compilation:** Compiling the PyTorch graph to NVIDIA TensorRT for aggressive layer fusion and precision calibration, potentially halving the 0.25s render time.
- **DeepCache / PrunaAI Integration:** Implementing caching mechanisms to skip high-frequency details on alternating diffusion steps for 2x-4x speedups without significant quality loss.
- **FP8/INT8 Quantization:** Quantizing model weights using `bitsandbytes` to alleviate VRAM bandwidth bottlenecks on Hopper/Ampere architectures.
- **Temporal Window Shrinking:** Reducing the inference batch size from 32 frames (1.28s) to 16 (0.64s) or 8 (0.32s) to drastically cut the minimum audio buffer requirement.
- **Pipeline & Transport Overlap:** Refactoring the generation loop to `yield` RGB frames sequentially as they are decoded, staggering the WebRTC push to shave off an additional 100-200ms.
- **Wav2Vec2 Concurrency:** Moving the audio feature extraction into an asynchronous micro-chunking thread so the embeddings are instantly ready for the video model.

## License
The SoulX-FlashHead model weights and architecture are licensed under the Apache 2.0 License.