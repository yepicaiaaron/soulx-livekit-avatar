# SoulX-FlashHead WebRTC Avatar

A fully real-time, low-latency, "echo-back" talking head integration using [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) and [LiveKit](https://livekit.io/) via Pipecat.

## 🚀 Key Achievements
- **Real-Time Lip-Sync:** Captures user audio from a WebRTC room and streams back a lip-synced video avatar in real time.
- **LightX2V Distilled VAE:** Integrates the distilled VAE, heavily reducing the decode time (saving ~35ms per chunk) and minimizing VRAM footprint.
- **Hardware Optimized:** Uses `enable_flash_sdp(True)` to shave ~3ms per denoising step over eager operations.
- **Intelligent Playback Queue:** Binds video frame delivery to the system atomic clock (`time.perf_counter()`) to guarantee mathematically accurate 25fps WebRTC streaming without robotic skipping or "queue drift." The avatar remains smoothly visible via an idle frame during periods of silence.

## 🛠️ Environment Setup & Installation

### 1. Repository Structure
Ensure you have cloned both the base `SoulX-FlashHead` library and this WebRTC service wrapper on the same machine.

1. Clone `SoulX-FlashHead` and download the models (you will need the 1.3B model and `wav2vec2-base-960h`).
2. Clone this repository.

### 2. Symlinks (Crucial Step)
The Python imports and model loaders require exact directory structures. From the root of this repo, create symlinks to your base installation:

```bash
# Link the core model engine
ln -s /path/to/SoulX-FlashHead/SoulX-FlashHead-src/flash_head flash_head

# Link the downloaded models (offline HuggingFace mode)
ln -s /path/to/SoulX-FlashHead/models models
```
*Note: If running with `HF_HUB_OFFLINE=1`, any broken symlinks will cause the engine to aggressively attempt (and fail) to download weights from the internet.*

### 3. Environment Variables
Create a `.env` file in the root of this repository with your LiveKit credentials:

```env
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

### 4. Configuration
You must configure the chunking behavior in `flash_head/configs/infer_params.yaml`. 
For stable streaming with standard SDPA math, use a 33-frame chunk size:
```yaml
frame_num: 33
tgt_fps: 25
sample_rate: 16000
```
*(Note: Reducing `frame_num` further without customized Triton kernels will cause PyTorch compilation `FakeTensor` shape mismatches).*

## 🏃 Running the Bot

Start the service using Python. The engine will undergo a ~3-minute `torch.compile` pre-warming sequence before connecting to the LiveKit room.

```bash
source .env
python3 webrtc_sync.py
```

1. Look for `SoulX Model fully loaded and GPU is pre-warmed.` in the logs.
2. Look for `Connected to soulx-flashhead-room`.
3. Use a generated LiveKit JWT to join the room. **Important:** Generic Meet links without authenticated tokens will drop you into empty fallback rooms.

## 🔮 Future Optimizations & Roadmap
To push the performance boundary even further (and safely lower `frame_num` to 9 for sub-360ms latency), the following low-level updates are required:
- **Flash Attention 3 Upgrade (YEP-49):** Rewrite the core attention blocks to use TMA asynchronously on Hopper/Blackwell hardware.
- **Custom Triton Kernels (YEP-48):** Port the bare-metal Flash Norm and RoPE Triton kernels to eliminate the PyTorch eager math bottlenecks completely.
