Real Time Diffusion WebRTC Avatar

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

## ⚡ Low-Latency Mode (YEP-48 + YEP-49) — IMPLEMENTED

The sub-1s latency work from the roadmap is now in the tree:

- **YEP-48 — Fused Triton kernels** (`flash_head/kernels/`): real-math fp32 RoPE
  (replacing the complex-float64 eager path — the source of both the fp64
  throughput cliff and the `FakeTensor` compile failures at small `frame_num`),
  single-pass RMSNorm, and a fused AdaLN `LN(x)*(1+scale)+shift` kernel. The
  RoPE op registers via `torch.library.custom_op`, so `torch.compile` treats it
  as opaque and `frame_num: 9` compiles cleanly. Disable with
  `FLASH_HEAD_FUSED_KERNELS=0`.
- **YEP-49 — FlashAttention-3 first**: attention dispatch now prefers FA3
  (TMA async pipeline on Hopper/Blackwell) over SageAttention/FA2/SDPA.
  Override with `FLASH_HEAD_ATTN=fa3|sage|fa2|sdpa`. Build FA3 into the Docker
  image with `--build-arg INSTALL_FA3=1`.
- **Latency profiles** (`FLASH_HEAD_PROFILE`):
  | Profile | frame_num | net new frames/chunk | chunk budget | audio lookahead |
  |---|---|---|---|---|
  | `default` | 33 | 28 | 1120 ms | 1320 ms |
  | `balanced` | 13 | 8 | 320 ms | 520 ms |
  | `lowlat` | 9 | 4 | 160 ms | 360 ms |

### Live test (one command, on the GPU box)
```bash
cp .env.example .env   # fill in your (ROTATED) LiveKit credentials
./deploy/live_test.sh  # gates: tests -> kernel parity -> quality PSNR A/B -> latency -> launch
```
The script refuses to launch unless fused kernels match eager numerics, the
PSNR quality gate passes (eager-vs-fused ≥ 35 dB), and the GPU sustains the
chosen profile's real-time budget; it auto-falls-back `lowlat → balanced` if
the 160 ms budget doesn't hold on your hardware.

> **Security note:** LiveKit credentials now come exclusively from the
> environment. The key/secret previously hardcoded in `webrtc_sync.py` are in
> git history — rotate them in the LiveKit dashboard.
