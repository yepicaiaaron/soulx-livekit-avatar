# SoulX-FlashHead Pipecat Bot

Real-time talking head generation bot using Pipecat orchestration framework with LiveKit WebRTC transport.

## Architecture

```
┌─────────────────┐     WebRTC      ┌──────────────────────────────────────────────────┐
│   LiveKit       │◄───Audio/Video──►│  SoulX Conversational Bot                        │
│   Server        │                  │                                                  │
│                 │                  │  ┌────────────┐    ┌─────────┐    ┌────────────┐ │
└─────────────────┘                  │  │  STT       │───►│   LLM   │───►│    TTS     │ │
        ▲                            │  └────────────┘    └─────────┘    └──────┬─────┘ │
        │                            │         ▲                                │       │
        │                            │         │                                ▼       │
   ┌────┴────┐                       │  ┌──────────────┐                 ┌────────────┐ │
   │  User   │◄──────────────────────┼──│ LiveKit      │                 │ SoulXFlash │ │
   │ Browser │    WebRTC Video/Audio │  │ Transport    │◄────────────────│ HeadService│ │
   └─────────┘                       │  └──────────────┘                 └────────────┘ │
                                    └──────────────────────────────────────────────────┘
```

## Features

- **Full Conversational Pipeline**: Low-latency STT (Deepgram), LLM (OpenAI), and TTS (Cartesia).
- **Real-time audio-to-video streaming**: Audio from TTS is streamed into SoulX-FlashHead model.
- **WebRTC output**: Generated video frames are streamed back via LiveKit.
- **Pipecat orchestration**: Modular pipeline architecture for rapid response.

## Files

| File | Description |
|------|-------------|
| `soulx_conversational_bot.py` | Main conversational orchestration script |
| `pipecat_soulx_service.py` | SoulXFlashHeadService implementation |
| `requirements_pipecat.txt` | Pipecat, LiveKit, and AI service dependencies |
| `run_conversational_bot.sh` | Launch script for the conversational bot |
| `setup_node.sh` | Node environment setup script |
| `Dockerfile` | Container image definition |

## Quick Start

### 1. Setup Environment

```bash
./setup_node.sh
pip install -r requirements_pipecat.txt
```

### 2. Configure API Keys and LiveKit

```bash
export OPENAI_API_KEY=your-openai-key
export DEEPGRAM_API_KEY=your-deepgram-key
export CARTESIA_API_KEY=your-cartesia-key
export LIVEKIT_URL=wss://your-livekit-server:7880
export LIVEKIT_TOKEN=your-access-token
export LIVEKIT_ROOM=soulx-flashhead-room
```

### 3. Run the Bot

```bash
./run_conversational_bot.sh
```

## Quick Start

### Prerequisites

- NVIDIA GPU (RTX 4090 for Lite model, dual RTX 5090 for Pro model real-time)
- CUDA 12.8+
- LiveKit server
- Model weights from HuggingFace

### 1. Setup Environment

```bash
./setup_node.sh
```

### 2. Download Models

```bash
# FlashHead model
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B

# Wav2Vec audio encoder
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
```

### 3. Configure LiveKit

```bash
export LIVEKIT_URL=wss://your-livekit-server:7880
export LIVEKIT_TOKEN=your-access-token
export LIVEKIT_ROOM=soulx-flashhead-room
```

### 4. Run the Bot

```bash
./run_bot.sh
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SOULX_CKPT_DIR` | `./models/SoulX-FlashHead-1_3B` | Model checkpoint directory |
| `SOULX_WAV2VEC_DIR` | `./models/wav2vec2-base-960h` | Wav2Vec model directory |
| `SOULX_MODEL_TYPE` | `lite` | Model type: `lite` or `pro` |
| `SOULX_COND_IMAGE` | `./examples/cond_image.png` | Condition image for generation |
| `SOULX_SEED` | `42` | Random seed |
| `SOULX_USE_FACE_CROP` | `false` | Enable face cropping |
| `LIVEKIT_URL` | `wss://localhost:7880` | LiveKit server URL |
| `LIVEKIT_TOKEN` | - | LiveKit access token |
| `LIVEKIT_ROOM` | `soulx-flashhead-room` | Room name |

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or with custom environment
docker-compose -e LIVEKIT_URL=wss://your-server:7880 up
```

## Pipeline Details

### Audio Flow

1. **LiveKitTransport (input)**: Receives 16kHz, 16-bit PCM audio via WebRTC
2. **SoulXFlashHeadService**: 
   - Buffers audio chunks (800ms for 25fps)
   - Runs wav2vec encoding
   - Generates video frames via FlashHead model
3. **LiveKitTransport (output)**: Sends generated video frames via WebRTC

### Video Flow

1. SoulX-FlashHead generates frames at 25fps
2. RawImageFrames (512x512 RGB) are yielded to transport
3. Transport encodes and streams via WebRTC

## Troubleshooting

### CUDA Out of Memory

- Use `lite` model instead of `pro`
- Reduce `frame_num` in config (requires model change)
- Enable model offloading

### High Latency

- Check GPU utilization with `nvidia-smi`
- Ensure SageAttention is installed for Pro model
- Verify LiveKit server proximity

### Audio/Video Sync Issues

- Check network latency to LiveKit server
- Monitor buffer levels in logs
- Adjust `cached_audio_duration` if needed

## References

- [YEP-16] LiveKit Transport Integration
- [YEP-17] Real-time Audio-to-Video Streaming
- [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead)
- [Pipecat](https://github.com/pipecat-ai/pipecat)
- [LiveKit](https://livekit.io/)

## License

See SoulX-FlashHead repository for model licensing.
