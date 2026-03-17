# SoulX WebRTC Continuous Sync - Checkpoint Documentation

This checkpoint represents a fully working, stable, zero-stutter WebRTC pipeline for the SoulX-FlashHead avatar.

## The Architecture
The `webrtc_sync.py` script replaces the closed ElevenLabs TTS loop with an open, continuously listening WebRTC tunnel.
1. The avatar idles in the LiveKit room, publishing a silent `AudioFrame` and a static base image `VideoFrame`.
2. As soon as the user speaks into the room microphone, the script receives the raw `AudioRawFrame` from LiveKit.
3. The audio is buffered into `1.28`-second chunks (matching the `slice_len` required by the FlashHead model).
4. The background generation loop (`_generation_loop`) passes the chunks to the GPU, which runs the Torchinductor-compiled graph to render 32 corresponding lip-sync video frames.
5. The `_video_loop` playback queue flawlessly merges the original raw audio bytes with the rendered RGBA video frames and publishes them back to the LiveKit room in real-time (~1 second delay).

## Critical Stability Fixes Applied
1. **Pre-warming the GPU:** The PyTorch Torchinductor JIT compile takes ~1-2 minutes the first time the model runs `forward()`. We now do a "dummy audio pass" inside `webrtc_sync.py` *before* joining the room, preventing 2-minute dropped-frame lags when the user starts speaking.
2. **TypeError Fix:** `np.zeros(self.sample_rate // self.tgt_fps, dtype=np.int16)` crashed Pipecat's idle loop because `16000 // 25.0` evaluates to `640.0` (a float). All instances are now wrapped in `int(...)` to prevent silent queue drops.
3. **Pipecat Idle Timeout:** `PipelineTask` inherently kills pipelines that haven't received audio after 5 minutes. We disabled this by explicitly setting `idle_timeout_secs=None` and `cancel_on_idle_timeout=False` inside `PipelineParams`.
4. **Immediate Publishing:** Instead of waiting for a downstream audio frame, we force the video and audio tracks to publish the moment the `StartFrame` flows through the pipeline. This ensures LiveKit Meet doesn't render a green spinner while waiting for the avatar to show up.
