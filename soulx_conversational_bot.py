"""Interruptible conversational SoulX avatar over LiveKit.

This is the real conversational loop that DEPLOYMENT.md referenced but was
never committed. Unlike webrtc_sync.py (echo-back: lip-syncs the user's own
mic), this wires a full duplex agent:

    mic -> VAD -> Deepgram STT -> OpenAI LLM -> ElevenLabs TTS
        -> SoulX avatar (lip-syncs the BOT's speech) -> LiveKit

Barge-in is the whole point. VAD is on and interruptions are enabled, so
when the user starts talking mid-response, Pipecat emits a
StartInterruptionFrame; we drop everything queued in the avatar generator
and playback buffers so the bot stops speaking *and* the face stops moving
within a frame or two, instead of talking over the user.

Env (see .env.example): LIVEKIT_URL/API_KEY/API_SECRET, DEEPGRAM_API_KEY,
OPENAI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, plus the SOULX_*
and FLASH_HEAD_* model/latency vars.

NOTE: not yet smoke-run end-to-end (needs a GPU host + the four API keys).
Frame-class names track Pipecat 0.0.10x; if a class moved, the try/except
imports below localize the fix.
"""

import asyncio
import collections
import os
import time

import cv2
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams

from livekit import api, rtc
import torch
from flash_head.inference import (
    get_pipeline, get_base_data, get_infer_params,
    get_audio_embedding, run_pipeline,
)


class SoulXAvatarProcessor(FrameProcessor):
    """Lip-syncs the bot's TTS audio and publishes synced A/V to LiveKit.

    Sits at the tail of the pipeline consuming TTSAudioRawFrame. Owns its own
    LiveKit video+audio tracks (SoulX must emit both together, frame-locked).
    Interruption frames flush every buffer so barge-in is immediate.
    """

    def __init__(self, transport, model_pipeline, **kwargs):
        super().__init__(**kwargs)
        self.transport = transport
        self.model_pipeline = model_pipeline
        self.width = self.height = 512

        p = get_infer_params()
        self.sample_rate = p["sample_rate"]
        self.tgt_fps = p["tgt_fps"]
        self.cached_audio_duration = p["cached_audio_duration"]
        self.frame_num = p["frame_num"]
        self.motion_frames_num = p["motion_frames_num"]
        self.slice_len = self.frame_num - self.motion_frames_num

        self.video_source = rtc.VideoSource(self.width, self.height)
        self.video_track = rtc.LocalVideoTrack.create_video_track("bot-video", self.video_source)
        self.audio_source = rtc.AudioSource(self.sample_rate, 1)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("bot-audio", self.audio_source)

        self.cached_len = self.sample_rate * self.cached_audio_duration
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.audio_dq = collections.deque([0.0] * self.cached_len, maxlen=self.cached_len)
        self.audio_slice_samples = self.slice_len * self.sample_rate // self.tgt_fps

        self.tts_float_buffer = []
        self.tts_byte_buffer = bytearray()
        self.playback_queue = collections.deque()
        self.generation_queue = asyncio.Queue()

        # Monotonically increasing epoch: bumped on every interruption. Audio
        # chunks and generated frames carry the epoch they were created under;
        # anything from a stale epoch is discarded rather than played.
        self.epoch = 0

        idle = cv2.imread(os.environ.get("SOULX_COND_IMAGE", "examples/omani_character.png"))
        idle = cv2.resize(idle, (512, 512)) if idle is not None else np.zeros((512, 512, 3), np.uint8)
        self.idle_rgba = cv2.cvtColor(idle, cv2.COLOR_BGR2RGBA)

        self._started = False

    # ------------------------------------------------------------------ #
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not self._started:
            self._started = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())

        if isinstance(frame, (StartInterruptionFrame,)):
            # Barge-in: abandon this response entirely.
            self._flush_for_interruption()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSAudioRawFrame):
            self._ingest_tts_audio(frame.audio)
            # Don't forward bot audio to transport.output — we publish our own
            # frame-locked audio track alongside the video instead.
            return

        await self.push_frame(frame, direction)

    def _flush_for_interruption(self):
        self.epoch += 1
        self.tts_float_buffer.clear()
        self.tts_byte_buffer.clear()
        # Drain the generation queue.
        try:
            while True:
                self.generation_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        self.playback_queue.clear()
        logger.info(f"[barge-in] flushed; epoch -> {self.epoch}")

    def _ingest_tts_audio(self, audio: bytes):
        self.tts_byte_buffer.extend(audio)
        floats = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self.tts_float_buffer.extend(floats.tolist())

        while len(self.tts_float_buffer) >= self.audio_slice_samples:
            chunk_floats = np.array(self.tts_float_buffer[:self.audio_slice_samples])
            self.tts_float_buffer = self.tts_float_buffer[self.audio_slice_samples:]
            nbytes = self.audio_slice_samples * 2
            chunk_bytes = bytes(self.tts_byte_buffer[:nbytes])
            self.tts_byte_buffer = self.tts_byte_buffer[nbytes:]
            self.generation_queue.put_nowait((self.epoch, chunk_floats, chunk_bytes))

    # ------------------------------------------------------------------ #
    async def _generation_loop(self):
        while True:
            epoch, chunk_floats, chunk_bytes = await self.generation_queue.get()
            if epoch != self.epoch:
                continue  # interrupted before we got to it
            gen_start = time.perf_counter()

            self.audio_dq.extend(chunk_floats.tolist())
            audio_array = np.array(self.audio_dq)

            def run_infer():
                emb = get_audio_embedding(self.model_pipeline, audio_array,
                                          self.audio_start_idx, self.audio_end_idx)
                video = run_pipeline(self.model_pipeline, emb)
                return video.cpu().numpy()

            try:
                video_np = await asyncio.to_thread(run_infer)
            except Exception as e:
                logger.error(f"inference error: {e}")
                continue

            if epoch != self.epoch:
                continue  # interruption landed while the GPU was busy

            n = video_np.shape[0]
            bytes_per = len(chunk_bytes) // n
            for i in range(n):
                v = video_np[i]
                if v.shape[0] == 3:
                    v = np.transpose(v, (1, 2, 0))
                rgba = cv2.cvtColor(v.astype(np.uint8), cv2.COLOR_RGB2RGBA)
                self.playback_queue.append((epoch, rgba, chunk_bytes[i * bytes_per:(i + 1) * bytes_per]))

            rtf = (time.perf_counter() - gen_start) / (self.slice_len / self.tgt_fps)
            logger.info(f"[latency] chunk RTF {rtf:.2f} (queue {len(self.playback_queue)})")

    async def _video_loop(self):
        while not (getattr(self.transport._client, "_room", None)
                   and self.transport._client._room.isconnected()):
            await asyncio.sleep(0.2)
        room = self.transport._client._room
        await room.local_participant.publish_track(
            self.video_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
        await room.local_participant.publish_track(
            self.audio_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

        idle_vf = rtc.VideoFrame(512, 512, rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes())
        while True:
            start = time.time()
            if self.playback_queue and self.playback_queue[0][0] == self.epoch:
                _, rgba, audio_bytes = self.playback_queue.popleft()
                self.video_source.capture_frame(
                    rtc.VideoFrame(512, 512, rtc.VideoBufferType.RGBA, rgba.tobytes()))
                await self.audio_source.capture_frame(rtc.AudioFrame(
                    data=bytes(audio_bytes), sample_rate=self.sample_rate,
                    num_channels=1, samples_per_channel=len(audio_bytes) // 2))
            else:
                # silent + idle when not speaking (or right after a barge-in)
                if self.playback_queue:
                    self.playback_queue.clear()
                self.video_source.capture_frame(idle_vf)
            await asyncio.sleep(max(0, (1.0 / self.tgt_fps) - (time.time() - start)))


def build_transport():
    url = os.environ["LIVEKIT_URL"]
    key = os.environ["LIVEKIT_API_KEY"]
    secret = os.environ["LIVEKIT_API_SECRET"]
    room = os.environ.get("LIVEKIT_ROOM", "soulx-flashhead-room")
    token = (api.AccessToken(key, secret)
             .with_identity("soulx-bot").with_name("SoulX Avatar")
             .with_grants(api.VideoGrants(room_join=True, room=room,
                                          can_publish=True, can_subscribe=True,
                                          can_publish_data=True)).to_jwt())
    return LiveKitTransport(
        url=url, token=token, room_name=room,
        params=LiveKitParams(
            audio_in_enabled=True, audio_out_enabled=False, video_out_enabled=False,
            audio_in_sample_rate=16000, audio_out_sample_rate=16000,
            vad_enabled=True, vad_analyzer=SileroVADAnalyzer(),
        ),
    )


async def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logger.info("Loading SoulX model + warming CUDA graphs...")
    model_pipeline = get_pipeline(
        world_size=1,
        ckpt_dir=os.environ.get("SOULX_CKPT_DIR", "models/SoulX-FlashHead-1_3B"),
        wav2vec_dir=os.environ.get("SOULX_WAV2VEC_DIR", "models/wav2vec2-base-960h"),
        model_type=os.environ.get("SOULX_MODEL_TYPE", "pro"),
    )
    get_base_data(model_pipeline,
                  cond_image_path_or_dir=os.environ.get("SOULX_COND_IMAGE", "examples/omani_character.png"),
                  base_seed=42, use_face_crop=False)
    p = get_infer_params()
    dummy = np.zeros(p["sample_rate"] * p["cached_audio_duration"], dtype=np.float32)
    end = p["cached_audio_duration"] * p["tgt_fps"]
    run_pipeline(model_pipeline, get_audio_embedding(model_pipeline, dummy, end - p["frame_num"], end))
    torch.cuda.synchronize()
    logger.info("SoulX warmed.")

    transport = build_transport()

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"],
                           model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
        sample_rate=16000,
    )

    context = OpenAILLMContext(messages=[{
        "role": "system",
        "content": ("You are a friendly real-time video avatar. Keep replies to one or "
                    "two short spoken sentences — you are being interrupted often and "
                    "must stay snappy. Never use markdown or emoji."),
    }])
    context_agg = llm.create_context_aggregator(context)
    avatar = SoulXAvatarProcessor(transport, model_pipeline)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_agg.user(),
        llm,
        tts,
        avatar,
        context_agg.assistant(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams(
        allow_interruptions=True,          # <-- barge-in enabled
        enable_metrics=True,
    ))
    logger.info("Conversational SoulX bot running. Join the room to talk (interrupt any time).")
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
