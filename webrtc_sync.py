import asyncio
import os
import time
import numpy as np
import cv2
import soxr
from loguru import logger
import collections

from pipecat.frames.frames import (
    OutputAudioRawFrame,
    OutputImageRawFrame,
    BotInterruptionFrame,
    LLMFullResponseEndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

import torch
from flash_head.inference import (
    get_pipeline, get_base_data, get_infer_params,
    get_audio_embedding, run_pipeline,
)

# ── Avatar mode ──────────────────────────────────────────────────────────────
LISTENING = "listening"
SPEAKING  = "speaking"

IDLE_ANIMATION_MULT     = 1.5
SPEAKING_ANIMATION_MULT = 1.5

# 24 kHz samples per video frame: 24000 / 25 fps = 960 samples = 40 ms
SAMPLES_24K_PER_FRAME = 960

# Turn-end timeouts (seconds):
#   CONTEXT_DONE  — LLMFullResponseEndFrame received: short debounce.
#   FALLBACK      — no context frame yet (inter-sentence gap). Must be longer
#                   than OpenAI TTS API latency per sentence (~1.5–2 s).
_CONTEXT_DONE_DEBOUNCE = 0.15
_FALLBACK_TIMEOUT      = float(os.environ.get("TURN_END_TIMEOUT", "2.5"))

# Idle feeder: keep this many slices queued ahead in _generation_queue
_IDLE_QUEUE_TARGET = 2


class WebRTCSyncPusher(FrameProcessor):
    """Pipecat FrameProcessor that drives the SoulX talking-head avatar.

    Speaking mode
    -------------
    - Intercepts OutputAudioRawFrame (24 kHz) from TTS
    - Accumulates audio; dispatches 1.24-second slices to GPU inference
    - Turn-end detected via LLMFullResponseEndFrame + 150 ms debounce, with
      2.5 s fallback for inter-sentence gaps
    - Resamples 24→16 kHz internally for SoulX inference
    - Pushes OutputImageRawFrame(sync_with_audio=True) + matching audio

    Listening (idle) mode
    ----------------------
    - _idle_feeder_loop continuously enqueues near-silence slices (16 kHz)
      into the shared _generation_queue
    - _generation_loop infers from them exactly as it does for speaking,
      using the same pipeline — so latent_motion_frames carries over
      automatically, giving pose-continuous idle↔speaking transitions
    - Idle frames are tagged (aud_frame=None) and pushed to screen only;
      they keep displaying until the first speaking frame is ready so
      there is no freeze gap during the speaking ramp-up
    """

    def __init__(self, transport, model_pipeline, **kwargs):
        super().__init__(**kwargs)
        self.transport      = transport
        self.model_pipeline = model_pipeline

        infer_params = get_infer_params()
        self.sample_rate           = infer_params['sample_rate']            # 16000
        self.tgt_fps               = infer_params['tgt_fps']                # 25
        self.cached_audio_duration = infer_params['cached_audio_duration']  # 8
        self.frame_num             = infer_params['frame_num']              # 33
        self.motion_frames_num     = infer_params['motion_frames_num']      # 2

        self.slice_len = self.frame_num - self.motion_frames_num  # 31

        self.cached_audio_length_sum = self.sample_rate * self.cached_audio_duration  # 128000
        self.audio_end_idx   = self.cached_audio_duration * self.tgt_fps              # 200
        self.audio_start_idx = self.audio_end_idx - self.frame_num                   # 167

        # Separate rolling audio context deques for speaking and idle.
        # Keeping them separate preserves speaking lip-sync quality while
        # still sharing the pipeline's latent_motion_frames for visual continuity.
        self.audio_dq = collections.deque(
            [0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum
        )
        self.idle_audio_dq = collections.deque(
            [0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum
        )

        # Slice sizes
        self.audio_slice_samples_16k = self.slice_len * self.sample_rate // self.tgt_fps
        self.audio_slice_samples_24k = self.audio_slice_samples_16k * 24000 // self.sample_rate
        self._slice_bytes_24k = self.audio_slice_samples_24k * 2  # int16 = 2 bytes/sample

        self._make_soxr()

        self._audio_buf_24k    = bytearray()
        self._generation_queue = asyncio.Queue()
        self._output_queue     = asyncio.Queue()

        # ── Turn-end state ────────────────────────────────────────────────────
        self._last_tts_time    = 0.0
        self._gen_in_flight    = 0   # speaking inferences dequeued but not done
        self._tts_context_done = False
        self._pending_flush    = False

        # ── Idle feeder ───────────────────────────────────────────────────────
        self._idle_feeder_task = None
        self._idle_rng         = np.random.default_rng()  # unseeded = variety

        self.mode        = LISTENING
        self._publishing = False
        self._generation_serial = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_soxr(self):
        self._soxr_stream = soxr.ResampleStream(
            in_rate=24000, out_rate=16000,
            num_channels=1, quality="VHQ", dtype="int16",
        )

    def _flush_buffers(self):
        """Discard pending audio and queued GPU work. Called on interruption."""
        self._audio_buf_24k    = bytearray()
        self._last_tts_time    = 0.0
        self._tts_context_done = False
        self._pending_flush    = False
        self._gen_in_flight    = 0
        self._make_soxr()
        _drain_async_queue(self._generation_queue)
        _drain_async_queue(self._output_queue)

    def _run_infer(self, audio_array_16k, mult):
        """Blocking GPU inference — called via asyncio.to_thread."""
        torch.cuda.synchronize()
        emb = get_audio_embedding(
            self.model_pipeline, audio_array_16k,
            self.audio_start_idx, self.audio_end_idx,
        )
        emb = emb * mult
        video = run_pipeline(self.model_pipeline, emb)
        torch.cuda.synchronize()
        return video.cpu().numpy()

    def _video_np_to_rgba_list(self, video_np):
        frames = []
        for i in range(video_np.shape[0]):
            vf = video_np[i]
            if vf.shape[0] == 3:
                vf = np.transpose(vf, (1, 2, 0))
            if vf.dtype != np.uint8:
                vf = vf.astype(np.uint8)
            frames.append(cv2.cvtColor(vf, cv2.COLOR_RGB2RGBA).tobytes())
        return frames

    # ── Idle feeder ───────────────────────────────────────────────────────────

    async def _idle_feeder_loop(self):
        """Continuously enqueue near-silence slices for idle inference.

        Uses the shared _generation_queue so idle and speaking inferences are
        serialized in _generation_loop.  This means pipeline.latent_motion_frames
        is updated by every idle inference, giving automatic pose-continuity
        when speaking starts — no crossfade or motion seeding needed.

        Near-silence (amplitude ~0.005, fresh random seed each slice) lets the
        model produce its natural trained behavior: organic micro-expressions,
        irregular blinks, subtle head movement.  No engineered sine waves that
        produce mechanical repetitive motion.
        """
        try:
            while True:
                if self._generation_queue.qsize() < _IDLE_QUEUE_TARGET:
                    silence = (
                        self._idle_rng.standard_normal(self.audio_slice_samples_16k)
                        .astype(np.float32) * 0.005
                    )
                    await self._generation_queue.put(('idle', None, silence))
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def _start_idle_feeder(self):
        if self._idle_feeder_task is not None and not self._idle_feeder_task.done():
            return
        self._idle_feeder_task = asyncio.create_task(self._idle_feeder_loop())
        logger.info("[IDLE] Feeder started — continuous idle generation")

    def _stop_idle_feeder(self):
        if self._idle_feeder_task is not None and not self._idle_feeder_task.done():
            self._idle_feeder_task.cancel()
            self._idle_feeder_task = None
            logger.info("[IDLE] Feeder stopped")

    # ── GPU generation loop ───────────────────────────────────────────────────

    async def _generation_loop(self):
        """Single serialized GPU inference loop for both idle and speaking.

        Serialization is the key: because all inferences (idle and speaking)
        go through this one loop, pipeline.latent_motion_frames is always
        consistent — the last idle frame feeds the first speaking frame and
        vice versa, giving visually smooth transitions for free.
        """
        logger.info("[GEN] Generation loop started")
        while True:
            mode, audio_24k_bytes, audio_16k_floats = await self._generation_queue.get()
            serial_before = self._generation_serial

            if mode == 'speaking':
                self._gen_in_flight += 1
                self.audio_dq.extend(audio_16k_floats.tolist())
                audio_array = np.array(self.audio_dq)
                mult = SPEAKING_ANIMATION_MULT
            else:
                self.idle_audio_dq.extend(audio_16k_floats.tolist())
                audio_array = np.array(self.idle_audio_dq)
                mult = IDLE_ANIMATION_MULT

            try:
                video_np = await asyncio.to_thread(
                    self._run_infer, audio_array, mult
                )
            except Exception as e:
                logger.error(f"[GEN] Inference error ({mode}): {e}")
                if mode == 'speaking':
                    self._gen_in_flight -= 1
                continue

            if self._generation_serial != serial_before:
                logger.debug(f"[GEN] Stale {mode} frames discarded (interrupted)")
                if mode == 'speaking':
                    self._gen_in_flight -= 1
                continue

            rgba_frames = self._video_np_to_rgba_list(video_np)

            for i, rgba_bytes in enumerate(rgba_frames):
                img_frame = OutputImageRawFrame(
                    image=rgba_bytes, size=(512, 512), format="RGBA"
                )
                if mode == 'speaking':
                    start = i * SAMPLES_24K_PER_FRAME * 2
                    end   = start + SAMPLES_24K_PER_FRAME * 2
                    audio_slice = audio_24k_bytes[start:end]
                    if len(audio_slice) < SAMPLES_24K_PER_FRAME * 2:
                        audio_slice = audio_slice + bytes(
                            SAMPLES_24K_PER_FRAME * 2 - len(audio_slice)
                        )
                    img_frame.sync_with_audio = True
                    aud_frame = OutputAudioRawFrame(
                        audio=audio_slice, sample_rate=24000, num_channels=1
                    )
                    await self._output_queue.put((img_frame, aud_frame))
                else:
                    # Idle: no audio output, aud_frame=None signals idle frame
                    await self._output_queue.put((img_frame, None))

            if mode == 'speaking':
                self._gen_in_flight -= 1

    # ── Output loop ───────────────────────────────────────────────────────────

    async def _output_loop(self):
        """Push frames downstream and manage idle↔speaking transitions.

        Frame routing:
          (img, aud_frame)  → speaking frame: push both, switch to SPEAKING mode
          (img, None)       → idle frame:
                              - if mode != SPEAKING: push (keeps avatar animated
                                during GPU ramp-up gap, no freeze)
                              - if mode == SPEAKING: discard (stale)

        Stale frame handling:
          - Speaking frame arriving after turn ended (_last_tts_time==0.0): discard
          - Idle frame arriving after mode already SPEAKING: discard
        """
        _CHECK_INTERVAL = 0.05

        while True:
            # ── Pull next frame ───────────────────────────────────────────────
            try:
                img_frame, aud_frame = await asyncio.wait_for(
                    self._output_queue.get(), timeout=_CHECK_INTERVAL
                )
                is_speaking_frame = aud_frame is not None

                if is_speaking_frame and self._last_tts_time == 0.0:
                    # Stale speaking frame — turn already ended
                    logger.debug("[GEN] Discarding stale speaking frame (turn ended)")

                elif not is_speaking_frame and self.mode == SPEAKING:
                    # Stale idle frame — speaking already active
                    pass

                else:
                    # Valid frame — push downstream
                    if is_speaking_frame and self.mode != SPEAKING:
                        # First speaking frame: transition from idle to speaking
                        self.mode = SPEAKING
                        logger.info("[AVATAR MODE] Speaking — first frame flowing")

                    await self.push_frame(img_frame)
                    if aud_frame is not None:
                        await self.push_frame(aud_frame)

                if self._output_queue.qsize() % 15 == 0 and self._output_queue.qsize() > 0:
                    logger.debug(
                        f"[SYNC] out_q={self._output_queue.qsize()} "
                        f"gen_q={self._generation_queue.qsize()} "
                        f"inf={self._gen_in_flight} "
                        f"buf={len(self._audio_buf_24k)}B "
                        f"ctx_done={self._tts_context_done} "
                        f"tts_age={time.monotonic()-self._last_tts_time:.2f}s"
                    )

            except asyncio.TimeoutError:
                pass

            # ── Turn-end check ────────────────────────────────────────────────
            if (self._output_queue.empty()
                    and self._generation_queue.empty()
                    and self._gen_in_flight == 0
                    and self._last_tts_time > 0.0):

                elapsed = time.monotonic() - self._last_tts_time
                effective_timeout = (
                    _CONTEXT_DONE_DEBOUNCE if self._tts_context_done
                    else _FALLBACK_TIMEOUT
                )

                if elapsed < effective_timeout:
                    continue

                # ── Step 1: flush partial audio buffer (last-word rescue) ─────
                if len(self._audio_buf_24k) > 0 and not self._pending_flush:
                    remaining  = len(self._audio_buf_24k)
                    chunk_24k  = bytes(self._audio_buf_24k) + bytes(
                        self._slice_bytes_24k - remaining
                    )
                    self._audio_buf_24k = bytearray()
                    pcm_24k    = np.frombuffer(chunk_24k, dtype=np.int16)
                    pcm_16k    = self._soxr_stream.resample_chunk(pcm_24k).astype(np.int16)
                    floats_16k = pcm_16k.astype(np.float32) / 32768.0
                    logger.info(
                        f"[FLUSH] Last-word rescue: {remaining}B → gen_q "
                        f"(ctx_done={self._tts_context_done}, age={elapsed:.2f}s)"
                    )
                    await self._generation_queue.put(('speaking', chunk_24k, floats_16k))
                    self._pending_flush = True
                    continue

                # ── Step 2: transition to idle ────────────────────────────────
                self.mode = LISTENING
                logger.info(
                    f"[AVATAR MODE] Listening — turn ended "
                    f"(age={elapsed:.2f}s ctx_done={self._tts_context_done} "
                    f"flush={self._pending_flush})"
                )
                self._last_tts_time    = 0.0
                self._pending_flush    = False
                self._tts_context_done = False

                # Restart idle feeder — first idle inference picks up from
                # pipeline.latent_motion_frames set by last speaking inference
                self._start_idle_feeder()

    # ── Pipecat frame processor ───────────────────────────────────────────────

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Start background tasks on first frame
        if not self._publishing:
            self._publishing = True
            logger.info("[PUSHER] Starting generation and output loops")
            asyncio.create_task(self._generation_loop())
            asyncio.create_task(self._output_loop())
            self._start_idle_feeder()

        # ── State machine ────────────────────────────────────────────────────

        if isinstance(frame, BotInterruptionFrame):
            logger.info("[AVATAR] Interrupted — flushing, returning to LISTENING")
            self._generation_serial += 1
            self.mode = LISTENING
            self._stop_idle_feeder()
            self._flush_buffers()
            self._start_idle_feeder()

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._last_tts_time > 0.0:
                logger.info(
                    "[TURN] LLMFullResponseEndFrame — no more sentences; "
                    f"switching to {_CONTEXT_DONE_DEBOUNCE*1000:.0f}ms debounce"
                )
            self._tts_context_done = True

        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.info("[USER] started speaking")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("[USER] stopped speaking")

        # ── Audio interception ───────────────────────────────────────────────
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, OutputAudioRawFrame):
            now = time.monotonic()
            if self._last_tts_time == 0.0:
                logger.info("[AVATAR MODE] Speaking — TTS audio stream started")
                # Stop idle feeder: no new idle slices queued from here on.
                # Any idle slices already in _generation_queue will complete and
                # their output frames discarded (aud_frame=None while mode==SPEAKING).
                # Idle frames already displayed keep the avatar animated during
                # the GPU ramp-up gap — no freeze.
                self._stop_idle_feeder()
            self._last_tts_time = now
            self._pending_flush = False

            self._audio_buf_24k.extend(frame.audio)

            while len(self._audio_buf_24k) >= self._slice_bytes_24k:
                chunk_24k  = bytes(self._audio_buf_24k[:self._slice_bytes_24k])
                self._audio_buf_24k = self._audio_buf_24k[self._slice_bytes_24k:]
                pcm_24k    = np.frombuffer(chunk_24k, dtype=np.int16)
                pcm_16k    = self._soxr_stream.resample_chunk(pcm_24k).astype(np.int16)
                floats_16k = pcm_16k.astype(np.float32) / 32768.0
                logger.debug(
                    f"[DISPATCH] Full slice → gen_q "
                    f"(depth={self._generation_queue.qsize()+1})"
                )
                await self._generation_queue.put(('speaking', chunk_24k, floats_16k))
        else:
            await self.push_frame(frame, direction)


# ── Utility ───────────────────────────────────────────────────────────────────

def _drain_async_queue(q: asyncio.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break
