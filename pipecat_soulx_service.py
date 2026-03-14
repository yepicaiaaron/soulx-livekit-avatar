import asyncio
import collections
import numpy as np
from typing import Optional

from pipecat.frames.frames import (
    AudioRawFrame,
    OutputImageRawFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger

from flash_head.inference import get_audio_embedding, run_pipeline


class SoulXFlashHeadService(FrameProcessor):
    """
    A custom Pipecat service that receives AudioRawFrames (from a TTS engine),
    buffers them using a sliding window, streams them into the SoulX-FlashHead
    inference pipeline, and yields OutputImageRawFrames (video frames) frame-by-frame.
    """

    def __init__(self, pipeline, infer_params: dict, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
        self._infer_params = infer_params

        # Audio parameters
        self._sample_rate = infer_params['sample_rate']
        self._tgt_fps = infer_params['tgt_fps']
        self._frame_num = infer_params['frame_num']
        self._motion_frames_num = infer_params['motion_frames_num']

        # Number of new frames generated per inference call
        self._slice_len = self._frame_num - self._motion_frames_num

        # Number of audio samples needed per inference slice
        self._audio_slice_len = self._slice_len * self._sample_rate // self._tgt_fps

        # Sliding window audio buffer
        self._cached_audio_duration = infer_params['cached_audio_duration']
        self._cached_audio_length = self._sample_rate * self._cached_audio_duration
        self._audio_buffer = collections.deque(
            [0.0] * self._cached_audio_length,
            maxlen=self._cached_audio_length
        )

        # Indices for audio embedding window
        self._audio_end_idx = self._cached_audio_duration * self._tgt_fps
        self._audio_start_idx = self._audio_end_idx - self._frame_num

        # Accumulator for incoming audio samples between inference calls
        self._audio_accumulator = []
        self._accumulated_samples = 0

        logger.info("SoulXFlashHeadService initialized")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._handle_audio_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Buffer audio and trigger video generation when enough samples are ready."""
        # Convert 16-bit PCM bytes to float32 numpy array
        audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

        self._audio_accumulator.append(audio_data)
        self._accumulated_samples += len(audio_data)

        while self._accumulated_samples >= self._audio_slice_len:
            # Concatenate and extract one slice
            all_audio = np.concatenate(self._audio_accumulator)
            audio_slice = all_audio[:self._audio_slice_len]

            # Keep the remainder for the next iteration
            remaining = all_audio[self._audio_slice_len:]
            self._audio_accumulator = [remaining] if len(remaining) > 0 else []
            self._accumulated_samples = len(remaining)

            # Advance the sliding window buffer
            self._audio_buffer.extend(audio_slice.tolist())

            # Run inference and push resulting video frames
            try:
                video_frames = await self._generate_video_frames()
                for v_frame in video_frames:
                    h, w = v_frame.shape[:2]
                    frame_bytes = v_frame.tobytes()
                    image_frame = OutputImageRawFrame(
                        image=frame_bytes,
                        size=(w, h),
                        format="RGB"
                    )
                    await self.push_frame(image_frame, direction)
            except Exception as e:
                logger.error(f"Error during SoulX inference: {e}")

        # Pass the original audio frame downstream so the user hears the TTS
        await self.push_frame(frame, direction)

    async def _generate_video_frames(self) -> list:
        """Run SoulX inference in a thread pool and return a list of HxWxC uint8 arrays."""
        audio_array = np.array(self._audio_buffer)

        # Step 1: encode audio into embedding
        audio_embedding = await asyncio.to_thread(
            get_audio_embedding,
            self._pipeline,
            audio_array,
            self._audio_start_idx,
            self._audio_end_idx
        )

        # Step 2: run the generative pipeline
        video_tensor = await asyncio.to_thread(
            run_pipeline,
            self._pipeline,
            audio_embedding
        )

        # Step 3: convert tensor frames to numpy arrays
        frames = []
        for i in range(video_tensor.shape[0]):
            frame_np = video_tensor[i].cpu().numpy().astype(np.uint8)
            # Ensure HWC layout
            if frame_np.ndim == 3 and frame_np.shape[0] == 3:
                frame_np = np.transpose(frame_np, (1, 2, 0))
            frames.append(frame_np)

        return frames
