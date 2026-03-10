#!/usr/bin/env python3
"""
SoulX-FlashHead Conversational Pipecat Bot with LiveKit Transport

This script implements a complete interactive talking head avatar experience using:
- LiveKitTransport for WebRTC audio/video streaming
- DeepgramSTTService for rapid speech-to-text
- OpenAILLMService for conversational intelligence
- CartesiaTTSService for ultra-fast text-to-speech
- ElevenLabsTTSService for high-quality text-to-speech
- SoulXFlashHeadService for real-time video generation
- Pipecat orchestration framework

The pipeline flow:
User Audio (LiveKit) -> STT -> LLM -> TTS -> SoulX -> User Video/Audio (LiveKit)
"""

import asyncio
import os
import sys
import signal
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# Pipecat imports
from pipecat.frames.frames import (
    AudioRawFrame,
    OutputImageRawFrame,
    OutputAudioRawFrame,
    Frame,
    SystemFrame,
    EndFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams

# Service imports
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService

import livekit.rtc as rtc

# Add SoulX-FlashHead to path
sys.path.insert(0, str(Path(__file__).parent))

from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
)

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")


@dataclass
class SoulXConfig:
    """Configuration for SoulX-FlashHead Service"""
    ckpt_dir: str = os.getenv("SOULX_CKPT_DIR", "./models/SoulX-FlashHead-1_3B")
    wav2vec_dir: str = os.getenv("SOULX_WAV2VEC_DIR", "./models/wav2vec2-base-960h")
    model_type: str = os.getenv("SOULX_MODEL_TYPE", "lite")  # "lite" or "pro"
    cond_image_path: str = os.getenv("SOULX_COND_IMAGE", "./examples/girl.png")
    base_seed: int = int(os.getenv("SOULX_SEED", "42"))
    use_face_crop: bool = os.getenv("SOULX_USE_FACE_CROP", "false").lower() == "true"
    sample_rate: int = 16000
    tgt_fps: int = 25
    
    # LiveKit configuration
    livekit_url: str = os.getenv("LIVEKIT_URL", "wss://localhost:7880")
    livekit_token: str = os.getenv("LIVEKIT_TOKEN", "")
    room_name: str = os.getenv("LIVEKIT_ROOM", "soulx-flashhead-room")
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB") # Adam


class SoulXFlashHeadService(FrameProcessor):
    """
    SoulX-FlashHead Pipecat Service for real-time talking head generation.
    
    Receives AudioRawFrames from TTS, buffers them, streams into
    SoulX-FlashHead inference pipeline, and yields OutputImageRawFrames.
    """
    
    def __init__(self, pipeline, infer_params: dict, config: SoulXConfig, video_source: rtc.VideoSource, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
        self._infer_params = infer_params
        self._config = config
        self._video_source = video_source
        
        # Audio buffering parameters
        self._sample_rate = infer_params['sample_rate']
        self._tgt_fps = infer_params['tgt_fps']
        self._frame_num = infer_params['frame_num']
        self._motion_frames_num = infer_params['motion_frames_num']
        self._slice_len = self._frame_num - self._motion_frames_num
        
        # Audio chunk size: slice_len frames worth of audio
        self._audio_slice_len = self._slice_len * self._sample_rate // self._tgt_fps
        
        # Streaming audio buffer (sliding window)
        from collections import deque
        self._cached_audio_duration = infer_params['cached_audio_duration']
        self._cached_audio_length = self._sample_rate * self._cached_audio_duration
        self._audio_buffer = deque([0.0] * self._cached_audio_length, maxlen=self._cached_audio_length)
        
        # Indices for audio embedding window
        self._audio_end_idx = self._cached_audio_duration * self._tgt_fps
        self._audio_start_idx = self._audio_end_idx - self._frame_num
        
        # Accumulator for audio chunks
        self._audio_accumulator = []
        self._accumulated_samples = 0
        
        logger.info(f"SoulXFlashHeadService initialized")
        
    async def process_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Process incoming frames"""
        if isinstance(frame, AudioRawFrame):
            # Only process audio going downstream (from TTS to user)
            if direction == FrameDirection.DOWNSTREAM:
                await self._handle_audio_frame(frame, direction)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (StartFrame, EndFrame)):
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
    
    async def _handle_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Buffer audio and trigger video generation"""
        # Convert bytes to float32
        audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        self._audio_accumulator.append(audio_data)
        self._accumulated_samples += len(audio_data)
        
        while self._accumulated_samples >= self._audio_slice_len:
            # Extract slice
            all_audio = np.concatenate(self._audio_accumulator)
            audio_slice = all_audio[:self._audio_slice_len]
            
            # Update accumulator
            remaining = all_audio[self._audio_slice_len:]
            self._audio_accumulator = [remaining] if len(remaining) > 0 else []
            self._accumulated_samples = len(remaining)
            
            # Update streaming buffer
            self._audio_buffer.extend(audio_slice.tolist())
            
            # Generate video
            try:
                # Run inference in a thread pool to avoid blocking
                video_frames = await self._generate_video_frame()
                
                # Push frames to LiveKit video source
                for v_frame in video_frames:
                    await self._push_video_frame(v_frame)
                    
            except Exception as e:
                logger.error(f"Error during SoulX inference: {e}")
    
    async def _generate_video_frame(self) -> list:
        """Run SoulX inference"""
        audio_array = np.array(self._audio_buffer)
        
        # Audio encoding
        audio_embedding = await asyncio.to_thread(
            get_audio_embedding,
            self._pipeline,
            audio_array,
            self._audio_start_idx,
            self._audio_end_idx
        )
        
        # Video generation
        video_tensor = await asyncio.to_thread(
            run_pipeline,
            self._pipeline,
            audio_embedding.to(self._pipeline.device)
        )
        
        # Extract frames
        frames = []
        for i in range(video_tensor.shape[0]):
            frame = video_tensor[i].cpu().numpy().astype(np.uint8)
            frames.append(frame)
        
        return frames
    
    async def _push_video_frame(self, frame_array: np.ndarray):
        """Push frame to LiveKit VideoSource"""
        import time
        import cv2
        
        # Ensure RGB [H, W, 3]
        if frame_array.shape[0] == 3:
            frame_array = np.transpose(frame_array, (1, 2, 0))
            
        h, w = frame_array.shape[:2]
        
        # Convert to I420 for LiveKit
        i420_frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2YUV_I420)
        frame_bytes = i420_frame.tobytes()
        
        # Create LiveKit frame (I420 = 5)
        lk_frame = rtc.VideoFrame(w, h, 5, frame_bytes)
        
        # Capture in video source
        ts_us = int(time.time() * 1000000)
        self._video_source.capture_frame(lk_frame, timestamp_us=ts_us)


class SoulXConversationalBot:
    """
    Main bot class that orchestrates the Conversational SoulX pipeline.
    """
    
    def __init__(self, config: SoulXConfig):
        self.config = config
        self.flash_pipeline = None
        self.runner = None
        self.video_source = rtc.VideoSource(512, 512)
        
    async def initialize_model(self):
        """Initialize SoulX-FlashHead"""
        logger.info("Initializing SoulX-FlashHead model...")
        
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.flash_pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=self.config.ckpt_dir,
            model_type=self.config.model_type,
            wav2vec_dir=self.config.wav2vec_dir
        )
        
        get_base_data(
            self.flash_pipeline,
            cond_image_path_or_dir=self.config.cond_image_path,
            base_seed=self.config.base_seed,
            use_face_crop=self.config.use_face_crop
        )
        
        self.infer_params = get_infer_params()
        logger.info("Model initialized.")

    def create_pipeline(self) -> Pipeline:
        """Create the Pipecat pipeline"""
        
        # 1. Transport
        params = LiveKitParams(
            url=self.config.livekit_url,
            token=self.config.livekit_token,
            room_name=self.config.room_name,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_width=512,
            video_out_height=512,
            video_out_framerate=self.config.tgt_fps
        )
        
        transport = LiveKitTransport(
            url=self.config.livekit_url,
            room_name=self.config.room_name,
            token=self.config.livekit_token,
            params=params
        )
        
        # Custom track publishing for video
        @transport.event_handler("on_connected")
        async def on_connected(transport, *args, **kwargs):
            track = rtc.LocalVideoTrack.create_video_track("bot-video", self.video_source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_CAMERA
            options.video_codec = rtc.VideoCodec.VP8
            options.simulcast = False
            await transport._client.room.local_participant.publish_track(track, options)
            logger.info("Published video track to LiveKit")

        # 2. STT
        stt = DeepgramSTTService(api_key=self.config.deepgram_api_key)
        
        # 3. LLM
        llm = OpenAILLMService(
            api_key=self.config.openai_api_key,
            model="gpt-4o"
        )
        
        # 4. TTS
        tts = ElevenLabsTTSService(
            api_key=self.config.elevenlabs_api_key,
            voice_id=self.config.elevenlabs_voice_id
        )
        
        # 5. SoulX Video Service
        soulx = SoulXFlashHeadService(
            pipeline=self.flash_pipeline,
            infer_params=self.infer_params,
            config=self.config,
            video_source=self.video_source
        )
        
        # 6. Build Pipeline
        # Input -> STT -> LLM -> TTS -> SoulX -> Output
        pipeline = Pipeline([
            transport.input(),
            stt,
            llm,
            tts,
            soulx,
            transport.output()
        ])
        
        return pipeline

    async def run(self):
        """Run the bot"""
        await self.initialize_model()
        
        pipeline = self.create_pipeline()
        self.runner = PipelineRunner()
        task = PipelineTask(pipeline)
        
        # Signal handling
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            
        logger.info("Bot is running. Join the LiveKit room to talk!")
        await self.runner.run(task)

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        if self.runner:
            await self.runner.cancel()


async def main():
    config = SoulXConfig()
    
    # Ensure LiveKit token is present or generated
    if not config.livekit_token:
        from livekit import api
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        if api_key and api_secret:
            token = api.AccessToken(api_key, api_secret) \
                .with_identity("soulx-conversational-bot") \
                .with_name("SoulX Avatar") \
                .with_grants(api.VideoGrants(room_join=True, room=config.room_name)) \
                .to_jwt()
            config.livekit_token = token
            
    bot = SoulXConversationalBot(config)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
