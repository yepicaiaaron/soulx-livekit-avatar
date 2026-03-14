#!/usr/bin/env python3
"""
SoulX-FlashHead Conversational Pipecat Bot with LiveKit Transport

This script implements a complete interactive talking head avatar experience using:
- LiveKitTransport for WebRTC audio/video streaming
- DeepgramSTTService for rapid speech-to-text
- OpenAILLMService for conversational intelligence
- ElevenLabsTTSService for high-quality text-to-speech
- SoulXFlashHeadService for real-time video generation
- Pipecat orchestration framework

The pipeline flow:
User Audio (LiveKit) -> STT -> LLM context -> LLM -> TTS -> SoulX -> LLM context -> User Video/Audio (LiveKit)
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
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextAggregator,
)

# Service imports
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService

# Add SoulX-FlashHead to path
sys.path.insert(0, str(Path(__file__).parent))

from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
)

from pipecat_soulx_service import SoulXFlashHeadService

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
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam


class SoulXConversationalBot:
    """
    Main bot class that orchestrates the Conversational SoulX pipeline.
    """

    def __init__(self, config: SoulXConfig):
        self.config = config
        self.flash_pipeline = None
        self.runner = None

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

    def create_pipeline(self):
        """Create the Pipecat pipeline. Returns (pipeline, task, transport)."""

        # 1. Transport — url/token/room_name go in the constructor, not in LiveKitParams
        params = LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_width=512,
            video_out_height=512,
            video_out_framerate=self.config.tgt_fps,
        )

        transport = LiveKitTransport(
            url=self.config.livekit_url,
            token=self.config.livekit_token,
            room_name=self.config.room_name,
            params=params,
        )

        # 2. STT
        stt = DeepgramSTTService(api_key=self.config.deepgram_api_key)

        # 3. LLM
        llm = OpenAILLMService(api_key=self.config.openai_api_key, model="gpt-4o")

        # 4. LLM context with system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, friendly conversational AI avatar. "
                    "Keep responses concise and natural for spoken conversation."
                ),
            }
        ]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # 5. TTS
        tts = ElevenLabsTTSService(
            api_key=self.config.elevenlabs_api_key,
            voice_id=self.config.elevenlabs_voice_id,
        )

        # 6. SoulX Video Service
        soulx = SoulXFlashHeadService(
            pipeline=self.flash_pipeline,
            infer_params=self.infer_params,
        )

        # 7. Build Pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            soulx,
            context_aggregator.assistant(),
            transport.output(),
        ])

        task = PipelineTask(pipeline)

        # 8. Event handler: greet the first participant and start the conversation
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant.identity)
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        return pipeline, task, transport

    async def run(self):
        """Run the bot"""
        await self.initialize_model()

        pipeline, task, transport = self.create_pipeline()
        self.runner = PipelineRunner()

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

    # Ensure LiveKit token is present or generate one from API key/secret
    if not config.livekit_token:
        from livekit import api
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        if api_key and api_secret:
            token = (
                api.AccessToken(api_key, api_secret)
                .with_identity("soulx-conversational-bot")
                .with_name("SoulX Avatar")
                .with_grants(api.VideoGrants(room_join=True, room=config.room_name))
                .to_jwt()
            )
            config.livekit_token = token

    bot = SoulXConversationalBot(config)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
