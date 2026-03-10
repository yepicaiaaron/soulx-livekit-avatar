import asyncio
import numpy as np
from typing import Optional

from pipecat.frames.frames import (
    AudioRawFrame,
    RawImageFrame,
    Frame,
    SystemFrame
)
from pipecat.services.base_service import BaseService
from loguru import logger

class SoulXFlashHeadService(BaseService):
    """
    A custom Pipecat service that receives AudioRawFrames (from a TTS engine like Cartesia),
    buffers them, streams them into the SoulX-FlashHead inference pipeline, 
    and yields RawImageFrames (video frames) frame-by-frame.
    """
    
    def __init__(self, pipeline, sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
        self._sample_rate = sample_rate
        self._audio_buffer = bytearray()
        
        # SoulX typically needs a specific chunk size of audio to generate a video frame slice
        # (e.g., 20ms of audio -> 1 frame at 25 fps, though exact sizes depend on the model slice_len)
        self._bytes_per_chunk = int(sample_rate * 2) # Assuming 16-bit PCM (2 bytes per sample) 
        
        # We need to know how many bytes to accumulate before doing a forward pass
        # This is a placeholder for the actual stride length in bytes
        self._chunk_size = 4000 
        logger.info("Initialized SoulX-FlashHead Pipecat Service")

    async def process_frame(self, frame: Frame):
        if isinstance(frame, AudioRawFrame):
            # 1. Buffer the incoming TTS audio
            self._audio_buffer.extend(frame.audio)
            
            # 2. Check if we have enough audio to do an inference step
            while len(self._audio_buffer) >= self._chunk_size:
                # Extract the chunk
                chunk_bytes = self._audio_buffer[:self._chunk_size]
                self._audio_buffer = self._audio_buffer[self._chunk_size:]
                
                # Convert 16-bit PCM bytes to float32 numpy array for SoulX
                audio_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # 3. Stream into SoulX-FlashHead in-memory pipeline
                # This runs the neural network forward pass on the audio chunk
                try:
                    # run_pipeline yields video tensors (e.g., shape [H, W, C])
                    video_tensors = await asyncio.to_thread(self._pipeline.process_audio_chunk, audio_array)
                    
                    # 4. Yield generated frames down the Pipecat pipeline instantly
                    for v_frame in video_tensors:
                        # Convert to bytes
                        frame_bytes = v_frame.tobytes()
                        # Assuming output is 512x512 RGB
                        image_frame = RawImageFrame(
                            pixels=frame_bytes,
                            size=(512, 512),
                            format="RGB"
                        )
                        await self.push_frame(image_frame)
                        
                except Exception as e:
                    logger.error(f"Error during SoulX inference step: {e}")
                    
            # We also pass the audio frame along so the user hears the TTS
            await self.push_frame(frame)
            
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame)
        else:
            await self.push_frame(frame)
