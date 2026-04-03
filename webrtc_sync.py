import asyncio
import os
import numpy as np
import cv2
import time
from loguru import logger
import collections

from pipecat.frames.frames import AudioRawFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

import torch
from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

class WebRTCSyncPusher(FrameProcessor):
    def __init__(self, transport, model_pipeline, **kwargs):
        super().__init__(**kwargs)
        self.transport = transport
        self.model_pipeline = model_pipeline
        self.width = 512
        self.height = 512
        
        # Audio config
        infer_params = get_infer_params()
        self.sample_rate = infer_params['sample_rate']
        self.tgt_fps = infer_params['tgt_fps']
        self.cached_audio_duration = infer_params['cached_audio_duration']
        self.frame_num = infer_params['frame_num']
        self.motion_frames_num = infer_params['motion_frames_num']
        self.slice_len = self.frame_num - self.motion_frames_num
        
        self.cached_audio_length_sum = self.sample_rate * self.cached_audio_duration
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.audio_dq = collections.deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
        
        self.audio_slice_samples = self.slice_len * self.sample_rate // self.tgt_fps
        
        self.audio_buffer = bytearray()
        self.audio_float_buffer = []
        
        self.playback_queue = collections.deque()
        self.generation_queue = asyncio.Queue()
        
        idle_img_path = os.environ.get("SOULX_COND_IMAGE", "examples/omani_character.png")
        idle_img = cv2.imread(idle_img_path)
        if idle_img is None:
            idle_img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            idle_img = cv2.resize(idle_img, (512, 512))
        self.idle_rgba = cv2.cvtColor(idle_img, cv2.COLOR_BGR2RGBA)
        
        self.is_publishing = False

    def _get_daily_call_client(self):
        """Return the underlying daily.CallClient from the pipecat DailyTransport."""
        return self.transport._client._call_client

    async def _generation_loop(self):
        logger.info("Starting background GPU generation loop...")
        while True:
            chunk_floats, chunk_bytes = await self.generation_queue.get()
            
            self.audio_dq.extend(chunk_floats.tolist())
            audio_array = np.array(self.audio_dq)
            
            try:
                def run_infer():
                    torch.cuda.synchronize()
                    audio_embedding = get_audio_embedding(self.model_pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
                    
                    # DYNAMIC ANIMATION MULTIPLIER
                    # Increase this to > 1.0 for more dramatic mouth/head movements
                    animation_multiplier = 1.5 
                    audio_embedding = audio_embedding * animation_multiplier
                    
                    video = run_pipeline(self.model_pipeline, audio_embedding)
                    torch.cuda.synchronize()
                    return video.cpu().numpy()
                
                video_np = await asyncio.to_thread(run_infer)
                
                num_frames = video_np.shape[0]
                bytes_per_video_frame = len(chunk_bytes) // num_frames
                
                for i in range(num_frames):
                    v_frame = video_np[i]
                    if v_frame.shape[0] == 3:
                        v_frame = np.transpose(v_frame, (1, 2, 0))
                    if v_frame.dtype != np.uint8:
                        v_frame = v_frame.astype(np.uint8)
                    rgba = cv2.cvtColor(v_frame, cv2.COLOR_RGB2RGBA)
                    
                    start = i * bytes_per_video_frame
                    end = start + bytes_per_video_frame
                    audio_slice = chunk_bytes[start:end]
                    
                    self.playback_queue.append((rgba, audio_slice))
                    
            except Exception as e:
                logger.error(f"Inference error: {e}")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if not self.is_publishing:
            logger.info('Forcing publish on first frame!')
            self.is_publishing = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())
            
        if isinstance(frame, StartFrame) and not self.is_publishing:
            self.is_publishing = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())
            
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, AudioRawFrame):
            self.audio_buffer.extend(frame.audio)
            audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_float_buffer.extend(audio_data.tolist())
            
            while len(self.audio_float_buffer) >= self.audio_slice_samples:
                chunk_floats = np.array(self.audio_float_buffer[:self.audio_slice_samples])
                self.audio_float_buffer = self.audio_float_buffer[self.audio_slice_samples:]
                
                chunk_bytes_len = self.audio_slice_samples * 2
                chunk_bytes = self.audio_buffer[:chunk_bytes_len]
                self.audio_buffer = self.audio_buffer[chunk_bytes_len:]
                
                await self.generation_queue.put((chunk_floats, chunk_bytes))

        else:
            await self.push_frame(frame, direction)

    async def _video_loop(self):
        from daily import VideoFrame as DailyVideoFrame, AudioData as DailyAudioData

        logger.info("Waiting for Daily.co room to connect before publishing media...")
        while True:
            try:
                call_client = self._get_daily_call_client()
                if call_client is not None and str(call_client.state()) == "joined":
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)

        call_client = self._get_daily_call_client()
        logger.info("Starting perfectly synced playback loop...")
        
        while True:
            start_time = time.time()
            
            try:
                if len(self.playback_queue) > 999:
                    logger.warning(f"Playback queue too large ({len(self.playback_queue)}). Catching up to real-time...")
                    while len(self.playback_queue) > 25:
                        self.playback_queue.popleft()
                        
                if len(self.playback_queue) > 0:
                    rgba, audio_bytes = self.playback_queue.popleft()
                    
                    vf = DailyVideoFrame(
                        buffer=rgba.tobytes(),
                        width=int(rgba.shape[1]),
                        height=int(rgba.shape[0]),
                        color_format="RGBA",
                    )
                    await asyncio.to_thread(call_client.send_video_frame, vf)
                    
                    af = DailyAudioData(
                        audio_data=bytes(audio_bytes),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                    await asyncio.to_thread(call_client.send_audio, af)
                else:
                    vf = DailyVideoFrame(
                        buffer=self.idle_rgba.tobytes(),
                        width=int(self.idle_rgba.shape[1]),
                        height=int(self.idle_rgba.shape[0]),
                        color_format="RGBA",
                    )
                    await asyncio.to_thread(call_client.send_video_frame, vf)
                    await asyncio.sleep(0.04)
                    continue
                    
            except Exception as e:
                logger.error(f"Error playing back frame: {e}")
            
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0/self.tgt_fps) - elapsed)
            await asyncio.sleep(sleep_time)

async def main():
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from pipecat.vad.silero import SileroVADAnalyzer
        from pipecat.vad.vad_analyzer import VADParams

        daily_room_url = os.environ.get("DAILY_ROOM_URL", "")
        daily_token = os.environ.get("DAILY_TOKEN", "")

        if not daily_room_url:
            logger.error("DAILY_ROOM_URL must be set.")
            return

        transport = DailyTransport(
            room_url=daily_room_url,
            token=daily_token or None,
            bot_name="SoulX Avatar",
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=512,
                camera_out_height=512,
                camera_out_framerate=25,
                camera_out_color_format="RGBA",
                vad_enabled=False,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
            )
        )
        
        logger.info("Loading heavy SoulX Model into VRAM... This may take a minute.")
        model_pipeline = get_pipeline(
            world_size=1, 
            ckpt_dir="models/SoulX-FlashHead-1_3B", 
            wav2vec_dir="models/wav2vec2-base-960h", 
            model_type="lite"
        )
        
        get_base_data(model_pipeline, cond_image_path_or_dir="examples/omani_character.png", base_seed=42, use_face_crop=False)
        
        logger.info("Pre-warming the GPU to build CUDA graphs. This will take ~30-40 seconds...")
        infer_params = get_infer_params()
        sample_rate = infer_params['sample_rate']
        tgt_fps = infer_params['tgt_fps']
        cached_audio_duration = infer_params['cached_audio_duration']
        frame_num = infer_params['frame_num']
        
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num
        
        dummy_audio = np.zeros(cached_audio_length_sum, dtype=np.float32)
        torch.cuda.synchronize()
        dummy_embedding = get_audio_embedding(model_pipeline, dummy_audio, audio_start_idx, audio_end_idx)
        run_pipeline(model_pipeline, dummy_embedding)
        torch.cuda.synchronize()
        logger.info("SoulX Model fully loaded and GPU is pre-warmed.")

        pusher = WebRTCSyncPusher(transport, model_pipeline)

        pipeline = Pipeline([
            transport.input(),
            pusher,
            transport.output(),
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False), idle_timeout_secs=None)
        
        runner = PipelineRunner()
        
        logger.info(f'Starting CONTINUOUS WebRTC streaming bot in Daily.co room...')
        await runner.run(task)

    except Exception as e:
        logger.error(f'Error: {e}')

if __name__ == '__main__':
    asyncio.run(main())