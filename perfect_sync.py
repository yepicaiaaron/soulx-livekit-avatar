import asyncio
import os
import random
import numpy as np
import cv2
import time
from loguru import logger
import collections

from pipecat.frames.frames import TextFrame, StartFrame, EndFrame, AudioRawFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from livekit import api, rtc
import torch
from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

PARAGRAPH = """
The quick brown fox jumps over the lazy dog. 
I am now testing the continuous streaming capabilities of the Soul X avatar. 
By providing a very long paragraph of text, the speech synthesis engine will generate a continuous stream of audio. 
This audio is broken down into exactly one point two eight second chunks. 
Each chunk is sent to the GPU, where the neural network renders thirty two consecutive frames of lip synced video. 
Because the GPU takes less than one second to render these frames, it can stay ahead of the real time playback. 
The playback loop takes the rendered video frames and the exact matching audio bytes, and pushes them down the WebRTC tunnel together. 
As long as there is audio in the queue, my lips should continue to move perfectly in sync with the words you are hearing. 
If this works correctly, there should be no stuttering and no freezing. 
I will just keep talking and talking and talking to prove that the pipeline is completely stable and running in real time without any blockages. 
This is the ultimate test of the synchronized queue and the continuous video transport. 
I hope you are having a wonderful day, Aaron.
"""

class PerfectSyncPusher(FrameProcessor):
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
        
        self.video_source = rtc.VideoSource(self.width, self.height)
        self.video_track = rtc.LocalVideoTrack.create_video_track("bot-video", self.video_source)
        
        self.audio_source = rtc.AudioSource(self.sample_rate, 1)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("bot-audio", self.audio_source)
        
        self.published = False
        
        self.cached_audio_length_sum = self.sample_rate * self.cached_audio_duration
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.audio_dq = collections.deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
        
        self.audio_slice_samples = self.slice_len * self.sample_rate // self.tgt_fps
        
        self.audio_buffer = bytearray()
        self.audio_float_buffer = []
        
        self.playback_queue = collections.deque()
        self.generation_queue = asyncio.Queue()
        
        idle_img = cv2.imread("examples/girl.png")
        if idle_img is None:
            idle_img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            idle_img = cv2.resize(idle_img, (512, 512))
        self.idle_rgba = cv2.cvtColor(idle_img, cv2.COLOR_BGR2RGBA)
        self.idle_vf = rtc.VideoFrame(self.idle_rgba.shape[1], self.idle_rgba.shape[0], rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes())
        
        self.silent_audio_frame = rtc.AudioFrame(
            data=np.zeros(self.sample_rate // self.tgt_fps, dtype=np.int16).tobytes(),
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=self.sample_rate // self.tgt_fps
        )

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
                    video = run_pipeline(self.model_pipeline, audio_embedding)
                    torch.cuda.synchronize()
                    return video.cpu().numpy()
                
                start_gen = time.time()
                video_np = await asyncio.to_thread(run_infer)
                end_gen = time.time()
                logger.info(f"GPU rendered {video_np.shape[0]} frames in {end_gen - start_gen:.2f} seconds")
                
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
        
        if isinstance(frame, StartFrame) and not self.published:
            self.published = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())
            await self.push_frame(frame, direction)
            
        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, AudioRawFrame):
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

        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSStoppedFrame):
            if len(self.audio_float_buffer) > 0:
                pad_len = self.audio_slice_samples - len(self.audio_float_buffer)
                if pad_len > 0:
                    self.audio_float_buffer.extend([0.0] * pad_len)
                    self.audio_buffer.extend(b'\x00' * (pad_len * 2))
                
                chunk_floats = np.array(self.audio_float_buffer[:self.audio_slice_samples])
                self.audio_float_buffer = []
                
                chunk_bytes_len = self.audio_slice_samples * 2
                chunk_bytes = self.audio_buffer[:chunk_bytes_len]
                self.audio_buffer = bytearray()
                
                await self.generation_queue.put((chunk_floats, chunk_bytes))
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _video_loop(self):
        logger.info("Waiting for room to connect before publishing media...")
        while not hasattr(self.transport._client, "_room") or self.transport._client._room is None or not self.transport._client._room.isconnected():
            await asyncio.sleep(0.5)
            
        room = self.transport._client._room
        
        v_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await room.local_participant.publish_track(self.video_track, v_options)
        
        a_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(self.audio_track, a_options)
        
        logger.info("Starting perfectly synced playback loop...")
        
        while True:
            start_time = time.time()
            
            try:
                if len(self.playback_queue) > 0:
                    rgba, audio_bytes = self.playback_queue.popleft()
                    
                    vf = rtc.VideoFrame(rgba.shape[1], rgba.shape[0], rtc.VideoBufferType.RGBA, rgba.tobytes())
                    self.video_source.capture_frame(vf)
                    
                    af = rtc.AudioFrame(
                        data=bytes(audio_bytes),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        samples_per_channel=len(audio_bytes) // 2
                    )
                    await self.audio_source.capture_frame(af)
                else:
                    vf = rtc.VideoFrame(self.idle_rgba.shape[1], self.idle_rgba.shape[0], rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes())
                    self.video_source.capture_frame(vf)
                    af = rtc.AudioFrame(
                        data=np.zeros(int(self.sample_rate // self.tgt_fps), dtype=np.int16).tobytes(),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        samples_per_channel=int(self.sample_rate // self.tgt_fps)
                    )
                    await self.audio_source.capture_frame(af)
                    
            except Exception as e:
                logger.error(f"Error playing back frame: {e}")
            
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0/self.tgt_fps) - elapsed)
            await asyncio.sleep(sleep_time)

async def sentence_injector(task: PipelineTask):
    await asyncio.sleep(10)
    logger.info("Injecting massive paragraph for continuous testing.")
    while True:
        for sentence in PARAGRAPH.split('.'):
            sentence = sentence.strip()
            if not sentence: continue
            logger.info(f"Injecting sentence: {sentence}")
            await task.queue_frames([TextFrame(text=sentence + ".")])
            await asyncio.sleep(1)
        await asyncio.sleep(2)

async def main():
    try:
        url = 'wss://chatgptme-sp76gr03.livekit.cloud'
        api_key = 'API6pGtbWcmZpMs'
        api_secret = 'dlXcUvEGjHF7Q6btM2nAefWojeK5YgS82AxKBt6U9ncA'
        room_name = 'aarons-private-soulx-room-20260315-1341'

        token = api.AccessToken(api_key, api_secret) \
            .with_identity('soulx-video-bot') \
            .with_name('SoulX Avatar') \
            .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True, can_publish_data=True)) \
            .to_jwt()

        transport = LiveKitTransport(
            url=url, 
            room_name=room_name,
            token=token,
            params=LiveKitParams(
                audio_in_enabled=False,
                audio_out_enabled=False, 
                video_out_enabled=False, 
                vad_enabled=False
            )
        )
        
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY', 'sk_1efd713557745a1b5c15bb8389e95d51023600689bc874a1')
        tts = ElevenLabsTTSService(api_key=elevenlabs_api_key, voice_id='EXAVITQu4vr4xnSDxMaL')
        
        logger.info("Loading heavy SoulX Model into VRAM... This may take a minute.")
        model_pipeline = get_pipeline(
            world_size=1, 
            ckpt_dir="models/SoulX-FlashHead-1_3B", 
            wav2vec_dir="models/wav2vec2-base-960h", 
            model_type="lite"
        )
        
        get_base_data(model_pipeline, cond_image_path_or_dir="examples/girl.png", base_seed=42, use_face_crop=False)
        
        logger.info("Pre-warming the GPU to build CUDA graphs. This will take ~30-40 seconds...")
        infer_params = get_infer_params()
        sample_rate = infer_params['sample_rate']
        tgt_fps = infer_params['tgt_fps']
        cached_audio_duration = infer_params['cached_audio_duration']
        frame_num = infer_params['frame_num']
        motion_frames_num = infer_params['motion_frames_num']
        slice_len = frame_num - motion_frames_num
        
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num
        
        dummy_audio = np.zeros(cached_audio_length_sum, dtype=np.float32)
        torch.cuda.synchronize()
        dummy_embedding = get_audio_embedding(model_pipeline, dummy_audio, audio_start_idx, audio_end_idx)
        run_pipeline(model_pipeline, dummy_embedding)
        torch.cuda.synchronize()
        logger.info("SoulX Model fully loaded and GPU is pre-warmed.")

        pusher = PerfectSyncPusher(transport, model_pipeline)

        pipeline = Pipeline([
            tts,
            pusher
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False), idle_timeout_secs=None)
        
        runner = PipelineRunner()
        asyncio.create_task(sentence_injector(task))
        
        logger.info(f'Starting PERFECTLY SYNCED SoulX bot in {room_name}...')
        await runner.run(task)

    except Exception as e:
        logger.error(f'Error: {e}')

if __name__ == '__main__':
    asyncio.run(main())
