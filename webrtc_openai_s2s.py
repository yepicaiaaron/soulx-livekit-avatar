import asyncio
import os
import numpy as np
import cv2
import time
from loguru import logger
import collections

from pipecat.frames.frames import AudioRawFrame, StartFrame, CancelFrame, InterruptionTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import EndFrame
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from livekit import api, rtc
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

import sys
sys.path.append(os.path.join(os.getcwd(), 'SoulX-FlashHead'))
from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

class WebRTCSyncPusher(FrameProcessor):
    def __init__(self, transport, model_pipeline, **kwargs):
        super().__init__(**kwargs)
        self.transport = transport
        self.model_pipeline = model_pipeline
        
        infer_params = get_infer_params()
        self.width = infer_params['width']
        self.height = infer_params['height']
        
        self.video_source = rtc.VideoSource(self.width, self.height)
        self.video_track = rtc.LocalVideoTrack.create_video_track("bot-video", self.video_source)
        
        self.sample_rate = infer_params['sample_rate']
        self.tgt_fps = infer_params['tgt_fps']
        self.frame_num = infer_params['frame_num']
        self.motion_frames_num = infer_params['motion_frames_num']
        self.slice_len = self.frame_num - self.motion_frames_num
        
        self.audio_source = rtc.AudioSource(self.sample_rate, 1)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("bot-audio", self.audio_source)
        
        self.cached_audio_length_sum = self.sample_rate * self.cached_audio_duration
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.audio_dq = collections.deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
        
        self.audio_slice_samples = self.slice_len * self.sample_rate // self.tgt_fps
        
        self.audio_buffer = bytearray()
        self.audio_float_buffer = []
        
        self.generation_queue = asyncio.Queue()
        self.playback_queue = collections.deque()
        self.is_publishing = False
        
        idle_bgr = cv2.imread("SoulX-FlashHead/examples/omani_character.png")
        if idle_bgr is None:
            idle_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            idle_img[:] = (0, 0, 255) # Red fallback block
        else:
            idle_img = cv2.resize(idle_bgr, (self.width, self.height))
        self.idle_rgba = cv2.cvtColor(idle_img, cv2.COLOR_BGR2RGBA)
        self._last_idle_frame = rtc.VideoFrame(self.idle_rgba.shape[1], self.idle_rgba.shape[0], rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes())
        
        self.silent_audio_frame = rtc.AudioFrame(
            data=np.zeros(int(self.sample_rate // self.tgt_fps), dtype=np.int16).tobytes(),
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=int(self.sample_rate // self.tgt_fps)
        )
        
    @property
    def cached_audio_duration(self):
        from flash_head.inference import get_infer_params
        return get_infer_params()['cached_audio_duration']
        
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
            
        if not isinstance(frame, AudioRawFrame) and not isinstance(frame, StartFrame) and not isinstance(frame, EndFrame):
            pass
            
        if isinstance(frame, (CancelFrame, InterruptionTaskFrame)) or type(frame).__name__ == "InterruptionTaskFrame" or type(frame).__name__ == "CancelFrame":
            logger.info("Interruption detected! Clearing all video and audio buffers.")
            self.audio_buffer.clear()
            self.audio_float_buffer.clear()
            
            while not self.generation_queue.empty():
                try:
                    self.generation_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            self.playback_queue.clear()
            self.audio_dq = collections.deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
            
        from pipecat.frames.frames import InputAudioRawFrame
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, AudioRawFrame) and not isinstance(frame, InputAudioRawFrame):
            audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            expected_sr = getattr(frame, 'sample_rate', 24000)
            if not hasattr(self, 'raw_audio_float_buffer'):
                self.raw_audio_float_buffer = []
                
            self.raw_audio_float_buffer.extend(audio_data.tolist())
            
            required_samples = self.slice_len * expected_sr // self.tgt_fps
            
            while len(self.raw_audio_float_buffer) >= required_samples:
                raw_chunk = np.array(self.raw_audio_float_buffer[:required_samples])
                self.raw_audio_float_buffer = self.raw_audio_float_buffer[required_samples:]
                
                if expected_sr != 16000:
                    import librosa
                    processed_chunk = librosa.resample(raw_chunk, orig_sr=expected_sr, target_sr=16000)
                else:
                    processed_chunk = raw_chunk
                    
                chunk_bytes = (processed_chunk * 32768.0).astype(np.int16).tobytes()
                await self.generation_queue.put((processed_chunk, chunk_bytes))

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
            if not hasattr(self, '_started_playing'):
                self._started_playing = True
                next_frame_time = time.perf_counter()
            
            try:
                if len(self.playback_queue) > 1000:
                    logger.warning(f"Playback queue too large ({len(self.playback_queue)}). This is a severe lag bug.")
                        
                if len(self.playback_queue) > 0:
                    rgba, audio_bytes = self.playback_queue.popleft()
                    
                    i420 = cv2.cvtColor(rgba, cv2.COLOR_RGBA2YUV_I420)
                    self._last_frame_bytes = i420.tobytes()
                    self._last_frame = rtc.VideoFrame(i420.shape[1], int(i420.shape[0] * 2 / 3), 5, self._last_frame_bytes) # 5 = I420
                    self.video_source.capture_frame(self._last_frame)
                    
                    af = rtc.AudioFrame(
                        data=bytes(audio_bytes),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        samples_per_channel=len(audio_bytes) // 2
                    )
                    await self.audio_source.capture_frame(af)
                else:
                    idle_i420 = cv2.cvtColor(self.idle_rgba, cv2.COLOR_RGBA2YUV_I420)
                    self._last_idle_bytes = idle_i420.tobytes()
                    self._last_idle_frame = rtc.VideoFrame(idle_i420.shape[1], int(idle_i420.shape[0] * 2 / 3), 5, self._last_idle_bytes)
                    self.video_source.capture_frame(self._last_idle_frame)
                    await self.audio_source.capture_frame(self.silent_audio_frame)
                    
            except Exception as e:
                logger.error(f"Error playing back frame: {e}")
            
            next_frame_time += (1.0 / self.tgt_fps)
            now = time.perf_counter()
            sleep_time = next_frame_time - now
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_frame_time = now
                
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
                    return video.float().cpu().numpy()
                
                t_start = time.perf_counter()
                video_np = await asyncio.to_thread(run_infer)
                t_end = time.perf_counter()
                logger.info(f"GPU rendered {video_np.shape[0]} frames in {t_end - t_start:.3f} seconds.")
                
                video_np = video_np[self.motion_frames_num:]
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

async def main():
    try:
        url = os.environ.get('LIVEKIT_URL', 'wss://chatgptme-sp76gr03.livekit.cloud')
        api_key = os.environ.get('LIVEKIT_API_KEY')
        api_secret = os.environ.get('LIVEKIT_API_SECRET')
        room_name = os.environ.get('LIVEKIT_ROOM', 'soulx-flashhead-room')

        token = api.AccessToken(api_key, api_secret) \
            .with_identity("soulx-video-bot") \
            .with_name("SoulX Avatar") \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
            )).to_jwt()

        
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        vad = SileroVADAnalyzer(params=VADParams(min_volume=0.2))
        
        transport = LiveKitTransport(
            url=url, 
            room_name=room_name,
            token=token,
            params=LiveKitParams(
                audio_in_enabled=True,
                audio_out_enabled=False, 
                video_out_enabled=False, 
                vad_analyzer=vad,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000
            )
        )

        
        logger.info("Loading heavy SoulX Model into VRAM... This may take a minute.")
        
        model_pipeline = get_pipeline(
            world_size=1, 
            ckpt_dir="models/SoulX-FlashHead-1_3B", 
            wav2vec_dir="models/wav2vec2-base-960h", 
            model_type="lite"
        )
        
        logger.info("Pre-warming the GPU to build CUDA graphs. This will take ~30-40 seconds...")
        infer_params = get_infer_params()
        cached_audio_length_sum = infer_params['sample_rate'] * infer_params['cached_audio_duration']
        audio_end_idx = int(infer_params['cached_audio_duration'] * infer_params['tgt_fps'])
        audio_start_idx = audio_end_idx - infer_params['frame_num']
        
        dummy_audio = np.zeros(cached_audio_length_sum, dtype=np.float32)
        get_base_data(model_pipeline, cond_image_path_or_dir="SoulX-FlashHead/examples/omani_character.png", base_seed=42, use_face_crop=False)
        torch.cuda.synchronize()
        dummy_embedding = get_audio_embedding(model_pipeline, dummy_audio, audio_start_idx, audio_end_idx)
        run_pipeline(model_pipeline, dummy_embedding)
        torch.cuda.synchronize()
        logger.info("SoulX Model fully loaded and GPU is pre-warmed.")

        pusher = WebRTCSyncPusher(transport, model_pipeline)
        
        llm = OpenAIRealtimeLLMService(
            api_key = os.environ.get('LIVEKIT_API_KEY'),
            model="gpt-4o-realtime-preview-2024-12-17"
        )
        
        context = OpenAILLMContext(messages=[{
            "role": "system",
            "content": "You are a concise, helpful assistant talking over pure audio. Never use lists."
        }])
        
        
        
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams, LLMAssistantAggregatorParams
        user_params = LLMUserAggregatorParams()
        from pipecat.turns.user_turn_strategies import UserTurnStrategies
        setattr(user_params, 'user_turn_strategies', UserTurnStrategies())
        user_params.vad_analyzer = vad
        assistant_params = LLMAssistantAggregatorParams(expect_stripped_words=False)
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=user_params,
            assistant_params=assistant_params,
        )



        
        pipeline = Pipeline([
            transport.input(),
            context_aggregator.user(),
            llm,
            pusher,
            transport.output(),
        ])

        task = PipelineTask(pipeline, params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
        ))

        @transport.event_handler("on_participant_connected")
        async def on_participant_connected(transport, participant):
            logger.info(f"Participant connected: {participant}")
            
        @transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant):
            logger.info(f"Participant disconnected: {participant}")
        
        runner = PipelineRunner()
        
        logger.info(f'Starting CONTINUOUS WebRTC streaming bot in {room_name}...')
        await runner.run(task)

    except Exception as e:
        import traceback
        logger.error(f'Error: {e}\n{traceback.format_exc()}')

if __name__ == '__main__':
    asyncio.run(main())
