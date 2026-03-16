import asyncio
import os
import json
import base64
import websockets
from loguru import logger

from pipecat.frames.frames import AudioRawFrame, StartFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from livekit import api

from dotenv import load_dotenv
load_dotenv()

class ElevenLabsAgentProcessor(FrameProcessor):
    def __init__(self, agent_id, sample_rate=16000, **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.sample_rate = sample_rate
        self.ws = None
        self.is_running = False
        self.agent_ready = False
        self._mic_audio_buffer = bytearray()

    async def _ws_receive_loop(self):
        try:
            while self.is_running and self.ws:
                msg = await self.ws.recv()
                data = json.loads(msg)
                
                if data.get('type') in ('conversation_initiation_metadata_event', 'conversation_initiation_metadata'):
                    logger.info(f'ElevenLabs Agent is ready to receive audio: {data.get("type")}')
                    self.agent_ready = True
                elif data.get('type') == 'audio':
                    audio_b64 = data.get('audio_event', {}).get('audio_base_64')
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        # Push audio downstream to LiveKit Output Transport
                        await self.push_frame(AudioRawFrame(audio=audio_bytes, sample_rate=self.sample_rate, num_channels=1), FrameDirection.DOWNSTREAM)
                elif data.get('type') == 'interruption':
                    logger.info('Agent was interrupted')
                elif data.get('type') == 'ping':
                    await self.ws.send(json.dumps({'type': 'pong', 'event_id': data.get('event_id')}))
        except websockets.exceptions.ConnectionClosed:
            logger.info('ElevenLabs WebSocket closed')
        except Exception as e:
            logger.error(f'Error in ElevenLabs WS receive loop: {e}')

    async def process_frame(self, frame, direction):
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
            self.is_running = True
            uri = f'wss://api.elevenlabs.io/v1/convai/conversation?agent_id={self.agent_id}'
            try:
                self.ws = await websockets.connect(uri)
                logger.info(f'Connected to ElevenLabs Agent {self.agent_id}')
                asyncio.create_task(self._ws_receive_loop())
            except Exception as e:
                logger.error(f'Failed to connect to ElevenLabs Agent: {e}')
            
        elif isinstance(frame, EndFrame):
            self.is_running = False
            if self.ws:
                await self.ws.close()
            await super().process_frame(frame, direction)
            
        elif isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            # Send mic audio to ElevenLabs
            if self.ws and getattr(self, 'agent_ready', False):
                self._mic_audio_buffer.extend(frame.audio)
                
                # ElevenLabs requires 16-bit PCM (2 bytes per sample). Ensure even byte length.
                chunk_size = 3200  # 200ms at 16kHz
                while len(self._mic_audio_buffer) >= chunk_size:
                    chunk = self._mic_audio_buffer[:chunk_size]
                    self._mic_audio_buffer = self._mic_audio_buffer[chunk_size:]
                    
                    audio_b64 = base64.b64encode(chunk).decode('utf-8')
                    try:
                        await self.ws.send(json.dumps({
                            'type': 'user_audio_chunk',
                            'user_audio_chunk': audio_b64
                        }))
                    except Exception as e:
                        logger.error(f'Failed to send audio to ElevenLabs: {e}')
            # Drop mic audio so it doesn't get echoed to the user
            return
        else:
            await super().process_frame(frame, direction)

async def main():
    try:
        url = os.getenv('LIVEKIT_URL', 'wss://chatgptme-sp76gr03.livekit.cloud')
        api_key = os.getenv('LIVEKIT_API_KEY', 'API6pGtbWcmZpMs')
        api_secret = os.getenv('LIVEKIT_API_SECRET', 'dlXcUvEGjHF7Q6btM2nAefWojeK5YgS82AxKBt6U9ncA')
        room_name = 'aarons-private-soulx-room-20260315-1341'

        token = api.AccessToken(api_key, api_secret)             .with_identity('elevenlabs-voice-only')             .with_name('ElevenLabs Agent')             .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True, can_publish_data=True))             .to_jwt()

        transport = LiveKitTransport(
            url=url, 
            room_name=room_name,
            token=token,
            params=LiveKitParams(
                audio_in_enabled=True,
                audio_out_enabled=True, 
                video_out_enabled=False, 
                vad_enabled=False,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000
            )
        )
        
        # Connect to ElevenLabs Conversational Agent
        agent_id = 'agent_3601kkw8mwj8e05t42gk47zaas1p'
        el_agent = ElevenLabsAgentProcessor(agent_id=agent_id)

        pipeline = Pipeline([
            transport.input(),
            el_agent,
            transport.output()
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True), idle_timeout_secs=None)
        
        runner = PipelineRunner()
        
        logger.info(f'Starting ElevenLabs Voice-Only test in {room_name}...')
        await runner.run(task)

    except Exception as e:
        logger.error(f'Error: {e}')

if __name__ == '__main__':
    asyncio.run(main())
