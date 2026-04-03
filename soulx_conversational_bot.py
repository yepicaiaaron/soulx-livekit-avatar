"""SoulX Conversational Brain

Full real-time pipeline orchestrated by Pipecat:
  User speaks via WebRTC → OpenAI Whisper STT → GPT-4o LLM (tool calling)
  → OpenAI TTS → WebRTCSyncPusher (SoulX avatar lip-sync + Daily.co publish)

Environment variables (see .env or Render dashboard):
  DAILY_ROOM_URL, DAILY_TOKEN
  OPENAI_API_KEY
  SOULX_MODEL_TYPE (lite|pro), SOULX_CKPT_DIR, SOULX_WAV2VEC_DIR
  SOULX_COND_IMAGE        path to avatar portrait (default: examples/omani_character.png)
  PERCEPTION_INTERVAL     seconds between visual analyses (default: 3.0)
"""

import ast
import asyncio
import operator as op
import os
import json
import numpy as np
import torch
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    StartFrame,
    LLMMessagesFrame,
    FunctionCallResultFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.services.openai import OpenAISTTService, OpenAILLMService, OpenAITTSService
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextAggregator,
)
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams

from webrtc_sync import WebRTCSyncPusher
from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
)
from perception_engine import PerceptionEngine

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are SoulX, an intelligent and empathetic AI avatar assistant powered by
the SoulX-FlashHead real-time diffusion engine. You communicate through a photorealistic
talking-head avatar in a live WebRTC session.

Guidelines:
- Be concise but thorough — aim for 1–3 sentences unless more is clearly needed.
- Show warmth and genuine curiosity about the user.
- When you use the get_visual_context tool, naturally acknowledge what you observe.
- Use calculate for any arithmetic the user requests.
- Your voice is rendered via text-to-speech, so avoid markdown, bullet points, or
  special characters that would sound unnatural when spoken.

Available tools:
  get_current_time   — current date and time
  get_visual_context — describes the user's webcam feed and shared screen in real time
  calculate          — evaluates a mathematical expression
"""

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_visual_context",
            "description": (
                "Return a real-time description of what the perception engine currently sees: "
                "the user's webcam feed and any screen being shared in the WebRTC session."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe mathematical expression, e.g. '2 + 2' or '(3 * 7) / 2'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (used by the 'calculate' tool)
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate a parsed AST node using only safe arithmetic ops."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](
            _safe_eval_node(node.left), _safe_eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """Parse and evaluate a numeric arithmetic expression safely via the AST."""
    tree = ast.parse(expression.strip(), mode="eval")
    result = _safe_eval_node(tree.body)
    # Return an integer string when the result is a whole number
    return str(int(result)) if result == int(result) else str(result)


# ---------------------------------------------------------------------------
# Tool handler factory
# ---------------------------------------------------------------------------
def make_tool_handlers(perception: PerceptionEngine):
    """Return per-tool async handlers bound to the given perception engine."""

    async def handle_get_current_time(
        function_name, tool_call_id, args, llm, context, result_callback
    ):
        result = {"time": datetime.now().strftime("%A, %d %B %Y %H:%M:%S")}
        await result_callback(result)

    async def handle_get_visual_context(
        function_name, tool_call_id, args, llm, context, result_callback
    ):
        observation = perception.get_visual_context()
        await result_callback({"observation": observation})

    async def handle_calculate(
        function_name, tool_call_id, args, llm, context, result_callback
    ):
        expression = args.get("expression", "")
        try:
            result = safe_calculate(expression)
            await result_callback({"result": result})
        except Exception as exc:
            await result_callback({"error": f"Could not evaluate expression: {exc}"})

    return {
        "get_current_time": handle_get_current_time,
        "get_visual_context": handle_get_visual_context,
        "calculate": handle_calculate,
    }


# ---------------------------------------------------------------------------
# GPU warmup helper
# ---------------------------------------------------------------------------
def warmup_gpu(model_pipeline):
    """Run a dummy inference pass to compile CUDA graphs before streaming starts."""
    logger.info("Pre-warming GPU (building CUDA graphs)… ~30–40 seconds.")
    infer_params = get_infer_params()
    sample_rate = infer_params["sample_rate"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    tgt_fps = infer_params["tgt_fps"]
    frame_num = infer_params["frame_num"]

    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    dummy_audio = np.zeros(cached_audio_length_sum, dtype=np.float32)
    torch.cuda.synchronize()
    dummy_embedding = get_audio_embedding(
        model_pipeline, dummy_audio, audio_start_idx, audio_end_idx
    )
    run_pipeline(model_pipeline, dummy_embedding)
    torch.cuda.synchronize()
    logger.info("GPU pre-warmed. SoulX model ready.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def main():
    load_dotenv()

    # ── Config ──────────────────────────────────────────────────────────────
    daily_room_url = os.environ.get("DAILY_ROOM_URL", "")
    daily_token = os.environ.get("DAILY_TOKEN", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    ckpt_dir = os.environ.get("SOULX_CKPT_DIR", "models/SoulX-FlashHead-1_3B")
    wav2vec_dir = os.environ.get("SOULX_WAV2VEC_DIR", "models/wav2vec2-base-960h")
    model_type = os.environ.get("SOULX_MODEL_TYPE", "lite")
    cond_image = os.environ.get("SOULX_COND_IMAGE", "examples/omani_character.png")

    perception_interval = float(os.environ.get("PERCEPTION_INTERVAL", "3.0"))

    if not daily_room_url:
        logger.error("DAILY_ROOM_URL must be set.")
        return
    if not openai_api_key:
        logger.error("OPENAI_API_KEY must be set.")
        return

    # ── Daily.co transport ───────────────────────────────────────────────────
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
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
    )

    # ── AI services ──────────────────────────────────────────────────────────
    stt = OpenAISTTService(api_key=openai_api_key, model="whisper-1")

    llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o")

    # OpenAI TTS — sample_rate=16000 to match SoulX-FlashHead inference config
    tts = OpenAITTSService(
        api_key=openai_api_key,
        voice="alloy",
        model="tts-1",
        sample_rate=16000,
    )

    # ── LLM context with tool definitions ────────────────────────────────────
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=TOOLS,
    )
    context_aggregator = llm.create_context_aggregator(context)

    # ── Perception engine ────────────────────────────────────────────────────
    perception = PerceptionEngine(
        daily_room_url=daily_room_url,
        daily_token=daily_token or None,
        openai_api_key=openai_api_key,
        capture_interval_secs=perception_interval,
    )

    # ── Register tool handlers ───────────────────────────────────────────────
    handlers = make_tool_handlers(perception)
    for name, handler in handlers.items():
        llm.register_function(name, handler)

    # ── Load SoulX model ─────────────────────────────────────────────────────
    logger.info("Loading SoulX model into VRAM… this may take a minute.")
    model_pipeline = get_pipeline(
        world_size=1,
        ckpt_dir=ckpt_dir,
        wav2vec_dir=wav2vec_dir,
        model_type=model_type,
    )
    get_base_data(
        model_pipeline,
        cond_image_path_or_dir=cond_image,
        base_seed=42,
        use_face_crop=False,
    )
    warmup_gpu(model_pipeline)

    # ── Avatar pusher ────────────────────────────────────────────────────────
    pusher = WebRTCSyncPusher(transport, model_pipeline)

    # ── Build pipeline ───────────────────────────────────────────────────────
    #
    #   Daily.co audio in (with VAD)
    #     → OpenAI Whisper STT             (audio → transcript)
    #     → LLM context aggregator (user)  (transcript → LLMMessagesFrame)
    #     → GPT-4o LLM with tool calling   (LLMMessagesFrame → text + tool calls)
    #     → OpenAI TTS                     (text → AudioRawFrame @ 16 kHz)
    #     → WebRTCSyncPusher               (AudioRawFrame → avatar video + audio in Daily.co)
    #     → LLM context aggregator (asst)  (assistant turn bookkeeping)
    #
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            pusher,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
        idle_timeout_secs=None,
    )
    runner = PipelineRunner()

    # Start perception engine concurrently with the pipeline
    logger.info(f"Starting SoulX Conversational Brain in Daily.co room: {daily_room_url}")
    await asyncio.gather(
        perception.start(),
        runner.run(task),
    )


if __name__ == "__main__":
    asyncio.run(main())
