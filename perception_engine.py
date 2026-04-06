"""SoulX Perception Engine

Real-time visual perception for the SoulX avatar. Subscribes to video tracks
in a Daily.co WebRTC session, periodically captures frames from:
  • participant webcam feeds
  • screen-share tracks

…and analyses them via the OpenAI Vision API (GPT-4o) to produce a natural-
language description that the conversational LLM can query as a tool.

Transport: Daily.co — requires `pip install daily-python`.

Usage (standalone):
  python perception_engine.py

Usage (embedded — imported by soulx_conversational_bot.py):
  from perception_engine import PerceptionEngine

Environment variables used when run standalone:
  DAILY_ROOM_URL        Daily.co room URL (required)
  DAILY_TOKEN           Daily.co participant token (optional)
  OPENAI_API_KEY
  PERCEPTION_INTERVAL   seconds between Vision-API calls (default: 3.0)
"""

import asyncio
import base64
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VisualObservation:
    """Snapshot of what the perception engine saw at a given moment."""

    timestamp: float
    webcam_description: Optional[str] = None
    screen_description: Optional[str] = None

    def to_context_string(self) -> str:
        parts: List[str] = []
        if self.webcam_description:
            parts.append(f"User webcam: {self.webcam_description}")
        if self.screen_description:
            parts.append(f"Shared screen: {self.screen_description}")
        return " | ".join(parts) if parts else "No visual content detected."


# ---------------------------------------------------------------------------
# Daily.co perception engine
# ---------------------------------------------------------------------------

class PerceptionEngine:
    """
    Joins a Daily.co room as a silent observer, subscribes to every video track,
    and uses OpenAI Vision to describe the frames on a configurable interval.

    Requires:
        pip install daily-python

    Thread-safety note: frame buffers are written from Daily.co callbacks (which
    run in a background thread) and read from the async analysis loop — writes
    are atomic dict assignments, which is safe in CPython.
    """

    def __init__(
        self,
        daily_room_url: str,
        openai_api_key: str,
        daily_token: Optional[str] = None,
        vision_model: str = "gpt-4o",
        capture_interval_secs: float = 3.0,
        max_frame_dimension: int = 768,
    ) -> None:
        self.daily_room_url = daily_room_url
        self.daily_token = daily_token
        self.openai = AsyncOpenAI(api_key=openai_api_key)
        self.vision_model = vision_model
        self.capture_interval_secs = capture_interval_secs
        self.max_frame_dimension = max_frame_dimension

        self._daily_client = None
        self._latest_observation: Optional[VisualObservation] = None
        # participant_identity → latest BGR frame (numpy)
        self._webcam_frames: Dict[str, np.ndarray] = {}
        self._screen_frames: Dict[str, np.ndarray] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        # Set of Daily.co participant IDs whose screenVideo track is currently active
        self._screenshare_ids: set = set()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def latest_observation(self) -> Optional[VisualObservation]:
        return self._latest_observation

    def get_visual_context(self) -> str:
        """Return the latest visual context string for the LLM tool handler."""
        if self._latest_observation is None:
            return "Perception engine is starting up — no visual context available yet."
        age = time.time() - self._latest_observation.timestamp
        return f"{self._latest_observation.to_context_string()} (captured {age:.1f}s ago)"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to the Daily.co room and begin background perception."""
        self._running = True
        logger.info(f"[Perception] Connecting to Daily.co room: {self.daily_room_url}")

        # Start the analysis loop (reads from the frame buffers)
        analysis_task = asyncio.create_task(self._analysis_loop())
        self._tasks.append(analysis_task)

        # Connect to Daily.co in a background thread (blocking SDK)
        try:
            await asyncio.to_thread(self._start_daily_sync)
        except ImportError:
            logger.warning(
                "[Perception] daily-python not installed — Daily.co capture unavailable. "
                "Install with: pip install daily-python"
            )
        except Exception as exc:
            logger.error(f"[Perception] Daily.co connection error: {exc}")

    async def stop(self) -> None:
        """Leave the Daily.co room and cancel background tasks."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._daily_client:
            try:
                self._daily_client.leave()
            except Exception:
                pass
        logger.info("[Perception] Stopped.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _start_daily_sync(self) -> None:
        """Blocking helper to initialise the Daily.co client (runs in a thread)."""
        from daily import CallClient, Daily, EventHandler  # type: ignore[import]

        perception_ref = self

        class _DailyHandler(EventHandler):
            def on_participant_joined(self, participant) -> None:
                logger.info(f"[Perception] Participant joined: {participant.get('id', 'unknown')}")

            def on_participant_updated(self, participant) -> None:
                """Track which participants have an active screenVideo stream."""
                pid = participant.get("id", "")
                tracks = participant.get("tracks", {})
                screen_state = tracks.get("screenVideo", {}).get("state", "")
                if screen_state == "playable":
                    perception_ref._screenshare_ids.add(pid)
                    logger.debug(f"[Perception] Screen-share active: {pid}")
                else:
                    perception_ref._screenshare_ids.discard(pid)

            def on_participant_left(self, participant, reason) -> None:
                pid = participant.get("id", "")
                perception_ref._screenshare_ids.discard(pid)
                perception_ref._webcam_frames.pop(pid, None)
                perception_ref._screen_frames.pop(pid, None)
                logger.info(f"[Perception] Participant '{pid}' left; cleared their frames.")

            def on_video_frame(self, participant_id: str, video_frame) -> None:
                try:
                    w, h = video_frame.width, video_frame.height
                    rgba = np.frombuffer(video_frame.buffer, dtype=np.uint8).reshape(h, w, 4)
                    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    if participant_id in perception_ref._screenshare_ids:
                        perception_ref._screen_frames[participant_id] = bgr
                    else:
                        perception_ref._webcam_frames[participant_id] = bgr
                except Exception as exc:
                    logger.warning(f"[Perception] Frame error ({participant_id}): {exc}")

        # Guard against double-init: pipecat's DailyTransportClient uses a class-level
        # flag to call Daily.init() exactly once per process. Reuse that flag so the
        # perception engine doesn't re-initialise the Daily Rust context and panic.
        try:
            from pipecat.transports.daily.transport import DailyTransportClient as _DTC
            if not _DTC._daily_initialized:
                _DTC._daily_initialized = True
                Daily.init()
        except ImportError:
            Daily.init()  # pipecat not present — standalone mode
        self._daily_client = CallClient(event_handler=_DailyHandler())

        join_kwargs: dict = {"url": self.daily_room_url}
        if self.daily_token:
            join_kwargs["token"] = self.daily_token

        self._daily_client.join(**join_kwargs)
        self._daily_client.update_subscription_profiles(
            {
                "base": {
                    "camera": "subscribed",
                    "screenVideo": "subscribed",
                }
            }
        )
        logger.info(f"[Perception] Joined Daily.co room: {self.daily_room_url}")

    async def _analysis_loop(self) -> None:
        """Periodically analyse buffered frames via OpenAI Vision."""
        logger.info("[Perception] Analysis loop started.")
        while self._running:
            await asyncio.sleep(self.capture_interval_secs)
            try:
                webcam_desc = await self._describe_frames(
                    dict(self._webcam_frames), context_label="user's webcam"
                )
                screen_desc = await self._describe_frames(
                    dict(self._screen_frames), context_label="shared screen"
                )
                if webcam_desc is not None or screen_desc is not None:
                    self._latest_observation = VisualObservation(
                        timestamp=time.time(),
                        webcam_description=webcam_desc,
                        screen_description=screen_desc,
                    )
                    logger.debug(
                        f"[Perception] Updated — "
                        f"webcam: {(webcam_desc or '')[:80]} | "
                        f"screen: {(screen_desc or '')[:80]}"
                    )
            except Exception as exc:
                logger.error(f"[Perception] Analysis error: {exc}")

    async def _describe_frames(
        self, frames: Dict[str, np.ndarray], context_label: str
    ) -> Optional[str]:
        """
        Encode frames as JPEG, submit them to OpenAI Vision, and return a
        concise description, or None if there are no frames.
        """
        if not frames:
            return None

        image_contents = []
        for pid, bgr in frames.items():
            try:
                h, w = bgr.shape[:2]
                if max(h, w) > self.max_frame_dimension:
                    scale = self.max_frame_dimension / max(h, w)
                    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
                _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                image_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    }
                )
            except Exception as exc:
                logger.warning(f"[Perception] Image encode error ({pid}): {exc}")

        if not image_contents:
            return None

        prompt = (
            f"You are analyzing a real-time {context_label} frame from a live video call. "
            "Describe in 1–2 sentences what you see: the person's appearance and current activity, "
            "any visible text, objects, or on-screen content. "
            "Focus on details an AI assistant would find contextually useful. Be concise."
        )

        response = await self.openai.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + image_contents,
                }
            ],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()


# Backward-compatible alias
DailyPerceptionEngine = PerceptionEngine


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

async def _run_standalone() -> None:
    load_dotenv()

    daily_url = os.environ.get("DAILY_ROOM_URL", "")
    if not daily_url:
        logger.error("[Perception] DAILY_ROOM_URL must be set.")
        return

    engine = PerceptionEngine(
        daily_room_url=daily_url,
        daily_token=os.environ.get("DAILY_TOKEN") or None,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        capture_interval_secs=float(os.environ.get("PERCEPTION_INTERVAL", "3.0")),
    )

    await engine.start()
    logger.info("[Perception] Running standalone. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(10)
            obs = engine.latest_observation
            if obs:
                logger.info(f"[Perception] Latest: {obs.to_context_string()}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(_run_standalone())
