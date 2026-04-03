"""SoulX Perception Engine

Real-time visual perception for the SoulX avatar. Subscribes to video tracks
in the WebRTC session, periodically captures frames from:
  • participant webcam feeds
  • screen-share tracks

…and analyses them via the OpenAI Vision API (GPT-4o) to produce a natural-
language description that the conversational LLM can query as a tool.

Primary transport: LiveKit (mirrors the transport used by the avatar bot).
Optional transport: Daily.co — requires `pip install daily-python`.

Usage (standalone):
  python perception_engine.py

Usage (embedded — imported by soulx_conversational_bot.py):
  from perception_engine import PerceptionEngine, DailyPerceptionEngine

Environment variables used when run standalone:
  LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_ROOM
  OPENAI_API_KEY
  PERCEPTION_INTERVAL   seconds between Vision-API calls (default: 3.0)
  DAILY_ROOM_URL        daily.co room URL (optional)
  DAILY_TOKEN           daily.co participant token (optional)
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

from livekit import api as livekit_api
from livekit import rtc


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
# LiveKit perception engine
# ---------------------------------------------------------------------------

class PerceptionEngine:
    """
    Joins a LiveKit room as a silent observer, subscribes to every video track,
    and uses OpenAI Vision to describe the frames on a configurable interval.

    Thread-safety note: frame buffers are written from asyncio callbacks and
    read from the analysis loop — both run on the same event loop, so a plain
    dict is safe here.
    """

    def __init__(
        self,
        livekit_url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        openai_api_key: str,
        vision_model: str = "gpt-4o",
        capture_interval_secs: float = 3.0,
        max_frame_dimension: int = 768,
    ) -> None:
        self.livekit_url = livekit_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_name = room_name
        self.openai = AsyncOpenAI(api_key=openai_api_key)
        self.vision_model = vision_model
        self.capture_interval_secs = capture_interval_secs
        self.max_frame_dimension = max_frame_dimension

        self._room: Optional[rtc.Room] = None
        self._latest_observation: Optional[VisualObservation] = None
        # participant_identity → latest BGR frame (numpy)
        self._webcam_frames: Dict[str, np.ndarray] = {}
        self._screen_frames: Dict[str, np.ndarray] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

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
        """Connect to the LiveKit room and begin background perception."""
        logger.info(f"[Perception] Connecting to LiveKit room '{self.room_name}' as observer…")
        self._running = True

        token = (
            livekit_api.AccessToken(self.api_key, self.api_secret)
            .with_identity("soulx-perception-observer")
            .with_name("SoulX Perception")
            .with_grants(
                livekit_api.VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_subscribe=True,
                    can_publish=False,
                )
            )
            .to_jwt()
        )

        self._room = rtc.Room()

        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                is_screen = publication.source == rtc.TrackSource.SOURCE_SCREENSHARE
                label = "screen-share" if is_screen else "webcam"
                logger.info(
                    f"[Perception] Subscribed to {label} from '{participant.identity}'"
                )
                task = asyncio.create_task(
                    self._consume_video_track(
                        track, participant.identity, is_screen=is_screen
                    )
                )
                self._tasks.append(task)

        @self._room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            pid = participant.identity
            self._webcam_frames.pop(pid, None)
            self._screen_frames.pop(pid, None)
            logger.info(f"[Perception] Participant '{pid}' left; cleared their frames.")

        await self._room.connect(self.livekit_url, token)
        logger.info("[Perception] Connected. Waiting for video tracks…")

        analysis_task = asyncio.create_task(self._analysis_loop())
        self._tasks.append(analysis_task)

    async def stop(self) -> None:
        """Cancel background tasks and disconnect from LiveKit."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._room:
            await self._room.disconnect()
        logger.info("[Perception] Stopped.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _consume_video_track(
        self, track: rtc.VideoTrack, participant_id: str, *, is_screen: bool
    ) -> None:
        """Read frames from a video track and store the most recent one."""
        store = self._screen_frames if is_screen else self._webcam_frames
        video_stream = rtc.VideoStream(track)
        async for event in video_stream:
            if not self._running:
                break
            try:
                frame: rtc.VideoFrame = event.frame
                # LiveKit delivers RGBA frames
                rgba = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                    frame.height, frame.width, 4
                )
                store[participant_id] = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            except Exception as exc:
                logger.warning(f"[Perception] Frame decode error ({participant_id}): {exc}")

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
        Encode up to `frames` as JPEG, submit them to OpenAI Vision, and
        return a concise description, or None if there are no frames.
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


# ---------------------------------------------------------------------------
# Daily.co perception engine (optional extension)
# ---------------------------------------------------------------------------

class DailyPerceptionEngine(PerceptionEngine):
    """
    Extends PerceptionEngine to also join a Daily.co room and subscribe to
    participant video tracks (webcam + screen share).

    Requires:
        pip install daily-python

    The Daily.co frames are stored in the same `_webcam_frames` /
    `_screen_frames` dicts as LiveKit frames, so the analysis loop picks them
    up automatically.

    Screen-share vs. webcam classification: the engine listens to the
    `on_participant_updated` event and tracks which participant IDs have an
    active `screenVideo` track (state == "playable"). Frames from those
    participants are routed to `_screen_frames`; all others go to
    `_webcam_frames`.
    """

    def __init__(
        self,
        daily_room_url: str,
        daily_token: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.daily_room_url = daily_room_url
        self.daily_token = daily_token
        self._daily_client = None
        # Set of Daily.co participant IDs whose screenVideo track is currently active
        self._daily_screenshare_ids: set = set()

    async def start(self) -> None:
        """Start both LiveKit and Daily.co perception concurrently."""
        await super().start()
        try:
            await asyncio.to_thread(self._start_daily_sync)
        except ImportError:
            logger.warning(
                "[Perception] daily-python not installed — Daily.co capture skipped. "
                "Install with: pip install daily-python"
            )
        except Exception as exc:
            logger.error(f"[Perception] Daily.co connection error: {exc}")

    def _start_daily_sync(self) -> None:
        """Blocking helper to initialise the Daily.co client (called via to_thread)."""
        from daily import CallClient, Daily  # type: ignore[import]

        perception_ref = self  # capture for closure

        class _DailyHandler:
            def on_participant_joined(self, participant):
                logger.info(f"[DailyPerception] Participant joined: {participant['id']}")

            def on_participant_updated(self, participant) -> None:
                """Track which participants have an active screenVideo stream."""
                pid = participant.get("id", "")
                tracks = participant.get("tracks", {})
                screen_state = tracks.get("screenVideo", {}).get("state", "")
                if screen_state == "playable":
                    perception_ref._daily_screenshare_ids.add(pid)
                    logger.debug(f"[DailyPerception] Screen-share active: {pid}")
                else:
                    perception_ref._daily_screenshare_ids.discard(pid)

            def on_participant_left(self, participant, reason) -> None:
                pid = participant.get("id", "")
                perception_ref._daily_screenshare_ids.discard(pid)
                perception_ref._webcam_frames.pop(f"daily_{pid}", None)
                perception_ref._screen_frames.pop(f"daily_{pid}", None)

            def on_video_frame(self, participant_id: str, video_frame) -> None:
                try:
                    w, h = video_frame.width, video_frame.height
                    rgba = np.frombuffer(video_frame.buffer, dtype=np.uint8).reshape(h, w, 4)
                    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    key = f"daily_{participant_id}"
                    # Route to screen or webcam based on reliably-tracked state
                    if participant_id in perception_ref._daily_screenshare_ids:
                        perception_ref._screen_frames[key] = bgr
                    else:
                        perception_ref._webcam_frames[key] = bgr
                except Exception as exc:
                    logger.warning(f"[DailyPerception] Frame error: {exc}")

        Daily.init()
        self._daily_client = CallClient(event_handler=_DailyHandler())

        join_kwargs = {"url": self.daily_room_url}
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
        logger.info(f"[DailyPerception] Joined Daily.co room: {self.daily_room_url}")

    async def stop(self) -> None:
        if self._daily_client:
            try:
                self._daily_client.leave()
            except Exception:
                pass
        await super().stop()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

async def _run_standalone() -> None:
    load_dotenv()

    daily_url = os.environ.get("DAILY_ROOM_URL", "")
    daily_token = os.environ.get("DAILY_TOKEN", "")

    common_kwargs = dict(
        livekit_url=os.environ.get("LIVEKIT_URL", "wss://chatgptme-sp76gr03.livekit.cloud"),
        api_key=os.environ.get("LIVEKIT_API_KEY", ""),
        api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
        room_name=os.environ.get("LIVEKIT_ROOM", "soulx-flashhead-room"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        capture_interval_secs=float(os.environ.get("PERCEPTION_INTERVAL", "3.0")),
    )

    if daily_url:
        engine: PerceptionEngine = DailyPerceptionEngine(
            daily_room_url=daily_url,
            daily_token=daily_token or None,
            **common_kwargs,
        )
    else:
        engine = PerceptionEngine(**common_kwargs)

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
