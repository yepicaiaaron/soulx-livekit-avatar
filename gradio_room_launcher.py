"""SoulX Avatar — Gradio Room Launcher

Provides a public Gradio URL that:
  1. Creates a new Daily.co room via the REST API (no pre-made room needed)
  2. Launches soulx_conversational_bot.py as a subprocess
  3. Embeds the Daily.co Prebuilt UI so users can speak to the avatar immediately

Required environment variables (set via Lightning.ai Secrets):
  DAILY_API_KEY   — Daily.co API key (creates rooms automatically)
  OPENAI_API_KEY  — OpenAI key for Whisper STT + GPT-4o LLM + TTS

Optional:
  SOULX_MODEL_TYPE      lite (default) or pro
  SOULX_COND_IMAGE      path to avatar portrait image
  PERCEPTION_INTERVAL   seconds between vision analyses (default: 3.0)
"""

import os
import time
import subprocess
import threading
import requests
import gradio as gr
from loguru import logger

# ---------------------------------------------------------------------------
# Config (pulled from environment — set via Lightning.ai Secrets panel)
# ---------------------------------------------------------------------------
DAILY_API_KEY = os.environ.get("DAILY_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Per-session bot processes: room_url -> Popen
_active_bots: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Daily.co helpers
# ---------------------------------------------------------------------------

def create_daily_room(api_key: str) -> str:
    """Call the Daily.co REST API to create a fresh room and return its URL."""
    resp = requests.post(
        "https://api.daily.co/v1/rooms",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "properties": {
                "exp": int(time.time()) + 7200,   # 2-hour expiry
                "enable_prejoin_ui": False,
                "enable_people_ui": False,
                "start_video_off": False,
                "start_audio_off": False,
            }
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    room_url: str = data["url"]
    logger.info(f"[Launcher] Daily.co room created: {room_url}")
    return room_url


def delete_daily_room(api_key: str, room_url: str) -> None:
    """Delete a Daily.co room when the session ends (best-effort)."""
    try:
        room_name = room_url.rstrip("/").split("/")[-1]
        requests.delete(
            f"https://api.daily.co/v1/rooms/{room_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        logger.info(f"[Launcher] Daily.co room deleted: {room_name}")
    except Exception as exc:
        logger.warning(f"[Launcher] Could not delete room: {exc}")


# ---------------------------------------------------------------------------
# Bot lifecycle
# ---------------------------------------------------------------------------

def _launch_bot(room_url: str) -> subprocess.Popen:
    """Start soulx_conversational_bot.py with the given Daily.co room URL."""
    env = os.environ.copy()
    env["DAILY_ROOM_URL"] = room_url
    env["OPENAI_API_KEY"] = OPENAI_API_KEY
    env["SOULX_MODEL_TYPE"] = os.environ.get("SOULX_MODEL_TYPE", "lite")
    env["SOULX_CKPT_DIR"] = os.environ.get("SOULX_CKPT_DIR", "./models/SoulX-FlashHead-1_3B")
    env["SOULX_WAV2VEC_DIR"] = os.environ.get("SOULX_WAV2VEC_DIR", "./models/wav2vec2-base-960h")
    env["SOULX_COND_IMAGE"] = os.environ.get("SOULX_COND_IMAGE", "./examples/omani_character.png")
    env["PERCEPTION_INTERVAL"] = os.environ.get("PERCEPTION_INTERVAL", "3.0")

    proc = subprocess.Popen(
        ["python", "soulx_conversational_bot.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    logger.info(f"[Launcher] Bot started (PID {proc.pid}) for room: {room_url}")
    return proc


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def start_session(state: dict):
    """Create a room, start the bot, return status + embedded UI."""
    if not DAILY_API_KEY:
        return (
            state,
            "❌ DAILY_API_KEY is not set.  "
            "Go to the Lightning.ai Studio → left sidebar → 🔒 Secrets → add DAILY_API_KEY.",
            "",
            "",
        )
    if not OPENAI_API_KEY:
        return (
            state,
            "❌ OPENAI_API_KEY is not set.  "
            "Go to the Lightning.ai Studio → left sidebar → 🔒 Secrets → add OPENAI_API_KEY.",
            "",
            "",
        )

    # If there is already a running session for this browser tab, reuse it
    if state.get("room_url") and state.get("pid"):
        room_url = state["room_url"]
        return state, "✅ Session already running!", room_url, _make_embed(room_url)

    try:
        room_url = create_daily_room(DAILY_API_KEY)
        proc = _launch_bot(room_url)

        with _lock:
            _active_bots[room_url] = proc

        state["room_url"] = room_url
        state["pid"] = proc.pid

        return (
            state,
            (
                "✅ Avatar is booting up — the GPU warmup takes ~40 seconds. "
                "Join the room below and you'll hear the avatar respond once it's ready."
            ),
            room_url,
            _make_embed(room_url),
        )

    except requests.HTTPError as exc:
        return state, f"❌ Daily.co API error: {exc.response.text}", "", ""
    except Exception as exc:
        return state, f"❌ Unexpected error: {exc}", "", ""


def end_session(state: dict):
    """Terminate the bot subprocess and delete the Daily.co room."""
    room_url = state.get("room_url")
    if not room_url:
        return state, "No active session to end.", "", ""

    with _lock:
        proc = _active_bots.pop(room_url, None)

    if proc and proc.poll() is None:
        proc.terminate()
        logger.info(f"[Launcher] Bot terminated (PID {proc.pid})")

    if DAILY_API_KEY:
        delete_daily_room(DAILY_API_KEY, room_url)

    state.clear()
    return state, "Session ended. Click 'Start Session' to begin a new one.", "", ""


# ---------------------------------------------------------------------------
# HTML embed builder
# ---------------------------------------------------------------------------

def _make_embed(room_url: str) -> str:
    return f"""
<div style="
    width: 100%;
    height: 620px;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 6px 32px rgba(0,0,0,0.25);
    background: #111;
">
  <iframe
    src="{room_url}"
    style="width: 100%; height: 100%; border: none;"
    allow="camera; microphone; fullscreen; display-capture; autoplay"
  ></iframe>
</div>
<p style="text-align:center; margin-top:10px; color:#888; font-size:13px;">
  Video call not loading?
  <a href="{room_url}" target="_blank" style="color:#4a9eff;">Open in a new tab ↗</a>
</p>
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="SoulX Live Avatar",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
        #start-btn { font-size: 1.1rem; padding: 12px 28px; }
        #end-btn   { font-size: 1.1rem; padding: 12px 24px; }
    """,
) as demo:
    session_state = gr.State({})

    gr.Markdown(
        """
        # 🤖 SoulX Live Talking Avatar
        A photorealistic AI avatar that listens, thinks, and speaks back — powered by
        **SoulX-FlashHead** (real-time diffusion) + **Pipecat** + **Daily.co**.

        Click **▶ Start Session** to create your private room and launch the avatar.
        The first response takes ~40 seconds while the GPU warms up.
        """
    )

    with gr.Row():
        start_btn = gr.Button("▶ Start Session", variant="primary", elem_id="start-btn", scale=3)
        end_btn   = gr.Button("⏹ End Session",   variant="stop",    elem_id="end-btn",   scale=1)

    status_box  = gr.Textbox(
        label="Status",
        value="Press ▶ Start Session to begin.",
        interactive=False,
        lines=2,
    )
    room_url_box = gr.Textbox(
        label="Room URL  (open this link in any browser to join)",
        interactive=False,
        placeholder="Your room URL will appear here…",
    )
    room_embed = gr.HTML()

    gr.Markdown(
        """
        ---
        ### Tips
        - 🎙️ **Allow microphone and camera** when the browser asks — the avatar needs them.
        - 🔇 Use headphones to prevent echo.
        - ⏱️ First response takes ~40 s (GPU warmup). Subsequent turns are real-time.
        - 🔁 Click **End Session** then **Start Session** to get a fresh room.
        """
    )

    start_btn.click(
        fn=start_session,
        inputs=[session_state],
        outputs=[session_state, status_box, room_url_box, room_embed],
    )
    end_btn.click(
        fn=end_session,
        inputs=[session_state],
        outputs=[session_state, status_box, room_url_box, room_embed],
    )


if __name__ == "__main__":
    logger.info("Starting SoulX Gradio Room Launcher on port 7860…")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,         # Lightning.ai exposes port 7860 publicly — no need for share=True
        show_error=True,
    )
