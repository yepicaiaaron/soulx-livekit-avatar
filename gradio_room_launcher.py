"""SoulX Avatar — Gradio Room Launcher

Features:
  - Avatar picker (existing portraits + custom upload)
  - TTS voice selector (OpenAI voices)
  - LLM model selector
  - Avatar sync tuning (TURN_END_TIMEOUT)
  - Live session log viewer with key-event highlighting
  - Real-time avatar state / queue metrics
"""

import os
import sys
import re
import time
import shutil
import subprocess
import threading
import requests
import gradio as gr
from collections import deque
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DAILY_API_KEY  = os.environ.get("DAILY_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

_REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(_REPO_DIR, "examples")
UPLOADS_DIR  = os.path.join(EXAMPLES_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Active sessions: room_url -> {"proc": Popen, "log": deque, "all_log": deque}
_active_bots: dict = {}
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Config options
# ---------------------------------------------------------------------------
TTS_VOICES = {
    "Nova — female, warm":        "nova",
    "Shimmer — female, soft":     "shimmer",
    "Alloy — neutral, balanced":  "alloy",
    "Echo — male, clear":         "echo",
    "Onyx — male, deep":          "onyx",
    "Fable — male, storytelling": "fable",
}

LLM_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

# Log lines to always keep (regardless of filter)
_KEY_PATTERNS = re.compile(
    r"(AVATAR MODE|FLUSH|DISPATCH|SYNC|IDLE|PUSHER|GEN\]|USER\]|STT|TTS AVATAR"
    r"|BotStarted|BotStopped|Interrupted|warmup|ready|Joined|Participant|ERROR|error)",
    re.IGNORECASE,
)


def _scan_avatars() -> dict:
    """Return {display_name: absolute_path} for all portrait images."""
    result = {}
    for fname in sorted(os.listdir(EXAMPLES_DIR)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            result[fname] = os.path.join(EXAMPLES_DIR, fname)
    for fname in sorted(os.listdir(UPLOADS_DIR)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            result[f"uploads/{fname}"] = os.path.join(UPLOADS_DIR, fname)
    return result


# ---------------------------------------------------------------------------
# Daily.co helpers
# ---------------------------------------------------------------------------
def create_daily_room(api_key: str) -> str:
    resp = requests.post(
        "https://api.daily.co/v1/rooms",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"properties": {
            "exp": int(time.time()) + 7200,
            "enable_prejoin_ui": False,
            "enable_people_ui": False,
            "start_video_off": False,
            "start_audio_off": False,
        }},
        timeout=10,
    )
    resp.raise_for_status()
    url = resp.json()["url"]
    logger.info(f"[Launcher] Room created: {url}")
    return url


def delete_daily_room(api_key: str, room_url: str) -> None:
    try:
        name = room_url.rstrip("/").split("/")[-1]
        requests.delete(
            f"https://api.daily.co/v1/rooms/{name}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        logger.info(f"[Launcher] Room deleted: {name}")
    except Exception as exc:
        logger.warning(f"[Launcher] Could not delete room: {exc}")


# ---------------------------------------------------------------------------
# Bot lifecycle
# ---------------------------------------------------------------------------
def _stream_logs(proc, buf: deque, all_buf: deque):
    """Background thread — drain subprocess stdout into deques and log file."""
    log_file = getattr(proc, "_log_file", None)
    try:
        for line in proc.stdout:
            stripped = line.rstrip()
            all_buf.append(stripped)
            if _KEY_PATTERNS.search(stripped):
                buf.append(stripped)
            if log_file:
                try:
                    log_file.write(stripped + "\n")
                    log_file.flush()
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass


def _launch_bot(room_url: str, avatar_path: str, tts_voice: str, llm_model: str,
                model_type: str = "lite", turn_end_timeout: float = 0.7):
    env = os.environ.copy()
    env["DAILY_ROOM_URL"]      = room_url
    env["OPENAI_API_KEY"]      = OPENAI_API_KEY
    env["SOULX_MODEL_TYPE"]    = model_type
    env["SOULX_CKPT_DIR"]      = os.environ.get("SOULX_CKPT_DIR",    "./models/SoulX-FlashHead-1_3B")
    env["SOULX_WAV2VEC_DIR"]   = os.environ.get("SOULX_WAV2VEC_DIR", "./models/wav2vec2-base-960h")
    env["SOULX_COND_IMAGE"]    = avatar_path
    env["PERCEPTION_INTERVAL"] = os.environ.get("PERCEPTION_INTERVAL", "3.0")
    env["SOULX_TTS_VOICE"]     = tts_voice
    env["SOULX_LLM_MODEL"]     = llm_model
    env["TURN_END_TIMEOUT"]    = str(turn_end_timeout)

    log_path = "/tmp/soulx_bot.log"
    log_file = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        [sys.executable, "-u", "soulx_conversational_bot.py"],
        env=env,
        cwd=_REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    proc._log_file = log_file
    buf     = deque(maxlen=300)   # filtered key events
    all_buf = deque(maxlen=600)   # all lines
    threading.Thread(target=_stream_logs, args=(proc, buf, all_buf), daemon=True).start()
    logger.info(
        f"[Launcher] Bot PID {proc.pid} | avatar={os.path.basename(avatar_path)} "
        f"| voice={tts_voice} | llm={llm_model} | soulx={model_type} "
        f"| turn_end_timeout={turn_end_timeout}s"
    )
    return proc, buf, all_buf


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------
def _extract_metrics(all_buf: deque) -> dict:
    """Parse recent log lines to extract live avatar metrics."""
    metrics = {
        "avatar_state": "UNKNOWN",
        "last_flush": "—",
        "last_user": "—",
        "last_bot": "—",
        "sync_info": "—",
        "errors": [],
    }
    for line in reversed(list(all_buf)):
        if "AVATAR MODE" in line:
            if "Speaking" in line and metrics["avatar_state"] == "UNKNOWN":
                metrics["avatar_state"] = "🗣️ SPEAKING"
            elif "Listening" in line and metrics["avatar_state"] == "UNKNOWN":
                metrics["avatar_state"] = "👂 LISTENING"
        if metrics["last_flush"] == "—" and "[FLUSH]" in line:
            metrics["last_flush"] = line.split("|")[-1].strip() if "|" in line else line.strip()
        if metrics["last_user"] == "—" and "[STT USER]" in line:
            metrics["last_user"] = line.split("→")[-1].strip() if "→" in line else line.strip()
        if metrics["last_bot"] == "—" and "[TTS AVATAR]" in line:
            metrics["last_bot"] = line.split("←")[-1].strip() if "←" in line else line.strip()
        if metrics["sync_info"] == "—" and "[SYNC]" in line:
            metrics["sync_info"] = line.split("|")[-1].strip() if "|" in line else line.strip()
        if "ERROR" in line or "error" in line.lower():
            if len(metrics["errors"]) < 3:
                metrics["errors"].append(line.strip())
    return metrics


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------
def start_session(state, avatar_choice, tts_label, llm_model, model_type, turn_end_timeout):
    if not DAILY_API_KEY:
        return state, "❌ DAILY_API_KEY not loaded — check .env or Secrets panel.", "", "", "", ""
    if not OPENAI_API_KEY:
        return state, "❌ OPENAI_API_KEY not loaded — check .env or Secrets panel.", "", "", "", ""

    if state.get("room_url") and state.get("pid"):
        room_url = state["room_url"]
        return state, "✅ Session already running.", room_url, _make_embed(room_url), "", ""

    try:
        avatars     = _scan_avatars()
        avatar_path = avatars.get(avatar_choice, os.path.join(EXAMPLES_DIR, "girl.png"))
        tts_voice   = TTS_VOICES.get(tts_label, "nova")

        room_url = create_daily_room(DAILY_API_KEY)
        proc, buf, all_buf = _launch_bot(
            room_url, avatar_path, tts_voice, llm_model, model_type, float(turn_end_timeout)
        )

        with _lock:
            _active_bots[room_url] = {"proc": proc, "log": buf, "all_log": all_buf}

        state["room_url"] = room_url
        state["pid"]      = proc.pid

        init_log = (
            f"Session started — PID {proc.pid}\n"
            f"Avatar           : {os.path.basename(avatar_path)}\n"
            f"Voice            : {tts_label}  ({tts_voice})\n"
            f"LLM              : {llm_model}\n"
            f"SoulX            : {model_type}\n"
            f"Turn-end timeout : {turn_end_timeout}s\n"
            f"Room             : {room_url}\n"
            f"---\n"
            f"GPU warmup in progress (~40s)...\n"
        )
        return (
            state,
            "✅ Avatar booting — GPU warmup ~40s. Join the room below.",
            room_url,
            _make_embed(room_url),
            init_log,
            _render_metrics({}),
        )
    except requests.HTTPError as exc:
        return state, f"❌ Daily.co error: {exc.response.text}", "", "", "", ""
    except Exception as exc:
        return state, f"❌ Error: {exc}", "", "", "", ""


def end_session(state):
    room_url = state.get("room_url")
    if not room_url:
        return state, "No active session.", "", "", "", ""

    with _lock:
        info = _active_bots.pop(room_url, None)

    if info:
        proc = info["proc"]
        if proc and proc.poll() is None:
            proc.terminate()
            logger.info(f"[Launcher] Bot terminated PID {proc.pid}")

    if DAILY_API_KEY:
        delete_daily_room(DAILY_API_KEY, room_url)

    state.clear()
    return state, "Session ended. Configure and press ▶ Start Session.", "", "", "", ""


def poll_logs(state, show_all_logs: bool) -> tuple:
    """Return (filtered_log, metrics_html) for the timer tick."""
    room_url = state.get("room_url") if state else None
    if not room_url:
        return "No active session — logs will appear here after starting.", _render_metrics({})
    with _lock:
        info = _active_bots.get(room_url)
    if not info:
        return "Session not found in active bots.", _render_metrics({})

    all_buf = info.get("all_log", deque())
    key_buf = info.get("log", deque())
    metrics = _extract_metrics(all_buf)

    if show_all_logs:
        lines = list(all_buf)
    else:
        lines = list(key_buf)

    log_text = "\n".join(lines) if lines else "Waiting for bot output..."
    return log_text, _render_metrics(metrics)


def _render_metrics(m: dict) -> str:
    state   = m.get("avatar_state", "—")
    flush   = m.get("last_flush",   "—")
    user    = m.get("last_user",    "—")
    bot     = m.get("last_bot",     "—")
    sync    = m.get("sync_info",    "—")
    errors  = m.get("errors",       [])

    err_html = ""
    if errors:
        err_html = "<br>".join(f'<span style="color:#ff6b6b">⚠ {e}</span>' for e in errors)
        err_html = f"<div style='margin-top:6px'>{err_html}</div>"

    return f"""
<div style="background:#1a1a2e;border-radius:10px;padding:14px;font-family:monospace;font-size:13px;color:#e0e0e0;">
  <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:10px;">
    <div><b style="color:#4fc3f7">Avatar State</b><br>
      <span style="font-size:1.3em">{state}</span></div>
    <div><b style="color:#4fc3f7">Last User</b><br>{user[:80]}</div>
    <div><b style="color:#4fc3f7">Last Bot</b><br>{bot[:80]}</div>
  </div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;">
    <div><b style="color:#81c784">Last Flush</b><br>{flush[:100]}</div>
    <div><b style="color:#81c784">Queue Sync</b><br>{sync[:100]}</div>
  </div>
  {err_html}
</div>
"""


def on_avatar_upload(file, current_choices):
    if file is None:
        return gr.update(), gr.update()
    fname = os.path.basename(file.name)
    dest  = os.path.join(UPLOADS_DIR, fname)
    shutil.copy(file.name, dest)
    avatars  = _scan_avatars()
    choices  = list(avatars.keys())
    key      = f"uploads/{fname}"
    selected = key if key in choices else choices[0]
    return gr.update(choices=choices, value=selected), gr.update(value=avatars.get(selected))


def on_avatar_change(choice):
    avatars = _scan_avatars()
    return gr.update(value=avatars.get(choice))


def _make_embed(room_url: str) -> str:
    return f"""
<div style="width:100%;height:620px;border-radius:14px;overflow:hidden;
            box-shadow:0 6px 32px rgba(0,0,0,.25);background:#111;">
  <iframe src="{room_url}" style="width:100%;height:100%;border:none;"
    allow="camera;microphone;fullscreen;display-capture;autoplay"></iframe>
</div>
<p style="text-align:center;margin-top:10px;color:#888;font-size:13px;">
  Not loading?
  <a href="{room_url}" target="_blank" style="color:#4a9eff;">Open in new tab ↗</a>
</p>
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_init_avatars = _scan_avatars()
_avatar_keys  = list(_init_avatars.keys())
_default_av   = "girl.png" if "girl.png" in _init_avatars else (_avatar_keys[0] if _avatar_keys else "")

with gr.Blocks(
    title="SoulX Live Avatar",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css="""
        #start-btn { font-size:1.1rem; padding:12px 28px; }
        #end-btn   { font-size:1.1rem; padding:12px 24px; }
        #log-box textarea { font-family: monospace !important; font-size: 12px !important; }
        #metrics-panel { margin-top:6px; }
    """,
) as demo:
    session_state = gr.State({})

    gr.Markdown(
        """# 🤖 SoulX Live Talking Avatar
Photorealistic AI avatar — **SoulX-FlashHead** diffusion + **Pipecat** + **Daily.co** + **OpenAI**.""")

    # ── Configuration panel ───────────────────────────────────────────────
    with gr.Accordion("⚙️ Session Configuration", open=True):
        with gr.Row():

            # Avatar column
            with gr.Column(scale=1):
                gr.Markdown("### 🎭 Avatar")
                avatar_radio = gr.Radio(
                    choices=_avatar_keys,
                    value=_default_av,
                    label="Select portrait",
                )
                avatar_preview = gr.Image(
                    value=_init_avatars.get(_default_av),
                    label="Preview",
                    height=170,
                    interactive=False,
                )
                avatar_upload = gr.File(
                    label="Upload custom portrait (.png / .jpg)",
                    file_types=[".png", ".jpg", ".jpeg"],
                )

            # Voice column
            with gr.Column(scale=1):
                gr.Markdown("### 🔊 TTS Voice")
                tts_radio = gr.Radio(
                    choices=list(TTS_VOICES.keys()),
                    value="Nova — female, warm",
                    label="OpenAI TTS voice",
                )
                gr.Markdown("""
| Voice | Gender | Character |
|-------|--------|-----------|
| Nova | Female | Warm, natural |
| Shimmer | Female | Soft, expressive |
| Alloy | Neutral | Balanced |
| Echo | Male | Clear, professional |
| Onyx | Male | Deep, authoritative |
| Fable | Male | Rich, storytelling |
""")

            # LLM + STT column
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 LLM Model")
                llm_dropdown = gr.Dropdown(
                    choices=LLM_MODELS,
                    value="gpt-4o",
                    label="OpenAI LLM",
                )
                gr.Markdown("### 🎥 SoulX Model")
                model_type_radio = gr.Radio(
                    choices=["lite", "pro"],
                    value="lite",
                    label="SoulX-FlashHead model (lite = faster, pro = higher quality)",
                )
                gr.Markdown("""
### 🎙️ STT Config *(fixed)*
| Setting | Value |
|---------|-------|
| Model | whisper-1 |
| VAD | Silero (stop: 1.5 s) |
| Interruptions | enabled |
""")

        # ── Avatar Sync Tuning ────────────────────────────────────────────
        with gr.Accordion("🎛️ Avatar Sync Tuning", open=True):
            gr.Markdown("""
**Turn-end timeout** — how long the avatar waits with no new TTS audio before
flushing the partial audio buffer (last-word rescue) and returning to idle.

- **Lower** (0.3–0.5 s): snappier idle return, but may trigger between sentences on slow LLMs
- **Higher** (0.8–1.5 s): more robust across inter-sentence gaps, slightly slower idle return
- Typical inter-sentence gap: **200–400 ms**
""")
            turn_end_slider = gr.Slider(
                minimum=0.3, maximum=2.0, step=0.1, value=0.7,
                label="Turn-end timeout (seconds)",
                info="Recommended: 0.7s. Increase if avatar idles between sentences.",
            )

    # ── Controls ───────────────────────────────────────────────────────────
    with gr.Row():
        start_btn = gr.Button("▶ Start Session", variant="primary", elem_id="start-btn", scale=3)
        end_btn   = gr.Button("⏹ End Session",   variant="stop",    elem_id="end-btn",   scale=1)

    status_box   = gr.Textbox(label="Status", value="Configure above then press ▶ Start Session.",
                               interactive=False, lines=2)
    room_url_box = gr.Textbox(label="Room URL", interactive=False,
                               placeholder="Room URL will appear here…")
    room_embed   = gr.HTML()

    # ── Live metrics ───────────────────────────────────────────────────────
    gr.Markdown("### 📊 Live Avatar Metrics")
    metrics_panel = gr.HTML(
        value=_render_metrics({}),
        elem_id="metrics-panel",
    )

    # ── Live log ───────────────────────────────────────────────────────────
    with gr.Accordion("📋 Live Session Log", open=True):
        with gr.Row():
            show_all_toggle = gr.Checkbox(
                label="Show full log (unfiltered)",
                value=False,
                info="Unchecked = key events only (avatar state, speech, sync, errors)",
            )
        log_box = gr.Textbox(
            label="Bot output (auto-refreshes every 3 s)",
            lines=22,
            max_lines=35,
            interactive=False,
            elem_id="log-box",
            placeholder="Logs appear here after starting a session…",
        )
        log_timer = gr.Timer(value=3)

    gr.Markdown("""---
### Tips
- 🎙️ Allow **microphone** and **camera** when the browser prompts.
- 🔇 Wear headphones to prevent echo.
- ⏱️ First response takes ~40 s (GPU warmup). Turns after that are real-time.
- 🔁 End then Start Session to get a fresh room with new settings.
- 🎛️ If avatar idles between sentences, increase Turn-end timeout and restart.""")

    # ── Event wiring ───────────────────────────────────────────────────────
    avatar_radio.change(fn=on_avatar_change, inputs=[avatar_radio], outputs=[avatar_preview])
    avatar_upload.change(fn=on_avatar_upload, inputs=[avatar_upload, avatar_radio],
                         outputs=[avatar_radio, avatar_preview])

    start_btn.click(
        fn=start_session,
        inputs=[session_state, avatar_radio, tts_radio, llm_dropdown, model_type_radio, turn_end_slider],
        outputs=[session_state, status_box, room_url_box, room_embed, log_box, metrics_panel],
    )
    end_btn.click(
        fn=end_session,
        inputs=[session_state],
        outputs=[session_state, status_box, room_url_box, room_embed, log_box, metrics_panel],
    )
    log_timer.tick(
        fn=poll_logs,
        inputs=[session_state, show_all_toggle],
        outputs=[log_box, metrics_panel],
    )


if __name__ == "__main__":
    logger.info("Starting SoulX Gradio Room Launcher on port 7860…")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
