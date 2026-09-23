"""Vaak web app: a thin HTTP layer over VoicePipeline plus a single static page.

  python -m web.server            # http://localhost:8765
  python -m web.server --port 9000

Endpoints:
  GET  /                    the app
  GET  /api/info            model and tool names for the header
  POST /api/ask/text        {"text": "...", "speak": true}
  POST /api/ask/audio       multipart: file=<audio>, speak=true|false
  GET  /api/audio/<name>    a synthesized reply (WAV)
"""
from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.agent_runtime import Agent, detect_language, load_tools
from core.pipeline import PipelineError, TurnResult, VoicePipeline
from core.tool_registry import ToolRegistry

TOOLS_MODULE = "tools.study_tools"
STATIC_DIR = Path(__file__).parent / "static"
OUT_DIR = Path("out")
UPLOAD_DIR = OUT_DIR / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
REPLY_NAME = re.compile(r"^reply_[0-9a-f]{8}\.wav$")
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".webm", ".flac", ".aiff"}

_LANGUAGE_LABELS = {
    ("ta-IN", "native"): "Tamil",
    ("ta-IN", "latin"): "Tanglish",
    ("hi-IN", "native"): "Hindi",
    ("hi-IN", "latin"): "Hinglish",
}

registry = ToolRegistry()
agent = Agent(registry, system_prompt=load_tools(TOOLS_MODULE, registry))
app = FastAPI(title="Vaak")


class TextQuery(BaseModel):
    text: str
    speak: bool = True


def language_label(lang: str, text: str) -> str:
    _, script = detect_language(text)
    return _LANGUAGE_LABELS.get((lang, script), "English")


def serialize(turn: TurnResult) -> dict:
    a = turn.agent
    return {
        "input_mode": turn.input_mode,
        "query": turn.query,
        "transcript": (
            {"text": turn.transcript.text, "language": turn.transcript.language_code,
             "provider": turn.transcript.provider}
            if turn.transcript else None
        ),
        "answer": a.answer,
        "route": a.route,
        "language": a.reply_language,
        "language_label": language_label(a.reply_language, turn.query),
        "tool_calls": [{"name": c.name, "arguments": c.arguments, "result": c.result} for c in a.tool_calls],
        "model": a.model,
        "timings_ms": {k: round(v) for k, v in turn.timings_ms.items()},
        "audio_url": f"/api/audio/{turn.speech.path.name}" if turn.speech else None,
        "errors": turn.errors,
    }


@app.get("/api/info")
def info() -> dict:
    return {"model": agent.model, "tools": registry.names()}


@app.post("/api/ask/text")
def ask_text(q: TextQuery) -> dict:
    text = q.text.strip()
    if not text:
        raise HTTPException(400, "Question is empty")
    return serialize(VoicePipeline(agent, speak=q.speak, out_dir=OUT_DIR).run_text(text))


@app.post("/api/ask/audio")
def ask_audio(file: UploadFile = File(...), speak: bool = Form(True)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(400, f"Unsupported audio type: {suffix or 'unknown'}")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file is larger than 10 MB")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    try:
        turn = VoicePipeline(agent, speak=speak, out_dir=OUT_DIR).run_audio(path)
    except PipelineError as e:
        # 422 with stage=stt tells the page to fall back to typing.
        return JSONResponse({"error": str(e), "stage": "stt"}, status_code=422)
    finally:
        path.unlink(missing_ok=True)
    return serialize(turn)


@app.get("/api/audio/{name}")
def audio(name: str) -> FileResponse:
    path = OUT_DIR / name
    if not REPLY_NAME.match(name) or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Vaak web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    shown = "localhost" if args.host == "127.0.0.1" else args.host
    print(f"Vaak running at http://{shown}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
