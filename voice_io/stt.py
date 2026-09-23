"""Speech-to-text wrapper.

Primary provider: Sarvam AI (saaras) — handles Hindi, Tamil, English and code-mixed speech.
Fallback provider: OpenAI Whisper.

Provider selection (env VAAK_STT_PROVIDER):
  auto   (default) try Sarvam if SARVAM_API_KEY is set, fall back to OpenAI on any error
  sarvam only Sarvam
  openai only OpenAI Whisper — use this if you don't have Sarvam access yet

This module is domain-agnostic: it must never import from tools/.

CLI:
  python -m voice_io.stt path/to/audio.wav [--lang hi-IN] [--provider openai]
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
REQUEST_TIMEOUT_S = 60
AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/x-m4a",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


class STTError(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    language_code: str  # BCP-47, e.g. "hi-IN"; "unknown" if the provider didn't say
    provider: str
    latency_ms: float


def transcribe(
    audio_path: str | Path,
    language_code: str = "unknown",
    provider: str | None = None,
) -> Transcript:
    """Transcribe an audio file.

    language_code: BCP-47 code ("hi-IN", "en-IN", ...) or "unknown" to auto-detect.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    provider = (provider or os.getenv("VAAK_STT_PROVIDER", "auto")).lower()

    if provider == "sarvam":
        return _transcribe_sarvam(audio_path, language_code)
    if provider == "openai":
        return _transcribe_openai(audio_path, language_code)
    if provider != "auto":
        raise ValueError(f"Unknown STT provider: {provider!r}")

    errors = []
    if os.getenv("SARVAM_API_KEY"):
        try:
            return _transcribe_sarvam(audio_path, language_code)
        except Exception as e:  # fall through to OpenAI
            errors.append(f"sarvam: {e}")
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _transcribe_openai(audio_path, language_code)
        except Exception as e:
            errors.append(f"openai: {e}")
    if not errors:
        raise STTError("No STT provider configured: set SARVAM_API_KEY and/or OPENAI_API_KEY in .env (repo root)")
    raise STTError("All STT providers failed -> " + " | ".join(errors))


def _transcribe_sarvam(audio_path: Path, language_code: str) -> Transcript:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise STTError("SARVAM_API_KEY is not set")

    data = {
        "model": os.getenv("SARVAM_STT_MODEL", "saaras:v3"),
        # "transcribe" keeps each language in its native script;
        # "codemix" writes English words in Latin script inside Hindi/Tamil sentences.
        "mode": os.getenv("SARVAM_STT_MODE", "transcribe"),
        "language_code": language_code,
    }
    start = time.perf_counter()
    with audio_path.open("rb") as f:
        resp = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": api_key},
            files={"file": (audio_path.name, f, _mime_type(audio_path))},
            data=data,
            timeout=REQUEST_TIMEOUT_S,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        raise STTError(f"Sarvam STT HTTP {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    return Transcript(
        text=(body.get("transcript") or "").strip(),
        language_code=body.get("language_code") or language_code,
        provider="sarvam",
        latency_ms=latency_ms,
    )


def _mime_type(audio_path: Path) -> str:
    """Sarvam rejects uploads without a Content-Type, so always send one."""
    return AUDIO_MIME_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")


def _transcribe_openai(audio_path: Path, language_code: str) -> Transcript:
    from openai import OpenAI  # imported lazily so Sarvam-only setups don't need it

    if not os.getenv("OPENAI_API_KEY"):
        raise STTError("OPENAI_API_KEY is not set")

    kwargs = {"model": os.getenv("OPENAI_STT_MODEL", "whisper-1")}
    if language_code and language_code != "unknown":
        kwargs["language"] = language_code.split("-")[0]  # Whisper wants ISO-639-1: "hi-IN" -> "hi"

    start = time.perf_counter()
    with audio_path.open("rb") as f:
        result = OpenAI().audio.transcriptions.create(file=f, **kwargs)
    latency_ms = (time.perf_counter() - start) * 1000

    return Transcript(
        text=result.text.strip(),
        language_code=language_code,
        provider="openai",
        latency_ms=latency_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("audio", help="path to audio file (wav, mp3, m4a, ...)")
    parser.add_argument("--lang", default="unknown", help='BCP-47 code, e.g. hi-IN, or "unknown"')
    parser.add_argument("--provider", choices=["auto", "sarvam", "openai"], default=None)
    args = parser.parse_args()

    t = transcribe(args.audio, language_code=args.lang, provider=args.provider)
    print(f"provider : {t.provider}")
    print(f"language : {t.language_code}")
    print(f"latency  : {t.latency_ms:.0f} ms")
    print(f"transcript: {t.text}")


if __name__ == "__main__":
    main()
