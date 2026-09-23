"""Text-to-speech wrapper.

Primary provider: Sarvam AI (bulbul) — natural Indian-accent Hindi/Tamil/English/code-mixed voices.
Fallback provider: OpenAI TTS.

Provider selection (env VAAK_TTS_PROVIDER):
  auto   (default) try Sarvam if SARVAM_API_KEY is set, fall back to OpenAI on any error
  sarvam only Sarvam
  openai only OpenAI TTS — use this if you don't have Sarvam access yet

Output is always a WAV file.

This module is domain-agnostic: it must never import from tools/.

CLI:
  python -m voice_io.tts "नमस्ते, मैं Vaak हूँ" -o out/hello.wav [--lang hi-IN] [--provider openai]
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TTS_LANGS = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
}
DEFAULT_LANG = "hi-IN"  # hi-IN voices read code-mixed Hindi/English naturally
MAX_CHUNK_CHARS = 1500  # under both bulbul:v2 (1500) and bulbul:v3 (2500) limits
REQUEST_TIMEOUT_S = 60


class TTSError(RuntimeError):
    pass


@dataclass
class SpeechResult:
    path: Path
    provider: str
    latency_ms: float


def synthesize(
    text: str,
    out_path: str | Path,
    language_code: str = DEFAULT_LANG,
    provider: str | None = None,
) -> SpeechResult:
    """Synthesize text to a WAV file at out_path."""
    text = text.strip()
    if not text:
        raise ValueError("text is empty")
    if language_code not in SARVAM_TTS_LANGS:
        language_code = DEFAULT_LANG  # e.g. "unknown" from STT auto-detect

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provider = (provider or os.getenv("VAAK_TTS_PROVIDER", "auto")).lower()

    if provider == "sarvam":
        return _synthesize_sarvam(text, out_path, language_code)
    if provider == "openai":
        return _synthesize_openai(text, out_path)
    if provider != "auto":
        raise ValueError(f"Unknown TTS provider: {provider!r}")

    errors = []
    if os.getenv("SARVAM_API_KEY"):
        try:
            return _synthesize_sarvam(text, out_path, language_code)
        except Exception as e:  # fall through to OpenAI
            errors.append(f"sarvam: {e}")
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _synthesize_openai(text, out_path)
        except Exception as e:
            errors.append(f"openai: {e}")
    if not errors:
        raise TTSError("No TTS provider configured: set SARVAM_API_KEY and/or OPENAI_API_KEY in .env (repo root)")
    raise TTSError("All TTS providers failed -> " + " | ".join(errors))


def _synthesize_sarvam(text: str, out_path: Path, language_code: str) -> SpeechResult:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise TTSError("SARVAM_API_KEY is not set")

    payload = {
        "model": os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
        "language_code": language_code,
    }
    speaker = os.getenv("SARVAM_TTS_SPEAKER")  # must match the model, e.g. "priya" for v3
    if speaker:
        payload["speaker"] = speaker

    start = time.perf_counter()
    wav_chunks = []
    for chunk in _split_text(text, MAX_CHUNK_CHARS):
        resp = requests.post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": api_key},
            json={**payload, "text": chunk},
            timeout=REQUEST_TIMEOUT_S,
        )
        if resp.status_code != 200:
            raise TTSError(f"Sarvam TTS HTTP {resp.status_code}: {resp.text[:300]}")
        wav_chunks.extend(base64.b64decode(a) for a in resp.json()["audios"])
    latency_ms = (time.perf_counter() - start) * 1000

    _write_wav(wav_chunks, out_path)
    return SpeechResult(path=out_path, provider="sarvam", latency_ms=latency_ms)


def _synthesize_openai(text: str, out_path: Path) -> SpeechResult:
    from openai import OpenAI  # imported lazily so Sarvam-only setups don't need it

    if not os.getenv("OPENAI_API_KEY"):
        raise TTSError("OPENAI_API_KEY is not set")

    client = OpenAI()
    start = time.perf_counter()
    wav_chunks = []
    for chunk in _split_text(text, MAX_CHUNK_CHARS):
        resp = client.audio.speech.create(
            model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
            voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
            input=chunk,
            response_format="wav",
        )
        wav_chunks.append(resp.content)
    latency_ms = (time.perf_counter() - start) * 1000

    _write_wav(wav_chunks, out_path)
    return SpeechResult(path=out_path, provider="openai", latency_ms=latency_ms)


_SPEECH_REPLACEMENTS = [
    (r"\\\(|\\\)|\\\[|\\\]|\$", ""),              # LaTeX delimiters
    (r"\*\*|__|`|^#+\s*", ""),                    # markdown emphasis, code, headings
    (r"\\times|\\cdot", " times "),                  # LaTeX operators
    (r"\(1/2\)|\b1/2\b", "half"),
    (r"\^\s*2\b|²", " squared"),
    (r"\^\s*3\b|³", " cubed"),
    (r"\^\s*\(?([\w.+-]+)\)?", r" to the power \1"),
    (r"\s*[*×]\s*", " times "),                   # "m * a", "I × R"
    (r"\s*÷\s*", " divided by "),
    (r"\s*±\s*", " plus or minus "),
    (r"\bsqrt\s*\(", "square root of ("),
    (r"\s*=\s*", " equals "),
    (r"\s{2,}", " "),
]


def prepare_for_speech(text: str) -> str:
    """Rewrite symbols a TTS engine reads badly ("v^2", "m * a", LaTeX) into words."""
    for pattern, repl in _SPEECH_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)
    return text.strip()


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split on sentence boundaries (incl. Hindi danda) so each chunk fits the API limit."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        while len(s) > max_chars:  # a single oversized sentence: hard-split it
            chunks.append(s[:max_chars])
            s = s[max_chars:]
        if current and len(current) + 1 + len(s) > max_chars:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


def _write_wav(wav_chunks: list[bytes], out_path: Path) -> None:
    """Write one WAV, concatenating PCM frames if the text was chunked."""
    if len(wav_chunks) == 1:
        out_path.write_bytes(wav_chunks[0])
        return
    with wave.open(str(out_path), "wb") as out:
        for i, chunk in enumerate(wav_chunks):
            with wave.open(io.BytesIO(chunk), "rb") as w:
                if i == 0:
                    out.setparams(w.getparams())
                out.writeframes(w.readframes(w.getnframes()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize text to a WAV file.")
    parser.add_argument("text")
    parser.add_argument("-o", "--out", default="out/tts_output.wav")
    parser.add_argument("--lang", default=DEFAULT_LANG, help="BCP-47 code, e.g. hi-IN, en-IN")
    parser.add_argument("--provider", choices=["auto", "sarvam", "openai"], default=None)
    args = parser.parse_args()

    r = synthesize(args.text, args.out, language_code=args.lang, provider=args.provider)
    print(f"provider : {r.provider}")
    print(f"latency  : {r.latency_ms:.0f} ms")
    print(f"wrote    : {r.path}")


if __name__ == "__main__":
    main()
