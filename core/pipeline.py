"""Voice pipeline: STT -> agent -> TTS, with a text-only fallback path.

    pipeline = VoicePipeline(agent)
    turn = pipeline.run_audio("question.m4a")   # spoken query
    turn = pipeline.run_text("What is 37*48?")  # typed query (the fallback path)

Failures degrade instead of crashing:
  - STT fails  -> PipelineError; callers fall back to asking for typed text
  - TTS fails  -> the turn still returns the text answer, with the error recorded

This module is domain-agnostic: it must never import from tools/.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from core.agent_runtime import Agent, AgentResult, detect_language
from voice_io.stt import Transcript, transcribe
from voice_io.tts import SARVAM_TTS_LANGS, SpeechResult, prepare_for_speech, synthesize


class PipelineError(RuntimeError):
    pass


@dataclass
class TurnResult:
    input_mode: str  # "audio" | "text"
    query: str
    agent: AgentResult
    transcript: Transcript | None = None
    speech: SpeechResult | None = None
    errors: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return self.agent.answer


class VoicePipeline:
    def __init__(self, agent: Agent, speak: bool = True, out_dir: str | Path = "out") -> None:
        self.agent = agent
        self.speak = speak
        self.out_dir = Path(out_dir)

    def run_audio(self, audio_path: str | Path) -> TurnResult:
        start = time.perf_counter()
        try:
            transcript = transcribe(audio_path)
        except Exception as e:
            raise PipelineError(f"Speech-to-text failed: {e}") from e
        if not transcript.text:
            raise PipelineError("Speech-to-text returned an empty transcript (silent audio?)")

        turn = self._answer(transcript.text, transcript.language_code, input_mode="audio")
        turn.transcript = transcript
        turn.timings_ms = {"stt": transcript.latency_ms, **turn.timings_ms}
        turn.timings_ms["total"] = (time.perf_counter() - start) * 1000
        return turn

    def run_text(self, text: str, language_code: str | None = None) -> TurnResult:
        start = time.perf_counter()
        turn = self._answer(text.strip(), language_code, input_mode="text")
        turn.timings_ms["total"] = (time.perf_counter() - start) * 1000
        return turn

    def _answer(self, query: str, language_code: str | None, input_mode: str) -> TurnResult:
        agent_result = self.agent.run(query, language_code=language_code)
        turn = TurnResult(input_mode=input_mode, query=query, agent=agent_result)
        turn.timings_ms["agent"] = agent_result.latency_ms

        if self.speak and agent_result.answer:
            out_path = self.out_dir / f"reply_{uuid.uuid4().hex[:8]}.wav"
            try:
                turn.speech = synthesize(
                    prepare_for_speech(agent_result.answer),
                    out_path,
                    language_code=_tts_language(agent_result),
                )
                turn.timings_ms["tts"] = turn.speech.latency_ms
            except Exception as e:  # text answer still stands
                turn.errors.append(f"Text-to-speech failed: {e}")
        return turn


def _tts_language(result: AgentResult) -> str:
    """Voice for the reply: the agent's reply language, else whatever the answer is written in."""
    if result.reply_language in SARVAM_TTS_LANGS:
        return result.reply_language
    detected, _ = detect_language(result.answer)
    return detected if detected in SARVAM_TTS_LANGS else "en-IN"
