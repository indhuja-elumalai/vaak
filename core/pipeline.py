"""Voice pipeline: STT -> agent -> TTS, with a text-only fallback path.

    pipeline = VoicePipeline(agent, conversation=Conversation())
    turn = pipeline.run_audio("question.m4a")   # spoken query
    turn = pipeline.run_text("and in grams?")   # typed follow-up (also the fallback path)

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

from core.agent_runtime import Agent, AgentResult, TraceStep, detect_language
from core.conversation import Conversation
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
    trace: list[TraceStep] = field(default_factory=list)  # stt + agent steps + tts, on one clock

    @property
    def answer(self) -> str:
        return self.agent.answer


class VoicePipeline:
    def __init__(
        self,
        agent: Agent,
        speak: bool = True,
        out_dir: str | Path = "out",
        conversation: Conversation | None = None,
    ) -> None:
        self.agent = agent
        self.speak = speak
        self.out_dir = Path(out_dir)
        self.conversation = conversation  # None = every question stands alone

    def run_audio(self, audio_path: str | Path) -> TurnResult:
        start = time.perf_counter()
        try:
            transcript = transcribe(audio_path)
        except Exception as e:
            raise PipelineError(f"Speech-to-text failed: {e}") from e
        if not transcript.text:
            raise PipelineError("Speech-to-text returned an empty transcript (silent audio?)")

        stt_ms = transcript.latency_ms
        turn = self._answer(transcript.text, transcript.language_code, input_mode="audio", offset_ms=stt_ms)
        turn.transcript = transcript
        turn.timings_ms = {"stt": stt_ms, **turn.timings_ms}
        turn.trace.insert(0, TraceStep(
            "stt", "Heard you", f"{transcript.provider} · {transcript.language_code}", 0, stt_ms,
        ))
        turn.timings_ms["total"] = (time.perf_counter() - start) * 1000
        return turn

    def run_text(self, text: str, language_code: str | None = None) -> TurnResult:
        start = time.perf_counter()
        turn = self._answer(text.strip(), language_code, input_mode="text")
        turn.timings_ms["total"] = (time.perf_counter() - start) * 1000
        return turn

    def _answer(
        self, query: str, language_code: str | None, input_mode: str, offset_ms: float = 0
    ) -> TurnResult:
        agent_result = self.agent.run(query, language_code=language_code, conversation=self.conversation)
        turn = TurnResult(input_mode=input_mode, query=query, agent=agent_result)
        turn.timings_ms["agent"] = agent_result.latency_ms
        turn.trace = [
            TraceStep(s.kind, s.label, s.detail, s.start_ms + offset_ms, s.duration_ms, s.error)
            for s in agent_result.trace
        ]

        if self.speak and agent_result.answer:
            out_path = self.out_dir / f"reply_{uuid.uuid4().hex[:8]}.wav"
            try:
                turn.speech = synthesize(
                    prepare_for_speech(agent_result.answer),
                    out_path,
                    language_code=_tts_language(agent_result),
                )
                turn.timings_ms["tts"] = turn.speech.latency_ms
                turn.trace.append(TraceStep(
                    "tts", "Spoke the answer", f"{turn.speech.provider} · {_tts_language(agent_result)}",
                    offset_ms + agent_result.latency_ms, turn.speech.latency_ms,
                ))
            except Exception as e:  # text answer still stands
                turn.errors.append(f"Text-to-speech failed: {e}")
                turn.trace.append(TraceStep(
                    "tts", "Couldn't speak the answer", str(e)[:110], offset_ms + agent_result.latency_ms, 0,
                    error=True,
                ))
        return turn


def _tts_language(result: AgentResult) -> str:
    """Voice for the reply: the agent's reply language, else whatever the answer is written in."""
    if result.reply_language in SARVAM_TTS_LANGS:
        return result.reply_language
    detected, _ = detect_language(result.answer)
    return detected if detected in SARVAM_TTS_LANGS else "en-IN"
