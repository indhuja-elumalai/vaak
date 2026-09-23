"""Agent runtime: input -> intent -> tool-or-direct -> output.

The LLM sees the registry's tool schemas and decides per query whether to call a tool
or answer directly. Tool results are fed back until the LLM produces a final answer.

LLM provider (env VAAK_LLM_PROVIDER):
  sarvam (default) Sarvam chat completions, OpenAI-compatible, uses SARVAM_API_KEY
  openai           OpenAI chat completions, uses OPENAI_API_KEY

This module is domain-agnostic: it must never import from tools/. Domain tools are
plugged in at runtime, either by the caller (Agent(registry=...)) or by module name
on the CLI:

  python -m core.agent_runtime --tools tools.study_tools "What is 37 times 48?"
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from core.conversation import Conversation
from core.grounding import GroundingResult, check_answer, correction_prompt, known_from_messages
from core.tool_registry import ToolRegistry

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
# Per LLM request; a hung request fails fast (after one retry) instead of stalling the voice loop.
LLM_TIMEOUT_S = float(os.getenv("VAAK_LLM_TIMEOUT_S", "30"))

BASE_SYSTEM_PROMPT = """\
You are a voice assistant. Your replies are read aloud by a text-to-speech engine.

Deciding how to answer:
- If one of your tools can do the task (a calculation, a unit conversion, a lookup), \
you MUST call the tool. Never do arithmetic, conversions or lookups in your head, \
even easy ones.
- If no tool fits (explanations, facts, definitions, opinions), answer directly \
without calling any tool.
- Follow-up questions ("and in grams?", "now calculate it for 2 kg") refer to the earlier \
conversation: work out the full request from context and call the tool again with the \
updated values. Never reuse an old tool result for a new number.
- Pass tool arguments in English and plain ASCII (e.g. "sqrt(144)", "km", "kinetic energy"), \
whatever language the user spoke.

Replying:
- Reply in the language given in the "Reply language" line below. Never switch to another language.
- Keep it short: one to three spoken sentences.
- Plain text only: no markdown, no LaTeX, no bullet points, no emojis.
- Write numbers as digits (55.12, not "fifty-five point one two"), and quote tool results \
exactly or rounded. Your numbers are checked against the tool results.
"""


# Romanized marker words, used only when the text is in Latin script.
_LATIN_MARKERS = {
    "ta-IN": {
        "enna", "yenna", "epdi", "eppadi", "evlo", "evvalavu", "pannu", "pannunga", "panna",
        "sollu", "sollunga", "solu", "ah", "la", "ku", "illa", "iruku", "irukku", "na",
        "aana", "kudu", "maathu", "maatru", "enga", "yaaru", "edhu", "ethu", "venum",
        "vendum", "theriyuma", "da", "di", "romba", "konjam", "ethana", "ethanai", "evvalavu",
        "tamil", "tamizh", "tamizhil", "irukiradhu", "irukkiradhu", "enakku", "unakku",
        "eppo", "yen", "illai", "sollungal", "vilakku", "kanakku", "podu", "paaru",
    },
    "hi-IN": {
        "kya", "hai", "hain", "ka", "ki", "ke", "ko", "mein", "karo", "kaise", "kitna",
        "kitne", "kitni", "batao", "samjhao", "hota", "hoti", "hote", "hoga", "kaun", "kyun",
        "aur", "nahi", "bhi", "wale", "wala", "wali", "kijiye", "bataiye", "matlab",
    },
}
# Spelling shapes common in romanized Tamil and rare in English (tamiZH, irukkiraDHU, ezhuthuKKAL).
_TAMIL_WORD_SHAPE = re.compile(r"zh|(dhu|kkal|ngal|ukku|ikku|aanga|unga)$")

_REPLY_INSTRUCTIONS = {
    ("en-IN", "latin"): "English.",
    ("unknown", "latin"): "the same language the user wrote in, using English (Latin) letters. "
    "If the user's words are English, reply in English. If they are romanized Hindi, reply in "
    "Hinglish. If they are romanized Tamil, reply in Tanglish (never Malayalam or Hindi).",
    ("hi-IN", "native"): "Hindi, written in Devanagari script.",
    ("hi-IN", "latin"): "Hinglish: Hindi written in English (Latin) letters, keeping English "
    "technical words as the user does. No Devanagari.",
    ("ta-IN", "native"): "Tamil, written in Tamil script.",
    ("ta-IN", "latin"): "Tanglish: Tamil written in English (Latin) letters, keeping English "
    "technical words as the user does. No Tamil script, no Hindi, no Malayalam.",
}


def detect_language(text: str) -> tuple[str, str]:
    """Return (BCP-47 code, "native" | "latin") for Hindi, Tamil or English text.

    Latin text with no Hindi/Tamil signal returns "unknown": it is probably English, but
    the LLM decides rather than being forced to English.
    """
    tamil = sum("஀" <= ch <= "௿" for ch in text)
    devanagari = sum("ऀ" <= ch <= "ॿ" for ch in text)
    if tamil or devanagari:
        return ("ta-IN" if tamil >= devanagari else "hi-IN"), "native"

    words = re.findall(r"[a-z]+", text.lower())
    scores = {lang: sum(w in markers for w in words) for lang, markers in _LATIN_MARKERS.items()}
    scores["ta-IN"] += sum(bool(_TAMIL_WORD_SHAPE.search(w)) for w in words)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best, "latin"
    return "unknown", "latin"


def reply_instruction(text: str, language_code: str | None = None) -> tuple[str, str]:
    """Pick the reply language; an STT-detected language_code overrides text detection."""
    detected, script = detect_language(text)
    lang = language_code if (language_code, script) in _REPLY_INSTRUCTIONS else detected
    return lang, _REPLY_INSTRUCTIONS[(lang, script)]


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: dict


@dataclass
class TraceStep:
    """One step of a turn, for the timeline: start and duration are ms from turn start."""
    kind: str  # stt | llm | tool | check | tts
    label: str
    detail: str
    start_ms: float
    duration_ms: float
    error: bool = False


@dataclass
class AgentResult:
    query: str
    answer: str
    reply_language: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    llm_calls: int = 0
    model: str = ""
    grounding: GroundingResult | None = None
    draft_answer: str | None = None  # the first answer, if grounding forced a rewrite
    trace: list[TraceStep] = field(default_factory=list)

    @property
    def route(self) -> str:
        return "tool" if self.tool_calls else "direct"

    @property
    def tools_used(self) -> list[str]:
        return [c.name for c in self.tool_calls]


def make_llm_client(provider: str | None = None) -> tuple[OpenAI, str, dict]:
    """Return (client, model, extra request params) for the configured provider."""
    provider = (provider or os.getenv("VAAK_LLM_PROVIDER", "sarvam")).lower()
    if provider == "sarvam":
        key = os.getenv("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY is not set in .env")
        client = OpenAI(
            api_key=key,
            base_url=SARVAM_BASE_URL,
            default_headers={"api-subscription-key": key},
            timeout=LLM_TIMEOUT_S,
            max_retries=1,
        )
        # sarvam-105b is a reasoning model; low effort keeps voice latency down.
        extra = {"extra_body": {"reasoning_effort": os.getenv("SARVAM_REASONING_EFFORT", "low")}}
        return client, os.getenv("SARVAM_LLM_MODEL", "sarvam-105b"), extra
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        return OpenAI(timeout=LLM_TIMEOUT_S, max_retries=1), os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"), {}
    raise ValueError(f"Unknown LLM provider: {provider!r}")


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        system_prompt: str = "",
        provider: str | None = None,
        max_steps: int = 4,
    ) -> None:
        self.registry = registry
        self.system_prompt = BASE_SYSTEM_PROMPT + ("\n" + system_prompt if system_prompt else "")
        self.client, self.model, self._extra = make_llm_client(provider)
        self.max_steps = max_steps

    def run(
        self,
        query: str,
        language_code: str | None = None,
        conversation: Conversation | None = None,
    ) -> AgentResult:
        """Answer one query, then check the answer's numbers against tool results.

        language_code: optional BCP-47 code from STT; otherwise detected from the text.
        conversation: earlier turns to use as context; this turn is appended to it.
        """
        lang, instruction = reply_instruction(query, language_code)
        result = AgentResult(query=query, answer="", reply_language=lang, model=self.model)
        history = conversation.messages() if conversation else []
        messages: list[dict] = [
            {"role": "system", "content": f"{self.system_prompt}\nReply language: {instruction}"},
            *history,
            {"role": "user", "content": query},
        ]
        turn_start = len(messages) - 1
        self._t0 = time.perf_counter()

        answer = self._loop(messages, result, rewrite=False)
        turn_messages = messages[turn_start:]
        grounding = self._check(answer, messages[1:], result)

        if grounding.verdict == "flagged":
            # One rewrite. The draft and the correction note are not kept in memory.
            result.draft_answer = answer
            work = messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": correction_prompt(grounding.unsupported)},
            ]
            correction_at = len(work)
            answer = self._loop(work, result, rewrite=True)
            new_tool_traffic = work[correction_at:]
            turn_messages = turn_messages + new_tool_traffic
            second = self._check(answer, messages[1:] + new_tool_traffic, result, after_rewrite=True)
            grounding = GroundingResult(
                "corrected" if second.verdict == "verified" else "flagged",
                checked=second.checked,
                unsupported=second.unsupported,
            )

        result.answer = answer
        result.grounding = grounding
        if conversation is not None:
            conversation.add_turn(turn_messages + [{"role": "assistant", "content": answer}])
        result.latency_ms = self._elapsed()
        return result

    def _loop(self, messages: list[dict], result: AgentResult, rewrite: bool) -> str:
        """LLM <-> tools until the LLM answers. Appends tool traffic to messages in place."""
        tools = self.registry.schemas()
        for _ in range(self.max_steps):
            t = self._elapsed()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                temperature=0,
                **self._extra,
            )
            result.llm_calls += 1
            msg = resp.choices[0].message

            if not msg.tool_calls:
                label = "Rewrote the answer to match tool results" if rewrite else "Wrote the answer"
                result.trace.append(TraceStep("llm", label, self.model, t, self._elapsed() - t))
                return (msg.content or "").strip()

            names = ", ".join(tc.function.name for tc in msg.tool_calls)
            result.trace.append(TraceStep("llm", f"Chose {names}", self.model, t, self._elapsed() - t))
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                t = self._elapsed()
                output = self.registry.call(tc.function.name, tc.function.arguments)
                args = _parse_args(tc.function.arguments)
                result.tool_calls.append(ToolCall(tc.function.name, args, output))
                result.trace.append(TraceStep(
                    "tool", f"Ran {tc.function.name}", _summarize(args, output), t, self._elapsed() - t,
                    error="error" in output,
                ))
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output, ensure_ascii=False)}
                )
        return "Sorry, I couldn't finish that request."

    def _check(
        self, answer: str, context: list[dict], result: AgentResult, after_rewrite: bool = False
    ) -> GroundingResult:
        t = self._elapsed()
        known, had_tools = known_from_messages(context)
        g = check_answer(answer, known, had_tools, turn_used_tools=bool(result.tool_calls))
        if g.verdict == "unverified":
            detail = "Direct answer from the model — no tool results to check against"
        elif not result.tool_calls:
            detail = f"{len(g.checked)} number(s) match earlier tool results"
        elif g.unsupported:
            detail = f"Not found in tool results: {', '.join(g.unsupported)}"
        elif g.checked:
            detail = f"{len(g.checked)} number(s) match tool results" + (" after rewrite" if after_rewrite else "")
        else:
            detail = "No numbers in the answer to check"
        result.trace.append(TraceStep(
            "check", "Checked numbers against tool results", detail, t, self._elapsed() - t,
            error=bool(g.unsupported),
        ))
        return g

    def _elapsed(self) -> float:
        return (time.perf_counter() - self._t0) * 1000


def _summarize(args: dict, output: dict, limit: int = 110) -> str:
    text = f"{json.dumps(args, ensure_ascii=False)} → {json.dumps(output, ensure_ascii=False)}"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _parse_args(arguments: str) -> dict:
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw": arguments}


def load_tools(module_name: str, registry: ToolRegistry) -> str:
    """Import a tools module by name, let it register itself, return its system prompt."""
    module = importlib.import_module(module_name)
    module.register(registry)
    return getattr(module, "SYSTEM_PROMPT", "")


def print_result(r: AgentResult) -> None:
    print(f"query   : {r.query}")
    print(f"route   : {r.route}   (reply language: {r.reply_language})")
    for c in r.tool_calls:
        print(f"  tool  : {c.name}({json.dumps(c.arguments, ensure_ascii=False)}) -> "
              f"{json.dumps(c.result, ensure_ascii=False)}")
    print(f"answer  : {r.answer}")
    if r.grounding:
        extra = f" (unsupported: {', '.join(r.grounding.unsupported)})" if r.grounding.unsupported else ""
        print(f"check   : {r.grounding.verdict}{extra}")
    if r.draft_answer:
        print(f"draft   : {r.draft_answer}")
    print(f"latency : {r.latency_ms:.0f} ms ({r.llm_calls} LLM calls, {r.model})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one text query through the agent.")
    parser.add_argument("query")
    parser.add_argument("--tools", default=None, help="tools module to load, e.g. tools.study_tools")
    parser.add_argument("--provider", choices=["sarvam", "openai"], default=None)
    args = parser.parse_args()

    registry = ToolRegistry()
    prompt = load_tools(args.tools, registry) if args.tools else ""
    print_result(Agent(registry, system_prompt=prompt, provider=args.provider).run(args.query))


if __name__ == "__main__":
    main()
