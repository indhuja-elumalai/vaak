"""Grounding: check that the numbers in an answer are backed by tool results.

The LLM words the final answer itself, so it can misquote a tool result ("55.21 pounds"
when the tool said 55.1156) or invent a number the tools never produced. This module
extracts every number in the answer and checks it against the numbers the turn is
allowed to use: tool arguments and results (this turn and earlier ones) and the user's
own messages. Rounding is accepted (55.12 matches 55.1156).

Verdicts:
  verified    tools were used and every number in the answer is supported
  corrected   the first draft had unsupported numbers; the rewrite passed
  flagged     unsupported numbers remain after the rewrite
  unverified  a direct answer whose numbers aren't in any tool result (facts we can't check)

This module is domain-agnostic: it must never import from tools/.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

# Plain or comma-grouped numbers (1,776 / 1,00,000 / 55.12), not glued to letters ("v2", "m1").
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?(?![\w])|(?<![\w.])\d+(?:\.\d+)?")


@dataclass
class NumberClaim:
    text: str
    value: float
    decimals: int


@dataclass
class GroundingResult:
    verdict: str
    checked: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> list[NumberClaim]:
    claims = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0).replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        claims.append(NumberClaim(m.group(0), float(raw), decimals))
    return claims


def collect_known(values) -> set[float]:
    """Every number inside tool arguments/results or text, walking nested JSON."""
    known: set[float] = set()

    def walk(v) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            known.add(abs(float(v)))
        elif isinstance(v, str):
            known.update(c.value for c in extract_numbers(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                walk(x)

    walk(values)
    return known


def known_from_messages(messages: list[dict]) -> tuple[set[float], bool]:
    """Numbers from user messages and tool traffic in a message history.

    Returns (known numbers, whether any tool results were present).
    """
    known: set[float] = set()
    had_tools = False
    for m in messages:
        role = m.get("role")
        if role == "user":
            known |= collect_known(m.get("content", ""))
        elif role == "tool":
            had_tools = True
            try:
                known |= collect_known(json.loads(m.get("content") or "{}"))
            except json.JSONDecodeError:
                known |= collect_known(m.get("content", ""))
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                try:
                    known |= collect_known(json.loads(tc["function"]["arguments"] or "{}"))
                except (KeyError, json.JSONDecodeError):
                    pass
    return known, had_tools


def is_supported(claim: NumberClaim, known: set[float]) -> bool:
    scale = 10 ** claim.decimals
    for k in known:
        if math.isclose(claim.value, k, rel_tol=1e-9, abs_tol=1e-9):
            return True
        # Rounded (55.12 <- 55.1156) or truncated (55.11 <- 55.1156) to the claim's precision.
        if abs(round(k, claim.decimals) - claim.value) < 1e-9:
            return True
        if abs(math.floor(k * scale) / scale - claim.value) < 1e-9:
            return True
    return False


def check_answer(
    answer: str, known: set[float], had_tools: bool, turn_used_tools: bool = True
) -> GroundingResult:
    """had_tools: any tool results in context; turn_used_tools: this turn called a tool.

    Only a turn that used tools can be flagged. A direct answer is "verified" when it
    just restates earlier tool results, otherwise "unverified" (facts we can't check).
    """
    claims = extract_numbers(answer)
    if not turn_used_tools:
        supported = had_tools and claims and all(is_supported(c, known) for c in claims)
        return GroundingResult("verified" if supported else "unverified", checked=[c.text for c in claims])
    unsupported = [c.text for c in claims if not is_supported(c, known)]
    return GroundingResult(
        "flagged" if unsupported else "verified",
        checked=[c.text for c in claims],
        unsupported=unsupported,
    )


def correction_prompt(unsupported: list[str]) -> str:
    nums = ", ".join(unsupported)
    return (
        f"Check failed: the number(s) {nums} in your answer do not appear in any tool result "
        "or in the user's question. Rewrite your answer using only numbers from the tool "
        "results above (rounding is fine). If a new number is needed, you may call a tool "
        "to compute it. Keep the same language and length."
    )
