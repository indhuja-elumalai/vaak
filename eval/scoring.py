"""Pure scoring functions for the eval harness (no API calls, unit-tested in tests/)."""
from __future__ import annotations

import math
import re
import unicodedata

from core.agent_runtime import detect_language
from core.grounding import extract_numbers, is_supported


def route_ok(expect_route: list[str], tools_used: list[str]) -> bool:
    """Pass if any acceptable route happened: a listed tool was called, or "direct" with no tools."""
    for route in expect_route:
        if route == "direct" and not tools_used:
            return True
        if route != "direct" and route in tools_used:
            return True
    return False


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def completion_ok(answer: str, expect_values: list[float] | None = None,
                  expect_text: list[str] | None = None) -> bool:
    """Task completion: the final answer contains an expected value (rounding-aware) or phrase."""
    if expect_values:
        claims = extract_numbers(answer)
        if not any(is_supported(c, {float(v)}) for c in claims for v in expect_values):
            return False
    if expect_text:
        squashed = _squash(answer)
        if not any(_squash(t) in squashed for t in expect_text):
            return False
    return bool(expect_values or expect_text)


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    # Keep letters, digits and Indic combining marks; drop punctuation.
    return re.findall(r"[\wऀ-ॿ஀-௿]+", text)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance / reference length."""
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def comparable_scripts(reference: str, hypothesis: str) -> bool:
    """WER is only meaningful when reference and transcript use the same script.

    Romanized Hinglish/Tanglish come back from STT in Devanagari/Tamil script, so a
    word-by-word comparison would measure transliteration, not recognition.
    """
    ref_lang, ref_script = detect_language(reference)
    hyp_lang, hyp_script = detect_language(hypothesis)
    if ref_script != hyp_script:
        return False
    return ref_script == "latin" or ref_lang == hyp_lang


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return math.nan
    ordered = sorted(values)
    k = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[k]
