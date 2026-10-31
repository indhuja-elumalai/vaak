# Portfolio and resume copy

All numbers below come from `eval/results/2026-10-31-text.md`, `eval/results/2026-10-31-voice.md`
and `pytest` on this branch. Update them if you re-run the eval.

## Resume bullets

- Built **Vaak**, a voice-first study agent in Python/FastAPI that answers in English, Hindi,
  Tamil and code-mixed speech and decides per question whether to call a deterministic tool or
  answer directly; on a 40-case, 47-turn multilingual eval it routed **97.9%** of turns correctly,
  completed **97.9%** in text and **91.5%** end to end by voice, at **0.48 s** median text latency.
- Designed a grounding check that verifies every number in a spoken answer against tool results
  and forces a rewrite on mismatch (**100%** of tool answers grounded in both eval modes), plus an
  eval harness with per-language breakdowns, **82** offline tests with a scripted model, CI, and
  Docker deployment with per-IP rate limits and a daily spend cap.

## Portfolio description

Vaak is a voice study helper for Indian students that listens in English, Hindi, Tamil or a mix,
and decides for each question whether to calculate, convert units, look up a formula, or simply
answer. Every number in its spoken reply is checked against the tool that produced it, and each
answer shows a step-by-step trace with timings. I built it as a reusable voice-and-agent runtime
with a strict boundary between the runtime and the domain tools, and measured it with a 40-case
multilingual eval: 97.9% routing accuracy, 97.9% task completion in text and 91.5% end to end by
voice, with every miss written up in the report.
