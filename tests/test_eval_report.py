"""Eval runner aggregation and the markdown report (pure functions, no API calls)."""
from __future__ import annotations

from dataclasses import asdict

from eval.eval_runner import ROOT, Row, recorded_audio, render_markdown, summarize


def row(case, language, route_ok=True, done=True, verdict="verified", tools=("calculate",),
        latency=500.0, error=None, audio=None):
    return Row(model="m", mode="voice" if audio else "text", case=case, turn=1, language=language,
               category="tool", query=f"{case}?", expect_route=["calculate"], tools_used=list(tools),
               route_ok=route_ok, completion_ok=done, verdict=verdict, answer="42", latency_ms=latency,
               error=error, audio_source=audio)


def test_summary_breaks_scores_down_by_language_and_excludes_errors():
    rows = [
        row("a", "English", latency=100),
        row("b", "English", done=False, latency=300),
        row("c", "Hindi", tools=(), verdict="unverified", latency=200),
        row("d", "Tamil", route_ok=False, done=False, verdict="error", tools=(), latency=0,
            error="APIStatusError: 402"),
    ]
    s = summarize(rows)
    assert s["turns"] == 3 and s["errors"] == 1
    assert s["routing_accuracy"] == 1.0
    assert abs(s["task_completion"] - 2 / 3) < 1e-9
    assert s["grounded_tool_turns"] == 1.0
    en = s["by_language"]["English"]
    assert en["turns"] == 2 and en["completion"] == 0.5 and en["latency_p50_ms"] == 200
    assert s["by_language"]["Hindi"]["grounded_tool_turns"] is None
    assert s["by_language"]["Tamil"] == {
        "turns": 0, "routing": None, "completion": None, "grounded_tool_turns": None,
        "latency_p50_ms": None, "latency_p95_ms": None, "errors": 1,
    }


def test_audio_sources_are_counted():
    s = summarize([row("a", "English", audio="recorded"), row("b", "Hindi", audio="synthetic"),
                   row("c", "Hindi", audio="synthetic")])
    assert s["audio_sources"] == {"recorded": 1, "synthetic": 2}


def test_markdown_report_has_summary_languages_and_misses():
    rows = [row("ok", "English"), row("miss", "Hindi", done=False)]
    report = {"eval_set": "study_eval_set", "eval_set_version": 3, "run_at": "2026-10-31T09:00:00+00:00",
              "git_commit": "abc1234", "modes": ["text"], "cases": 2, "turns_in_set": 2,
              "summary": [{"model": "m", "mode": "text", **summarize(rows)}],
              "rows": [asdict(r) for r in rows]}
    md = render_markdown(report)
    assert md.startswith("# Vaak eval report — 2026-10-31")
    assert "| m | text | 2 | 100.0% | 50.0% | 100.0% | 500 ms | 500 ms | 0 |" in md
    assert "| Hindi | 1 | 100.0% | 0.0% |" in md
    assert "miss#1" in md and "ok#1" not in md


def test_recorded_audio_falls_back_to_other_extensions(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.eval_runner.ROOT", tmp_path)
    (tmp_path / "eval" / "audio").mkdir(parents=True)
    (tmp_path / "eval" / "audio" / "q.wav").write_bytes(b"x")
    assert recorded_audio({"audio": "eval/audio/q.m4a"}) == tmp_path / "eval" / "audio" / "q.wav"
    assert recorded_audio({"audio": "eval/audio/missing.m4a"}) is None
    assert recorded_audio({}) is None
    assert ROOT.name  # the real root is untouched
