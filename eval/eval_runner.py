"""Eval harness: score the full pipeline against eval/study_eval_set.json.

For every case (one or more turns sharing a conversation) and every mode:
  text   the question goes straight to the agent
  voice  STT -> agent -> TTS (the real loop) on a recorded question from eval/audio/
         when the turn has one, otherwise on the question synthesized with TTS

Scores per turn: routing (right tool or direct), task completion (correct final
answer), grounding verdict, latency, and in voice mode transcript word error rate.
Every score is also broken down by language.

  python eval/eval_runner.py                          # both modes, configured model
  python eval/eval_runner.py --mode text              # faster: text only
  python eval/eval_runner.py --mode text --limit 5    # first 5 cases only
  python eval/eval_runner.py --models sarvam-105b,sarvam-105b-conversations --mode text
  python eval/eval_runner.py --cases followup-en,fact-en -v
  python eval/eval_runner.py --mode text --min-routing 0.85   # CI gate: exit 1 below threshold

Results are printed as tables and saved to eval/results/latest-<mode>.json
(or eval/results/model_comparison.json when comparing several models, or --out),
plus a readable report at eval/results/<date>-<mode>.md. Text and voice runs keep separate files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent_runtime import Agent, load_tools  # noqa: E402
from core.conversation import Conversation  # noqa: E402
from core.pipeline import VoicePipeline  # noqa: E402
from core.tool_registry import ToolRegistry  # noqa: E402
from eval.scoring import (  # noqa: E402
    comparable_scripts, completion_ok, percentile, route_ok, word_error_rate,
)
from voice_io.tts import synthesize  # noqa: E402

DEFAULT_SET = ROOT / "eval" / "study_eval_set.json"
RESULTS_DIR = ROOT / "eval" / "results"
AUDIO_CACHE = ROOT / "eval" / ".cache"
REPLY_DIR = ROOT / "out" / "eval"
TOOLS_MODULE = "tools.study_tools"


@dataclass
class Row:
    model: str
    mode: str
    case: str
    turn: int
    language: str
    category: str
    query: str
    expect_route: list[str]
    tools_used: list[str]
    route_ok: bool
    completion_ok: bool
    verdict: str
    answer: str
    latency_ms: float
    stt_ms: float | None = None
    agent_ms: float | None = None
    tts_ms: float | None = None
    transcript: str | None = None
    wer: float | None = None
    error: str | None = None
    audio_source: str | None = None  # voice mode: "recorded" | "synthetic"


AUDIO_EXTENSIONS = (".m4a", ".wav", ".mp3", ".ogg", ".webm", ".flac")


def recorded_audio(turn: dict) -> Path | None:
    """The turn's recorded question, if it has one on disk (any common audio extension)."""
    if not turn.get("audio"):
        return None
    path = ROOT / turn["audio"]
    for candidate in (path, *(path.with_suffix(ext) for ext in AUDIO_EXTENSIONS)):
        if candidate.is_file():
            return candidate
    return None


def query_audio(text: str, voice: str) -> Path:
    """TTS the question once and cache it, so repeated runs only pay for STT onwards."""
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
    path = AUDIO_CACHE / f"{key}.wav"
    if not path.exists():
        synthesize(text, path, language_code=voice)
    return path


def run_case(agent: Agent, case: dict, mode: str) -> list[Row]:
    pipeline = VoicePipeline(agent, speak=(mode == "voice"), out_dir=REPLY_DIR, conversation=Conversation())
    rows = []
    for i, turn in enumerate(case["turns"], 1):
        base = dict(model=agent.model, mode=mode, case=case["id"], turn=i, language=case["language"],
                    category=case["category"], query=turn["query"], expect_route=turn["expect_route"])
        recorded = recorded_audio(turn) if mode == "voice" else None
        if mode == "voice":
            base["audio_source"] = "recorded" if recorded else "synthetic"
        try:
            if mode == "voice":
                result = pipeline.run_audio(recorded or query_audio(turn["query"], case["voice"]))
            else:
                result = pipeline.run_text(turn["query"])
        except Exception as e:  # a failed turn breaks the conversation, so stop the case
            rows.append(Row(**base, tools_used=[], route_ok=False, completion_ok=False, verdict="error",
                            answer="", latency_ms=0, error=f"{type(e).__name__}: {e}"[:300]))
            break

        a = result.agent
        t = result.timings_ms
        row = Row(
            **base,
            tools_used=a.tools_used,
            route_ok=route_ok(turn["expect_route"], a.tools_used),
            completion_ok=completion_ok(a.answer, turn.get("expect_values"), turn.get("expect_text")),
            verdict=a.grounding.verdict if a.grounding else "none",
            answer=a.answer,
            latency_ms=t.get("total", a.latency_ms),
            stt_ms=t.get("stt"), agent_ms=t.get("agent"), tts_ms=t.get("tts"),
            error="; ".join(result.errors) or None,
        )
        if result.transcript:
            row.transcript = result.transcript.text
            if comparable_scripts(turn["query"], result.transcript.text):
                row.wer = word_error_rate(turn["query"], result.transcript.text)
        rows.append(row)
    return rows


def summarize(rows: list[Row]) -> dict:
    # Turns that crashed (API outage, timeout, out of credits) are infrastructure errors,
    # not wrong answers: they are counted in "errors" and left out of accuracy.
    errored = [r for r in rows if r.verdict == "error"]
    rows = [r for r in rows if r.verdict != "error"]
    n = len(rows)
    tool_turns = [r for r in rows if r.tools_used]
    lat = [r.latency_ms for r in rows if r.latency_ms]
    wers = [r.wer for r in rows if r.wer is not None]
    verdicts = defaultdict(int)
    for r in rows:
        verdicts[r.verdict] += 1
    by_category = {}
    for cat in sorted({r.category for r in rows}):
        cr = [r for r in rows if r.category == cat]
        by_category[cat] = {
            "turns": len(cr),
            "routing": sum(r.route_ok for r in cr) / len(cr),
            "completion": sum(r.completion_ok for r in cr) / len(cr),
        }
    by_language = {}
    for lang in sorted({r.language for r in rows} | {r.language for r in errored}):
        lr = [r for r in rows if r.language == lang]
        lt = [r for r in lr if r.tools_used]
        ll = [r.latency_ms for r in lr if r.latency_ms]
        by_language[lang] = {
            "turns": len(lr),
            "routing": sum(r.route_ok for r in lr) / len(lr) if lr else None,
            "completion": sum(r.completion_ok for r in lr) / len(lr) if lr else None,
            "grounded_tool_turns": (
                sum(r.verdict in ("verified", "corrected") for r in lt) / len(lt) if lt else None
            ),
            "latency_p50_ms": statistics.median(ll) if ll else None,
            "latency_p95_ms": percentile(ll, 95) if ll else None,
            "errors": sum(r.language == lang for r in errored),
        }
    return {
        "turns": n,
        "routing_accuracy": sum(r.route_ok for r in rows) / n if n else 0,
        "task_completion": sum(r.completion_ok for r in rows) / n if n else 0,
        "tool_turns": len(tool_turns),
        "grounded_tool_turns": (
            sum(r.verdict in ("verified", "corrected") for r in tool_turns) / len(tool_turns)
            if tool_turns else None
        ),
        "verdicts": dict(verdicts),
        "latency_ms": {
            "p50": statistics.median(lat) if lat else None,
            "p95": percentile(lat, 95) if lat else None,
            "max": max(lat) if lat else None,
        },
        "transcript_wer": {"mean": statistics.mean(wers) if wers else None, "comparable_turns": len(wers)},
        "errors": len(errored),
        "warnings": sum(r.error is not None for r in rows),  # e.g. TTS failed but the answer stands
        "by_category": by_category,
        "by_language": by_language,
        "audio_sources": dict(sorted(
            {src: sum(r.audio_source == src for r in rows) for src in {r.audio_source for r in rows}
             if src}.items()
        )),
    }


# ---------- printing ----------

def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:4.0f}%"


def _secs(ms: float | None) -> str:
    return "   n/a" if ms is None else f"{ms / 1000:5.1f}s"


def _cut(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def print_row(r: Row, verbose: bool) -> None:
    got = ",".join(r.tools_used) or "direct"
    route = f"{'✓' if r.route_ok else '✗'} {got}"
    wer = "" if r.wer is None else f"  wer {r.wer:.2f}"
    print(f"  {r.case + '#' + str(r.turn):<24} {r.language:<9} {route:<26} "
          f"{'✓' if r.completion_ok else '✗'} done  {r.verdict:<10} {_secs(r.latency_ms)}{wer}")
    if r.error:
        print(f"      error: {_cut(r.error, 110)}")
    if verbose or not (r.route_ok and r.completion_ok):
        if r.transcript:
            print(f"      heard : {_cut(r.transcript, 110)}")
        print(f"      answer: {_cut(r.answer, 110)}")


def print_summary(runs: dict[tuple[str, str], dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'model':<28}{'mode':<7}{'scored':>7}{'routing':>9}{'complete':>10}{'grounded':>10}"
          f"{'p50':>8}{'p95':>8}{'max':>8}{'WER':>7}{'errors':>8}")
    print("-" * 100)
    for (model, mode), s in runs.items():
        wer = s["transcript_wer"]["mean"]
        print(f"{model:<28}{mode:<7}{s['turns']:>7}{_pct(s['routing_accuracy']):>9}"
              f"{_pct(s['task_completion']):>10}{_pct(s['grounded_tool_turns']):>10}"
              f"{_secs(s['latency_ms']['p50']):>8}{_secs(s['latency_ms']['p95']):>8}"
              f"{_secs(s['latency_ms']['max']):>8}{'  n/a' if wer is None else f'{wer:6.2f}':>7}{s['errors']:>8}")
    print("=" * 100)
    print("scored   = turns that ran; turns that crashed (API outage, timeout, no credits) count as errors, not as wrong answers")
    print("routing  = right tool, or no tool for direct questions   complete = final answer contains the correct value")
    print("grounded = tool answers whose numbers match tool results (verified or corrected)")
    print("WER      = transcript word error rate, voice mode, same-script turns only (romanized input is transliterated by STT)")
    for (model, mode), s in runs.items():
        cats = "  ".join(f"{c} {v['completion'] * 100:.0f}%" for c, v in s["by_category"].items())
        v = s["verdicts"]
        print(f"\n{model} / {mode}: completion by category → {cats}")
        print(f"  checks → verified {v.get('verified', 0)}, corrected {v.get('corrected', 0)}, "
              f"flagged {v.get('flagged', 0)}, unverified {v.get('unverified', 0)}")
        if s.get("audio_sources"):
            print("  audio  → " + ", ".join(f"{k} {n}" for k, n in s["audio_sources"].items()))
        print(f"  {'language':<11}{'scored':>7}{'routing':>9}{'complete':>10}{'grounded':>10}"
              f"{'p50':>8}{'p95':>8}{'errors':>8}")
        for lang, b in s["by_language"].items():
            print(f"  {lang:<11}{b['turns']:>7}{_pct(b['routing']):>9}{_pct(b['completion']):>10}"
                  f"{_pct(b['grounded_tool_turns']):>10}{_secs(b['latency_p50_ms']):>8}"
                  f"{_secs(b['latency_p95_ms']):>8}{b['errors']:>8}")


# ---------- markdown report ----------

def _md_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _md_ms(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms:,.0f} ms"


def _md_cell(text: str, n: int = 90) -> str:
    return _cut(text, n).replace("|", "\\|")


def render_markdown(report: dict) -> str:
    """A human-readable report of one eval run (numbers exactly as measured, one decimal)."""
    out = [
        f"# Vaak eval report — {report['run_at'][:10]}",
        "",
        f"- Eval set: `{report['eval_set']}` v{report['eval_set_version']}"
        f" ({report['cases']} cases, {report['turns_in_set']} turns)",
        f"- Run at: {report['run_at']} (UTC), commit `{report['git_commit'] or 'unknown'}`",
        f"- Modes: {', '.join(report['modes'])}",
        "",
        "## Summary",
        "",
        "| Model | Mode | Scored turns | Routing | Task completion | Grounded tool answers |"
        " Latency p50 | Latency p95 | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report["summary"]:
        out.append(
            f"| {s['model']} | {s['mode']} | {s['turns']} | {_md_pct(s['routing_accuracy'])} |"
            f" {_md_pct(s['task_completion'])} | {_md_pct(s['grounded_tool_turns'])} |"
            f" {_md_ms(s['latency_ms']['p50'])} | {_md_ms(s['latency_ms']['p95'])} | {s['errors']} |"
        )
    for s in report["summary"]:
        out += [
            "", f"## By language — {s['model']} / {s['mode']}", "",
            "| Language | Scored turns | Routing | Task completion | Grounded tool answers |"
            " Latency p50 | Latency p95 | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for lang, b in s["by_language"].items():
            out.append(
                f"| {lang} | {b['turns']} | {_md_pct(b['routing'])} | {_md_pct(b['completion'])} |"
                f" {_md_pct(b['grounded_tool_turns'])} | {_md_ms(b['latency_p50_ms'])} |"
                f" {_md_ms(b['latency_p95_ms'])} | {b['errors']} |"
            )
        v = s["verdicts"]
        wer = s["transcript_wer"]
        out += [
            "",
            f"Grounding verdicts: verified {v.get('verified', 0)}, corrected {v.get('corrected', 0)},"
            f" flagged {v.get('flagged', 0)}, unverified {v.get('unverified', 0)}"
            " (unverified = direct answers, which have no tool result to check against).",
        ]
        if s.get("audio_sources"):
            out.append("Voice input: " + ", ".join(f"{k} {n}" for k, n in s["audio_sources"].items()) + " turns.")
        if wer["mean"] is not None:
            out.append(f"Transcript WER: {wer['mean']:.3f} mean over {wer['comparable_turns']} same-script turns.")

    misses = [r for r in report["rows"] if not (r["route_ok"] and r["completion_ok"])]
    out += ["", "## Misses", ""]
    if not misses:
        out.append("None: every scored turn routed correctly and contained the expected answer.")
    else:
        out += ["| Model / mode | Case | Language | Query | Expected route | Got | Done | Answer or error |",
                "|---|---|---|---|---|---|---|---|"]
        for r in misses:
            got = ",".join(r["tools_used"]) or ("error" if r["verdict"] == "error" else "direct")
            detail = r["error"] if r["verdict"] == "error" else r["answer"]
            out.append(
                f"| {r['model']} / {r['mode']} | {r['case']}#{r['turn']} | {r['language']} |"
                f" {_md_cell(r['query'], 60)} | {', '.join(r['expect_route'])} | {got} |"
                f" {'yes' if r['completion_ok'] else 'no'} | {_md_cell(detail or '')} |"
            )
    out += [
        "",
        "Routing = the right tool was called, or no tool for direct questions. "
        "Task completion = the final answer contains the expected value (rounding allowed) or phrase. "
        "Grounded = tool answers whose numbers all match tool results (verified or corrected). "
        "Turns that crashed (API outage, timeout, no credits) are counted as errors and left out of the percentages.",
        "",
    ]
    return "\n".join(out)


def _shown(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the Vaak pipeline against an eval set.")
    parser.add_argument("--set", default=str(DEFAULT_SET), help="eval set JSON")
    parser.add_argument("--mode", choices=["text", "voice", "both"], default="both")
    parser.add_argument("--models", default=None, help="comma-separated model names to compare")
    parser.add_argument("--provider", choices=["sarvam", "openai"], default=None)
    parser.add_argument("--cases", default=None, help="comma-separated case ids to run")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument("--no-md", action="store_true", help="don't write eval/results/<date>.md")
    parser.add_argument("--out", default=None,
                        help="report path (default: results/latest-<mode>.json, or results/model_comparison.json with --models)")
    parser.add_argument("--min-routing", type=float, default=None, help="exit 1 if routing accuracy is lower")
    parser.add_argument("--min-completion", type=float, default=None, help="exit 1 if task completion is lower")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every answer")
    args = parser.parse_args()

    spec = json.loads(Path(args.set).read_text(encoding="utf-8"))
    cases = spec["cases"]
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    modes = ["text", "voice"] if args.mode == "both" else [args.mode]
    models = args.models.split(",") if args.models else [None]

    registry = ToolRegistry()
    prompt = load_tools(TOOLS_MODULE, registry)
    all_rows: list[Row] = []
    runs: dict[tuple[str, str], dict] = {}
    started = time.perf_counter()

    for model in models:
        agent = Agent(registry, system_prompt=prompt, provider=args.provider, model=model)
        for mode in modes:
            print(f"\n▶ {agent.model} · {mode} mode · {len(cases)} cases")
            rows = []
            for case in cases:
                for row in run_case(agent, case, mode):
                    print_row(row, args.verbose)
                    rows.append(row)
            runs[(agent.model, mode)] = summarize(rows)
            all_rows += rows

    print_summary(runs)
    print(f"\nfinished in {time.perf_counter() - started:.0f}s")

    if all(s["turns"] == 0 for s in runs.values()):
        print("\nNo turn was scored (every turn errored — check API keys/credits). Report not saved.")
        return 2
    default_name = "model_comparison.json" if len(models) > 1 else f"latest-{args.mode}.json"
    out = Path(args.out) if args.out else RESULTS_DIR / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "eval_set": spec["name"], "eval_set_version": spec.get("version"),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "modes": modes,
        "cases": len(cases), "turns_in_set": sum(len(c["turns"]) for c in cases),
        "summary": [{"model": m, "mode": md, **s} for (m, md), s in runs.items()],
        "rows": [asdict(r) for r in all_rows],
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {_shown(out)}")
    if not args.no_md:
        if args.out:
            md = out.with_suffix(".md")
        else:
            stem = f"{datetime.now():%Y-%m-%d}-{args.mode}" + ("-model-comparison" if len(models) > 1 else "")
            md = RESULTS_DIR / f"{stem}.md"
        md.write_text(render_markdown(report), encoding="utf-8")
        print(f"saved {_shown(md)}")

    failed = False
    for (model, mode), s in runs.items():
        if args.min_routing is not None and s["routing_accuracy"] < args.min_routing:
            print(f"FAIL: {model}/{mode} routing {s['routing_accuracy']:.2f} < {args.min_routing}")
            failed = True
        if args.min_completion is not None and s["task_completion"] < args.min_completion:
            print(f"FAIL: {model}/{mode} completion {s['task_completion']:.2f} < {args.min_completion}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
