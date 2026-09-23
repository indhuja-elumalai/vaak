# Vaak

A voice-first agent that answers questions by deciding, per query, whether to reason directly or call a real tool — built as a reusable voice+agent runtime, demonstrated first on a study-Q&A use case.

## Problem Statement

Most "voice AI" demos are a thin STT→LLM→TTS wrapper: they sound impressive but never actually *do* anything — every answer is generated text, nothing is verified, nothing is measured. Two separate problems get conflated as one:

1. **Voice as an interface** — accurately transcribing and synthesizing natural (including Hindi/English code-mixed) speech.
2. **Agentic decision-making** — knowing when a query needs a deterministic tool (a calculation, a lookup, an API call) versus when a direct language-model answer is correct and sufficient, and being able to prove that decision was right.

Almost no project at the "quick demo" level treats problem 2 as a real engineering problem with a measurable answer — accuracy, task-completion rate, and latency get asserted, not shown.

## Solution

Vaak is a small, deliberately reusable **voice + agent runtime** with a hard boundary between:
- the **runtime** (STT → intent/tool-routing → TTS, generic, domain-agnostic), and
- the **tools** (whatever real actions a specific use case needs — today, study-helper tools; later, potentially tools from other projects).

It ships with an evaluation harness that scores every run against a fixed test set — transcript accuracy, correct tool-vs-direct-answer decisions, and end-to-end latency — so quality claims are backed by numbers, not vibes.

**Today's demo domain:** a study-helper voice agent that answers questions in Hindi/English/code-mixed speech, using three tools (`calculate`, `convert_units`, `lookup_formula`) where appropriate, and direct LLM answers otherwise.

## Architecture

```
 Voice/text input
        │
        ▼
   io/stt.py  ──────►  transcript
        │
        ▼
 core/agent_runtime.py
   ├── decides: direct answer OR tool call
   ├── tool_registry.py (generic dispatch)
   └── calls into tools/*.py (domain-specific, pluggable)
        │
        ▼
   io/tts.py  ──────►  spoken response
        │
        ▼
 eval/eval_runner.py (scores against eval/*.json test sets)
```

**Design rule:** `core/` and `io/` never import from `tools/`. Tools register themselves with the runtime. This is what makes the runtime reusable across domains without rewrites — see "Future integration" below.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python |
| STT / TTS | Sarvam AI API (Indian language/accent support), fallback: OpenAI Whisper + TTS API |
| Agent / tool-calling | OpenAI or Anthropic function-calling API |
| Eval | Custom lightweight script (`eval/eval_runner.py`), no heavy framework |
| Dev environment | Local script / Colab notebook |
| Version control | Git, branch-per-feature workflow (see below) |

## Repo Structure

```
vaak/
  core/
    agent_runtime.py      # input → intent → tool-or-direct → output loop
    tool_registry.py      # register_tool(name, schema, handler)
  io/
    stt.py                 # speech-to-text wrapper
    tts.py                 # text-to-speech wrapper
  tools/
    study_tools.py         # calculate(), convert_units(), lookup_formula()
  eval/
    eval_runner.py         # scoring harness
    study_eval_set.json    # today's test cases
  cli.py                   # wires everything together for the demo
  README.md
```

## Git Workflow — branch-wise, with manual test checkpoints

Each branch below must pass its own manual test checklist before merging to `main`. After each merge, tick it off in the **Progress Tracker** further down and commit that README update as part of the merge.

| Branch | Goal | Manual test before merge |
|---|---|---|
| `feat/stt-io` | STT + TTS wrappers working round-trip | Run `io/stt.py` on a sample audio file → transcript printed correctly. Run `io/tts.py` on sample text → audio file produced and playable. |
| `feat/agent-core` | Agent decides direct-answer vs tool-call correctly | Run `core/agent_runtime.py` on 5–10 hardcoded text queries → confirm each routes to the correct tool or direct answer by manual inspection. |
| `feat/full-loop` | STT → agent → TTS wired end to end | Speak/upload one query → hear a correct spoken response, no crashes. Confirm text-only fallback path also works. |
| `feat/eval-harness` | Eval script scores the full pipeline | Run `eval/eval_runner.py` against `study_eval_set.json` → a results table prints with accuracy, task-completion rate, and latency per query. |
| `docs/demo` | README finalized, Loom recorded | README's Progress Tracker fully checked. Loom recorded showing one live query + eval output. |

## Progress Tracker

- [ ] `feat/stt-io` merged
- [ ] `feat/agent-core` merged
- [ ] `feat/full-loop` merged
- [ ] `feat/eval-harness` merged
- [ ] `docs/demo` merged — Loom recorded

## How to Run

```bash
pip install -r requirements.txt
python cli.py            # interactive demo
python eval/eval_runner.py   # run the eval suite
```

## Future integration

The `core/` runtime is domain-agnostic by design. A future `tools/*.py` file (e.g. tools that call into a separate project's API) can be registered without touching `core/` or `io/` at all — the voice+agent+eval infrastructure built here is meant to be reused, not rebuilt, for any future project that needs a natural-language front end over real actions.

## License

MIT (or your preference)
