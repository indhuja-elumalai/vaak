# Vaak

A voice-first agent that answers questions by deciding, per query, whether to reason directly or call a real tool — built as a reusable voice+agent runtime, demonstrated first on a study-Q&A use case.

## Problem Statement

Most "voice AI" demos are a thin STT→LLM→TTS wrapper: they sound impressive but never actually *do* anything — every answer is generated text, nothing is verified, nothing is measured. Two separate problems get conflated as one:

1. **Voice as an interface** — accurately transcribing and synthesizing natural (including Hindi/Tamil/English code-mixed) speech.
2. **Agentic decision-making** — knowing when a query needs a deterministic tool (a calculation, a lookup, an API call) versus when a direct language-model answer is correct and sufficient, and being able to prove that decision was right.

Almost no project at the "quick demo" level treats problem 2 as a real engineering problem with a measurable answer — accuracy, task-completion rate, and latency get asserted, not shown.

## Solution

Vaak is a small, deliberately reusable **voice + agent runtime** with a hard boundary between:
- the **runtime** (STT → intent/tool-routing → TTS, generic, domain-agnostic), and
- the **tools** (whatever real actions a specific use case needs — today, study-helper tools; later, potentially tools from other projects).

It ships with an evaluation harness that scores every run against a fixed test set — transcript accuracy, correct tool-vs-direct-answer decisions, and end-to-end latency — so quality claims are backed by numbers, not vibes.

**Today's demo domain:** a study-helper voice agent that answers questions in Hindi/Tamil/English/code-mixed speech, using three tools (`calculate`, `convert_units`, `lookup_formula`) where appropriate, and direct LLM answers otherwise.

## Architecture

```
 Voice/text input
        │
        ▼
   voice_io/stt.py ──────►  transcript
        │
        ▼
 core/agent_runtime.py
   ├── decides: direct answer OR tool call
   ├── tool_registry.py (generic dispatch)
   └── calls into tools/*.py (domain-specific, pluggable)
        │
        ▼
   voice_io/tts.py ──────►  spoken response
        │
        ▼
 eval/eval_runner.py (scores against eval/*.json test sets)
```

**Design rule:** `core/` and `voice_io/` never import from `tools/`. Tools register themselves with the runtime. This is what makes the runtime reusable across domains without rewrites — see "Future integration" below.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python |
| STT / TTS | Sarvam AI API (Indian language/accent support), fallback: OpenAI Whisper + TTS API |
| Agent / tool-calling | Sarvam `sarvam-105b` (OpenAI-compatible function calling), switchable to OpenAI via `VAAK_LLM_PROVIDER` |
| Web UI | FastAPI + plain HTML/CSS/JS (no framework, no build step) |
| Eval | Custom lightweight script (`eval/eval_runner.py`), no heavy framework |
| Dev environment | Local script / Colab notebook |
| Version control | Git, branch-per-feature workflow (see below) |

## Repo Structure

```
vaak/
  core/
    agent_runtime.py      # input → intent → tool-or-direct → output loop
    pipeline.py           # STT → agent → TTS, with text-only fallback
    tool_registry.py      # register_tool(name, schema, handler)
  voice_io/               # named voice_io, not io: a local `io` package collides with the Python stdlib
    stt.py                 # speech-to-text wrapper
    tts.py                 # text-to-speech wrapper
  tools/
    study_tools.py         # calculate(), convert_units(), lookup_formula()
  eval/
    eval_runner.py         # scoring harness
    study_eval_set.json    # today's test cases
  smoke_agent.py           # 13-query routing check (feat/agent-core manual test)
  cli.py                   # wires everything together for the demo
  web/
    server.py              # FastAPI: /api/ask/text, /api/ask/audio, reply audio
    static/                # single-page UI (HTML/CSS/JS, no build step)
  README.md
```

## Git Workflow — branch-wise, with manual test checkpoints

Each branch below must pass its own manual test checklist before merging to `main`. After each merge, tick it off in the **Progress Tracker** further down and commit that README update as part of the merge.

| Branch | Goal | Manual test before merge |
|---|---|---|
| `feat/stt-io` | STT + TTS wrappers working round-trip | Run `python -m voice_io.stt` on a sample audio file → transcript printed correctly. Run `python -m voice_io.tts` on sample text → audio file produced and playable. |
| `feat/agent-core` | Agent decides direct-answer vs tool-call correctly | Run `python smoke_agent.py` on 13 hardcoded text queries (English, Hindi, Hinglish, Tamil, Tanglish) → confirm each routes to the correct tool or direct answer by manual inspection. |
| `feat/full-loop` | STT → agent → TTS wired end to end | Speak/upload one query → hear a correct spoken response, no crashes. Confirm text-only fallback path also works. |
| `feat/web-ui` | Clean, professional web interface over the full loop | Open the local web app → ask by mic and by typing → see transcript, tool used, language and latency, and hear the reply. Fallback to typing works when the mic is unavailable. |
| `feat/memory` | Multi-turn conversations: follow-ups use earlier context | Ask a question, then a follow-up that depends on it ("and in grams?") in the web UI and CLI → the follow-up is answered correctly. "New chat" starts fresh. |
| `feat/grounding` | Answers verified against tool results, with a per-turn trace | Numbers in answers match tool outputs (mismatches are corrected or flagged); direct answers are labelled unverified; each answer shows a step-by-step timeline in the UI. |
| `feat/eval-harness` | Eval script scores the full pipeline | Run `eval/eval_runner.py` against `study_eval_set.json` (single-turn + follow-ups, text + voice) → a results table prints with transcript accuracy, routing accuracy, task completion, grounding rate and latency; model comparison; CI runs the eval on every PR. |
| `docs/demo` | README finalized, Loom recorded | README's Progress Tracker fully checked. Loom recorded showing one live query + eval output. |

## Progress Tracker

- [x] `feat/stt-io` merged
- [x] `feat/agent-core` merged
- [x] `feat/full-loop` merged
- [x] `feat/web-ui` merged
- [x] `feat/memory` merged
- [ ] `feat/grounding` merged
- [ ] `feat/eval-harness` merged
- [ ] `docs/demo` merged — Loom recorded

## How to Run

```bash
pip install -r requirements.txt
cp .env.example .env     # add SARVAM_API_KEY (and optionally OPENAI_API_KEY)
python cli.py                              # interactive: type, or /audio <path>
python cli.py --audio samples/question.m4a # spoken question from a file
python cli.py --text "37 times 48?" --text-only
python -m web.server                       # web app at http://localhost:8765
python eval/eval_runner.py   # run the eval suite
```

## Future integration

The `core/` runtime is domain-agnostic by design. A future `tools/*.py` file (e.g. tools that call into a separate project's API) can be registered without touching `core/` or `voice_io/` at all — the voice+agent+eval infrastructure built here is meant to be reused, not rebuilt, for any future project that needs a natural-language front end over real actions.

**Planned: `feat/second-domain`** — plug a second tool module (e.g. finance tools from a separate project) into the same runtime with `--tools tools.finance_tools`, with zero changes to `core/` or `voice_io/`, to prove the runtime/tools boundary in practice.

## License

MIT (or your preference)
