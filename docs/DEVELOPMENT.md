# Development notes

How Vaak was built, branch by branch, and how to work on it.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # Python 3.11+ (math.cbrt)
pip install -r requirements-dev.txt
cp .env.example .env                                     # add SARVAM_API_KEY
```

## Checks

```bash
ruff check .                         # lint (config in ruff.toml)
pytest -q                            # offline: no keys or network needed
python smoke_agent.py                # live: 13 routing checks (uses API credits)
python smoke_agent.py --followups    # live: 4 follow-up conversations
python smoke_agent.py --grounding    # live: grounding verdicts
python eval/eval_runner.py --mode text            # live: full eval, writes eval/results/
python eval/eval_runner.py --mode text --limit 5  # live: first 5 cases only
```

CI (`.github/workflows/ci.yml`) runs ruff and pytest on every push and pull request. The live
eval runs only when started by hand (Actions → CI → Run workflow → tick "run_eval"), using
the `SARVAM_API_KEY` repository secret.

## Rules of the codebase

- `core/` and `voice_io/` never import from `tools/`. Domain tools register themselves.
- The web server's public-deployment safeguards (`web/ops.py`) are off unless their
  environment variables are set, so local runs behave the same as without them.
- Eval numbers in the README come from a real run in `eval/results/`; they are never edited by hand.

## Branch history and manual test checkpoints

Each branch below must pass its own manual test checklist before merging to `main`. After each merge it was ticked off in the progress tracker below.

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

### Progress tracker

- [x] `feat/stt-io` merged
- [x] `feat/agent-core` merged
- [x] `feat/full-loop` merged
- [x] `feat/web-ui` merged
- [x] `feat/memory` merged
- [x] `feat/grounding` merged
- [x] `feat/eval-harness` merged
- [ ] `docs/demo` merged — Loom recorded

### Polish branch

- [x] `polish/eval-ui-deploy`: 40-case eval with per-language results, route and agent tests, CI with ruff, UI redesign, deployment (Dockerfile, Render, Hugging Face), README rewrite
