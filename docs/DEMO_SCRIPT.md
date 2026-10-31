# Demo script (60–90 seconds)

What it shows: per-question routing (tool vs direct), grounding, follow-up memory, the trace,
and four languages. Record the browser window at about 1280 × 800 with system audio on.

**Before recording:** open the app (local `python -m web.server`, or the deployed URL a minute
early so it's awake), click **New chat**, keep spoken replies on, and allow the microphone.

| Time | Do | Say or ask | Point at |
|---|---|---|---|
| 0:00 | Show the empty page | "This is Vaak, a voice study helper for English, Hindi, Tamil and code-mixed speech." | The hero and language chips |
| 0:07 | Click the big mic, speak, click again | **"What is 37 times 48?"** | Chips: *Tool: calculate*, *Verified*, the latency. The spoken answer plays |
| 0:17 | Open "How Vaak answered" | "Every answer shows its steps: speech to text, the model choosing a tool, the tool result, the number check, and speech." | The timeline durations |
| 0:27 | Speak | **"25 கிலோகிராம் எத்தனை பவுண்டு?"** | Tamil reply, *Tool: convert_units*, *Verified* |
| 0:38 | Speak (Hinglish) | **"Kinetic energy ka formula kya hai?"** | *Tool: lookup_formula*, reply in Hinglish |
| 0:47 | Speak the follow-up | **"Ab 2 kg aur 3 m/s ke liye calculate karo"** | *Used context from 1 earlier turn*, *Tool: calculate*, answer 9 |
| 0:58 | Type (show the text path) | **"Who wrote India's national anthem?"** | *Direct answer* and *Unverified*: "No tool fits, so it answers directly and says it couldn't verify." |
| 1:08 | Switch to the README | "Every change is scored on a 40-question eval across five languages." | The evaluation results table |
| 1:18 | End | "Code, eval and deployment are on GitHub." | — |

**Tips**
- If speech recognition mishears a language, set the language picker next to the mic and repeat.
- Keep each question short; total turn time is shown on the latency chip.
- For a silent GIF for the README, record just 0:07–0:27 (one voice question plus the trace),
  for example with Kap or the macOS screenshot toolbar (Cmd+Shift+5), then convert to GIF and
  save it as `docs/demo.gif`.
