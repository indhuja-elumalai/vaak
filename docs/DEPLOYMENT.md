# Deploying Vaak

Vaak ships as one Docker image (`Dockerfile`) that runs on any host that can run a container.
Two free options are covered here. Both take about 15 minutes.

| | Option A: Render | Option B: Hugging Face Spaces |
|---|---|---|
| Free tier | Free web service | Free CPU Space |
| Port | Render sets `$PORT` (default 10000); the image reads it | 7860 (the image's default) |
| Sleeps when idle | After 15 minutes without traffic | After a period of inactivity |
| Deploys from | Your GitHub repo, automatically on push | Files you upload to the Space |

## Before you start

1. **A Sarvam API key with credits.** Pay as you go has no monthly fee; credits don't expire.
   An OpenAI key is optional (it's only used as a fallback for speech).
2. **Push this repo to GitHub** (Option A deploys straight from it).
3. **Keys only ever go in the host's secret settings.** Never put them in the Dockerfile,
   `render.yaml`, the README or a commit. `.env` is git-ignored and excluded from the image by
   `.dockerignore`.

## Environment variables

| Variable | Where | Value | Purpose |
|---|---|---|---|
| `SARVAM_API_KEY` | **Secret** | your key | Required. LLM, speech-to-text and text-to-speech |
| `OPENAI_API_KEY` | **Secret** | your key, or leave unset | Optional speech fallback |
| `VAAK_RATE_LIMIT_PER_MIN` | Variable | `5` | Questions per minute per visitor |
| `VAAK_RATE_LIMIT_PER_DAY` | Variable | `30` | Questions per day per visitor |
| `VAAK_DAILY_CAP_TOTAL` | Variable | `100` | Questions per day for **everyone together**: the hard limit on your API spend |
| `VAAK_TRUST_PROXY` | Variable | `1` | Read the visitor's IP from the host's proxy header |
| `VAAK_AUDIO_TTL_MIN` | Variable | `30` | Delete spoken replies older than 30 minutes |
| `VAAK_AUDIO_MAX_FILES` | Variable | `200` | Keep at most 200 reply files |
| `VAAK_LLM_PROVIDER`, `VAAK_STT_PROVIDER`, `VAAK_TTS_PROVIDER`, model overrides | Variable | see `.env.example` | Optional; the defaults are fine |

The Dockerfile already sets the limit and cleanup values above, so on Hugging Face you only
need to add the secrets. On Render, `render.yaml` sets them too.

**Budget.** At Sarvam's pay-as-you-go prices a spoken question costs well under ₹1, so a
daily cap of 100 keeps a worst-case day to a small, predictable amount. Start low and raise it
only if real visitors hit the limit. Limits are in memory and reset when the service restarts
(which also happens when it wakes from sleep). The per-visitor limits are best-effort; the
daily total cap is what protects your credits.

## Option A: Render (free web service, from the Dockerfile)

1. Sign in at [render.com](https://render.com) with GitHub.
2. Click **New → Blueprint** and pick the `vaak` repository. Render reads `render.yaml` and
   shows one web service called `vaak` (Docker, free plan, health check `/health`).
3. Render asks for the values of `SARVAM_API_KEY` and `OPENAI_API_KEY`. Paste your Sarvam
   key; leave the OpenAI one empty if you don't have one. Click **Apply**.
4. Wait for the build and deploy (a few minutes the first time). The service URL looks like
   `https://vaak-xxxx.onrender.com`.
5. To change a limit later: open the service → **Environment** → edit the value → **Save**
   (Render redeploys).

*Without the Blueprint:* **New → Web Service** → pick the repo → Language **Docker** →
Instance type **Free** → add the variables above under **Environment** (keys as secrets) →
set **Health Check Path** to `/health` → **Create Web Service**.

**Cold starts:** a free Render service sleeps after 15 minutes without traffic, and the first
request after that wakes it, which can take a while. Open the URL a minute before a demo.

## Option B: Hugging Face Spaces (Docker SDK, port 7860)

1. Sign in at [huggingface.co](https://huggingface.co) → **New Space**.
   Name `vaak`, SDK **Docker** → **Blank**, hardware **CPU basic (free)**, visibility
   **Public** → **Create Space**.
2. In the Space: **Settings → Variables and secrets → New secret**: add `SARVAM_API_KEY`
   (and `OPENAI_API_KEY` if you have one). Secrets are passed to the app as environment
   variables at runtime.
3. Upload the app from your Mac (run in the repo folder):

   ```bash
   pip install -U huggingface_hub
   huggingface-cli login            # paste a write token from huggingface.co/settings/tokens
   SPACE=<your-hf-username>/vaak
   huggingface-cli upload $SPACE . . --repo-type=space \
     --include "Dockerfile" ".dockerignore" "requirements.txt" "core/*" "voice_io/*" "tools/*" "web/*" "web/static/*" \
     --exclude "**/__pycache__/*"
   huggingface-cli upload $SPACE docs/hf-space/README.md README.md --repo-type=space
   ```

   The second command uploads the Space's own README, whose header (`sdk: docker`,
   `app_port: 7860`) tells Hugging Face how to run it. The `.env` file is never uploaded.
4. The Space builds and starts on its own (watch the **Logs** tab). The app is at
   `https://<your-hf-username>-vaak.hf.space`. Share that direct URL rather than the
   huggingface.co/spaces page: the Space page shows the app inside a frame, and some browsers
   block the microphone inside frames.
5. To update after code changes, run the two upload commands again.

**Sleep:** free CPU Spaces pause after a period of inactivity and restart on the next visit,
which takes a little while. Files written inside the container (reply audio) are lost on
restart, which is fine for Vaak.

## Smoke checklist after deploying

Run these against your public URL (`https://…`):

- [ ] `GET /health` returns `{"status":"ok"}` (open `https://…/health` in a browser).
- [ ] The page loads and the header shows the model name.
- [ ] English text: **What is 37 times 48?** → 1776, chip "Tool: calculate", verified.
- [ ] Hindi text: **480 का 25 प्रतिशत कितना है?** → 120, reply in Hindi.
- [ ] Tamil text: **25 கிலோகிராம் எத்தனை பவுண்டு?** → about 55.12, reply in Tamil.
- [ ] Hinglish or Tanglish: **Ohm's law formula enna?** → V = I × R, reply in Tanglish.
- [ ] Voice: click the mic, ask a question, stop → transcript appears, answer is spoken.
      (The browser only allows the mic on `https://` pages, which both hosts provide.)
- [ ] Follow-up: ask **25 kg in pounds?** then **and in grams?** → 25000.
- [ ] Trace: open the timeline under an answer and check each stage has a duration.
- [ ] New chat clears the conversation.
- [ ] Rate limit: asking more than 5 questions within a minute shows the "please wait" message.

## Troubleshooting

- **Build succeeds but the service never becomes healthy:** the app could not start, usually
  because `SARVAM_API_KEY` is missing. Check the logs, add the secret, redeploy.
- **Every answer fails with "No credits available" (402):** top up Sarvam credits.
- **"The demo has reached its daily question limit":** the total cap was reached; it resets at
  midnight UTC (5:30 am IST), or raise `VAAK_DAILY_CAP_TOTAL`.
- **Mic button says the mic is unavailable:** the page must be opened over `https://`, and the
  browser must be allowed to use the microphone. Typing always works.
