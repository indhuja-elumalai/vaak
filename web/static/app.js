"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

const els = {
  thread: $("#thread"), scroller: $("#scroller"), input: $("#input"), composer: $("#composer"),
  micBtn: $("#micBtn"), sendBtn: $("#sendBtn"), attachBtn: $("#attachBtn"), fileInput: $("#fileInput"),
  status: $("#status"), toast: $("#toast"), player: $("#player"), voiceToggle: $("#voiceToggle"),
  themeToggle: $("#themeToggle"), modelPill: $("#modelPill"), footnote: $("#footnote"),
  newChat: $("#newChat"),
};

/* ================= Atom =================
   A nucleus with three tilted orbits. Speed, spread and pulse follow the app state:
   idle (slow orbit), listening (orbits widen with mic level), thinking (fast spin),
   speaking (nucleus pulses with the reply audio). */

const STATE_PARAMS = {
  idle:      { speed: 0.9, spread: 1.0,  glow: 0.30, precess: 0.05 },
  listening: { speed: 1.8, spread: 1.0,  glow: 0.55, precess: 0.12 },
  thinking:  { speed: 4.6, spread: 0.88, glow: 0.60, precess: 0.9 },
  speaking:  { speed: 1.5, spread: 1.0,  glow: 0.55, precess: 0.15 },
};
let atomSeq = 0;

class Atom {
  constructor(host) {
    const id = `atom${atomSeq++}`;
    this.host = host;
    const small = host.classList.contains("atom-sm") || host.classList.contains("atom-xs");
    const stroke = small ? 4.2 : 2.2, electronR = small ? 6.5 : 4.4, nucleusR = small ? 13 : 11;
    host.innerHTML = `
      <svg viewBox="-60 -60 120 120" aria-hidden="true">
        <defs>
          <radialGradient id="${id}g" cx="35%" cy="30%" r="80%">
            <stop offset="0" style="stop-color:var(--accent-2)"/>
            <stop offset="1" style="stop-color:var(--accent)"/>
          </radialGradient>
          <filter id="${id}b" x="-150%" y="-150%" width="400%" height="400%"><feGaussianBlur stdDeviation="8"/></filter>
        </defs>
        <circle class="glow" r="${nucleusR * 2}" fill="url(#${id}g)" filter="url(#${id}b)" opacity=".3"/>
        ${[0, 1, 2].map(() => `<ellipse class="orbit" rx="46" ry="17" fill="none" stroke="currentColor" stroke-width="${stroke}" opacity=".32"/>`).join("")}
        ${[0, 1, 2].map(i => `<circle class="electron" r="${electronR}" style="fill:var(${i === 1 ? "--accent-2" : "--accent"})"/>`).join("")}
        <circle class="nucleus" r="${nucleusR}" fill="url(#${id}g)"/>
      </svg>`;
    this.orbits = [...host.querySelectorAll(".orbit")];
    this.electrons = [...host.querySelectorAll(".electron")];
    this.glow = host.querySelector(".glow");
    this.nucleus = host.querySelector(".nucleus");
    this.nucleusR = nucleusR;
    this.glowScale = small ? 0.45 : 1;
    this.phase = [0, 2.1, 4.2];
    this.tilt = Math.random() * Math.PI;
    this.frozen = false;
    this.draw({ ...STATE_PARAMS.idle, pulse: 1 }, 0);
  }

  draw(p, dt) {
    this.tilt += dt * p.precess;
    for (let i = 0; i < 3; i++) {
      this.phase[i] += dt * p.speed * (1 + i * 0.17);
      const a = this.tilt + (i * Math.PI) / 3;
      const rx = 46 * p.spread, ry = 17 * p.spread;
      const orbit = this.orbits[i];
      orbit.setAttribute("rx", rx.toFixed(2));
      orbit.setAttribute("ry", ry.toFixed(2));
      orbit.setAttribute("transform", `rotate(${((a * 180) / Math.PI).toFixed(2)})`);
      const x0 = rx * Math.cos(this.phase[i]), y0 = ry * Math.sin(this.phase[i]);
      this.electrons[i].setAttribute("cx", (x0 * Math.cos(a) - y0 * Math.sin(a)).toFixed(2));
      this.electrons[i].setAttribute("cy", (x0 * Math.sin(a) + y0 * Math.cos(a)).toFixed(2));
    }
    this.nucleus.setAttribute("r", (this.nucleusR * p.pulse).toFixed(2));
    this.glow.setAttribute("r", (this.nucleusR * 2 * p.pulse).toFixed(2));
    this.glow.setAttribute("opacity", (p.glow * this.glowScale).toFixed(2));
  }
}

const atoms = new Set();
function mountAtoms(root) {
  root.querySelectorAll("[data-atom]").forEach(host => {
    if (!host.atom) { host.atom = new Atom(host); atoms.add(host.atom); }
  });
}

/* ================= State + animation loop ================= */

let state = "idle";
function setState(next) {
  state = next;
  document.body.dataset.state = next;
}

const current = { ...STATE_PARAMS.idle, pulse: 1 };
let level = 0; // smoothed 0..1 audio level from mic or reply
let lastT = performance.now();

function frame(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  const t = now / 1000;

  const analyser = state === "listening" ? audio.mic : state === "speaking" ? audio.reply : null;
  const raw = analyser ? readLevel(analyser) : 0;
  level += (raw - level) * Math.min(1, dt * 12);
  els.micBtn.style.setProperty("--level", state === "listening" ? level.toFixed(3) : "0");

  const base = STATE_PARAMS[state];
  const target = { ...base, pulse: 1 };
  if (state === "idle") target.pulse = 1 + 0.045 * Math.sin(t * 1.8);
  if (state === "thinking") target.pulse = 1 + 0.08 * Math.sin(t * 7);
  if (state === "listening") { target.spread = 1 + level * 0.35; target.pulse = 1 + level * 0.3; }
  if (state === "speaking") { target.spread = 1 + level * 0.12; target.pulse = 1 + level * 0.5; }
  if (reducedMotion) { target.speed *= 0.2; target.precess *= 0.2; }

  const k = Math.min(1, dt * 6);
  for (const key of Object.keys(target)) current[key] += (target[key] - current[key]) * k;
  for (const atom of atoms) if (!atom.frozen) atom.draw(current, dt);
  requestAnimationFrame(frame);
}

/* ================= Audio: levels, recording, playback ================= */

const audio = { ctx: null, mic: null, reply: null, replySource: null };

function ensureAudioContext() {
  if (!audio.ctx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    audio.ctx = new Ctx();
  }
  if (audio.ctx.state === "suspended") audio.ctx.resume();
  if (!audio.replySource) {
    // Route the reply <audio> through an analyser so the atom can pulse with it.
    audio.replySource = audio.ctx.createMediaElementSource(els.player);
    audio.reply = audio.ctx.createAnalyser();
    audio.reply.fftSize = 512;
    audio.replySource.connect(audio.reply);
    audio.reply.connect(audio.ctx.destination);
  }
  return audio.ctx;
}

const levelBuf = new Uint8Array(512);
function readLevel(analyser) {
  analyser.getByteTimeDomainData(levelBuf);
  let sum = 0;
  for (let i = 0; i < analyser.fftSize; i++) { const v = (levelBuf[i] - 128) / 128; sum += v * v; }
  return Math.min(1, Math.sqrt(sum / analyser.fftSize) * 4.5);
}

const rec = { recorder: null, stream: null, chunks: [], started: 0, timer: null, cancelled: false };
const MAX_RECORD_MS = 30000;

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) return micUnavailable();
  stopPlayback();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  } catch (err) {
    return micUnavailable(err);
  }
  const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"]
    .find(t => MediaRecorder.isTypeSupported(t));
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  Object.assign(rec, { recorder, stream, chunks: [], started: performance.now(), cancelled: false });

  const ctx = ensureAudioContext();
  if (ctx) {
    audio.mic = ctx.createAnalyser();
    audio.mic.fftSize = 512;
    ctx.createMediaStreamSource(stream).connect(audio.mic);
  }

  recorder.ondataavailable = e => { if (e.data.size) rec.chunks.push(e.data); };
  recorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    clearInterval(rec.timer);
    audio.mic = null;
    const duration = performance.now() - rec.started;
    if (state === "listening") setState("idle");
    els.micBtn.setAttribute("aria-pressed", "false");
    setStatus("");
    if (rec.cancelled) return;
    if (duration < 500) return toast("That was too short — hold on a moment longer next time.");
    const type = recorder.mimeType || "audio/webm";
    const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
    ask({ blob: new Blob(rec.chunks, { type }), filename: `voice.${ext}` });
  };

  recorder.start(250);
  setState("listening");
  els.micBtn.setAttribute("aria-pressed", "true");
  const tick = () => {
    const s = Math.floor((performance.now() - rec.started) / 1000);
    setStatus(`<span class="dot"></span>Listening… tap the mic to send <span class="timer">0:${String(s).padStart(2, "0")}</span> · Esc to cancel`, true);
    if (performance.now() - rec.started > MAX_RECORD_MS) stopRecording();
  };
  tick();
  rec.timer = setInterval(tick, 250);
}

function stopRecording(cancel = false) {
  if (rec.recorder?.state === "recording") { rec.cancelled = cancel; rec.recorder.stop(); }
}

function micUnavailable(err) {
  const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
  toast(denied ? "Microphone access was blocked — type your question instead."
               : "No microphone available here — type your question instead.");
  setStatus("Voice input unavailable. You can still type, or upload an audio file with the clip.");
  els.input.focus();
}

function play(url) {
  ensureAudioContext();
  els.player.src = url;
  els.player.play().catch(() => toast("Press play on the answer to hear it."));
}
function stopPlayback() {
  if (!els.player.paused) els.player.pause();
}
els.player.addEventListener("play", () => setState("speaking"));
["pause", "ended"].forEach(ev => els.player.addEventListener(ev, () => { if (state === "speaking") setState("idle"); }));

/* ================= Conversation ================= */

let busy = false;
let voiceOn = store.get("vaak-voice") !== "off";

// One conversation per page: the server keeps its memory under this id.
const newSessionId = () =>
  crypto.randomUUID?.() ?? `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
let sessionId = newSessionId();

async function ask({ text, blob, filename }) {
  if (busy) return;
  stopPlayback();
  document.body.classList.add("has-thread");

  const userLi = addUserMessage(text ?? "Voice message", Boolean(blob));
  const botLi = addAssistantMessage();
  busy = true;
  updateSendButton();
  setState("thinking");
  setStatus(blob ? "Transcribing and thinking…" : "");

  try {
    let res;
    if (blob) {
      const form = new FormData();
      form.append("file", blob, filename);
      form.append("speak", String(voiceOn));
      form.append("session_id", sessionId);
      res = await fetch("/api/ask/audio", { method: "POST", body: form });
    } else {
      res = await fetch("/api/ask/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, speak: voiceOn, session_id: sessionId }),
      });
    }
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      if (data.stage === "stt") {
        // Voice fallback: keep the conversation going by typing.
        userLi.classList.add("failed");
        $(".text", userLi).textContent = "Couldn't make out that audio.";
        botLi.remove();
        toast("Couldn't understand the audio — type your question instead.");
        setStatus("Voice didn't work this time. Type your question below.");
        els.input.focus();
        return;
      }
      throw new Error(data.detail || data.error || `Request failed (${res.status})`);
    }

    if (data.transcript) {
      $(".text", userLi).textContent = data.transcript.text;
      $(".heard-label", userLi).textContent = `Heard · ${data.language_label}`;
    }
    renderAnswer(botLi, data);
    setStatus("");
    if (data.audio_url && voiceOn) play(data.audio_url);
  } catch (err) {
    renderError(botLi, err.message || "Something went wrong.");
    setStatus("");
  } finally {
    busy = false;
    if (state === "thinking") setState("idle");
    updateSendButton();
    scrollToEnd();
  }
}

function addUserMessage(text, isVoice) {
  const li = $("#tplUser").content.firstElementChild.cloneNode(true);
  $(".text", li).textContent = text;
  $(".heard", li).hidden = !isVoice;
  els.thread.append(li);
  scrollToEnd();
  return li;
}

function addAssistantMessage() {
  const li = $("#tplAssistant").content.firstElementChild.cloneNode(true);
  els.thread.append(li);
  mountAtoms(li);
  scrollToEnd();
  return li;
}

const ICON_TOOL = '<svg class="icon" viewBox="0 0 24 24"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.4-.6-.6-2.4z"/></svg>';
const ICON_CHEVRON = '<svg class="icon tool-chevron" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>';
const ICON_CONTEXT = '<svg class="icon" viewBox="0 0 24 24"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>';
const ICON_PLAY = '<svg class="icon" viewBox="0 0 24 24"><path d="M11 5 6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/></svg>';

function renderAnswer(li, data) {
  li.classList.remove("pending");
  freezeAtom(li);
  $(".answer", li).textContent = data.answer || "(no answer)";

  const tools = $(".tools", li);
  for (const call of data.tool_calls) {
    const res = call.result || {};
    const primary = res.error ?? res.result ?? res.formula ?? JSON.stringify(res);
    const args = Object.values(call.arguments || {}).join(", ");
    const d = document.createElement("details");
    d.className = "tool" + (res.error ? " error" : "");
    d.innerHTML = `<summary><span class="tool-icon">${ICON_TOOL}</span><span class="tool-name"></span><span class="tool-value"></span>${ICON_CHEVRON}</summary><pre></pre>`;
    $(".tool-name", d).textContent = call.name;
    $(".tool-value", d).textContent = `${args} → ${primary}`;
    $("pre", d).textContent =
      `arguments  ${JSON.stringify(call.arguments, null, 2)}\nresult     ${JSON.stringify(res, null, 2)}`;
    tools.append(d);
  }

  const meta = $(".meta", li);
  meta.append(chip(data.language_label));
  meta.append(chip(data.route === "tool" ? "Used a tool" : "Direct answer", data.route === "tool" ? "route-tool" : ""));
  const earlier = (data.turns_in_memory || 0) - 1;
  if (earlier > 0) {
    const c = chip(`Used context from ${earlier} earlier ${earlier === 1 ? "turn" : "turns"}`, "context");
    c.insertAdjacentHTML("afterbegin", ICON_CONTEXT);
    c.title = "Vaak remembers this conversation. Use New chat to start fresh.";
    meta.append(c);
  }
  const t = data.timings_ms || {};
  const parts = [["stt", "STT"], ["agent", "Agent"], ["tts", "TTS"]]
    .filter(([k]) => t[k] != null).map(([k, label]) => `${label} ${(t[k] / 1000).toFixed(1)}s`);
  const timing = document.createElement("span");
  timing.className = "timing";
  timing.textContent = parts.join(" · ");
  timing.title = `Total ${(t.total / 1000).toFixed(1)}s · ${data.model}`;
  meta.append(timing);

  if (data.audio_url) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "replay";
    btn.innerHTML = `${ICON_PLAY}<span>Play</span>`;
    btn.addEventListener("click", () => play(data.audio_url));
    meta.append(btn);
  }

  if (data.errors?.length) {
    const warn = $(".warn", li);
    warn.hidden = false;
    warn.textContent = "Couldn't generate the spoken reply, so here's the text.";
    warn.title = data.errors.join("\n");
  }
}

function renderError(li, message) {
  li.classList.remove("pending");
  freezeAtom(li);
  const warn = $(".warn", li);
  warn.hidden = false;
  warn.textContent = message;
}

function freezeAtom(li) {
  const host = $("[data-atom]", li);
  if (host?.atom) host.atom.frozen = true;
}

function chip(text, cls = "") {
  const s = document.createElement("span");
  s.className = `chip ${cls}`.trim();
  s.textContent = text;
  return s;
}

function newChat() {
  if (busy) return;
  stopRecording(true);
  stopPlayback();
  fetch(`/api/session/${sessionId}`, { method: "DELETE" }).catch(() => {});
  sessionId = newSessionId();
  els.thread.replaceChildren();
  for (const atom of atoms) if (!atom.host.isConnected) atoms.delete(atom);
  document.body.classList.remove("has-thread");
  setStatus("");
  els.input.value = "";
  autoGrow();
  updateSendButton();
  els.input.focus();
}

/* ================= UI helpers ================= */

function setStatus(content, isHTML = false) {
  if (isHTML) els.status.innerHTML = content; else els.status.textContent = content;
}

let toastTimer;
function toast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove("show"), 4200);
}

function scrollToEnd() {
  requestAnimationFrame(() => els.scroller.scrollTo({ top: els.scroller.scrollHeight, behavior: "smooth" }));
}

function updateSendButton() {
  els.sendBtn.disabled = busy || !els.input.value.trim();
}

function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 180)}px`;
}

function submitText() {
  const text = els.input.value.trim();
  if (!text || busy) return;
  els.input.value = "";
  autoGrow();
  updateSendButton();
  ask({ text });
}

function applyVoiceToggle() {
  els.voiceToggle.setAttribute("aria-pressed", String(voiceOn));
  els.voiceToggle.title = voiceOn ? "Spoken replies on" : "Spoken replies off";
}

/* ================= Wiring ================= */

els.composer.addEventListener("submit", e => { e.preventDefault(); submitText(); });
els.input.addEventListener("input", () => { autoGrow(); updateSendButton(); });
els.input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submitText(); }
});

els.micBtn.addEventListener("click", () => {
  if (state === "listening") stopRecording();
  else if (!busy) startRecording();
});

els.attachBtn.addEventListener("click", () => els.fileInput.click());
els.newChat.addEventListener("click", newChat);
els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files[0];
  els.fileInput.value = "";
  if (file) ask({ blob: file, filename: file.name });
});

document.querySelectorAll(".suggestion").forEach(btn =>
  btn.addEventListener("click", () => ask({ text: btn.dataset.q })));

els.voiceToggle.addEventListener("click", () => {
  voiceOn = !voiceOn;
  store.set("vaak-voice", voiceOn ? "on" : "off");
  applyVoiceToggle();
  if (!voiceOn) stopPlayback();
  toast(voiceOn ? "Spoken replies on" : "Spoken replies off — text only");
});

els.themeToggle.addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  store.set("vaak-theme", root.dataset.theme);
});

document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  if (state === "listening") { stopRecording(true); toast("Recording cancelled"); }
  else stopPlayback();
});

// Browsers only allow audio after a user gesture; unlock it on the first one.
document.addEventListener("pointerdown", () => ensureAudioContext(), { once: true });

fetch("/api/info").then(r => r.json()).then(info => {
  els.modelPill.textContent = `${info.model} · ${info.tools.length} tools`;
  els.modelPill.hidden = false;
  els.footnote.textContent = `Direct answers come from ${info.model} and can be wrong. Tool results (${info.tools.join(", ")}) are computed exactly.`;
}).catch(() => {});

applyVoiceToggle();
mountAtoms(document);
requestAnimationFrame(frame);
