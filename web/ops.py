"""Opt-in safeguards for running the web app in public: rate limits and reply-audio cleanup.

Everything here is off unless its environment variable is set, so a local
`python -m web.server` behaves exactly as before.

Rate limits (requests to /api/ask/* only; 0 or unset = no limit):
  VAAK_RATE_LIMIT_PER_MIN   per client IP, sliding 60-second window
  VAAK_RATE_LIMIT_PER_DAY   per client IP, per UTC day
  VAAK_DAILY_CAP_TOTAL      all clients together, per UTC day: the hard guard on API spend
  VAAK_TRUST_PROXY          number of reverse proxies in front of the app (e.g. 1 on Render or
                            Hugging Face). The client IP is then read from X-Forwarded-For;
                            with 0 (default) the socket address is used.

Reply-audio cleanup (generated reply_*.wav files in the output folder; 0 or unset = keep all):
  VAAK_AUDIO_TTL_MIN        delete replies older than this many minutes
  VAAK_AUDIO_MAX_FILES      keep at most this many replies (newest kept)

Limits are kept in memory, so they reset when the process restarts. That is enough for a
single-instance demo; a multi-instance deployment would need a shared store instead.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPLY_FILE = re.compile(r"^reply_[0-9a-f]{8}\.wav$")


def env_int(name: str, default: int = 0) -> int:
    try:
        return max(int(os.getenv(name, default) or 0), 0)
    except ValueError:
        return default


# ---------- rate limiting ----------

@dataclass
class Decision:
    allowed: bool
    message: str = ""
    retry_after_s: int = 0


class RateLimiter:
    def __init__(self, per_minute: int = 0, per_day: int = 0, daily_total: int = 0, clock=time.time) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self.daily_total = daily_total
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        self._daily: dict[str, int] = defaultdict(int)
        self._total = 0

    @classmethod
    def from_env(cls) -> RateLimiter:
        return cls(env_int("VAAK_RATE_LIMIT_PER_MIN"), env_int("VAAK_RATE_LIMIT_PER_DAY"),
                   env_int("VAAK_DAILY_CAP_TOTAL"))

    @property
    def enabled(self) -> bool:
        return bool(self.per_minute or self.per_day or self.daily_total)

    def check(self, client: str) -> Decision:
        """Count one request from `client` if it is within every limit."""
        if not self.enabled:
            return Decision(True)
        now = self._clock()
        with self._lock:
            self._roll_day(now)
            if self.daily_total and self._total >= self.daily_total:
                return Decision(False, "The demo has reached its daily question limit. Please try again tomorrow.",
                                self._seconds_to_midnight(now))
            if self.per_day and self._daily[client] >= self.per_day:
                return Decision(False, f"You've reached the limit of {self.per_day} questions today. "
                                "Please try again tomorrow.", self._seconds_to_midnight(now))
            recent = self._recent[client]
            while recent and now - recent[0] >= 60:
                recent.popleft()
            if self.per_minute and len(recent) >= self.per_minute:
                wait = max(int(60 - (now - recent[0])) + 1, 1)
                return Decision(False, f"Too many questions in a minute. Please wait {wait} seconds.", wait)
            recent.append(now)
            self._daily[client] += 1
            self._total += 1
            if len(self._recent) > 10_000:  # bound memory: drop idle clients
                for key in [k for k, q in self._recent.items() if not q or now - q[-1] >= 60]:
                    del self._recent[key]
            return Decision(True)

    def _roll_day(self, now: float) -> None:
        day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        if day != self._day:
            self._day, self._daily, self._total = day, defaultdict(int), 0

    @staticmethod
    def _seconds_to_midnight(now: float) -> int:
        return int(86400 - now % 86400) + 1


def client_ip(peer: str | None, forwarded_for: str | None, trusted_proxies: int) -> str:
    """The client address: the socket peer, or the X-Forwarded-For entry added by the
    outermost trusted proxy (counting from the right, so clients can't spoof it)."""
    if trusted_proxies and forwarded_for:
        hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
        if hops:
            return hops[-min(trusted_proxies, len(hops))]
    return peer or "unknown"


# ---------- reply-audio cleanup ----------

class AudioJanitor:
    """Deletes old reply_*.wav files; runs at most once per `interval_s`."""

    def __init__(self, out_dir: Path, ttl_min: int = 0, max_files: int = 0, interval_s: float = 60,
                 clock=time.time) -> None:
        self.out_dir = Path(out_dir)
        self.ttl_min = ttl_min
        self.max_files = max_files
        self.interval_s = interval_s
        self._clock = clock
        self._last = float("-inf")
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, out_dir: Path) -> AudioJanitor:
        return cls(out_dir, env_int("VAAK_AUDIO_TTL_MIN"), env_int("VAAK_AUDIO_MAX_FILES"))

    @property
    def enabled(self) -> bool:
        return bool(self.ttl_min or self.max_files)

    def maybe_run(self) -> int:
        if not self.enabled:
            return 0
        now = self._clock()
        with self._lock:
            if now - self._last < self.interval_s:
                return 0
            self._last = now
        return self.run(now)

    def run(self, now: float | None = None) -> int:
        """Delete expired replies and the oldest ones beyond max_files. Returns how many."""
        now = self._clock() if now is None else now
        if not self.out_dir.is_dir():
            return 0
        replies = []
        for path in self.out_dir.iterdir():
            if REPLY_FILE.match(path.name) and path.is_file():
                try:
                    replies.append((path.stat().st_mtime, path))
                except FileNotFoundError:
                    pass
        replies.sort(reverse=True)  # newest first
        doomed = []
        for i, (mtime, path) in enumerate(replies):
            too_old = self.ttl_min and now - mtime > self.ttl_min * 60
            too_many = self.max_files and i >= self.max_files
            if too_old or too_many:
                doomed.append(path)
        for path in doomed:
            path.unlink(missing_ok=True)
        return len(doomed)
