"""Deployment safeguards: /health, HOST/PORT, rate limits and reply-audio cleanup."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import web.server as server
from web.ops import AudioJanitor, RateLimiter, client_ip


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_health(monkeypatch):
    assert TestClient(server.app).get("/health").json() == {"status": "ok"}


def test_limits_are_off_by_default():
    limiter = RateLimiter()
    assert not limiter.enabled
    assert all(limiter.check("1.2.3.4").allowed for _ in range(1000))
    assert not AudioJanitor("out").enabled


def test_per_minute_limit_uses_a_sliding_window():
    clock = Clock()
    limiter = RateLimiter(per_minute=2, clock=clock)
    assert limiter.check("a").allowed and limiter.check("a").allowed
    blocked = limiter.check("a")
    assert not blocked.allowed and 1 <= blocked.retry_after_s <= 61
    assert limiter.check("b").allowed  # other clients are unaffected
    clock.t += 61
    assert limiter.check("a").allowed


def test_daily_limits_per_client_and_in_total():
    clock = Clock()
    limiter = RateLimiter(per_day=2, daily_total=3, clock=clock)
    assert limiter.check("a").allowed and limiter.check("a").allowed
    assert "today" in limiter.check("a").message
    assert limiter.check("b").allowed
    assert "daily question limit" in limiter.check("c").message  # total cap reached
    clock.t += 86400  # next UTC day resets both
    assert limiter.check("a").allowed and limiter.check("c").allowed


def test_client_ip_only_trusts_configured_proxies():
    assert client_ip("10.0.0.1", "6.6.6.6, 1.2.3.4", 0) == "10.0.0.1"
    assert client_ip("10.0.0.1", "6.6.6.6, 1.2.3.4", 1) == "1.2.3.4"  # spoofed left entry ignored
    assert client_ip("10.0.0.1", "6.6.6.6, 1.2.3.4", 2) == "6.6.6.6"
    assert client_ip(None, None, 1) == "unknown"


def test_ask_routes_return_429_when_limited(monkeypatch):
    monkeypatch.setattr(server, "limiter", RateLimiter(per_minute=1))
    client = TestClient(server.app)
    assert client.post("/api/ask/text", json={"text": "  "}).status_code == 400  # counted, then rejected
    res = client.post("/api/ask/text", json={"text": "  "})
    assert res.status_code == 429
    assert res.json()["stage"] == "rate_limit" and int(res.headers["Retry-After"]) >= 1
    assert client.get("/api/info").status_code == 200  # other routes are never limited
    assert client.get("/health").status_code == 200


def _reply(dir_, name, age_s, now):
    path = dir_ / name
    path.write_bytes(b"x")
    os.utime(path, (now - age_s, now - age_s))
    return path


def test_janitor_deletes_old_and_excess_replies_only(tmp_path):
    now = 1_800_000_000.0
    old = _reply(tmp_path, "reply_00000001.wav", 3600, now)
    new = [_reply(tmp_path, f"reply_0000001{i}.wav", 60 * i, now) for i in range(4)]
    keep = tmp_path / "te_question.wav"
    keep.write_bytes(b"x")
    os.utime(keep, (now - 99999, now - 99999))
    janitor = AudioJanitor(tmp_path, ttl_min=30, max_files=3, clock=lambda: now)
    assert janitor.run(now) == 2
    assert not old.exists() and not new[3].exists()
    assert all(p.exists() for p in new[:3]) and keep.exists()


def test_janitor_runs_at_most_once_per_interval(tmp_path):
    clock = Clock()
    janitor = AudioJanitor(tmp_path, max_files=1, interval_s=60, clock=clock)
    for i in range(3):
        (tmp_path / f"reply_0000000{i}.wav").write_bytes(b"x")
    assert janitor.maybe_run() == 2
    (tmp_path / "reply_000000aa.wav").write_bytes(b"x")
    assert janitor.maybe_run() == 0
    clock.t += 61
    assert janitor.maybe_run() == 1


@pytest.mark.parametrize("env,expected", [({}, ("127.0.0.1", 8765)), ({"HOST": "0.0.0.0", "PORT": "7860"}, ("0.0.0.0", 7860))])
def test_host_and_port_come_from_env(monkeypatch, env, expected):
    for k in ("HOST", "PORT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    seen = {}
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, **kw: seen.update(host=host, port=port))
    monkeypatch.setattr("sys.argv", ["web.server"])
    server.main()
    assert (seen["host"], seen["port"]) == expected
