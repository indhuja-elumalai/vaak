"""FastAPI routes, end to end through the real agent loop, tools and grounding.

The LLM, STT and TTS are replaced with fakes, so these tests need no keys or network.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import core.pipeline as pipeline_mod
import web.server as server
from voice_io.stt import Transcript
from voice_io.tts import SpeechResult

SESSION = "test-session-0001"


@pytest.fixture
def client(monkeypatch, tmp_path, fake_llm):
    monkeypatch.setattr(server.agent, "client", fake_llm)
    monkeypatch.setattr(server, "OUT_DIR", tmp_path)
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    server.sessions.delete(SESSION)
    return TestClient(server.app)


@pytest.fixture
def fake_tts(monkeypatch):
    def synthesize(text, out_path, language_code="hi-IN", provider=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RIFF0000WAVEfmt ")
        return SpeechResult(path=out_path, provider="fake", latency_ms=12.0)

    monkeypatch.setattr(pipeline_mod, "synthesize", synthesize)


def fake_stt(monkeypatch, text="What is 37 times 48?", language="en-IN", fail=False):
    def transcribe(path, language_code="unknown", provider=None):
        if fail:
            raise RuntimeError("provider down")
        return Transcript(text=text, language_code=language, provider="fake", latency_ms=40.0)

    monkeypatch.setattr(pipeline_mod, "transcribe", transcribe)


def ask(client, text, **extra):
    return client.post("/api/ask/text", json={"text": text, "speak": False, **extra})


def test_info_lists_model_tools_and_languages(client):
    body = client.get("/api/info").json()
    assert body["tools"] == ["calculate", "convert_units", "lookup_formula"]
    assert body["model"]
    assert body["languages"] == {"en-IN": "English", "hi-IN": "Hindi", "ta-IN": "Tamil"}


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert 'id="micBtn"' in res.text and "app.js" in res.text


def test_tool_question_routes_to_tool_and_is_verified(client):
    res = ask(client, "What is 37 times 48?")
    assert res.status_code == 200
    body = res.json()
    assert body["route"] == "tool"
    assert body["tool_calls"][0]["name"] == "calculate"
    assert body["tool_calls"][0]["result"]["result"] == 1776
    assert "1776" in body["answer"]
    assert body["grounding"]["verdict"] == "verified"
    assert [s["kind"] for s in body["trace"]] == ["llm", "tool", "llm", "check"]
    assert body["input_mode"] == "text" and body["transcript"] is None
    assert body["audio_url"] is None
    assert {"agent", "total"} <= set(body["timings_ms"])


def test_direct_question_is_answered_without_tools(client):
    body = ask(client, "Who wrote India's national anthem?").json()
    assert body["route"] == "direct" and body["tool_calls"] == []
    assert body["grounding"]["verdict"] == "unverified"
    assert body["language_label"] == "English"


def test_language_label_for_code_mixed_text(client):
    assert ask(client, "144 ka square root kya hai? 12 times 12").json()["language_label"] == "Hinglish"


def test_empty_question_is_rejected(client):
    assert ask(client, "   ").status_code == 400


def test_invalid_session_id_is_rejected(client):
    assert ask(client, "hi", session_id="bad id!").status_code == 400


def test_unsupported_language_gets_a_clear_422(client):
    res = ask(client, "నమస్కారం, ఇది ఏమిటి?")
    assert res.status_code == 422
    body = res.json()
    assert body["stage"] == "language" and body["detected"] == "te-IN"


def test_follow_ups_share_memory_until_new_chat(client):
    first = ask(client, "25 kg in pounds?", session_id=SESSION).json()
    assert first["turns_in_memory"] == 1
    second = ask(client, "Who wrote India's national anthem?", session_id=SESSION).json()
    assert second["turns_in_memory"] == 2
    assert client.delete(f"/api/session/{SESSION}").json() == {"ok": True}
    assert ask(client, "What is 2 times 3?", session_id=SESSION).json()["turns_in_memory"] == 1


def test_no_session_means_no_memory(client):
    assert ask(client, "What is 2 times 3?").json()["turns_in_memory"] == 0


def test_spoken_reply_is_saved_and_served(client, fake_tts):
    body = client.post("/api/ask/text", json={"text": "What is 6 times 7?", "speak": True}).json()
    assert body["audio_url"].startswith("/api/audio/reply_")
    assert body["trace"][-1]["kind"] == "tts"
    res = client.get(body["audio_url"])
    assert res.status_code == 200 and res.headers["content-type"] == "audio/wav"


def test_audio_route_rejects_unknown_names(client):
    assert client.get("/api/audio/../../.env").status_code == 404
    assert client.get("/api/audio/reply_00000000.wav").status_code == 404


def test_voice_question_runs_stt_then_agent(client, monkeypatch, tmp_path):
    fake_stt(monkeypatch)
    res = client.post("/api/ask/audio", files={"file": ("q.webm", b"fake-audio", "audio/webm")},
                      data={"speak": "false", "session_id": SESSION})
    assert res.status_code == 200
    body = res.json()
    assert body["input_mode"] == "audio"
    assert body["transcript"] == {"text": "What is 37 times 48?", "language": "en-IN", "provider": "fake"}
    assert body["trace"][0]["kind"] == "stt"
    assert "stt" in body["timings_ms"]
    assert not any((tmp_path / "uploads").iterdir())  # the upload is deleted after the turn


def test_voice_failure_asks_the_page_to_fall_back_to_typing(client, monkeypatch):
    fake_stt(monkeypatch, fail=True)
    res = client.post("/api/ask/audio", files={"file": ("q.wav", b"x", "audio/wav")})
    assert res.status_code == 422 and res.json()["stage"] == "stt"


def test_voice_in_unsupported_language(client, monkeypatch):
    fake_stt(monkeypatch, text="ఇది ఏమిటి", language="te-IN")
    res = client.post("/api/ask/audio", files={"file": ("q.wav", b"x", "audio/wav")})
    assert res.status_code == 422 and res.json()["stage"] == "language"


def test_audio_upload_validation(client, monkeypatch):
    assert client.post("/api/ask/audio", files={"file": ("q.txt", b"x", "text/plain")}).status_code == 400
    assert client.post("/api/ask/audio", files={"file": ("q.wav", b"x", "audio/wav")},
                       data={"language": "te-IN"}).status_code == 400
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 4)
    assert client.post("/api/ask/audio", files={"file": ("q.wav", b"12345", "audio/wav")}).status_code == 413
