"""Shared test fixtures: a scripted stand-in for the LLM so no test needs keys or network."""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest

# web.server builds its Agent at import time, which needs a key to construct the client.
# Creating the client makes no network call; every test swaps it for FakeLLM below.
os.environ.setdefault("SARVAM_API_KEY", "test-key-not-real")
os.environ["VAAK_LLM_PROVIDER"] = "sarvam"


def _message(content: str | None = None, tool_calls: list | None = None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _tool_call(name: str, args: dict, n: int = 0):
    return SimpleNamespace(id=f"call_{n}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeLLM:
    """Mimics client.chat.completions.create with a few fixed behaviours.

    - "A times B"               -> calls calculate, then answers with the tool result
    - "N kg in pounds"          -> calls convert_units, then answers with the tool result
    - after a tool result       -> "The answer is <result>." (or a wrong number if misquote=True,
                                   until the grounding check asks for a rewrite)
    - anything else             -> a direct answer with no tool call
    """

    def __init__(self, misquote: bool = False) -> None:
        self.misquote = misquote
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, tools=None, **kwargs):
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        last = messages[-1]
        if last["role"] == "tool":
            result = json.loads(last["content"]).get("result")
            if self.misquote and not any("Check failed" in str(m.get("content")) for m in messages):
                return _message(f"The answer is {result + 1}.")
            return _message(f"The answer is {result}.")
        if last["role"] == "user" and "Check failed" in last["content"]:
            tool_results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            return _message(f"The answer is {tool_results[-1]['result']}.")

        query = last["content"]
        if m := re.search(r"(\d+) times (\d+)", query):
            return _message(tool_calls=[_tool_call("calculate", {"expression": f"{m[1]}*{m[2]}"})])
        if m := re.search(r"(\d+) kg in pounds", query):
            return _message(tool_calls=[_tool_call(
                "convert_units", {"value": float(m[1]), "from_unit": "kg", "to_unit": "lb"})])
        return _message("Rabindranath Tagore wrote it.")


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def llm_factory():
    """Build a FakeLLM with options, e.g. llm_factory(misquote=True)."""
    return FakeLLM
