"""Tool registry contract and the agent loop (routing, grounding rewrite, memory) with a fake LLM."""
from __future__ import annotations

import pytest

from core.agent_runtime import Agent, load_tools
from core.conversation import Conversation
from core.tool_registry import ToolRegistry


def make_agent(llm) -> Agent:
    registry = ToolRegistry()
    agent = Agent(registry, system_prompt=load_tools("tools.study_tools", registry))
    agent.client = llm
    return agent


# --- registry ---------------------------------------------------------------------

def test_registry_exposes_openai_style_schemas():
    registry = ToolRegistry()
    load_tools("tools.study_tools", registry)
    schemas = registry.schemas()
    assert [s["function"]["name"] for s in schemas] == registry.names()
    for s in schemas:
        assert s["type"] == "function"
        assert s["function"]["description"]
        assert s["function"]["parameters"]["type"] == "object"


def test_registry_rejects_duplicates_and_wraps_plain_results():
    registry = ToolRegistry()
    registry.register_tool("double", {"description": "x2"}, lambda n: n * 2)
    with pytest.raises(ValueError):
        registry.register_tool("double", {"description": "again"}, lambda n: n)
    assert registry.call("double", '{"n": 21}') == {"result": 42}
    assert registry.call("double", {"n": 2}) == {"result": 4}
    assert registry.schemas()[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_registry_turns_failures_into_error_results():
    registry = ToolRegistry()
    registry.register_tool("boom", {"description": "fails"}, lambda: 1 / 0)
    assert "ZeroDivisionError" in registry.call("boom", "{}")["error"]
    assert "error" in registry.call("boom", "{not json")
    assert registry.call("missing", "{}") == {"error": "Unknown tool: missing"}


# --- agent loop ---------------------------------------------------------------------

def test_agent_calls_the_tool_and_verifies_the_answer(llm_factory):
    llm = llm_factory()
    result = make_agent(llm).run("What is 37 times 48?")
    assert result.route == "tool" and result.tools_used == ["calculate"]
    assert result.answer == "The answer is 1776."
    assert result.grounding.verdict == "verified"
    assert result.llm_calls == 2
    assert llm.calls[0]["tools"]  # the LLM was offered the tool schemas


def test_agent_answers_directly_when_no_tool_fits(llm_factory):
    result = make_agent(llm_factory()).run("Who wrote India's national anthem?")
    assert result.route == "direct" and result.grounding.verdict == "unverified"


def test_misquoted_number_is_flagged_and_rewritten(llm_factory):
    result = make_agent(llm_factory(misquote=True)).run("What is 37 times 48?")
    assert result.draft_answer == "The answer is 1777."
    assert result.answer == "The answer is 1776."
    assert result.grounding.verdict == "corrected"
    assert any("Rewrote" in s.label for s in result.trace)


def test_reply_language_instruction_reaches_the_model(llm_factory):
    llm = llm_factory()
    make_agent(llm).run("144 ka square root kya hai?")
    assert "Hinglish" in llm.calls[0]["messages"][0]["content"]


def test_conversation_keeps_tool_traffic_for_follow_ups(llm_factory):
    conv = Conversation()
    llm = llm_factory()
    agent = make_agent(llm)
    agent.run("25 kg in pounds?", conversation=conv)
    agent.run("Who wrote India's national anthem?", conversation=conv)
    roles = [m["role"] for m in conv.messages()]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    # the second request carried the first turn as context
    assert any(m["role"] == "tool" for m in llm.calls[-1]["messages"])
