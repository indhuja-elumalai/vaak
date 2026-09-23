"""Conversation memory: the message history the agent sees on follow-up questions.

History is stored as whole turns (user message, any assistant tool calls and tool
results, final answer) so trimming never splits a tool call from its result.
Only the most recent max_turns turns are kept, which bounds prompt size.

This module is domain-agnostic: it must never import from tools/.
"""
from __future__ import annotations

import threading
from collections import OrderedDict, deque

DEFAULT_MAX_TURNS = 8


class Conversation:
    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._turns: deque[list[dict]] = deque(maxlen=max_turns)
        # Serializes turns within one conversation (e.g. two browser requests at once).
        self.lock = threading.Lock()

    def messages(self) -> list[dict]:
        return [m for turn in self._turns for m in turn]

    def add_turn(self, messages: list[dict]) -> None:
        self._turns.append(list(messages))

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


class ConversationStore:
    """In-memory conversations keyed by session id, evicting the least recently used."""

    def __init__(self, max_sessions: int = 500) -> None:
        self._sessions: OrderedDict[str, Conversation] = OrderedDict()
        self._max = max_sessions
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Conversation:
        with self._lock:
            conv = self._sessions.pop(session_id, None) or Conversation()
            self._sessions[session_id] = conv
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
            return conv

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
