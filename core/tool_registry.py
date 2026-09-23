"""Generic tool registry.

Domain code registers tools here; the agent runtime only ever sees the registry.
This module is domain-agnostic: it must never import from tools/.

    registry = ToolRegistry()
    registry.register_tool(
        "calculate",
        {"description": "...", "parameters": {<JSON schema>}},
        handler,
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the handler's keyword arguments
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register_tool(self, name: str, schema: dict, handler: Callable[..., Any]) -> None:
        """schema: {"description": str, "parameters": <JSON schema object>}"""
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = Tool(
            name=name,
            description=schema["description"],
            parameters=schema.get("parameters", {"type": "object", "properties": {}}),
            handler=handler,
        )

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        """Tool definitions in OpenAI function-calling format (also used by Sarvam)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments: str | dict) -> dict:
        """Run a tool. Never raises: errors come back as {"error": ...} so the LLM can recover."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            args = json.loads(arguments or "{}") if isinstance(arguments, str) else arguments
            result = tool.handler(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return result if isinstance(result, dict) else {"result": result}
