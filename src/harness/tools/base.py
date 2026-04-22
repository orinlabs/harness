"""Tool contract.

One protocol, two origins:
  - Built-in tools: Python classes in `harness/tools/` that satisfy `Tool`. They
    may touch harness internals (memory, runtime_api, context).
  - External tools: `ExternalTool(spec)` wraps an `ExternalToolSpec` and
    satisfies `Tool` by POSTing args to `spec.url`.

Both go into a single `tool_map: dict[str, Tool]` assembled by `build_tool_map`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from harness.context import RunContext


@dataclass
class ToolResult:
    text: str
    images: list[str] | None = None  # base64-encoded strings


@dataclass
class ToolSchema:
    """OpenAI-format tool definition for the `tools` param on an LLM call."""

    name: str
    description: str
    parameters: dict  # JSON Schema

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict

    @property
    def schema(self) -> ToolSchema: ...

    def call(self, args: dict, ctx: RunContext) -> ToolResult: ...
