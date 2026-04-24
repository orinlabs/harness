"""Assemble the tool map for a Harness instance.

Built-in tools first, then every external tool from every adapter. Name
collisions raise — no silent override.
"""
from __future__ import annotations

from harness.config import AdapterConfig, ExternalToolSpec
from harness.tools.base import Tool
from harness.tools.external import ExternalTool
from harness.tools.sleep import SleepTool


def _builtins() -> list[Tool]:
    return [SleepTool()]


def build_tool_map(adapters: list[AdapterConfig]) -> dict[str, Tool]:
    tool_map: dict[str, Tool] = {}

    for tool in _builtins():
        if tool.name in tool_map:
            raise ValueError(f"duplicate built-in tool: {tool.name!r}")
        tool_map[tool.name] = tool

    for adapter in adapters:
        for entry in adapter.tools:
            # Eval-time fake adapters pass already-instantiated Tool objects
            # instead of ExternalToolSpec. They satisfy the Tool protocol and
            # are dispatched in-process; no HTTP wrapping needed.
            if isinstance(entry, ExternalToolSpec):
                if entry.name in tool_map:
                    raise ValueError(
                        f"tool name collision: {entry.name!r} already registered "
                        f"(adapter={adapter.name!r})"
                    )
                tool_map[entry.name] = ExternalTool(entry)
            else:
                name = getattr(entry, "name", None)
                call = getattr(entry, "call", None)
                if not (isinstance(name, str) and callable(call)):
                    raise TypeError(
                        f"adapter {adapter.name!r} has a tool entry that is neither "
                        f"ExternalToolSpec nor a Tool instance: {entry!r}"
                    )
                if name in tool_map:
                    raise ValueError(
                        f"tool name collision: {name!r} already registered "
                        f"(adapter={adapter.name!r})"
                    )
                tool_map[name] = entry

    return tool_map
