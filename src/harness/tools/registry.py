"""Assemble the tool map for a Harness instance.

Built-in tools first, then every external tool from every adapter. Name
collisions raise — no silent override.
"""
from __future__ import annotations

from harness.config import AdapterConfig
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
        for spec in adapter.tools:
            if spec.name in tool_map:
                raise ValueError(
                    f"tool name collision: {spec.name!r} already registered "
                    f"(adapter={adapter.name!r})"
                )
            tool_map[spec.name] = ExternalTool(spec)

    return tool_map
