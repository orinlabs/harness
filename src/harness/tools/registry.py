"""Assemble the tool map for a Harness instance.

Built-in tools first, then every tool from the config. Name collisions raise --
no silent override.
"""

from __future__ import annotations

import logging

from harness.config import ExternalToolSpec
from harness.tools.base import Tool
from harness.tools.external import ExternalTool
from harness.tools.sleep import SleepTool

logger = logging.getLogger(__name__)


def _builtins() -> list[Tool]:
    return [SleepTool()]


def build_tool_map(tools: list[ExternalToolSpec | Tool]) -> dict[str, Tool]:
    tool_map: dict[str, Tool] = {}

    builtin_tools = _builtins()
    for tool in builtin_tools:
        if tool.name in tool_map:
            raise ValueError(f"duplicate built-in tool: {tool.name!r}")
        tool_map[tool.name] = tool
    logger.info(
        "build_tool_map: registered %d built-in tool(s): %s",
        len(builtin_tools),
        [t.name for t in builtin_tools],
    )

    logger.info("build_tool_map: received %d tool(s) from config", len(tools))

    for entry in tools:
        # Eval-time fakes pass already-instantiated Tool objects instead of
        # ExternalToolSpec. They satisfy the Tool protocol and are dispatched
        # in-process; no HTTP wrapping needed.
        if isinstance(entry, ExternalToolSpec):
            if entry.name in tool_map:
                raise ValueError(f"tool name collision: {entry.name!r} already registered")
            tool_map[entry.name] = ExternalTool(entry)
        else:
            name = getattr(entry, "name", None)
            call = getattr(entry, "call", None)
            if not (isinstance(name, str) and callable(call)):
                raise TypeError(
                    f"tools list contains an entry that is neither "
                    f"ExternalToolSpec nor a Tool instance: {entry!r}"
                )
            if name in tool_map:
                raise ValueError(f"tool name collision: {name!r} already registered")
            tool_map[name] = entry

    logger.info(
        "build_tool_map: final tool_map has %d tool(s): %s",
        len(tool_map),
        list(tool_map.keys()),
    )
    return tool_map
