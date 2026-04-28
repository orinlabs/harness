"""Assemble the tool map for a Harness instance.

Built-in tools first, then every external tool from every adapter. Name
collisions raise — no silent override.
"""
from __future__ import annotations

import logging

from harness.config import AdapterConfig, ExternalToolSpec
from harness.tools.base import Tool
from harness.tools.external import ExternalTool
from harness.tools.sleep import SleepTool

logger = logging.getLogger(__name__)


def _builtins() -> list[Tool]:
    return [SleepTool()]


def build_tool_map(adapters: list[AdapterConfig]) -> dict[str, Tool]:
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

    logger.info(
        "build_tool_map: received %d adapter(s): %s",
        len(adapters),
        [a.name for a in adapters],
    )

    for adapter in adapters:
        adapter_tool_names: list[str] = []
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
                adapter_tool_names.append(entry.name)
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
                adapter_tool_names.append(name)
        logger.info(
            "build_tool_map: adapter=%r contributed %d tool(s): %s",
            adapter.name,
            len(adapter_tool_names),
            adapter_tool_names,
        )

    logger.info(
        "build_tool_map: final tool_map has %d tool(s): %s",
        len(tool_map),
        list(tool_map.keys()),
    )
    return tool_map
