"""Assemble the tool map for a Harness instance.

Built-in tools first, then every tool from the config. Name collisions raise --
no silent override -- with ONE exception: a config tool named ``sleep`` is
captured as the environment's *sleep listener* instead of colliding with the
built-in. The built-in ``SleepTool`` stays the only model-visible ``sleep``
schema; it forwards every sleep call to the captured listener so the
environment can react (e.g. advance a simulated clock / schedule a wake).
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


def build_tool_map(
    tools: list[ExternalToolSpec | Tool],
) -> tuple[dict[str, Tool], Tool | None]:
    """Build the model-visible tool map plus the optional env sleep listener.

    Returns ``(tool_map, env_sleep)``. ``env_sleep`` is the config tool named
    ``"sleep"`` (wrapped as an ``ExternalTool`` when it came in as a spec), or
    ``None`` when the config doesn't register one. It is deliberately NOT in
    ``tool_map`` so it never reaches the model's tool schemas.
    """
    tool_map: dict[str, Tool] = {}
    env_sleep: Tool | None = None

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

    def capture_env_sleep(tool: Tool) -> None:
        nonlocal env_sleep
        if env_sleep is not None:
            raise ValueError("duplicate env sleep listener: 'sleep' registered twice in config")
        env_sleep = tool
        logger.info("build_tool_map: captured env sleep listener (not model-visible)")

    for entry in tools:
        # Eval-time fakes pass already-instantiated Tool objects instead of
        # ExternalToolSpec. They satisfy the Tool protocol and are dispatched
        # in-process; no HTTP wrapping needed.
        if isinstance(entry, ExternalToolSpec):
            if entry.name == SleepTool.name:
                capture_env_sleep(ExternalTool(entry))
                continue
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
            if name == SleepTool.name:
                capture_env_sleep(entry)
                continue
            if name in tool_map:
                raise ValueError(f"tool name collision: {name!r} already registered")
            tool_map[name] = entry

    logger.info(
        "build_tool_map: final tool_map has %d tool(s): %s (env sleep listener: %s)",
        len(tool_map),
        list(tool_map.keys()),
        "yes" if env_sleep is not None else "no",
    )
    return tool_map, env_sleep
