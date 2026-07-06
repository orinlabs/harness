from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_agent_id: ContextVar[str | None] = ContextVar("harness_agent_id", default=None)


def set_agent_id(agent_id: str) -> None:
    _agent_id.set(agent_id)


def get_agent_id() -> str:
    v = _agent_id.get()
    if v is None:
        raise RuntimeError(
            "agent_id not set in context. Call set_agent_id() at the top of Harness.run()."
        )
    return v


@dataclass
class RunContext:
    agent_id: str
    run_id: str
    turn: int = 0
    sleep_requested: bool = False
    # Populated by Harness after building the tool map. Lets built-in tools
    # (e.g. SleepTool) look up and invoke sibling tools -- e.g. sleep calls
    # list_notifications to refuse sleep while attention items are pending.
    # Typed as Any to avoid a context <-> tools import cycle.
    tool_map: dict[str, Any] = field(default_factory=dict)
    # Populated by Harness.__init__ from the injected (or autoconfigured)
    # ``AgentRuntime``. Built-in tools that need platform-scoped operations
    # (currently just SleepTool) invoke them through here. Typed as Any to
    # avoid a context <-> core/runtime import cycle at dataclass-decoration time.
    runtime: Any = None
    # Hard cap on how far into the future the agent may sleep (aware UTC
    # datetime). Populated by Harness.__init__ from --max-utc-sleep.
    # SleepTool clamps any later (or "indefinite") sleep request down to
    # this moment. None means uncapped.
    max_sleep_until: datetime | None = None
    # Agent-local timezone name (e.g. "America/Los_Angeles"). SleepTool
    # treats naive sleep timestamps as local to this zone and keeps
    # user-facing sleep/clamp messages in this zone. None means UTC.
    timezone_name: str | None = None
