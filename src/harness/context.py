from contextvars import ContextVar
from dataclasses import dataclass, field
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
    # The environment's optional sleep listener: a config tool named "sleep",
    # captured by ``build_tool_map`` instead of colliding with the built-in.
    # The built-in SleepTool forwards every sleep call here (best-effort) so
    # the environment can react -- e.g. advance a simulated clock or schedule
    # a wake. Never model-visible. Typed as Any to avoid an import cycle.
    env_sleep_tool: Any = None
