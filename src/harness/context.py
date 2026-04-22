from contextvars import ContextVar
from dataclasses import dataclass

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
