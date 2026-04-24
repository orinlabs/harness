from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from harness.constants import DEFAULT_TOOL_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from harness.tools.base import Tool


@dataclass(frozen=True)
class ExternalToolSpec:
    """A tool registered by the platform, invoked by HTTP POST to `url`.

    The harness never sees the tool's implementation. When the model calls this
    tool, harness POSTs `{"args": {...}, "agent_id": "...", "run_id": "..."}`
    to `url` and expects back `{"text": str, "images": list[str] | None}`.
    """

    name: str
    description: str
    parameters: dict
    url: str
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS


@dataclass(frozen=True)
class AdapterConfig:
    """Adapter declaration for an agent.

    ``tools`` may contain either:
      - ``ExternalToolSpec`` -- invoked via HTTP by ``ExternalTool`` at dispatch
        time (the production path).
      - An already-instantiated in-process ``Tool`` object -- used by the eval
        framework's fake adapters (``harness.evals.fakes``) so scenarios can
        plug a Python implementation directly into ``AgentConfig.adapters``
        without spinning up an HTTP endpoint.

    ``build_tool_map`` in ``harness.tools.registry`` dispatches on the
    entry's type.
    """

    name: str
    description: str
    tools: "list[Union[ExternalToolSpec, Tool]]"


@dataclass(frozen=True)
class AgentConfig:
    id: str
    model: str
    system_prompt: str
    adapters: list[AdapterConfig] = field(default_factory=list)
    reasoning_effort: str | None = None
