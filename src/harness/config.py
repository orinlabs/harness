from dataclasses import dataclass, field

from harness.constants import DEFAULT_TOOL_TIMEOUT_SECONDS


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
    name: str
    description: str
    tools: list[ExternalToolSpec]


@dataclass(frozen=True)
class AgentConfig:
    id: str
    model: str
    system_prompt: str
    adapters: list[AdapterConfig] = field(default_factory=list)
    reasoning_effort: str | None = None
