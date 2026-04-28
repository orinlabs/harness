from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from harness.constants import DEFAULT_TOOL_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from harness.tools.base import Tool


@dataclass(frozen=True)
class ToolAuth:
    """How ``ExternalTool`` should authenticate when calling a tool URL.

    Kinds:
      * ``none`` (default) -- no Authorization header sent.
      * ``bearer_env`` -- resolve the bearer token from ``token_env`` at call
        time. Used for Bedrock-proxied tools, where ``token_env="BEDROCK_TOKEN"``.
      * ``bearer_literal`` -- use the literal ``token`` value. Don't check this
        into YAML; intended for programmatic configs.
      * ``headers`` -- send ``headers`` verbatim. Escape hatch for APIs with
        custom auth shapes.
    """

    kind: Literal["none", "bearer_env", "bearer_literal", "headers"] = "none"
    token_env: str | None = None
    token: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalToolSpec:
    """A tool invoked by HTTP POST to ``url``.

    The harness never sees the tool's implementation. When the model calls
    this tool, harness POSTs ``{"args": {...}, "agent_id": ..., "run_id": ...}``
    to ``url`` and expects back ``{"text": str, "images": list[str] | None}``.

    Bedrock-proxied tools additionally need ``auth`` (bearer on $BEDROCK_TOKEN)
    and ``forward_trace_context=True`` so Bedrock's adapter runtime can nest
    its own tool span under our active trace. Standalone tools usually have
    ``auth=ToolAuth(kind="none")`` and ``forward_trace_context=False``.
    """

    name: str
    description: str
    parameters: dict
    url: str
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS
    auth: ToolAuth = field(default_factory=ToolAuth)
    forward_trace_context: bool = False


@dataclass(frozen=True)
class AgentConfig:
    """A flat agent config.

    ``tools`` may contain either:
      - ``ExternalToolSpec`` -- invoked via HTTP by ``ExternalTool`` at dispatch
        time (the production path).
      - An already-instantiated in-process ``Tool`` object -- used by the eval
        framework's fakes so scenarios can plug a Python implementation
        directly into ``AgentConfig.tools`` without spinning up an HTTP
        endpoint.

    ``build_tool_map`` in ``harness.tools.registry`` dispatches on the entry's
    type. Bedrock serves its config as nested adapters over the wire;
    ``harness.cloud.bedrock.config.fetch_harness_config`` flattens that into
    ``tools`` on ingest, so the rest of the harness only ever sees the flat
    list here.
    """

    id: str
    model: str
    system_prompt: str
    tools: "list[ExternalToolSpec | Tool]" = field(default_factory=list)
    reasoning_effort: str | None = None
    # When True, ``MemoryService`` defers all summarization to end-of-run
    # and uses the past-tense-only prompt variant. Off by default so
    # existing agents keep the original 1m -> 5m -> hourly -> ... cascade
    # and mid-run summary triggers. Plumbed from the platform's agent
    # config (e.g. the ``summarizer_v2`` feature flag on Bedrock).
    summarizer_v2: bool = False
