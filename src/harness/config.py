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
class MemoryConfig:
    """Per-agent memory selection.

    Defaults reproduce the harness's historical behavior (tiered sqlite
    summaries, cheap summarizer model), so configs that omit the block
    entirely -- including every existing Bedrock payload and standalone
    YAML -- behave identically. ``system`` picks the backend in
    ``harness.memory.build_memory``; unknown values fail at startup.
    """

    system: str = "tiered_sqlite"
    # Summarization always runs on a cheap model, not the agent's
    # configured model. Otherwise every turn's `update_summaries()`
    # fires N summary-generation LLM calls at whatever the agent
    # happens to be using (Opus, Sonnet, etc.) -- easily >$1/turn on
    # agents with deep history.
    #
    # None means "whatever the running harness defaults to"
    # (``harness.spec.DEFAULT_SUMMARIZER_MODEL``), resolved by
    # ``harness.memory.build_memory`` at construction time. A set value
    # pins the model. Rendered configs carry null unless they pin one, so
    # a fleet-wide summarizer upgrade is a harness change, not an
    # every-config edit.
    summarizer_model: str | None = None


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
    max_tokens: int | None = None
    timezone: str | None = None
    # Generic per-agent feature flags. Maps flag name -> stored value
    # (typically ``"on"`` / ``"off"`` but free-form strings are allowed
    # so non-boolean flags work — e.g. a tier name or a model variant).
    # Plumbed from the platform's agent config (Bedrock's ``feature_flags``
    # field on the ``harness-config`` payload, or ``feature_flags:`` in
    # standalone YAML configs). Use ``is_enabled(name)`` for the common
    # boolean check; use ``feature_flags.get(name, default)`` for value
    # reads.
    feature_flags: dict[str, str] = field(default_factory=dict)
    # Memory backend selection; see MemoryConfig. Set from the config's
    # optional `memory:` block; everything else gets the defaults.
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def is_enabled(self, flag: str) -> bool:
        """Return True if ``feature_flags[flag]`` resolves to ``"on"``.

        Matches Bedrock's ``FeatureFlag.is_enabled(name, agent)`` semantics:
        any value other than the literal string ``"on"`` (case-insensitive)
        — including the empty string and a missing flag — counts as off.
        Use ``feature_flags.get(name, default)`` directly when a flag is
        modeled as a free-form string instead of a boolean toggle.
        """
        return (self.feature_flags.get(flag, "") or "").strip().lower() == "on"
