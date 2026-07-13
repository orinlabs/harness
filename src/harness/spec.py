"""Harness runtime-config input spec.

The harness runtime takes exactly one input: a flat agent config -- a
YAML/JSON file on disk, or the JSON Bedrock serves through its
``harness-config`` endpoint. This module is the *published contract* for
that input:

* ``AgentConfigSpec`` is a strict Pydantic mirror of what
  ``harness.config_loader.build_agent_config`` accepts. Unknown keys are
  errors here (``extra="forbid"``) even where the loader would ignore
  them: ``harness validate-config`` uses this model precisely so that a
  config written for a *newer* harness fails loudly instead of having its
  new fields silently dropped.
* ``harness-spec.json`` at the repo root is *generated* from
  ``AgentConfigSpec.model_json_schema()`` and committed, so external
  consumers (Bedrock, the agents repo's CI) can check whether a rendered
  config is compatible with ANY harness commit by fetching that commit's
  copy -- no harness process required. A unit test asserts the committed
  file matches the models. Regenerate with:

      uv run python -m harness.spec > harness-spec.json

Versioning is by git commit -- the schema travels with the code, so
"compatible with harness @ SHA" means "validates against harness-spec.json
@ SHA" (fast path) or "``harness validate-config`` exits 0 at that
checkout" (authoritative path; it additionally runs the real loader).

The *authoring* format for repo-managed agents (bundle folders,
``index.yaml``, prompt fragments, shipped files) is owned by the private
agents repo and its renderer. The harness knows nothing about it;
renderers must emit configs that satisfy THIS spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

DEFAULT_SUMMARIZER_MODEL = "openai/gpt-5-nano"

# The memory systems this checkout can construct (see
# ``harness.memory.build_memory``). New backends register here so configs
# that ask for them fail validation on checkouts that predate them.
KNOWN_MEMORY_SYSTEMS = ("tiered_sqlite",)


class ToolAuthSpec(BaseModel):
    """Auth for an external tool call (see ``harness.config.ToolAuth``)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["none", "bearer_env", "bearer_literal", "headers"] = "none"
    token_env: str | None = None
    token: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class ExternalToolSpecModel(BaseModel):
    """One entry of ``tools[]`` -- an HTTP-invoked tool.

    Mirrors ``harness.config.ExternalToolSpec``; the harness POSTs
    ``{"args", "agent_id", "run_id"}`` to ``url`` and expects back
    ``{"text", "images"?}``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict = Field(description="JSON Schema for the tool's arguments.")
    url: str
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="null/omitted uses the harness default timeout.",
    )
    auth: ToolAuthSpec = Field(default_factory=ToolAuthSpec)
    forward_trace_context: bool = False


class MemorySpec(BaseModel):
    """Per-agent memory selection (see ``harness.config.MemoryConfig``).

    Omitting the block entirely is always safe: every field's null/absent
    state means "whatever the running harness decides". Memory tunables
    only become spec fields when their absence has a well-defined
    "harness decides" meaning -- otherwise every rendered config would
    bake in whatever the default was on the day it was rendered.
    """

    model_config = ConfigDict(extra="forbid")

    system: Literal["tiered_sqlite"] = "tiered_sqlite"
    summarizer_model: str | None = Field(
        default=None,
        description=(
            "Model for memory summarization. null/omitted means 'whatever the "
            f"running harness defaults to' (currently {DEFAULT_SUMMARIZER_MODEL!r}, "
            "resolved by the memory factory at construction time); a set value "
            "pins it. The default is deliberately NOT baked in here: if it were, "
            "every rendered config would silently pin today's default, and a "
            "fleet-wide summarizer upgrade would require editing every config."
        ),
    )


class AgentConfigSpec(BaseModel):
    """The flat agent config the harness runtime accepts.

    Strict mirror of ``harness.config_loader.build_agent_config`` /
    ``harness.config.AgentConfig``. ``id`` is the runtime identity,
    injected by whoever instantiates the agent (a Bedrock UUID in
    production, a trial id in evals) -- authoring formats upstream of
    this spec deliberately have no ``id``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    model: str
    system_prompt: str
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    timezone: str | None = Field(
        default=None,
        validation_alias=AliasChoices("timezone", "time_zone"),
        description="IANA timezone for agent-local, user-visible times.",
    )
    feature_flags: dict[str, str] = Field(
        default_factory=dict,
        description='Per-agent flags, name -> stored value (typically "on"/"off").',
    )
    memory: MemorySpec = Field(default_factory=MemorySpec)
    tools: list[ExternalToolSpecModel] = Field(
        default_factory=list,
        description="Flat tool list -- no adapter grouping ever reaches the harness.",
    )


def spec_json_schema() -> dict:
    """The JSON Schema committed as ``harness-spec.json``."""
    schema = AgentConfigSpec.model_json_schema()
    schema["$comment"] = (
        "Input spec: the agent config accepted by this harness commit. "
        "Generated from harness.spec; do not edit by hand: "
        "uv run python -m harness.spec > harness-spec.json"
    )
    return schema


def validate_config_data(data: dict) -> list[str]:
    """Validate a parsed config dict against this checkout. Returns errors.

    Two layers, both must pass:

    1. The strict spec model -- catches unknown keys (config newer than
       this harness) and shape/type errors.
    2. The real loader (``build_agent_config``) -- the code that will
       actually consume the config at run time, for semantic checks the
       schema can't express.
    """
    from pydantic import ValidationError

    from harness.config_loader import build_agent_config

    errors: list[str] = []
    try:
        AgentConfigSpec.model_validate(data)
    except ValidationError as e:
        errors.extend(
            f"spec: {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
    try:
        build_agent_config(data)
    except Exception as e:  # noqa: BLE001 -- loader raises plain ValueError
        errors.append(f"loader: {e}")
    return errors


def validate_config_file(path: Path) -> list[str]:
    """Parse ``path`` (YAML or JSON) and validate it. Returns errors."""
    try:
        raw = path.read_text()
    except OSError as e:
        return [f"cannot read {path}: {e}"]
    try:
        data = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        return [f"{path}: not valid YAML/JSON: {e}"]
    if not isinstance(data, dict):
        return [f"{path}: expected a mapping at the top level"]
    return validate_config_data(data)


if __name__ == "__main__":
    import sys

    json.dump(spec_json_schema(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
