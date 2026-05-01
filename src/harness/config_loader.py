"""Load ``AgentConfig`` from a YAML / JSON file on disk.

Two entry points:

* ``load_agent_config_from_path(path)`` -- given an explicit file path,
  parses YAML (``yaml.safe_load`` handles JSON too) and builds the Python
  types.

* ``load_agent_config_by_name(name)`` -- looks up ``./agents/<name>.{yaml,
  yml,json}`` relative to the cwd. Raises ``FileNotFoundError`` when the
  file isn't there, so the CLI can fall back to Bedrock cleanly.

YAML schema (JSON is equivalent):

    id: my-agent              # required
    model: claude-haiku-4-5   # required
    system_prompt: |          # required
      ...
    reasoning_effort: medium  # optional
    summarizer_v2: false      # optional, legacy — prefer feature_flags below
    feature_flags:            # optional; per-agent overrides keyed by name
      summarizer_v2: "on"
      auto_associative_memory: "off"

    tools:                    # flat list -- no adapter grouping
      - name: get_forecast
        description: Five-day forecast for a city.
        parameters:           # JSON Schema
          type: object
          properties:
            city: { type: string }
          required: [city]
        url: http://localhost:9001/weather/get_forecast
        timeout_seconds: 30          # optional; default 60
        auth:                        # optional; default {kind: none}
          kind: bearer_env           # none | bearer_env | bearer_literal | headers
          token_env: OPENWEATHER_API_KEY
        forward_trace_context: false # optional; default false

Bedrock serves its config with a nested ``adapters: [{name, tools: [...]}]``
shape. ``harness.cloud.bedrock.config.fetch_harness_config`` flattens that
on ingest -- only this (flat) schema reaches ``AgentConfig``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from harness.config import AgentConfig, ExternalToolSpec, ToolAuth

# Fixed, repo-relative. "./agents/<name>.yaml".
_AGENTS_DIR_NAME = "agents"
_SUPPORTED_EXTS = (".yaml", ".yml", ".json")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_agent_config_from_path(path: Path) -> AgentConfig:
    """Parse ``path`` (YAML or JSON) into an ``AgentConfig``."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    raw = path.read_text()
    if path.suffix == ".json":
        data = json.loads(raw)
    else:
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return build_agent_config(data)


def load_agent_config_by_name(name: str) -> AgentConfig:
    """Resolve ``<cwd>/agents/<name>.{yaml,yml,json}`` and load it.

    Raises ``FileNotFoundError`` (with a message listing the paths we tried)
    if nothing matches. Callers in ``harness.cli`` catch this to fall back
    to Bedrock.
    """
    agents_dir = Path.cwd() / _AGENTS_DIR_NAME
    candidates = [agents_dir / f"{name}{ext}" for ext in _SUPPORTED_EXTS]
    for p in candidates:
        if p.exists():
            return load_agent_config_from_path(p)
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"no local agent config found for '{name}'. Tried:\n  {tried}")


def build_agent_config(data: dict[str, Any]) -> AgentConfig:
    """Build an ``AgentConfig`` from a plain dict.

    Used by both the YAML loader above and Bedrock's JSON-returning fetch
    helper (which pre-flattens its nested adapter shape into ``tools``), so
    the same schema validation runs on both paths.
    """
    _require_keys(data, ("id", "model", "system_prompt"), where="agent config")

    return AgentConfig(
        id=str(data["id"]),
        model=str(data["model"]),
        system_prompt=str(data["system_prompt"]),
        reasoning_effort=_opt_str(data.get("reasoning_effort")),
        feature_flags=_feature_flags(data.get("feature_flags")),
        summarizer_v2=bool(data.get("summarizer_v2", False)),
        tools=[_tool(t) for t in data.get("tools", []) or []],
    )


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _tool(data: dict[str, Any]) -> ExternalToolSpec:
    _require_keys(
        data,
        ("name", "description", "parameters", "url"),
        where="tool",
    )
    timeout = data.get("timeout_seconds")
    auth_data = data.get("auth") or {}
    return ExternalToolSpec(
        name=str(data["name"]),
        description=str(data["description"]),
        parameters=dict(data["parameters"]),
        url=str(data["url"]),
        **({"timeout_seconds": float(timeout)} if timeout is not None else {}),
        auth=_tool_auth(auth_data),
        forward_trace_context=bool(data.get("forward_trace_context", False)),
    )


def _tool_auth(data: dict[str, Any]) -> ToolAuth:
    if not data:
        return ToolAuth()
    kind = str(data.get("kind", "none"))
    if kind not in ("none", "bearer_env", "bearer_literal", "headers"):
        raise ValueError(
            f"tool auth.kind must be one of none|bearer_env|bearer_literal|headers, got {kind!r}"
        )
    return ToolAuth(
        kind=kind,  # type: ignore[arg-type]
        token_env=_opt_str(data.get("token_env")),
        token=_opt_str(data.get("token")),
        headers=dict(data.get("headers") or {}),
    )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _feature_flags(value: Any) -> dict[str, str]:
    """Coerce the optional ``feature_flags`` block into ``dict[str, str]``.

    Accepts ``None`` (no flags), a real mapping (the typical case), or any
    other value (rejected with a clear error). Values are stringified so a
    YAML ``true`` / ``false`` becomes the literal string ``"True"`` /
    ``"False"`` — but new YAML configs should quote ``"on"`` / ``"off"``
    explicitly to match Bedrock's wire format.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"feature_flags must be a mapping of name -> string value, got {type(value).__name__}"
        )
    return {str(k): str(v) for k, v in value.items()}


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], *, where: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"{where}: missing required key(s): {', '.join(missing)}")
