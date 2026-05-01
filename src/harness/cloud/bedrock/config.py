"""Bedrock agent-config fetch + dev/eval agent creation.

Owns every call to ``/api/cloud/agents/*`` and ``/api/templates/``.
Returns already-parsed Python types (``AgentConfig``, ``dict``) so the CLI
layer never has to know Bedrock's JSON shape.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

from harness.cloud.bedrock.client import auth_header, http, platform_url
from harness.config import AgentConfig
from harness.config_loader import build_agent_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config shape translation
# ---------------------------------------------------------------------------


def _flatten_and_stamp(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten Bedrock's nested ``adapters: [{tools: [...]}]`` into a flat
    ``tools: [...]`` list, and stamp Bedrock-specific auth/trace fields on
    each tool.

    Bedrock's wire shape groups tools by adapter (for UI/categorization);
    the harness only cares about the flat list. We also attach:

    * ``Authorization: Bearer $BEDROCK_TOKEN`` (Bedrock's adapter runtime
      expects it on every tool request)
    * ``trace_id`` / ``parent_span_id`` in the POST body (so the adapter
      runtime can nest its own tool-span under our active trace)

    These are Bedrock-specific, so we stamp them here rather than defaulting
    them in ``ExternalToolSpec`` (which standalone YAML configs would
    otherwise inherit by accident). Mutates ``cfg`` in place and returns it.
    """
    flat: list[dict[str, Any]] = []
    for adapter in cfg.get("adapters", []) or []:
        for tool in adapter.get("tools", []) or []:
            tool.setdefault("auth", {"kind": "bearer_env", "token_env": "BEDROCK_TOKEN"})
            tool.setdefault("forward_trace_context", True)
            flat.append(tool)
    cfg["tools"] = flat
    cfg.pop("adapters", None)
    return cfg


def _config_from_bedrock_json(
    cfg: dict[str, Any],
    *,
    model_override: str | None = None,
    reasoning_override: str | None = None,
) -> AgentConfig:
    """Build an ``AgentConfig`` from Bedrock's harness-config JSON payload."""
    _flatten_and_stamp(cfg)
    if model_override:
        cfg["model"] = model_override
    if reasoning_override:
        cfg["reasoning_effort"] = reasoning_override
    return build_agent_config(cfg)


# ---------------------------------------------------------------------------
# HTTP entry points
# ---------------------------------------------------------------------------


def fetch_harness_config(
    agent_id: str,
    *,
    model_override: str | None = None,
    reasoning_override: str | None = None,
) -> AgentConfig:
    """GET /api/cloud/agents/{agent_id}/harness-config/ and parse the result.

    ``model_override`` / ``reasoning_override`` let the CLI stamp --model /
    --reasoning-effort onto the result without mutating Bedrock's record.
    """
    url = f"{platform_url()}/api/cloud/agents/{agent_id}/harness-config/"
    logger.info("fetching harness config from %s", url)
    resp = http().get(url, headers=auth_header(), timeout=10.0)
    resp.raise_for_status()
    return _config_from_bedrock_json(
        resp.json(),
        model_override=model_override,
        reasoning_override=reasoning_override,
    )


def resolve_template(explicit: str | None) -> str | None:
    """Resolve a template UUID-or-name to a UUID.

    Templates are optional on agent create (Bedrock allows agents without
    a template), so this returns ``None`` when ``explicit`` is ``None``.

    1. ``None`` -> ``None`` (caller skips ``template`` in the create body).
    2. UUID-shaped string -> returned as-is.
    3. Otherwise treated as a template name and resolved via
       GET /api/templates/, scoped to the caller's organization.
       Raises ``SystemExit`` if no/multiple matches.
    """
    if not explicit:
        return None
    try:
        return str(uuid.UUID(explicit))
    except ValueError:
        pass

    url = f"{platform_url()}/api/templates/"
    resp = http().get(url, headers=auth_header(), timeout=10.0)
    resp.raise_for_status()
    templates = resp.json()
    if isinstance(templates, dict) and "results" in templates:
        templates = templates["results"]
    if not isinstance(templates, list):
        raise SystemExit(f"unexpected templates response shape from {url}")
    matches = [t for t in templates if t.get("name") == explicit]
    if not matches:
        raise SystemExit(f"no template named {explicit!r} visible to this API key")
    if len(matches) > 1:
        raise SystemExit(f"multiple templates named {explicit!r}; pass --template <uuid>")
    return matches[0]["id"]


def create_dev_agent(
    *,
    template_id: str | None,
    model: str,
    system_prompt: str | None,
    branch: str,
    sha: str,
) -> dict[str, Any]:
    """POST /api/cloud/agents/ with purpose=dev. Returns the created agent JSON.

    ``template_id`` is optional — Bedrock agents can be created ad-hoc
    without a template (organization is inferred from the API key).
    """
    body: dict[str, Any] = {
        "name": f"dev-{uuid.uuid4().hex[:8]}",
        "purpose": "dev",
        "model": model,
        "tags": [f"git-ref:{branch}", f"git-sha:{sha}"],
    }
    if template_id:
        body["template"] = template_id
    if system_prompt:
        body["system_prompt"] = system_prompt
    url = f"{platform_url()}/api/cloud/agents/"
    resp = http().post(url, json=body, headers=auth_header(), timeout=15.0)
    if resp.status_code >= 400:
        print(f"create agent failed: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()


def create_eval_agent(
    *,
    scenario_name: str,
    template_id: str | None,
    model: str,
    system_prompt: str,
    branch: str,
    sha: str,
) -> dict[str, Any]:
    """POST /api/cloud/agents/ with purpose=eval. Returns the created agent JSON.

    Only called when the user explicitly asks for a Bedrock-backed eval
    (``harness eval`` with ``--bedrock-url``/token set). Standalone evals
    don't create an agent row; they synthesize a local uuid instead.
    """
    body: dict[str, Any] = {
        "name": f"eval-{scenario_name}",
        "purpose": "eval",
        "model": model,
        "system_prompt": system_prompt or "",
        "tags": [
            f"scenario:{scenario_name}",
            f"git-ref:{branch}",
            f"git-sha:{sha}",
        ],
    }
    if template_id:
        body["template"] = template_id
    url = f"{platform_url()}/api/cloud/agents/"
    resp = http().post(url, json=body, headers=auth_header(), timeout=15.0)
    if resp.status_code >= 400:
        print(
            f"create eval agent failed: {resp.status_code} {resp.text}",
            file=sys.stderr,
        )
        resp.raise_for_status()
    return resp.json()
