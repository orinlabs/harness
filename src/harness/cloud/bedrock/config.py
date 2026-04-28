"""Bedrock agent-config fetch + dev/eval agent creation.

Owns every call to ``/api/cloud/agents/*`` and ``/api/products/products/``.
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
            tool.setdefault(
                "auth", {"kind": "bearer_env", "token_env": "BEDROCK_TOKEN"}
            )
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


def resolve_product(explicit: str | None) -> str:
    """Resolve a product UUID.

    1. ``explicit`` wins if provided.
    2. Else GET /api/products/products/ -- if exactly one visible, use it.
    3. Else raise ``SystemExit`` with a helpful message.
    """
    if explicit:
        return explicit
    url = f"{platform_url()}/api/products/products/"
    resp = http().get(url, headers=auth_header(), timeout=10.0)
    resp.raise_for_status()
    products = resp.json()
    if isinstance(products, dict) and "results" in products:
        products = products["results"]
    if not isinstance(products, list) or not products:
        raise SystemExit("no products visible to this API key")
    if len(products) > 1:
        raise SystemExit("multiple products visible, pass --product <uuid>")
    return products[0]["id"]


def create_dev_agent(
    *,
    product_id: str,
    model: str,
    system_prompt: str | None,
    template: str | None,
    branch: str,
    sha: str,
) -> dict[str, Any]:
    """POST /api/cloud/agents/ with purpose=dev. Returns the created agent JSON."""
    body: dict[str, Any] = {
        "name": f"dev-{uuid.uuid4().hex[:8]}",
        "purpose": "dev",
        "product": product_id,
        "model": model,
        "tags": [f"git-ref:{branch}", f"git-sha:{sha}"],
    }
    if system_prompt:
        body["system_prompt"] = system_prompt
    if template:
        logger.warning(
            "# TODO(Phase 2): template not yet implemented server-side; "
            "ignoring --template=%s",
            template,
        )
    url = f"{platform_url()}/api/cloud/agents/"
    resp = http().post(url, json=body, headers=auth_header(), timeout=15.0)
    if resp.status_code >= 400:
        print(f"create agent failed: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()


def create_eval_agent(
    *,
    scenario_name: str,
    product_id: str,
    model: str,
    system_prompt: str,
    template: str | None,
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
        "product": product_id,
        "model": model,
        "system_prompt": system_prompt or "",
        "tags": [
            f"scenario:{scenario_name}",
            f"git-ref:{branch}",
            f"git-sha:{sha}",
        ],
    }
    if template:
        logger.warning(
            "# TODO(Phase 2): template not yet implemented server-side; "
            "ignoring --template=%s",
            template,
        )
    url = f"{platform_url()}/api/cloud/agents/"
    resp = http().post(url, json=body, headers=auth_header(), timeout=15.0)
    if resp.status_code >= 400:
        print(
            f"create eval agent failed: {resp.status_code} {resp.text}",
            file=sys.stderr,
        )
        resp.raise_for_status()
    return resp.json()
