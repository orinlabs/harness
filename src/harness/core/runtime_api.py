"""Runtime API client.

Agent-scoped runtime endpoints on the infra platform. Currently just sleep:
  - POST /api/cloud/agents/{agent_id}/sleep/
    body: { until: iso8601 | "indefinite", reason: str }

Callers pass agent_id explicitly (it's always known at the call site).
"""
from __future__ import annotations

import os
from typing import Any

import httpx

_client: httpx.Client | None = None


def _platform_url() -> str:
    url = os.environ.get("BEDROCK_URL")
    if not url:
        raise RuntimeError(
            "BEDROCK_URL is not set. Core clients need it to reach the infra platform."
        )
    return url.rstrip("/")


def _auth_header() -> dict[str, str]:
    token = os.environ.get("BEDROCK_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=10.0)
    return _client


def sleep(agent_id: str, until: str, reason: str = "") -> dict[str, Any]:
    """Request the platform put this agent to sleep.

    `until` is either an ISO-8601 timestamp or the string "indefinite".
    Returns the platform's JSON response body.
    """
    resp = _http().post(
        f"{_platform_url()}/api/cloud/agents/{agent_id}/sleep/",
        json={"until": until, "reason": reason},
        headers=_auth_header(),
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}
