"""Shared HTTP helpers for the Bedrock backend.

All Bedrock HTTP goes through this module so there's exactly one place that
reads ``BEDROCK_URL`` / ``BEDROCK_TOKEN`` and owns the shared ``httpx.Client``.
"""
from __future__ import annotations

import os

import httpx

_client: httpx.Client | None = None


def platform_url() -> str:
    """Return the Bedrock base URL. Raises if unset.

    Use ``platform_url_or_none`` if you want best-effort behavior (e.g. the
    trace sink swallows failures, so it short-circuits when unconfigured).
    """
    url = platform_url_or_none()
    if url is None:
        raise RuntimeError(
            "BEDROCK_URL is not set. Bedrock clients require it to reach the platform."
        )
    return url


def platform_url_or_none() -> str | None:
    url = os.environ.get("BEDROCK_URL")
    return url.rstrip("/") if url else None


def auth_header() -> dict[str, str]:
    token = os.environ.get("BEDROCK_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=10.0)
    return _client
