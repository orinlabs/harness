"""HTTP client for the `current` workflow platform.

All workflow-mode HTTP goes through this module so there's exactly one place
that reads ``CURRENT_URL`` / ``CURRENT_RUN_TOKEN`` and owns the shared
``httpx.Client`` (same layout as ``harness.cloud.bedrock.client``).

Contract (v0, authoritative copy in orinlabs/workflows designs/002):

    GET  {CURRENT_URL}/api/workflows/runs/{run_id}/definition/
    POST {CURRENT_URL}/api/workflows/runs/{run_id}/transitions/
    POST {CURRENT_URL}/api/workflows/runs/{run_id}/records/

Every request carries ``Authorization: Bearer $CURRENT_RUN_TOKEN`` (a
run-scoped token minted by current when it dispatches the sandbox).

Postgres on the current side is the source of truth; the sandbox running
this client is disposable. That's why transient failures (connection
errors, 5xx) get bounded retries with backoff — losing a transition or a
record to a network blip would strand the run's journal — while 4xx
responses fail fast: they mean the runner sent something the platform
rejected (bad token, unknown run, schema-invalid record) and retrying
can't fix it.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None

_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.5
# Indirection so tests can observe/skip the waits without patching `time`.
_sleep = time.sleep


class CurrentAPIError(RuntimeError):
    """Non-2xx response from current (or exhausted transport retries).

    Carries ``status_code`` (None when the failure was transport-level) and
    the response ``body`` so callers can branch — e.g. the workflow runner
    treats a 422 on the records endpoint as a step failure, not a crash.
    """

    def __init__(self, message: str, *, status_code: int | None = None, body: str = ""):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=15.0)
    return _client


class CurrentClient:
    """Thin client over the three run-scoped current endpoints."""

    def __init__(self, base_url: str, token: str, run_id: str):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self.run_id = run_id

    @classmethod
    def from_env(cls) -> CurrentClient:
        """Build a client from ``CURRENT_URL`` / ``CURRENT_RUN_TOKEN`` /
        ``HARNESS_RUN_ID``. Raises with a per-variable message when any is
        missing so a misdispatched sandbox fails loudly at startup."""
        missing = [
            k
            for k in ("CURRENT_URL", "CURRENT_RUN_TOKEN", "HARNESS_RUN_ID")
            if not os.environ.get(k)
        ]
        if missing:
            raise RuntimeError(
                f"workflow mode requires env: {', '.join(missing)}. "
                "current sets these when dispatching the sandbox."
            )
        return cls(
            os.environ["CURRENT_URL"],
            os.environ["CURRENT_RUN_TOKEN"],
            os.environ["HARNESS_RUN_ID"],
        )

    # -- endpoints ----------------------------------------------------------

    def get_definition(self) -> dict[str, Any]:
        """GET the run definition + journal (steps_state, decisions)."""
        resp = self._request("GET", self._url("definition"))
        return resp.json()

    def post_transition(
        self,
        *,
        step_id: str,
        status: str,
        attempt: int,
        error: str | None = None,
    ) -> None:
        """POST a step transition. Idempotent server-side; 200/201 = success."""
        self._request(
            "POST",
            self._url("transitions"),
            json={"step_id": step_id, "status": status, "attempt": attempt, "error": error},
        )

    def post_record(
        self,
        *,
        record_type: str,
        data: dict[str, Any],
        step_id: str | None,
        project: str | None = None,
        extras: list[Any] | None = None,
        produced_at: str | None = None,
    ) -> dict[str, Any]:
        """POST a record row. Returns the created body (``{"id": uuid}``).

        A 422 means the platform rejected the record against its declared
        schema — surfaced as ``CurrentAPIError(status_code=422)`` so the
        runner can report it as a step failure rather than crashing.
        """
        resp = self._request(
            "POST",
            self._url("records"),
            json={
                "record_type": record_type,
                "project": project,
                "step_id": step_id,
                "data": data,
                "extras": extras or [],
                "produced_at": produced_at,
            },
        )
        return resp.json() if resp.content else {}

    # -- plumbing -----------------------------------------------------------

    def _url(self, leaf: str) -> str:
        return f"{self._base_url}/api/workflows/runs/{self.run_id}/{leaf}/"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _request(self, method: str, url: str, *, json: Any | None = None) -> httpx.Response:
        """Issue a request with bounded retries on transport errors and 5xx.

        4xx fails immediately (the request is wrong; a retry can't help).
        """
        last_error: str = ""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = _http().request(method, url, json=json, headers=self._headers())
            except httpx.TransportError as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "current %s %s transport error (attempt %d/%d): %s",
                    method,
                    url,
                    attempt,
                    _MAX_ATTEMPTS,
                    last_error,
                )
            else:
                if resp.status_code < 300:
                    return resp
                body = resp.text[:2000]
                if resp.status_code >= 500 and attempt < _MAX_ATTEMPTS:
                    last_error = f"HTTP {resp.status_code}: {body}"
                    logger.warning(
                        "current %s %s -> %d (attempt %d/%d); retrying",
                        method,
                        url,
                        resp.status_code,
                        attempt,
                        _MAX_ATTEMPTS,
                    )
                else:
                    raise CurrentAPIError(
                        f"current {method} {url} failed: HTTP {resp.status_code}: {body}",
                        status_code=resp.status_code,
                        body=body,
                    )
            if attempt < _MAX_ATTEMPTS:
                _sleep(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
        raise CurrentAPIError(
            f"current {method} {url} failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )
