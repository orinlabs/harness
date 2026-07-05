"""Transient-failure retries in the OpenRouter client (network-free).

Regression suite for the incident where a single flaky TLS read
(`httpx.ReadError: [SSL: SSLV3_ALERT_BAD_RECORD_MAC] ...`) on one OpenRouter
call crashed the whole `harness agent` process and the supervising rollout
runner scored the episode as-is. The chat-completions POST is stateless, so
transport failures and HTTP 502/503/529 get bounded retries with backoff;
everything else keeps its fail-fast behavior.

All tests run against `httpx.MockTransport` — no network, no API key needed
(a fake one is injected so `_api_key()` passes).
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from harness.core import llm

MODEL = "openai/gpt-4o-mini"

_SSE_OK_BODY = (
    'data: {"id":"gen-1","model":"openai/gpt-4o-mini","choices":'
    '[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}\n'
    "\n"
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6,"cost":0.001}}\n'
    "\n"
    "data: [DONE]\n"
)


def _sse_response(body: str = _SSE_OK_BODY) -> httpx.Response:
    return httpx.Response(
        200,
        content=body.encode(),
        headers={"content-type": "text/event-stream"},
    )


class _ExplodingStream(httpx.SyncByteStream):
    """Yields some SSE bytes, then dies mid-stream like a flaky TLS read."""

    def __init__(self, chunks: list[bytes], exc: Exception):
        self._chunks = chunks
        self._exc = exc

    def __iter__(self):
        yield from self._chunks
        raise self._exc


@pytest.fixture
def mock_openrouter(monkeypatch):
    """Route `llm._http()` at an in-memory transport and neuter backoff sleeps.

    Returns a harness object; tests assign `harness.handler` (a callable
    `request -> httpx.Response`, which may raise) and can inspect
    `harness.requests` (every request the client actually sent) and
    `harness.sleeps` (backoff waits that would have happened).
    """

    class Harness:
        def __init__(self):
            self.requests: list[httpx.Request] = []
            self.sleeps: list[float] = []
            self.handler = None

    h = Harness()

    def transport_handler(request: httpx.Request) -> httpx.Response:
        h.requests.append(request)
        assert h.handler is not None, "test must set mock_openrouter.handler"
        return h.handler(request)

    client = httpx.Client(transport=httpx.MockTransport(transport_handler))
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "_sleep", lambda s: h.sleeps.append(s))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.delenv("HARNESS_LLM_RETRIES", raising=False)

    yield h
    client.close()


def _complete() -> llm.LLMResponse:
    return llm.complete(
        model=MODEL,
        system="You are a test assistant.",
        messages=[{"role": "user", "content": "say OK"}],
    )


def _retry_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "openrouter transient failure" in r.getMessage()
    ]


def test_read_error_retried_until_success(mock_openrouter, caplog):
    """ReadError on attempts 1 and 2, success on 3 → complete() returns
    normally with the streamed payload, and each retry logged one WARNING
    carrying attempt number, wait, exception class + message, and model."""
    caplog.set_level(logging.WARNING, logger="harness.core.llm")

    def handler(request):
        if len(mock_openrouter.requests) < 3:
            raise httpx.ReadError(
                "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:2570)"
            )
        return _sse_response()

    mock_openrouter.handler = handler

    resp = _complete()

    assert resp.text == "OK"
    assert resp.finish_reason == "stop"
    assert resp.usage.total_cost == 0.001
    assert len(mock_openrouter.requests) == 3
    assert len(mock_openrouter.sleeps) == 2

    warnings = _retry_warnings(caplog)
    assert len(warnings) == 2
    for n, record in enumerate(warnings, start=1):
        msg = record.getMessage()
        assert f"attempt {n}/4" in msg
        assert "ReadError" in msg
        assert "SSLV3_ALERT_BAD_RECORD_MAC" in msg
        assert MODEL in msg
        assert "retrying in" in msg


def test_mid_stream_failure_discards_partial_output_and_retries(mock_openrouter, caplog):
    """A ReadError *after* partial SSE chunks must retry the whole request and
    return only the clean re-sent response — no partial-content duplication."""
    caplog.set_level(logging.WARNING, logger="harness.core.llm")

    def handler(request):
        if len(mock_openrouter.requests) == 1:
            partial = (
                'data: {"id":"gen-0","model":"openai/gpt-4o-mini","choices":'
                '[{"index":0,"delta":{"content":"PARTIAL"},"finish_reason":null}]}\n\n'
            )
            return httpx.Response(
                200,
                stream=_ExplodingStream(
                    [partial.encode()],
                    httpx.ReadError("connection dropped mid-stream"),
                ),
                headers={"content-type": "text/event-stream"},
            )
        return _sse_response()

    mock_openrouter.handler = handler

    resp = _complete()

    assert resp.text == "OK", "partial first-attempt output must be discarded"
    assert "PARTIAL" not in resp.text
    assert len(mock_openrouter.requests) == 2

    warnings = _retry_warnings(caplog)
    assert len(warnings) == 1
    assert "ReadError" in warnings[0].getMessage()


def test_exhausted_retries_reraise_original_read_error(mock_openrouter, caplog):
    """When every attempt fails, the transport error propagates to the caller
    exactly as before retries existed."""
    caplog.set_level(logging.WARNING, logger="harness.core.llm")

    def handler(request):
        raise httpx.ReadError("still broken")

    mock_openrouter.handler = handler

    with pytest.raises(httpx.ReadError, match="still broken"):
        _complete()

    # 1 initial attempt + 3 retries (default budget), each retry slept once.
    assert len(mock_openrouter.requests) == 4
    assert len(mock_openrouter.sleeps) == 3
    assert len(_retry_warnings(caplog)) == 3


@pytest.mark.parametrize("status", [400, 429])
def test_client_errors_are_not_retried(mock_openrouter, status):
    """Deterministic / quota HTTP errors keep today's fail-fast OpenRouterError
    semantics: exactly one request, no retries, status + body preserved."""
    error_body = json.dumps({"error": {"message": "nope", "code": status}})

    def handler(request):
        return httpx.Response(status, content=error_body.encode())

    mock_openrouter.handler = handler

    with pytest.raises(llm.OpenRouterError) as excinfo:
        _complete()

    assert excinfo.value.status_code == status
    assert "nope" in excinfo.value.body
    assert excinfo.value.model == MODEL
    assert len(mock_openrouter.requests) == 1
    assert mock_openrouter.sleeps == []


def test_http_503_is_retried(mock_openrouter, caplog):
    """502/503/529 are 'upstream temporarily unavailable' — re-sending helps."""
    caplog.set_level(logging.WARNING, logger="harness.core.llm")

    def handler(request):
        if len(mock_openrouter.requests) == 1:
            return httpx.Response(503, content=b'{"error":{"message":"overloaded"}}')
        return _sse_response()

    mock_openrouter.handler = handler

    resp = _complete()

    assert resp.text == "OK"
    assert len(mock_openrouter.requests) == 2

    warnings = _retry_warnings(caplog)
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "OpenRouterError" in msg
    assert "503" in msg


def test_env_zero_disables_retries(mock_openrouter, monkeypatch):
    """HARNESS_LLM_RETRIES=0 means one attempt, fail immediately."""
    monkeypatch.setenv("HARNESS_LLM_RETRIES", "0")

    def handler(request):
        raise httpx.ReadError("flaky")

    mock_openrouter.handler = handler

    with pytest.raises(httpx.ReadError, match="flaky"):
        _complete()

    assert len(mock_openrouter.requests) == 1
    assert mock_openrouter.sleeps == []


def test_env_overrides_retry_budget(mock_openrouter, monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_RETRIES", "1")

    def handler(request):
        raise httpx.ReadError("flaky")

    mock_openrouter.handler = handler

    with pytest.raises(httpx.ReadError):
        _complete()

    assert len(mock_openrouter.requests) == 2


def test_invalid_retry_env_fails_loudly(mock_openrouter, monkeypatch):
    """A silently misparsed retry budget would only surface mid-incident."""
    monkeypatch.setenv("HARNESS_LLM_RETRIES", "lots")

    mock_openrouter.handler = lambda request: _sse_response()

    with pytest.raises(ValueError, match="HARNESS_LLM_RETRIES"):
        _complete()

    assert mock_openrouter.requests == []


def test_backoff_schedule_is_roughly_1_4_10_with_jitter():
    """Retry waits follow ~1s/4s/10s exponential backoff, jittered ±25%."""
    for retry_number, base in ((1, 1.0), (2, 4.0), (3, 10.0)):
        waits = [llm._backoff_seconds(retry_number) for _ in range(50)]
        assert all(base * 0.75 <= w <= base * 1.25 for w in waits)
