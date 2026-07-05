"""OpenRouter client.

Single entry point: `complete(...)`. Returns a structured response with text,
parsed tool calls, and usage including total cost (in USD).

OpenRouter speaks OpenAI chat-completions format, so no structured request type is
needed — harness code builds plain dicts for `messages` and `tools`.

To swap in a different LLM provider, branch the repo and rewrite this file.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- transient-failure retry policy -----------------------------------------
#
# One flaky TLS read on a single OpenRouter call (e.g. `httpx.ReadError:
# [SSL: SSLV3_ALERT_BAD_RECORD_MAC]`) used to crash the whole agent process,
# and the supervising rollout runner scores the episode as-is. A
# chat-completions POST is stateless and safely re-sendable, so transport
# failures — including mid-stream ones, where we just discard partial SSE
# output — get bounded retries with exponential backoff + jitter.
#
# Retried: the `httpx.TransportError` family (ReadError, ConnectError,
# RemoteProtocolError, the timeout subclasses, ...) and HTTP 502/503/529 from
# OpenRouter. Anything else (400/401/403/404/429, parse errors, ...) keeps
# today's fail-fast behavior.
_DEFAULT_LLM_RETRIES = 3  # override via HARNESS_LLM_RETRIES; 0 disables
_RETRYABLE_STATUS_CODES = frozenset({502, 503, 529})
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 10.0
# Indirection so tests can observe/skip the waits without patching `time`.
_sleep = time.sleep
_DEFAULT_ANTHROPIC_MAX_TOKENS = 8192
_REASONING_EFFORT_RATIOS = {
    "xhigh": 0.95,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2,
    "minimal": 0.1,
}


class OpenRouterError(RuntimeError):
    """Non-2xx response from OpenRouter.

    Carries the raw response body so the tracer's generic `f"{type(e).__name__}:
    {e}"` error capture surfaces the JSON `{error: {message, code}}` payload
    OpenRouter returns — which is usually the only place that explains *why*
    a request 400'd (invalid model slug, schema rejected, quota, etc.).
    """

    def __init__(self, *, status_code: int, body: str, model: str):
        self.status_code = status_code
        self.body = body
        self.model = model
        super().__init__(f"openrouter HTTP {status_code} for model {model}: {body}")


# ---------------------------------------------------------------------------
# Model slug translation
# ---------------------------------------------------------------------------
#
# Bedrock speaks Anthropic API IDs for Claude models (`claude-opus-4-7`,
# `claude-sonnet-4-6`, ...). OpenRouter uses its own namespaced slugs
# (`anthropic/claude-opus-4.7`, `anthropic/claude-sonnet-4.6`, ...).
# Passing the Anthropic ID directly yields a 400 from OpenRouter ("Invalid
# model"), so translate at the edge.
#
# Keep this map in sync with whatever Cursor's bedrock hands us. Anything
# that already looks like an OpenRouter slug (contains `/`) or that isn't
# recognised here is passed through untouched so OpenAI/Google/etc. still
# work.
_MODEL_MAP: dict[str, str] = {
    # Current Anthropic flagship models (per docs.anthropic.com and
    # openrouter.ai/anthropic as of 2026-04). Some bedrock IDs carry a
    # `-YYYYMMDD` suffix; we key on both the aliased and dated forms.
    "claude-opus-4-7": "anthropic/claude-opus-4.7",
    "claude-opus-4-6": "anthropic/claude-opus-4.6",
    "claude-opus-4-5": "anthropic/claude-opus-4.5",
    "claude-opus-4-5-20251101": "anthropic/claude-opus-4.5",
    "claude-opus-4-1": "anthropic/claude-opus-4.1",
    "claude-opus-4-1-20250805": "anthropic/claude-opus-4.1",
    "claude-opus-4-0": "anthropic/claude-opus-4",
    "claude-opus-4-20250514": "anthropic/claude-opus-4",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4-5-20250929": "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4-0": "anthropic/claude-sonnet-4",
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
}


def _translate_model(model: str) -> str:
    """Translate a bedrock/Anthropic model id to an OpenRouter slug.

    Pass-through rules:
      - Already namespaced (`foo/bar`): leave alone.
      - Explicit entry in `_MODEL_MAP`: use it.
      - Otherwise: return untouched and trust the caller.

    We log at INFO the first time a translation happens per-process to make
    misconfigured model names obvious in the trace.
    """
    if "/" in model:
        return model
    mapped = _MODEL_MAP.get(model)
    if mapped is not None and mapped != model:
        logger.info("translating model slug %r -> %r for openrouter", model, mapped)
        return mapped
    return model


def _is_anthropic_model(model: str) -> bool:
    """True for any OpenRouter slug that resolves to an Anthropic model.

    Tolerates OpenRouter's routing prefixes/suffixes:
      - ``~anthropic/claude-opus-latest`` — the ``~`` provider prefix that
        pins routing to a specific provider and supports ``-latest`` aliases.
      - ``anthropic/claude-sonnet-4.6:floor`` / ``:nitro`` / ``:free`` —
        suffixes that pick a routing strategy but don't change the
        underlying provider.

    A naive ``startswith("anthropic/")`` check missed the ``~`` form,
    which silently disabled prompt caching, the Anthropic ``max_tokens``
    default, and the effort-based reasoning budget for any agent
    configured with a ``~anthropic/...`` slug -- exactly the symptom we
    saw with ``~anthropic/claude-opus-latest`` returning
    ``cached_tokens=0`` and ``cache_write_tokens=0`` on every turn even
    after caching was wired up.

    Substring containment is safe here: no non-Anthropic OpenRouter
    provider uses ``anthropic/`` in its slug.
    """
    return "anthropic/" in model


def _effective_max_tokens(*, model: str, max_tokens: int | None) -> int | None:
    """Choose a request max_tokens that keeps Anthropic reasoning valid.

    Anthropic derives reasoning budget from top-level max_tokens when
    reasoning.effort is used. OpenRouter requires max_tokens to be strictly
    higher than the resulting reasoning budget; because Anthropic also floors
    reasoning budgets at 1024, any explicit Anthropic max_tokens must exceed
    1024.
    """
    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if _is_anthropic_model(model) and max_tokens <= 1024:
            raise ValueError(
                "Anthropic reasoning requires max_tokens > 1024 so the final "
                f"response has room after the minimum reasoning budget; got {max_tokens}"
            )
        return max_tokens
    if _is_anthropic_model(model):
        return _DEFAULT_ANTHROPIC_MAX_TOKENS
    return None


def _anthropic_reasoning_max_tokens(
    *, max_tokens: int, reasoning_effort: str | None
) -> int | None:
    if reasoning_effort == "none":
        return None
    ratio = _REASONING_EFFORT_RATIOS.get(reasoning_effort or "medium")
    if ratio is None:
        raise ValueError(
            "reasoning_effort must be one of minimal|low|medium|high|xhigh|none "
            f"for Anthropic models, got {reasoning_effort!r}"
        )
    budget = int(max_tokens * ratio)
    return max(min(budget, 128_000), 1024)


def _build_chat_completion_body(
    *,
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    model = _translate_model(model)
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(_prepare_replay_messages(messages))

    body: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
        "usage": {"include": True},
    }
    effective_max_tokens = _effective_max_tokens(model=model, max_tokens=max_tokens)
    if effective_max_tokens is not None:
        body["max_tokens"] = effective_max_tokens
    # Enable Anthropic prompt caching in OpenRouter's "automatic" mode.
    #
    # Top-level `cache_control` tells OpenRouter to advance an ephemeral
    # cache breakpoint to the last cacheable block on every request, which
    # is exactly the multi-turn agent pattern: stable system prompt + tool
    # schemas + a growing message log. Without this flag Anthropic returns
    # zero cached tokens and we pay full input price on every replay.
    #
    # Caveats encoded by this branch:
    #   - Top-level `cache_control` is only honoured when the request is
    #     routed to Anthropic directly. OpenRouter automatically excludes
    #     Bedrock/Vertex endpoints when it's present, which matches what
    #     we want -- the harness already targets `anthropic/...` slugs.
    #   - Other providers (OpenAI, DeepSeek, Gemini 2.5) cache implicitly
    #     and need no flag, so we skip this for non-Anthropic models to
    #     avoid sending a body OpenRouter would route oddly.
    #   - Default 5-minute TTL is fine: the agent loop's LLM↔tool cycle
    #     is way under that. Long idle pauses fall back to a cache write
    #     at the same rate as the very first request.
    if _is_anthropic_model(model):
        body["cache_control"] = {"type": "ephemeral"}
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    # Turn reasoning on by default and request plain-text summaries.
    #
    # OpenRouter's `reasoning` object is a no-op on non-reasoning models,
    # so always sending it is safe. But the defaults-per-field matter:
    #
    #   - `enabled: true` is required to activate extended thinking on
    #     Anthropic (Claude 3.7 / 4.x) and Gemini thinking models. Without
    #     it, those providers return zero reasoning tokens even though the
    #     model is capable -- which is what caused "Opus 4.7 isn't
    #     reasoning" until we set this explicitly.
    #   - `summary: "auto"` opts OpenAI o-series / gpt-5 into a human-
    #     readable summary instead of the default encrypted blob. Ignored
    #     by providers that don't encrypt.
    #   - An explicit `effort` (or the summarizer's `"minimal"` override)
    #     wins over the implied medium effort from `enabled: true`.
    #   - Anthropic needs top-level `max_tokens` with effort-based
    #     reasoning; when callers don't provide one we set a conservative
    #     default above so OpenRouter can derive a valid reasoning budget.
    reasoning_cfg: dict[str, Any] = {"enabled": True, "summary": "auto"}
    if _is_anthropic_model(model):
        if reasoning_effort == "none":
            reasoning_cfg = {"effort": "none"}
        else:
            assert effective_max_tokens is not None
            reasoning_cfg["max_tokens"] = _anthropic_reasoning_max_tokens(
                max_tokens=effective_max_tokens,
                reasoning_effort=reasoning_effort,
            )
    elif reasoning_effort:
        reasoning_cfg["effort"] = reasoning_effort
    body["reasoning"] = reasoning_cfg
    return body


def _max_retries() -> int:
    """Number of *retries* (attempts beyond the first) for transient failures.

    Reads ``HARNESS_LLM_RETRIES`` on every call so a supervisor can tune it
    per-run; 0 disables retries entirely. Bad values fail loudly — a silently
    misparsed retry budget would only surface mid-incident.
    """
    raw = os.environ.get("HARNESS_LLM_RETRIES")
    if raw is None or raw.strip() == "":
        return _DEFAULT_LLM_RETRIES
    try:
        retries = int(raw)
    except ValueError:
        raise ValueError(
            f"HARNESS_LLM_RETRIES must be a non-negative integer, got {raw!r}"
        ) from None
    if retries < 0:
        raise ValueError(f"HARNESS_LLM_RETRIES must be a non-negative integer, got {raw!r}")
    return retries


def _is_retryable(exc: BaseException) -> bool:
    """True only when re-sending the identical request is safe and likely to help.

    - ``httpx.TransportError`` covers the whole flaky-network family:
      ReadError/WriteError/ConnectError/RemoteProtocolError plus the timeout
      subclasses (ConnectTimeout, ReadTimeout, PoolTimeout, ...).
    - HTTP 502/503/529 are OpenRouter/upstream "temporarily unavailable"
      responses. Other statuses (400/401/403/404/429) are deterministic or
      quota-related and keep their fail-fast behavior.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, OpenRouterError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


def _backoff_seconds(retry_number: int) -> float:
    """Exponential backoff (~1s/4s/10s for retries 1/2/3) with ±25% jitter."""
    base = min(_BACKOFF_BASE_SECONDS * (4 ** (retry_number - 1)), _BACKOFF_MAX_SECONDS)
    return base * random.uniform(0.75, 1.25)


def _request_with_retries(body: dict, *, timeout: httpx.Timeout, model: str) -> dict:
    """Run `_stream_chat_completion` with bounded retries on transient failures.

    The chat-completions POST is stateless, so a failed attempt — even one
    that died mid-stream after partial SSE chunks — can be re-sent verbatim;
    partial output is local to `_stream_chat_completion` and is discarded.
    After the retry budget is exhausted the last exception propagates
    unchanged, so callers see exactly the error they would have seen without
    retries.
    """
    retries = _max_retries()
    attempt = 0
    while True:
        attempt += 1
        try:
            return _stream_chat_completion(body, timeout=timeout, model=model)
        except Exception as e:
            if attempt > retries or not _is_retryable(e):
                raise
            wait = _backoff_seconds(attempt)
            logger.warning(
                "openrouter transient failure on attempt %d/%d, retrying in %.1fs "
                "(%s: %s) model=%s",
                attempt,
                retries + 1,
                wait,
                type(e).__name__,
                e,
                model,
            )
            _sleep(wait)


def _http() -> httpx.Client:
    global _client
    if _client is None:
        # Default client timeout is intentionally lenient; each call overrides
        # with an `httpx.Timeout` that splits connect / read / write phases.
        _client = httpx.Client(timeout=httpx.Timeout(60.0))
    return _client


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. LLM calls need it to reach OpenRouter.")
    return key


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float  # USD; 0.0 if OpenRouter didn't return a cost
    model: str = ""
    cached_tokens: int = 0
    # `cache_write_tokens` is populated only on the request that establishes
    # a new cache entry (Anthropic, Alibaba, Gemini explicit). Subsequent
    # cache hits report 0 here and a positive `cached_tokens`. Tracking it
    # separately keeps the per-turn span faithful to the OpenRouter usage
    # payload and lets cost-investigation queries distinguish "cold start"
    # turns (write) from "warm" turns (read).
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0  # populated only for thinking models (o1/gpt-5/claude-thinking)
    llm_calls: int = 1

    def to_dict(self) -> dict:
        """Full usage dict, suitable for `turn_N` / run-level `usage` metadata."""
        return {
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "cost_usd": self.total_cost,
            "total_cost_usd": self.total_cost,
            "llm_calls": self.llm_calls,
            "cache_hit_rate": (
                self.cached_tokens / self.prompt_tokens if self.prompt_tokens > 0 else 0.0
            ),
        }

    def to_llm_cost_dict(self) -> dict:
        """Compact per-call cost dict, suitable for LLM-span `llm_cost` metadata."""
        return {
            "model": self.model,
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_cost_usd": self.total_cost,
            "cache_hit_rate": (
                self.cached_tokens / self.prompt_tokens if self.prompt_tokens > 0 else 0.0
            ),
        }


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: Usage
    reasoning: str | None = None
    raw: dict = field(repr=False, default_factory=dict)


def complete(
    *,
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    timeout_seconds: float = 120.0,
) -> LLMResponse:
    """Send one chat completion request to OpenRouter.

    Args:
        model: OpenRouter model slug (e.g. "openai/gpt-4o-mini").
        system: system prompt text; prepended as a system-role message.
        messages: list of OpenAI-format message dicts (assistant, user, tool).
        tools: optional list of OpenAI-format tool (function) definitions.
        tool_choice: "auto" | "required" | "none" | {"type": "function", ...}.
            When `tools` is provided, the harness loop passes "required" so every
            turn emits at least one tool call. Pass None to let the provider
            default (usually "auto") take over.
        reasoning_effort: optional minimal|low|medium|high|xhigh for reasoning models.
        max_tokens: optional completion token budget. Anthropic reasoning models
            need this with effort-based reasoning; if omitted, Anthropic requests
            get a conservative default.
        timeout_seconds: request timeout.
    """
    body = _build_chat_completion_body(
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )
    model = body["model"]
    full_messages = body["messages"]

    logger.info(
        "openrouter complete() received: model=%s messages=%d tools=%s "
        "tool_choice=%s reasoning_effort=%s max_tokens=%s",
        model,
        len(messages),
        "None" if tools is None else f"len={len(tools)}",
        tool_choice,
        reasoning_effort or "-",
        body.get("max_tokens", "-"),
    )
    if tools:
        tool_names = [((t.get("function") or {}).get("name") or "?") for t in tools]
        logger.info("openrouter complete() tool names: %s", tool_names)

    # Stream the response. Reasoning models (gpt-5 thinking, o1, Claude with
    # extended thinking) commonly take minutes before emitting any non-streaming
    # body bytes, which looks indistinguishable from a deadlock. Streaming gives
    # us (a) SSE keepalive pings from OpenRouter (`: OPENROUTER PROCESSING`) so
    # the per-read timeout doesn't trip spuriously during the thinking phase,
    # and (b) token-by-token progress logs so a real hang is obvious.
    body["stream"] = True

    # Split timeouts so a stalled DNS/TLS handshake fails fast but a slow
    # provider generation doesn't. `read` is "max idle between bytes" — with
    # SSE keepalives every few seconds this is generously forgiving.
    timeout = httpx.Timeout(
        connect=15.0,
        read=timeout_seconds,
        write=30.0,
        pool=30.0,
    )

    body_bytes = len(json.dumps(body, default=str))
    logger.info(
        "openrouter POST start (stream) model=%s messages=%d tools=%d "
        "reasoning_effort=%s max_tokens=%s body_bytes=%d read_timeout=%.1fs",
        model,
        len(full_messages),
        len(tools) if tools else 0,
        reasoning_effort or "-",
        body.get("max_tokens", "-"),
        body_bytes,
        timeout_seconds,
    )
    t0 = time.monotonic()
    try:
        data = _request_with_retries(body, timeout=timeout, model=model)
    except httpx.TimeoutException as e:
        logger.error(
            "openrouter POST timeout after %.2fs model=%s (%s: %s)",
            time.monotonic() - t0,
            model,
            type(e).__name__,
            e,
        )
        raise
    except Exception as e:
        logger.exception(
            "openrouter POST failed after %.2fs model=%s (%s)",
            time.monotonic() - t0,
            model,
            type(e).__name__,
        )
        raise
    elapsed = time.monotonic() - t0
    logger.info(
        "openrouter POST done elapsed=%.2fs model=%s",
        elapsed,
        model,
    )

    choice = data["choices"][0]
    msg = choice.get("message", {})
    text = msg.get("content") or ""
    finish_reason = choice.get("finish_reason") or ""
    reasoning = _parse_reasoning(msg)

    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            logger.warning("tool_call had unparseable arguments: %r", raw_args)
            args = {"_raw": raw_args}
        tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), args=args))

    usage_data = data.get("usage") or {}
    completion_details = usage_data.get("completion_tokens_details") or {}
    prompt_details = usage_data.get("prompt_tokens_details") or {}
    usage = Usage(
        prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
        completion_tokens=int(usage_data.get("completion_tokens") or 0),
        total_tokens=int(usage_data.get("total_tokens") or 0),
        total_cost=_extract_total_cost(usage_data),
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
        cache_write_tokens=int(prompt_details.get("cache_write_tokens") or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        model=str(data.get("model") or model),
        llm_calls=1,
    )

    logger.info(
        "openrouter response finish=%s tool_calls=%d "
        "tokens in=%d out=%d cached=%d cache_write=%d reasoning=%d cost=$%.5f",
        finish_reason or "-",
        len(tool_calls),
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cached_tokens,
        usage.cache_write_tokens,
        usage.reasoning_tokens,
        usage.total_cost,
    )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        reasoning=reasoning,
        raw=data,
    )


def _stream_chat_completion(body: dict, *, timeout: httpx.Timeout, model: str) -> dict:
    """POST with `stream: true` and reassemble SSE deltas into a dict that
    matches the non-streaming response shape the rest of `complete()` expects.

    Handles OpenRouter-specific quirks:
      - Keepalive SSE comments (`: OPENROUTER PROCESSING`) arrive every few
        seconds while the upstream provider is thinking; we log them at DEBUG
        and reset a "last progress" timer so a real stall is visible.
      - Tool call fragments stream as `delta.tool_calls[i].function.arguments`
        chunks that must be concatenated by `index` (not by id, which only
        appears on the first fragment).
      - Usage arrives in a final delta after `finish_reason`, or alongside it
        on the last choice chunk.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_details: list[dict] = []
    tool_calls_by_index: dict[int, dict] = {}
    finish_reason: str = ""
    usage: dict = {}
    response_model: str = model
    response_id: str = ""
    chunk_count = 0
    last_progress = time.monotonic()

    with _http().stream(
        "POST",
        _OPENROUTER_URL,
        json=body,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        timeout=timeout,
    ) as resp:
        logger.info(
            "openrouter stream opened status=%d model=%s",
            resp.status_code,
            model,
        )
        if resp.status_code >= 400:
            body_bytes = resp.read()
            body_text = body_bytes[:4000].decode("utf-8", errors="replace")
            logger.error(
                "openrouter stream error status=%d body=%s",
                resp.status_code,
                body_text,
            )
            # Default httpx.HTTPStatusError only shows the URL + status, which
            # drops the useful part — OpenRouter returns a JSON `{error: {...}}`
            # body on 4xx/5xx explaining *why* (invalid model, quota, schema
            # mismatch, etc.). Raise our own error so the span's error field
            # (captured from `str(e)`) includes the body.
            raise OpenRouterError(
                status_code=resp.status_code,
                body=body_text,
                model=model,
            )

        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(":"):
                # SSE comment / keepalive. Reset progress timer so we don't
                # log a "no progress" warning during legit reasoning phases.
                logger.debug("openrouter keepalive: %s", line[:120])
                last_progress = time.monotonic()
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("openrouter bad SSE payload: %r", payload[:200])
                continue

            chunk_count += 1
            if chunk.get("id") and not response_id:
                response_id = chunk["id"]
            if chunk.get("model"):
                response_model = chunk["model"]
            if chunk.get("usage"):
                usage = chunk["usage"]

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}

                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)

                reasoning_piece = delta.get("reasoning")
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    reasoning_parts.append(reasoning_piece)

                details = delta.get("reasoning_details")
                if isinstance(details, list):
                    reasoning_details.extend(d for d in details if isinstance(d, dict))

                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    entry = tool_calls_by_index.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc_delta.get("id"):
                        entry["id"] = tc_delta["id"]
                    if tc_delta.get("type"):
                        entry["type"] = tc_delta["type"]
                    fn_delta = tc_delta.get("function") or {}
                    if fn_delta.get("name"):
                        entry["function"]["name"] = fn_delta["name"]
                    if "arguments" in fn_delta:
                        entry["function"]["arguments"] += fn_delta["arguments"] or ""

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            now = time.monotonic()
            if now - last_progress > 5.0:
                logger.info(
                    "openrouter streaming progress chunks=%d "
                    "content_chars=%d reasoning_chars=%d tool_calls=%d",
                    chunk_count,
                    sum(len(p) for p in content_parts),
                    sum(len(p) for p in reasoning_parts),
                    len(tool_calls_by_index),
                )
                last_progress = now

    tool_calls_list = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    message: dict[str, Any] = {"role": "assistant"}
    message["content"] = "".join(content_parts) if content_parts else None
    if tool_calls_list:
        message["tool_calls"] = tool_calls_list
    if reasoning_parts:
        message["reasoning"] = "".join(reasoning_parts)
    if reasoning_details:
        message["reasoning_details"] = _merge_streamed_reasoning_details(reasoning_details)

    return {
        "id": response_id,
        "model": response_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _merge_streamed_reasoning_details(details: list[dict]) -> list[dict]:
    """Collapse streaming reasoning deltas into replayable reasoning blocks.

    OpenRouter streams Anthropic ``reasoning_details`` as small deltas: several
    ``reasoning.text`` chunks with the same index, followed by a signature-only
    chunk. Replaying that fragmented sequence makes Anthropic reject the next
    tool-continuation request with ``Invalid signature in thinking block``.
    The chat history needs the same logical block OpenRouter would return from
    a non-streaming response: concatenated text plus the final signature.
    """
    merged: list[dict] = []
    by_key: dict[tuple[int | None, str | None, str | None], dict] = {}

    for detail in details:
        if not isinstance(detail, dict):
            continue
        key = (
            detail.get("index") if isinstance(detail.get("index"), int) else None,
            detail.get("type") if isinstance(detail.get("type"), str) else None,
            detail.get("format") if isinstance(detail.get("format"), str) else None,
        )
        block = by_key.get(key)
        if block is None:
            block = {
                k: v
                for k, v in detail.items()
                if k not in {"text", "summary", "content", "data", "signature"}
            }
            by_key[key] = block
            merged.append(block)

        for text_key in ("text", "summary", "content", "data"):
            value = detail.get(text_key)
            if isinstance(value, str) and value:
                block[text_key] = f"{block.get(text_key, '')}{value}"

        signature = detail.get("signature")
        if isinstance(signature, str) and signature:
            block["signature"] = signature

    return merged


def _extract_total_cost(usage_data: dict) -> float:
    """Pick the right USD cost field from an OpenRouter usage payload.

    OpenRouter reports cost differently depending on how the request was
    billed:

    - **Platform-billed (``is_byok: false``).** ``cost`` is the total USD
      drawn from the user's OpenRouter credits and includes any provider
      markup. ``cost_details.upstream_inference_cost`` mirrors the same
      number minus the markup.
    - **BYOK (``is_byok: true``).** The user paid the upstream provider
      directly with their own key, so OpenRouter reports ``cost: 0`` --
      it didn't bill anything. The actual spend with the upstream
      provider is in ``cost_details.upstream_inference_cost``. Reading
      only the top-level ``cost`` made every BYOK turn report ``$0`` on
      both the per-call ``llm_cost`` span metadata and the run-level
      usage rollup, which silently zeroed out cost tracking on any
      account using BYOK for one of its providers (e.g. an Anthropic
      key plumbed in via OpenRouter).

    Prefer the top-level ``cost`` when it's positive (so platform-billed
    calls keep including the OpenRouter markup), otherwise fall back to
    ``cost_details.upstream_inference_cost`` so BYOK runs still surface
    real spend.
    """
    direct = float(usage_data.get("cost") or 0.0)
    if direct > 0:
        return direct
    cost_details = usage_data.get("cost_details") or {}
    return float(cost_details.get("upstream_inference_cost") or 0.0)


def _parse_reasoning(msg: dict) -> str | None:
    """Extract plain-text reasoning from an OpenRouter chat/completions message.

    Two shapes show up in practice:
      - `message.reasoning`: a top-level string (Anthropic, some Gemini, and
        OpenAI when the reasoning isn't encrypted).
      - `message.reasoning_details`: a list of structured blocks; summaries
        carry the plain text even when the raw reasoning is encrypted
        (OpenAI o1/gpt-5 with `summary: "auto"`).

    Returns None when we only got encrypted content.
    """
    direct = msg.get("reasoning")
    if isinstance(direct, str) and direct.strip():
        return direct

    details = msg.get("reasoning_details") or []
    summaries: list[str] = []
    for d in details:
        if not isinstance(d, dict):
            continue
        # Plain-text variants
        for key in ("summary", "text", "content"):
            val = d.get(key)
            if isinstance(val, str) and val.strip() and val.strip() != "[redacted]":
                summaries.append(val)
                break
    if summaries:
        return "\n\n".join(summaries)
    return None


def _prepare_replay_messages(messages: list[dict]) -> list[dict]:
    """Return chat messages safe to replay into a new OpenRouter request."""
    return _drop_orphan_tool_messages(_strip_provider_reasoning(messages))


def _strip_provider_reasoning(messages: list[dict]) -> list[dict]:
    """Remove non-replayable thinking fields from assistant messages.

    ``reasoning`` is plaintext trace output, not chat history. Preserve
    ``reasoning_details`` though: OpenRouter's tool-use guidance requires
    replaying those signed blocks verbatim so Anthropic can continue thinking
    after tool results.
    """
    sanitized: list[dict] = []
    dropped = 0

    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue

        clean = dict(msg)
        if "reasoning" in clean:
            clean.pop("reasoning", None)
            dropped += 1

        content = clean.get("content")
        if isinstance(content, list):
            filtered_content = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in {"thinking", "redacted_thinking", "reasoning"}
                )
            ]
            if len(filtered_content) != len(content):
                clean["content"] = filtered_content
                dropped += len(content) - len(filtered_content)

        sanitized.append(clean)

    if dropped:
        logger.warning(
            "Stripped %d provider reasoning field/block(s) from replay context",
            dropped,
        )

    return sanitized


def _drop_orphan_tool_messages(messages: list[dict]) -> list[dict]:
    """Remove ``role: tool`` messages whose ``tool_call_id`` has no prior call.

    OpenAI Chat Completions (and OpenRouter, which speaks the same shape) is
    strict about tool-call/tool-result pairing: every ``role: tool`` message
    must reference a ``tool_calls[*].id`` from an earlier ``role: assistant``
    message in the same request, or the API rejects the whole call with
    ``messages.*.tool_call_id: tool message has no matching tool call``.

    Memory replay can violate that invariant in two ways:
      - Summarisation/trimming drops the assistant turn that issued the
        original ``tool_calls`` while the matching ``tool`` row is still
        within the recent-message window.
      - Provider switches (Claude tool_use blocks ↔ OpenAI tool_calls)
        leave historical results whose original call shape is no longer
        replayable.

    Dropping only the orphan ``tool`` rows preserves every valid tool
    exchange and keeps replay robust across context windows. The orphan
    set is logged at warn so the underlying summariser/trimming bug stays
    visible if it ever spikes.
    """
    seen_call_ids: set[str] = set()
    filtered: list[dict] = []
    dropped = 0

    for msg in messages:
        if not isinstance(msg, dict):
            filtered.append(msg)
            continue

        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if isinstance(tc_id, str) and tc_id:
                    seen_call_ids.add(tc_id)
            filtered.append(msg)
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id in seen_call_ids:
                filtered.append(msg)
            else:
                dropped += 1
            continue

        filtered.append(msg)

    if dropped:
        logger.warning(
            "Dropped %d orphan tool message(s) from replay context "
            "(no matching assistant.tool_calls[].id earlier in the window)",
            dropped,
        )

    return filtered
