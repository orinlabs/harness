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
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        super().__init__(
            f"openrouter HTTP {status_code} for model {model}: {body}"
        )


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
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. LLM calls need it to reach OpenRouter."
        )
    return key


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float  # USD; 0.0 if OpenRouter didn't return a cost
    model: str = ""
    cached_tokens: int = 0
    reasoning_tokens: int = 0  # populated only for thinking models (o1/gpt-5/claude-thinking)
    llm_calls: int = 1

    def to_dict(self) -> dict:
        """Full usage dict, suitable for `turn_N` / run-level `usage` metadata."""
        return {
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
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
        reasoning_effort: optional "low" | "medium" | "high" for reasoning models.
        timeout_seconds: request timeout.
    """
    # Translate Anthropic/bedrock model IDs (`claude-opus-4-7`) into
    # OpenRouter slugs (`anthropic/claude-opus-4.7`). OpenRouter 400s on the
    # un-namespaced form. No-op for OpenAI/Google/etc. slugs.
    model = _translate_model(model)

    logger.info(
        "openrouter complete() received: model=%s messages=%d tools=%s "
        "tool_choice=%s reasoning_effort=%s",
        model,
        len(messages),
        "None" if tools is None else f"len={len(tools)}",
        tool_choice,
        reasoning_effort or "-",
    )
    if tools:
        tool_names = [
            ((t.get("function") or {}).get("name") or "?") for t in tools
        ]
        logger.info("openrouter complete() tool names: %s", tool_names)

    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(_drop_orphan_tool_messages(messages))

    body: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
        "usage": {"include": True},
    }
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
    reasoning_cfg: dict[str, Any] = {"enabled": True, "summary": "auto"}
    if reasoning_effort:
        reasoning_cfg["effort"] = reasoning_effort
    body["reasoning"] = reasoning_cfg

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
        "reasoning_effort=%s body_bytes=%d read_timeout=%.1fs",
        model,
        len(full_messages),
        len(tools) if tools else 0,
        reasoning_effort or "-",
        body_bytes,
        timeout_seconds,
    )
    t0 = time.monotonic()
    try:
        data = _stream_chat_completion(body, timeout=timeout, model=model)
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
        tool_calls.append(
            ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), args=args)
        )

    usage_data = data.get("usage") or {}
    completion_details = usage_data.get("completion_tokens_details") or {}
    usage = Usage(
        prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
        completion_tokens=int(usage_data.get("completion_tokens") or 0),
        total_tokens=int(usage_data.get("total_tokens") or 0),
        total_cost=float(usage_data.get("cost") or 0.0),
        cached_tokens=int(
            (usage_data.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        ),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        model=str(data.get("model") or model),
        llm_calls=1,
    )

    logger.info(
        "openrouter response finish=%s tool_calls=%d "
        "tokens in=%d out=%d cached=%d reasoning=%d cost=$%.5f",
        finish_reason or "-",
        len(tool_calls),
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cached_tokens,
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


def _stream_chat_completion(
    body: dict, *, timeout: httpx.Timeout, model: str
) -> dict:
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
        message["reasoning_details"] = reasoning_details

    return {
        "id": response_id,
        "model": response_model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": usage,
    }


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
