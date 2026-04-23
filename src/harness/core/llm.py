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


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=120.0)
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
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    body: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
        "usage": {"include": True},
    }
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    # Always request plain-text reasoning summaries. OpenAI reasoning models
    # (o1, gpt-5) encrypt raw reasoning by default; `summary: "auto"` opts
    # into a human-readable summary. No-op on non-reasoning models.
    reasoning_cfg: dict[str, Any] = {"summary": "auto"}
    if reasoning_effort:
        reasoning_cfg["effort"] = reasoning_effort
    body["reasoning"] = reasoning_cfg

    body_bytes = len(json.dumps(body, default=str))
    logger.info(
        "openrouter POST start model=%s messages=%d tools=%d "
        "reasoning_effort=%s body_bytes=%d timeout=%.1fs",
        model,
        len(full_messages),
        len(tools) if tools else 0,
        reasoning_effort or "-",
        body_bytes,
        timeout_seconds,
    )
    t0 = time.monotonic()
    try:
        resp = _http().post(
            _OPENROUTER_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )
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
        "openrouter POST done status=%d elapsed=%.2fs model=%s",
        resp.status_code,
        elapsed,
        model,
    )
    resp.raise_for_status()
    data = resp.json()

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
