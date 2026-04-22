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
    raw: dict = field(repr=False)


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
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}

    resp = _http().post(
        _OPENROUTER_URL,
        json=body,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    msg = choice.get("message", {})
    text = msg.get("content") or ""
    finish_reason = choice.get("finish_reason") or ""

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
    usage = Usage(
        prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
        completion_tokens=int(usage_data.get("completion_tokens") or 0),
        total_tokens=int(usage_data.get("total_tokens") or 0),
        total_cost=float(usage_data.get("cost") or 0.0),
    )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        raw=data,
    )
