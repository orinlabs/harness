"""Chunk 4 verification: real OpenRouter API calls."""

from __future__ import annotations

import pytest

CHEAP_MODEL = "openai/gpt-4o-mini"


def test_plain_completion_returns_text_and_cost(openrouter_key):
    from harness.core import llm

    resp = llm.complete(
        model=CHEAP_MODEL,
        system="You are a test assistant. Reply with exactly the word OK.",
        messages=[{"role": "user", "content": "say OK"}],
    )

    assert "OK" in resp.text.upper()
    assert resp.finish_reason in ("stop", "length")
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    assert resp.usage.total_cost > 0, "OpenRouter should return cost when include=True"
    assert resp.tool_calls == []


def test_tool_call_is_parsed(openrouter_key):
    from harness.core import llm

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time in a timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {"tz": {"type": "string"}},
                    "required": ["tz"],
                },
            },
        }
    ]

    resp = llm.complete(
        model=CHEAP_MODEL,
        system="You have a tool called get_time. When asked about the time, you must call it.",
        messages=[{"role": "user", "content": "What time is it in UTC? Use the tool."}],
        tools=tools,
    )

    assert len(resp.tool_calls) >= 1, f"expected a tool call, got text: {resp.text!r}"
    tc = resp.tool_calls[0]
    assert tc.name == "get_time"
    assert isinstance(tc.args, dict)
    assert "tz" in tc.args
    assert resp.usage.total_cost > 0


def test_translate_model_maps_bedrock_ids_to_openrouter_slugs():
    """The bedrock platform hands us Anthropic API IDs; OpenRouter 400s on
    those directly. `_translate_model` normalises them to the namespaced slug.

    Regression: passing `claude-opus-4-7` (unmapped) to OpenRouter returned
    "HTTPStatusError: Client error '400 Bad Request'" from the chat endpoint.
    """
    from harness.core import llm

    # Current flagship models — translate to namespaced slugs.
    assert llm._translate_model("claude-opus-4-7") == "anthropic/claude-opus-4.7"
    assert llm._translate_model("claude-opus-4-6") == "anthropic/claude-opus-4.6"
    assert llm._translate_model("claude-sonnet-4-6") == "anthropic/claude-sonnet-4.6"
    assert llm._translate_model("claude-sonnet-4-5") == "anthropic/claude-sonnet-4.5"
    assert llm._translate_model("claude-haiku-4-5") == "anthropic/claude-haiku-4.5"

    # Dated bedrock aliases (what you get back from the Claude API /models
    # endpoint and from AWS Bedrock sometimes) still resolve.
    assert llm._translate_model("claude-sonnet-4-5-20250929") == "anthropic/claude-sonnet-4.5"
    assert llm._translate_model("claude-opus-4-1-20250805") == "anthropic/claude-opus-4.1"

    # Already-namespaced slugs are pass-through (don't re-prefix).
    assert llm._translate_model("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert llm._translate_model("anthropic/claude-opus-4.7") == "anthropic/claude-opus-4.7"
    assert llm._translate_model("google/gemini-2.0-flash") == "google/gemini-2.0-flash"

    # Unknown / non-Anthropic bare slugs are also pass-through; let
    # OpenRouter decide whether they resolve.
    assert llm._translate_model("gpt-4o-mini") == "gpt-4o-mini"


def test_anthropic_reasoning_requests_set_default_max_tokens():
    """Anthropic effort-based reasoning needs top-level max_tokens.

    Without this, OpenRouter has to infer the reasoning budget from provider
    defaults, and Anthropic can reject or silently under-budget thinking.
    """
    from harness.core import llm

    body = llm._build_chat_completion_body(
        model="claude-haiku-4-5",
        system="",
        messages=[{"role": "user", "content": "think"}],
        reasoning_effort="high",
    )

    assert body["model"] == "anthropic/claude-haiku-4.5"
    assert body["max_tokens"] == 8192
    assert body["reasoning"] == {
        "enabled": True,
        "summary": "auto",
        "max_tokens": 6553,
    }


def test_anthropic_reasoning_requests_honor_configured_max_tokens():
    from harness.core import llm

    body = llm._build_chat_completion_body(
        model="anthropic/claude-opus-4.7",
        system="",
        messages=[{"role": "user", "content": "think"}],
        reasoning_effort="xhigh",
        max_tokens=32768,
    )

    assert body["max_tokens"] == 32768
    assert body["reasoning"]["max_tokens"] == 31129


def test_anthropic_reasoning_rejects_max_tokens_at_minimum_budget():
    from harness.core import llm

    with pytest.raises(ValueError, match="requires max_tokens > 1024"):
        llm._build_chat_completion_body(
            model="anthropic/claude-sonnet-4.6",
            system="",
            messages=[{"role": "user", "content": "think"}],
            reasoning_effort="low",
            max_tokens=1024,
        )


def test_non_anthropic_reasoning_does_not_invent_max_tokens():
    from harness.core import llm

    body = llm._build_chat_completion_body(
        model=CHEAP_MODEL,
        system="",
        messages=[{"role": "user", "content": "think"}],
        reasoning_effort="minimal",
    )

    assert "max_tokens" not in body
    assert body["reasoning"]["effort"] == "minimal"


def test_complete_translates_anthropic_slug_before_sending(openrouter_key):
    """End-to-end: calling `complete()` with a bedrock-style Anthropic ID
    must succeed (not 400), because translation runs before the HTTP call.

    Uses the cheapest Claude model we can map (Haiku 4.5) to keep the bill
    tiny while still exercising the real code path a 400-producing request
    would hit.
    """
    from harness.core import llm

    resp = llm.complete(
        model="claude-haiku-4-5",
        system="You are a test assistant. Reply with exactly the word OK.",
        messages=[{"role": "user", "content": "say OK"}],
    )

    # Before the fix, this raised `HTTPStatusError: 400 Bad Request` because
    # OpenRouter doesn't recognise the bare Anthropic ID. Succeeding here is
    # the real assertion.
    assert "OK" in resp.text.upper()
    # Echoed model must be in the anthropic namespace (OpenRouter may
    # canonicalise the slug, e.g. `anthropic/claude-4.5-haiku-20251001`).
    assert resp.usage.model.startswith("anthropic/"), (
        f"expected openrouter to route to an anthropic model, got {resp.usage.model!r}"
    )
    assert "haiku" in resp.usage.model.lower()
    # We deliberately do NOT assert total_cost > 0 here. OpenRouter
    # occasionally canonicalises Haiku 4.5 to a date-pinned variant
    # (e.g. `anthropic/claude-4.5-haiku-20251001`) whose pricing row
    # hasn't been populated yet; the call still succeeds and returns
    # tokens, but `usage.cost` comes back as 0. The dedicated cost
    # assertion lives in `test_plain_completion_returns_text_and_cost`
    # against gpt-4o-mini, which has stable pricing.
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0


def test_multi_turn_message_history(openrouter_key):
    """Prove messages are threaded through correctly (not just system+last)."""
    from harness.core import llm

    resp = llm.complete(
        model=CHEAP_MODEL,
        system="You are concise.",
        messages=[
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "What number did I say? Reply with only the digits."},
        ],
    )

    assert "42" in resp.text
    assert resp.usage.total_cost > 0


def test_drop_orphan_tool_messages_filters_unmatched_results():
    """`role: tool` rows whose ``tool_call_id`` isn't in any prior
    ``assistant.tool_calls[].id`` get filtered out, while the rest of the
    log is preserved untouched.

    Regression: when summarisation/trimming evicts the assistant turn
    that issued ``tool_calls``, the matching ``role: tool`` row becomes
    an orphan. OpenRouter (and OpenAI Chat Completions) reject the entire
    request with ``messages.*.tool_call_id: tool message has no matching
    tool call``. Dropping only the orphan preserves valid exchanges.
    """
    from harness.core import llm

    messages = [
        {"role": "tool", "tool_call_id": "missing_call", "content": "stale"},
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "think", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "ok"},
    ]

    filtered = llm._drop_orphan_tool_messages(messages)

    assert [m["role"] for m in filtered] == ["user", "assistant", "tool"]
    assert filtered[0]["content"] == "continue"
    assert filtered[2]["tool_call_id"] == "call_123"


def test_drop_orphan_tool_messages_passthrough_when_paired():
    """Sanity check: a well-formed log with paired tool calls/results is
    returned unchanged."""
    from harness.core import llm

    messages = [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "y", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "x done"},
        {"role": "tool", "tool_call_id": "call_b", "content": "y done"},
        {"role": "assistant", "content": "all done"},
    ]

    filtered = llm._drop_orphan_tool_messages(messages)

    assert filtered == messages


def test_prepare_replay_messages_preserves_reasoning_details_for_tool_replay():
    """Signed reasoning_details are required for Anthropic tool continuations.

    Top-level ``reasoning`` is trace output and should not be replayed, but
    OpenRouter documents ``reasoning_details`` as the continuity mechanism for
    reasoning models when a tool call pauses the response.
    """
    from harness.core import llm

    messages = [
        {"role": "user", "content": "check state"},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "I should inspect notifications.",
            "reasoning_details": [
                {"type": "reasoning.text", "text": "I should inspect notifications."},
                {"type": "reasoning.text", "signature": "provider-signature"},
            ],
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "type": "function",
                    "function": {"name": "list_notifications", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_123", "content": "none"},
    ]

    filtered = llm._prepare_replay_messages(messages)

    assert filtered[1] == {
        "role": "assistant",
        "content": None,
        "reasoning_details": [
            {"type": "reasoning.text", "text": "I should inspect notifications."},
            {"type": "reasoning.text", "signature": "provider-signature"},
        ],
        "tool_calls": [
            {
                "id": "toolu_123",
                "type": "function",
                "function": {"name": "list_notifications", "arguments": "{}"},
            }
        ],
    }
    assert filtered[2]["tool_call_id"] == "toolu_123"


def test_merge_streamed_reasoning_details_combines_text_with_signature():
    from harness.core import llm

    merged = llm._merge_streamed_reasoning_details(
        [
            {
                "type": "reasoning.text",
                "text": "first ",
                "format": "anthropic-claude-v1",
                "index": 0,
            },
            {
                "type": "reasoning.text",
                "text": "second",
                "format": "anthropic-claude-v1",
                "index": 0,
            },
            {
                "type": "reasoning.text",
                "signature": "signed",
                "format": "anthropic-claude-v1",
                "index": 0,
            },
        ]
    )

    assert merged == [
        {
            "type": "reasoning.text",
            "format": "anthropic-claude-v1",
            "index": 0,
            "text": "first second",
            "signature": "signed",
        }
    ]


def test_prepare_replay_messages_strips_anthropic_thinking_content_blocks():
    """Anthropic thinking blocks can also appear inside message content lists."""
    from harness.core import llm

    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "secret", "signature": "bad"},
                {"type": "text", "text": "Visible answer"},
            ],
        }
    ]

    assert llm._prepare_replay_messages(messages) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Visible answer"}]}
    ]
