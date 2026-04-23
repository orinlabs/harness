"""Chunk 4 verification: real OpenRouter API calls."""
from __future__ import annotations

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
    assert (
        llm._translate_model("claude-sonnet-4-5-20250929")
        == "anthropic/claude-sonnet-4.5"
    )
    assert (
        llm._translate_model("claude-opus-4-1-20250805")
        == "anthropic/claude-opus-4.1"
    )

    # Already-namespaced slugs are pass-through (don't re-prefix).
    assert llm._translate_model("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert (
        llm._translate_model("anthropic/claude-opus-4.7")
        == "anthropic/claude-opus-4.7"
    )
    assert llm._translate_model("google/gemini-2.0-flash") == "google/gemini-2.0-flash"

    # Unknown / non-Anthropic bare slugs are also pass-through; let
    # OpenRouter decide whether they resolve.
    assert llm._translate_model("gpt-4o-mini") == "gpt-4o-mini"


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
    assert resp.usage.total_cost > 0


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
