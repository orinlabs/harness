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
