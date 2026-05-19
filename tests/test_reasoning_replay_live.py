"""Live OpenRouter: replay history with full reasoning preserved.

Run:
  uv run pytest tests/test_reasoning_replay_live.py -v -s

Requires OPENROUTER_API_KEY.
"""

from __future__ import annotations

from typing import Any

import pytest

from harness.core import llm

PROVIDER_MODELS: list[tuple[str, str]] = [
    ("anthropic", "anthropic/claude-haiku-4.5"),
    ("openai", "openai/gpt-5-nano"),
    ("gemini", "google/gemini-2.5-flash"),
]

SLEEP_TOOL = {
    "type": "function",
    "function": {
        "name": "sleep",
        "description": "Stop the agent loop when finished.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}

SYSTEM = (
    "You are a test assistant. Think step by step before acting. "
    "You must call the sleep tool when done explaining your plan."
)

USER_TURN1 = (
    "Before calling any tool, reason about what you will do. "
    "Then call the sleep tool with empty arguments."
)


def _first_tool_call_turn(model: str) -> tuple[dict[str, Any], llm.LLMResponse]:
    resp = llm.complete(
        model=model,
        system=SYSTEM,
        messages=[{"role": "user", "content": USER_TURN1}],
        tools=[SLEEP_TOOL],
        tool_choice="auto",
        reasoning_effort="low",
        timeout_seconds=180.0,
    )
    assistant = resp.raw["choices"][0]["message"]
    if not resp.tool_calls:
        pytest.fail(
            f"{model}: turn-1 expected a tool call, got finish={resp.finish_reason!r} "
            f"text={resp.text[:200]!r}"
        )
    return assistant, resp


@pytest.mark.parametrize("provider,model", PROVIDER_MODELS)
def test_replay_preserves_reasoning_fields(openrouter_key, provider: str, model: str):
    """_prepare_replay_messages leaves assistant reasoning intact."""
    assistant, _ = _first_tool_call_turn(model)
    tc_id = assistant.get("tool_calls", [{}])[0].get("id", "toolu_test")
    history = [
        {"role": "user", "content": USER_TURN1},
        assistant,
        {"role": "tool", "tool_call_id": tc_id, "content": '{"ok": true}'},
    ]
    prepared = llm._prepare_replay_messages(history)
    assert prepared[1] == assistant


@pytest.mark.parametrize("provider,model", PROVIDER_MODELS)
def test_post_tool_replay_with_full_reasoning(openrouter_key, provider: str, model: str):
    """Continuation after a tool result succeeds with reasoning in history."""
    assistant, turn1 = _first_tool_call_turn(model)
    tc = turn1.tool_calls[0]
    history = [
        {"role": "user", "content": USER_TURN1},
        assistant,
        {"role": "tool", "tool_call_id": tc.id, "content": '{"ok": true}'},
    ]
    resp = llm.complete(
        model=model,
        system=SYSTEM,
        messages=history,
        tools=[SLEEP_TOOL],
        tool_choice="auto",
        reasoning_effort="low",
        timeout_seconds=180.0,
    )
    print(
        f"\n{provider}: finish={resp.finish_reason} "
        f"reasoning_tokens={resp.usage.reasoning_tokens} "
        f"completion_tokens={resp.usage.completion_tokens}"
    )
    # Replay success means no 400; some providers return stop with 0 completion
    # tokens on short post-tool continuations.
    assert resp.usage.prompt_tokens > 0
