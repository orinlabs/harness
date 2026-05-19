"""Wire tests for the summarizer + sanitize integration.

Asserts:
  - Base64 image bytes never reach the summarizer LLM prompt.
  - The ``summarizer_call`` span carries ``sanitized`` / ``content_chars``
    bookkeeping when stripping ran.
  - OpenRouter context-length 400s flag ``context_overflow`` on the span
    instead of crashing the run.

All tests stub ``harness.core.llm.complete`` so no network is required.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime

import pytest

from harness.core.llm import LLMResponse, OpenRouterError, Usage
from harness.core.tracing import InMemoryTraceSink
from harness.memory.summarizer import SummaryUpdater, _is_context_overflow_error
from harness.memory.types import PeriodType

NS_PER_MINUTE = 60 * 1_000_000_000


def _ok_response(text: str = "I worked with the user.") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            total_cost=0.0,
            model="openai/gpt-5-nano",
            llm_calls=1,
        ),
    )


def _insert_message(storage_env, ts_ns: int, msg: dict) -> None:
    storage_env.db.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        (str(uuid.uuid4()), ts_ns, str(msg.get("role") or "user"), json.dumps(msg)),
    )


def _big_base64_user_message() -> dict:
    """Mirror of the followup ``_build_image_followup_message`` emits."""
    payload = "B" * 50_000  # 50 KB; uniquely identifiable in captured prompts
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "[Image attachment from tool 'browser_screenshot'.]",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            },
        ],
    }


def _install_inmemory_tracer(monkeypatch) -> InMemoryTraceSink:
    from harness.core import tracer

    sink = InMemoryTraceSink()
    tracer.set_trace_sink(sink)
    return sink


@pytest.fixture
def six_min_ago_ns() -> int:
    """A timestamp guaranteed to land in a completed 5-minute bucket."""
    return time.time_ns() - 6 * NS_PER_MINUTE


def test_five_minute_summary_strips_base64_before_llm(
    storage_env, monkeypatch, six_min_ago_ns
):
    captured: list[dict] = []

    def fake_complete(**kwargs):
        captured.append(kwargs)
        return _ok_response()

    from harness.core import llm as llm_mod

    monkeypatch.setattr(llm_mod, "complete", fake_complete)
    _install_inmemory_tracer(monkeypatch)

    _insert_message(
        storage_env,
        six_min_ago_ns,
        {"role": "user", "content": "Look at this screenshot."},
    )
    _insert_message(storage_env, six_min_ago_ns + 1, _big_base64_user_message())
    _insert_message(
        storage_env,
        six_min_ago_ns + 2,
        {
            "role": "assistant",
            "content": "I see the home page.",
            "reasoning_details": [
                {
                    "type": "reasoning.text",
                    "text": "There's a login button.",
                    "signature": "SIG" * 1000,
                }
            ],
        },
    )

    updater = SummaryUpdater(model="openai/gpt-5-nano")
    updater._update_five_minute_summaries(datetime.now(UTC))

    assert len(captured) == 1, "expected exactly one summarizer LLM call"
    prompt = captured[0]["messages"][0]["content"]
    assert "data:image/png;base64," not in prompt
    assert "B" * 50_000 not in prompt
    # The semantic envelope survives: label, tool output text, reasoning text.
    assert "browser_screenshot" in prompt
    assert "home page" in prompt
    assert "login button" in prompt
    # Signatures get scrubbed.
    assert "SIG" * 100 not in prompt


def test_summarizer_span_records_sanitization_metadata(
    storage_env, monkeypatch, six_min_ago_ns
):
    sink = _install_inmemory_tracer(monkeypatch)
    from harness.core import llm as llm_mod

    monkeypatch.setattr(llm_mod, "complete", lambda **_: _ok_response())

    _insert_message(storage_env, six_min_ago_ns, _big_base64_user_message())
    _insert_message(storage_env, six_min_ago_ns + 1, _big_base64_user_message())

    updater = SummaryUpdater(model="openai/gpt-5-nano")
    updater._update_five_minute_summaries(datetime.now(UTC))

    summarizer_spans = [
        s for s in sink.spans_closed.values() if s.get("metadata", {}).get(
            "period_type"
        ) == PeriodType.FIVE_MINUTE.value
    ]
    assert len(summarizer_spans) == 1
    metadata = summarizer_spans[0]["metadata"]
    assert metadata["sanitized"] is True
    assert metadata["stripped_image_parts"] == 2
    assert metadata["content_chars"] < 5000, (
        f"content_chars unexpectedly large after stripping: {metadata['content_chars']}"
    )


def test_openrouter_context_overflow_flagged_on_span(
    storage_env, monkeypatch, six_min_ago_ns
):
    sink = _install_inmemory_tracer(monkeypatch)
    from harness.core import llm as llm_mod

    body = (
        '{"error":{"message":"This endpoint\'s maximum context length is 400000 '
        'tokens. However, you requested about 601261 tokens. Please reduce '
        'the length of either one, or use the context-compression plugin to '
        'compress your prompt automatically.","code":400}}'
    )

    def boom(**_):
        raise OpenRouterError(status_code=400, body=body, model="openai/gpt-5-nano")

    monkeypatch.setattr(llm_mod, "complete", boom)

    _insert_message(
        storage_env,
        six_min_ago_ns,
        {"role": "user", "content": "a normal text message"},
    )

    updater = SummaryUpdater(model="openai/gpt-5-nano")
    # Should not raise; should return [] (or whatever the no-summary path
    # produces) and stamp the span with context_overflow=True.
    updater._update_five_minute_summaries(datetime.now(UTC))

    summarizer_spans = [
        s for s in sink.spans_closed.values() if s.get("metadata", {}).get(
            "period_type"
        ) == PeriodType.FIVE_MINUTE.value
    ]
    assert len(summarizer_spans) == 1
    metadata = summarizer_spans[0]["metadata"]
    assert metadata.get("context_overflow") is True
    assert metadata.get("openrouter_status") == 400
    assert "601261" in metadata.get("openrouter_error_body", "")


def test_is_context_overflow_error_detects_known_phrases():
    overflow = OpenRouterError(
        status_code=400,
        body='{"error":{"message":"maximum context length is 400000 tokens."}}',
        model="openai/gpt-5-nano",
    )
    assert _is_context_overflow_error(overflow) is True

    plugin_hint = OpenRouterError(
        status_code=400,
        body='{"error":{"message":"use the context-compression plugin"}}',
        model="openai/gpt-5-nano",
    )
    assert _is_context_overflow_error(plugin_hint) is True


def test_is_context_overflow_error_ignores_unrelated_400s():
    not_overflow = OpenRouterError(
        status_code=400,
        body='{"error":{"message":"Invalid model slug"}}',
        model="openai/gpt-5-nano",
    )
    assert _is_context_overflow_error(not_overflow) is False

    # 5xx is upstream availability, not a payload-shape issue.
    server_error = OpenRouterError(
        status_code=502,
        body='{"error":{"message":"maximum context length"}}',
        model="openai/gpt-5-nano",
    )
    assert _is_context_overflow_error(server_error) is False
