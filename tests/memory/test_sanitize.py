"""Tests for ``harness.memory.sanitize``.

Pure unit tests that exercise ``sanitize_messages_for_summarization`` against
the message shapes the harness actually persists today: multipart user
messages from ``_build_image_followup_message``, assistant rows with
OpenRouter-style ``reasoning_details``, and provider-native ``thinking``
content blocks. No SQLite, no LLM.
"""

from __future__ import annotations

import copy

from harness.memory.sanitize import (
    SanitizationResult,
    sanitize_messages_for_summarization,
)


def _base64_image_message(label: str, *, big_payload_chars: int = 4096) -> dict:
    """Mirror of ``harness.harness._build_image_followup_message`` output."""
    payload = "A" * big_payload_chars
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": label},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            },
        ],
    }


def test_strips_base64_image_url_keeps_label_text():
    label = "[Image attachment from tool 'browser_screenshot'.]"
    msgs = [_base64_image_message(label, big_payload_chars=10_000)]

    result = sanitize_messages_for_summarization(msgs)

    [cleaned] = result.messages
    assert cleaned["role"] == "user"
    assert cleaned["content"] == [{"type": "text", "text": label}]
    assert result.stripped_image_parts == 1
    assert result.stripped_signatures == 0
    assert result.sanitized


def test_keeps_plain_url_image_parts():
    """Non-base64 image URLs are tiny; keep them so the summarizer sees
    that an image was attached without flagging false positives."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/screenshot.png"},
                },
            ],
        }
    ]

    result = sanitize_messages_for_summarization(msgs)

    [cleaned] = result.messages
    assert cleaned["content"][1]["image_url"]["url"] == "https://example.com/screenshot.png"
    assert result.stripped_image_parts == 0


def test_strips_signatures_in_reasoning_details_keeps_text():
    msgs = [
        {
            "role": "assistant",
            "content": "Looking at the screenshot now.",
            "reasoning": "The user wants me to inspect the page.",
            "reasoning_details": [
                {
                    "type": "reasoning.text",
                    "text": "I should examine the layout first.",
                    "signature": "ANTHROPIC_OPAQUE_SIGNATURE_BLOB" * 100,
                },
                {
                    "type": "reasoning.encrypted",
                    "data": "ENCRYPTED_OAI_BODY" * 100,
                    "signature": "more-signature",
                },
            ],
        }
    ]

    result = sanitize_messages_for_summarization(msgs)

    [cleaned] = result.messages
    assert cleaned["reasoning"] == "The user wants me to inspect the page."
    text_block, encrypted_block = cleaned["reasoning_details"]
    assert text_block["text"] == "I should examine the layout first."
    assert "signature" not in text_block
    assert "signature" not in encrypted_block
    assert "data" not in encrypted_block
    # One signature on each block plus the encrypted data blob = 3 strips.
    assert result.stripped_signatures == 3
    assert result.stripped_image_parts == 0


def test_strips_signature_on_inline_thinking_blocks():
    """Provider-native content blocks (Anthropic-shaped ``thinking``) carry
    their signature inline. Sanitize should drop the signature without
    removing the block or its plaintext."""
    msgs = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "let me think...",
                    "signature": "OPAQUE-SIG",
                },
                {"type": "text", "text": "Here's the answer."},
            ],
        }
    ]

    result = sanitize_messages_for_summarization(msgs)

    [cleaned] = result.messages
    thinking, text = cleaned["content"]
    assert thinking == {"type": "thinking", "thinking": "let me think..."}
    assert text == {"type": "text", "text": "Here's the answer."}
    assert result.stripped_signatures == 1


def test_preserves_assistant_tool_calls():
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "open_telegram_admin_attachment",
                        "arguments": '{"message_id": 123}',
                    },
                }
            ],
        }
    ]

    result = sanitize_messages_for_summarization(msgs)

    assert result.messages[0] == msgs[0]
    assert not result.sanitized


def test_preserves_role_tool_content():
    """Tool output text is preserved intact — no truncation in v1."""
    msgs = [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"status": "ok", "data": "a" * 10000}',
        }
    ]

    result = sanitize_messages_for_summarization(msgs)

    assert result.messages == msgs
    assert not result.sanitized


def test_does_not_mutate_input():
    msgs = [_base64_image_message("x")]
    snapshot = copy.deepcopy(msgs)
    sanitize_messages_for_summarization(msgs)
    assert msgs == snapshot, "input list was mutated"


def test_handles_non_dict_messages_gracefully():
    msgs = ["not-a-dict", _base64_image_message("y"), 42]  # type: ignore[list-item]
    result = sanitize_messages_for_summarization(msgs)
    assert result.messages[0] == "not-a-dict"
    assert result.messages[2] == 42
    # The middle message still got sanitized.
    assert result.stripped_image_parts == 1


def test_data_url_without_base64_marker_is_preserved():
    """A ``data:`` URL that doesn't declare ``;base64`` carries percent-
    encoded text, not bytes. It's typically tiny; leave it alone."""
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/svg+xml,<svg/>"},
                }
            ],
        }
    ]

    result = sanitize_messages_for_summarization(msgs)

    assert result.stripped_image_parts == 0
    assert result.messages[0]["content"][0]["image_url"]["url"].startswith(
        "data:image/svg+xml,"
    )


def test_sanitization_result_dataclass_flags_combined_strip():
    msgs = [
        _base64_image_message("img"),
        {
            "role": "assistant",
            "reasoning_details": [{"type": "reasoning.text", "signature": "x"}],
        },
    ]

    result = sanitize_messages_for_summarization(msgs)

    assert isinstance(result, SanitizationResult)
    assert result.stripped_image_parts == 1
    assert result.stripped_signatures == 1
    assert result.sanitized
