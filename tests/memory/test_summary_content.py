"""Tests for summarizer input sanitization."""

from __future__ import annotations

from harness.memory.summary_content import (
    DEFAULT_MAX_SUMMARY_INPUT_CHARS,
    flatten_message_content,
    sanitize_messages_for_summary,
)


def test_flatten_message_content_replaces_image_blocks():
    content = [
        {"type": "text", "text": "Photo from site:"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + ("A" * 100_000)},
        },
    ]
    assert flatten_message_content(content) == "Photo from site: [image attachment]"


def test_sanitize_messages_for_summary_strips_large_image_payloads():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Images returned by open_sms_attachment:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + ("Z" * 500_000)},
                },
            ],
        },
        {"role": "assistant", "content": "Saved attachments to disk."},
    ]

    rendered = sanitize_messages_for_summary(messages)

    assert "ZZZZ" not in rendered
    assert "[image attachment]" in rendered
    assert "Saved attachments to disk." in rendered
    assert len(rendered) < 10_000


def test_sanitize_messages_for_summary_truncates_oversized_payload():
    chunk = "x" * 8_000
    messages = [{"role": "tool", "content": chunk} for _ in range(50)]

    rendered = sanitize_messages_for_summary(messages)

    assert len(rendered) <= DEFAULT_MAX_SUMMARY_INPUT_CHARS + 200
    assert "...[truncated" in rendered
