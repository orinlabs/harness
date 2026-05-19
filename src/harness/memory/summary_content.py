"""Helpers for building summarizer LLM inputs from stored chat messages."""

from __future__ import annotations

import json
from typing import Any

from harness.core.llm import _parse_reasoning

# Stay well under OpenRouter's 400k-token context ceiling once the
# summarizer prompt wrapper is included. ~4 chars/token is conservative
# for JSON-ish text; 300k chars ~= 75k tokens of payload.
DEFAULT_MAX_SUMMARY_INPUT_CHARS = 300_000

# Per-message cap so one tool result cannot dominate an entire bucket.
DEFAULT_MAX_MESSAGE_CHARS = 8_000


def flatten_message_content(content: Any) -> str:
    """Render OpenAI-style message content as plain text for summarization."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text") or ""))
        elif block_type == "image_url":
            parts.append("[image attachment]")
        elif block_type == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                parts.append(thinking.strip())
        elif block_type == "redacted_thinking":
            parts.append("[redacted thinking]")
        elif block_type == "reasoning":
            text = block.get("reasoning") or block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        else:
            parts.append(f"[{block_type or 'block'}]")
    return " ".join(part for part in parts if part)


def message_body_for_summary(msg: dict[str, Any]) -> str:
    """Plain-text body for one chat message, including model reasoning when present."""
    parts: list[str] = []
    reasoning = _parse_reasoning(msg)
    if reasoning:
        parts.append(f"[reasoning] {reasoning}")
    content_text = flatten_message_content(msg.get("content"))
    if content_text:
        parts.append(content_text)
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def sanitize_messages_for_summary(
    messages: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_SUMMARY_INPUT_CHARS,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> str:
    """Convert stored chat messages into compact text safe for summarizer calls.

    Base64 image payloads are replaced with a short placeholder because they
    dominate token count but carry no summarizable semantics. Individual
    messages and the final payload are capped so one bucket cannot exceed
    model context limits.
    """
    rendered: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown")
        body = _truncate(message_body_for_summary(msg), max_message_chars)
        entry: dict[str, str] = {"role": role, "content": body}
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            entry["tool_call_id"] = str(tool_call_id)
        rendered.append(entry)

    content = json.dumps(rendered, ensure_ascii=False)
    return _truncate(content, max_chars)


def sanitize_summary_input(
    content: str,
    *,
    max_chars: int = DEFAULT_MAX_SUMMARY_INPUT_CHARS,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> str:
    """Sanitize any summarizer payload before it is embedded in the LLM prompt.

    Raw message buckets (five-minute tier) arrive as JSON arrays and are
    stripped of image bytes. Higher-tier rollups arrive as plain text and
    are truncated to stay within context limits.
    """
    stripped = content.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            pass
        else:
            if (
                isinstance(parsed, list)
                and parsed
                and all(isinstance(item, dict) for item in parsed)
            ):
                return sanitize_messages_for_summary(
                    parsed,
                    max_chars=max_chars,
                    max_message_chars=max_message_chars,
                )
    return _truncate(content, max_chars)
