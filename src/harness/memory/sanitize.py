"""Sanitize messages before summarization.

The summarizer LLM operates on raw chat-format messages serialized as JSON.
Two classes of payload bloat the prompt without contributing to a useful
summary:

  - **Base64 image bytes** embedded in ``image_url.url`` data URLs. A single
    screenshot followup can easily exceed a megabyte of base64; a vision-
    heavy 5-minute bucket has been observed to blow past the summarizer
    model's 400K-token context window with one set of buckets producing
    ~601K input tokens.
  - **Signed ``reasoning_details`` blocks** (Anthropic extended thinking,
    OpenAI o-series encrypted reasoning). The ``signature`` and encrypted
    ``data`` fields are opaque tokens used only for tool-continuation
    replay; the human-readable reasoning text in ``text`` / ``summary`` /
    ``thinking`` / ``content`` is what actually helps summarization.

This module returns sanitized copies so the durable ``messages`` log is
unaffected — only the summarizer prompt sees the trimmed form. Tool
content, ``tool_calls``, ``reasoning`` plaintext, and assistant text are
all preserved verbatim.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

# Conservative match: data URLs whose value carries base64-encoded bytes.
# Per RFC 2397: ``data:<media-type>[;base64],<data>``. We match the prefix
# rather than the body to keep the regex small; dropping the part entirely
# is what evicts the megabyte-class payloads.
_DATA_URL_BASE64_PREFIX = re.compile(r"^data:[^;,]*;base64,", re.IGNORECASE)

# Inline content-block types that may carry a ``signature`` field.
# ``thinking`` / ``redacted_thinking`` are Anthropic's content-block names
# when the assistant message uses provider-native blocks instead of the
# OpenAI-style ``reasoning_details`` array. ``reasoning`` covers a smaller
# subset of providers that surface reasoning inline.
_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning"})

# ``reasoning_details`` entries that carry an opaque encrypted body in
# ``data`` instead of plaintext in ``text`` / ``summary`` / ``content``.
# The ciphertext is useful only for replay continuation, never for
# summarization, so drop it alongside the signature.
_ENCRYPTED_REASONING_TYPES = frozenset({"reasoning.encrypted", "reasoning_encrypted"})


@dataclass
class SanitizationResult:
    """Result of sanitizing a batch of messages for summarization."""

    messages: list[dict]
    stripped_image_parts: int = 0
    stripped_signatures: int = 0

    @property
    def sanitized(self) -> bool:
        return self.stripped_image_parts > 0 or self.stripped_signatures > 0


def sanitize_messages_for_summarization(messages: list[dict]) -> SanitizationResult:
    """Return deep-copied messages with summarization-irrelevant bulk removed.

    Strips:
      - Multipart ``content`` parts whose ``image_url.url`` is a base64
        data URL. Other parts on the same message (text labels, plain
        URL images) are preserved.
      - ``signature`` (and encrypted ``data``) fields from
        ``reasoning_details`` entries and inline thinking content blocks.

    Preserves:
      - ``role`` and all non-image content (tool output, assistant text,
        user text, system text).
      - ``tool_calls`` on assistant messages, including ``function.arguments``.
      - ``reasoning`` plaintext and the human-readable text fields of
        ``reasoning_details`` (``text``, ``summary``, ``content``, ``thinking``).

    The original messages list is not mutated.
    """
    out: list[dict] = []
    image_count = 0
    sig_count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        cleaned, img_n, sig_n = _sanitize_one(msg)
        image_count += img_n
        sig_count += sig_n
        out.append(cleaned)
    return SanitizationResult(
        messages=out,
        stripped_image_parts=image_count,
        stripped_signatures=sig_count,
    )


def _sanitize_one(msg: dict) -> tuple[dict, int, int]:
    cleaned = copy.deepcopy(msg)
    image_count = 0
    sig_count = 0

    content = cleaned.get("content")
    if isinstance(content, list):
        new_parts: list = []
        for part in content:
            if _is_base64_image_part(part):
                image_count += 1
                continue
            if isinstance(part, dict) and part.get("type") in _THINKING_BLOCK_TYPES:
                if part.pop("signature", None) is not None:
                    sig_count += 1
            new_parts.append(part)
        cleaned["content"] = new_parts

    details = cleaned.get("reasoning_details")
    if isinstance(details, list):
        for d in details:
            if not isinstance(d, dict):
                continue
            if d.pop("signature", None) is not None:
                sig_count += 1
            if d.get("type") in _ENCRYPTED_REASONING_TYPES and d.pop("data", None) is not None:
                sig_count += 1

    return cleaned, image_count, sig_count


def _is_base64_image_part(part: object) -> bool:
    """True if ``part`` is an OpenAI-style ``image_url`` block with a
    base64 data URL."""
    if not isinstance(part, dict):
        return False
    if part.get("type") != "image_url":
        return False
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
    elif isinstance(image_url, str):
        url = image_url
    else:
        return False
    return isinstance(url, str) and bool(_DATA_URL_BASE64_PREFIX.match(url))
