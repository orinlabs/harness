"""Tool-emitted images must reach the next LLM turn.

Background: ``ExternalTool`` parses the wire response into
``ToolResult(text, images)`` (``tools/external.py:_parse_success``), but
before this passthrough landed the main loop only logged ``result.text``
into memory and silently discarded ``result.images``. Every subsequent
LLM turn saw ``image_tokens: 0`` and the model would loop trying to
"open" an image it had supposedly already received — exact symptom of
the May 1 webchat / Telegram Admin traces shipped from the bedrock-api
side. These tests lock in the fix so future refactors can't regress
the round-trip.
"""
from __future__ import annotations

import base64
import importlib
from pathlib import Path

import pytest

from harness.harness import (
    _build_image_followup_message,
    _sniff_image_mime,
)

# Tiny valid 1x1 PNG; this is the byte signature the ``image_to_image_block``
# helper in bedrock-api emits today for most operator-image pipelines.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x04\x00\x00\x00\xb5\x1c\x0c\x02\x00\x00\x00\x0bIDATx\xdac\xfc"
    b"\xff\x1f\x00\x03\x03\x02\x00\xef\xbf\xa7\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
)
_TINY_PNG_B64 = base64.b64encode(_TINY_PNG).decode()


# ---------------------------------------------------------------------------
# Unit tests on the small helpers
# ---------------------------------------------------------------------------


class TestSniffImageMime:
    def test_png_magic(self):
        assert _sniff_image_mime(_TINY_PNG) == "image/png"

    def test_jpeg_magic(self):
        # FFD8FF is the JPEG SOI marker that every JPEG opens with.
        assert _sniff_image_mime(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "image/jpeg"

    def test_gif_magic(self):
        assert _sniff_image_mime(b"GIF89a..............") == "image/gif"
        assert _sniff_image_mime(b"GIF87a..............") == "image/gif"

    def test_webp_magic(self):
        # WebP container: ``RIFF<size:4>WEBP``.
        assert (
            _sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ......")
            == "image/webp"
        )

    def test_unknown_bytes_default_to_png(self):
        # Bedrock-api's invoke endpoint already canonicalizes images via
        # ``image_to_image_block`` so PNG is the right safe default for
        # bytes that don't match a known signature.
        assert _sniff_image_mime(b"this-is-not-an-image") == "image/png"


class TestBuildImageFollowupMessage:
    def test_returns_none_when_no_images(self):
        assert _build_image_followup_message(tool_name="x", images=[]) is None
        assert _build_image_followup_message(tool_name="x", images=None) is None  # type: ignore[arg-type]

    def test_skips_garbage_entries_individually(self):
        # The whole batch must not be dropped because one entry is bad —
        # most image-producing tools today emit exactly one image, but
        # multi-image tools shouldn't lose a valid attachment to a
        # neighbour's malformed payload.
        msg = _build_image_followup_message(
            tool_name="open_telegram_admin_attachment",
            images=[_TINY_PNG_B64, "", 123, _TINY_PNG_B64],  # type: ignore[list-item]
        )
        assert msg is not None
        image_parts = [
            p for p in msg["content"] if p.get("type") == "image_url"
        ]
        assert len(image_parts) == 2

    def test_returns_none_when_every_entry_is_unusable(self):
        # If every image is bad we should NOT log an empty user message
        # with just the label — that would inject a meaningless turn into
        # context. ``None`` means "nothing to forward".
        msg = _build_image_followup_message(
            tool_name="x",
            images=["", 0, None],  # type: ignore[list-item]
        )
        assert msg is None

    def test_emits_openai_multipart_user_shape(self):
        msg = _build_image_followup_message(
            tool_name="open_telegram_admin_attachment",
            images=[_TINY_PNG_B64],
        )
        assert msg is not None
        assert msg["role"] == "user"
        # Multipart content — first a text label naming the source tool,
        # then one image_url part per image.
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {
            "type": "text",
            "text": (
                "[Image attachment from tool 'open_telegram_admin_attachment'. "
                "View it as if you had received it directly from the user.]"
            ),
        }
        assert msg["content"][1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
        }


# ---------------------------------------------------------------------------
# Integration test: harness loop logs the followup message into memory and
# the very next LLM call sees the image
# ---------------------------------------------------------------------------


@pytest.fixture
def harness_storage(tmp_path, monkeypatch):
    """Reset storage to a temp dir so the test owns its sqlite file."""
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    mig_dir = Path(__file__).parent.parent / "src/harness/memory/migrations"
    monkeypatch.setenv("HARNESS_MIGRATIONS_DIR", str(mig_dir))
    from harness.core import storage

    importlib.reload(storage)
    yield


class _RecordingLLM:
    """Drop-in replacement for ``harness.core.llm.complete`` that returns a
    pre-baked sequence of responses and records every call's messages."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        # Snapshot the messages list (deep enough — we only assert dict
        # shape, the harness doesn't mutate prior entries) so the test
        # can read what the LLM saw on each call.
        self.calls.append(
            {
                "model": kwargs.get("model"),
                "messages": [dict(m) for m in kwargs.get("messages", [])],
            }
        )
        if not self._responses:
            raise RuntimeError("RecordingLLM ran out of programmed responses")
        return self._responses.pop(0)


def _make_assistant_response(*, tool_calls: list[dict], text: str = ""):
    """Build a minimal ``LLMResponse`` matching what
    ``harness.core.llm.complete`` returns to the harness."""
    from harness.core.llm import LLMResponse, ToolCall, Usage

    return LLMResponse(
        text=text,
        tool_calls=[
            ToolCall(id=tc["id"], name=tc["name"], args=tc.get("args") or {})
            for tc in tool_calls
        ],
        finish_reason="tool_calls",
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            total_cost=0.0,
            model="test/recording",
            llm_calls=1,
        ),
        reasoning=None,
        raw={
            "id": "rec-1",
            "model": "test/recording",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": "{}",
                                },
                            }
                            for tc in tool_calls
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


class _FakeImageTool:
    """Built-in tool that returns ``ToolResult(text, images)`` — same shape
    ``ExternalTool`` produces from a real ``open_telegram_admin_attachment``
    /invoke/ response."""

    name = "open_telegram_admin_attachment"
    description = "fake attachment opener that always returns an image"
    parameters: dict = {"type": "object", "properties": {}}

    @property
    def schema(self):
        from harness.tools.base import ToolSchema

        return ToolSchema(self.name, self.description, self.parameters)

    def call(self, args, ctx):
        from harness.tools.base import ToolResult

        return ToolResult(
            text="Image attachment description: a pretend description.",
            images=[_TINY_PNG_B64],
        )


def test_tool_emitted_images_reach_next_llm_turn(harness_storage, monkeypatch):
    """End-to-end: a tool returns ``ToolResult.images``, the harness logs
    the followup user message, and the very next LLM call's messages
    include the image bytes as an OpenAI multipart ``image_url`` part.

    Locks in the regression: before this fix the harness silently dropped
    ``result.images``, so the second call's messages held only the
    ``role: tool`` entry with the description text — no image, no
    ``image_tokens`` on the next turn, and the model would loop forever
    trying to "open" the attachment it had already opened.
    """
    from harness import AgentConfig
    from harness.core import llm as llm_mod
    from harness.harness import Harness

    # Programmed LLM: turn 0 calls the image tool, turn 1 calls sleep.
    recording = _RecordingLLM(
        [
            _make_assistant_response(
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "open_telegram_admin_attachment",
                        "args": {},
                    }
                ]
            ),
            _make_assistant_response(
                tool_calls=[{"id": "call-2", "name": "sleep", "args": {}}]
            ),
        ]
    )
    monkeypatch.setattr(llm_mod, "complete", recording)

    config = AgentConfig(
        id="agent-image-passthrough",
        model="test/recording",
        system_prompt="test agent",
    )

    h = Harness(config, run_id="run-image-1")
    # Inject our fake image-emitting tool. The real SleepTool stays so
    # turn 1 can cleanly exit the loop without invoking ``runtime_api``.
    h.tool_map["open_telegram_admin_attachment"] = _FakeImageTool()

    h.run()

    assert len(recording.calls) >= 2, (
        f"expected at least 2 LLM calls, got {len(recording.calls)}: "
        f"{recording.calls}"
    )

    # The second LLM call's messages must include exactly one user-role
    # multipart message whose parts include an ``image_url`` carrying
    # the tool's image bytes as a base64 ``data:`` URL. That's the whole
    # contract this fix exists to satisfy.
    second_call_messages = recording.calls[1]["messages"]
    multipart_users = [
        m
        for m in second_call_messages
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    assert multipart_users, (
        "expected at least one multipart user message on the second LLM call "
        f"but got messages={second_call_messages}"
    )

    image_urls: list[str] = []
    for m in multipart_users:
        for part in m["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                image_urls.append(url)

    assert any(_TINY_PNG_B64 in url for url in image_urls), (
        "expected the tool's base64 image to appear in a multipart "
        f"image_url part on the next LLM turn; saw {image_urls}"
    )
    # Tag should be PNG (sniffed from magic bytes), not the safe-default
    # only when bytes are unrecognizable.
    assert any(url.startswith("data:image/png;base64,") for url in image_urls)

    # The tool message itself must still be present (tool_call_id pairing
    # is what every provider validates against), and must NOT itself
    # carry multipart content — providers are picky about ``role: tool``
    # content shape.
    tool_messages = [
        m for m in second_call_messages if m.get("role") == "tool"
    ]
    assert tool_messages, "tool_call_id pairing message missing from replay"
    for m in tool_messages:
        assert isinstance(m.get("content"), str), (
            "tool messages must keep string content for cross-provider "
            f"compatibility; got {type(m.get('content')).__name__}"
        )
