"""Tests for ``AgentConfig`` config plumbing.

Covers:
- ``max_tokens`` for Anthropic reasoning budget control
- Backward compatibility: legacy feature-flag keys (``summarizer_v2``,
  ``feature_flags``) are silently ignored by the loader now that the
  feature-flag mechanism is removed end to end (Bedrock no longer sends
  it; nothing in the harness ever consumed a flag at runtime).
"""

from __future__ import annotations

import pytest

from harness.config import AgentConfig
from harness.config_loader import build_agent_config


def _minimal_data(**overrides):
    """Build the minimum ``build_agent_config`` payload, with overrides."""
    base: dict = {
        "id": "agent-1",
        "model": "openai/gpt-4o-mini",
        "system_prompt": "hi",
    }
    base.update(overrides)
    return base


def test_build_agent_config_forwards_max_tokens():
    cfg = build_agent_config(_minimal_data(max_tokens=8192))

    assert cfg.max_tokens == 8192


def test_build_agent_config_rejects_invalid_max_tokens():
    with pytest.raises(ValueError, match="expected positive integer"):
        build_agent_config(_minimal_data(max_tokens=0))


# ---------------------------------------------------------------------------
# Backward compatibility: legacy feature-flag keys are silently ignored
# ---------------------------------------------------------------------------


def test_legacy_summarizer_v2_yaml_key_is_silently_ignored():
    """Older agent YAMLs still ship ``summarizer_v2: ...`` at the top level.
    The flag was removed when v1 became the only path, so the loader must
    not raise -- it should drop the key and produce a usable config."""
    cfg = build_agent_config(_minimal_data(summarizer_v2=True))
    assert cfg.id == "agent-1"
    assert not hasattr(cfg, "summarizer_v2")


def test_legacy_feature_flags_block_is_silently_ignored():
    """Older agent YAMLs / Bedrock payloads may still carry a whole
    ``feature_flags`` block. The mechanism is gone from ``AgentConfig``;
    the loader must drop the block and keep loading."""
    cfg = build_agent_config(
        _minimal_data(feature_flags={"summarizer_v2": "on", "anything": "off"})
    )
    assert cfg.id == "agent-1"
    assert not hasattr(cfg, "feature_flags")
    assert not hasattr(cfg, "is_enabled")


def test_agent_config_has_no_feature_flag_surface():
    cfg = AgentConfig(id="a", model="m", system_prompt="s")
    assert not hasattr(cfg, "feature_flags")
    assert cfg.max_tokens is None
