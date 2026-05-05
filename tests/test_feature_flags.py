"""Tests for ``AgentConfig`` config plumbing.

Covers:
- ``AgentConfig.feature_flags`` defaults and ``is_enabled`` semantics
- ``config_loader`` reading ``feature_flags`` from YAML / dict input
- The Bedrock-JSON path (``_config_from_bedrock_json``) forwarding
  ``feature_flags`` from the platform's ``harness-config`` payload
- ``max_tokens`` for Anthropic reasoning budget control
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


# ---------------------------------------------------------------------------
# AgentConfig dataclass
# ---------------------------------------------------------------------------


def test_feature_flags_defaults_to_empty_dict():
    cfg = AgentConfig(id="a", model="m", system_prompt="s")
    assert cfg.feature_flags == {}
    assert cfg.max_tokens is None


def test_is_enabled_true_only_for_literal_on():
    cfg = AgentConfig(
        id="a",
        model="m",
        system_prompt="s",
        feature_flags={"summarizer_v2": "on"},
    )
    assert cfg.is_enabled("summarizer_v2") is True


def test_is_enabled_case_insensitive():
    cfg = AgentConfig(
        id="a",
        model="m",
        system_prompt="s",
        feature_flags={"flag_a": "ON", "flag_b": " On ", "flag_c": "oN"},
    )
    assert cfg.is_enabled("flag_a") is True
    assert cfg.is_enabled("flag_b") is True
    assert cfg.is_enabled("flag_c") is True


@pytest.mark.parametrize(
    "value",
    ["off", "", "true", "1", "enabled", "yes", "OFFISH"],
)
def test_is_enabled_false_for_non_on_values(value):
    cfg = AgentConfig(
        id="a",
        model="m",
        system_prompt="s",
        feature_flags={"f": value},
    )
    assert cfg.is_enabled("f") is False


def test_is_enabled_false_for_missing_flag():
    cfg = AgentConfig(id="a", model="m", system_prompt="s")
    assert cfg.is_enabled("never_defined") is False


def test_feature_flags_carries_freeform_values():
    """Non-boolean flags work — e.g. a tier name or a model variant."""
    cfg = AgentConfig(
        id="a",
        model="m",
        system_prompt="s",
        feature_flags={"approval_tier": "tier_3", "model_variant": "preview"},
    )
    assert cfg.feature_flags.get("approval_tier") == "tier_3"
    assert cfg.feature_flags.get("model_variant") == "preview"
    # Non-"on" values still don't satisfy is_enabled (it's the boolean shortcut).
    assert cfg.is_enabled("approval_tier") is False


# ---------------------------------------------------------------------------
# config_loader.build_agent_config (YAML + Bedrock-JSON share this)
# ---------------------------------------------------------------------------


def test_build_agent_config_no_feature_flags_block():
    cfg = build_agent_config(_minimal_data())
    assert cfg.feature_flags == {}


def test_build_agent_config_reads_feature_flags_block():
    cfg = build_agent_config(
        _minimal_data(feature_flags={"summarizer_v2": "on", "auto_associative_memory": "off"})
    )
    assert cfg.feature_flags == {"summarizer_v2": "on", "auto_associative_memory": "off"}
    assert cfg.is_enabled("summarizer_v2") is True
    assert cfg.is_enabled("auto_associative_memory") is False


def test_build_agent_config_stringifies_non_string_values():
    cfg = build_agent_config(_minimal_data(feature_flags={"x": 42, "y": True}))
    # Coerced to str so the wire shape is uniform downstream.
    assert cfg.feature_flags == {"x": "42", "y": "True"}


def test_build_agent_config_rejects_non_mapping_feature_flags():
    bad_payload = _minimal_data(feature_flags=["summarizer_v2", "auto_associative_memory"])
    with pytest.raises(ValueError, match="feature_flags must be a mapping"):
        build_agent_config(bad_payload)


def test_build_agent_config_forwards_max_tokens():
    cfg = build_agent_config(_minimal_data(max_tokens=8192))

    assert cfg.max_tokens == 8192


def test_build_agent_config_rejects_invalid_max_tokens():
    with pytest.raises(ValueError, match="expected positive integer"):
        build_agent_config(_minimal_data(max_tokens=0))


# ---------------------------------------------------------------------------
# Bedrock JSON ingest path
# ---------------------------------------------------------------------------


def test_bedrock_json_forwards_feature_flags():
    """``_config_from_bedrock_json`` flattens adapters but must preserve
    top-level keys including ``feature_flags``."""
    from harness.cloud.bedrock.config import _config_from_bedrock_json

    bedrock_payload = {
        "id": "agent-1",
        "model": "anthropic/claude-opus-4.7",
        "system_prompt": "hi",
        "reasoning_effort": "medium",
        "max_tokens": 16384,
        "feature_flags": {"summarizer_v2": "on", "elicitation_v2": "off"},
        "adapters": [
            {
                "name": "Contacts",
                "description": "Contacts adapter",
                "tools": [
                    {
                        "name": "list_contacts",
                        "description": "list",
                        "parameters": {"type": "object", "properties": {}},
                        "url": "http://bedrock/api/.../list_contacts/invoke/",
                    }
                ],
            }
        ],
    }
    cfg = _config_from_bedrock_json(bedrock_payload)
    assert cfg.feature_flags == {"summarizer_v2": "on", "elicitation_v2": "off"}
    assert cfg.max_tokens == 16384
    assert cfg.is_enabled("summarizer_v2") is True
    assert cfg.is_enabled("elicitation_v2") is False
    # Sanity: tools still flattened.
    assert len(cfg.tools) == 1
    assert cfg.tools[0].name == "list_contacts"


# ---------------------------------------------------------------------------
# Backward compatibility: legacy summarizer_v2 bool still works
# ---------------------------------------------------------------------------


def test_legacy_summarizer_v2_bool_still_loads():
    """Existing YAML configs with ``summarizer_v2: true`` keep working."""
    cfg = build_agent_config(_minimal_data(summarizer_v2=True))
    assert cfg.summarizer_v2 is True
    # ``is_enabled`` only consults ``feature_flags`` — the legacy bool is
    # honored separately by ``Harness.__init__``. Verify that here.
    assert cfg.is_enabled("summarizer_v2") is False


def test_feature_flags_and_legacy_bool_coexist():
    cfg = build_agent_config(
        _minimal_data(summarizer_v2=False, feature_flags={"summarizer_v2": "on"})
    )
    assert cfg.summarizer_v2 is False
    assert cfg.is_enabled("summarizer_v2") is True
