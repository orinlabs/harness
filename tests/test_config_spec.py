"""Tests for the runtime-config input spec, validate-config, and memory plumbing.

The spec (``harness.spec.AgentConfigSpec``) is the published compatibility
contract: external callers decide "is this config compatible with harness @
SHA" by validating against the committed ``harness-spec.json`` at that SHA
(fast path) or running ``harness validate-config`` from that checkout
(authoritative path). These tests pin both paths and the strictness rule
that makes them sound: unknown keys are errors, never silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.cli import main
from harness.config import MemoryConfig
from harness.config_loader import build_agent_config
from harness.spec import spec_json_schema, validate_config_data, validate_config_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides) -> dict:
    data = {
        "id": "agent-1",
        "model": "anthropic/claude-sonnet-4-5",
        "system_prompt": "You are an agent.",
        "reasoning_effort": "high",
        "max_tokens": 8192,
        "timezone": "America/New_York",
        "memory": {"system": "tiered_sqlite", "summarizer_model": None},
        "tools": [
            {
                "name": "send_email",
                "description": "Send an email.",
                "parameters": {"type": "object", "properties": {}},
                "url": "http://127.0.0.1:9999/tools/send_email",
                "timeout_seconds": 30,
                "auth": {"kind": "bearer_env", "token_env": "BEDROCK_TOKEN"},
                "forward_trace_context": True,
            }
        ],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Spec validation (the compatibility contract)
# ---------------------------------------------------------------------------


def test_full_config_validates_and_loads():
    assert validate_config_data(_config()) == []
    cfg = build_agent_config(_config())
    assert cfg.id == "agent-1"
    assert cfg.tools[0].auth.kind == "bearer_env"


def test_minimal_config_validates():
    minimal = {"id": "a", "model": "m", "system_prompt": "s"}
    assert validate_config_data(minimal) == []


def test_unknown_top_level_key_is_an_error():
    """The strictness rule: a config written for a NEWER harness (carrying a
    field this checkout doesn't know) must fail loudly, not silently drop
    the field. This is what makes 'validates == compatible' sound."""
    errors = validate_config_data(_config(wake_schedule={"cron": "* * * * *"}))
    assert any("wake_schedule" in e for e in errors)


def test_unknown_memory_key_is_an_error():
    errors = validate_config_data(_config(memory={"system": "tiered_sqlite", "nope": 1}))
    assert any("nope" in e for e in errors)


def test_unknown_memory_system_is_an_error():
    errors = validate_config_data(_config(memory={"system": "quantum"}))
    assert errors


def test_missing_required_key_is_an_error():
    errors = validate_config_data({"model": "m", "system_prompt": "s"})
    assert any("id" in e for e in errors)


def test_time_zone_alias_accepted():
    data = {"id": "a", "model": "m", "system_prompt": "s", "time_zone": "UTC"}
    assert validate_config_data(data) == []
    assert build_agent_config(data).timezone == "UTC"


def test_validate_config_file_yaml_and_json(tmp_path: Path):
    yaml_path = tmp_path / "agent.yaml"
    yaml_path.write_text("id: a\nmodel: m\nsystem_prompt: s\n")
    assert validate_config_file(yaml_path) == []

    json_path = tmp_path / "agent.json"
    json_path.write_text(json.dumps(_config()))
    assert validate_config_file(json_path) == []

    bad = tmp_path / "bad.yaml"
    bad.write_text("id: a\nmodel: m\nsystem_prompt: s\nnot_a_field: 1\n")
    assert validate_config_file(bad)


def test_committed_sample_config_validates():
    assert validate_config_file(REPO_ROOT / "agents" / "demo.yaml") == []


# ---------------------------------------------------------------------------
# Committed harness-spec.json stays in sync with the models
# ---------------------------------------------------------------------------


def test_committed_spec_json_matches_models():
    committed = json.loads((REPO_ROOT / "harness-spec.json").read_text())
    assert committed == spec_json_schema(), (
        "harness-spec.json is stale; regenerate with "
        "`uv run harness generate-spec > harness-spec.json`"
    )


# ---------------------------------------------------------------------------
# Memory config plumbing (config -> MemoryConfig -> build_memory)
# ---------------------------------------------------------------------------


def test_memory_defaults_and_override():
    cfg = build_agent_config({"id": "a", "model": "m", "system_prompt": "s"})
    assert cfg.memory == MemoryConfig()
    assert cfg.memory.summarizer_model is None  # "harness decides"
    cfg2 = build_agent_config(
        {
            "id": "a",
            "model": "m",
            "system_prompt": "s",
            "memory": {"summarizer_model": "openai/gpt-6-nano"},
        }
    )
    assert cfg2.memory.summarizer_model == "openai/gpt-6-nano"
    assert cfg2.memory.system == "tiered_sqlite"


def test_build_memory_resolves_null_summarizer_to_harness_default():
    """Unset summarizer_model means "harness decides": the factory resolves
    it to DEFAULT_SUMMARIZER_MODEL at construction time, so backends always
    receive a concrete model string."""
    from harness.memory import build_memory
    from harness.spec import DEFAULT_SUMMARIZER_MODEL

    cfg = build_agent_config(
        {"id": "a", "model": "m", "system_prompt": "s", "memory": {"summarizer_model": None}}
    )
    assert cfg.memory.summarizer_model is None
    backend = build_memory(cfg, timezone_name="UTC")
    assert backend.model == DEFAULT_SUMMARIZER_MODEL


def test_build_memory_honors_pinned_summarizer_model():
    from harness.memory import build_memory

    cfg = build_agent_config(
        {
            "id": "a",
            "model": "m",
            "system_prompt": "s",
            "memory": {"summarizer_model": "openai/gpt-6-nano"},
        }
    )
    backend = build_memory(cfg, timezone_name="UTC")
    assert backend.model == "openai/gpt-6-nano"


def test_memory_unknown_key_rejected_by_loader():
    with pytest.raises(ValueError, match="unknown key"):
        build_agent_config(
            {"id": "a", "model": "m", "system_prompt": "s", "memory": {"nope": 1}}
        )


def test_build_memory_unknown_system_fails():
    from harness.memory import build_memory

    cfg = build_agent_config(
        {"id": "a", "model": "m", "system_prompt": "s", "memory": {"system": "quantum"}}
    )
    with pytest.raises(ValueError, match="unknown memory system"):
        build_memory(cfg, timezone_name="UTC")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_validate_config_ok(tmp_path: Path, capsys):
    path = tmp_path / "agent.yaml"
    path.write_text("id: a\nmodel: m\nsystem_prompt: s\n")
    rc = main(["validate-config", str(path)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_cli_validate_config_fails_on_unknown_key(tmp_path: Path, capsys):
    path = tmp_path / "agent.yaml"
    path.write_text("id: a\nmodel: m\nsystem_prompt: s\nfrom_the_future: 1\n")
    rc = main(["validate-config", str(path)])
    assert rc == 1
    assert "from_the_future" in capsys.readouterr().err


def test_cli_generate_spec_check_passes(capsys):
    rc = main(["generate-spec", "--check"])
    assert rc == 0
    assert "up to date" in capsys.readouterr().out


def test_cli_generate_spec_prints_schema(capsys):
    rc = main(["generate-spec"])
    assert rc == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema == spec_json_schema()
