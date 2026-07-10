"""Tests for repo-managed agent bundles: spec, loader, hashing, CLI, extends."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from harness.bundles import (
    BundleError,
    expand_extends,
    list_bundle_names,
    load_bundle,
    render_bundle_payload,
    render_sync_manifest,
    scan_for_secrets,
)
from harness.cli import main
from harness.config import MemoryConfig
from harness.config_loader import build_agent_config, load_agent_config_from_path
from harness.spec import SPEC_VERSION, spec_json_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_bundle(
    agents_dir: Path,
    name: str = "po-3v",
    *,
    manifest_extra: str = "",
) -> Path:
    """A minimal but representative bundle: prompt file, shared fragment,
    one document, one sandbox file."""
    agents_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = agents_dir / name
    (bundle_dir / "sops").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "sandbox").mkdir(parents=True, exist_ok=True)
    (agents_dir / "shared").mkdir(exist_ok=True)

    (bundle_dir / "system.md").write_text("You are the PO agent.")
    (agents_dir / "shared" / "etiquette.md").write_text("Be concise.")
    (bundle_dir / "sops" / "build-po.md").write_text("# SOP: build a PO\nSteps...")
    (bundle_dir / "sandbox" / "po_flow_write.py").write_text("print('hi')\n")

    manifest = agents_dir / f"{name}.yaml"
    manifest.write_text(
        f"""\
spec_version: {SPEC_VERSION}
name: {name}
model: anthropic/claude-sonnet-4-5
reasoning_effort: high
timezone: America/New_York
max_sleep_hours: 24
system_prompt_file: {name}/system.md
prompt_fragments:
  - shared/etiquette.md
files:
  - path: {name}/sops/build-po.md
    target: document
    title: "SOP: build a PO"
  - path: {name}/sandbox/po_flow_write.py
    target: sandbox
    dest: tools/po_flow_write.py
adapters: [Documents, Computer]
expected_tools: [computer_exec, get_document]
feature_flags:
  auto_associative_memory: "on"
{manifest_extra}"""
    )
    return manifest


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    _write_bundle(d)
    return d


# ---------------------------------------------------------------------------
# Loading + rendering
# ---------------------------------------------------------------------------


def test_load_bundle_renders_prompt_in_order(agents_dir: Path):
    bundle = load_bundle("po-3v", agents_dir)
    assert bundle.system_prompt == "You are the PO agent.\n\nBe concise."
    assert bundle.bundle_hash.startswith("sha256:")
    assert bundle.manifest.memory == type(bundle.manifest.memory)()  # defaults


def test_bundle_name_must_match_file_stem(tmp_path: Path):
    d = tmp_path / "agents"
    _write_bundle(d, "po-3v")
    (d / "renamed.yaml").write_text((d / "po-3v.yaml").read_text())
    with pytest.raises(BundleError, match="must match the file stem"):
        load_bundle("renamed", d)


def test_missing_referenced_file_fails(agents_dir: Path):
    (agents_dir / "shared" / "etiquette.md").unlink()
    with pytest.raises(BundleError, match="etiquette.md"):
        load_bundle("po-3v", agents_dir)


def test_prompt_literal_and_file_mutually_exclusive(tmp_path: Path):
    d = tmp_path / "agents"
    _write_bundle(d)
    manifest = d / "po-3v.yaml"
    manifest.write_text(manifest.read_text() + 'system_prompt: "inline too"\n')
    with pytest.raises(BundleError, match="exactly one of"):
        load_bundle("po-3v", d)


def test_newer_spec_version_rejected(tmp_path: Path):
    d = tmp_path / "agents"
    manifest = _write_bundle(d)
    manifest.write_text(
        manifest.read_text().replace(
            f"spec_version: {SPEC_VERSION}", f"spec_version: {SPEC_VERSION + 1}"
        )
    )
    with pytest.raises(BundleError, match="newer than this harness checkout"):
        load_bundle("po-3v", d)


def test_bundle_hash_changes_with_shared_fragment(agents_dir: Path):
    before = load_bundle("po-3v", agents_dir).bundle_hash
    (agents_dir / "shared" / "etiquette.md").write_text("Be very concise.")
    after = load_bundle("po-3v", agents_dir).bundle_hash
    assert before != after


def test_bundle_hash_stable_across_loads(agents_dir: Path):
    first = load_bundle("po-3v", agents_dir).bundle_hash
    second = load_bundle("po-3v", agents_dir).bundle_hash
    assert first == second


def test_render_bundle_payload_shape(agents_dir: Path):
    payload = render_bundle_payload(load_bundle("po-3v", agents_dir))
    assert payload["name"] == "po-3v"
    assert payload["spec_version"] == SPEC_VERSION
    assert payload["config"]["system_prompt"].startswith("You are the PO agent.")
    assert payload["config"]["max_sleep_hours"] == 24
    assert payload["documents"] == [
        {"title": "SOP: build a PO", "kind": "skill", "content": "# SOP: build a PO\nSteps..."}
    ]
    decoded = base64.b64decode(payload["sandbox_files"]["tools/po_flow_write.py"])
    assert decoded == b"print('hi')\n"


def test_render_sync_manifest_lists_all_bundles(tmp_path: Path):
    d = tmp_path / "agents"
    _write_bundle(d, "po-3v")
    _write_bundle(d, "other-agent")
    manifest = render_sync_manifest(d, harness_git_sha="a" * 40)
    assert manifest["spec_version"] == SPEC_VERSION
    assert manifest["harness_git_sha"] == "a" * 40
    assert sorted(a["name"] for a in manifest["agents"]) == ["other-agent", "po-3v"]


def test_list_bundle_names_skips_legacy_configs(tmp_path: Path):
    d = tmp_path / "agents"
    _write_bundle(d)
    # Legacy standalone config (has `id`, no spec_version) must be ignored.
    (d / "demo.yaml").write_text("id: demo\nmodel: m\nsystem_prompt: hi\n")
    assert list_bundle_names(d) == ["po-3v"]


def test_secret_scan_flags_token(agents_dir: Path):
    (agents_dir / "po-3v" / "system.md").write_text(
        "Use key sk-abcdefghijklmnopqrstuvwxyz123456 for everything."
    )
    findings = scan_for_secrets(load_bundle("po-3v", agents_dir))
    assert findings and "secret key" in findings[0]


def test_path_traversal_rejected(tmp_path: Path):
    d = tmp_path / "agents"
    manifest = _write_bundle(d)
    manifest.write_text(
        manifest.read_text().replace("po-3v/sops/build-po.md", "../outside.md")
    )
    with pytest.raises(BundleError, match="relative path"):
        load_bundle("po-3v", d)


# ---------------------------------------------------------------------------
# extends: bundle + trial-config overlay
# ---------------------------------------------------------------------------


def test_expand_extends_merges_bundle_and_overlay(agents_dir: Path):
    data = {
        "id": "trial-123",
        "extends": "po-3v",
        "system_prompt_prefix": "CONTRACT PREAMBLE",
        "system_prompt_suffix": "Task: build POs for batch 7.",
        "feature_flags": {"auto_associative_memory": "off"},
        "tools": [],
    }
    merged = expand_extends(data, agents_dir=agents_dir)
    cfg = build_agent_config(merged)
    assert cfg.id == "trial-123"
    assert cfg.model == "anthropic/claude-sonnet-4-5"  # from bundle
    assert cfg.system_prompt == (
        "CONTRACT PREAMBLE\n\nYou are the PO agent.\n\nBe concise."
        "\n\nTask: build POs for batch 7."
    )
    # Overlay wins on flag collision.
    assert cfg.feature_flags["auto_associative_memory"] == "off"
    assert cfg.timezone == "America/New_York"


def test_expand_extends_model_override(agents_dir: Path):
    merged = expand_extends(
        {"id": "t", "extends": "po-3v", "model": "openai/gpt-5"}, agents_dir=agents_dir
    )
    assert merged["model"] == "openai/gpt-5"


def test_expand_extends_requires_id(agents_dir: Path):
    with pytest.raises(BundleError, match="must set `id`"):
        expand_extends({"extends": "po-3v"}, agents_dir=agents_dir)


def test_expand_extends_rejects_system_prompt(agents_dir: Path):
    with pytest.raises(BundleError, match="must not also set"):
        expand_extends(
            {"id": "t", "extends": "po-3v", "system_prompt": "x"}, agents_dir=agents_dir
        )


def test_loader_expands_extends_via_env(agents_dir: Path, tmp_path: Path, monkeypatch):
    """The environments-repo path: trial JSON in the workdir, agents dir via env."""
    monkeypatch.setenv("HARNESS_AGENTS_DIR", str(agents_dir))
    trial = tmp_path / "trial.json"
    trial.write_text(
        json.dumps(
            {
                "id": "trial-9",
                "extends": "po-3v",
                "system_prompt_suffix": "Do the task.",
                "tools": [],
            }
        )
    )
    cfg = load_agent_config_from_path(trial)
    assert cfg.id == "trial-9"
    assert cfg.system_prompt.endswith("Do the task.")
    assert cfg.memory.system == "tiered_sqlite"


# ---------------------------------------------------------------------------
# Memory config plumbing
# ---------------------------------------------------------------------------


def test_memory_defaults_and_override():
    cfg = build_agent_config({"id": "a", "model": "m", "system_prompt": "s"})
    assert cfg.memory == MemoryConfig()
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


def test_memory_unknown_key_rejected():
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
# Committed harness-spec.json stays in sync with the models
# ---------------------------------------------------------------------------


def test_committed_spec_json_matches_models():
    committed = json.loads(
        (Path(__file__).resolve().parents[1] / "harness-spec.json").read_text()
    )
    assert committed == spec_json_schema(), (
        "harness-spec.json is stale; regenerate with "
        "`uv run python -m harness.spec > harness-spec.json`"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_validate_agents(agents_dir: Path, capsys):
    rc = main(["validate-agents", "--agents-dir", str(agents_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("po-3v sha256:")


def test_cli_validate_agents_fails_on_bad_bundle(agents_dir: Path, capsys):
    (agents_dir / "po-3v" / "system.md").unlink()
    rc = main(["validate-agents", "--agents-dir", str(agents_dir)])
    assert rc == 1
    assert "system.md" in capsys.readouterr().err


def test_cli_render_agent(agents_dir: Path, capsys):
    rc = main(["render-agent", "po-3v", "--agents-dir", str(agents_dir)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "po-3v"
    assert payload["bundle_hash"].startswith("sha256:")


def test_cli_render_manifest(agents_dir: Path, capsys):
    rc = main(["render-manifest", "--agents-dir", str(agents_dir), "--commit", "b" * 40])
    assert rc == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["harness_git_sha"] == "b" * 40
    assert [a["name"] for a in manifest["agents"]] == ["po-3v"]


def test_cli_generate_spec_check_passes(capsys):
    rc = main(["generate-spec", "--check"])
    assert rc == 0
    assert "up to date" in capsys.readouterr().out


def test_cli_generate_spec_prints_schema(capsys):
    rc = main(["generate-spec"])
    assert rc == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema == spec_json_schema()


def test_example_bundle_validates(capsys):
    examples = Path(__file__).resolve().parents[1] / "examples" / "bundles"
    rc = main(["validate-agents", "--agents-dir", str(examples)])
    assert rc == 0
    assert capsys.readouterr().out.startswith("example-po sha256:")
