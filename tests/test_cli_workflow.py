"""Tests for `harness workflow`'s `--data-root` argument: precedence
between an explicit flag, the CURRENT_DATA_ROOT env var current sets at
dispatch, and the `/data` fallback for standalone runs.

These only exercise argument parsing (via a stubbed `_cmd_workflow`), not
a real run -- that's covered end-to-end by `test_workflow_runner.py`
against `tests/fake_current.py`.
"""

from __future__ import annotations


def _captured_data_root(monkeypatch, argv: list[str]) -> str:
    import harness.cli as cli

    captured: dict[str, str] = {}

    def fake_cmd_workflow(args, parser):  # noqa: ARG001 - matches real signature
        captured["data_root"] = args.data_root
        return 0

    monkeypatch.setattr(cli, "_cmd_workflow", fake_cmd_workflow)
    assert cli.main(argv) == 0
    return captured["data_root"]


def test_data_root_defaults_to_current_data_root_env(monkeypatch):
    monkeypatch.setenv("CURRENT_DATA_ROOT", "/mnt/workspace-data")
    assert _captured_data_root(monkeypatch, ["workflow", "run-id-123"]) == "/mnt/workspace-data"


def test_data_root_falls_back_to_slash_data_when_env_unset(monkeypatch):
    monkeypatch.delenv("CURRENT_DATA_ROOT", raising=False)
    assert _captured_data_root(monkeypatch, ["workflow", "run-id-123"]) == "/data"


def test_data_root_explicit_flag_wins_over_env(monkeypatch):
    monkeypatch.setenv("CURRENT_DATA_ROOT", "/mnt/workspace-data")
    argv = ["workflow", "run-id-123", "--data-root", "/mnt/custom"]
    assert _captured_data_root(monkeypatch, argv) == "/mnt/custom"
