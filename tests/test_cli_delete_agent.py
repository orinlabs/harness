from __future__ import annotations

import importlib


def test_delete_agent_cli_deletes_local_storage(tmp_path, monkeypatch, capsys):
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(storage_root))
    monkeypatch.delenv("HARNESS_TURSO_ORG", raising=False)
    monkeypatch.delenv("HARNESS_TURSO_PLATFORM_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    from harness.core import storage

    importlib.reload(storage)
    conn = storage.load("agent-cli-delete")
    conn.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        ("m1", 1, "user", "{}"),
    )
    storage.flush()

    db_path = storage._db_path("agent-cli-delete")
    assert db_path.exists()

    from harness.cli import main

    assert main(["delete-agent", "agent-cli-delete", "--log-level", "ERROR"]) == 0

    captured = capsys.readouterr()
    assert "id=agent-cli-delete" in captured.err
    assert "local=True" in captured.err
    assert "remote=False" in captured.err
    assert not db_path.exists()
