from __future__ import annotations

import importlib


def test_reset_memory_cli_resets_local_memory(tmp_path, monkeypatch, capsys):
    storage_root = tmp_path / "storage"
    monkeypatch.chdir(tmp_path)

    from harness.core import storage

    importlib.reload(storage)
    monkeypatch.setattr(storage, "_STORAGE_ROOT", storage_root)
    conn = storage.load("agent-cli-reset")
    conn.execute(
        "INSERT INTO messages (id, ts_ns, role, content_json) VALUES (?, ?, ?, ?)",
        ("m1", 1, "user", "{}"),
    )
    storage.flush()

    from harness.cli import main

    assert main(["reset-memory", "agent-cli-reset", "--log-level", "ERROR"]) == 0

    captured = capsys.readouterr()
    assert "id=agent-cli-reset" in captured.err
    assert "local=True" in captured.err

    conn = storage.load("agent-cli-reset")
    assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 0
