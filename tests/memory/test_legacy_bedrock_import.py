from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_import_legacy_bedrock_memory_writes_per_agent_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "HARNESS_MIGRATIONS_DIR",
        str(Path(__file__).parent.parent.parent / "src" / "harness" / "memory" / "migrations"),
    )

    from harness.core import storage as storage_module

    importlib.reload(storage_module)

    from harness.memory.legacy_bedrock_import import import_legacy_bedrock_memory

    payload_path = tmp_path / "agent-1.json"
    payload_path.write_text(
        json.dumps(
            {
                "agent_id": "agent-1",
                "messages": [
                    {
                        "id": "bedrock:1",
                        "ts_ns": 1_700_000_000_000_000_000,
                        "role": "user",
                        "content": {"role": "user", "content": "remember the blue door"},
                    }
                ],
                "one_minute_summaries": [
                    {
                        "id": "bedrock:one",
                        "date": "2026-04-24",
                        "hour": 10,
                        "minute": 30,
                        "summary": "one minute",
                        "message_count": 1,
                        "created_at_ns": 1,
                    }
                ],
                "five_minute_summaries": [],
                "hourly_summaries": [],
                "daily_summaries": [
                    {
                        "id": "bedrock:daily",
                        "date": "2026-04-24",
                        "summary": "daily",
                        "message_count": 1,
                        "created_at_ns": 2,
                    }
                ],
                "weekly_summaries": [],
                "monthly_summaries": [],
            }
        )
    )

    counts = import_legacy_bedrock_memory("agent-1", payload_path)
    assert counts.messages == 1
    assert counts.one_minute_summaries == 1
    assert counts.daily_summaries == 1

    conn = storage_module.load("agent-1")
    try:
        message = conn.execute("SELECT role, content_json FROM messages").fetchone()
        assert message["role"] == "user"
        assert json.loads(message["content_json"]) == {
            "role": "user",
            "content": "remember the blue door",
        }
        assert conn.execute("SELECT COUNT(*) AS c FROM daily_summaries").fetchone()["c"] == 1
    finally:
        storage_module.close()


def test_import_legacy_bedrock_memory_rejects_wrong_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))

    from harness.memory.legacy_bedrock_import import import_legacy_bedrock_memory

    payload_path = tmp_path / "agent-1.json"
    payload_path.write_text(json.dumps({"agent_id": "agent-2"}))

    try:
        import_legacy_bedrock_memory("agent-1", payload_path)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected wrong-agent payload to fail")


def test_import_legacy_bedrock_memory_batches_across_chunk_boundary(
    tmp_path, monkeypatch
):
    """Imports that exceed BULK_INSERT_BATCH_ROWS must land all rows.

    Regression guard: the legacy per-row ``executemany()`` implementation was
    replaced with a chunked multi-row VALUES path. An off-by-one in the chunk
    flush — e.g. forgetting the tail batch, or emitting a malformed VALUES
    clause on odd-sized tails — would silently drop rows. Force enough rows
    to span at least 2 full batches and a partial tail, then assert every
    row round-trips back out of storage.
    """
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "HARNESS_MIGRATIONS_DIR",
        str(Path(__file__).parent.parent.parent / "src" / "harness" / "memory" / "migrations"),
    )

    from harness.core import storage as storage_module

    importlib.reload(storage_module)

    from harness.memory import legacy_bedrock_import
    from harness.memory.legacy_bedrock_import import import_legacy_bedrock_memory

    # Pick a row count that is not a multiple of BULK_INSERT_BATCH_ROWS so the
    # tail-flush path is exercised too. 2 full batches + a partial.
    total_messages = legacy_bedrock_import.BULK_INSERT_BATCH_ROWS * 2 + 7
    total_one_minute = legacy_bedrock_import.BULK_INSERT_BATCH_ROWS + 3

    messages = [
        {
            "id": f"bedrock:msg:{i}",
            "ts_ns": 1_700_000_000_000_000_000 + i,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": {"role": "user", "content": f"msg {i}"},
        }
        for i in range(total_messages)
    ]
    one_minute = [
        {
            "id": f"bedrock:om:{i}",
            "date": "2026-04-24",
            "hour": (i // 60) % 24,
            "minute": i % 60,
            "summary": f"one minute {i}",
            "message_count": i,
            "created_at_ns": i,
        }
        for i in range(total_one_minute)
    ]

    payload_path = tmp_path / "agent-bulk.json"
    payload_path.write_text(
        json.dumps(
            {
                "agent_id": "agent-bulk",
                "messages": messages,
                "one_minute_summaries": one_minute,
                "five_minute_summaries": [],
                "hourly_summaries": [],
                "daily_summaries": [],
                "weekly_summaries": [],
                "monthly_summaries": [],
            }
        )
    )

    counts = import_legacy_bedrock_memory("agent-bulk", payload_path)
    assert counts.messages == total_messages
    assert counts.one_minute_summaries == total_one_minute

    conn = storage_module.load("agent-bulk")
    try:
        assert (
            conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
            == total_messages
        )
        assert (
            conn.execute("SELECT COUNT(*) AS c FROM one_minute_summaries").fetchone()[
                "c"
            ]
            == total_one_minute
        )
        # Spot-check that the first, a middle, and the last row all round-trip
        # intact — proves the batched VALUES stream correctly binds per-row
        # parameters rather than bleeding across rows.
        first = conn.execute(
            "SELECT role, content_json FROM messages WHERE id = ?",
            ["bedrock:msg:0"],
        ).fetchone()
        assert first["role"] == "user"
        assert json.loads(first["content_json"]) == {"role": "user", "content": "msg 0"}

        middle_index = legacy_bedrock_import.BULK_INSERT_BATCH_ROWS + 3
        middle = conn.execute(
            "SELECT role FROM messages WHERE id = ?",
            [f"bedrock:msg:{middle_index}"],
        ).fetchone()
        assert middle is not None
        assert middle["role"] == ("user" if middle_index % 2 == 0 else "assistant")

        last = conn.execute(
            "SELECT role FROM messages WHERE id = ?",
            [f"bedrock:msg:{total_messages - 1}"],
        ).fetchone()
        assert last is not None
    finally:
        storage_module.close()


def test_bulk_upsert_flushes_when_byte_cap_reached_before_row_cap():
    """Large individual rows must flush before hitting the row-count cap.

    Turso / Fly.io edge rejects requests above a few MB with
    ``stream not found``. A single heavy Bedrock message can be >1 MB
    (large tool-call payloads), so packing 500 of those into one request
    would blow past the edge limit long before hitting ``batch_size``. The
    byte-aware batcher must flush early on a byte trigger.
    """
    from harness.memory.legacy_bedrock_import import _bulk_upsert

    flush_calls: list[int] = []

    class _RecordingConn:
        def execute(self, sql, parameters=None):
            # parameters is flat across all rows in the batch; divide by
            # column count to recover rows-per-batch.
            assert isinstance(parameters, tuple)
            flush_calls.append(len(parameters) // 2)

    # Each row is ~600 bytes of string. With batch_size=1000 we'd never
    # flush on row-count. With max_batch_bytes=1500 a flush must fire
    # roughly every 2-3 rows.
    big_blob = "x" * 600
    rows = [(f"id-{i}", big_blob) for i in range(10)]
    total = _bulk_upsert(
        _RecordingConn(),
        table="t",
        columns=("id", "blob"),
        rows=rows,
        batch_size=1000,
        max_batch_bytes=1500,
    )
    assert total == 10
    # All 10 rows delivered, but split into multiple batches driven by
    # bytes, not rows.
    assert sum(flush_calls) == 10
    assert len(flush_calls) >= 4, (
        f"expected byte-cap to force multiple flushes, got {flush_calls}"
    )
    # No single batch exceeds ~3 rows (3 × 600 = 1800 > 1500, triggers flush).
    assert max(flush_calls) <= 3


def test_bulk_upsert_admits_single_oversize_row_alone():
    """One row bigger than max_batch_bytes must still be delivered.

    A row can't be split across batches. The batcher must let an oversize
    row flush in a batch-of-one rather than hang or loop.
    """
    from harness.memory.legacy_bedrock_import import _bulk_upsert

    flush_sizes: list[int] = []

    class _RecordingConn:
        def execute(self, sql, parameters=None):
            assert isinstance(parameters, tuple)
            flush_sizes.append(len(parameters) // 2)

    # One huge row + a couple small rows.
    rows = [
        ("huge", "y" * 5000),
        ("small-1", "a"),
        ("small-2", "b"),
    ]
    total = _bulk_upsert(
        _RecordingConn(),
        table="t",
        columns=("id", "blob"),
        rows=rows,
        batch_size=1000,
        max_batch_bytes=1000,
    )
    assert total == 3
    # Huge row flushes alone; the two small rows can share a batch.
    assert flush_sizes[0] == 1  # oversize row solo
    assert sum(flush_sizes) == 3


def test_bulk_upsert_passes_tuple_params_to_execute():
    """libsql_experimental (Turso) only accepts tuples for ``parameters``.

    sqlite3 happily accepts any sequence, so a regression back to ``list``
    would pass every test that exercises the real storage layer — and then
    blow up at deploy time against remote libSQL with
    ``TypeError: 'list' object cannot be converted to 'PyTuple'``.

    This guards the contract at the ``conn.execute`` boundary directly.
    """
    from harness.memory.legacy_bedrock_import import _bulk_upsert

    captured: list[tuple[str, object]] = []

    class _RecordingConn:
        def execute(self, sql, parameters=None):
            # The real production backend raises TypeError on non-tuple
            # parameters, so model that here.
            if parameters is not None and not isinstance(parameters, tuple):
                raise TypeError(
                    "argument 'parameters': "
                    f"{type(parameters).__name__!r} object cannot be converted to 'PyTuple'"
                )
            captured.append((sql, parameters))

    rows = [(f"id-{i}", i, "x") for i in range(3)]
    total = _bulk_upsert(
        _RecordingConn(),
        table="t",
        columns=("id", "n", "s"),
        rows=rows,
        batch_size=2,
    )
    assert total == 3
    # 3 rows with batch_size=2 → one full + one tail = 2 batches = 2 calls.
    assert len(captured) == 2
    assert all(isinstance(params, tuple) for _, params in captured)
    # Full batch: 2 rows × 3 cols = 6 params.
    assert len(captured[0][1]) == 6
    # Tail batch: 1 row × 3 cols = 3 params.
    assert len(captured[1][1]) == 3


def test_import_legacy_bedrock_memory_is_idempotent(tmp_path, monkeypatch):
    """Re-running the same import must not duplicate rows.

    Prod-deploy migrations may retry per-agent imports (timeout, pod restart,
    operator rerun). Legacy ids are stable, and the upsert path relies on
    ``INSERT OR REPLACE``. A regression that regresses back to plain INSERT
    — or that forgets to include the primary key in the VALUES clause — would
    leak into row-count doubling on replay.
    """
    monkeypatch.setenv("HARNESS_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "HARNESS_MIGRATIONS_DIR",
        str(Path(__file__).parent.parent.parent / "src" / "harness" / "memory" / "migrations"),
    )

    from harness.core import storage as storage_module

    importlib.reload(storage_module)

    from harness.memory.legacy_bedrock_import import import_legacy_bedrock_memory

    payload = {
        "agent_id": "agent-replay",
        "messages": [
            {
                "id": "bedrock:replay:1",
                "ts_ns": 1,
                "role": "user",
                "content": {"role": "user", "content": "once"},
            }
        ],
        "one_minute_summaries": [
            {
                "id": "bedrock:om:1",
                "date": "2026-04-24",
                "hour": 1,
                "minute": 2,
                "summary": "first",
                "message_count": 1,
                "created_at_ns": 10,
            }
        ],
        "five_minute_summaries": [],
        "hourly_summaries": [],
        "daily_summaries": [],
        "weekly_summaries": [],
        "monthly_summaries": [],
    }
    payload_path = tmp_path / "agent-replay.json"
    payload_path.write_text(json.dumps(payload))

    import_legacy_bedrock_memory("agent-replay", payload_path)
    # Mutate the summary text so REPLACE is observable and we can distinguish
    # the second run from the first.
    payload["one_minute_summaries"][0]["summary"] = "second"
    payload_path.write_text(json.dumps(payload))
    import_legacy_bedrock_memory("agent-replay", payload_path)

    conn = storage_module.load("agent-replay")
    try:
        assert conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 1
        assert (
            conn.execute("SELECT COUNT(*) AS c FROM one_minute_summaries").fetchone()[
                "c"
            ]
            == 1
        )
        assert (
            conn.execute(
                "SELECT summary FROM one_minute_summaries WHERE id = ?",
                ["bedrock:om:1"],
            ).fetchone()["summary"]
            == "second"
        )
    finally:
        storage_module.close()
