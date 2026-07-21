"""A local HTTP server implementing the `current` workflow contract for tests.

Same philosophy as ``tests/fake_platform.py``: the harness talks to this via
real HTTP so serialization, auth headers, and error paths are exercised for
real; only the implementation behind the wire is fake.

Endpoints (v0 contract):
  GET  /api/workflows/runs/{run_id}/definition/     -> the configured definition
                                                     response (definition +
                                                     steps_state + approvals)
  POST /api/workflows/runs/{run_id}/transitions/  -> 200, recorded in order
  POST /api/workflows/runs/{run_id}/records/      -> 201 {"id": uuid}, or a
                                                     test-injected status per
                                                     record_type (e.g. 422)

Tests mutate ``definition_response`` between runner invocations to simulate
current's journal advancing (e.g. a gate decision landing before a resume).
``events`` interleaves transitions and records in arrival order so tests can
assert journal-first ordering across both endpoints.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


@dataclass
class FakeCurrent:
    run_id: str = "run-1"
    host: str = "127.0.0.1"
    port: int = 0
    definition_response: dict[str, Any] = field(default_factory=dict)
    # record_type -> (status, body) override; e.g. {"agent_proposal": (422, {...})}
    record_failures: dict[str, tuple[int, Any]] = field(default_factory=dict)
    transitions: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    events: list[tuple[str, dict]] = field(default_factory=list)
    auth_headers: list[str] = field(default_factory=list)
    _server: HTTPServer | None = None
    _thread: threading.Thread | None = None

    # -- convenience views -------------------------------------------------

    def transition_tuples(self) -> list[tuple[str, str, int]]:
        return [(t["step_id"], t["status"], t["attempt"]) for t in self.transitions]

    def records_of(self, record_type: str) -> list[dict]:
        return [r for r in self.records if r["record_type"] == record_type]

    def reset_journal(self) -> None:
        """Clear captured traffic (keeps the definition response) between runs."""
        self.transitions.clear()
        self.records.clear()
        self.events.clear()
        self.auth_headers.clear()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._server = HTTPServer((self.host, self.port), _make_handler(self))
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        _wait_for_port(self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> FakeCurrent:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _wait_for_port(host: str, port: int, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.01)
    raise RuntimeError(f"fake_current did not start on {host}:{port}")


def _make_handler(fake: FakeCurrent):
    prefix = f"/api/workflows/runs/{fake.run_id}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            pass

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            return json.loads(raw) if raw else None

        def _write(self, status: int, body: Any) -> None:
            data = json.dumps(body).encode() if body is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if data:
                self.wfile.write(data)

        def do_GET(self):
            fake.auth_headers.append(self.headers.get("Authorization") or "")
            if self.path == f"{prefix}/definition/":
                self._write(200, fake.definition_response)
                return
            self._write(404, {"error": f"no route for GET {self.path}"})

        def do_POST(self):
            fake.auth_headers.append(self.headers.get("Authorization") or "")
            body = self._read_json()

            if self.path == f"{prefix}/transitions/":
                fake.transitions.append(body)
                fake.events.append(("transition", body))
                self._write(200, {"ok": True})
                return

            if self.path == f"{prefix}/records/":
                failure = fake.record_failures.get(body.get("record_type"))
                if failure is not None:
                    status, resp_body = failure
                    self._write(status, resp_body)
                    return
                fake.records.append(body)
                fake.events.append(("record", body))
                self._write(201, {"id": str(uuid.uuid4())})
                return

            self._write(404, {"error": f"no route for POST {self.path}"})

    return Handler
