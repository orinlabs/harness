"""A local HTTP server that implements the infra platform contract for tests.

The harness code talks to this via real HTTP, so tests exercise real serialization
and real error paths. Only the implementation on this side is fake; the wire
protocol is identical to what the production platform will serve.

Endpoints:
  POST   /api/tracing/traces/                  record trace open
  PATCH  /api/tracing/traces/{id}/             record trace close (+ metadata)
  POST   /api/tracing/spans/                   record span open
  PATCH  /api/tracing/spans/{id}/              record span close (+ metadata)
  POST   /api/cloud/agents/{agent_id}/sleep/   record sleep request
  POST   /fake_tools/{name}                    dispatch to a test-registered
                                                tool handler

Tests use `FakePlatform` as a context manager or via the `fake_platform` pytest
fixture. Call `.record` to inspect received requests, and `.register_tool` to
bind a handler function to a tool name.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


@dataclass
class RecordedRequest:
    method: str
    path: str
    body: Any
    headers: dict[str, str]


ToolHandler = Callable[[dict, dict], dict]
"""Signature: (args, envelope) -> response body.

`args` is `envelope["args"]`. `envelope` has `agent_id`, `run_id`, etc.
Return value is merged into the HTTP response body as JSON.

To test error paths, raise a `FakeToolError` from inside the handler.
"""


@dataclass
class FakeToolError(Exception):
    status: int
    body: Any


@dataclass
class FakePlatform:
    host: str = "127.0.0.1"
    port: int = 0
    _server: HTTPServer | None = None
    _thread: threading.Thread | None = None
    requests: list[RecordedRequest] = field(default_factory=list)
    traces_open: dict[str, dict] = field(default_factory=dict)
    traces_closed: dict[str, dict] = field(default_factory=dict)
    spans_open: dict[str, dict] = field(default_factory=dict)
    spans_closed: dict[str, dict] = field(default_factory=dict)
    sleep_requests: list[dict] = field(default_factory=list)
    tool_handlers: dict[str, ToolHandler] = field(default_factory=dict)

    # Convenience views
    def spans_by_name(self, name: str) -> list[dict]:
        """Return all opened spans with the given name."""
        return [s for s in self.spans_open.values() if s["name"] == name]

    def closed_span(self, span_id: str) -> dict | None:
        return self.spans_closed.get(span_id)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def register_tool(self, name: str, handler: ToolHandler) -> None:
        self.tool_handlers[name] = handler

    def start(self) -> None:
        handler_cls = _make_handler(self)
        self._server = HTTPServer((self.host, self.port), handler_cls)
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

    def __enter__(self) -> FakePlatform:
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
    raise RuntimeError(f"fake_platform did not start on {host}:{port}")


def _make_handler(platform: FakePlatform):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            pass  # silence default stderr logging

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw.decode("utf-8", errors="replace")

        def _record(self, body: Any) -> None:
            platform.requests.append(
                RecordedRequest(
                    method=self.command,
                    path=self.path,
                    body=body,
                    headers=dict(self.headers),
                )
            )

        def _write(self, status: int, body: Any) -> None:
            data = json.dumps(body).encode() if body is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if data:
                self.wfile.write(data)

        def do_POST(self):
            body = self._read_json()
            self._record(body)

            if self.path == "/api/tracing/traces/":
                platform.traces_open[body["id"]] = body
                self._write(200, {"ok": True})
                return

            if self.path == "/api/tracing/spans/":
                platform.spans_open[body["id"]] = body
                self._write(200, {"ok": True})
                return

            if (
                self.path.startswith("/api/cloud/agents/")
                and self.path.endswith("/sleep/")
            ):
                agent_id = self.path[
                    len("/api/cloud/agents/"):-len("/sleep/")
                ]
                platform.sleep_requests.append({"agent_id": agent_id, **(body or {})})
                self._write(200, {"ok": True})
                return

            if self.path.startswith("/fake_tools/"):
                name = self.path[len("/fake_tools/"):]
                handler = platform.tool_handlers.get(name)
                if handler is None:
                    self._write(404, {"error": f"no handler for {name}"})
                    return
                envelope = body or {}
                args = envelope.get("args", {})
                try:
                    result = handler(args, envelope)
                    self._write(200, result)
                except FakeToolError as e:
                    if isinstance(e.body, (dict, list)):
                        self._write(e.status, e.body)
                    else:
                        data = str(e.body).encode()
                        self.send_response(e.status)
                        self.send_header("Content-Type", "text/plain")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                return

            self._write(404, {"error": f"no route for POST {self.path}"})

        def do_PATCH(self):
            body = self._read_json()
            self._record(body)

            if (
                self.path.startswith("/api/tracing/traces/")
                and self.path.endswith("/")
                and self.path.count("/") == 5
            ):
                trace_id = self.path[len("/api/tracing/traces/"):-1]
                platform.traces_closed[trace_id] = body
                self._write(200, {"ok": True})
                return

            if (
                self.path.startswith("/api/tracing/spans/")
                and self.path.endswith("/")
                and self.path.count("/") == 5
            ):
                span_id = self.path[len("/api/tracing/spans/"):-1]
                platform.spans_closed[span_id] = body
                self._write(200, {"ok": True})
                return

            self._write(404, {"error": f"no route for PATCH {self.path}"})

    return Handler
