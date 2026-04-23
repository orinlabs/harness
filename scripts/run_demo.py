"""Run a real agent end-to-end.

Starts the fake platform server in-process, mounts a mock SMS inbox as two
external tools (read + send), and runs Harness against live OpenRouter. The
user's message comes in through the inbox — no direct memory injection.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from fake_platform import FakePlatform  # noqa: E402


# -----------------------------------------------------------------------------
# Mock SMS inbox. Lives in-process; the fake platform exposes it as tool URLs.
# -----------------------------------------------------------------------------


@dataclass
class SmsMessage:
    id: str
    from_number: str
    body: str
    read: bool = False


@dataclass
class MockSmsInbox:
    agent_number: str = "+15550000000"
    incoming: list[SmsMessage] = field(default_factory=list)
    outgoing: list[dict] = field(default_factory=list)

    def seed(self, from_number: str, body: str) -> None:
        self.incoming.append(
            SmsMessage(
                id=f"msg-{len(self.incoming) + 1}",
                from_number=from_number,
                body=body,
            )
        )

    def check(self) -> list[dict]:
        unread = [m for m in self.incoming if not m.read]
        for m in unread:
            m.read = True
        return [{"id": m.id, "from": m.from_number, "body": m.body} for m in unread]

    def send(self, to: str, body: str) -> dict:
        entry = {"id": f"out-{len(self.outgoing) + 1}", "to": to, "body": body}
        self.outgoing.append(entry)
        return entry


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set. Put it in .env or export it.")
        sys.exit(1)

    platform = FakePlatform()
    platform.start()
    print(f"[fake_platform] listening at {platform.url}")

    tmp = Path(tempfile.mkdtemp(prefix="harness_demo_"))
    os.environ["HARNESS_PLATFORM_URL"] = platform.url
    os.environ["HARNESS_PLATFORM_TOKEN"] = "demo-token"
    os.environ["HARNESS_STORAGE_ROOT"] = str(tmp)
    print(f"[storage] sqlite files under {tmp}")

    inbox = MockSmsInbox(agent_number="+15550000000")
    inbox.seed(from_number="+15551234567", body="hey what's the weather in Tokyo?")

    def weather_handler(args, envelope):
        city = str(args.get("city") or "").lower()
        data = {
            "san francisco": {"temp_f": 62, "cond": "foggy"},
            "new york": {"temp_f": 71, "cond": "partly cloudy"},
            "tokyo": {"temp_f": 78, "cond": "clear"},
        }.get(city, {"temp_f": 68, "cond": "unknown"})
        return {"text": f"{city.title()}: {data['temp_f']}F, {data['cond']}"}

    def sms_check_handler(args, envelope):
        messages = inbox.check()
        if not messages:
            return {"text": "Inbox is empty. No new messages."}
        return {
            "text": "New SMS messages (now marked read):\n"
            + json.dumps(messages, indent=2)
        }

    def sms_send_handler(args, envelope):
        to = str(args.get("to") or "")
        body = str(args.get("body") or "")
        if not to or not body:
            return {"text": "Error: both 'to' and 'body' are required."}
        sent = inbox.send(to=to, body=body)
        return {"text": f"SMS sent (id={sent['id']}) to {to}."}

    platform.register_tool("get_weather", weather_handler)
    platform.register_tool("sms_check_inbox", sms_check_handler)
    platform.register_tool("sms_send", sms_send_handler)

    from harness import AdapterConfig, AgentConfig, ExternalToolSpec, Harness

    weather_tool = ExternalToolSpec(
        name="get_weather",
        description="Get current weather for a US or major world city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name."}},
            "required": ["city"],
        },
        url=f"{platform.url}/fake_tools/get_weather",
    )

    sms_check_tool = ExternalToolSpec(
        name="sms_check_inbox",
        description=(
            "Read any unread SMS messages addressed to you. Returns the sender's "
            "phone number and message body for each. Automatically marks them as read."
        ),
        parameters={"type": "object", "properties": {}},
        url=f"{platform.url}/fake_tools/sms_check_inbox",
    )

    sms_send_tool = ExternalToolSpec(
        name="sms_send",
        description="Send an SMS reply to a phone number.",
        parameters={
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "E.164 phone number, e.g. +15551234567.",
                },
                "body": {
                    "type": "string",
                    "description": "SMS body. Keep under 160 characters.",
                },
            },
            "required": ["to", "body"],
        },
        url=f"{platform.url}/fake_tools/sms_send",
    )

    config = AgentConfig(
        id="demo-sms-agent",
        model="anthropic/claude-opus-4.7",
        system_prompt=(
            f"You are an SMS concierge assistant. Your phone number is {inbox.agent_number}. "
            "On every wake cycle you should:\n"
            "1. Call `sms_check_inbox` to see if anyone texted you.\n"
            "2. For each message, answer the question (use tools like `get_weather` "
            "as needed) and reply with `sms_send` to the sender's number.\n"
            "3. Once the inbox is empty and you have sent all replies, call `sleep` "
            'with until="2099-01-01T00:00:00Z" and a reason describing what you did.\n'
            "Keep SMS replies short and friendly. Never reply with tool-call JSON as text."
        ),
        adapters=[
            AdapterConfig(
                name="weather",
                description="Weather lookups",
                tools=[weather_tool],
            ),
            AdapterConfig(
                name="sms",
                description="SMS inbox read/write",
                tools=[sms_check_tool, sms_send_tool],
            ),
        ],
    )

    banner("RUNNING HARNESS")
    Harness(config, run_id="demo-sms-run-1").run()
    banner("DONE")

    print(
        f"\n  turns: {sum(1 for s in platform.spans_open.values() if s['name'].startswith('turn_'))}"
    )
    print(
        f"  LLM calls: {sum(1 for s in platform.spans_open.values() if s['span_type'] == 'llm')}"
    )
    print(
        f"  tool calls: {sum(1 for s in platform.spans_open.values() if s['span_type'] == 'tool')}"
    )

    total_cost = 0.0
    for span_id, span in platform.spans_closed.items():
        if platform.spans_open.get(span_id, {}).get("span_type") == "llm":
            total_cost += (
                span.get("metadata", {}).get("llm_cost", {}).get("total_cost_usd", 0.0)
            )
    print(f"  total cost: ${total_cost:.6f}")

    if platform.sleep_requests:
        sr = platform.sleep_requests[0]
        print(f"  slept until: {sr.get('until')} (reason: {sr.get('reason')!r})")

    banner("SMS INBOX STATE")
    print("Incoming:")
    for m in inbox.incoming:
        flag = "[read]  " if m.read else "[UNREAD]"
        print(f"  {flag} {m.from_number} -> {m.body!r}")
    print("\nOutgoing:")
    for m in inbox.outgoing:
        print(f"  -> {m['to']}: {m['body']!r}")

    banner("TOOL INVOCATIONS")
    for req in platform.requests:
        if req.path.startswith("/fake_tools/"):
            name = req.path[len("/fake_tools/") :]
            print(f"  {name}({json.dumps(req.body.get('args'))})")

    banner("MESSAGE LOG IN SQLITE")
    from harness.core import storage

    storage.load("demo-sms-agent")
    for row in storage.db.execute(
        "SELECT role, content_json FROM messages ORDER BY ts_ns"
    ):
        content = json.loads(row["content_json"])
        if content.get("tool_calls"):
            names = [tc["function"]["name"] for tc in content["tool_calls"]]
            preview = f"[tool_calls] {json.dumps(names)}"
        else:
            preview = content.get("content") or "(empty)"
        if isinstance(preview, str) and len(preview) > 180:
            preview = preview[:180] + "..."
        print(f"  [{row['role']:9}] {preview}")
    storage.close()

    platform.stop()


if __name__ == "__main__":
    main()
