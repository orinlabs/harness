"""Run a Harness against a live bedrock agent.

Fetches the agent's harness-config from bedrock, constructs an AgentConfig,
and runs. Traces flow to bedrock's /api/tracing/* endpoints; storage goes to
Turso; external tool calls hit bedrock's /api/cloud/agents/{id}/tools/{name}/invoke/.

Env (from .env):
    HARNESS_PLATFORM_URL        (e.g. http://127.0.0.1:8000)
    HARNESS_PLATFORM_TOKEN      (bedrock product API key)
    HARNESS_TURSO_ORG           (for agent DB storage)
    HARNESS_TURSO_PLATFORM_TOKEN
    HARNESS_DATABASE_TOKEN
    OPENROUTER_API_KEY

Usage:
    uv run python scripts/run_live.py <AGENT_UUID>
    AGENT_ID=<uuid> uv run python scripts/run_live.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def fetch_config(agent_id: str) -> dict:
    base = os.environ["HARNESS_PLATFORM_URL"].rstrip("/")
    token = os.environ["HARNESS_PLATFORM_TOKEN"]
    r = httpx.get(
        f"{base}/api/cloud/agents/{agent_id}/harness-config/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    agent_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AGENT_ID")
    if not agent_id:
        print("usage: run_live.py <AGENT_UUID>  (or set AGENT_ID=<uuid>)")
        sys.exit(1)

    for required in (
        "HARNESS_PLATFORM_URL",
        "HARNESS_PLATFORM_TOKEN",
        "HARNESS_TURSO_ORG",
        "HARNESS_TURSO_PLATFORM_TOKEN",
        "HARNESS_DATABASE_TOKEN",
        "OPENROUTER_API_KEY",
    ):
        if not os.environ.get(required):
            print(f"missing env: {required}")
            sys.exit(1)

    banner(f"FETCH harness-config for agent {agent_id}")
    cfg = fetch_config(agent_id)
    print(f"  model: {cfg['model']}")
    print(f"  adapters: {[a['name'] for a in cfg.get('adapters', [])]}")
    tool_names = [t["name"] for a in cfg.get("adapters", []) for t in a.get("tools", [])]
    print(f"  tools: {tool_names}")

    from harness import AdapterConfig, AgentConfig, ExternalToolSpec, Harness

    # Optional model override via MODEL env var; handy for A/B-ing against
    # gpt-5, claude-opus-4.7, etc. without mutating bedrock's Agent row.
    model = os.environ.get("MODEL") or cfg["model"]
    reasoning_effort = os.environ.get("REASONING_EFFORT") or cfg.get("reasoning_effort")
    if model != cfg["model"]:
        print(f"  [override] model: {cfg['model']} -> {model}")
    if reasoning_effort and reasoning_effort != cfg.get("reasoning_effort"):
        print(f"  [override] reasoning_effort: {reasoning_effort}")

    config = AgentConfig(
        id=cfg["id"],
        model=model,
        system_prompt=cfg["system_prompt"],
        reasoning_effort=reasoning_effort,
        adapters=[
            AdapterConfig(
                name=a["name"],
                description=a["description"],
                tools=[
                    ExternalToolSpec(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["parameters"],
                        url=t["url"],
                        timeout_seconds=t.get("timeout_seconds", 60.0),
                    )
                    for t in a["tools"]
                ],
            )
            for a in cfg.get("adapters", [])
        ],
    )

    run_id = str(uuid.uuid4())
    banner(f"RUN  run_id={run_id}")
    Harness(config, run_id=run_id).run()
    banner("DONE")

    # Pull the trace back from bedrock so we can confirm it landed.
    base = os.environ["HARNESS_PLATFORM_URL"].rstrip("/")
    token = os.environ["HARNESS_PLATFORM_TOKEN"]
    r = httpx.get(
        f"{base}/api/tracing/traces/",
        headers={"Authorization": f"Bearer {token}"},
        params={"agent_id": agent_id},
        timeout=10.0,
    )
    traces = r.json()
    if isinstance(traces, dict):
        traces = traces.get("results", [])
    # Newest first; find the one matching run_agent for this agent
    for t in traces[:3]:
        print(
            f"  trace {t['id']}  name={t['name']}  ended={bool(t.get('ended_at'))}  "
            f"error={t.get('error')!r}"
        )
        md = t.get("metadata") or {}
        usage = md.get("usage") or {}
        if usage:
            print(
                f"    cost: ${usage.get('cost_usd', 0):.6f}  "
                f"tokens: in={usage.get('input_tokens')} out={usage.get('output_tokens')}  "
                f"llm_calls={usage.get('llm_calls')}"
            )

    print(
        f"\n  View this run in bedrock:\n  {base}/admin/tracing/trace/  "
        f"(filter by agent id {agent_id})"
    )


if __name__ == "__main__":
    main()
