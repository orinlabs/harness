"""Bedrock-backed ``AgentRuntime``.

Routes the built-in ``SleepTool`` through Bedrock's agent-scoped lifecycle
endpoint. Bedrock is then responsible for SIGTERMing this process and
spawning a fresh one at ``until``.
"""

from __future__ import annotations

from typing import Any

from harness.cloud.bedrock.client import auth_header, http, platform_url


class BedrockAgentRuntime:
    def sleep(self, agent_id: str, *, until: str, reason: str) -> dict[str, Any]:
        """POST /api/cloud/agents/{agent_id}/sleep/ and return the JSON body."""
        resp = http().post(
            f"{platform_url()}/api/cloud/agents/{agent_id}/sleep/",
            json={"until": until, "reason": reason},
            headers=auth_header(),
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}
