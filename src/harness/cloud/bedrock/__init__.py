"""Bedrock cloud integration.

The only module in the tree that reads ``BEDROCK_URL`` / ``BEDROCK_TOKEN`` or
speaks Bedrock's HTTP. Imports from here are guarded by ``autoconfigure()``
(or explicit opt-in from the CLI).
"""

from harness.cloud.bedrock.config import (
    create_dev_agent,
    create_eval_agent,
    fetch_harness_config,
    resolve_product,
)
from harness.cloud.bedrock.runtime import BedrockAgentRuntime
from harness.cloud.bedrock.trace_sink import BedrockTraceSink

__all__ = [
    "BedrockAgentRuntime",
    "BedrockTraceSink",
    "create_dev_agent",
    "create_eval_agent",
    "fetch_harness_config",
    "resolve_product",
]
