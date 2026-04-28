from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests.fake_platform import FakePlatform

load_dotenv(Path(__file__).parent.parent / ".env")

# Belt-and-suspenders: `.env` is the production config (real Daytona API key,
# real platform URL). Tests must *never* accidentally talk to those services,
# so we unset the relevant vars globally here. Individual tests that genuinely
# want a live platform re-set them via `monkeypatch` or fixtures.
for _var in (
    "DAYTONA_API_KEY",
    "DAYTONA_API_URL",
    "DAYTONA_TARGET",
    "BEDROCK_URL",
    "BEDROCK_TOKEN",
):
    os.environ.pop(_var, None)


@pytest.fixture(autouse=True)
def _reset_trace_sink_between_tests():
    """Force the tracer to re-pick its sink each test.

    The tracer caches its sink on first access (see ``get_trace_sink``). If
    one test activates Bedrock via ``fake_platform`` and a later test runs
    without Bedrock env, the cached Bedrock sink would leak into the second
    test. Clearing the cache before every test makes the autoconfig run
    against the current env.
    """
    from harness.core import tracer

    tracer._reset_sink_for_tests()
    yield
    tracer._reset_sink_for_tests()


@pytest.fixture
def fake_platform(monkeypatch):
    """Start a FakePlatform and point Bedrock env vars at it.

    Tests that take this fixture get a ``BedrockTraceSink`` and
    ``BedrockAgentRuntime`` autoconfigured by the tracer + Harness, both
    aimed at the in-process fake server. The protocol is identical to what
    real Bedrock serves, so coverage here catches real wire-format bugs.
    """
    platform = FakePlatform()
    platform.start()
    monkeypatch.setenv("BEDROCK_URL", platform.url)
    monkeypatch.setenv("BEDROCK_TOKEN", "test-token")

    # The autouse reset fixture already cleared the cached sink; next tracer
    # access will pick up a BedrockTraceSink pointing at the fake's URL.
    try:
        yield platform
    finally:
        platform.stop()


@pytest.fixture
def openrouter_key():
    """Return the OpenRouter API key. Fail loudly if missing (never skip)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.fail(
            "OPENROUTER_API_KEY not set. Live-API tests fail loudly on purpose — "
            "set the key or remove the test."
        )
    return key
