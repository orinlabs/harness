from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests.fake_platform import FakePlatform

load_dotenv(Path(__file__).parent.parent / ".env")

# Belt-and-suspenders: `.env` is the production config (real Turso creds, real
# platform URL). Tests must *never* accidentally talk to those services, so we
# unset the relevant vars globally here. Individual tests that genuinely want
# remote Turso or a live platform re-set them via `monkeypatch` or fixtures.
for _var in (
    "HARNESS_DATABASE_URL",
    "HARNESS_DATABASE_TOKEN",
    "HARNESS_TURSO_ORG",
    "HARNESS_TURSO_GROUP",
    "HARNESS_TURSO_PLATFORM_TOKEN",
    "BEDROCK_URL",
    "BEDROCK_TOKEN",
):
    os.environ.pop(_var, None)


@pytest.fixture
def fake_platform(monkeypatch):
    """Start a FakePlatform, wire env vars, reload core clients so they pick them up."""
    platform = FakePlatform()
    platform.start()
    monkeypatch.setenv("BEDROCK_URL", platform.url)
    monkeypatch.setenv("BEDROCK_TOKEN", "test-token")

    from harness.core import runtime_api, tracer

    importlib.reload(tracer)
    importlib.reload(runtime_api)

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
