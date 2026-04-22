"""Chunk 1 verification: package imports cleanly, config constructs, import is cheap."""
import subprocess
import sys

import pytest


def test_cold_import_under_100ms():
    """Fresh interpreter import of `harness` should be well under 100ms.

    Measured in a subprocess so prior imports in the test process don't hide the cost.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import time; t = time.perf_counter(); import harness; print(time.perf_counter() - t)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = float(result.stdout.strip())
    assert elapsed < 0.1, f"harness import took {elapsed * 1000:.1f}ms, budget is 100ms"


def test_config_construction_no_io(monkeypatch):
    """Constructing AgentConfig + related dataclasses performs no filesystem or network I/O."""
    import socket

    def forbid_socket(*args, **kwargs):
        raise AssertionError("network I/O at config construction time")

    monkeypatch.setattr(socket, "socket", forbid_socket)

    from harness import AdapterConfig, AgentConfig, ExternalToolSpec

    tool = ExternalToolSpec(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {}},
        url="http://example.com/echo",
    )
    adapter = AdapterConfig(name="test", description="test adapter", tools=[tool])
    config = AgentConfig(
        id="agent-1",
        model="openai/gpt-4o-mini",
        system_prompt="hello",
        adapters=[adapter],
    )
    assert config.id == "agent-1"
    assert config.adapters[0].tools[0].url == "http://example.com/echo"


def test_core_subpackage_not_imported_by_default():
    """`core/` modules should not load during `import harness`.

    This protects cold-start time and proves no global state is established until
    Harness.run() is called.
    """
    # Run in a subprocess to get a fresh sys.modules.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import harness; import sys; "
            "bad = [m for m in sys.modules if m.startswith('harness.core')]; "
            "print(bad)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = result.stdout.strip()
    assert loaded == "[]", f"harness.core modules loaded during import: {loaded}"


@pytest.mark.parametrize(
    "module_name",
    ["harness", "harness.config", "harness.context", "harness.constants"],
)
def test_modules_importable(module_name):
    __import__(module_name)
