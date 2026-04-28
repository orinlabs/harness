from harness.config import AgentConfig, ExternalToolSpec, ToolAuth
from harness.context import RunContext

__all__ = [
    "AgentConfig",
    "ExternalToolSpec",
    "Harness",
    "RunContext",
    "ToolAuth",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy-load `Harness` so `import harness` stays fast.

    Importing `Harness` pulls in core/*, memory/*, tools/*, which load httpx etc.
    and we don't want that on the cold-start path for callers who just want the
    config dataclasses (e.g. the platform building an AgentConfig).
    """
    if name == "Harness":
        from harness.harness import Harness

        return Harness
    raise AttributeError(f"module 'harness' has no attribute {name!r}")
