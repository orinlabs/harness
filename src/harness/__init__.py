from harness.config import AdapterConfig, AgentConfig, ExternalToolSpec
from harness.context import RunContext

__all__ = [
    "AdapterConfig",
    "AgentConfig",
    "ExternalToolSpec",
    "Harness",
    "RunContext",
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
