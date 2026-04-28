"""Cloud integrations.

Opt-in glue that connects ``harness.core`` to an external control plane.
Every submodule here is the *only* place in the tree that reads the env vars
or speaks the HTTP of its particular backend. ``harness.core`` never imports
from ``harness.cloud``; the only bridge is ``harness.cloud.autoconfig``,
which is called from ``Harness.__init__`` (and only imports the
backend-specific sink when the corresponding env vars are present).

Currently only Bedrock is implemented.
"""
from harness.cloud.autoconfig import autoconfigure

__all__ = ["autoconfigure"]
