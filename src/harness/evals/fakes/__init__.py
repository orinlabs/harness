"""In-process fake adapters for eval scenarios.

Each submodule exposes an ``*Adapter`` class with a ``make_tools()``
classmethod that returns a flat ``list[Tool]`` suitable for splicing into
``AgentConfig.tools``, plus an ``inject_inbound_*`` helper for scenarios
to simulate inbound traffic.

Migrations:
    Before using any fake adapter, scenarios must ensure the extra tables
    exist by calling ``harness.evals.fakes.base.apply_migrations()`` after
    ``harness.core.storage.load(agent_id)``. The production
    ``storage.load()`` path intentionally does not apply these migrations.
"""
from __future__ import annotations

from . import base, computer, contacts, email, sms
from .computer import FakeComputerAdapter
from .contacts import FakeContactsAdapter
from .email import FakeEmailAdapter
from .sms import FakeSMSAdapter

__all__ = [
    "FakeComputerAdapter",
    "FakeContactsAdapter",
    "FakeEmailAdapter",
    "FakeSMSAdapter",
    "base",
    "computer",
    "contacts",
    "email",
    "sms",
]
