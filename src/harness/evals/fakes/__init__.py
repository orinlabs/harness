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

from . import base, computer, contacts, documents, email, projects, sms
from .computer import FakeComputerAdapter
from .contacts import FakeContactsAdapter, TestContactsAdapter
from .documents import FakeDocumentsAdapter, TestDocumentsAdapter
from .email import FakeEmailAdapter
from .projects import FakeProjectsAdapter, TestProjectsAdapter
from .sms import FakeSMSAdapter

__all__ = [
    "FakeComputerAdapter",
    "FakeContactsAdapter",
    "FakeDocumentsAdapter",
    "FakeEmailAdapter",
    "FakeProjectsAdapter",
    "FakeSMSAdapter",
    "TestContactsAdapter",
    "TestDocumentsAdapter",
    "TestProjectsAdapter",
    "base",
    "computer",
    "contacts",
    "documents",
    "email",
    "projects",
    "sms",
]
