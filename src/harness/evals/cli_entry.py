"""`harness eval` subcommand entrypoint.

Placed in ``harness.evals.cli_entry`` so that the top-level ``harness.cli``
module can defer importing ``harness.evals`` (and its eval-only dependencies
like the simulation runner / fake adapters) until the user actually asks
for the eval path.

Evals run standalone by default: the scenario's fake adapters live in-process
and span-sink/agent-runtime come from ``autoconfigure()``. When
``BEDROCK_URL`` + ``BEDROCK_TOKEN`` are set, spans flow to Bedrock (via
``BedrockTraceSink``) -- but we don't create a Bedrock agent row, so the
spans are visible under whatever agent_id we synthesize locally. Users who
want a dedicated Bedrock eval agent should create one via the Bedrock API
and pass its id via ``AGENT_ID`` env (TODO).
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import pkgutil
import sys
import uuid

from harness.cli import (
    DEFAULT_ENV,
    _apply_bedrock_env,
    _bedrock_configured,
    _ensure_secrets_env,
    _load_env,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------


def _discover_simulation_classes():
    """Walk ``harness.evals.scenarios`` and return all Simulation subclasses.

    Returns a list of (module_name, class) tuples. Modules that fail to
    import are logged and skipped so one broken scenario doesn't take the
    whole runner down.
    """
    from harness.evals import scenarios as scenarios_pkg
    from harness.evals.base import Simulation

    out: list[tuple[str, type[Simulation]]] = []
    for mod_info in pkgutil.walk_packages(
        scenarios_pkg.__path__, prefix=f"{scenarios_pkg.__name__}."
    ):
        try:
            mod = importlib.import_module(mod_info.name)
        except Exception as e:
            logger.warning("scenario module %s failed to import: %s", mod_info.name, e)
            continue
        for attr in vars(mod).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, Simulation)
                and attr is not Simulation
                and getattr(attr, "name", "")
            ):
                out.append((mod_info.name, attr))
    return out


def _find_scenario(name: str):
    candidates = _discover_simulation_classes()
    for _mod, cls in candidates:
        if cls.name == name:
            return cls
    available = sorted({cls.name for _m, cls in candidates if cls.name})
    raise SystemExit(
        f"scenario {name!r} not found. available: {', '.join(available) or '(none)'}"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run(args, parser: argparse.ArgumentParser) -> int:
    _load_env()

    for k, v in DEFAULT_ENV.items():
        os.environ.setdefault(k, v)

    _apply_bedrock_env(args)
    _ensure_secrets_env(parser)

    scenario_cls = _find_scenario(args.scenario)
    logger.info("found scenario %s from %s", scenario_cls.name, scenario_cls.__module__)

    model = (
        args.model
        or scenario_cls.agent_overrides.model
        or "claude-haiku-4-5"
    )
    system_prompt = scenario_cls.agent_overrides.system_prompt or ""
    reasoning_effort = (
        args.reasoning_effort
        or scenario_cls.agent_overrides.reasoning_effort
        or None
    )

    # No Bedrock agent row is created. Spans flow to Bedrock via the
    # autoconfigured trace sink when the env is set; the agent_id below is
    # local and synthesized.
    agent_id = f"eval-{scenario_cls.name}-{uuid.uuid4().hex[:8]}"
    print(f"eval agent id: {agent_id}", file=sys.stderr)

    # Build AgentConfig with every fake adapter wired in.
    from harness.config import AgentConfig
    from harness.evals.fakes import (
        FakeComputerAdapter,
        FakeEmailAdapter,
        FakeSMSAdapter,
        TestContactsAdapter,
        TestDocumentsAdapter,
        TestProjectsAdapter,
    )

    config = AgentConfig(
        id=agent_id,
        model=model,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        tools=[
            *FakeEmailAdapter.make_tools(),
            *FakeSMSAdapter.make_tools(),
            *TestContactsAdapter.make_tools(),
            *TestProjectsAdapter.make_tools(),
            *TestDocumentsAdapter.make_tools(),
            *FakeComputerAdapter.make_tools(),
        ],
    )

    from harness.evals.runner import SimulationRunner

    runner = SimulationRunner(scenario_cls, agent_id=agent_id, agent_config=config)
    try:
        result = runner.execute()
    except KeyboardInterrupt:
        logger.warning("eval interrupted")
        return 130
    except Exception:
        logger.exception("eval failed")
        raise

    passed = sum(1 for cp in runner.checkpoint_results if cp["passed"])
    total = len(runner.checkpoint_results)
    print(
        f"eval complete: {passed}/{total} checkpoints passed",
        file=sys.stderr,
    )
    if _bedrock_configured():
        base = os.environ["BEDROCK_URL"].rstrip("/")
        print(
            f"View traces: {base}/admin/tracing/trace/ (filter by agent_id={agent_id})",
            file=sys.stderr,
        )

    _ = result
    return 0 if total > 0 and passed == total else (1 if total > 0 else 0)
