"""`harness eval` subcommand entrypoint.

Placed in `harness.evals.cli_entry` so that the top-level `harness.cli`
module can defer importing `harness.evals` (and its eval-only dependencies
like the simulation runner / fake adapters) until the user actually asks
for the eval path.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
import subprocess
import sys
import uuid

import httpx

from harness.cli import (
    DEFAULT_ENV,
    _ensure_secrets_env,
    _load_env,
    _resolve_bedrock_token,
    _resolve_bedrock_url,
    _resolve_product,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------


def _discover_simulation_classes():
    """Walk `harness.evals.scenarios` and return all Simulation subclasses.

    Returns a list of (module_name, class) tuples. Modules that fail to
    import are logged and skipped so one broken scenario doesn't take the
    whole runner down.
    """
    from harness.evals.base import Simulation
    from harness.evals import scenarios as scenarios_pkg

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
# Agent creation for evals
# ---------------------------------------------------------------------------


def _git_branch() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()[:40] or "unknown"
    except Exception:
        return "unknown"


def _create_eval_agent(
    bedrock_url: str,
    bedrock_token: str,
    *,
    scenario_name: str,
    product_id: str,
    model: str,
    system_prompt: str,
    template: str | None,
) -> dict:
    body = {
        "name": f"eval-{scenario_name}",
        "purpose": "eval",
        "product": product_id,
        "model": model,
        "system_prompt": system_prompt or "",
        "tags": [
            f"scenario:{scenario_name}",
            f"git-ref:{_git_branch()}",
            f"git-sha:{_git_sha()}",
        ],
    }
    if template:
        logger.warning(
            "# TODO(Phase 2): template not yet implemented server-side; "
            "ignoring --template=%s", template,
        )
    url = f"{bedrock_url.rstrip('/')}/api/cloud/agents/"
    resp = httpx.post(
        url,
        json=body,
        headers={"Authorization": f"Bearer {bedrock_token}"},
        timeout=15.0,
    )
    if resp.status_code >= 400:
        print(f"create eval agent failed: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run(args, parser: argparse.ArgumentParser) -> int:
    _load_env()
    import os

    for k, v in DEFAULT_ENV.items():
        os.environ.setdefault(k, v)

    bedrock_url = _resolve_bedrock_url(args)
    os.environ["BEDROCK_URL"] = bedrock_url
    bedrock_token = _resolve_bedrock_token(args)
    if not bedrock_token:
        parser.error("--bedrock-token is required (pass as flag or set $BEDROCK_TOKEN)")
    os.environ["BEDROCK_TOKEN"] = bedrock_token
    _ensure_secrets_env(parser)

    scenario_cls = _find_scenario(args.scenario)
    logger.info("found scenario %s from %s", scenario_cls.name, scenario_cls.__module__)

    product_id = _resolve_product(bedrock_url, bedrock_token, args.product)

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

    agent = _create_eval_agent(
        bedrock_url,
        bedrock_token,
        scenario_name=scenario_cls.name,
        product_id=product_id,
        model=model,
        system_prompt=system_prompt,
        template=args.template,
    )
    agent_id = agent["id"]
    print(f"created eval agent: {agent_id}", file=sys.stderr)

    # Build AgentConfig with every fake adapter wired in. Scenarios can
    # opt-in/out per-adapter later via AgentOverrides.adapters; for now
    # we always attach all four so every tool surface is available.
    from harness.config import AgentConfig
    from harness.evals.fakes import (
        FakeComputerAdapter,
        FakeContactsAdapter,
        FakeEmailAdapter,
        FakeSMSAdapter,
    )

    config = AgentConfig(
        id=agent_id,
        model=model,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        adapters=[
            FakeEmailAdapter.make_adapter_config(),
            FakeSMSAdapter.make_adapter_config(),
            FakeContactsAdapter.make_adapter_config(),
            FakeComputerAdapter.make_adapter_config(),
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

    # Summarise checkpoint results.
    passed = sum(1 for cp in runner.checkpoint_results if cp["passed"])
    total = len(runner.checkpoint_results)
    print(
        f"eval complete: {passed}/{total} checkpoints passed",
        file=sys.stderr,
    )
    print(f"View agent: {bedrock_url.rstrip('/')}/agents/{agent_id}", file=sys.stderr)

    _ = result
    _ = uuid  # reserved for future run_id plumbing
    return 0 if total > 0 and passed == total else (1 if total > 0 else 0)
