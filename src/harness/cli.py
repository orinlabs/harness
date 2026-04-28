"""Console entry point for the harness.

Two subcommands:

    harness agent [AGENT_ID] [options]    # Run a single agent run.
    harness delete-agent AGENT_ID [--purge]
                                          # Archive (default) or fully purge
                                          # agent-owned harness storage.
    harness reset-memory AGENT_ID         # Reset agent memory storage (keeps
                                          # the Daytona sandbox around).
    harness eval  SCENARIO   [options]    # Run a scenario eval end-to-end.

The `agent` path is the production-adjacent path. The `eval` path is for
offline evaluation against the in-process fake adapters (imports are
deferred into the `eval` branch so cold-start of `harness agent` never
pays the price of pulling `harness.evals` / `harness.evals.fakes`).

Environment is loaded from `.env` in cwd if present. Non-secret identifiers
(platform URL) have baked-in defaults.

Required env (secrets) common to both subcommands:
    DAYTONA_API_KEY       Daytona API key. Harness provisions one sandbox per
                          agent (labeled `harness.agent_id=<id>`) and keeps
                          that agent's sqlite DB inside it.
    OPENROUTER_API_KEY

Optional env (override defaults):
    BEDROCK_URL                       default: http://127.0.0.1:8000
    DAYTONA_API_URL                   default: https://app.daytona.io/api
    DAYTONA_TARGET                    SDK default region if unset
    HARNESS_DAYTONA_AUTO_STOP_MINUTES passed as `auto_stop_interval`
    MODEL                             override the agent's configured model
    REASONING_EFFORT                  override reasoning_effort (low|medium|high)
    LOG_LEVEL                         default: INFO
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


DEFAULT_ENV: dict[str, str] = {
    "BEDROCK_URL": "http://127.0.0.1:8000",
}

REQUIRED_ENV = (
    "DAYTONA_API_KEY",
    "OPENROUTER_API_KEY",
)


# ---------------------------------------------------------------------------
# Process-lifecycle helpers (shared by both subcommands)
# ---------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(level=numeric, handlers=[handler], force=True)

    if numeric > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _install_shutdown_handlers() -> None:
    from harness.core import tracer

    def _handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        logger.warning("received %s, closing open traces and exiting", sig_name)
        raise KeyboardInterrupt(sig_name)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError) as e:
            logger.debug("could not install handler for %s: %s", sig, e)

    def _atexit_close() -> None:
        try:
            tracer.close_all_open("harness process exiting")
        except Exception:
            logger.exception("tracer: close_all_open failed during atexit")

    atexit.register(_atexit_close)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env load")
        return
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        logger.debug("loading env from %s", env_path)
        load_dotenv(env_path)


# ---------------------------------------------------------------------------
# Shared: bedrock URL/token + env checks + git stamping
# ---------------------------------------------------------------------------


def _resolve_bedrock_url(args) -> str:
    """Precedence: --bedrock-url > --local > $BEDROCK_URL > default."""
    if getattr(args, "bedrock_url", None):
        return args.bedrock_url
    if getattr(args, "local", False):
        return "http://127.0.0.1:8000"
    return os.environ.get("BEDROCK_URL") or DEFAULT_ENV["BEDROCK_URL"]


def _resolve_bedrock_token(args) -> str | None:
    return getattr(args, "bedrock_token", None) or os.environ.get("BEDROCK_TOKEN")


def _ensure_secrets_env(parser: argparse.ArgumentParser) -> None:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        logger.error("missing required env: %s", ", ".join(missing))
        print(f"missing required env: {', '.join(missing)}", file=sys.stderr)
        parser.exit(2)


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


def _resolve_product(bedrock_url: str, bedrock_token: str, explicit: str | None) -> str:
    """Resolve product UUID for auto-created agents.

    1. `--product` flag wins.
    2. Else GET /api/products/products/ -- if exactly one visible, use it.
    3. Else raise.
    """
    if explicit:
        return explicit
    url = f"{bedrock_url.rstrip('/')}/api/products/products/"
    resp = httpx.get(url, headers={"Authorization": f"Bearer {bedrock_token}"}, timeout=10.0)
    resp.raise_for_status()
    products = resp.json()
    if isinstance(products, dict) and "results" in products:
        products = products["results"]
    if not isinstance(products, list) or not products:
        raise SystemExit("no products visible to this API key")
    if len(products) > 1:
        raise SystemExit("multiple products visible, pass --product <uuid>")
    return products[0]["id"]


# ---------------------------------------------------------------------------
# Config-building for `harness agent` (existing id path)
# ---------------------------------------------------------------------------


def _fetch_config(bedrock_url: str, bedrock_token: str, agent_id: str) -> dict:
    url = f"{bedrock_url.rstrip('/')}/api/cloud/agents/{agent_id}/harness-config/"
    logger.info("fetching harness config from %s", url)
    resp = httpx.get(url, headers={"Authorization": f"Bearer {bedrock_token}"}, timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def _build_config_from_harness_config(cfg: dict, *, model_override: str | None = None,
                                      reasoning_override: str | None = None):
    from harness import AdapterConfig, AgentConfig, ExternalToolSpec

    model = model_override or os.environ.get("MODEL") or cfg["model"]
    reasoning_effort = (
        reasoning_override
        or os.environ.get("REASONING_EFFORT")
        or cfg.get("reasoning_effort")
    )

    return AgentConfig(
        id=cfg["id"],
        model=model,
        system_prompt=cfg["system_prompt"],
        reasoning_effort=reasoning_effort,
        adapters=[
            AdapterConfig(
                name=a["name"],
                description=a["description"],
                tools=[
                    ExternalToolSpec(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["parameters"],
                        url=t["url"],
                        timeout_seconds=t.get("timeout_seconds", 60.0),
                    )
                    for t in a["tools"]
                ],
            )
            for a in cfg.get("adapters", [])
        ],
    )


def _create_dev_agent(
    bedrock_url: str,
    bedrock_token: str,
    *,
    product_id: str,
    model: str,
    system_prompt: str | None,
    template: str | None,
    branch: str,
    sha: str,
) -> dict:
    body = {
        "name": f"dev-{uuid.uuid4().hex[:8]}",
        "purpose": "dev",
        "product": product_id,
        "model": model,
        "tags": [f"git-ref:{branch}", f"git-sha:{sha}"],
    }
    if system_prompt:
        body["system_prompt"] = system_prompt
    if template:
        # Server-side template support is Phase 2. For now pass it through so
        # any future serializer can accept it, and warn loudly.
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
        print(f"create agent failed: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Subcommand: agent
# ---------------------------------------------------------------------------


def _cmd_agent(args, parser: argparse.ArgumentParser) -> int:
    _load_env()
    for k, v in DEFAULT_ENV.items():
        os.environ.setdefault(k, v)

    bedrock_url = _resolve_bedrock_url(args)
    os.environ["BEDROCK_URL"] = bedrock_url

    bedrock_token = _resolve_bedrock_token(args)
    if not bedrock_token:
        parser.error("--bedrock-token is required (pass as flag or set $BEDROCK_TOKEN)")
    os.environ["BEDROCK_TOKEN"] = bedrock_token

    _ensure_secrets_env(parser)

    agent_id = args.agent_id
    if agent_id:
        cfg = _fetch_config(bedrock_url, bedrock_token, agent_id)
        config = _build_config_from_harness_config(
            cfg,
            model_override=args.model,
            reasoning_override=args.reasoning_effort,
        )
    else:
        product_id = _resolve_product(bedrock_url, bedrock_token, args.product)
        model = args.model or "claude-haiku-4-5"
        created = _create_dev_agent(
            bedrock_url,
            bedrock_token,
            product_id=product_id,
            model=model,
            system_prompt=args.system_prompt,
            template=args.template,
            branch=_git_branch(),
            sha=_git_sha(),
        )
        agent_id = created["id"]
        print(f"created dev agent: {agent_id}", file=sys.stderr)
        # Fetch the harness-config view so we get the product's default
        # adapters wired into this agent (same shape as existing-id path).
        cfg = _fetch_config(bedrock_url, bedrock_token, agent_id)
        config = _build_config_from_harness_config(
            cfg,
            model_override=args.model,
            reasoning_override=args.reasoning_effort,
        )

    logger.info(
        "built agent config: id=%s model=%s adapters=%d",
        config.id, config.model, len(config.adapters),
    )
    for adapter in config.adapters:
        tool_names = [getattr(t, "name", "<unnamed>") for t in adapter.tools]
        logger.info(
            "built agent config: adapter=%r has %d tool(s): %s",
            adapter.name, len(adapter.tools), tool_names,
        )
    total_tools = sum(len(a.tools) for a in config.adapters)
    logger.info(
        "built agent config: %d total tool(s) across %d adapter(s) "
        "(built-ins added separately)",
        total_tools, len(config.adapters),
    )

    from harness import Harness

    run_id = args.run_id or str(uuid.uuid4())
    logger.info("starting harness run agent=%s run=%s", agent_id, run_id)
    try:
        Harness(config, run_id=run_id).run()
    except KeyboardInterrupt:
        logger.warning("harness run interrupted agent=%s run=%s", agent_id, run_id)
        return 130
    except Exception:
        logger.exception("harness run failed agent=%s run=%s", agent_id, run_id)
        raise
    logger.info("harness run complete agent=%s run=%s", agent_id, run_id)
    return 0


def _cmd_delete_agent(args, parser: argparse.ArgumentParser) -> int:
    _load_env()

    from harness.core import storage

    try:
        result = storage.delete_agent_storage(
            args.agent_id,
            require_remote=bool(os.environ.get("DAYTONA_API_KEY")),
            purge=args.purge,
        )
    except RuntimeError as e:
        parser.exit(1, f"delete agent failed: {e}\n")

    mode = "purged" if args.purge else "archived"
    logger.info(
        "deleted agent storage (%s): id=%s local=%s remote=%s",
        mode, args.agent_id, result["local"], result["remote"],
    )
    print(
        f"deleted agent storage ({mode}): "
        f"id={args.agent_id} local={result['local']} remote={result['remote']}",
        file=sys.stderr,
    )
    return 0


def _cmd_reset_memory(args, parser: argparse.ArgumentParser) -> int:
    _load_env()

    from harness.core import storage

    try:
        result = storage.reset_agent_memory(
            args.agent_id,
            require_remote=bool(os.environ.get("DAYTONA_API_KEY")),
        )
    except RuntimeError as e:
        parser.exit(1, f"reset memory failed: {e}\n")

    logger.info(
        "reset agent memory: id=%s local=%s remote=%s",
        args.agent_id, result["local"], result["remote"],
    )
    print(
        "reset agent memory: "
        f"id={args.agent_id} local={result['local']} remote={result['remote']}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# Argparse + entrypoint
# ---------------------------------------------------------------------------


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bedrock-url", default=None,
                   help="Override $BEDROCK_URL (CLI wins).")
    p.add_argument("--local", action="store_true",
                   help="Sugar for --bedrock-url http://127.0.0.1:8000.")
    p.add_argument("--bedrock-token", default=None,
                   help="Bedrock product API key. Defaults to $BEDROCK_TOKEN.")
    p.add_argument("--log-level",
                   default=os.environ.get("LOG_LEVEL", "INFO"),
                   help="Log level: DEBUG|INFO|WARNING|ERROR.")
    p.add_argument("--model", default=None, help="Override the model.")
    p.add_argument("--reasoning-effort", default=None,
                   help="Override reasoning effort.")
    p.add_argument("--template", default=None,
                   help="Template uuid-or-name (forward-compat; Phase 2).")
    p.add_argument("--product", default=None,
                   help="Product UUID for auto-created agents. If omitted the "
                        "single product visible to the API key is used.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Run a harness agent run or a scenario eval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    agent_p = subparsers.add_parser("agent", help="Run a single agent run.")
    agent_p.add_argument("agent_id", nargs="?", default=os.environ.get("AGENT_ID"),
                         help="Agent UUID. Omit to auto-create a dev agent.")
    agent_p.add_argument("--run-id", default=None,
                         help="Run UUID. Defaults to a fresh uuid4.")
    agent_p.add_argument("--system-prompt", default=None,
                         help="System prompt override (dev auto-create only).")
    _add_common_flags(agent_p)

    delete_p = subparsers.add_parser(
        "delete-agent",
        help="Archive (default) or purge harness-owned storage for an agent.",
    )
    delete_p.add_argument("agent_id", help="Agent UUID to delete storage for.")
    delete_p.add_argument(
        "--purge",
        action="store_true",
        help="Fully delete the Daytona sandbox (no recovery). Default is to "
             "archive, which moves state to cheap object storage and lets the "
             "next `harness agent` resurrect it.",
    )
    delete_p.add_argument("--log-level",
                          default=os.environ.get("LOG_LEVEL", "INFO"),
                          help="Log level: DEBUG|INFO|WARNING|ERROR.")

    reset_p = subparsers.add_parser(
        "reset-memory",
        help="Reset memory storage for an agent.",
    )
    reset_p.add_argument("agent_id", help="Agent UUID to reset memory for.")
    reset_p.add_argument("--log-level",
                         default=os.environ.get("LOG_LEVEL", "INFO"),
                         help="Log level: DEBUG|INFO|WARNING|ERROR.")

    eval_p = subparsers.add_parser("eval", help="Run a scenario eval end-to-end.")
    eval_p.add_argument("scenario",
                        help="Scenario name (matches Simulation.name).")
    _add_common_flags(eval_p)

    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    _install_shutdown_handlers()

    if args.command == "agent":
        return _cmd_agent(args, agent_p)
    if args.command == "delete-agent":
        return _cmd_delete_agent(args, delete_p)
    if args.command == "reset-memory":
        return _cmd_reset_memory(args, reset_p)
    if args.command == "eval":
        # Lazy import — `harness.evals` must not be pulled on the agent path.
        from harness.evals.cli_entry import run as _cmd_eval

        return _cmd_eval(args, eval_p)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
