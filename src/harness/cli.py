"""Console entry point for the harness.

Subcommands:

    harness agent [AGENT_ID_OR_NAME] [options]
                                          # Run a single agent run. Resolves
                                          # local ./agents/<name>.yaml first;
                                          # falls back to Bedrock's harness-
                                          # config/ endpoint when the env has
                                          # BEDROCK_URL + BEDROCK_TOKEN set.
                                          # Auto-creates a dev agent on
                                          # Bedrock when no id is provided.
    harness delete-agent AGENT_ID [--purge]
                                          # Archive (default) or fully purge
                                          # agent-owned harness storage.
    harness reset-memory AGENT_ID         # Reset agent memory storage.
    harness eval  SCENARIO   [options]    # Run a scenario eval end-to-end.

Environment is loaded from the first `.env` found by walking up from cwd
(so a per-repo `.env` shadows an org-level `.env` one or more directories
up). Required secrets:
    OPENROUTER_API_KEY

Optional:
    DAYTONA_API_KEY       per-agent Daytona sandbox (falls back to local sqlite)
    DAYTONA_API_URL       default: https://app.daytona.io/api
    DAYTONA_TARGET        SDK default region if unset
    HARNESS_DAYTONA_AUTO_STOP_MINUTES  passed as auto_stop_interval
    MODEL                 override the agent's configured model
    REASONING_EFFORT      override reasoning_effort (low|medium|high)
    LOG_LEVEL             default: INFO
    BEDROCK_URL           enables Bedrock lookup + tracing to Bedrock
    BEDROCK_TOKEN         bedrock org-scoped API key
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

logger = logging.getLogger(__name__)


DEFAULT_ENV: dict[str, str] = {}

REQUIRED_ENV = ("OPENROUTER_API_KEY",)


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
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env load")
        return
    env_path = find_dotenv(usecwd=True)
    if env_path:
        logger.debug("loading env from %s", env_path)
        load_dotenv(env_path)


# ---------------------------------------------------------------------------
# Shared: Bedrock URL/token resolution + env checks + git stamping
# ---------------------------------------------------------------------------


def _resolve_bedrock_url(args) -> str | None:
    """Precedence: ``--bedrock-url`` > ``--local`` > ``$BEDROCK_URL`` > None.

    Returns None when no Bedrock URL is configured anywhere. Callers treat
    that as "standalone mode; don't fall back to Bedrock".
    """
    if getattr(args, "bedrock_url", None):
        return args.bedrock_url
    if getattr(args, "local", False):
        return "http://127.0.0.1:8000"
    return os.environ.get("BEDROCK_URL") or None


def _resolve_bedrock_token(args) -> str | None:
    return getattr(args, "bedrock_token", None) or os.environ.get("BEDROCK_TOKEN")


def _apply_bedrock_env(args) -> tuple[str | None, str | None]:
    """Normalize --bedrock-url/--bedrock-token/--local onto env for downstream.

    Returns ``(url, token)``; either may be None (standalone mode). The
    tracer / Bedrock clients read these off env, so we export whatever we
    resolved onto the process env before any sink or runtime is built.
    """
    url = _resolve_bedrock_url(args)
    token = _resolve_bedrock_token(args)
    if url:
        os.environ["BEDROCK_URL"] = url
    if token:
        os.environ["BEDROCK_TOKEN"] = token
    return url, token


def _bedrock_configured() -> bool:
    return bool(os.environ.get("BEDROCK_URL") and os.environ.get("BEDROCK_TOKEN"))


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
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip()[:40] or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Config resolution for `harness agent`
# ---------------------------------------------------------------------------


def _resolve_agent_config(args, parser: argparse.ArgumentParser):
    """Resolve ``args.agent_id`` into an ``AgentConfig``.

    1. If ``agent_id`` is given and ``./agents/<agent_id>.{yaml,yml,json}``
       exists, load it locally (standalone).
    2. Else if Bedrock env is set, fetch the harness-config from Bedrock.
    3. Else error.

    When ``agent_id`` is None, we auto-create a dev agent on Bedrock
    (requires env). Standalone auto-create would need somewhere to put the
    resulting YAML; unsupported for now.
    """
    from harness.config_loader import load_agent_config_by_name

    agent_id: str | None = args.agent_id

    if agent_id:
        try:
            cfg = load_agent_config_by_name(agent_id)
        except FileNotFoundError as e:
            if not _bedrock_configured():
                parser.exit(
                    1,
                    f"{e}\n(No BEDROCK_URL + BEDROCK_TOKEN in env, so can't "
                    "fall back to the platform.)\n",
                )
            from harness.cloud.bedrock import fetch_harness_config

            cfg = fetch_harness_config(
                agent_id,
                model_override=args.model,
                reasoning_override=args.reasoning_effort,
            )
        else:
            # Apply CLI overrides to the locally-loaded config.
            if args.model or args.reasoning_effort:
                cfg = _apply_runtime_overrides(
                    cfg,
                    model_override=args.model,
                    reasoning_override=args.reasoning_effort,
                )
        return cfg, agent_id

    # No id: auto-create on Bedrock.
    if not _bedrock_configured():
        parser.exit(
            2,
            "agent_id is required in standalone mode. Pass an id that matches "
            "./agents/<name>.yaml, or set BEDROCK_URL + BEDROCK_TOKEN to "
            "auto-create a dev agent on the platform.\n",
        )
    from harness.cloud.bedrock import create_dev_agent, fetch_harness_config, resolve_template

    template_id = resolve_template(args.template)
    model = args.model or "claude-haiku-4-5"
    created = create_dev_agent(
        template_id=template_id,
        model=model,
        system_prompt=args.system_prompt,
        branch=_git_branch(),
        sha=_git_sha(),
    )
    agent_id = created["id"]
    print(f"created dev agent: {agent_id}", file=sys.stderr)
    cfg = fetch_harness_config(
        agent_id,
        model_override=args.model,
        reasoning_override=args.reasoning_effort,
    )
    return cfg, agent_id


def _apply_runtime_overrides(cfg, *, model_override, reasoning_override):
    from dataclasses import replace

    return replace(
        cfg,
        model=model_override or cfg.model,
        reasoning_effort=reasoning_override or cfg.reasoning_effort,
    )


# ---------------------------------------------------------------------------
# Subcommand: agent
# ---------------------------------------------------------------------------


def _cmd_agent(args, parser: argparse.ArgumentParser) -> int:
    _load_env()
    for k, v in DEFAULT_ENV.items():
        os.environ.setdefault(k, v)

    _apply_bedrock_env(args)
    _ensure_secrets_env(parser)

    config, agent_id = _resolve_agent_config(args, parser)

    tool_names = [getattr(t, "name", "<unnamed>") for t in config.tools]
    logger.info(
        "built agent config: id=%s model=%s tools=%d (%s) (built-ins added separately)",
        config.id,
        config.model,
        len(config.tools),
        tool_names,
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
        mode,
        args.agent_id,
        result["local"],
        result["remote"],
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
        args.agent_id,
        result["local"],
        result["remote"],
    )
    print(
        f"reset agent memory: id={args.agent_id} local={result['local']} remote={result['remote']}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# Argparse + entrypoint
# ---------------------------------------------------------------------------


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bedrock-url", default=None, help="Override $BEDROCK_URL (CLI wins).")
    p.add_argument(
        "--local", action="store_true", help="Sugar for --bedrock-url http://127.0.0.1:8000."
    )
    p.add_argument(
        "--bedrock-token",
        default=None,
        help="Bedrock org-scoped API key. Defaults to $BEDROCK_TOKEN.",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Log level: DEBUG|INFO|WARNING|ERROR.",
    )
    p.add_argument("--model", default=None, help="Override the model.")
    p.add_argument("--reasoning-effort", default=None, help="Override reasoning effort.")
    p.add_argument(
        "--template",
        default=None,
        help="Optional AgentTemplate uuid-or-name to base auto-created "
        "agents on. Omit to create a templateless agent (Bedrock "
        "infers organization from $BEDROCK_TOKEN).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Run a harness agent run or a scenario eval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    agent_p = subparsers.add_parser(
        "agent",
        help="Run a single agent run.",
        description=(
            "Run an agent. AGENT_ID_OR_NAME resolves to ./agents/<name>.yaml "
            "first; falls back to Bedrock's harness-config endpoint when "
            "BEDROCK_URL + BEDROCK_TOKEN are set. Omit the arg to auto-create "
            "a dev agent on Bedrock (requires env)."
        ),
    )
    agent_p.add_argument(
        "agent_id",
        nargs="?",
        default=os.environ.get("AGENT_ID"),
        help="Agent UUID or local config name. Omit to auto-create a dev agent on Bedrock.",
    )
    agent_p.add_argument("--run-id", default=None, help="Run UUID. Defaults to a fresh uuid4.")
    agent_p.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt override (Bedrock dev auto-create only).",
    )
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
    delete_p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Log level: DEBUG|INFO|WARNING|ERROR.",
    )

    reset_p = subparsers.add_parser(
        "reset-memory",
        help="Reset memory storage for an agent.",
    )
    reset_p.add_argument("agent_id", help="Agent UUID to reset memory for.")
    reset_p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Log level: DEBUG|INFO|WARNING|ERROR.",
    )

    eval_p = subparsers.add_parser("eval", help="Run a scenario eval end-to-end.")
    eval_p.add_argument("scenario", help="Scenario name (matches Simulation.name).")
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
