"""Console entry point for the harness.

Installed as the `harness` script via `[project.scripts]`. After `uv sync` (or
`uv pip install -e .`), both of these work:

    uv run harness <AGENT_UUID>
    python -m harness <AGENT_UUID>

The CLI fetches the agent's harness-config from the bedrock platform, builds an
`AgentConfig`, and runs a single `Harness` run to completion. Environment is
loaded from `.env` in the current working directory if present.

Non-secret identifiers (platform URL, Turso org/group) have baked-in defaults
below. Only the actual secrets must be provided via env.

Required CLI args:
    --bedrock-token               bedrock product API key (also settable via
                                  $BEDROCK_TOKEN; CLI flag wins)

Required env (secrets):
    HARNESS_TURSO_PLATFORM_TOKEN  Turso platform API token (for DB provisioning)
    HARNESS_DATABASE_TOKEN        Turso group-scoped data token
    OPENROUTER_API_KEY

Optional env (override defaults):
    BEDROCK_URL                   default: http://127.0.0.1:8000
    HARNESS_TURSO_ORG             default: bryanhoulton
    HARNESS_TURSO_GROUP           default: default
    MODEL                         override the agent's configured model
    REASONING_EFFORT              override reasoning_effort (low|medium|high)
    LOG_LEVEL                     default: DEBUG (DEBUG|INFO|WARNING|ERROR)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)


# Non-secret identifiers. Safe to commit; env overrides still win via
# os.environ.setdefault below.
DEFAULT_ENV: dict[str, str] = {
    "BEDROCK_URL": "http://127.0.0.1:8000",
    "HARNESS_TURSO_ORG": "bryanhoulton",
    "HARNESS_TURSO_GROUP": "default",
}

REQUIRED_ENV = (
    "HARNESS_TURSO_PLATFORM_TOKEN",
    "HARNESS_DATABASE_TOKEN",
    "OPENROUTER_API_KEY",
)


def _configure_logging(level: str) -> None:
    """Configure root logging for the CLI process.

    Writes to stdout (not stderr) so the live harness trace — turns, LLM calls,
    tool calls — interleaves with ordinary process output and is easy to pipe
    into `tee` / `grep`. Uses `force=True` so this wins even if some transitive
    import (e.g. httpx, libsql) called `basicConfig` first.
    """
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

    # Third-party libraries are chatty at DEBUG; keep them at INFO unless the
    # user explicitly asked for DEBUG on the root.
    if numeric > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Ensure stdout is line-buffered so each log record flushes immediately —
    # critical when the process hangs mid-LLM-call and we need to see the last
    # log that made it out.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _load_env() -> None:
    """Load .env from cwd if `python-dotenv` is available. Silent no-op otherwise.

    Kept optional so the CLI still works in production subprocesses (bedrock
    spawns us with env already populated and no dotenv installed).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env load")
        return
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        logger.debug("loading env from %s", env_path)
        load_dotenv(env_path)


def _fetch_config(agent_id: str) -> dict:
    base = os.environ["BEDROCK_URL"].rstrip("/")
    token = os.environ["BEDROCK_TOKEN"]
    url = f"{base}/api/cloud/agents/{agent_id}/harness-config/"
    logger.info("fetching harness config from %s", url)
    resp = httpx.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()


def _build_config(cfg: dict) -> "AgentConfig":  # noqa: F821
    from harness import AdapterConfig, AgentConfig, ExternalToolSpec

    model = os.environ.get("MODEL") or cfg["model"]
    reasoning_effort = os.environ.get("REASONING_EFFORT") or cfg.get("reasoning_effort")

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Run a harness against a live bedrock agent.",
    )
    parser.add_argument(
        "agent_id",
        nargs="?",
        default=os.environ.get("AGENT_ID"),
        help="Agent UUID. Defaults to $AGENT_ID.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run UUID. Defaults to a fresh uuid4.",
    )
    parser.add_argument(
        "--bedrock-token",
        default=None,
        help="Bedrock product API key. Defaults to $BEDROCK_TOKEN.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "DEBUG"),
        help="Log level: DEBUG|INFO|WARNING|ERROR. Defaults to $LOG_LEVEL or DEBUG.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    if not args.agent_id:
        parser.error("agent_id is required (pass as arg or set $AGENT_ID)")

    _load_env()

    for k, v in DEFAULT_ENV.items():
        os.environ.setdefault(k, v)

    bedrock_token = args.bedrock_token or os.environ.get("BEDROCK_TOKEN")
    if not bedrock_token:
        parser.error("--bedrock-token is required (pass as flag or set $BEDROCK_TOKEN)")
    # Propagate so downstream modules (tracer, runtime_api, external tools)
    # can pick it up via the usual os.environ lookup.
    os.environ["BEDROCK_TOKEN"] = bedrock_token

    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        logger.error("missing required env: %s", ", ".join(missing))
        print(f"missing required env: {', '.join(missing)}", file=sys.stderr)
        return 2

    cfg = _fetch_config(args.agent_id)
    config = _build_config(cfg)
    logger.info(
        "built agent config: id=%s model=%s adapters=%d",
        config.id,
        config.model,
        len(config.adapters),
    )

    from harness import Harness

    run_id = args.run_id or str(uuid.uuid4())
    logger.info("starting harness run agent=%s run=%s", args.agent_id, run_id)
    try:
        Harness(config, run_id=run_id).run()
    except Exception:
        logger.exception("harness run failed agent=%s run=%s", args.agent_id, run_id)
        raise
    logger.info("harness run complete agent=%s run=%s", args.agent_id, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
