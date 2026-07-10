"""Console entry point for the harness.

Subcommands:

    harness boot [AGENT_ID]               # In-sandbox bootstrap. Resolves
                                          # the agent id, run id, and target
                                          # git SHA from CLI flags or
                                          # HARNESS_* env vars; updates the
                                          # local checkout to that SHA, runs
                                          # `uv sync --frozen`, then execs
                                          # into `harness agent ...`. The
                                          # only command bedrock needs to
                                          # know about.
    harness agent [AGENT_ID_OR_NAME] [options]
                                          # Run a single agent run. Resolves
                                          # local ./agents/<name>.yaml first;
                                          # falls back to Bedrock's harness-
                                          # config/ endpoint when the env has
                                          # BEDROCK_URL + BEDROCK_TOKEN set.
                                          # Auto-creates a dev agent on
                                          # Bedrock when no id is provided.
    harness reset-memory AGENT_ID         # Reset agent memory storage.
    harness forget-memory AGENT_ID --minutes N
                                          # Forget memory from the last N
                                          # minutes (raw messages + the
                                          # summary buckets covering them);
                                          # older memory is preserved.
    harness export-memory AGENT_ID        # Print the agent's tiered memory
                                          # summary block to stdout, wrapped
                                          # in sentinel markers so callers
                                          # can extract it from combined
                                          # exec output.
    harness eval  SCENARIO   [options]    # Run a scenario eval end-to-end.
    harness validate-agents [--agents-dir DIR]
                                          # Validate every repo-managed
                                          # agent bundle in an agents dir
                                          # against harness.spec; print
                                          # `<name> <bundle_hash>` per
                                          # bundle. CI gate for the
                                          # agents repo.
    harness render-agent NAME [--agents-dir DIR]
                                          # Render one bundle (resolved
                                          # prompt, documents, sandbox
                                          # files, hash) as JSON.
    harness render-manifest [--agents-dir DIR] [--commit SHA]
                                          # Render the full sync manifest
                                          # (all bundles) that the agents
                                          # repo CI POSTs to Bedrock.

Environment is loaded from the first `.env` found by walking up from cwd
(so a per-repo `.env` shadows an org-level `.env` one or more directories
up). Required secrets:
    OPENROUTER_API_KEY

Optional:
    HARNESS_AGENT_ID      default agent id for `harness boot`
    HARNESS_RUN_ID        default run id for `harness boot`
    HARNESS_COMMIT_SHA    git SHA to check out before booting
    HARNESS_REPO_DIR      checkout path (default: /workspace/harness)
    GITHUB_TOKEN          private-repo fetch auth (used by `boot` only)
    HARNESS_SIM_START_TIME  simulated start time (epoch seconds); all time
                          reads become this moment + elapsed run time
    HARNESS_AGENTS_DIR    directory of repo-managed agent bundles; used by
                          the bundle subcommands and by configs that
                          `extends:` a bundle (defaults to ./agents)
    MODEL                 override the agent's configured model
    REASONING_EFFORT      override reasoning_effort (minimal|low|medium|high|xhigh)
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
from pathlib import Path

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


def _activate_sim_clock(args, parser: argparse.ArgumentParser) -> None:
    """Install the simulated clock offset if a start time was provided.

    Precedence: ``--sim-start-time`` > ``$HARNESS_SIM_START_TIME`` > off.
    Once set, every harness time read (memory timestamps, summarization
    buckets, trace span times, ...) returns the given epoch plus real
    elapsed run time. Runs before any subcommand body so nothing observes
    wall-clock first.
    """
    _load_env()  # idempotent; needed here so a .env-provided fallback is visible
    epoch = getattr(args, "sim_start_time", None)
    if epoch is None:
        raw = os.environ.get("HARNESS_SIM_START_TIME")
        if not raw:
            return
        try:
            epoch = float(raw)
        except ValueError:
            parser.exit(2, f"HARNESS_SIM_START_TIME must be epoch seconds, got {raw!r}\n")

    from harness.core import clock

    clock.set_start_epoch(epoch)
    # Persist onto env so exec'd children (`boot` -> `agent`) inherit the
    # same simulated start even when the flag isn't forwarded.
    os.environ["HARNESS_SIM_START_TIME"] = str(epoch)
    logger.info(
        "simulated clock active: start=%s (epoch %s), offset=%s",
        clock.now().isoformat(),
        epoch,
        clock.get_offset(),
    )


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
                max_tokens_override=args.max_tokens,
            )
        else:
            # Apply CLI overrides to the locally-loaded config.
            if args.model or args.reasoning_effort or args.max_tokens is not None:
                cfg = _apply_runtime_overrides(
                    cfg,
                    model_override=args.model,
                    reasoning_override=args.reasoning_effort,
                    max_tokens_override=args.max_tokens,
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
        max_tokens_override=args.max_tokens,
    )
    return cfg, agent_id


def _apply_runtime_overrides(cfg, *, model_override, reasoning_override, max_tokens_override):
    from dataclasses import replace

    return replace(
        cfg,
        model=model_override or cfg.model,
        reasoning_effort=reasoning_override or cfg.reasoning_effort,
        max_tokens=max_tokens_override if max_tokens_override is not None else cfg.max_tokens,
    )


# ---------------------------------------------------------------------------
# Subcommand: agent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Subcommand: boot
#
# The single-command contract bedrock targets in-sandbox. Bedrock's job is:
#
#   1. find or create a Daytona sandbox labeled `harness.agent_id=<id>`
#   2. set HARNESS_AGENT_ID, HARNESS_RUN_ID, HARNESS_COMMIT_SHA in the env
#      (plus BEDROCK_TOKEN, BEDROCK_URL, OPENROUTER_API_KEY, GITHUB_TOKEN
#      if the harness repo is private)
#   3. process.exec("harness boot")
#
# Everything else (git checkout, uv sync, agent loop) is the harness's
# problem. `boot` exec's into `harness agent ...` once the checkout is
# ready -- that exec is important: any code we just pulled wouldn't be
# loaded by the running interpreter, so the new process is the only way
# to actually use the new code.
# ---------------------------------------------------------------------------


def _git(repo_dir: Path, *args: str) -> str:
    """Run a git command in ``repo_dir`` and return stdout (stripped)."""
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve_repo_dir(repo_dir_arg: str | None) -> Path:
    """Default precedence: --repo-dir > $HARNESS_REPO_DIR > /workspace/harness > cwd.

    /workspace/harness wins when present so an in-sandbox `harness boot`
    "just works" without flags. cwd is the local-debug fallback.
    """
    if repo_dir_arg:
        return Path(repo_dir_arg).resolve()
    if env := os.environ.get("HARNESS_REPO_DIR"):
        return Path(env).resolve()
    if Path("/workspace/harness/.git").is_dir():
        return Path("/workspace/harness")
    return Path.cwd().resolve()


def _sync_repo(repo_dir: Path, target_sha: str | None) -> None:
    """Bring ``repo_dir`` to ``target_sha`` if not already there.

    No-op if ``target_sha`` is None (caller wants HEAD as-is) or if HEAD
    already matches. Auth: if ``$GITHUB_TOKEN`` is set, the fetch URL is
    rewritten to include it for that one fetch only -- the persistent
    ``origin`` URL stays public so a token rotation doesn't strand the
    sandbox with a stale credential baked into ``.git/config``.
    """
    if not target_sha:
        logger.info("boot: no target SHA -- using current HEAD as-is")
        return

    head = _git(repo_dir, "rev-parse", "HEAD")
    if head == target_sha or head.startswith(target_sha):
        logger.info("boot: HEAD already at %s -- skipping fetch", target_sha)
        return

    origin = _git(repo_dir, "remote", "get-url", "origin")
    fetch_url = origin
    token = os.environ.get("GITHUB_TOKEN")
    if token and origin.startswith("https://github.com/") and "@" not in origin:
        fetch_url = origin.replace("https://", f"https://x-access-token:{token}@", 1)

    logger.info("boot: fetching %s into %s", target_sha, repo_dir)
    _git(repo_dir, "fetch", "--depth=1", fetch_url, target_sha)
    _git(repo_dir, "checkout", "--detach", target_sha)


def _sync_deps(repo_dir: Path) -> None:
    """Run ``uv sync --frozen`` in ``repo_dir``. No-op when nothing changed."""
    logger.info("boot: uv sync --frozen (cwd=%s)", repo_dir)
    subprocess.run(
        ["uv", "sync", "--frozen"],
        cwd=str(repo_dir),
        check=True,
    )


def _build_agent_cmd(agent_id: str, run_id: str | None, args) -> list[str]:
    """Construct the argv that ``boot`` will exec into.

    Forwards bedrock/runtime overrides from the boot invocation onto the
    agent invocation so callers don't have to pass the same flag twice
    (once to boot, once to agent). ``--bedrock-url`` and ``--local`` are
    kept distinct: ``--local`` is short for ``--bedrock-url
    http://127.0.0.1:8000`` and shouldn't be forwarded blindly.
    """
    cmd = ["uv", "run", "--frozen", "harness", "agent", agent_id]
    if run_id:
        cmd += ["--run-id", run_id]
    if getattr(args, "bedrock_token", None):
        cmd += ["--bedrock-token", args.bedrock_token]
    if getattr(args, "bedrock_url", None):
        cmd += ["--bedrock-url", args.bedrock_url]
    elif getattr(args, "local", False):
        cmd += ["--local"]
    if getattr(args, "trace_sink", None):
        cmd += ["--trace-sink", args.trace_sink]
    if getattr(args, "model", None):
        cmd += ["--model", args.model]
    if getattr(args, "reasoning_effort", None):
        cmd += ["--reasoning-effort", args.reasoning_effort]
    if getattr(args, "max_tokens", None) is not None:
        cmd += ["--max-tokens", str(args.max_tokens)]
    if getattr(args, "log_level", None):
        cmd += ["--log-level", args.log_level]
    if getattr(args, "sim_start_time", None) is not None:
        cmd += ["--sim-start-time", str(args.sim_start_time)]
    if getattr(args, "max_utc_sleep", None):
        cmd += ["--max-utc-sleep", args.max_utc_sleep]
    return cmd


def _cmd_boot(args, parser: argparse.ArgumentParser) -> int:
    _load_env()

    agent_id = args.agent_id or os.environ.get("HARNESS_AGENT_ID")
    if not agent_id:
        parser.exit(
            2,
            "boot: agent_id required (positional arg or $HARNESS_AGENT_ID)\n",
        )

    run_id = args.run_id or os.environ.get("HARNESS_RUN_ID")
    target_sha = args.commit or os.environ.get("HARNESS_COMMIT_SHA")
    repo_dir = _resolve_repo_dir(args.repo_dir)

    if not (repo_dir / ".git").is_dir():
        parser.exit(
            1,
            f"boot: {repo_dir} is not a git checkout. Set --repo-dir or $HARNESS_REPO_DIR.\n",
        )

    try:
        _sync_repo(repo_dir, target_sha)
        _sync_deps(repo_dir)
    except subprocess.CalledProcessError as e:
        cmd = " ".join(map(str, e.cmd or [])) or "<unknown>"
        stderr = (
            e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(errors="replace")
        )
        parser.exit(
            1,
            f"boot: command failed (exit {e.returncode}): {cmd}\nstderr: {stderr}\n",
        )

    cmd = _build_agent_cmd(agent_id, run_id, args)
    logger.info("boot: exec %s", " ".join(cmd))
    os.chdir(str(repo_dir))
    os.execvp(cmd[0], cmd)
    # execvp does not return; appease the type-checker.
    return 0


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
        Harness(config, run_id=run_id, max_sleep_until=args.max_utc_sleep).run()
    except KeyboardInterrupt:
        logger.warning("harness run interrupted agent=%s run=%s", agent_id, run_id)
        return 130
    except Exception:
        logger.exception("harness run failed agent=%s run=%s", agent_id, run_id)
        raise
    logger.info("harness run complete agent=%s run=%s", agent_id, run_id)
    return 0


def _cmd_reset_memory(args, parser: argparse.ArgumentParser) -> int:
    _load_env()

    from harness.core import storage

    try:
        result = storage.reset_agent_memory(args.agent_id)
    except RuntimeError as e:
        parser.exit(1, f"reset memory failed: {e}\n")

    logger.info(
        "reset agent memory: id=%s local=%s",
        args.agent_id,
        result["local"],
    )
    print(
        f"reset agent memory: id={args.agent_id} local={result['local']}",
        file=sys.stderr,
    )
    return 0


def _cmd_forget_memory(args, parser: argparse.ArgumentParser) -> int:
    _load_env()

    if args.minutes <= 0:
        parser.exit(2, "--minutes must be a positive integer\n")

    from harness.core import storage
    from harness.memory import forget_recent_minutes

    # Open (or create) the agent DB before mutating it. A never-run agent
    # yields an empty schema-applied DB, so the deletes are simply no-ops.
    storage.load(args.agent_id)
    try:
        counts = forget_recent_minutes(args.minutes, timezone_name=args.timezone)
    except (RuntimeError, ValueError) as e:
        parser.exit(1, f"forget memory failed: {e}\n")
    finally:
        storage.flush()

    logger.info(
        "forget agent memory: id=%s minutes=%s deleted=%s",
        args.agent_id,
        args.minutes,
        counts,
    )
    print(
        f"forget agent memory: id={args.agent_id} minutes={args.minutes} deleted={counts}",
        file=sys.stderr,
    )
    return 0


def _cmd_export_memory(args, parser: argparse.ArgumentParser) -> int:
    _load_env()

    from harness.core import storage
    from harness.memory import export_memory_context, wrap_export

    # Open (or create) the agent DB. A never-run agent yields an empty
    # schema-applied DB, so the export is simply an empty block.
    storage.load(args.agent_id)
    try:
        rendered = export_memory_context(
            timezone_name=args.timezone,
            max_tokens=args.max_tokens,
        )
    except (RuntimeError, ValueError) as e:
        parser.exit(1, f"export memory failed: {e}\n")

    # Payload goes to stdout between sentinel markers. Everything else on
    # the stream (log lines, etc.) is noise the consumer discards.
    print(wrap_export(rendered), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Subcommands: bundle tooling (validate-agents / render-agent / render-manifest)
# ---------------------------------------------------------------------------


def _cmd_validate_agents(args, parser: argparse.ArgumentParser) -> int:
    """Validate every bundle in the agents dir; print name + hash per bundle.

    Exit 0 when everything validates, 1 with per-bundle errors otherwise.
    Also runs the secret tripwire scan -- findings are hard failures, the
    whole point of bundles is that their contents are safe to replicate.
    """
    import json as _json

    from harness.bundles import (
        BundleError,
        list_bundle_names,
        load_bundle,
        resolve_agents_dir,
        scan_for_secrets,
    )

    agents_dir = resolve_agents_dir(args.agents_dir)
    if not agents_dir.is_dir():
        parser.exit(1, f"validate-agents: {agents_dir} is not a directory\n")

    names = list_bundle_names(agents_dir)
    if not names:
        parser.exit(1, f"validate-agents: no bundle manifests found in {agents_dir}\n")

    errors: list[str] = []
    results: list[dict] = []
    for name in names:
        try:
            bundle = load_bundle(name, agents_dir)
        except BundleError as e:
            errors.append(str(e))
            continue
        findings = scan_for_secrets(bundle)
        if findings:
            errors.extend(f"{name}: {f}" for f in findings)
            continue
        results.append({"name": name, "bundle_hash": bundle.bundle_hash})
        print(f"{name} {bundle.bundle_hash}")

    if args.json:
        print(_json.dumps({"agents": results, "errors": errors}))
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


def _cmd_render_agent(args, parser: argparse.ArgumentParser) -> int:
    import json as _json

    from harness.bundles import BundleError, load_bundle, render_bundle_payload

    try:
        bundle = load_bundle(args.name, args.agents_dir)
    except BundleError as e:
        parser.exit(1, f"render-agent: {e}\n")
    print(_json.dumps(render_bundle_payload(bundle), indent=2))
    return 0


def _cmd_render_manifest(args, parser: argparse.ArgumentParser) -> int:
    import json as _json

    from harness.bundles import BundleError, render_sync_manifest

    try:
        manifest = render_sync_manifest(args.agents_dir, harness_git_sha=args.commit)
    except BundleError as e:
        parser.exit(1, f"render-manifest: {e}\n")
    print(_json.dumps(manifest))
    return 0


# ---------------------------------------------------------------------------
# Argparse + entrypoint
# ---------------------------------------------------------------------------


def _add_sim_clock_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sim-start-time",
        type=float,
        default=None,
        metavar="EPOCH",
        help="Simulated start time as epoch seconds. Every harness time read "
        "becomes this moment plus real elapsed run time. Falls back to "
        "$HARNESS_SIM_START_TIME; omit both to use wall-clock.",
    )


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    _add_sim_clock_flag(p)
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
    p.add_argument(
        "--trace-sink",
        choices=("bedrock", "stdout", "null"),
        default=None,
        help=(
            "Where trace spans go. Defaults to automatic: bedrock when "
            "BEDROCK_URL + BEDROCK_TOKEN are set, else stdout (machine-"
            "readable @@trace lines). Overrides $HARNESS_TRACE_SINK."
        ),
    )
    p.add_argument("--model", default=None, help="Override the model.")
    p.add_argument(
        "--reasoning-effort",
        default=None,
        help="Override reasoning effort: minimal|low|medium|high|xhigh.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Override completion max_tokens. Anthropic effort-based reasoning "
            "uses this to derive its thinking budget."
        ),
    )
    p.add_argument(
        "--template",
        default=None,
        help="Optional AgentTemplate uuid-or-name to base auto-created "
        "agents on. Omit to create a templateless agent (Bedrock "
        "infers organization from $BEDROCK_TOKEN).",
    )
    p.add_argument(
        "--max-utc-sleep",
        default=None,
        metavar="ISO8601",
        help=(
            "Hard cap on how far into the future the agent may sleep "
            '(e.g. "2026-07-07T06:00:00Z"). Sleep requests past this '
            "moment (including indefinite) are clamped to it. Omit for no cap."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Run a harness agent run or a scenario eval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    boot_p = subparsers.add_parser(
        "boot",
        help="In-sandbox bootstrap: update checkout, uv sync, exec into `harness agent`.",
        description=(
            "The single command bedrock invokes inside a Daytona sandbox. "
            "Bedrock's only job is: (1) find or create the sandbox by "
            "`harness.agent_id=<id>` label, (2) call `process.exec` with "
            "the env contract below set, (3) wait for exit. Everything "
            "else (git checkout, uv sync, agent loop) is the harness's "
            "responsibility.\n\n"
            "Env contract bedrock should set on the exec call:\n"
            "  HARNESS_AGENT_ID    (required) agent UUID to run\n"
            "  HARNESS_RUN_ID      run UUID for tracing/logging\n"
            "  HARNESS_COMMIT_SHA  git SHA to check out before running\n"
            "  HARNESS_REPO_DIR    checkout path, default /workspace/harness\n"
            "  BEDROCK_URL         bedrock endpoint\n"
            "  BEDROCK_TOKEN       bedrock product API key\n"
            "  OPENROUTER_API_KEY  forwarded to the harness loop\n"
            "  GITHUB_TOKEN        only needed if the harness repo is "
            "private (currently public; safe to omit).\n\n"
            "Steps: resolve agent_id/run_id/target SHA from flags or env, "
            "fetch + `git checkout --detach <sha>` if HEAD differs (no-op "
            "otherwise), run `uv sync --frozen` to align deps, then exec "
            "into `uv run --frozen harness agent ...`. The exec is what "
            "ensures any code we just pulled is actually loaded -- the "
            "old interpreter is gone after this point."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    boot_p.add_argument(
        "agent_id",
        nargs="?",
        default=None,
        help="Agent UUID. Falls back to $HARNESS_AGENT_ID.",
    )
    boot_p.add_argument(
        "--commit",
        default=None,
        help="Target git SHA to check out before running. Falls back to "
        "$HARNESS_COMMIT_SHA. If unset, uses the current HEAD as-is "
        "(no fetch).",
    )
    boot_p.add_argument(
        "--run-id",
        default=None,
        help="Run UUID. Falls back to $HARNESS_RUN_ID. If still unset, "
        "the agent subcommand will allocate a fresh uuid4.",
    )
    boot_p.add_argument(
        "--repo-dir",
        default=None,
        help="Path to the harness checkout. Falls back to $HARNESS_REPO_DIR, "
        "then /workspace/harness, then cwd.",
    )
    _add_common_flags(boot_p)

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

    forget_p = subparsers.add_parser(
        "forget-memory",
        help="Forget the last N minutes of an agent's memory.",
        description=(
            "Delete raw messages logged in the last --minutes minutes plus "
            "the tiered-summary buckets covering them. Memory older than the "
            "cutoff is preserved; the summarizer rebuilds the boundary bucket "
            "from any surviving messages on its next run."
        ),
    )
    forget_p.add_argument("agent_id", help="Agent UUID to forget memory for.")
    forget_p.add_argument(
        "--minutes",
        type=int,
        required=True,
        help="How many minutes of recent memory to forget (e.g. 30).",
    )
    forget_p.add_argument(
        "--timezone",
        default=os.environ.get("HARNESS_FORGET_TIMEZONE", "UTC"),
        help="Timezone the summary buckets are keyed in (default: UTC, "
        "matching the summarizer's production default).",
    )
    forget_p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Log level: DEBUG|INFO|WARNING|ERROR.",
    )
    _add_sim_clock_flag(forget_p)

    export_p = subparsers.add_parser(
        "export-memory",
        help="Print an agent's tiered memory summary block to stdout.",
        description=(
            "Render the agent's temporally tiered memory (monthly through "
            "5-minute summaries, no raw messages) and print it to stdout "
            "between sentinel marker lines. Consumers (e.g. Bedrock's voice "
            "webhook) extract the text between the markers."
        ),
    )
    export_p.add_argument("agent_id", help="Agent UUID to export memory for.")
    export_p.add_argument(
        "--timezone",
        default=os.environ.get("HARNESS_TIMEZONE", "UTC"),
        help="Timezone the summary buckets are keyed in (default: UTC).",
    )
    export_p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Cap the rendered block at roughly this many tokens "
            "(tier-aware: finest tiers are dropped first). Omit to keep "
            "the renderer's default."
        ),
    )
    export_p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Log level: DEBUG|INFO|WARNING|ERROR.",
    )
    _add_sim_clock_flag(export_p)

    eval_p = subparsers.add_parser("eval", help="Run a scenario eval end-to-end.")
    eval_p.add_argument("scenario", help="Scenario name (matches Simulation.name).")
    _add_common_flags(eval_p)

    def _add_agents_dir_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--agents-dir",
            default=None,
            help="Directory holding <name>.yaml bundle manifests. Defaults to "
            "$HARNESS_AGENTS_DIR, then ./agents.",
        )
        p.add_argument(
            "--log-level",
            default=os.environ.get("LOG_LEVEL", "INFO"),
            help="Log level: DEBUG|INFO|WARNING|ERROR.",
        )

    validate_p = subparsers.add_parser(
        "validate-agents",
        help="Validate every agent bundle in an agents dir; print name + bundle hash.",
        description=(
            "CI gate for the agents repo. Validates each <name>.yaml against "
            "the harness.spec schema, checks every referenced file exists, "
            "runs a secret tripwire scan, and prints `<name> <bundle_hash>` "
            "per bundle. The bundle hash covers the manifest plus its "
            "resolved file closure, so shared-fragment edits change every "
            "dependent bundle's hash."
        ),
    )
    _add_agents_dir_flag(validate_p)
    validate_p.add_argument(
        "--json",
        action="store_true",
        help="Additionally print a JSON summary line ({agents, errors}).",
    )

    render_agent_p = subparsers.add_parser(
        "render-agent",
        help="Render one bundle (prompt, documents, sandbox files, hash) as JSON.",
    )
    render_agent_p.add_argument("name", help="Bundle name (file stem of <name>.yaml).")
    _add_agents_dir_flag(render_agent_p)

    render_manifest_p = subparsers.add_parser(
        "render-manifest",
        help="Render the full sync manifest the agents-repo CI POSTs to Bedrock.",
    )
    _add_agents_dir_flag(render_manifest_p)
    render_manifest_p.add_argument(
        "--commit",
        default=None,
        help="Git SHA of the agents-repo commit this manifest was rendered from.",
    )

    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    _activate_sim_clock(args, parser)
    _install_shutdown_handlers()

    # Normalize --trace-sink onto env for downstream (autoconfigure reads
    # $HARNESS_TRACE_SINK), same pattern as --bedrock-url -> $BEDROCK_URL.
    if getattr(args, "trace_sink", None):
        os.environ["HARNESS_TRACE_SINK"] = args.trace_sink

    # Validate eagerly so a malformed cap fails the invocation up front rather
    # than mid-run when the agent first tries to sleep.
    if getattr(args, "max_utc_sleep", None):
        from harness.core import clock

        try:
            clock.parse_utc(args.max_utc_sleep)
        except ValueError:
            parser.exit(
                2,
                f"--max-utc-sleep must be an ISO-8601 timestamp "
                f"(e.g. 2026-07-07T06:00:00Z), got {args.max_utc_sleep!r}\n",
            )

    if args.command == "boot":
        return _cmd_boot(args, boot_p)
    if args.command == "agent":
        return _cmd_agent(args, agent_p)
    if args.command == "reset-memory":
        return _cmd_reset_memory(args, reset_p)
    if args.command == "forget-memory":
        return _cmd_forget_memory(args, forget_p)
    if args.command == "export-memory":
        return _cmd_export_memory(args, export_p)
    if args.command == "validate-agents":
        return _cmd_validate_agents(args, validate_p)
    if args.command == "render-agent":
        return _cmd_render_agent(args, render_agent_p)
    if args.command == "render-manifest":
        return _cmd_render_manifest(args, render_manifest_p)
    if args.command == "eval":
        # Lazy import — `harness.evals` must not be pulled on the agent path.
        from harness.evals.cli_entry import run as _cmd_eval

        return _cmd_eval(args, eval_p)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
