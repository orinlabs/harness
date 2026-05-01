"""Tests for the `harness boot` subcommand.

The production path is: bedrock starts a sandbox, sets HARNESS_* env vars
+ optionally GITHUB_TOKEN, runs `harness boot`. `boot` then drives real
git + real `uv sync` + real `os.execvp` into `harness agent`.

These tests run against real `git` (not a mock) so they catch the kinds
of bugs that only surface when the harness's helper actually shells out:
wrong cwd, wrong remote URL handling, HEAD comparison off-by-one, etc.
We can't realistically run `uv sync` here (it'd resolve over the network
and install into a real venv), and we can't run `os.execvp` (it'd
replace the test process), so those parts are tested separately:
``_build_agent_cmd`` is unit-tested for shape, and the integration
between the helpers + the subcommand is exercised via the helper-level
tests below.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def real_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A real git repo in a tmp dir with two commits and an `origin` remote.

    The remote URL is local (``file://...``) so ``git fetch`` works
    without network access; this exactly mirrors what ``boot`` does
    against ``https://github.com/orinlabs/harness.git`` in production.

    Returns ``(work_dir, sha1, sha2)`` -- HEAD is detached at ``sha1``,
    and ``sha2`` is published on the upstream's ``main`` ref so tests
    can ``fetch + checkout --detach sha2`` to advance.
    """
    upstream = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "--bare", "-q", str(upstream)], check=True)

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q")
    _git(work, "config", "user.email", "test@harness")
    _git(work, "config", "user.name", "Test Harness")

    (work / "README").write_text("commit 1\n")
    _git(work, "add", "README")
    _git(work, "commit", "-q", "-m", "first")
    sha1 = _git(work, "rev-parse", "HEAD")

    (work / "README").write_text("commit 2\n")
    _git(work, "add", "README")
    _git(work, "commit", "-q", "-m", "second")
    sha2 = _git(work, "rev-parse", "HEAD")

    _git(work, "remote", "add", "origin", f"file://{upstream}")
    _git(work, "push", "-q", "origin", "HEAD:refs/heads/main")

    # Detach the working repo to sha1 so tests can fetch sha2 forward.
    _git(work, "checkout", "-q", "--detach", sha1)

    return work, sha1, sha2


def _git(repo_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_sync_repo_noop_when_target_sha_unset(real_repo):
    """No --commit / no $HARNESS_COMMIT_SHA: ``boot`` runs against the
    sandbox's current HEAD (whatever that happens to be, e.g. the
    snapshot-baked version). _sync_repo must not fetch."""
    from harness.cli import _sync_repo

    work, sha1, _ = real_repo
    _sync_repo(work, target_sha=None)
    assert _git(work, "rev-parse", "HEAD") == sha1


def test_sync_repo_noop_when_head_already_at_target(real_repo):
    """If HEAD already matches the target, ``boot`` skips the fetch entirely.
    This is the warm-restart path: bedrock issues the same SHA twice in a
    row (e.g. retry), and we don't want a wasteful network round-trip."""
    from harness.cli import _sync_repo

    work, sha1, _ = real_repo
    _sync_repo(work, target_sha=sha1)
    assert _git(work, "rev-parse", "HEAD") == sha1


def test_sync_repo_fetches_and_detaches_to_target_sha(real_repo):
    """The advance path: HEAD is at sha1, target is sha2, we fetch + detach.

    This is the main bedrock production behavior: a freshly-spawned
    sandbox is at the snapshot-baked commit, bedrock asks for HEAD-of-
    main, _sync_repo brings the working tree forward.
    """
    from harness.cli import _sync_repo

    work, sha1, sha2 = real_repo
    assert sha1 != sha2  # sanity: fixture set up a real advance

    _sync_repo(work, target_sha=sha2)

    assert _git(work, "rev-parse", "HEAD") == sha2

    # Detached HEAD: no branch should be checked out.
    branch = subprocess.run(
        ["git", "-C", str(work), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
        text=True,
    )
    assert branch.returncode != 0, "expected detached HEAD after checkout --detach"

    # The README must reflect the new commit's contents.
    assert (work / "README").read_text() == "commit 2\n"


def test_sync_repo_keeps_origin_url_clean_even_with_github_token_set(real_repo, monkeypatch):
    """The harness repo is public, so $GITHUB_TOKEN won't usually be set
    in production -- but if it ever is (e.g. the repo goes private
    later, or someone leaks it via env), ``_sync_repo`` must NOT persist
    it into ``.git/config``. The post-condition is: ``origin`` URL after
    sync is byte-identical to the URL before sync, regardless of whether
    a token was in env. Otherwise a rotation would strand the sandbox
    with a stale credential and we'd have to manually scrub it."""
    from harness.cli import _sync_repo

    work, _, sha2 = real_repo
    origin_before = _git(work, "remote", "get-url", "origin")

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_unit_test_fake_token")
    _sync_repo(work, target_sha=sha2)

    assert _git(work, "remote", "get-url", "origin") == origin_before
    assert _git(work, "rev-parse", "HEAD") == sha2


def test_resolve_repo_dir_precedence(tmp_path, monkeypatch):
    """--repo-dir > $HARNESS_REPO_DIR > /workspace/harness > cwd."""
    from harness.cli import _resolve_repo_dir

    monkeypatch.delenv("HARNESS_REPO_DIR", raising=False)

    # 1. Explicit flag wins.
    assert _resolve_repo_dir(str(tmp_path)) == tmp_path.resolve()

    # 2. Env var is the fallback.
    env_dir = tmp_path / "envdir"
    env_dir.mkdir()
    monkeypatch.setenv("HARNESS_REPO_DIR", str(env_dir))
    assert _resolve_repo_dir(None) == env_dir.resolve()

    # 3. With no flag and no env: cwd (assuming /workspace/harness doesn't
    #    exist on the test host, which is the dev-laptop case).
    monkeypatch.delenv("HARNESS_REPO_DIR")
    monkeypatch.chdir(tmp_path)
    if not Path("/workspace/harness/.git").is_dir():
        assert _resolve_repo_dir(None) == tmp_path.resolve()


def test_build_agent_cmd_minimal():
    """Just an agent_id, no run_id, no overrides: build the smallest valid
    invocation. ``uv run --frozen`` ensures we don't accidentally
    re-resolve deps after `boot` already ran `uv sync --frozen`."""
    from harness.cli import _build_agent_cmd

    args = argparse.Namespace(
        bedrock_token=None,
        bedrock_url=None,
        local=False,
        model=None,
        reasoning_effort=None,
        log_level=None,
    )
    cmd = _build_agent_cmd("agent-xyz", run_id=None, args=args)

    assert cmd == ["uv", "run", "--frozen", "harness", "agent", "agent-xyz"]


def test_build_agent_cmd_forwards_all_flags():
    """All boot-time overrides flow through to the agent invocation, in a
    stable order so the resulting argv is greppable in logs."""
    from harness.cli import _build_agent_cmd

    args = argparse.Namespace(
        bedrock_token="bt-123",
        bedrock_url="http://b.example.com",
        local=False,
        model="claude-haiku-4-5",
        reasoning_effort="medium",
        log_level="DEBUG",
    )
    cmd = _build_agent_cmd("agent-xyz", run_id="run-abc", args=args)

    assert cmd == [
        "uv",
        "run",
        "--frozen",
        "harness",
        "agent",
        "agent-xyz",
        "--run-id",
        "run-abc",
        "--bedrock-token",
        "bt-123",
        "--bedrock-url",
        "http://b.example.com",
        "--model",
        "claude-haiku-4-5",
        "--reasoning-effort",
        "medium",
        "--log-level",
        "DEBUG",
    ]


def test_build_agent_cmd_local_sugar_not_combined_with_bedrock_url():
    """``--local`` is shorthand for ``--bedrock-url 127.0.0.1:8000``. If
    the boot caller passed both, --bedrock-url wins (explicit > sugar)
    and --local is dropped, otherwise we'd send conflicting flags."""
    from harness.cli import _build_agent_cmd

    args = argparse.Namespace(
        bedrock_token=None,
        bedrock_url="https://b.example.com",
        local=True,  # ignored when bedrock_url is set
        model=None,
        reasoning_effort=None,
        log_level=None,
    )
    cmd = _build_agent_cmd("agent-1", run_id=None, args=args)

    assert "--local" not in cmd
    assert "--bedrock-url" in cmd
    assert cmd[cmd.index("--bedrock-url") + 1] == "https://b.example.com"

    # And vice-versa: --local alone forwards as --local.
    args.bedrock_url = None
    cmd2 = _build_agent_cmd("agent-1", run_id=None, args=args)
    assert "--local" in cmd2


def test_cmd_boot_errors_when_agent_id_missing(monkeypatch, tmp_path):
    """The contract is `harness boot` (no flags) when env is set, or
    `harness boot AGENT_ID` for explicit. With neither, exit 2 with a
    pointer at the env var."""
    monkeypatch.delenv("HARNESS_AGENT_ID", raising=False)
    monkeypatch.delenv("HARNESS_RUN_ID", raising=False)
    monkeypatch.delenv("HARNESS_COMMIT_SHA", raising=False)
    monkeypatch.setenv("HARNESS_REPO_DIR", str(tmp_path))

    from harness.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["boot"])
    assert excinfo.value.code == 2


def test_cmd_boot_errors_when_repo_dir_is_not_a_git_checkout(monkeypatch, tmp_path, capsys):
    """If --repo-dir / $HARNESS_REPO_DIR points at a non-git directory,
    fail loudly before doing anything destructive (no git fetch on a
    random folder, no `uv sync` in the wrong place)."""
    not_a_repo = tmp_path / "blank"
    not_a_repo.mkdir()
    monkeypatch.setenv("HARNESS_AGENT_ID", "agent-test")
    monkeypatch.setenv("HARNESS_REPO_DIR", str(not_a_repo))

    from harness.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["boot", "--log-level", "ERROR"])
    assert excinfo.value.code == 1

    err = capsys.readouterr().err
    assert "not a git checkout" in err


@pytest.mark.skipif(
    sys.platform == "win32", reason="execvp + os.chdir + uv subprocess; unix-only path"
)
def test_cmd_boot_execs_into_agent_when_everything_is_in_place(real_repo, monkeypatch, tmp_path):
    """End-to-end: a real git repo, sync_repo advances HEAD, sync_deps is
    short-circuited to a known invocation, and execvp is captured via a
    fake. The point of this test is to assert ``boot`` does the steps
    in the right order: sync git -> sync deps -> exec agent (cwd=repo).

    We replace `_sync_deps` and `os.execvp` with thin spies because
    actually running `uv sync` would need a real pyproject.toml and a
    real network, and `os.execvp` would replace the test process. The
    spies preserve the production call shape, just don't do the work.
    """
    work, sha1, sha2 = real_repo
    assert sha1 != sha2

    monkeypatch.setenv("HARNESS_AGENT_ID", "agent-test")
    monkeypatch.setenv("HARNESS_RUN_ID", "run-test")
    monkeypatch.setenv("HARNESS_COMMIT_SHA", sha2)
    monkeypatch.setenv("HARNESS_REPO_DIR", str(work))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # boot doesn't need it

    from harness import cli

    sync_deps_calls: list[Path] = []
    monkeypatch.setattr(cli, "_sync_deps", lambda d: sync_deps_calls.append(d))

    exec_calls: list[tuple[str, list[str]]] = []

    def fake_execvp(file: str, argv: list[str]) -> None:
        exec_calls.append((file, list(argv)))
        raise SystemExit(0)  # mimic execvp's "this function does not return"

    monkeypatch.setattr(cli.os, "execvp", fake_execvp)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["boot", "--log-level", "ERROR"])
    assert excinfo.value.code == 0

    # 1. The git checkout advanced.
    assert _git(work, "rev-parse", "HEAD") == sha2

    # 2. _sync_deps was called once, with the resolved repo dir.
    assert sync_deps_calls == [work.resolve()] or sync_deps_calls == [work]

    # 3. execvp was called exactly once, into uv run harness agent.
    assert len(exec_calls) == 1
    file, argv = exec_calls[0]
    assert file == "uv"
    assert argv[:5] == ["uv", "run", "--frozen", "harness", "agent"]
    assert "agent-test" in argv
    assert "--run-id" in argv and argv[argv.index("--run-id") + 1] == "run-test"
