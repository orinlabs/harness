# Daytona snapshot image for harness.
#
# Built and published to GHCR by .github/workflows/publish-daytona-snapshot.yml
# on every push to main. The Daytona snapshot named `harness-latest` is then
# pointed at the new sha-tagged image, so consumers can always create a sandbox
# from `harness-latest` and get the most recent harness without changing their
# config.
#
# Layout matches what bedrock expects when spawning an agent run. Bedrock's
# exec command is roughly:
#
#   cd /workspace/harness \
#     && git fetch --depth=1 origin "$HARNESS_COMMIT_SHA" \
#     && git checkout --detach "$HARNESS_COMMIT_SHA" \
#     && uv sync \
#     && uv run harness agent "$HARNESS_AGENT_ID" ...
#
# So this snapshot ships:
#   1. A real git repo at /workspace/harness with `origin` pointing at the
#      public clone URL (auth is bedrock's job at exec time).
#   2. The harness source already committed locally as a "snapshot baseline"
#      so `git checkout --detach <sha>` doesn't trip on untracked files.
#   3. A pre-warmed venv at /workspace/harness/.venv so bedrock's `uv sync`
#      after checkout is a near-no-op for unchanged deps.

FROM python:3.12-slim

# uv: fast Python package manager. Grab the static binary from the upstream
# image rather than `pip install uv` (no pip bootstrap needed).
COPY --from=ghcr.io/astral-sh/uv:0.5.20 /uv /uvx /usr/local/bin/

# Tools commonly useful inside a sandbox: git for the runtime fetch+checkout,
# sqlite3 CLI for inspecting agent memory, ca-certificates so HTTPS works.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/harness

# Two-step install for better layer caching: dependencies first (changes
# rarely; cached across most commits), then the project source (changes on
# every commit).
COPY pyproject.toml uv.lock README.md .gitignore ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY agents/ ./agents/
RUN uv sync --frozen --no-dev

# Initialize a git repo and commit the COPYed source as a single "snapshot
# baseline" commit. This is what makes `git checkout --detach <sha>` work at
# runtime: with a tracked baseline, git can replace the working tree
# in-place without complaining about untracked files. The baseline commit
# itself is throw-away -- bedrock fetches the real commit by SHA and detaches
# onto it. `.venv` is in .gitignore, so it stays out of the baseline (and
# survives the runtime checkout as an untracked, ignored directory).
RUN git init -q . \
    && git config user.email snapshot@harness \
    && git config user.name harness-snapshot \
    && git remote add origin https://github.com/orinlabs/harness.git \
    && git add -A \
    && git commit -q -m "snapshot baseline"

# Put the venv on PATH so `harness ...` works without `uv run` inside the
# sandbox. Daytona's `process.exec(...)` invokes commands via a plain shell.
ENV PATH="/workspace/harness/.venv/bin:$PATH"

# `~/.harness` is the default storage root (see src/harness/core/storage.py).
# Pre-creating it means the first `harness agent` doesn't need to mkdir.
RUN mkdir -p /root/.harness/agents

# Daytona requires a long-running entrypoint; the SDK execs commands into the
# already-running container.
CMD ["sleep", "infinity"]
