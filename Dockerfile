# Daytona snapshot image for harness.
#
# Built and published to GHCR by .github/workflows/publish-daytona-snapshot.yml
# on every push to main. The Daytona snapshot named `harness-latest` is then
# pointed at the new sha-tagged image, so consumers can always create a sandbox
# from `harness-latest` and get the most recent harness without changing their
# config.
#
# This is a build-time install (not a runtime download) so cold-start of a new
# sandbox is instant: the venv is already on disk inside the snapshot.

FROM python:3.12-slim

# uv: fast Python package manager. We grab the static binary from the upstream
# image rather than `pip install uv` so we don't need to bootstrap pip first.
COPY --from=ghcr.io/astral-sh/uv:0.5.20 /uv /uvx /usr/local/bin/

# Tools commonly useful inside a sandbox: git for any clone-time fetches,
# sqlite3 CLI for inspecting agent memory, ca-certificates so HTTPS works.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/harness

# Two-step install for better layer caching: dependencies first (changes
# rarely; cached across most commits), then the project source (changes on
# every commit).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY agents/ ./agents/
RUN uv sync --frozen --no-dev

# Put the venv on PATH so `harness ...` works without `uv run` inside the
# sandbox. Daytona's `process.exec(...)` invokes commands via a plain shell.
ENV PATH="/opt/harness/.venv/bin:$PATH"

# `~/.harness` is the default storage root (see src/harness/core/storage.py).
# Pre-creating it means the first `harness agent` doesn't need to mkdir.
RUN mkdir -p /root/.harness/agents

# Daytona requires a long-running entrypoint; the SDK execs commands into the
# already-running container.
CMD ["sleep", "infinity"]
