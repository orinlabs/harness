# Harness

Harness is the local runner for Bedrock agents.

It fetches an agent config from Bedrock, runs the agent loop in this process,
calls models through OpenRouter, calls external tools through Bedrock adapter
URLs, records traces back to Bedrock, and stores memory in libSQL/Turso for the
CLI workflows.

## Setup

Install dependencies:

```bash
uv sync --dev
```

The CLI loads `.env` from the repo root. A typical `.env` looks like this:

```bash
OPENROUTER_API_KEY=...
BEDROCK_TOKEN=...
HARNESS_TURSO_PLATFORM_TOKEN=...
HARNESS_DATABASE_TOKEN=...

# Usually optional
HARNESS_TURSO_ORG=<your-turso-org>
HARNESS_TURSO_GROUP=default
BEDROCK_URL=http://127.0.0.1:8000
```

Notes:

- `BEDROCK_TOKEN` is a Bedrock product API key.
- `BEDROCK_URL` defaults to `http://127.0.0.1:8000`.
- Missing API keys fail loudly. Tests and evals do not silently skip live calls.

## Use With Local Bedrock

Use this when Bedrock is running on your machine.

Run an existing agent:

```bash
uv run harness agent <AGENT_UUID> --local
```

Or create a temporary dev agent and run it:

```bash
uv run harness agent --local --product <PRODUCT_UUID> --system-prompt "You are helpful."
```

If your API key can see exactly one product, you can omit `--product`.

Useful local flags:

```bash
uv run harness agent <AGENT_UUID> --local --model claude-haiku-4-5
uv run harness agent <AGENT_UUID> --local --reasoning-effort medium
uv run harness agent <AGENT_UUID> --local --run-id <RUN_UUID>
```

Spin down an agent:

```bash
uv run harness delete-agent <AGENT_UUID>
```

This deletes all of the agent's side effects in the world. Bedrock calls this
once when deleting an agent.

Reset an agent's memory:

```bash
uv run harness reset-memory <AGENT_UUID>
```

Bedrock calls this when an agent's memory should be wiped. The next run starts
from empty memory.

## Use With Remote Bedrock

Point the CLI at the remote Bedrock URL.

```bash
BEDROCK_URL=https://your-bedrock.example.com uv run harness agent <AGENT_UUID>
```

Or pass it as a flag:

```bash
uv run harness agent <AGENT_UUID> --bedrock-url https://your-bedrock.example.com
```

You can also create a temporary dev agent remotely:

```bash
uv run harness agent \
  --bedrock-url https://your-bedrock.example.com \
  --product <PRODUCT_UUID> \
  --system-prompt "You are helpful."
```

## Run Without Bedrock

For the quickest end-to-end check, run the demo:

```bash
uv run python scripts/run_demo.py
```

The demo starts an in-process fake platform, registers fake weather and SMS
tools, uses local sqlite storage, and makes real OpenRouter calls. It only
needs `OPENROUTER_API_KEY`.

## Run An Eval Locally

Run the eval runner from this repo, usually against local Bedrock:

```bash
uv run harness eval smoke --local
```

Run a larger scenario:

```bash
uv run harness eval group-lunch-memory --local --model claude-haiku-4-5
```

Run the same eval against remote Bedrock:

```bash
uv run harness eval smoke --bedrock-url https://your-bedrock.example.com
```

How evals work:

- The eval simulation and fake adapters run locally in this process.
- The eval still creates a real eval agent in Bedrock.
- Model calls still go through OpenRouter.
- CLI storage uses Turso/libSQL unless you are using the standalone demo path.
- Results print to stdout, and the final line includes the Bedrock agent URL.

Available scenarios:

```text
smoke
group-lunch-fresh
group-lunch-memory
multi-stakeholder-scheduling
preference-channel-fidelity
blue-red-shoes-light
blue-red-shoes-medium
blue-red-shoes-heavy
curl-wget-light
curl-wget-heavy
draft-before-send-heavy
thursday-no-meetings-heavy
vegetarian-restaurant-heavy
vending-bench
```

## Make A Change Locally

Use this loop for normal development:

```bash
git checkout -b <branch-name>
uv sync --dev
# edit code
uv run ruff check .
uv run pytest
uv run harness eval smoke --local
```

For targeted checks:

```bash
uv run pytest tests/test_harness_e2e.py
uv run pytest tests/memory -o "addopts="
RUN_LIVE_TURSO=1 uv run pytest tests/integration -o "addopts="
```

Testing expectations:

- Default tests exclude `tests/memory` and `tests/integration`.
- LLM-related tests use real OpenRouter calls.
- Live Turso tests require `RUN_LIVE_TURSO=1`.
- If an API key is required and missing, the run should fail.

## Common Commands

```bash
# Install
uv sync --dev

# Run the default test suite
uv run pytest

# Run formatting/lint checks
uv run ruff check .

# Run the no-Bedrock demo
uv run python scripts/run_demo.py

# Run an existing local Bedrock agent
uv run harness agent <AGENT_UUID> --local

# Run an existing remote Bedrock agent
uv run harness agent <AGENT_UUID> --bedrock-url https://your-bedrock.example.com

# Run a local eval
uv run harness eval smoke --local
```

## Repo Map

```text
src/harness/cli.py              CLI entrypoint: harness agent / harness eval
src/harness/harness.py          Main agent loop
src/harness/config.py           AgentConfig, AdapterConfig, ExternalToolSpec
src/harness/core/llm.py         OpenRouter client
src/harness/core/storage.py     sqlite or Turso/libSQL storage
src/harness/core/tracer.py      Bedrock tracing
src/harness/tools/              Built-in and external tool wrappers
src/harness/memory/             SQL-backed memory and summaries
src/harness/evals/              Eval runner, fake adapters, scenarios
scripts/run_demo.py             No-Bedrock demo
scripts/run_live.py             Lower-level live-agent helper
tests/                          Unit, e2e, memory, and integration tests
```

## Lower-Level Live Script

Most runs should use `uv run harness agent ...`. If you need the older explicit
script entrypoint, it is still available:

```bash
uv run python scripts/run_live.py <AGENT_UUID> --bedrock-token "$BEDROCK_TOKEN"
```
