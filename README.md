# Harness

Harness is a local agent runtime. Give it a config (model, system prompt,
HTTP-backed tools) and it runs the agent loop in-process: calls models through
OpenRouter, dispatches tool calls, stores memory in per-agent sqlite, and
emits traces.

Harness runs in two modes:

- **Standalone.** Everything in-process; config comes from `./agents/<name>.yaml`.
  Needs `OPENROUTER_API_KEY`. Nothing talks to any cloud besides OpenRouter.
- **Cloud (Bedrock).** Config is fetched from Bedrock; traces + sleep go
  back to Bedrock. Triggered automatically when `BEDROCK_URL` and
  `BEDROCK_TOKEN` are set.

The mode is picked per-run based on what's in the environment; the same
binary handles both.

## Storage

Per-agent sqlite, on the local filesystem. Default location is
`~/.harness/agents/<agent_id>.sqlite`; override with `HARNESS_STORAGE_ROOT`
to point storage somewhere else (the test suite uses this).

When the harness itself runs on durable infrastructure (e.g. a Daytona
sandbox), the sqlite file persists with the host. There's no separate
storage backend to configure.

## Daytona snapshot

Every commit to `main` builds and publishes a Docker image to GHCR and
points the Daytona snapshot named `harness-latest` at the new image. See
[`Dockerfile`](./Dockerfile) for the build; see
[`.github/workflows/publish-daytona-snapshot.yml`](./.github/workflows/publish-daytona-snapshot.yml)
for the publish step.

Consumers create sandboxes with `snapshot="harness-latest"` and always
get the most recent harness — the snapshot name is constant; only the
image SHA behind it changes.

```python
from daytona import Daytona, CreateSandboxFromSnapshotParams

sb = Daytona().create(CreateSandboxFromSnapshotParams(snapshot="harness-latest"))
sb.process.exec("harness agent demo")
```

**Required repo secrets** (Settings → Secrets → Actions):

- `DAYTONA_API_KEY` — used by the publish workflow to delete + recreate
  the snapshot.
- `DAYTONA_API_URL` (optional) — defaults to `https://app.daytona.io/api`.

**One-time Daytona setup:** add a [GHCR registry](https://www.daytona.io/docs/en/snapshots/#github-container-registry-ghcr)
to your Daytona org so the snapshot can pull from
`ghcr.io/orinlabs/harness:sha-...`. Or make the GHCR package public and
skip the registry step.

## Setup

```bash
uv sync --dev
```

A minimal `.env`:

```bash
OPENROUTER_API_KEY=...

# Optional: override where per-agent sqlite files live.
# HARNESS_STORAGE_ROOT=~/.harness/agents

# Optional: enable Bedrock mode. When unset, harness runs standalone.
BEDROCK_URL=http://127.0.0.1:8000
BEDROCK_TOKEN=...
```

Notes:

- Missing API keys fail loudly; Harness does not silently skip live calls.

## Standalone Quickstart

1. Drop an agent config into `./agents/<name>.yaml`.
2. Run it:

```bash
uv run harness agent <name>
```

There's a worked example at `agents/demo.yaml`. Run:

```bash
uv run harness agent demo
```

### Config schema

Flat list of tools (no adapter grouping). Bedrock serves a nested
`adapters: [{name, tools: [...]}]` shape; the Bedrock client flattens that
on ingest so the harness sees the same flat list regardless of source.

```yaml
id: my-agent
model: claude-haiku-4-5
system_prompt: |
  You are helpful.
reasoning_effort: medium        # optional
summarizer_v2: false            # optional

tools:                          # flat list; no adapter grouping
  - name: get_forecast
    description: Five-day forecast for a city.
    parameters:
      type: object
      properties:
        city: { type: string }
      required: [city]
    url: http://localhost:9001/weather/get_forecast
    timeout_seconds: 30
    auth:                              # optional; default kind=none
      kind: bearer_env                 # none | bearer_env | bearer_literal | headers
      token_env: OPENWEATHER_API_KEY
    forward_trace_context: false       # optional; default false
```

### Sleep semantics in standalone

`ctx.runtime.sleep(...)` is a no-op in standalone mode. The harness still
exits cleanly at the end of the current turn (the `SleepTool` sets
`ctx.sleep_requested = True`), but nothing re-spawns the process at
`until`. Use Bedrock mode when you need durable sleep/wake.

### Usage tracking in standalone

Usage totals (tokens, cost, per-model breakdown) are computed per run. They
only persist via the `TraceSink`:

- With Bedrock configured, totals land on the `run_agent` trace metadata.
- Without, they're logged to stderr at end-of-run and not persisted.

## Use With Bedrock

Set `BEDROCK_URL` + `BEDROCK_TOKEN` in the environment (or pass
`--bedrock-url`/`--bedrock-token` on the CLI; `--local` is sugar for
`--bedrock-url http://127.0.0.1:8000`). With those present:

- `harness agent <uuid>` fetches the config from Bedrock (if no local
  `./agents/<uuid>.yaml` shadows it).
- Traces + sleep POST back to Bedrock.

```bash
uv run harness agent <AGENT_UUID>
uv run harness agent <AGENT_UUID> --local
uv run harness agent <AGENT_UUID> --bedrock-url https://your-bedrock.example.com
```

Create a temporary dev agent (Bedrock auto-creates the row):

```bash
uv run harness agent --local --product <PRODUCT_UUID> --system-prompt "You are helpful."
```

If your API key can see exactly one product, `--product` is optional.

Useful flags:

```bash
uv run harness agent <AGENT_UUID> --local --model claude-haiku-4-5
uv run harness agent <AGENT_UUID> --local --reasoning-effort medium
uv run harness agent <AGENT_UUID> --local --run-id <RUN_UUID>
```

### Resolution order for `harness agent <X>`

1. `./agents/<X>.{yaml,yml,json}` (local).
2. Else `GET /api/cloud/agents/<X>/harness-config/` if Bedrock env is set.
3. Else error.

`harness agent` with no id auto-creates a dev agent on Bedrock (requires
env); standalone mode errors because there's nowhere to put a generated
config.

### Reset

```bash
uv run harness reset-memory <AGENT_UUID>
```

Wipes the agent's local sqlite (and WAL/SHM sidecars). The next
`harness agent` recreates it empty and reapplies migrations.

## Evals

```bash
uv run harness eval smoke
uv run harness eval group-lunch-memory --model claude-haiku-4-5
```

Evals run standalone by default. When Bedrock env is set, spans flow to
Bedrock under a synthesized `eval-<scenario>-<hex>` agent_id (no agent row
is created in Bedrock).

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

## Development Loop

```bash
git checkout -b <branch-name>
uv sync --dev
# edit code
uv run ruff check .
uv run pytest
uv run harness eval smoke
```

Targeted:

```bash
uv run pytest tests/test_harness_e2e.py
uv run pytest tests/memory -o "addopts="
```

Testing expectations:

- Default tests exclude `tests/memory`.
- LLM-related tests use real OpenRouter calls.
- Storage runs against per-test tmp dirs via `HARNESS_STORAGE_ROOT`; nothing
  ever writes into `~/.harness/`.
- If an API key is required and missing, the run fails loudly.

## Repo Map

```text
src/harness/cli.py                 CLI entrypoint: harness agent / eval / reset-memory / inspect
src/harness/harness.py             Main agent loop
src/harness/config.py              AgentConfig, ExternalToolSpec, ToolAuth
src/harness/config_loader.py       Load AgentConfig from ./agents/<name>.yaml
src/harness/core/llm.py            OpenRouter client
src/harness/core/storage.py        Per-agent local sqlite (~/.harness/agents/)
src/harness/core/tracer.py         Trace/span context manager; delegates HTTP to TraceSink
src/harness/core/tracing.py        TraceSink protocol + NullTraceSink + InMemoryTraceSink
src/harness/core/runtime.py        AgentRuntime protocol + LocalAgentRuntime
src/harness/cloud/autoconfig.py    Pick TraceSink + AgentRuntime from env
src/harness/cloud/bedrock/         Bedrock integration (trace sink, runtime, config fetch)
src/harness/tools/                 Built-in and external tool wrappers
src/harness/memory/                SQL-backed memory and summaries
src/harness/evals/                 Eval runner, fake adapters, scenarios
agents/                            Local agent configs (YAML)
tests/                             Unit, e2e, and memory tests
```
