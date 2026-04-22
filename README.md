# Harness

A runnable agent harness designed for fast iteration without requiring changes to surrounding infrastructure, tools, or adapters. Harness separates the *agent loop* (prompts, memory, control flow, evals) from the *platform* (deployment, tool implementations, adapters, storage of logs/runs) so each can evolve independently.

Harness is always paired with a **cloud-based infrastructure platform** that owns deployment and durable state. Every run — local or deployed — streams its logs, traces, and eval results to that platform, so local dev and production share one source of truth.

## Goals

- **Fast harness iteration w/o changing infra/tools** — ship new agent logic without touching deploy pipelines or tool implementations.
- **Run evals against any harness version** — pin an eval suite to any commit and reproduce results.
- **Update deployed agents to new versions** — promote a harness commit into an existing deployed agent without rewiring its tools/adapters.
- **Local test + centralized builtins** — run the full harness on your laptop against the same builtins, and have every run show up in the platform alongside deployed runs.

## Architecture

Three layers:

### Builtins

Shared, centralized primitives consumed by every harness version. Stable, versioned independently, owned by the platform. Builtins are the bridge between local runs and the cloud — they're how a `harness run` on a laptop ends up with logs and eval results stored in the same place as a production agent.

- Logging / trace utils (stream to platform)
- LLM routing
- Cost tracking
- OpenRouter integration

### Harness

A single git repo containing the agent loop itself. Every harness version is a commit; the platform pins deployed agents to a specific commit.

- Standard `Runnable` interface
- Single git repo (this one)
- Unified API contract for tools and storage — harness code does not know which concrete tool/adapter it's talking to
- Memory lives inside the harness
- Migration scripts between harness versions
- Extensible evals class with examples

### Platform (cloud infra)

A separate cloud service responsible for deploying agents and persisting everything observable.

- Deploy agents by selecting `{harness commit, model, adapters, tools}`
- Out-of-the-box tools and adapters
- Update an existing deployed agent to a new harness commit
- Run evals against deployed agents
- **Centralized storage of all logs, traces, and eval runs — including runs executed locally**

## Runs: local and deployed

Harness runs flow through the same pipeline regardless of where they execute:

```
[ harness (local laptop OR deployed agent) ]
              │
              ▼
        [ builtins ]  ──▶  logs / traces / eval results
              │                     │
              ▼                     ▼
         [ LLM / tools ]      [ cloud platform ]
```

A run is identified by `{harness commit, model, adapter set, run id}`. The platform is the durable home for all of it, so:

- You can diff a local run against a deployed run.
- Evals run on your laptop are visible to the team immediately.
- Regressions between harness commits are attributable to a specific commit range.

## Contracts

Harness talks to the outside world through two interfaces defined in this repo:

- **Tools** — callable capabilities injected by the platform; harness only sees the contract.
- **Storage / adapters** — persistence and I/O; harness only sees the contract.

Because these are contracts, the platform can swap implementations (e.g. a new vector store, a new search tool) without requiring a harness change — and vice versa.

## Repo Layout (planned)

```
harness/
  src/
    runnable/        # Standard Runnable entrypoint
    memory/          # In-harness memory
    contracts/       # Tool + storage interfaces
    builtins/        # Clients for platform-owned builtins (logging, LLM routing, etc.)
    evals/           # Extensible evals base class + examples
  migrations/        # Version-to-version migration scripts
  examples/
  README.md
```

## Workflows

**Iterate on the agent loop**
1. Branch this repo.
2. Change harness code.
3. Run evals locally — results land in the cloud platform automatically.
4. Compare against a previous harness commit in the platform UI.
5. Merge.

**Deploy a new version to an existing agent**
1. Pick a harness commit.
2. Platform updates the agent to point at that commit.
3. Run migration scripts if needed.
4. Re-run evals; they land next to the previous version's runs.

**Add a new tool or adapter**
1. Implement against the contract defined here.
2. Register it in the platform.
3. No harness change required.
