# pde-sim binding for petscagent-bench

Exposes the pde-sim pipeline as the **Purple agent** (agent-under-test) in
[petscagent-bench](https://github.com/petsc/petscagent-bench), so pde-sim can be
scored by the same gates/metrics/quality evaluators as any other code-generation
agent — **without modifying any bench source**.

## How it works

The bench Green agent is decoupled from the Purple *implementation*: it reads the
purple **URL** from its task message and talks A2A/HTTP only. This binding is a
standalone A2A server implementing the Purple contract:

1. Green sends a plain-text `problem_description`.
2. `pdesim_purple.py` drives pde-sim non-interactively via `claude -p` in
   **autonomous** mode, writing artifacts under a per-request study dir.
3. It reads the pipeline's `results-manifest.json`, then returns:
   - a `TextPart`: `Code generation successful` / `nsize:` / `cli_args:`
     (`nsize` = the chosen run's `mpi_ranks`; `cli_args` = its `command` with the
     `mpiexec -n N ./exe` prefix stripped),
   - a `FilePart` per generated source (main `.c` first, plus `.cu` /
     `*.kokkos.cxx` dependencies). The makefile is intentionally **not** sent —
     the bench's MCP server builds its own.
4. Green uploads the source via MCP, compiles, runs with `nsize`/`cli_args`, and
   scores it.

## Prerequisites

- The `claude` CLI on `PATH` (or set `PDESIM_CLAUDE_BIN`), logged in, able to run
  the `/pde-sim` pipeline in this repo.
- A working PETSc build reachable via `PETSC_DIR` / `PETSC_ARCH` in the
  environment (pde-sim compiles/verifies; the bench independently recompiles).
- The A2A server deps (`a2a-sdk[http-server]`, `uvicorn`). Simplest is to run
  this file from petscagent-bench's own `uv` env.

## Run it (no bench code changes)

Use the bench's "separate components" path — **not** `main.py launch`, which
hardcodes the built-in purple.

Export the needed environment first (the MCP server and the pde-sim pipeline need
`PETSC_DIR`/`PETSC_ARCH`; the Green judge needs its LLM key — `ARGO_API_KEY` for
Argo). `uv run` does not read the bench's `.env`, so export them in the shells:

```bash
export PETSC_DIR=/path/to/petsc PETSC_ARCH=your-arch ARGO_API_KEY=your_anl_username

# 1) Green agent (from the bench repo/venv) — :9001
cd ~/petscagent-bench && uv run src/green_agent/server.py &

# 2) PETSc compile/run MCP server — :8080
#    petsc_mcp_servers ships as a bench dependency (petsc-ai-servers-clients),
#    so its module is importable from the bench env. main() defaults to
#    streamable-http on :8080 with the /mcp endpoint:
cd ~/petscagent-bench && \
  uv run python -c "from petsc_compile_run_mcp_server import main; main()" &

# 3) This binding as the Purple agent — :9002
#    Run from the bench env so a2a-sdk/uvicorn are available:
cd ~/petscagent-bench && \
  uv run python ~/petsc/.agents/pde-sim/bindings/petscagent-bench/pdesim_purple.py &

# 4) Trigger the run, pointing Green at this binding's URL
cd ~/petscagent-bench && uv run src/client_cli.py \
  --green-url http://localhost:9001 \
  --purple-url http://localhost:9002 \
  --mcp-server-url http://localhost:8080/mcp
```

Results land in the bench's `output/` as usual. The bench's
`config/purple_agent_config.yaml` is **unused** here (it configures the built-in
purple's LLM); the Green config still applies.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PDESIM_CLAUDE_BIN` | `claude` | Claude CLI binary |
| `PDESIM_REPO_DIR` | repo root (inferred) | cwd for the pipeline |
| `PDESIM_ARTIFACTS_DIR` | `<repo>/artifacts` | where studies/manifests are searched |
| `PDESIM_TIMEOUT` | `2900` | per-problem seconds. Hard ceiling is the bench's Green→Purple HTTP read timeout, hardcoded `read=3000.0` in `src/util/a2a_comm.py` (no env override), so the default leaves ~100s to kill the CLI, save the transcript and answer. The full config-3 pipeline can still exceed this on the heavier problems; going higher needs that bench-side read timeout raised too. A run killed at this limit is still recorded in the usage sidecar (see below) |
| `CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS` | `1200000` (this binding; CLI default is `600000`) | Claude CLI knob (milliseconds): a Task subagent with no streamed progress for this long is aborted and reported to the orchestrator as a stall. The binding raises the default to 20 min via `setdefault` so a long code-generation step is not reaped mid-run; export your own value to override (e.g. `3600000` for 1 h). There is no documented value that fully disables it — set it large. |
| `PDESIM_PERMISSION_FLAG` | `--permission-mode bypassPermissions` | Claude CLI permission flag |
| `PDESIM_CLAUDE_MODEL` | *(unset)* | optional `--model` override |
| `PDESIM_CLAUDE_EXTRA_ARGS` | *(unset)* | extra CLI args (shell-split) |
| `PDESIM_MAX_NSIZE` | `64` | cap on returned `nsize` (bench rejects larger) |
| `PDESIM_CONFIG` | `3` | ablation rung: `1` general agent (no skills, no subagents), `2` general + skills (no subagents), `3` full pde-sim pipeline |
| `PDESIM_DISALLOWED_TOOLS` | per-config | Claude CLI `--disallowed-tools` list gating the config; default `Task Skill` (1), `Task` (2), empty (3). Override if the installed CLI names these tools differently. |
| `PDESIM_USAGE_DIR` | *(unset)* | if set, append one JSON record per request (tokens, `total_cost_usd`, `num_turns`, subagent counts, `model_name`, and per-stage attempts — see below) to `<dir>/usage-pdesim-<model>-c<config>.jsonl` — named from the self-reported model to match the Green output file; the ablation driver sets it to aggregate generation cost |
| `PDESIM_USAGE_LOG` | *(unset)* | explicit single-file override for the usage record (used when `PDESIM_USAGE_DIR` is not set) |
| `PDESIM_SAVE_TRANSCRIPT` | `1` | (also gates spec companions) save the run's model I/O to the study dir, **live**, and — at the end — a companion `<name>.md` for each math-bearing spec JSON the pipeline wrote (`problem-spec.json`, `numerical-plan.json`, …), rendering its `*_latex`/formula fields as `$$` blocks so the equations are readable in a Markdown-math viewer. Model I/O: The CLI runs with `--output-format stream-json`, so `transcript.stream.jsonl` is appended one event per line as the run happens (`tail -f` it to watch it grow). Readable `.md` files are re-rendered every ~3s and once at the end: each turn timestamped `[HH:MM:SS]` local, each tool result showing the tool and its duration (e.g. `(Bash, 5.4s)`), plus a same-named `.json` sibling with the full, UNtruncated events (look up any `…[truncated]` there), plus `transcript-stderr.log`. Single-agent runs get `transcript.{md,json}`; config 3 splits by role into `transcript-{orchestrator,pde-modeling,numerical-analysis,code-generation,visualization}.{md,json}` (split by `parent_tool_use_id`, roles from the dispatch prompt). Each per-role file grows **live**: it appears the moment its subagent is dispatched and gains one timestamped progress line per tool the subagent runs (from the CLI's `task_*` events, which are the only material available before the subagent's full turns arrive), so a long code-generation step is visible step-by-step and a stall shows as a growing time gap; the coarse progress lines are superseded by the full detailed turns once the subagent completes. Set `0` to disable |

### Per-stage attempts in the usage sidecar

Each `PDESIM_USAGE_DIR` record carries two attempt counts, recorded **binding-side
only** (they are deliberately *not* part of the bench `petscagent.telemetry.v1`
contract, so `_build_telemetry` does not forward them to the Green agent — how the
bench should aggregate attempts is left to the bench maintainers):

- `stage_attempts` — a map of how many times the orchestrator dispatched each
  specialist, keyed by the same `role[-stage]` label as the split transcripts
  (e.g. `{"pde-modeling": 1, "numerical-analysis-plan": 1,
  "numerical-analysis-assess": 1, "code-generation": 1}`). Normally 1 per stage (2
  for numerical-analysis: plan and assess); a value `>1` means the orchestrator
  re-dispatched that stage, e.g. a revise-plan feedback loop. Empty for configs 1–2
  (no subagents).
- `code_generation_iterations` — the length of the code-generation stage's own
  build/run/format fix loop, taken from the manifest's `attempts[]` journal (the
  number of tries that stage needed, distinct from how many times it was
  *dispatched*).

A usage record is written on **every** path — success, timeout, or failure — so a
run's consumption is never silently lost. A run killed at `PDESIM_TIMEOUT` emits
no final `result` event and therefore reports no cost, so its record carries
`"partial": true` and `"total_cost_usd": null`; its token counts are summed from
the individual turns, which is how billing actually accrues, so the run can still
be priced after the fact.

## Ablation configurations (`PDESIM_CONFIG`)

The binding doubles as the agent-under-test for the three-rung ablation in
`EXPERIMENT_PLAN.md` (repo root). One code path, selected by `PDESIM_CONFIG`:

| # | Config | Gating (`--disallowed-tools`) | Prompt |
|---|--------|-------------------------------|--------|
| 1 | General coding agent | `Task` (subagent) + `Skill` blocked | plain "write PETSc code" |
| 2 | Single-context pipeline | `Task` blocked | run the pde-sim workflow itself: read the orchestration `SKILL.md` (steps) + domain skills; no subagents |
| 3 | Specialized + scoped skills | none | full pde-sim pipeline, orchestrator dispatches role agents (default) |

Config 2 reads the **same** pde-sim knowledge as config 3 — the orchestration
`SKILL.md` for the end-to-end workflow plus the domain skills — but performs every
step itself in one context (the dispatch parts of the orchestration skill are inert
without the Agent tool). So **2→3 isolates agent specialization alone**, not the
knowledge available; **1→2 isolates the pde-sim knowledge** (workflow + skills).

All three run at the **repo root**, so they share identical project context
(`CLAUDE.md`/`AGENTS.md`); the only thing that changes across the ladder is the
tool gating. The output/deliverable contract (output discipline, direct run
command, manifest, `PDESIM_DONE` marker) is shared verbatim across all three so
scoring differences reflect the wiring, not prompt wording.

Gating **verified** against Claude CLI 2.1.281: `--disallowed-tools "Task Skill"`
removes both the subagent tool (displayed as `Agent`; `Task` is its disallow alias)
and `Skill`; `--disallowed-tools Task` removes the subagent tool while keeping
`Skill`. Names stay overridable via `PDESIM_DISALLOWED_TOOLS` should a future CLI
rename them.

## Known limitations / things to watch

- **Grader mismatch is the main risk.** The bench's `numerical_accuracy` metric
  reads the *trailing N numeric lines* of stdout (N = length of the problem's
  hidden `expected_output`) and compares them to that reference; only
  `problem_description` is given to the agent — `test_cases` (`args` /
  `expected_output`) are not. The binding's prompt therefore tells pde-sim to
  emit **only** the output the description asks for (e.g. the final solution via
  `VecView()`) and to keep extra numeric diagnostics off stdout. A correct
  solution can still miss the metric if the description's output format is
  ambiguous; remaining cases need per-problem handling in the bench evaluators.
- **Run selection is a heuristic.** The binding sends the first successful run's
  `command`/`mpi_ranks`. If a problem needs a different run, adjust `_pick_run`.
- **cli_args are passed verbatim** after stripping the launcher/exe prefix,
  including MMS-specific flags (e.g. `-mms_levels 4`, `-mms_csv convergence.csv`);
  the latter just writes a file in the bench work dir.
- **A2A SDK / telemetry.** The bench uses the protobuf-based **a2a-sdk 1.x**; this
  binding imports its A2A types and helpers from the bench's own shims
  (`src.util.a2a_v1`, `src.util.telemetry`), so it must be launched from the bench
  repo root (it adds the cwd to `sys.path`). It reports token/cost/effort to the
  bench in a versioned data Part (`petscagent.telemetry.v1`) via `_build_telemetry()`.
- **Cost/time.** A full pde-sim pipeline (multi-agent, MMS studies, retries) uses
  far more tokens/time per problem than a single-shot purple agent. The binding runs
  the CLI with `--output-format json`; the reported token counts are the **top-level
  agent's** (subagent tokens excluded), so the whole-run `cost_usd` /
  `total_cost_usd` is the trustworthy total-cost figure. It is also written to the
  `PDESIM_USAGE_DIR` sidecar and aggregated into the report's `Gen $` column.
- **`bypassPermissions` runs arbitrary Bash** — only run against a sandboxed
  PETSc build.
