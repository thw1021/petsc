# pedsim — a multi-agent system for PETSc-based PDE simulation

A multi-agent system that takes a **text description of a physical phenomenon**
and drives it through modeling → discretization → code generation →
verification → simulation → analysis, producing verified, analyzed, and
visualized numerical solutions.

The design is split into a framework-neutral **core** (contracts, agent role
prompts, domain-knowledge skills, example components, tests) and a thin per-tool
**binding**. Inside PETSc the core lives under `.agents/`: the domain **skills**
in `.agents/skills/` (alongside PETSc's other agent skills), the four agent role
prompts in `.agents/agents/`, and the contracts, components, tests, and docs under
`.agents/pde-pipeline/` (this directory). The only binding built so far is for
**Claude Code**; see [Running under Claude Code](#running-under-claude-code) and
[Portability](#portability).

> Paths written as `contracts/…`, `components/…`, `tests/…`, `docs/…` below are
> relative to this `.agents/pde-pipeline/` directory. Skills and agent role prompts
> are referenced by their repo-root paths under `.agents/skills/` and `.agents/agents/`.

## Architecture at a glance

```
                               user: "simulate X"
                                        │
                                        ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │  ORCHESTRATOR  (main session · skills/orchestration)            │
       │  only it dispatches subagents · every handoff flows through it  │
       └─────────────────────────────────────────────────────────────────┘
                                        │
                        dispatches each agent in pipeline
                        order, routing each output onward
                                        │
                                        ▼
  reads                               agent             produces
  ───────────────────────────────────────────────────────────────────────────────
  phenomenon text          ─▶ [ pde-modeling ]       ─▶ problem-spec.json
  problem-spec.json        ─▶ [ numerical-analysis ] ─▶ numerical-plan.json
  geometry/grid/fields     ─▶ [ visualization ]      ─▶ vis-spec.json
  numerical-plan.json
  + vis-spec in-situ items ─▶ [ code-generation ]    ─▶ sim code incl. IN-SITU viz;
                                                        builds · runs · MMS-verifies;
                                                        results-manifest.json
  results-manifest.json    ─▶ [ numerical-analysis ] ─▶ numerical-assessment.json
  results-manifest.json    ─▶ [ visualization ]      ─▶ generates viz code (post-processors) & RUNS it;
                                                        analysis-report.json

  revision loop: orchestrator routes assessment / report back ─▶ [ numerical-analysis ] or [ pde-modeling ]

  Visualization code has two homes:
    • in-situ   — compiled INTO the simulation binary        → code-generation (per vis-spec in-situ items)
    • post-hoc  — standalone scripts run after the simulation → visualization   (→ analysis-report.json)
```

Three kinds of building block:

| Kind | Location | What it is |
|------|----------|------------|
| **Agents** (roles) | `.agents/agents/*.md` | Subagents with their own context and tools (model inherited from the session). Dispatched by the orchestrator. |
| **Skills** (knowledge) | `.agents/skills/*/SKILL.md` | On-demand expertise, loaded by whichever agent needs it. Two are **shared**. |
| **Contracts** (interfaces) | `contracts/*.schema.json` | JSON Schemas for every inter-agent handoff. The robustness backbone. |

### Why the orchestrator is the *main session*, not a subagent
Only the top-level (main) session can dispatch subagents (subagents cannot spawn
subagents). So the orchestrator is realized as the `orchestration` **skill** that
the main session loads; the four specialists are subagents it dispatches. All
coordination flows through the orchestrator — specialists never call each other.

## The agents

| Agent | Consumes | Produces | Loads skills |
|-------|----------|----------|--------------|
| `pde-modeling` | phenomenon text | `problem-spec.json` | `pde-formulation` |
| `numerical-analysis` | `problem-spec.json`; `results-manifest.json` | `numerical-plan.json`; `numerical-assessment.json` | `numerical-methods`, `petsc-solvers` |
| `code-generation` | `numerical-plan.json` (+ in-situ items of `vis-spec.json`) | simulation code incl. output + `results-manifest.json` | `petsc-codegen`, `petsc-solvers`, `pde-visualization` |
| `visualization` | geometry/grid/fields; `results-manifest.json` | `vis-spec.json`; standalone post-processors it runs; `analysis-report.json` | `pde-visualization` |

`petsc-solvers` and `pde-visualization` are **shared** skills — this is why
skills are named by knowledge domain, not by owning agent.

## The contracts (handoffs)

| Schema | From → To | Purpose |
|--------|-----------|---------|
| `problem-spec.schema.json` | pde-modeling → numerical-analysis | continuous math model (equations, geometry, BC/IC, parameters, scales) |
| `numerical-plan.schema.json` | numerical-analysis → code-generation | grid class, discretization, coefficient functions, solver stack, MMS, verification plan |
| `vis-spec.schema.json` | visualization → code-generation (in-situ items) | visualizations/analytics to produce, each tagged `in_situ` or `post_hoc` |
| `results-manifest.schema.json` | code-generation → orchestrator | inventory of build + runs + output file paths + convergence data |
| `numerical-assessment.schema.json` | numerical-analysis → orchestrator | convergence/conservation verdicts + recommended revisions |
| `analysis-report.schema.json` | visualization → orchestrator | artifacts produced + **factual** diagnostics (no interpretation) |

Large data (solution fields) is **referenced by path**, never inlined. At runtime,
per-study artifacts are written under `artifacts/<study-id>/` (a scratch directory,
git-ignored). A complete worked study (2-D Poisson) is checked in under
`examples/poisson2d/`, and its reusable solver component under
`components/poisson2d-dmda/`.

## Pipeline stages (orchestrator)

1. **Model** — pde-modeling: text → `problem-spec.json`.
2. **Discretize** — numerical-analysis: → `numerical-plan.json` (incl. MMS +
   expected convergence rates). Optionally visualization → `vis-spec.json`.
3. **Generate & verify** — code-generation: write code, build, run MMS/
   convergence study → `results-manifest.json`. Back up a stage on failure.
4. **Run studies** — orchestrator plans a campaign; code-generation executes;
   manifest is fanned out to numerical-analysis (interpret) and visualization
   (render + diagnose).
5. **Deliver** — verified, analyzed, visualized solutions.

Then **escalate**: more physics, parametric sweeps, or design optimization.

## Failure handling

- **Compile/runtime errors** are fixed by `code-generation` in its own
  build→run→fix loop (retry budget ~5; stop on a repeated error).
- **Unfixable-in-place failures** escalate via the Results Manifest's
  `escalation` block (`suspected_cause`, `target_agent`, `summary`, `attempts`);
  the orchestrator routes them to `numerical-analysis` (plan), `pde-modeling`
  (model), or resolves environment issues itself. A run that completes but
  converges wrongly is not an escalation — it flows to `numerical-analysis` for
  assessment.

## Human-in-the-loop

The orchestrator shares the session with the human, so it pauses at decision
gates rather than only at tool-permission prompts. **Default autonomy is
`interactive`** (pause at every gate); `checkpointed` gates only the model +
production runs; `autonomous` runs through and stops only on escalation. Gates:
after the Problem Spec, after the Numerical Plan, before production/large runs,
at results + interpretation, and before higher-fidelity escalation.
`open_questions` and `environment`/`unknown` escalations are always surfaced —
never silently assumed. See `.agents/skills/orchestration/SKILL.md` and
`docs/DECISIONS.md` (D17).

## Reuse & self-improvement (Level 1)

The system accumulates verified knowledge across runs, guarded by the MMS gate:
- **Case index** (`components/case-index.json`) — a registry of solved, verified
  studies (problem class → plan → confirmed order) the orchestrator consults
  before planning from scratch.
- **Component library** (`components/`) — reusable building blocks that passed
  MMS (seeded by the verified 2-D Poisson solver).
- **Regression suite** (`tests/run_regression.sh`) — rebuilds/reruns components
  and re-validates artifacts; a promotion or skill edit is trusted only if this
  still passes. Every change is a revertible git commit, human-gated (D17).

See `docs/DECISIONS.md` (D18). Levels 2–3 (a curator step proposing skill diffs;
eval-driven plan optimization) are deferred.

## Key design decisions (open for team review)

> Full rationale, alternatives, supersessions, and open questions:
> [`docs/DECISIONS.md`](docs/DECISIONS.md).

1. **Compile/run is a tool of `code-generation`** (it holds `Bash`), not a
   separate agent — keeps the edit→compile→fix loop inside one context.
2. **Who runs code?** Orchestrator *decides/dispatches*; code-generation
   *executes* and returns the manifest.
3. **Results go to the orchestrator**, which fans them out — not Code→Vis
   directly.
4. **Visualization reports facts, not interpretation.** Numerical root-cause is
   `numerical-analysis`; model issues are `pde-modeling`.
5. **Code splits by artifact, not by "who codes".** `code-generation` owns code
   inside/linked to the simulation binary (solver, coefficient routines,
   solution output, in-situ rendering). The `visualization` agent owns
   *standalone* post-processors that read the output files (pvpython/matplotlib/
   pyvista) and runs them itself. The shared `pde-visualization` skill is the
   VTK/ParaView knowledge base both use; Vis Spec items are tagged `in_situ` or
   `post_hoc` to route them.
6. **Viz stack = VTK / ParaView + matplotlib** (working assumption — change if
   the target libraries differ).

## Running under Claude Code

PETSc keeps the tool-agnostic core under `.agents/` and **git-ignores `.claude/`**
(see `.gitignore`), so the Claude Code binding is not committed — you create it
locally. Claude Code discovers subagents in `.claude/agents/` and skills in
`.claude/skills/`; point those at the core with symlinks from the repo root:

```
mkdir -p .claude
ln -s ../.agents/agents .claude/agents
ln -s ../.agents/skills .claude/skills
```

Then start Claude Code at the repo root and invoke the orchestrator:

```
/orchestration simulate 2-D steady heat conduction on the unit square with
homogeneous Dirichlet boundaries
```

The orchestrator dispatches the specialists in turn, writing the contract
artifacts under a scratch `artifacts/<study-id>/` directory (add it to your local
`.git/info/exclude`; `.claude/` is already ignored). The `.claude/skills` symlink
also surfaces PETSc's own review/codegraph skills to Claude Code, which is
harmless.

## Repository layout

```
.agents/                        # tracked, tool-agnostic core
├── skills/                     # domain knowledge (loaded on demand)
│   ├── orchestration/          #   pipeline driver (main session)
│   ├── numerical-methods/  petsc-solvers/        # (petsc-solvers shared)
│   ├── pde-formulation/    pde-visualization/    # (pde-visualization shared)
│   └── petsc-codegen/
│                               #   (PETSc's own review-*/codegraph skills also live here)
├── agents/                     # role definitions (subagents), one per specialist
│   ├── pde-modeling.md         numerical-analysis.md
│   └── code-generation.md      visualization.md
└── pde-pipeline/               # framework-neutral core (this directory)
    ├── contracts/    # JSON Schemas for inter-agent handoffs
    ├── components/   # verified, reusable building blocks + case-index.json (reuse registry)
    ├── examples/     # complete worked studies (e.g. examples/poisson2d)
    ├── tests/        # regression suite (guards self-improvement) + schema validator
    ├── docs/         # DECISIONS.md
    └── README.md     # this file

.claude/          # Claude Code binding — created locally, git-ignored (see above)
artifacts/        # per-study runtime output (scratch, regenerated per run)
```

## Portability

The design is framework-neutral; only the **binding** is tool-specific. The neutral
core is `.agents/pde-pipeline/contracts/`, the agent role prompts in
`.agents/agents/`, the `.agents/skills/` knowledge, and the `examples/`,
`components/`, and `tests/` under `.agents/pde-pipeline/` — no hardcoded model, and prose free of tool
assumptions (Skill/Agent-tool mechanics are flagged as binding-specific and fall
back to reading files directly).

The **Claude Code binding** is the local `.claude/` layout described in
[Running under Claude Code](#running-under-claude-code): agents and skills
discovered there (via symlinks into `.agents/`) and the agent frontmatter dialect
(`tools:`). PETSc git-ignores `.claude/`, so the binding is not committed.

To add another tool (e.g. **opencode**) as a second binding, you would add its own
config pointing at the same core — for opencode, `.opencode/agents/*.md` with
`mode`/`permission`/`provider/model-id` frontmatter and a `{file:...}` include of each
role prompt. No second binding is built yet (see `docs/DECISIONS.md` D15).
