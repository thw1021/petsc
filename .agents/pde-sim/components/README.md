# Component library

Reusable, **MMS-verified** building blocks promoted from completed studies. Part
of the Level-1 self-improvement loop (see `docs/DECISIONS.md` D18): the
`code-generation` agent starts from the closest component instead of writing from
scratch, and the orchestrator finds candidates via `components/case-index.json`.

## Rules for this library
- A component enters here **only after it passed its MMS/convergence checks** in a
  study. Unverified code stays in `artifacts/<study-id>/`.
- Each component carries a `component.json` (what it is, problem class, what it
  provides, how it was verified, provenance) and a short `README.md`.
- Changes are guarded by `tests/run_regression.sh` — a component edit is trusted
  only if convergence still holds. Every change is a revertible git commit.

## Contents
| Component | Problem class | Geometry | Verified |
|-----------|---------------|----------|----------|
| `poisson2d-dmda` | elliptic | structured (`DMDA`) | order ≈ 2.00 (L2, L∞) |
