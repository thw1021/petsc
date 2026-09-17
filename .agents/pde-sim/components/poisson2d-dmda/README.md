# poisson2d-dmda

Verified 2-D Poisson solver on a structured `DMDA`, with an MMS convergence
harness. See `component.json` for metadata and provenance.

## Build & run
Requires `PETSC_DIR` / `PETSC_ARCH` set and `mpiexec` available.

```bash
make poisson
mpiexec -n 4 ./poisson -mms_base 17 -mms_levels 4 -pc_type gamg -mms_csv convergence.csv
```

Expected: `L2` and `Linf` errors quarter as the grid halves (second order).

## Reuse
This is a starting point for related elliptic problems on structured grids.
Adapt:
- `forcing(x,y)` — the right-hand side (or MMS forcing),
- `uExact(x,y)` — the manufactured solution used for error norms,
- `gBC(x,y)` — the boundary data,
- the 5-point stencil in `ComputeMatrix` — the operator.

Keep the MMS harness and the observed-order reporting so any adaptation stays
verifiable. Validate changes with `tests/run_regression.sh`.
