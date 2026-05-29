# PETSc SINDy + Lorenz tutorial

## Context

Port the hand-rolled Python SINDy example at `~/proj/sindy-bstollnitz/sindy/lorenz-custom/` (three files: `1_generate_data.py`, `2_fit.py`, `3_predict.py`, with shared code in `common.py`) to a single PETSc C tutorial. SINDy (Sparse Identification of Nonlinear Dynamics) discovers an ODE system from trajectory data by:

1. Simulating the system to get state snapshots `U ∈ ℝ^{n × 3}`,
2. Building a candidate-function "library" `Θ(U) ∈ ℝ^{n × p}` (here, polynomials up to order 2 in `x, y, z` — `p = 10` features),
3. Estimating derivatives `U' = dU/dt ∈ ℝ^{n × 3}` from `U`,
4. Solving the sparse regression problem `Θ Ξ = U'` for the coefficient matrix `Ξ ∈ ℝ^{p × 3}`. Sparsity selects which library terms appear in each equation.

The Python uses Sequential Thresholded Least-Squares for step (4); per user decision, this port uses **pure LASSO via `PetscRegressor`** (one fit per state column, `REGRESSOR_LINEAR_LASSO`). PETSc TS does step (1); `PetscRegressor` does step (4). Steps (2) and (3) are short helper routines on dense `Mat`s.

The tutorial demonstrates an end-to-end workflow joining `TS` and `PetscRegressor`, the first such example in the tree.

## Files to create

All new. Locations and roles:

- `src/ml/regressor/tutorials/makefile` — aggregator makefile for the new tutorials directory. 4 levels deep:
  ```
  -include ../../../../petscdir.mk
  MANSEC    = ML
  SUBMANSEC = PetscRegressor
  include ${PETSC_DIR}/lib/petsc/conf/variables
  include ${PETSC_DIR}/lib/petsc/conf/rules_doc.mk
  ```
  Mirrors `src/ml/regressor/makefile` (which uses `rules_doc.mk`).

- `src/ml/regressor/tutorials/sindy/makefile` — leaf makefile, 5 levels deep:
  ```
  -include ../../../../../petscdir.mk
  include ${PETSC_DIR}/lib/petsc/conf/variables
  include ${PETSC_DIR}/lib/petsc/conf/rules
  ```
  Mirrors `src/ml/regressor/tests/makefile` style (leaf, uses `rules`).

- `src/ml/regressor/tutorials/sindy/ex1.c` — the tutorial source (single file). Structure below.

- `src/ml/regressor/tutorials/sindy/output/ex1.out` — CI reference output (created during dev by running the example once and capturing stdout).

PETSc's recursive directory walker discovers new subdirs automatically; no edit to the parent `src/ml/regressor/makefile` is needed.

## `ex1.c` structure

Single binary, three phases driven from `main`:

1. **Data generation.** Set up `TS` with `LorenzRHS` (truth: σ=10, ρ=28, β=8/3). Use `TSRK` with `TSRK4` and a fixed `dt`. A monitor stashes each step's solution into row `step` of a seq-dense `Mat U` of shape `(nsteps+1) × 3`. Assemble `U` once after `TSSolve`.

2. **Fit.** Build `Mat Theta` (shape `(nsteps+1) × 10`) and `Mat Uprime` (shape `(nsteps+1) × 3`) from `U`. Loop `j = 0..2`: get column `j` of `Uprime` as a `Vec` via `MatDenseGetColumnVecRead`, call `PetscRegressorFit(regressor, Theta, y_j)`, `PetscRegressorLinearGetCoefficients(regressor, &xi_j)`, copy `xi_j` into column `j` of a seq-dense `Mat Xi` (shape `10 × 3`). Call `PetscRegressorReset(regressor)` between columns (required — `PetscRegressorReset` in `interface/regressor.c` flips `setupcalled = PETSC_FALSE`; without it the second `Fit` reuses stale internal state from the linear impl's setup).

3. **Predict** (skippable via `-sindy_skip_predict`). Build a second `TS` whose RHS is `SindyRHS` driven by `Xi`. Integrate from the same IC over a user-controlled horizon (`-sindy_predict_tmax`, default 5 — short enough that the chaotic divergence between true and discovered systems stays bounded). Print the final state and the t at which integration completed.

Print the discovered system via `ViewDiscoveredEquations` (see "CI reproducibility" below).

### Function-level breakdown

- `LorenzRHS(TS, t, Vec X, Vec F, void *ctx)` — direct port of `1_generate_data.py:lorenz`. `f[0]=σ(x[1]−x[0])`, `f[1]=x[0]*(ρ−x[2])−x[1]`, `f[2]=x[0]*x[1]−β*x[2]`. Pattern from `src/ts/tutorials/ex16.c:79`.

- `SnapshotMonitor(TS, PetscInt step, PetscReal t, Vec X, void *ctx)` — `ctx->U` is the dense snapshot Mat. `MatSetValues(U, 1, &step, 3, {0,1,2}, x_array, INSERT_VALUES)`. Do not assemble inside the monitor — assemble once after `TSSolve`.

- `BuildLibraryRow(const PetscScalar x[3], PetscScalar row[10])` — fills `{1, x, y, z, x², xy, xz, y², yz, z²}`. Factored as a small helper so `BuildLibrary` and `SindyRHS` share one implementation and cannot drift apart.

- `BuildLibrary(Mat U, Mat Theta)` — `MatDenseGetArrayRead(U,&u)` + `MatDenseGetArrayWrite(Theta,&th)`, loop rows, call `BuildLibraryRow`. PETSc dense matrices are column-major: column `j` of an `m × n` Mat starts at `array[j*m]`. Use raw arrays (not `MatSetValues`) for speed.

- `FiniteDifferenceDerivatives(Mat U, Mat Uprime, PetscReal dt)` — 2nd-order central interior, 1st-order forward/backward at endpoints. Matches `derivative.dxdt(..., kind="finite_difference", k=1)` (which is `np.gradient`). Operate on raw dense-array pointers.

- `SindyRHS(TS, t, Vec X, Vec F, void *ctx)` — `ctx` holds `Mat Xi` and a persistent length-10 `Vec theta`. Build `theta` from `X` via `BuildLibraryRow`, then `MatMultTranspose(Xi, theta, F)` produces `F = Xiᵀ θ ∈ ℝ³`.

- `ViewDiscoveredEquations(Mat Xi, PetscReal display_thresh, PetscViewer)` — see next section.

### Application context

```c
typedef struct {
  PetscReal sigma, rho, beta;     /* truth params for LorenzRHS */
  Mat       U;                    /* (nsteps+1) × 3 snapshots */
  /* predict-phase fields: */
  Mat       Xi;                   /* 10 × 3 coefficients */
  Vec       theta_row;            /* length-10 scratch */
} AppCtx;
```

The data-gen TS and predict TS each set their own RHS context but share the struct.

## CI reproducibility

LASSO output is not bit-stable (`src/ml/regressor/tests/output/ex3_lasso_1.out` already shows entries at ~`1e-6`, ~`1e-7` magnitudes that vary across BLAS/architectures). Strategy: **pretty-print only coefficients above a display threshold** (default `-sindy_display_threshold 0.01`). `ViewDiscoveredEquations` produces:

```
dx/dt = -10.000 x + 10.000 y
dy/dt = 28.000 x - 1.000 y - 1.000 xz
dz/dt = -2.667 z + 1.000 xy
```

Hard-code the 10 feature-name strings `{"1","x","y","z","x^2","xy","xz","y^2","yz","z^2"}` and 3 state names `{"dx/dt","dy/dt","dz/dt"}`. Read `Xi` via `MatDenseGetArrayRead`. Format each non-zero coefficient with `%.3f`, joining with `" + "` / `" - "` based on sign, suppressing constant "+0.000".

The display threshold (0.01) sits above the LASSO noise floor and below the smallest true Lorenz coefficient (β ≈ 2.667), so the symbolic output is robust to LASSO's per-run wobble.

For development/debugging only, expose `-sindy_view_xi_raw` that does a full `MatView(Xi, PETSC_VIEWER_STDOUT_WORLD)`. Not used in the CI test.

## Options

- `-lorenz_sigma`, `-lorenz_rho`, `-lorenz_beta` — truth ODE params (defaults 10, 28, 8/3).
- `-sindy_ic_x`, `-sindy_ic_y`, `-sindy_ic_z` — initial condition (defaults −8, 8, 27, matching Python).
- `-sindy_dt`, `-sindy_tmax` — data-generation time step / horizon (defaults 0.01, 10.0).
- `-sindy_predict_tmax` — predict-phase horizon (default 5.0).
- `-sindy_skip_predict` — skip phase 3.
- `-sindy_lambda` — regularizer weight (default 0.05). Forwarded to `PetscRegressorSetRegularizerWeight`.
- `-sindy_display_threshold` — pretty-print cutoff (default 0.01).
- `-sindy_view_xi_raw` — debug `MatView` of full Xi.
- `-sindy_fit_intercept` — default **off**. Because Theta has a constant "1" column, fitting a separate intercept would double-count and shift the discovered coefficients. Default off; expose the option for experimentation. The Python reference also does not fit an intercept.
- TS options (`-ts_type`, `-ts_rk_type`, `-ts_max_steps`, ...) flow through `TSSetFromOptions` — don't shadow them.

## MPI scope

Single rank. The TS state is 3 DOFs; the snapshot/Theta/Xi matrices are seq-dense; the work doesn't parallelize meaningfully at this size. Use the ex16.c idiom:

```c
PetscMPIInt size;
PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,
           "This is a uniprocessor example only!");
```

TEST block uses `nsize: 1`.

## CI test block

At the bottom of `ex1.c`:

```
/*TEST

   build:
     requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
     suffix: lasso
     nsize: 1
     args: -ts_type rk -ts_rk_type 4 -sindy_dt 0.01 -sindy_tmax 10.0 \
           -sindy_lambda 0.05 -sindy_predict_tmax 5.0

TEST*/
```

Reference output `output/ex1.out` is captured from a working run and committed alongside the source. The build `requires:` line matches `src/ml/regressor/tests/ex3.c`.

## Verification

Run from `$PETSC_DIR`:

1. Build: `make -C src/ml/regressor/tutorials/sindy ex1`
2. Quick smoke run: `./src/ml/regressor/tutorials/sindy/ex1 -ts_type rk -ts_rk_type 4 -sindy_dt 0.01 -sindy_tmax 10.0 -sindy_lambda 0.05`
3. Expect output containing the three discovered equations matching Lorenz (σ=10, ρ=28, β≈2.667) to ~3 digits. If recovery is poor, tune `-sindy_lambda` (try 0.01) and `-sindy_dt` (smaller for cleaner finite-difference derivatives).
4. Sanity-check predict phase: confirm the SINDy TS doesn't blow up over `predict_tmax`. Final state should be on the same order as the truth at t=5 (not requiring trajectory match — Lorenz is chaotic).
5. Capture stdout as `src/ml/regressor/tutorials/sindy/output/ex1.out`.
6. Run via PETSc test harness: `make test search='ml_regressor_tutorials_sindy*'`.
7. Style: `make clangformat` then `make checkbadSource`.

If LASSO with `λ=0.05` leaves visible noise terms above the display threshold, the first knob to turn is `λ` (raise to 0.1) before lowering `display_threshold`. The smallest true coefficient is β=2.667, so a display threshold up to ~1.0 is safe if the noise floor is unexpectedly high.
