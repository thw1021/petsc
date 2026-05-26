# Plan: Add a Nonlinear Least Squares `PetscRegressor` Type

## Goal

Add a new `PetscRegressor` implementation, registered as `-regressor_type nlls`,
that fits a user-supplied nonlinear model `f(X, p)` to observation data `y` in
the least squares sense. The user-facing intent is the same as
`scipy.optimize.curve_fit`, but the API follows PETSc conventions: the model
and Jacobian callbacks are attached to the `PetscRegressor` via setters (à la
`SNESSetFunction()` / `SNESSetJacobian()` / `TaoSetResidualRoutine()`),
not passed into `PetscRegressorFit()`.

Internally the implementation wraps `TAOBRGN` (the bound-constrained
regularized Gauss-Newton solver already used by the linear regressor) so we
inherit a tested nonlinear LS engine and a Jacobian-free finite-difference
fallback.

The first iteration deliberately mirrors only the *required* surface of
`curve_fit`. Optional features (`sigma`, `absolute_sigma`, `bounds`,
covariance computation, method selection) are explicitly out of scope and
left for a follow-up.

## Mapping `scipy.optimize.curve_fit` to PETSc

| `curve_fit` argument | PETSc equivalent |
|---|---|
| `f(x, *params)` (model) | `PetscRegressorNLLSSetFunction(reg, f, fn, ctx)` — `fn` receives the data matrix `X` and parameters `p`, and produces model evaluations |
| `xdata` | `X` passed into `PetscRegressorFit(reg, X, y)` and `PetscRegressorPredict(reg, X, y)` |
| `ydata` | `y` passed into `PetscRegressorFit(reg, X, y)` |
| `p0` (initial guess) | `PetscRegressorNLLSSetInitialParameters(reg, p0)` |
| `jac` (analytic Jacobian) | `PetscRegressorNLLSSetJacobian(reg, J, Jpre, jacfn, ctx)` (optional; otherwise `TAOBRGN` does finite-difference) |
| Returned `popt` | `PetscRegressorNLLSGetParameters(reg, &p)` |
| `sigma`, `absolute_sigma`, `bounds`, `pcov`, `method`, `full_output` | **Out of scope.** |

### Callback shape: `X` is an explicit argument

The model and Jacobian callbacks **receive the data matrix `X` directly** as
an argument. They do not close over it through the user ctx.

This decision aligns the callbacks with the existing
`PetscRegressorFit(reg, X, y)` / `PetscRegressorPredict(reg, X, y)` usage
model — in both entry points `X` is provided explicitly per call, and the
callbacks see the same `X` the user just handed in. Concretely:

- During `Fit`, the wrapper invokes the model with `X = reg->training`.
- During `Predict(reg, X_new, y)`, the wrapper invokes the model with
  `X = X_new`.

This avoids two footguns:

- Forcing users to mutate a ctx field between `Fit` and `Predict`.
- Forcing the implementation to stash a pointer to the prediction `X` on
  the regressor for the callback to find.

### Model vs. residual

The callback returns the **model evaluation `f(X, p)`** (length `M`), not
the residual `f(X, p) - y`. The NLLS impl subtracts `y` internally when
forming the Tao residual. This matches `curve_fit`'s mental model and lets
`PetscRegressorPredict()` simply reuse the same callback with the fitted
parameters — no need to invent a separate predict path.

## File-level changes

```
include/petscregressor.h                            (modified)
src/ml/regressor/interface/regressorregi.c          (modified — register nlls)
src/ml/regressor/impls/makefile                     (no change — recursive descent)
src/ml/regressor/impls/nlls/                        (new directory)
src/ml/regressor/impls/nlls/makefile                (new)
src/ml/regressor/impls/nlls/nllsimpl.h              (new — private header)
src/ml/regressor/impls/nlls/nlls.c                  (new — implementation)
src/ml/regressor/tests/ex_nlls.c                    (new — exponential-fit test)
src/ml/regressor/tests/output/ex_nlls_*.out         (new — expected outputs)
```

No changes to `include/petsc/private/regressorimpl.h` — the existing
`_PetscRegressorOps` table already exposes `setup`, `fit`, `predict`,
`setfromoptions`, `view`, `reset`, `destroy`, and the training matrix and
target vector are already on the base struct.

## Public API additions (in `include/petscregressor.h`)

```c
#define PETSCREGRESSORNLLS "nlls"

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscRegressorNLLSFunctionFn(PetscRegressor reg, Mat X, Vec p, Vec f, void *ctx);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscRegressorNLLSJacobianFn(PetscRegressor reg, Mat X, Vec p, Mat J, Mat Jpre, void *ctx);

PETSC_EXTERN PetscErrorCode PetscRegressorNLLSSetFunction(PetscRegressor, Vec, PetscRegressorNLLSFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSGetFunction(PetscRegressor, Vec *, PetscRegressorNLLSFunctionFn **, void **);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSSetJacobian(PetscRegressor, Mat, Mat, PetscRegressorNLLSJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSGetJacobian(PetscRegressor, Mat *, Mat *, PetscRegressorNLLSJacobianFn **, void **);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSSetInitialParameters(PetscRegressor, Vec);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSGetParameters(PetscRegressor, Vec *);
```

Notes:

- **Function callback signature**: `(reg, X, p, f, ctx)`. Argument order
  matches `scipy.optimize.curve_fit`'s `f(x, *params)`: the independent
  variable first, then the parameters. `X` is the data matrix (rows are
  samples), `p` is the parameter vector (length `N`), and `f` is the
  output vector of model evaluations (length `M`, layout matching `y`).
  The "function" name follows PETSc convention — it is the thing being
  driven toward a target, not the residual.

- **Jacobian callback signature**: `(reg, X, p, J, Jpre, ctx)`. Same
  `(X, p)` ordering as the function callback. Computes `df/dp` (an
  `M x N` matrix). Since `r = f - y` and `y` is constant in `p`, this is
  also `dr/dp`.

- The `Vec` argument to `SetFunction` is an optional pre-allocated
  output template for the model evaluations (same calling convention as
  `SNESSetFunction(snes, F, ...)` and `TaoSetResidualRoutine(tao, F, ...)`,
  but here it holds `f(X, p)`, not a residual). If `NULL`, `SetUp`
  duplicates `reg->target` to obtain one. If supplied, it must have the
  same layout as `y`.

- `SetInitialParameters` takes a `Vec` (parallel-aware). Its layout
  determines `N`; we duplicate it into the solution vector handed to Tao.

## Implementation sketch

### `nllsimpl.h`

```c
#pragma once
#include <petsc/private/regressorimpl.h>
#include <petsctao.h>

typedef struct {
  /* User callbacks */
  PetscRegressorNLLSFunctionFn *modelfn;
  void                         *modelctx;
  Vec                           f_template; /* template for model output f(X, p); layout matches y; passed to Tao as its residual vector */

  PetscRegressorNLLSJacobianFn *jacfn;
  void                         *jacctx;
  Mat                           J, Jpre;           /* Jacobian (M x N) */

  /* Solver state */
  Vec parameters;     /* fitted parameters (length N); Tao's solution vector */
  Vec parameters0;    /* user-provided initial guess */

  /* Matrix currently in play for the callback (set per Fit/Predict call). */
  Mat current_X;
} PetscRegressor_NLLS;
```

`current_X` is set immediately before any path that may invoke Tao (so the
residual/Jacobian wrappers can hand it to the user callback) and cleared
afterward. During `Fit` it points at `reg->training`; during `Predict` it
points at the `X` argument.

### `nlls.c` — Tao adapter functions

```c
/* Tao calls these with the parameter vector; we evaluate the user model
 * against the currently-active X and form the residual. */
static PetscErrorCode NLLSEvaluateResidual(Tao tao, Vec p, Vec r, void *ptr)
{
  PetscRegressor       reg  = (PetscRegressor)ptr;
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)reg->data;

  PetscFunctionBegin;
  PetscCall((*nlls->modelfn)(reg, nlls->current_X, p, r, nlls->modelctx));
  PetscCall(VecAXPY(r, -1.0, reg->target));   /* r = f(X, p) - y */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NLLSEvaluateJacobian(Tao tao, Vec p, Mat J, Mat Jpre, void *ptr)
{
  PetscRegressor       reg  = (PetscRegressor)ptr;
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)reg->data;

  PetscFunctionBegin;
  if (nlls->jacfn) PetscCall((*nlls->jacfn)(reg, nlls->current_X, p, J, Jpre, nlls->jacctx));
  /* else: no-op; rely on TAOBRGN finite-difference Jacobian. */
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### `PetscRegressorSetUp_NLLS`

1. Verify `nlls->modelfn` is set; `SETERRQ` otherwise.
2. Verify `nlls->parameters0` is set; `SETERRQ` otherwise — unlike Python
   introspection, we cannot infer `N` automatically.
3. `PetscRegressorGetTao(reg, &tao)` (creates the Tao if needed).
4. `VecDuplicate(nlls->parameters0, &nlls->parameters); VecCopy(p0, params)`.
5. If `nlls->f_template` is `NULL`, `VecDuplicate(reg->target,
   &nlls->f_template)`.
6. If `nlls->J` is `NULL`, create a default dense `M x N` Jacobian using the
   row layout of `reg->target` and the column layout of
   `nlls->parameters0`. (Dense default is fine for small `N`; flag in the
   manpage that for large `N` the user should pre-allocate a sparse `J` via
   `SetJacobian`.)
7. `TaoSetType(tao, TAOBRGN)`.
8. `TaoSetSolution(tao, nlls->parameters)`.
9. `TaoSetResidualRoutine(tao, nlls->f_template, NLLSEvaluateResidual, reg)`.
10. `TaoSetJacobianResidualRoutine(tao, nlls->J, nlls->Jpre ? nlls->Jpre : nlls->J, NLLSEvaluateJacobian, reg)`.
11. Set the Tao options prefix to match `regressor` prefix +
    `regressor_nlls_` (mirrors `linear.c:158–160`).
12. `TaoBRGNSetRegularizerWeight(tao, reg->regularizer_weight)` if the user
    has set one; otherwise leave at Tao default. (No regularization-type
    plumbing in v1.)
13. `TaoSetFromOptions(tao)`.

### `PetscRegressorFit_NLLS`

```c
PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)reg->data;

PetscFunctionBegin;
nlls->current_X = reg->training;
PetscCall(TaoSolve(reg->tao));
nlls->current_X = NULL;
PetscFunctionReturn(PETSC_SUCCESS);
```

The fitted parameters live in `nlls->parameters` because that is the `Vec`
passed to `TaoSetSolution`.

### `PetscRegressorPredict_NLLS`

```c
PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)reg->data;

PetscFunctionBegin;
PetscCheck(nlls->modelfn, ..., PETSC_ERR_ARG_WRONGSTATE, "model function not set");
PetscCall((*nlls->modelfn)(reg, X, nlls->parameters, y, nlls->modelctx));
PetscFunctionReturn(PETSC_SUCCESS);
```

`X` is the matrix the caller passed to `PetscRegressorPredict()`; `y`
receives the model evaluations directly — no residual subtraction in this
path.

### `PetscRegressorReset_NLLS` / `PetscRegressorDestroy_NLLS`

- Reset: `VecDestroy(&nlls->f_template); VecDestroy(&nlls->parameters); VecDestroy(&nlls->parameters0); MatDestroy(&nlls->J); MatDestroy(&nlls->Jpre); nlls->current_X = NULL;`
- Destroy: composed-function unregistrations + `PetscFree(reg->data)`,
  matching the pattern at `linear.c:199–212`.

### `PetscRegressorView_NLLS`

Print `"Number of parameters: N"`, whether an analytic Jacobian is in use,
and let the base routine call `TaoView` (it already does — see
`regressor.c:128`).

### `PetscRegressorSetFromOptions_NLLS`

Minimal in v1: nothing NLLS-specific needs an options-database key beyond
what already flows through to the Tao subsolver via the prefix. Leave the
no-op in place so future flags can be added without an interface churn.

## Setters / getters (`PetscObjectComposeFunction` plumbing)

Follow the exact pattern in `linear.c:553–559`. Each public
`PetscRegressorNLLSXxx()` function calls `PetscUseMethod` / `PetscTryMethod`
on a `_C` suffix, and `Create_NLLS` composes the type-specific
implementation. `Destroy_NLLS` must `NULL` them out (`linear.c:202–208`).

`PetscRegressorNLLSSetFunction_NLLS` should reference the supplied output
template `f` (`PetscObjectReference((PetscObject)f)`) and destroy any
prior one, mirroring how `PetscRegressorFit` handles `X` and `y` in
`regressor.c:262–271`.

`PetscRegressorNLLSSetJacobian_NLLS` should likewise reference both `J` and
`Jpre` and destroy any prior matrices.

## Registration

In `src/ml/regressor/interface/regressorregi.c`:

```c
PETSC_EXTERN PetscErrorCode PetscRegressorCreate_NLLS(PetscRegressor);
...
#if !PetscDefined(USE_COMPLEX)
  PetscCall(PetscRegressorRegister(PETSCREGRESSORLINEAR, PetscRegressorCreate_Linear));
  PetscCall(PetscRegressorRegister(PETSCREGRESSORNLLS,   PetscRegressorCreate_NLLS));
#endif
```

The existing `!complex` guard fits NLLS as well, since `TAOBRGN` is
real-only.

## Makefile

New `src/ml/regressor/impls/nlls/makefile` is a copy of the linear one:

```make
-include ../../../../../petscdir.mk
MANSEC    = ML
SUBMANSEC = PetscRegressor
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules_doc.mk
```

`src/ml/regressor/impls/makefile` already descends into all subdirectories,
so no edit there.

## Test: `src/ml/regressor/tests/ex_nlls.c`

Fit an exponential `f(x, p) = p_0 * exp(p_1 * x) + p_2` to synthetic data
generated from known parameters. Verify recovered parameters within a
tolerance.

Outline:

```c
static char help[] = "Tests PETSCREGRESSORNLLS by fitting an exponential model.\n\n";
#include <petscregressor.h>

/* model: y_i = p0 * exp(p1 * x_i) + p2; X is M x 1 holding x_i */
static PetscErrorCode Model(PetscRegressor reg, Mat X, Vec p, Vec f, void *ctx);
static PetscErrorCode Jacobian(PetscRegressor reg, Mat X, Vec p, Mat J, Mat Jpre, void *ctx);

int main(...) {
  /* generate M=20 samples from p_true = {2.5, -0.3, 0.5} */
  /* PetscRegressorCreate; SetType NLLS; SetFromOptions */
  /* PetscRegressorNLLSSetFunction(reg, NULL, Model, NULL); */
  /* if (analytic_jac) PetscRegressorNLLSSetJacobian(reg, J, J, Jacobian, NULL); */
  /* p0 = {1, 0, 0}; PetscRegressorNLLSSetInitialParameters(reg, p0) */
  /* PetscRegressorFit(reg, X, y) */
  /* PetscRegressorNLLSGetParameters(reg, &p_fit); VecView(...) */
  /* PetscRegressorPredict(reg, X, y_predicted); VecView(...) */
  /* cleanup; single PetscFinalize at the end */
}

/*TEST
  build:
    requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

  test:
    suffix: analytic_jac
    args: -regressor_view ::ascii_info_detail

  test:
    suffix: fd_jac
    args: -regressor_view ::ascii_info_detail -no_analytic_jacobian
TEST*/
```

Two test cases ensure both the analytic-Jacobian path and the
finite-difference fallback (handled by `TAOBRGN`) work. Expected outputs
land in `src/ml/regressor/tests/output/ex_nlls_analytic_jac.out` and
`ex_nlls_fd_jac.out`. Generate them by running the test once after the
implementation is in.

Tolerance: convergence of `TAOBRGN` against synthetic noise-free data
should give parameters accurate to many digits. Use `filter` to truncate to
~4 decimal places so the test is robust across platforms (see existing
regressor test outputs for the style).

## Style / convention checklist (per `CLAUDE.md`)

- `PetscFunctionBegin` / `PetscFunctionReturn(PETSC_SUCCESS)` in every PETSc
  routine; `PetscFunctionBeginUser` in the test's `main()` and user
  callbacks.
- All PETSc calls wrapped with `PetscCall(...)`.
- Single-statement `if`/`else` without braces.
- Header prototypes omit parameter names; the function typedefs include them.
- No `PetscFinalize()` inside a conditional in the test.
- Group local variable declarations by type at the top of each block.
- One blank line between local declarations and `PetscFunctionBegin`.
- `make checkbadSource` and `make checkclangformat` clean before submitting.

## Documentation touchpoints

- Man pages: every new public function has a `/*@ ... @*/` block with the
  correct collectivity tag, parameter list, `Level:`, `Options Database
  Key:` (none for NLLS in v1), and a `.seealso:` chain including
  `PETSCREGRESSORNLLS`, `PetscRegressor`, `TAOBRGN`.
- `/*MC ... M*/` block for `PETSCREGRESSORNLLS` matching the existing
  `PETSCREGRESSORLINEAR` block at `linear.c:512–529`.
- The `Notes:` block on `PETSCREGRESSORNLLS` should call out: (a) the
  callback receives the data matrix `X` directly as an argument (same `X`
  the caller passes into `PetscRegressorFit()` / `PetscRegressorPredict()`);
  (b) the default Jacobian storage is dense `M x N` and the user should
  pre-allocate a sparse `J` for large `N`; (c) prediction calls the same
  model callback with the fitted parameters and the prediction `X`.

## Out of scope for v1 (track separately)

- `sigma`-weighted residual.
- Parameter bounds (would require switching from `TAOBRGN` to a different
  Tao type, or threading bounds through).
- Parameter covariance estimate `pcov`.
- Method selection (`'lm'` vs `'trf'` vs `'dogbox'`).
- Auto-default `p0` to all-ones.
- petsc4py bindings.

## Implementation order

1. Header (`include/petscregressor.h`) additions — typedefs, defines,
   prototypes.
2. `src/ml/regressor/impls/nlls/` skeleton (header, makefile, empty `.c`
   with `PetscRegressorCreate_NLLS` stub).
3. Wire registration in `regressorregi.c`; verify `make` builds and
   `-regressor_type nlls` is recognised (no-op fit will fail cleanly).
4. Fill in `SetUp`, `Fit`, the Tao residual/Jacobian wrappers.
5. Add `SetFunction`, `SetJacobian`, `SetInitialParameters`,
   `GetParameters`, plus `_C`-composed implementations and corresponding
   `Reset`/`Destroy` cleanup.
6. Add `Predict_NLLS`.
7. Add `View_NLLS` and (optional) `SetFromOptions_NLLS`.
8. Write `ex_nlls.c`; iterate on tolerances and `filter` until output is
   reproducible.
9. Run `make checkclangformat checkbadSource` and `make test
   search='regressor%nlls'`. Capture expected output.

## Verification

- `make -C src/ml/regressor` builds clean.
- `make test search='regressor%nlls'` passes both subtests.
- `make test search='regressor%linear'` still passes (no regression in the
  base interface).
- `make checkclangformat` and `make checkbadSource` are clean.
