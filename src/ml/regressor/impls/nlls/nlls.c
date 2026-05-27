#include <../src/ml/regressor/impls/nlls/nllsimpl.h> /*I "petscregressor.h" I*/

static PetscErrorCode NLLSEvaluateResidual(Tao tao, Vec p, Vec r, void *ptr)
{
  PetscRegressor       regressor = (PetscRegressor)ptr;
  PetscRegressor_NLLS *nlls      = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  PetscCall((*nlls->modelfn)(regressor, nlls->current_X, p, r, nlls->modelctx));
  PetscCall(VecAXPY(r, -1.0, regressor->target)); /* r = f(X, p) - y */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NLLSEvaluateJacobian(Tao tao, Vec p, Mat J, Mat Jpre, void *ptr)
{
  PetscRegressor       regressor = (PetscRegressor)ptr;
  PetscRegressor_NLLS *nlls      = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  if (nlls->jacfn) PetscCall((*nlls->jacfn)(regressor, nlls->current_X, p, J, Jpre, nlls->jacctx));
  /* If no user Jacobian, rely on TAOBRGN's finite-difference Jacobian. */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorNLLSSetFunction_NLLS(PetscRegressor regressor, Vec f, PetscRegressorNLLSFunctionFn *fn, void *ctx)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  if (f) {
    PetscCall(PetscObjectReference((PetscObject)f));
    PetscCall(VecDestroy(&nlls->f_template));
    nlls->f_template = f;
  }
  if (fn) nlls->modelfn = fn;
  if (ctx) nlls->modelctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorNLLSGetFunction_NLLS(PetscRegressor regressor, Vec *f, PetscRegressorNLLSFunctionFn **fn, void **ctx)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  if (f) *f = nlls->f_template;
  if (fn) *fn = nlls->modelfn;
  if (ctx) *ctx = nlls->modelctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorNLLSSetJacobian_NLLS(PetscRegressor regressor, Mat J, Mat Jpre, PetscRegressorNLLSJacobianFn *fn, void *ctx)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  if (J) {
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&nlls->J));
    nlls->J = J;
  }
  if (Jpre) {
    PetscCall(PetscObjectReference((PetscObject)Jpre));
    PetscCall(MatDestroy(&nlls->Jpre));
    nlls->Jpre = Jpre;
  }
  if (fn) nlls->jacfn = fn;
  if (ctx) nlls->jacctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorNLLSGetJacobian_NLLS(PetscRegressor regressor, Mat *J, Mat *Jpre, PetscRegressorNLLSJacobianFn **fn, void **ctx)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  if (J) *J = nlls->J;
  if (Jpre) *Jpre = nlls->Jpre;
  if (fn) *fn = nlls->jacfn;
  if (ctx) *ctx = nlls->jacctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorNLLSSetInitialParameters_NLLS(PetscRegressor regressor, Vec p0)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)p0));
  PetscCall(VecDestroy(&nlls->parameters0));
  nlls->parameters0 = p0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorNLLSGetParameters_NLLS(PetscRegressor regressor, Vec *p)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  *p = nlls->parameters;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorSetUp_NLLS(PetscRegressor regressor)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;
  Tao                  tao;
  const char          *prefix;

  PetscFunctionBegin;
  PetscCheck(nlls->modelfn, PetscObjectComm((PetscObject)regressor), PETSC_ERR_ARG_WRONGSTATE, "Model function not set; call PetscRegressorNLLSSetFunction() first");
  PetscCheck(nlls->parameters0, PetscObjectComm((PetscObject)regressor), PETSC_ERR_ARG_WRONGSTATE, "Initial parameter vector not set; call PetscRegressorNLLSSetInitialParameters() first");
  PetscCheck(regressor->target, PetscObjectComm((PetscObject)regressor), PETSC_ERR_ARG_WRONGSTATE, "Target vector not set; pass it to PetscRegressorFit()");

  PetscCall(VecDestroy(&nlls->parameters));
  PetscCall(VecDuplicate(nlls->parameters0, &nlls->parameters));
  PetscCall(VecCopy(nlls->parameters0, nlls->parameters));

  if (!nlls->f_template) PetscCall(VecDuplicate(regressor->target, &nlls->f_template));

  if (!nlls->J) {
    PetscInt M, N, m, n;

    PetscCall(VecGetSize(regressor->target, &M));
    PetscCall(VecGetLocalSize(regressor->target, &m));
    PetscCall(VecGetSize(nlls->parameters0, &N));
    PetscCall(VecGetLocalSize(nlls->parameters0, &n));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)regressor), m, n, M, N, NULL, &nlls->J));
    PetscCall(MatAssemblyBegin(nlls->J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(nlls->J, MAT_FINAL_ASSEMBLY));
  }

  if (!regressor->tao) PetscCall(PetscRegressorGetTao(regressor, &tao));
  else tao = regressor->tao;

  PetscCall(TaoSetType(tao, TAOBRGN));
  PetscCall(TaoSetSolution(tao, nlls->parameters));
  PetscCall(TaoSetResidualRoutine(tao, nlls->f_template, NLLSEvaluateResidual, regressor));
  PetscCall(TaoSetJacobianResidualRoutine(tao, nlls->J, nlls->Jpre ? nlls->Jpre : nlls->J, NLLSEvaluateJacobian, regressor));

  PetscCall(PetscRegressorGetOptionsPrefix(regressor, &prefix));
  PetscCall(TaoSetOptionsPrefix(tao, prefix));
  PetscCall(TaoAppendOptionsPrefix(tao, "regressor_nlls_"));
  PetscCall(TaoBRGNSetRegularizerWeight(tao, regressor->regularizer_weight));
  PetscCall(TaoSetFromOptions(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorFit_NLLS(PetscRegressor regressor)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  nlls->current_X = regressor->training;
  PetscCall(TaoSolve(regressor->tao));
  nlls->current_X = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorPredict_NLLS(PetscRegressor regressor, Mat X, Vec y)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  PetscCheck(nlls->modelfn, PetscObjectComm((PetscObject)regressor), PETSC_ERR_ARG_WRONGSTATE, "Model function not set");
  PetscCall((*nlls->modelfn)(regressor, X, nlls->parameters, y, nlls->modelctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorReset_NLLS(PetscRegressor regressor)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&nlls->f_template));
  PetscCall(VecDestroy(&nlls->parameters));
  PetscCall(VecDestroy(&nlls->parameters0));
  PetscCall(MatDestroy(&nlls->J));
  PetscCall(MatDestroy(&nlls->Jpre));
  nlls->current_X = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorDestroy_NLLS(PetscRegressor regressor)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSSetFunction_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSGetFunction_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSSetJacobian_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSGetJacobian_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSSetInitialParameters_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSGetParameters_C", NULL));
  PetscCall(PetscRegressorReset_NLLS(regressor));
  PetscCall(PetscFree(regressor->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRegressorView_NLLS(PetscRegressor regressor, PetscViewer viewer)
{
  PetscRegressor_NLLS *nlls = (PetscRegressor_NLLS *)regressor->data;
  PetscBool            isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (nlls->parameters0) {
      PetscInt N;

      PetscCall(VecGetSize(nlls->parameters0, &N));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Number of parameters: %" PetscInt_FMT "\n", N));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "Analytic Jacobian: %s\n", nlls->jacfn ? "yes" : "no (finite-difference)"));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegressorNLLSSetFunction - Sets the model function to be fit by a `PETSCREGRESSORNLLS` regressor.

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context (must be of type `PETSCREGRESSORNLLS`)
. f         - [optional] output vector template for the model evaluation $f(X, p)$; must have the same layout as the target vector `y` passed to `PetscRegressorFit()`. If `NULL`, a vector is duplicated from the target during `PetscRegressorSetUp()`.
. fn        - the function evaluating the model
- ctx       - [optional] user-defined context for the model function

  Calling Sequence of `fn`:
+ regressor - the `PetscRegressor` context
. X         - data matrix of independent variables
. p         - vector of model parameters
. f         - output vector of model evaluations $f(X, p)$
- ctx       - the user-defined context

  Level: beginner

  Notes:
  The callback receives `X` directly as a parameter, mirroring the usage model of
  `PetscRegressorFit(regressor, X, y)` and `PetscRegressorPredict(regressor, X, y)`:
  during fitting, `X` is the training matrix passed to `PetscRegressorFit()`; during
  prediction, `X` is the matrix passed to `PetscRegressorPredict()`. There is no need
  to stash `X` in the user context.

  Unlike `SNESSetFunction()` or `TaoSetResidualRoutine()`, the `fn` returns the model
  values $f(X, p)$ rather than the residual $f(X, p) - y$; the implementation forms the
  residual internally. This matches the convention of `scipy.optimize.curve_fit`.

.seealso: `PetscRegressor`, `PETSCREGRESSORNLLS`, `PetscRegressorNLLSGetFunction()`, `PetscRegressorNLLSSetJacobian()`, `PetscRegressorNLLSFunctionFn`
@*/
PetscErrorCode PetscRegressorNLLSSetFunction(PetscRegressor regressor, Vec f, PetscRegressorNLLSFunctionFn *fn, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (f) PetscValidHeaderSpecific(f, VEC_CLASSID, 2);
  PetscTryMethod(regressor, "PetscRegressorNLLSSetFunction_C", (PetscRegressor, Vec, PetscRegressorNLLSFunctionFn *, void *), (regressor, f, fn, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegressorNLLSGetFunction - Returns the model function set with `PetscRegressorNLLSSetFunction()`.

  Not Collective

  Input Parameter:
. regressor - the `PetscRegressor` context (must be of type `PETSCREGRESSORNLLS`)

  Output Parameters:
+ f   - [optional] the output vector template (may be `NULL` if none was set or `PetscRegressorSetUp()` has not been called)
. fn  - [optional] the model function
- ctx - [optional] the user-defined context

  Level: advanced

.seealso: `PetscRegressor`, `PETSCREGRESSORNLLS`, `PetscRegressorNLLSSetFunction()`
@*/
PetscErrorCode PetscRegressorNLLSGetFunction(PetscRegressor regressor, Vec *f, PetscRegressorNLLSFunctionFn **fn, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscUseMethod(regressor, "PetscRegressorNLLSGetFunction_C", (PetscRegressor, Vec *, PetscRegressorNLLSFunctionFn **, void **), (regressor, f, fn, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegressorNLLSSetJacobian - Sets the Jacobian routine for a `PETSCREGRESSORNLLS` regressor.

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context (must be of type `PETSCREGRESSORNLLS`)
. J         - [optional] Jacobian matrix; if `NULL`, a dense $M \times N$ matrix is allocated during `PetscRegressorSetUp()`
. Jpre      - [optional] preconditioner matrix (may be the same as `J`)
. fn        - [optional] function evaluating the Jacobian $\partial f/\partial p$; if `NULL`, the underlying `TAOBRGN` solver uses a finite-difference approximation
- ctx       - [optional] user-defined context for the Jacobian function

  Calling Sequence of `fn`:
+ regressor - the `PetscRegressor` context
. X         - data matrix of independent variables
. p         - vector of model parameters
. J         - Jacobian matrix to fill
. Jpre      - preconditioner matrix
- ctx       - the user-defined context

  Level: intermediate

  Note:
  The default dense Jacobian storage is convenient for small numbers of parameters.
  For large parameter counts, pre-allocate a sparse `J` and pass it to this routine.

.seealso: `PetscRegressor`, `PETSCREGRESSORNLLS`, `PetscRegressorNLLSGetJacobian()`, `PetscRegressorNLLSSetFunction()`, `PetscRegressorNLLSJacobianFn`
@*/
PetscErrorCode PetscRegressorNLLSSetJacobian(PetscRegressor regressor, Mat J, Mat Jpre, PetscRegressorNLLSJacobianFn *fn, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (J) PetscValidHeaderSpecific(J, MAT_CLASSID, 2);
  if (Jpre) PetscValidHeaderSpecific(Jpre, MAT_CLASSID, 3);
  PetscTryMethod(regressor, "PetscRegressorNLLSSetJacobian_C", (PetscRegressor, Mat, Mat, PetscRegressorNLLSJacobianFn *, void *), (regressor, J, Jpre, fn, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegressorNLLSGetJacobian - Returns the Jacobian routine set with `PetscRegressorNLLSSetJacobian()`.

  Not Collective

  Input Parameter:
. regressor - the `PetscRegressor` context (must be of type `PETSCREGRESSORNLLS`)

  Output Parameters:
+ J    - [optional] the Jacobian matrix
. Jpre - [optional] the preconditioner matrix
. fn   - [optional] the Jacobian function
- ctx  - [optional] the user-defined context

  Level: advanced

.seealso: `PetscRegressor`, `PETSCREGRESSORNLLS`, `PetscRegressorNLLSSetJacobian()`
@*/
PetscErrorCode PetscRegressorNLLSGetJacobian(PetscRegressor regressor, Mat *J, Mat *Jpre, PetscRegressorNLLSJacobianFn **fn, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscUseMethod(regressor, "PetscRegressorNLLSGetJacobian_C", (PetscRegressor, Mat *, Mat *, PetscRegressorNLLSJacobianFn **, void **), (regressor, J, Jpre, fn, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorNLLSSetInitialParameters - Sets the initial guess for the parameters of a `PETSCREGRESSORNLLS` regressor.

  Logically Collective

  Input Parameters:
+ regressor - the `PetscRegressor` context (must be of type `PETSCREGRESSORNLLS`)
- p0        - vector containing the initial parameter values

  Level: beginner

  Note:
  The length of `p0` determines the number of model parameters `N`. This routine must be
  called before `PetscRegressorFit()`.

.seealso: `PetscRegressor`, `PETSCREGRESSORNLLS`, `PetscRegressorNLLSGetParameters()`
@*/
PetscErrorCode PetscRegressorNLLSSetInitialParameters(PetscRegressor regressor, Vec p0)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscValidHeaderSpecific(p0, VEC_CLASSID, 2);
  PetscTryMethod(regressor, "PetscRegressorNLLSSetInitialParameters_C", (PetscRegressor, Vec), (regressor, p0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRegressorNLLSGetParameters - Returns the vector of fitted parameters from a `PETSCREGRESSORNLLS` regressor.

  Not Collective, but the vector is parallel if the regressor is

  Input Parameter:
. regressor - the `PetscRegressor` context (must be of type `PETSCREGRESSORNLLS`)

  Output Parameter:
. p - the vector of fitted parameters

  Level: beginner

  Note:
  The vector is owned by the regressor; do not destroy it. It is valid only after
  `PetscRegressorFit()` has been called.

.seealso: `PetscRegressor`, `PETSCREGRESSORNLLS`, `PetscRegressorNLLSSetInitialParameters()`, `PetscRegressorFit()`
@*/
PetscErrorCode PetscRegressorNLLSGetParameters(PetscRegressor regressor, Vec *p)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(p, 2);
  PetscUseMethod(regressor, "PetscRegressorNLLSGetParameters_C", (PetscRegressor, Vec *), (regressor, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PETSCREGRESSORNLLS - Nonlinear least squares regression model.

   Level: beginner

   Notes:
   Fits a user-supplied nonlinear model $f(X, p)$ to observation data $y$ by minimizing
   $\|f(X, p) - y\|^2$ over the parameter vector $p$. The model function is provided via
   `PetscRegressorNLLSSetFunction()` and an optional analytic Jacobian via
   `PetscRegressorNLLSSetJacobian()`. An initial parameter guess must be supplied with
   `PetscRegressorNLLSSetInitialParameters()`. After `PetscRegressorFit()`, the fitted
   parameters are accessible via `PetscRegressorNLLSGetParameters()`.

   The model and Jacobian callbacks receive the data matrix `X` directly as an argument,
   matching the usage model of `PetscRegressorFit(regressor, X, y)` and
   `PetscRegressorPredict(regressor, X, y)`. The argument order $(X, p)$ matches the
   convention of `scipy.optimize.curve_fit`.

   Internally this implementation drives the `TAOBRGN` solver. If no Jacobian function is
   set, `TAOBRGN` uses a finite-difference Jacobian.

   The default Jacobian storage is dense $M \times N$. For large $N$, pre-allocate a
   sparse matrix and pass it through `PetscRegressorNLLSSetJacobian()`.

.seealso: `PetscRegressorCreate()`, `PetscRegressor`, `PetscRegressorSetType()`, `PetscRegressorNLLSSetFunction()`,
          `PetscRegressorNLLSSetJacobian()`, `PetscRegressorNLLSSetInitialParameters()`, `PetscRegressorNLLSGetParameters()`, `TAOBRGN`
M*/
PETSC_EXTERN PetscErrorCode PetscRegressorCreate_NLLS(PetscRegressor regressor)
{
  PetscRegressor_NLLS *nlls;

  PetscFunctionBegin;
  PetscCall(PetscNew(&nlls));
  regressor->data = (void *)nlls;

  regressor->ops->setup   = PetscRegressorSetUp_NLLS;
  regressor->ops->reset   = PetscRegressorReset_NLLS;
  regressor->ops->destroy = PetscRegressorDestroy_NLLS;
  regressor->ops->view    = PetscRegressorView_NLLS;
  regressor->ops->fit     = PetscRegressorFit_NLLS;
  regressor->ops->predict = PetscRegressorPredict_NLLS;

  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSSetFunction_C", PetscRegressorNLLSSetFunction_NLLS));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSGetFunction_C", PetscRegressorNLLSGetFunction_NLLS));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSSetJacobian_C", PetscRegressorNLLSSetJacobian_NLLS));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSGetJacobian_C", PetscRegressorNLLSGetJacobian_NLLS));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSSetInitialParameters_C", PetscRegressorNLLSSetInitialParameters_NLLS));
  PetscCall(PetscObjectComposeFunction((PetscObject)regressor, "PetscRegressorNLLSGetParameters_C", PetscRegressorNLLSGetParameters_NLLS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
