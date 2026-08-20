/*
    User interface for the nonlinear solvers package.
*/
#pragma once

#include <petscksp.h>
#include <petscdmtypes.h>
#include <petscfvtypes.h>
#include <petscdmdatypes.h>
#include <petscsnestypes.h>

/* SUBMANSEC = SNES */

/* Logging support */
PETSC_EXTERN PetscClassId SNES_CLASSID;
PETSC_EXTERN PetscClassId DMSNES_CLASSID;

PETSC_EXTERN PetscErrorCode SNESInitializePackage(void);
PETSC_EXTERN PetscErrorCode SNESFinalizePackage(void);

PETSC_EXTERN PetscErrorCode SNESCreate(MPI_Comm, SNES *);
PETSC_EXTERN PetscErrorCode SNESParametersInitialize(SNES);
PETSC_EXTERN PetscErrorCode SNESReset(SNES);
PETSC_EXTERN PetscErrorCode SNESDestroy(SNES *);
PETSC_EXTERN PetscErrorCode SNESSetType(SNES, SNESType);
PETSC_EXTERN PetscErrorCode SNESMonitor(SNES, PetscInt, PetscReal);
PETSC_EXTERN PetscErrorCode SNESMonitorSet(SNES, PetscErrorCode (*)(SNES, PetscInt, PetscReal, PetscCtx), PetscCtx, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode SNESMonitorSetFromOptions(SNES, const char[], const char[], const char[], PetscErrorCode (*)(SNES, PetscInt, PetscReal, PetscViewerAndFormat *), PetscErrorCode (*)(SNES, PetscViewerAndFormat *));
PETSC_EXTERN PetscErrorCode SNESMonitorCancel(SNES);
PETSC_EXTERN PetscErrorCode SNESMonitorSAWs(SNES, PetscInt, PetscReal, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESMonitorSAWsCreate(SNES, void **);
PETSC_EXTERN PetscErrorCode SNESMonitorSAWsDestroy(PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESSetConvergenceHistory(SNES, PetscReal[], PetscInt[], PetscInt, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetConvergenceHistory(SNES, PetscReal *[], PetscInt *[], PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetUp(SNES);
PETSC_EXTERN PetscErrorCode SNESSolve(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESSetErrorIfNotConverged(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetErrorIfNotConverged(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESConverged(SNES, PetscInt, PetscReal, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode SNESSetWorkVecs(SNES, PetscInt);

PETSC_EXTERN PetscErrorCode SNESAddOptionsChecker(PetscErrorCode (*)(SNES));

PETSC_EXTERN PetscErrorCode SNESRegister(const char[], PetscErrorCode (*)(SNES));

PETSC_EXTERN PetscErrorCode SNESGetKSP(SNES, KSP *);
PETSC_EXTERN PetscErrorCode SNESSetKSP(SNES, KSP);
PETSC_EXTERN PetscErrorCode SNESSetSolution(SNES, Vec);
PETSC_EXTERN PetscErrorCode SNESGetSolution(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESGetSolutionUpdate(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESGetRhs(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESView(SNES, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESLoad(SNES, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESConvergedReasonViewSet(SNES, PetscErrorCode (*)(SNES, PetscCtx), PetscCtx, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode SNESViewFromOptions(SNES, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode SNESConvergedReasonView(SNES, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESConvergedReasonViewFromOptions(SNES);
PETSC_EXTERN PetscErrorCode SNESConvergedReasonViewCancel(SNES);

PETSC_DEPRECATED_FUNCTION(3, 14, 0, "SNESConvergedReasonView()", ) static inline PetscErrorCode SNESReasonView(SNES snes, PetscViewer v)
{
  return SNESConvergedReasonView(snes, v);
}
PETSC_DEPRECATED_FUNCTION(3, 14, 0, "SNESConvergedReasonViewFromOptions()", ) static inline PetscErrorCode SNESReasonViewFromOptions(SNES snes)
{
  return SNESConvergedReasonViewFromOptions(snes);
}

#define SNES_FILE_CLASSID 1211224

PETSC_EXTERN PetscErrorCode SNESSetOptionsPrefix(SNES, const char[]);
PETSC_EXTERN PetscErrorCode SNESAppendOptionsPrefix(SNES, const char[]);
PETSC_EXTERN PetscErrorCode SNESGetOptionsPrefix(SNES, const char *[]);
PETSC_EXTERN PetscErrorCode SNESSetFromOptions(SNES);
PETSC_EXTERN PetscErrorCode SNESResetFromOptions(SNES);

PETSC_EXTERN PetscErrorCode SNESSetUseMatrixFree(SNES, PetscBool, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetUseMatrixFree(SNES, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode MatCreateSNESMF(SNES, Mat *);
PETSC_EXTERN PetscErrorCode MatSNESMFGetSNES(Mat, SNES *);
PETSC_EXTERN PetscErrorCode MatSNESMFSetReuseBase(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatSNESMFGetReuseBase(Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMFFDComputeJacobian(SNES, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode MatCreateSNESMFMore(SNES, Vec, Mat *);
PETSC_EXTERN PetscErrorCode MatSNESMFMoreSetParameters(Mat, PetscReal, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode SNESGetType(SNES, SNESType *);
PETSC_EXTERN PetscErrorCode SNESMonitorDefaultSetUp(SNES, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorDefault(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorScaling(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorRange(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorRatio(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorRatioSetUp(SNES, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorSolution(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorResidual(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorSolutionUpdate(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_DEPRECATED_FUNCTION(3, 26, 0, "SNESMonitorDefault()", ) static inline PetscErrorCode SNESMonitorDefaultShort(SNES snes, PetscInt n, PetscReal fgnorm, PetscViewerAndFormat *vf)
{
  return SNESMonitorDefault(snes, n, fgnorm, vf);
}
PETSC_EXTERN PetscErrorCode SNESMonitorDefaultField(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorJacUpdateSpectrum(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorFields(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode KSPMonitorSNESResidual(KSP, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode KSPMonitorSNESResidualDrawLG(KSP, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode KSPMonitorSNESResidualDrawLGCreate(PetscViewer, PetscViewerFormat, PetscCtx, PetscViewerAndFormat **);

PETSC_EXTERN PetscErrorCode SNESSetTolerances(SNES, PetscReal, PetscReal, PetscReal, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode SNESSetDivergenceTolerance(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESGetTolerances(SNES, PetscReal *, PetscReal *, PetscReal *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESGetDivergenceTolerance(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESGetForceIteration(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESSetForceIteration(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetIterationNumber(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetIterationNumber(SNES, PetscInt);

PETSC_EXTERN PetscErrorCode SNESNewtonTRSetPreCheck(SNES, PetscErrorCode (*)(SNES, Vec, Vec, PetscBool *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRGetPreCheck(SNES, PetscErrorCode (**)(SNES, Vec, Vec, PetscBool *, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetPostCheck(SNES, PetscErrorCode (*)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRGetPostCheck(SNES, PetscErrorCode (**)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetFallbackType(SNES, SNESNewtonTRFallbackType);
PETSC_EXTERN PetscErrorCode SNESNewtonTRPreCheck(SNES, Vec, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESNewtonTRPostCheck(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetNormType(SNES, NormType);

PETSC_EXTERN PetscErrorCode SNESNewtonTRSetQNType(SNES, SNESNewtonTRQNType);

PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 22, 0, "SNESNewtonTRSetTolerances()", ) PetscErrorCode SNESSetTrustRegionTolerance(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetTolerances(SNES, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode SNESNewtonTRGetTolerances(SNES, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetUpdateParameters(SNES, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode SNESNewtonTRGetUpdateParameters(SNES, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode SNESNewtonTRDCGetRhoFlag(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESNewtonTRDCSetPreCheck(SNES, PetscErrorCode (*)(SNES, Vec, Vec, PetscBool *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRDCGetPreCheck(SNES, PetscErrorCode (**)(SNES, Vec, Vec, PetscBool *, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESNewtonTRDCSetPostCheck(SNES, PetscErrorCode (*)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRDCGetPostCheck(SNES, PetscErrorCode (**)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, PetscCtx), PetscCtxRt);

PETSC_EXTERN PetscErrorCode SNESGetNonlinearStepFailures(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetMaxNonlinearStepFailures(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetMaxNonlinearStepFailures(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESGetNumberFunctionEvals(SNES, PetscInt *);

PETSC_EXTERN PetscErrorCode SNESSetLagPreconditioner(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetLagPreconditioner(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetLagJacobian(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetLagJacobian(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetLagPreconditionerPersists(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESSetLagJacobianPersists(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESSetGridSequence(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetGridSequence(SNES, PetscInt *);

PETSC_EXTERN PetscErrorCode SNESGetLinearSolveIterations(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESGetLinearSolveFailures(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetMaxLinearSolveFailures(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetMaxLinearSolveFailures(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetCountersReset(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESResetCounters(SNES);

PETSC_EXTERN PetscErrorCode SNESKSPSetUseEW(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESKSPGetUseEW(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESKSPSetParametersEW(SNES, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode SNESKSPGetParametersEW(SNES, PetscInt *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode SNESMonitorLGRange(SNES, PetscInt, PetscReal, PetscCtx);

PETSC_EXTERN PetscErrorCode SNESSetApplicationContext(SNES, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESGetApplicationContext(SNES, PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESSetComputeApplicationContext(SNES, PetscErrorCode (*)(SNES, PetscCtxRt), PetscCtxDestroyFn *);

PETSC_EXTERN PetscErrorCode SNESPythonSetType(SNES, const char[]);
PETSC_EXTERN PetscErrorCode SNESPythonGetType(SNES, const char *[]);

PETSC_EXTERN PetscErrorCode SNESSetFunctionDomainError(SNES);
PETSC_EXTERN PetscErrorCode SNESSetObjectiveDomainError(SNES);
PETSC_EXTERN PetscErrorCode SNESSetJacobianDomainError(SNES);
PETSC_EXTERN PetscErrorCode SNESSetCheckJacobianDomainError(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetCheckJacobianDomainError(SNES, PetscBool *);

/*MC
   SNES_CONVERGED_FNORM_ABS - $||F|| \le abstol$

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_CONVERGED_FNORM_RELATIVE - $||F|| \le rtol*||F(x_0)||$ where $x_0$ is the initial guess

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
  SNES_CONVERGED_SNORM_RELATIVE - The 2-norm of the last step $\le stol * ||x||$ where `x` is the current
  solution and `stol` is the 4th argument to `SNESSetTolerances()`

  Options Database Key:
  -snes_stol stol - the step tolerance

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_FUNCTION_COUNT - The user provided function has been called more times then the final
   argument to `SNESSetTolerances()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_DTOL - The norm of the function has increased by a factor of divtol set with `SNESSetDivergenceTolerance()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`, `SNESSetDivergenceTolerance()`
M*/

/*MC
   SNES_DIVERGED_FUNCTION_NANORINF - the 2-norm of the current function evaluation is not-a-number (NaN) or infinity, this
   is usually caused by a division of 0 by 0, or infinity.  See `SNESSetFunctionDomainError()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_FUNCTION_DOMAIN - the function provided with `SNESSetFunction()` called `SNESSetFunctionDomainError()` and
   the solver could not recoverer.

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_OBJECTIVE_DOMAIN - the function provided with `SNESSetObjective()` called `SNESSetObjectiveDomainError()` and
   the solver could not recoverer.

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_JACOBIAN_DOMAIN - the function provided with `SNESSetJacobian()` called `SNESSetJacobianDomainError()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_MAX_IT - SNESSolve() has reached the maximum number of iterations requested

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_LINE_SEARCH - The line search has failed. This only occurs for a `SNES` solvers that use a line search

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`, `SNESLineSearch`
M*/

/*MC
   SNES_DIVERGED_LOCAL_MIN - the algorithm seems to have stagnated at a local minimum that is not zero.
   See the manual page for `SNESConvergedReason` for more details

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_CONERGED_ITERATING - this only occurs if `SNESGetConvergedReason()` is called during the `SNESSolve()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

PETSC_EXTERN PetscErrorCode SNESSetConvergenceTest(SNES, PetscErrorCode (*)(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, PetscCtx), PetscCtx, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode SNESConvergedDefault(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESConvergedSkip(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESConvergedCorrectPressure(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESGetConvergedReason(SNES, SNESConvergedReason *);
PETSC_EXTERN PetscErrorCode SNESGetConvergedReasonString(SNES, const char *[]);
PETSC_EXTERN PetscErrorCode SNESSetConvergedReason(SNES, SNESConvergedReason);

PETSC_DEPRECATED_FUNCTION(3, 5, 0, "SNESConvergedSkip()", ) static inline void SNESSkipConverged(void)
{ /* never called */
}
#define SNESSkipConverged (SNESSkipConverged, SNESConvergedSkip)

/* --------- Solving systems of nonlinear equations --------------- */
PETSC_EXTERN PetscErrorCode SNESSetFunction(SNES, Vec, SNESFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESGetFunction(SNES, Vec *, SNESFunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESComputeFunction(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESComputeMFFunction(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESSetInitialFunction(SNES, Vec);

PETSC_EXTERN PetscErrorCode SNESSetJacobian(SNES, Mat, Mat, SNESJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESGetJacobian(SNES, Mat *, Mat *, SNESJacobianFn **, PetscCtxRt);
PETSC_EXTERN SNESFunctionFn SNESObjectiveComputeFunctionDefaultFD;
PETSC_EXTERN SNESJacobianFn SNESComputeJacobianDefault;
PETSC_EXTERN SNESJacobianFn SNESComputeJacobianDefaultColor;
PETSC_EXTERN PetscErrorCode SNESPruneJacobianColor(SNES, Mat, Mat);
PETSC_EXTERN PetscErrorCode SNESSetComputeInitialGuess(SNES, SNESInitialGuessFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESSetPicard(SNES, Vec, SNESFunctionFn *, Mat, Mat, SNESJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESGetPicard(SNES, Vec *, SNESFunctionFn **, Mat *, Mat *, SNESJacobianFn **, PetscCtxRt);
PETSC_EXTERN SNESFunctionFn SNESPicardComputeFunction;
PETSC_EXTERN SNESFunctionFn SNESPicardComputeMFFunction;
PETSC_EXTERN SNESJacobianFn SNESPicardComputeJacobian;

PETSC_EXTERN PetscErrorCode SNESSetObjective(SNES, SNESObjectiveFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESGetObjective(SNES, SNESObjectiveFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESComputeObjective(SNES, Vec, PetscReal *);

PETSC_EXTERN PetscErrorCode SNESSetUpdate(SNES, SNESUpdateFn *);

/*MC
   SNES_NORM_NONE - Don't compute function and its L2 norm when possible

   Level: advanced

   Note:
   This is most useful for stationary solvers with a fixed number of iterations used as smoothers.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_DEFAULT`
M*/

/*MC
   SNES_NORM_ALWAYS - Compute the function and its L2 norm at each iteration.

   Level: advanced

   Note:
   Most solvers will use this no matter what norm type is passed to them.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_NONE`
M*/

/*MC
   SNES_NORM_INITIAL_ONLY - Compute the function and its L2 at iteration 0, but do not update it.

   Level: advanced

   Notes:
   This method is useful in composed methods, when a true solution might actually be found before `SNESSolve()` is called.
   This option enables the solve to abort on the zeroth iteration if this is the case.

   For solvers that require the computation of the L2 norm of the function as part of the method, this merely cancels
   the norm computation at the last iteration (if possible).

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_FINAL_ONLY`, `SNES_NORM_INITIAL_FINAL_ONLY`
M*/

/*MC
   SNES_NORM_FINAL_ONLY - Compute the function and its L2 norm on only the final iteration.

   Level: advanced

   Note:
   For solvers that require the computation of the L2 norm of the function as part of the method, behaves
   exactly as `SNES_NORM_DEFAULT`.  This method is useful when the function is gotten after `SNESSolve()` and
   used in subsequent computation for methods that do not need the norm computed during the rest of the
   solution procedure.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_INITIAL_ONLY`, `SNES_NORM_INITIAL_FINAL_ONLY`
M*/

/*MC
   SNES_NORM_INITIAL_FINAL_ONLY - Compute the function and its L2 norm on only the initial and final iterations.

   Level: advanced

   Note:
   This method combines the benefits of `SNES_NORM_INITIAL_ONLY` and `SNES_NORM_FINAL_ONLY`.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_SNES_NORM_INITIAL_ONLY`, `SNES_NORM_FINAL_ONLY`
M*/

PETSC_EXTERN PetscErrorCode SNESSetNormSchedule(SNES, SNESNormSchedule);
PETSC_EXTERN PetscErrorCode SNESGetNormSchedule(SNES, SNESNormSchedule *);
PETSC_EXTERN PetscErrorCode SNESSetFunctionNorm(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESGetFunctionNorm(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESGetUpdateNorm(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESGetSolutionNorm(SNES, PetscReal *);

PETSC_EXTERN PetscErrorCode SNESSetFunctionType(SNES, SNESFunctionType);
PETSC_EXTERN PetscErrorCode SNESGetFunctionType(SNES, SNESFunctionType *);

PETSC_EXTERN PetscErrorCode SNESSetNGS(SNES, SNESNGSFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESGetNGS(SNES, SNESNGSFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESComputeNGS(SNES, Vec, Vec);

PETSC_EXTERN PetscErrorCode SNESNGSSetSweeps(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESNGSGetSweeps(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESNGSSetTolerances(SNES, PetscReal, PetscReal, PetscReal, PetscInt);
PETSC_EXTERN PetscErrorCode SNESNGSGetTolerances(SNES, PetscReal *, PetscReal *, PetscReal *, PetscInt *);

PETSC_EXTERN PetscErrorCode SNESSetAlwaysComputesFinalResidual(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetAlwaysComputesFinalResidual(SNES, PetscBool *);

PETSC_EXTERN PetscErrorCode SNESShellGetContext(SNES, PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESShellSetContext(SNES, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESShellSetSolve(SNES, PetscErrorCode (*)(SNES, Vec));

/* --------- Routines specifically for line search methods --------------- */

PETSC_EXTERN PetscFunctionList SNESList;
PETSC_EXTERN PetscClassId      SNESLINESEARCH_CLASSID;
PETSC_EXTERN PetscFunctionList SNESLineSearchList;

#define SNES_LINESEARCH_ORDER_LINEAR    1
#define SNES_LINESEARCH_ORDER_QUADRATIC 2
#define SNES_LINESEARCH_ORDER_CUBIC     3

PETSC_EXTERN PetscErrorCode SNESLineSearchCreate(MPI_Comm, SNESLineSearch *);
PETSC_EXTERN PetscErrorCode SNESLineSearchReset(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchView(SNESLineSearch, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESLineSearchDestroy(SNESLineSearch *);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetType(SNESLineSearch, SNESLineSearchType *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetType(SNESLineSearch, SNESLineSearchType);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetFromOptions(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetFunction(SNESLineSearch, PetscErrorCode (*)(SNES, Vec, Vec));
PETSC_EXTERN PetscErrorCode SNESLineSearchSetUp(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchApply(SNESLineSearch, Vec, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode SNESLineSearchPreCheck(SNESLineSearch, Vec, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESLineSearchPostCheck(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetWorkVecs(SNESLineSearch, PetscInt);

/* set the functions for precheck and postcheck */

PETSC_EXTERN PetscErrorCode SNESLineSearchSetPreCheck(SNESLineSearch, PetscErrorCode (*)(SNESLineSearch, Vec, Vec, PetscBool *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetPostCheck(SNESLineSearch, PetscErrorCode (*)(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *, PetscCtx), PetscCtx);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetPreCheck(SNESLineSearch, PetscErrorCode (**)(SNESLineSearch, Vec, Vec, PetscBool *, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetPostCheck(SNESLineSearch, PetscErrorCode (**)(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *, PetscCtx), PetscCtxRt);

/* set the functions for VI-specific line search operations */

PETSC_EXTERN PetscErrorCode SNESLineSearchSetVIFunctions(SNESLineSearch, SNESLineSearchVIProjectFn *, SNESLineSearchVINormFn *, SNESLineSearchVIDirDerivFn *);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetVIFunctions(SNESLineSearch, SNESLineSearchVIProjectFn **, SNESLineSearchVINormFn **, SNESLineSearchVIDirDerivFn **);

/* pointers to the associated SNES in order to be able to get the function evaluation out */
PETSC_EXTERN PetscErrorCode SNESLineSearchSetSNES(SNESLineSearch, SNES);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetSNES(SNESLineSearch, SNES *);

/* set and get the parameters and vectors */
PETSC_EXTERN PetscErrorCode SNESLineSearchGetTolerances(SNESLineSearch, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetTolerances(SNESLineSearch, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscInt);

PETSC_EXTERN PetscErrorCode SNESLineSearchPreCheckPicard(SNESLineSearch, Vec, Vec, PetscBool *, PetscCtx);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetLambda(SNESLineSearch, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetLambda(SNESLineSearch, PetscReal);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetDamping(SNESLineSearch, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetDamping(SNESLineSearch, PetscReal);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetOrder(SNESLineSearch, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetOrder(SNESLineSearch, PetscInt);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetReason(SNESLineSearch, SNESLineSearchReason *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetReason(SNESLineSearch, SNESLineSearchReason);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetVecs(SNESLineSearch, Vec *, Vec *, Vec *, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetVecs(SNESLineSearch, Vec, Vec, Vec, Vec, Vec);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetNorms(SNESLineSearch, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetNorms(SNESLineSearch, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode SNESLineSearchComputeNorms(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetComputeNorms(SNESLineSearch, PetscBool);

PETSC_EXTERN PetscErrorCode SNESLineSearchMonitor(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchMonitorSet(SNESLineSearch, PetscErrorCode (*)(SNESLineSearch, PetscCtx), PetscCtx, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode SNESLineSearchMonitorSetFromOptions(SNESLineSearch, const char[], const char[], const char[], PetscErrorCode (*)(SNESLineSearch, PetscViewerAndFormat *), PetscErrorCode (*)(SNESLineSearch, PetscViewerAndFormat *));
PETSC_EXTERN PetscErrorCode SNESLineSearchMonitorCancel(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetDefaultMonitor(SNESLineSearch, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetDefaultMonitor(SNESLineSearch, PetscViewer *);
PETSC_EXTERN PetscErrorCode SNESLineSearchMonitorSolutionUpdate(SNESLineSearch, PetscViewerAndFormat *);

PETSC_EXTERN PetscErrorCode SNESLineSearchAppendOptionsPrefix(SNESLineSearch, const char[]);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetOptionsPrefix(SNESLineSearch, const char *[]);

/* Shell interface functions */
PETSC_EXTERN PetscErrorCode SNESLineSearchShellSetApply(SNESLineSearch, SNESLineSearchShellApplyFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESLineSearchShellGetApply(SNESLineSearch, SNESLineSearchShellApplyFn **, PetscCtxRt);

PETSC_DEPRECATED_FUNCTION(3, 21, 0, "SNESLinesearchShellSetApply()", ) static inline PetscErrorCode SNESLineSearchShellSetUserFunc(SNESLineSearch ls, SNESLineSearchShellApplyFn *f, PetscCtx ctx)
{
  return SNESLineSearchShellSetApply(ls, f, ctx);
}

PETSC_DEPRECATED_FUNCTION(3, 21, 0, "SNESLinesearchShellGetApply()", ) static inline PetscErrorCode SNESLineSearchShellGetUserFunc(SNESLineSearch ls, SNESLineSearchShellApplyFn **f, PetscCtxRt ctx)
{
  return SNESLineSearchShellGetApply(ls, f, ctx);
}

/* BT interface functions */
PETSC_EXTERN PetscErrorCode SNESLineSearchBTSetAlpha(SNESLineSearch, PetscReal);
PETSC_EXTERN PetscErrorCode SNESLineSearchBTGetAlpha(SNESLineSearch, PetscReal *);

/*register line search types */
PETSC_EXTERN PetscErrorCode SNESLineSearchRegister(const char[], PetscErrorCode (*)(SNESLineSearch));

/* Routines for VI solver */
PETSC_EXTERN PetscErrorCode SNESVISetVariableBounds(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESVIGetVariableBounds(SNES, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode SNESVISetComputeVariableBounds(SNES, PetscErrorCode (*)(SNES, Vec, Vec));
PETSC_EXTERN PetscErrorCode SNESVIGetInactiveSet(SNES, IS *);
PETSC_EXTERN PetscErrorCode SNESVIGetActiveSetIS(SNES, Vec, Vec, IS *);
PETSC_EXTERN PetscErrorCode SNESVIComputeInactiveSetFnorm(SNES, Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESVIComputeInactiveSetFtY(SNES, Vec, Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode SNESVISetRedundancyCheck(SNES, PetscErrorCode (*)(SNES, IS, IS *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESVIComputeMeritFunction(Vec, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESVIComputeFunction(SNES, Vec, Vec, PetscCtx);
PETSC_EXTERN PetscErrorCode DMSetVI(DM, IS);
PETSC_EXTERN PetscErrorCode DMDestroyVI(DM);

PETSC_EXTERN PetscErrorCode SNESTestLocalMin(SNES);

/* Should this routine be private? */
PETSC_EXTERN PetscErrorCode SNESComputeJacobian(SNES, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode SNESTestJacobian(SNES, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESTestFunction(SNES);

PETSC_EXTERN PetscErrorCode SNESSetDM(SNES, DM);
PETSC_EXTERN PetscErrorCode SNESGetDM(SNES, DM *);
PETSC_EXTERN PetscErrorCode SNESSetNPC(SNES, SNES);
PETSC_EXTERN PetscErrorCode SNESGetNPC(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESHasNPC(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESApplyNPC(SNES, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESGetNPCFunction(SNES, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESComputeFunctionDefaultNPC(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESSetNPCSide(SNES, PCSide);
PETSC_EXTERN PetscErrorCode SNESGetNPCSide(SNES, PCSide *);
PETSC_EXTERN PetscErrorCode SNESSetLineSearch(SNES, SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESGetLineSearch(SNES, SNESLineSearch *);

PETSC_DEPRECATED_FUNCTION(3, 4, 0, "SNESGetLineSearch()", ) static inline PetscErrorCode SNESGetSNESLineSearch(SNES snes, SNESLineSearch *ls)
{
  return SNESGetLineSearch(snes, ls);
}
PETSC_DEPRECATED_FUNCTION(3, 4, 0, "SNESSetLineSearch()", ) static inline PetscErrorCode SNESSetSNESLineSearch(SNES snes, SNESLineSearch ls)
{
  return SNESSetLineSearch(snes, ls);
}

PETSC_EXTERN PetscErrorCode SNESSetUpMatrices(SNES);
PETSC_EXTERN PetscErrorCode DMSNESSetFunction(DM, SNESFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESGetFunction(DM, SNESFunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMSNESSetFunctionContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMSNESSetMFFunction(DM, SNESFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESSetNGS(DM, SNESNGSFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESGetNGS(DM, SNESNGSFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMSNESSetJacobian(DM, SNESJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESGetJacobian(DM, SNESJacobianFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMSNESSetJacobianContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMSNESSetPicard(DM, SNESFunctionFn *, SNESJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESGetPicard(DM, SNESFunctionFn **, SNESJacobianFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMSNESSetObjective(DM, SNESObjectiveFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESGetObjective(DM, SNESObjectiveFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMCopyDMSNES(DM, DM);

PETSC_EXTERN PetscErrorCode DMDASNESSetFunctionLocal(DM, InsertMode, DMDASNESFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMDASNESSetJacobianLocal(DM, DMDASNESJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMDASNESSetObjectiveLocal(DM, DMDASNESObjectiveFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMDASNESSetPicardLocal(DM, InsertMode, DMDASNESFunctionFn *, DMDASNESJacobianFn *, PetscCtx);

PETSC_EXTERN PetscErrorCode DMDASNESSetFunctionLocalVec(DM, InsertMode, DMDASNESFunctionVecFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMDASNESSetJacobianLocalVec(DM, DMDASNESJacobianVecFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMDASNESSetObjectiveLocalVec(DM, DMDASNESObjectiveVecFn *, PetscCtx);

PETSC_EXTERN PetscErrorCode DMSNESSetBoundaryLocal(DM, PetscErrorCode (*)(DM, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESSetObjectiveLocal(DM, PetscErrorCode (*)(DM, Vec, PetscReal *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESSetFunctionLocal(DM, PetscErrorCode (*)(DM, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESSetJacobianLocal(DM, PetscErrorCode (*)(DM, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESGetBoundaryLocal(DM, PetscErrorCode (**)(DM, Vec, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMSNESGetObjectiveLocal(DM, PetscErrorCode (**)(DM, Vec, PetscReal *, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMSNESGetFunctionLocal(DM, PetscErrorCode (**)(DM, Vec, Vec, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMSNESGetJacobianLocal(DM, PetscErrorCode (**)(DM, Vec, Mat, Mat, PetscCtx), PetscCtxRt);

/* Routines for Multiblock solver */
PETSC_EXTERN PetscErrorCode SNESMultiblockSetFields(SNES, const char[], PetscInt, const PetscInt *);
PETSC_EXTERN PetscErrorCode SNESMultiblockSetIS(SNES, const char[], IS);
PETSC_EXTERN PetscErrorCode SNESMultiblockSetBlockSize(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESMultiblockSetType(SNES, PCCompositeType);
PETSC_EXTERN PetscErrorCode SNESMultiblockGetSubSNES(SNES, PetscInt *, SNES *[]);

PETSC_EXTERN PetscErrorCode SNESMSRegister(SNESMSType, PetscInt, PetscInt, PetscReal, const PetscReal[], const PetscReal[], const PetscReal[]);
PETSC_EXTERN PetscErrorCode SNESMSRegisterAll(void);
PETSC_EXTERN PetscErrorCode SNESMSGetType(SNES, SNESMSType *);
PETSC_EXTERN PetscErrorCode SNESMSSetType(SNES, SNESMSType);
PETSC_EXTERN PetscErrorCode SNESMSGetDamping(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESMSSetDamping(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESMSFinalizePackage(void);
PETSC_EXTERN PetscErrorCode SNESMSInitializePackage(void);
PETSC_EXTERN PetscErrorCode SNESMSRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode SNESNGMRESSetRestartType(SNES, SNESNGMRESRestartType);
PETSC_EXTERN PetscErrorCode SNESNGMRESSetSelectType(SNES, SNESNGMRESSelectType);
PETSC_EXTERN PetscErrorCode SNESNGMRESSetRestartFmRise(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESNGMRESGetRestartFmRise(SNES, PetscBool *);

PETSC_EXTERN PetscErrorCode SNESNCGSetType(SNES, SNESNCGType);

PETSC_EXTERN PetscErrorCode SNESQNSetType(SNES, SNESQNType);
PETSC_EXTERN PetscErrorCode SNESQNSetScaleType(SNES, SNESQNScaleType);
PETSC_EXTERN PetscErrorCode SNESQNSetRestartType(SNES, SNESQNRestartType);

PETSC_EXTERN PetscErrorCode SNESNASMGetType(SNES, PCASMType *);
PETSC_EXTERN PetscErrorCode SNESNASMSetType(SNES, PCASMType);
PETSC_EXTERN PetscErrorCode SNESNASMGetSubdomains(SNES, PetscInt *, SNES *[], VecScatter *[], VecScatter *[], VecScatter *[]);
PETSC_EXTERN PetscErrorCode SNESNASMSetSubdomains(SNES, PetscInt, SNES[], VecScatter[], VecScatter[], VecScatter[]);
PETSC_EXTERN PetscErrorCode SNESNASMSetDamping(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESNASMGetDamping(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESNASMGetSubdomainVecs(SNES, PetscInt *, Vec *[], Vec *[], Vec *[], Vec *[]);
PETSC_EXTERN PetscErrorCode SNESNASMSetComputeFinalJacobian(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESNASMGetSNES(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESNASMGetNumber(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESNASMSetWeight(SNES, Vec);

PETSC_EXTERN PetscErrorCode SNESCompositeSetType(SNES, SNESCompositeType);
PETSC_EXTERN PetscErrorCode SNESCompositeAddSNES(SNES, SNESType);
PETSC_EXTERN PetscErrorCode SNESCompositeGetSNES(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESCompositeGetNumber(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESCompositeSetDamping(SNES, PetscInt, PetscReal);

PETSC_EXTERN PetscErrorCode SNESPatchSetDiscretisationInfo(SNES, PetscInt, DM[], PetscInt[], PetscInt[], const PetscInt **, const PetscInt[], PetscInt, const PetscInt[], PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode SNESPatchSetComputeOperator(SNES, PetscErrorCode (*func)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESPatchSetComputeFunction(SNES, PetscErrorCode (*func)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESPatchSetConstructType(SNES, PCPatchConstructType, PetscErrorCode (*func)(PC, PetscInt *, IS **, IS *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode SNESPatchSetCellNumbering(SNES, PetscSection);

/* called on the finest level FAS instance*/
PETSC_EXTERN PetscErrorCode SNESFASSetType(SNES, SNESFASType);
PETSC_EXTERN PetscErrorCode SNESFASGetType(SNES, SNESFASType *);
PETSC_EXTERN PetscErrorCode SNESFASSetLevels(SNES, PetscInt, MPI_Comm *);
PETSC_EXTERN PetscErrorCode SNESFASGetLevels(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESFASGetCycleSNES(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASSetNumberSmoothUp(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetNumberSmoothDown(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetCycles(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetMonitor(SNES, PetscViewerAndFormat *, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASSetLog(SNES, PetscBool);

PETSC_EXTERN PetscErrorCode SNESFASSetGalerkin(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASGetGalerkin(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESFASGalerkinFunctionDefault(SNES, Vec, Vec, PetscCtx);

/* called on any level -- "Cycle" FAS instance */
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmoother(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmootherUp(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmootherDown(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetCorrection(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetInterpolation(SNES, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetRestriction(SNES, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetInjection(SNES, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetRScale(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESFASCycleSetCycles(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASCycleIsFine(SNES, PetscBool *);

/* called on the (outer) finest level FAS to set/get parameters on any level instance */
PETSC_EXTERN PetscErrorCode SNESFASSetInterpolation(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetInterpolation(SNES, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASSetRestriction(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetRestriction(SNES, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASSetInjection(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetInjection(SNES, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASSetRScale(SNES, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode SNESFASGetRScale(SNES, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode SNESFASSetContinuation(SNES, PetscBool);

PETSC_EXTERN PetscErrorCode SNESFASGetSmoother(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASGetSmootherUp(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASGetSmootherDown(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASGetCoarseSolve(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASSetUseCoarseCorrectionLineSearch(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASGetCoarseCorrectionLineSearch(SNES, PetscInt, SNESLineSearch *);

/* parameters for full FAS */
PETSC_EXTERN PetscErrorCode SNESFASFullSetDownSweep(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASCreateCoarseVec(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESFASRestrict(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESFASFullSetTotal(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASFullGetTotal(SNES, PetscBool *);

PETSC_EXTERN PetscErrorCode DMPlexSetSNESVariableBounds(DM, SNES);
PETSC_EXTERN PetscErrorCode DMSNESCheckDiscretization(SNES, DM, PetscReal, Vec, PetscReal, PetscReal[]);
PETSC_EXTERN PetscErrorCode DMSNESCheckResidual(SNES, DM, Vec, PetscReal, PetscReal *);
PETSC_EXTERN PetscErrorCode DMSNESCheckJacobian(SNES, DM, Vec, PetscReal, PetscBool *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMSNESCheckFromOptions(SNES, Vec);
PETSC_EXTERN PetscErrorCode DMSNESComputeJacobianAction(DM, Vec, Vec, Vec, PetscCtx);
PETSC_EXTERN PetscErrorCode DMSNESCreateJacobianMF(DM, Vec, PetscCtx, Mat *);

PETSC_EXTERN PetscErrorCode SNESNewtonALSetFunction(SNES, SNESFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode SNESNewtonALGetFunction(SNES, SNESFunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode SNESNewtonALComputeFunction(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESNewtonALGetLoadParameter(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESNewtonALSetDiagonalScaling(SNES snes, Vec v);

PETSC_EXTERN PetscErrorCode SNESNewtonALSetCorrectionType(SNES, SNESNewtonALCorrectionType);
