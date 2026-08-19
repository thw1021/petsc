/*
   User interface for the timestepping package. This package
   is for use in solving time-dependent PDEs.
*/
#pragma once

#include <petscsnes.h>
#include <petscconvest.h>
#include <petsctstypes.h>

/*I <petscts.h> I*/

/* SUBMANSEC = TS */

/*MC
   TS_CONVERGED_ITERATING - this only occurs if `TSGetConvergedReason()` is called during the `TSSolve()`

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`
M*/

/*MC
   TS_CONVERGED_TIME - the final time was reached

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxTime()`, `TSGetMaxTime()`, `TSGetSolveTime()`
M*/

/*MC
   TS_CONVERGED_ITS - the maximum number of iterations (time-steps) was reached prior to the final time

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxSteps()`, `TSGetMaxSteps()`
M*/

/*MC
   TS_CONVERGED_USER - user requested termination

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`
M*/

/*MC
   TS_CONVERGED_EVENT - user requested termination on event detection

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`
M*/

/*MC
   TS_CONVERGED_PSEUDO_FRTOL - stops when function norm decreased by a set amount, used only for `TSPSEUDO`

   Options Database Key:
.   -ts_pseudo_frtol rtol - use specified rtol

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`, `TS_CONVERGED_PSEUDO_FATOL`
M*/

/*MC
   TS_CONVERGED_PSEUDO_FATOL - stops when function norm decreases below a set amount, used only for `TSPSEUDO`

   Options Database Key:
.   -ts_pseudo_fatol atol - use specified atol

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`, `TS_CONVERGED_PSEUDO_FRTOL`
M*/

/*MC
   TS_DIVERGED_NONLINEAR_SOLVE - too many nonlinear solves failed

   Level: beginner

   Note:
   See `TSSetMaxSNESFailures()` for how to allow more nonlinear solver failures.

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSGetSNES()`, `SNESGetConvergedReason()`, `TSSetMaxSNESFailures()`
M*/

/*MC
   TS_DIVERGED_STEP_REJECTED - too many steps were rejected

   Level: beginner

   Notes:
   See `TSSetMaxStepRejections()` for how to allow more step rejections.

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxStepRejections()`
M*/

/* Logging support */
PETSC_EXTERN PetscClassId TS_CLASSID;
PETSC_EXTERN PetscClassId DMTS_CLASSID;
PETSC_EXTERN PetscClassId TSADAPT_CLASSID;

PETSC_EXTERN PetscErrorCode TSInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSFinalizePackage(void);

PETSC_EXTERN PetscErrorCode TSCreate(MPI_Comm, TS *);
PETSC_EXTERN PetscErrorCode TSClone(TS, TS *);
PETSC_EXTERN PetscErrorCode TSDestroy(TS *);

PETSC_EXTERN PetscErrorCode TSSetProblemType(TS, TSProblemType);
PETSC_EXTERN PetscErrorCode TSGetProblemType(TS, TSProblemType *);
PETSC_EXTERN PetscErrorCode TSMonitor(TS, PetscInt, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSMonitorSet(TS, PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscCtx), PetscCtx, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode TSMonitorSetFromOptions(TS, const char[], const char[], const char[], PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *), PetscErrorCode (*)(TS, PetscViewerAndFormat *));
PETSC_EXTERN PetscErrorCode TSMonitorCancel(TS);

PETSC_EXTERN PetscErrorCode TSSetOptionsPrefix(TS, const char[]);
PETSC_EXTERN PetscErrorCode TSAppendOptionsPrefix(TS, const char[]);
PETSC_EXTERN PetscErrorCode TSGetOptionsPrefix(TS, const char *[]);
PETSC_EXTERN PetscErrorCode TSSetFromOptions(TS);
PETSC_EXTERN PetscErrorCode TSSetUp(TS);
PETSC_EXTERN PetscErrorCode TSReset(TS);

PETSC_EXTERN PetscErrorCode TSSetSolution(TS, Vec);
PETSC_EXTERN PetscErrorCode TSGetSolution(TS, Vec *);

PETSC_EXTERN PetscErrorCode TS2SetSolution(TS, Vec, Vec);
PETSC_EXTERN PetscErrorCode TS2GetSolution(TS, Vec *, Vec *);

PETSC_EXTERN PetscErrorCode TSGetSolutionComponents(TS, PetscInt *, Vec *);
PETSC_EXTERN PetscErrorCode TSGetAuxSolution(TS, Vec *);
PETSC_EXTERN PetscErrorCode TSGetTimeError(TS, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode TSSetTimeError(TS, Vec);

PETSC_EXTERN PetscErrorCode TSSetRHSJacobianP(TS, Mat, PetscErrorCode (*)(TS, PetscReal, Vec, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetRHSJacobianP(TS, Mat *, PetscErrorCode (**)(TS, PetscReal, Vec, Mat, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TSComputeRHSJacobianP(TS, PetscReal, Vec, Mat);
PETSC_EXTERN PetscErrorCode TSSetIJacobianP(TS, Mat, PetscErrorCode (*)(TS, PetscReal, Vec, Vec, PetscReal, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetIJacobianP(TS, Mat *, PetscErrorCode (**)(TS, PetscReal, Vec, Vec, PetscReal, Mat, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TSComputeIJacobianP(TS, PetscReal, Vec, Vec, PetscReal, Mat, PetscBool);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSGetQuadratureTS() then TSComputeRHSJacobianP()", ) PetscErrorCode TSComputeDRDPFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSGetQuadratureTS() then TSComputeRHSJacobian()", ) PetscErrorCode TSComputeDRDUFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSSetIHessianProduct(TS, Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, PetscCtx), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, PetscCtx), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, PetscCtx), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionUU(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionUP(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionPU(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionPP(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSSetRHSHessianProduct(TS, Vec[], PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, PetscCtx), Vec[], PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, PetscCtx), Vec[], PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, PetscCtx), Vec[], PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionUU(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionUP(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionPU(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionPP(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSSetCostHessianProducts(TS, PetscInt, Vec[], Vec[], Vec);
PETSC_EXTERN PetscErrorCode TSGetCostHessianProducts(TS, PetscInt *, Vec *[], Vec *[], Vec *);
PETSC_EXTERN PetscErrorCode TSComputeSNESJacobian(TS, Vec, Mat, Mat);

PETSC_EXTERN PetscFunctionList TSTrajectoryList;
PETSC_EXTERN PetscClassId      TSTRAJECTORY_CLASSID;
PETSC_EXTERN PetscBool         TSTrajectoryRegisterAllCalled;

PETSC_EXTERN PetscErrorCode TSSetSaveTrajectory(TS);
PETSC_EXTERN PetscErrorCode TSResetTrajectory(TS);
PETSC_EXTERN PetscErrorCode TSRemoveTrajectory(TS);

PETSC_EXTERN PetscErrorCode TSTrajectoryCreate(MPI_Comm, TSTrajectory *);
PETSC_EXTERN PetscErrorCode TSTrajectoryReset(TSTrajectory);
PETSC_EXTERN PetscErrorCode TSTrajectoryDestroy(TSTrajectory *);
PETSC_EXTERN PetscErrorCode TSTrajectoryView(TSTrajectory, PetscViewer);
PETSC_EXTERN PetscErrorCode TSTrajectorySetType(TSTrajectory, TS, TSTrajectoryType);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetType(TSTrajectory, TS, TSTrajectoryType *);
PETSC_EXTERN PetscErrorCode TSTrajectorySet(TSTrajectory, TS, PetscInt, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSTrajectoryGet(TSTrajectory, TS, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetVecs(TSTrajectory, TS, PetscInt, PetscReal *, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetUpdatedHistoryVecs(TSTrajectory, TS, PetscReal, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetNumSteps(TSTrajectory, PetscInt *);
PETSC_EXTERN PetscErrorCode TSTrajectoryRestoreUpdatedHistoryVecs(TSTrajectory, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TSTrajectorySetFromOptions(TSTrajectory, TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryRegister(const char[], PetscErrorCode (*)(TSTrajectory, TS));
PETSC_EXTERN PetscErrorCode TSTrajectoryRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSTrajectorySetUp(TSTrajectory, TS);
PETSC_EXTERN PetscErrorCode TSTrajectorySetUseHistory(TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMonitor(TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSTrajectorySetVariableNames(TSTrajectory, const char *const *);
PETSC_EXTERN PetscErrorCode TSTrajectorySetTransform(TSTrajectory, PetscErrorCode (*)(PetscCtx, Vec, Vec *), PetscCtxDestroyFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSTrajectorySetSolutionOnly(TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetSolutionOnly(TSTrajectory, PetscBool *);
PETSC_EXTERN PetscErrorCode TSTrajectorySetKeepFiles(TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSTrajectorySetDirname(TSTrajectory, const char[]);
PETSC_EXTERN PetscErrorCode TSTrajectorySetFiletemplate(TSTrajectory, const char[]);
PETSC_EXTERN PetscErrorCode TSGetTrajectory(TS, TSTrajectory *);

PETSC_EXTERN PetscErrorCode TSTrajectoryMemorySetType(TSTrajectory, TSTrajectoryMemoryType);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMaxCpsRAM(TSTrajectory, PetscInt);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMaxCpsDisk(TSTrajectory, PetscInt);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMaxUnitsRAM(TSTrajectory, PetscInt);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMaxUnitsDisk(TSTrajectory, PetscInt);

PETSC_EXTERN PetscErrorCode TSSetCostGradients(TS, PetscInt, Vec[], Vec[]);
PETSC_EXTERN PetscErrorCode TSGetCostGradients(TS, PetscInt *, Vec *[], Vec *[]);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 12, 0, "TSCreateQuadratureTS() and TSForwardSetSensitivities()", ) PetscErrorCode TSSetCostIntegrand(TS, PetscInt, Vec, PetscErrorCode (*)(TS, PetscReal, Vec, Vec, PetscCtx), PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, PetscCtx), PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, PetscCtx), PetscBool, PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetCostIntegral(TS, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeCostIntegrand(TS, PetscReal, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSCreateQuadratureTS(TS, PetscBool, TS *);
PETSC_EXTERN PetscErrorCode TSGetQuadratureTS(TS, PetscBool *, TS *);

PETSC_EXTERN PetscErrorCode TSAdjointSetFromOptions(TS, PetscOptionItems);
PETSC_EXTERN PetscErrorCode TSAdjointMonitor(TS, PetscInt, PetscReal, Vec, PetscInt, Vec[], Vec[]);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorSet(TS, PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, PetscCtx), PetscCtx, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorCancel(TS);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorSetFromOptions(TS, const char[], const char[], const char[], PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, PetscViewerAndFormat *), PetscErrorCode (*)(TS, PetscViewerAndFormat *));

PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSSetRHSJacobianP()", ) PetscErrorCode TSAdjointSetRHSJacobian(TS, Mat, PetscErrorCode (*)(TS, PetscReal, Vec, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSComputeRHSJacobianP()", ) PetscErrorCode TSAdjointComputeRHSJacobian(TS, PetscReal, Vec, Mat);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSGetQuadratureTS()", ) PetscErrorCode TSAdjointComputeDRDPFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSGetQuadratureTS()", ) PetscErrorCode TSAdjointComputeDRDYFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSAdjointSolve(TS);
PETSC_EXTERN PetscErrorCode TSAdjointSetSteps(TS, PetscInt);

PETSC_EXTERN PetscErrorCode TSAdjointStep(TS);
PETSC_EXTERN PetscErrorCode TSAdjointSetUp(TS);
PETSC_EXTERN PetscErrorCode TSAdjointReset(TS);
PETSC_EXTERN PetscErrorCode TSAdjointCostIntegral(TS);
PETSC_EXTERN PetscErrorCode TSAdjointSetForward(TS, Mat);
PETSC_EXTERN PetscErrorCode TSAdjointResetForward(TS);

PETSC_EXTERN PetscErrorCode TSForwardSetSensitivities(TS, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode TSForwardGetSensitivities(TS, PetscInt *, Mat *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 12, 0, "TSCreateQuadratureTS()", ) PetscErrorCode TSForwardSetIntegralGradients(TS, PetscInt, Vec[]);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 12, 0, "TSForwardGetSensitivities()", ) PetscErrorCode TSForwardGetIntegralGradients(TS, PetscInt *, Vec *[]);
PETSC_EXTERN PetscErrorCode TSForwardSetUp(TS);
PETSC_EXTERN PetscErrorCode TSForwardReset(TS);
PETSC_EXTERN PetscErrorCode TSForwardCostIntegral(TS);
PETSC_EXTERN PetscErrorCode TSForwardStep(TS);
PETSC_EXTERN PetscErrorCode TSForwardSetInitialSensitivities(TS, Mat);
PETSC_EXTERN PetscErrorCode TSForwardGetStages(TS, PetscInt *, Mat *[]);

PETSC_EXTERN PetscErrorCode TSSetMaxSteps(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSGetMaxSteps(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetRunSteps(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSGetRunSteps(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetMaxTime(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetMaxTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetExactFinalTime(TS, TSExactFinalTimeOption);
PETSC_EXTERN PetscErrorCode TSGetExactFinalTime(TS, TSExactFinalTimeOption *);
PETSC_EXTERN PetscErrorCode TSSetEvaluationTimes(TS, PetscInt, PetscReal[]);
PETSC_EXTERN PetscErrorCode TSGetEvaluationTimes(TS, PetscInt *, const PetscReal *[]);
PETSC_EXTERN PetscErrorCode TSGetEvaluationSolutions(TS, PetscInt *, const PetscReal *[], Vec *[]);
PETSC_EXTERN PetscErrorCode TSSetTimeSpan(TS, PetscInt, PetscReal[]);

/*@
  TSGetTimeSpan - gets the time span set with `TSSetTimeSpan()`

  Not Collective

  Input Parameter:
. ts - the time-stepper

  Output Parameters:
+ n          - number of the time points (>=2)
- span_times - array of the time points. The first element and the last element are the initial time and the final time respectively.

  Level: deprecated

  Note:
  Deprecated, use `TSGetEvaluationTimes()`.

  The values obtained are valid until the `TS` object is destroyed.

  Both `n` and `span_times` can be `NULL`.

.seealso: [](ch_ts), `TS`, `TSGetEvaluationTimes()`, `TSSetTimeSpan()`, `TSSetEvaluationTimes()`, `TSGetEvaluationSolutions()`
 @*/
PETSC_DEPRECATED_FUNCTION(3, 23, 0, "TSGetEvaluationTimes()", ) static inline PetscErrorCode TSGetTimeSpan(TS ts, PetscInt *n, const PetscReal *span_times[])
{
  return TSGetEvaluationTimes(ts, n, span_times);
}

/*@
  TSGetTimeSpanSolutions - Get the number of solutions and the solutions at the time points specified by the time span.

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameters:
+ nsol - the number of solutions
- Sols - the solution vectors

  Level: deprecated

  Notes:
  Deprecated, use `TSGetEvaluationSolutions()`.

  Both `nsol` and `Sols` can be `NULL`.

  Some time points in the time span may be skipped by `TS` so that `nsol` is less than the number of points specified by `TSSetTimeSpan()`.
  For example, manipulating the step size, especially with a reduced precision, may cause `TS` to step over certain points in the span.
  This issue is alleviated in `TSGetEvaluationSolutions()` by returning the solution times that `Sols` were recorded at.

.seealso: [](ch_ts), `TS`, `TSGetEvaluationSolutions()`, `TSSetTimeSpan()`, `TSGetEvaluationTimes()`, `TSSetEvaluationTimes()`
 @*/
PETSC_DEPRECATED_FUNCTION(3, 23, 0, "TSGetEvaluationSolutions()", ) static inline PetscErrorCode TSGetTimeSpanSolutions(TS ts, PetscInt *nsol, Vec **Sols)
{
  return TSGetEvaluationSolutions(ts, nsol, NULL, Sols);
}

PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSSetTime()", ) PetscErrorCode TSSetInitialTimeStep(TS, PetscReal, PetscReal);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSSetMax()", ) PetscErrorCode TSSetDuration(TS, PetscInt, PetscReal);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSGetMax()", ) PetscErrorCode TSGetDuration(TS, PetscInt *, PetscReal *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSGetStepNumber()", ) PetscErrorCode TSGetTimeStepNumber(TS, PetscInt *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSGetStepNumber()", ) PetscErrorCode TSGetTotalSteps(TS, PetscInt *);

PETSC_EXTERN PetscErrorCode TSMonitorDefault(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorWallClockTime(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorWallClockTimeSetUp(TS, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorExtreme(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);

PETSC_EXTERN PetscErrorCode TSMonitorDrawCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TSMonitorDrawCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorDrawCtxDestroy(TSMonitorDrawCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorDrawSolution(TS, PetscInt, PetscReal, Vec, PetscCtx);
PETSC_EXTERN PetscErrorCode TSMonitorDrawSolutionPhase(TS, PetscInt, PetscReal, Vec, PetscCtx);
PETSC_EXTERN PetscErrorCode TSMonitorDrawError(TS, PetscInt, PetscReal, Vec, PetscCtx);
PETSC_EXTERN PetscErrorCode TSMonitorDrawSolutionFunction(TS, PetscInt, PetscReal, Vec, PetscCtx);

PETSC_EXTERN PetscErrorCode TSAdjointMonitorDefault(TS, PetscInt, PetscReal, Vec, PetscInt, Vec[], Vec[], PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorDrawSensi(TS, PetscInt, PetscReal, Vec, PetscInt, Vec[], Vec[], PetscCtx);

PETSC_EXTERN PetscErrorCode TSMonitorSolution(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionSetup(TS, PetscViewerAndFormat *);

PETSC_EXTERN PetscErrorCode TSMonitorSolutionVTK(TS, PetscInt, PetscReal, Vec, TSMonitorVTKCtx);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionVTKDestroy(TSMonitorVTKCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionVTKCtxCreate(const char *, TSMonitorVTKCtx *);

PETSC_EXTERN PetscErrorCode TSStep(TS);
PETSC_EXTERN PetscErrorCode TSEvaluateWLTE(TS, NormType, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSEvaluateStep(TS, PetscInt, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode TSSolve(TS, Vec);
PETSC_EXTERN PetscErrorCode TSGetEquationType(TS, TSEquationType *);
PETSC_EXTERN PetscErrorCode TSSetEquationType(TS, TSEquationType);
PETSC_EXTERN PetscErrorCode TSGetConvergedReason(TS, TSConvergedReason *);
PETSC_EXTERN PetscErrorCode TSSetConvergedReason(TS, TSConvergedReason);
PETSC_EXTERN PetscErrorCode TSGetSolveTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSGetSNESIterations(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSGetKSPIterations(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSGetStepRejections(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetMaxStepRejections(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSGetSNESFailures(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetMaxSNESFailures(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSSetErrorIfStepFails(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSRestartStep(TS);
PETSC_EXTERN PetscErrorCode TSRollBack(TS);
PETSC_EXTERN PetscErrorCode TSGetStepRollBack(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode TSGetStages(TS, PetscInt *, Vec *[]);

PETSC_EXTERN PetscErrorCode TSGetTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetTime(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetPrevTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSGetTimeStep(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetStepNumber(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetStepNumber(TS, PetscInt);

PETSC_EXTERN PetscErrorCode TSSetRHSFunction(TS, Vec, TSRHSFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetRHSFunction(TS, Vec *, TSRHSFunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode TSSetRHSJacobian(TS, Mat, Mat, TSRHSJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetRHSJacobian(TS, Mat *, Mat *, TSRHSJacobianFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode TSRHSJacobianSetReuse(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSSetSolutionFunction(TS, TSSolutionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSSetForcingFunction(TS, TSForcingFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSSetIFunction(TS, Vec, TSIFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetIFunction(TS, Vec *, TSIFunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode TSSetIJacobian(TS, Mat, Mat, TSIJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetIJacobian(TS, Mat *, Mat *, TSIJacobianFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode TSSetI2Function(TS, Vec, TSI2FunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetI2Function(TS, Vec *, TSI2FunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode TSSetI2Jacobian(TS, Mat, Mat, TSI2JacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetI2Jacobian(TS, Mat *, Mat *, TSI2JacobianFn **, PetscCtxRt);

PETSC_EXTERN PetscErrorCode TSRHSSplitSetIS(TS, const char[], IS);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetIS(TS, const char[], IS *);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetRHSFunction(TS, const char[], Vec, TSRHSFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetIFunction(TS, const char[], Vec, TSIFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetIJacobian(TS, const char[], Mat, Mat, TSIJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetSubTS(TS, const char[], TS *);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetSubTSs(TS, PetscInt *, TS *[]);
PETSC_EXTERN PetscErrorCode TSSetUseSplitRHSFunction(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSGetUseSplitRHSFunction(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetSNES(TS, SNES *);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetSNES(TS, SNES);

PETSC_EXTERN TSRHSFunctionFn TSComputeRHSFunctionLinear;
PETSC_EXTERN TSRHSJacobianFn TSComputeRHSJacobianConstant;
PETSC_EXTERN PetscErrorCode  TSComputeIFunctionLinear(TS, PetscReal, Vec, Vec, Vec, PetscCtx);
PETSC_EXTERN PetscErrorCode  TSComputeIJacobianConstant(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, PetscCtx);
PETSC_EXTERN PetscErrorCode  TSComputeSolutionFunction(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode  TSComputeForcingFunction(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode  TSComputeIJacobianDefaultColor(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, PetscCtx);
PETSC_EXTERN PetscErrorCode  TSPruneIJacobianColor(TS, Mat, Mat);

PETSC_EXTERN PetscErrorCode TSSetPreStep(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetPreStage(TS, PetscErrorCode (*)(TS, PetscReal));
PETSC_EXTERN PetscErrorCode TSSetPostStage(TS, PetscErrorCode (*)(TS, PetscReal, PetscInt, Vec *));
PETSC_EXTERN PetscErrorCode TSSetPostEvaluate(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetPostStep(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetResize(TS, PetscBool, PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscBool *, PetscCtx), PetscErrorCode (*)(TS, PetscInt, Vec[], Vec[], PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TSPreStep(TS);
PETSC_EXTERN PetscErrorCode TSPreStage(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSPostStage(TS, PetscReal, PetscInt, Vec[]);
PETSC_EXTERN PetscErrorCode TSPostEvaluate(TS);
PETSC_EXTERN PetscErrorCode TSPostStep(TS);
PETSC_EXTERN PetscErrorCode TSResize(TS);
PETSC_EXTERN PetscErrorCode TSResizeRetrieveVec(TS, const char *, Vec *);
PETSC_EXTERN PetscErrorCode TSResizeRegisterVec(TS, const char *, Vec);
PETSC_EXTERN PetscErrorCode TSGetStepResize(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode TSInterpolate(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSSetTolerances(TS, PetscReal, Vec, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSGetTolerances(TS, PetscReal *, Vec *, PetscReal *, Vec *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedNorm(TS, Vec, Vec, NormType, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedENorm(TS, Vec, Vec, Vec, NormType, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetCFLTimeLocal(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetCFLTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetFunctionDomainError(TS, PetscErrorCode (*)(TS, PetscReal, Vec, PetscBool *));
PETSC_EXTERN PetscErrorCode TSFunctionDomainError(TS, PetscReal, Vec, PetscBool *);

PETSC_EXTERN PetscErrorCode TSPseudoSetTimeStep(TS, PetscErrorCode (*)(TS, PetscReal *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TSPseudoTimeStepDefault(TS, PetscReal *, PetscCtx);
PETSC_EXTERN PetscErrorCode TSPseudoSetMaxTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSPseudoSetVerifyTimeStep(TS, PetscErrorCode (*)(TS, Vec, PetscCtx, PetscReal *, PetscBool *), PetscCtx);
PETSC_EXTERN PetscErrorCode TSPseudoSetTimeStepIncrement(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSPseudoIncrementDtFromInitialDt(TS);
PETSC_EXTERN PetscErrorCode TSPseudoComputeFunction(TS, Vec, Vec *, PetscReal *);

PETSC_EXTERN PetscErrorCode TSPythonSetType(TS, const char[]);
PETSC_EXTERN PetscErrorCode TSPythonGetType(TS, const char *[]);

PETSC_EXTERN PetscErrorCode TSComputeRHSFunction(TS, PetscReal, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSComputeRHSJacobian(TS, PetscReal, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TSComputeIFunction(TS, PetscReal, Vec, Vec, Vec, PetscBool);
PETSC_EXTERN PetscErrorCode TSComputeIJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, PetscBool);
PETSC_EXTERN PetscErrorCode TSComputeI2Function(TS, PetscReal, Vec, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSComputeI2Jacobian(TS, PetscReal, Vec, Vec, Vec, PetscReal, PetscReal, Mat, Mat);
PETSC_EXTERN PetscErrorCode TSComputeLinearStability(TS, PetscReal, PetscReal, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode TSVISetVariableBounds(TS, Vec, Vec);

PETSC_EXTERN PetscErrorCode DMTSSetBoundaryLocal(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSSetIFunctionPre(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSSetRHSFunction(DM, TSRHSFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetRHSFunction(DM, TSRHSFunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetRHSFunctionContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetRHSJacobian(DM, TSRHSJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetRHSJacobian(DM, TSRHSJacobianFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetRHSJacobianContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetIFunction(DM, TSIFunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetIFunction(DM, TSIFunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetIFunctionContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetIJacobian(DM, TSIJacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetIJacobian(DM, TSIJacobianFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetIJacobianContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetI2Function(DM, TSI2FunctionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetI2Function(DM, TSI2FunctionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetI2FunctionContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetI2Jacobian(DM, TSI2JacobianFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetI2Jacobian(DM, TSI2JacobianFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetI2JacobianContextDestroy(DM, PetscCtxDestroyFn *);

PETSC_EXTERN PetscErrorCode TSSetTransientVariable(TS, TSTransientVariableFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSSetTransientVariable(DM, TSTransientVariableFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetTransientVariable(DM, TSTransientVariableFn **, PetscCtx);
PETSC_EXTERN PetscErrorCode TSComputeTransientVariable(TS, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSHasTransientVariable(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode DMTSSetSolutionFunction(DM, TSSolutionFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetSolutionFunction(DM, TSSolutionFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetForcingFunction(DM, TSForcingFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetForcingFunction(DM, TSForcingFn **, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSCheckResidual(TS, DM, PetscReal, Vec, Vec, PetscReal, PetscReal *);
PETSC_EXTERN PetscErrorCode DMTSCheckJacobian(TS, DM, PetscReal, Vec, Vec, PetscReal, PetscBool *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMTSCheckFromOptions(TS, Vec);

PETSC_EXTERN PetscErrorCode DMTSGetIFunctionLocal(DM, PetscErrorCode (**)(DM, PetscReal, Vec, Vec, Vec, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetIFunctionLocal(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetIJacobianLocal(DM, PetscErrorCode (**)(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetIJacobianLocal(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSGetRHSFunctionLocal(DM, PetscErrorCode (**)(DM, PetscReal, Vec, Vec, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMTSSetRHSFunctionLocal(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode DMTSCreateRHSMassMatrix(DM);
PETSC_EXTERN PetscErrorCode DMTSCreateRHSMassMatrixLumped(DM);
PETSC_EXTERN PetscErrorCode DMTSDestroyRHSMassMatrix(DM);

PETSC_EXTERN PetscErrorCode DMTSSetIFunctionSerialize(DM, PetscErrorCode (*)(PetscCtx, PetscViewer), PetscErrorCode (*)(PetscCtx *, PetscViewer));
PETSC_EXTERN PetscErrorCode DMTSSetIJacobianSerialize(DM, PetscErrorCode (*)(PetscCtx, PetscViewer), PetscErrorCode (*)(PetscCtx *, PetscViewer));

PETSC_EXTERN PetscErrorCode DMDATSSetRHSFunctionLocal(DM, InsertMode, DMDATSRHSFunctionLocalFn *, void *);
PETSC_EXTERN PetscErrorCode DMDATSSetRHSJacobianLocal(DM, DMDATSRHSJacobianLocalFn *, void *);
PETSC_EXTERN PetscErrorCode DMDATSSetIFunctionLocal(DM, InsertMode, DMDATSIFunctionLocalFn *, void *);
PETSC_EXTERN PetscErrorCode DMDATSSetIJacobianLocal(DM, DMDATSIJacobianLocalFn *, void *);

PETSC_EXTERN PetscErrorCode TSMonitorDMDARayDestroy(PetscCtxRt);
PETSC_EXTERN PetscErrorCode TSMonitorDMDARay(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGDMDARay(TS, PetscInt, PetscReal, Vec, void *);

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscFunctionList TSList;
PETSC_EXTERN PetscErrorCode    TSGetType(TS, TSType *);
PETSC_EXTERN PetscErrorCode    TSSetType(TS, TSType);
PETSC_EXTERN PetscErrorCode    TSRegister(const char[], PetscErrorCode (*)(TS));

PETSC_EXTERN PetscErrorCode TSGetSNES(TS, SNES *);
PETSC_EXTERN PetscErrorCode TSSetSNES(TS, SNES);
PETSC_EXTERN PetscErrorCode TSGetKSP(TS, KSP *);
PETSC_EXTERN PetscErrorCode TSIsImplicit(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode TSView(TS, PetscViewer);
PETSC_EXTERN PetscErrorCode TSLoad(TS, PetscViewer);
PETSC_EXTERN PetscErrorCode TSViewFromOptions(TS, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode TSTrajectoryViewFromOptions(TSTrajectory, PetscObject, const char[]);

#define TS_FILE_CLASSID 1211225

PETSC_EXTERN PetscErrorCode TSSetApplicationContext(TS, PetscCtx);
PETSC_EXTERN PetscErrorCode TSGetApplicationContext(TS, PetscCtxRt);

PETSC_EXTERN PetscErrorCode TSMonitorLGCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TSMonitorLGCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxDestroy(TSMonitorLGCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorLGTimeStep(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSolution(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSetVariableNames(TS, const char *const *);
PETSC_EXTERN PetscErrorCode TSMonitorLGGetVariableNames(TS, const char *const **);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxSetVariableNames(TSMonitorLGCtx, const char *const *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSetDisplayVariables(TS, const char *const *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxSetDisplayVariables(TSMonitorLGCtx, const char *const *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSetTransform(TS, PetscErrorCode (*)(void *, Vec, Vec *), PetscCtxDestroyFn *, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxSetTransform(TSMonitorLGCtx, PetscErrorCode (*)(void *, Vec, Vec *), PetscCtxDestroyFn *, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGError(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSNESIterations(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGKSPIterations(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorError(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSDMSwarmMonitorMoments(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);

struct _n_TSMonitorLGCtxNetwork {
  PetscInt     nlg;
  PetscDrawLG *lg;
  PetscBool    semilogy;
  PetscInt     howoften; /* when > 0 uses step % howoften, when negative only final solution plotted */
};

PETSC_EXTERN PetscErrorCode TSMonitorLGCtxNetworkDestroy(TSMonitorLGCtxNetwork *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxNetworkCreate(TS, const char[], const char[], int, int, int, int, PetscInt, TSMonitorLGCtxNetwork *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxNetworkSolution(TS, PetscInt, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode TSMonitorEnvelopeCtxCreate(TS, TSMonitorEnvelopeCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorEnvelope(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorEnvelopeGetBounds(TS, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TSMonitorEnvelopeCtxDestroy(TSMonitorEnvelopeCtx *);

PETSC_EXTERN PetscErrorCode TSMonitorSPEigCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TSMonitorSPEigCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorSPEigCtxDestroy(TSMonitorSPEigCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorSPEig(TS, PetscInt, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode TSMonitorSPCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, PetscInt, PetscBool, PetscBool, TSMonitorSPCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorSPCtxDestroy(TSMonitorSPCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorSPSwarmSolution(TS, PetscInt, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode TSMonitorHGCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, PetscInt, PetscInt, PetscBool, TSMonitorHGCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorHGSwarmSolution(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorHGCtxDestroy(TSMonitorHGCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorHGSwarmSolution(TS, PetscInt, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode TSSetEventHandler(TS, PetscInt, PetscInt[], PetscBool[], PetscErrorCode (*)(TS, PetscReal, Vec, PetscReal[], void *), PetscErrorCode (*)(TS, PetscInt, PetscInt[], PetscReal, Vec, PetscBool, void *), void *);
PETSC_EXTERN PetscErrorCode TSSetPostEventStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSetPostEventSecondStep(TS, PetscReal);
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TSSetPostEventSecondStep()", ) static inline PetscErrorCode TSSetPostEventIntervalStep(TS ts, PetscReal dt)
{
  return TSSetPostEventSecondStep(ts, dt);
}
PETSC_EXTERN PetscErrorCode TSSetEventTolerances(TS, PetscReal, PetscReal[]);
PETSC_EXTERN PetscErrorCode TSGetNumEvents(TS, PetscInt *);

PETSC_EXTERN PetscErrorCode    TSSSPSetType(TS, TSSSPType);
PETSC_EXTERN PetscErrorCode    TSSSPGetType(TS, TSSSPType *);
PETSC_EXTERN PetscErrorCode    TSSSPSetNumStages(TS, PetscInt);
PETSC_EXTERN PetscErrorCode    TSSSPGetNumStages(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode    TSSSPInitializePackage(void);
PETSC_EXTERN PetscErrorCode    TSSSPFinalizePackage(void);
PETSC_EXTERN PetscFunctionList TSSSPList;

PETSC_EXTERN PetscErrorCode TSGetAdapt(TS, TSAdapt *);
PETSC_EXTERN PetscErrorCode TSAdaptRegister(const char[], PetscErrorCode (*)(TSAdapt));
PETSC_EXTERN PetscErrorCode TSAdaptInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSAdaptFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSAdaptCreate(MPI_Comm, TSAdapt *);
PETSC_EXTERN PetscErrorCode TSAdaptSetType(TSAdapt, TSAdaptType);
PETSC_EXTERN PetscErrorCode TSAdaptGetType(TSAdapt, TSAdaptType *);
PETSC_EXTERN PetscErrorCode TSAdaptSetOptionsPrefix(TSAdapt, const char[]);
PETSC_EXTERN PetscErrorCode TSAdaptCandidatesClear(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCandidateAdd(TSAdapt, const char[], PetscInt, PetscInt, PetscReal, PetscReal, PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptCandidatesGet(TSAdapt, PetscInt *, const PetscInt **, const PetscInt **, const PetscReal **, const PetscReal **);
PETSC_EXTERN PetscErrorCode TSAdaptChoose(TSAdapt, TS, PetscReal, PetscInt *, PetscReal *, PetscBool *);
PETSC_EXTERN PetscErrorCode TSAdaptCheckStage(TSAdapt, TS, PetscReal, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode TSAdaptView(TSAdapt, PetscViewer);
PETSC_EXTERN PetscErrorCode TSAdaptLoad(TSAdapt, PetscViewer);
PETSC_EXTERN PetscErrorCode TSAdaptSetFromOptions(TSAdapt, PetscOptionItems);
PETSC_EXTERN PetscErrorCode TSAdaptReset(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptDestroy(TSAdapt *);
PETSC_EXTERN PetscErrorCode TSAdaptSetMonitor(TSAdapt, PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptSetAlwaysAccept(TSAdapt, PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptSetSafety(TSAdapt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetSafety(TSAdapt, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetMaxIgnore(TSAdapt, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetMaxIgnore(TSAdapt, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetClip(TSAdapt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetClip(TSAdapt, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetScaleSolveFailed(TSAdapt, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetScaleSolveFailed(TSAdapt, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetStepLimits(TSAdapt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetStepLimits(TSAdapt, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetCheckStage(TSAdapt, PetscErrorCode (*)(TSAdapt, TS, PetscReal, Vec, PetscBool *));
PETSC_EXTERN PetscErrorCode TSAdaptHistorySetHistory(TSAdapt, PetscInt, PetscReal[], PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptHistorySetTrajectory(TSAdapt, TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptHistoryGetStep(TSAdapt, PetscInt, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetTimeStepIncreaseDelay(TSAdapt, PetscInt);
PETSC_EXTERN PetscErrorCode TSAdaptDSPSetFilter(TSAdapt, const char *);
PETSC_EXTERN PetscErrorCode TSAdaptDSPSetPID(TSAdapt, PetscReal, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode TSGLLEAdaptRegister(const char[], PetscErrorCode (*)(TSGLLEAdapt));
PETSC_EXTERN PetscErrorCode TSGLLEAdaptInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptCreate(MPI_Comm, TSGLLEAdapt *);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptSetType(TSGLLEAdapt, TSGLLEAdaptType);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptSetOptionsPrefix(TSGLLEAdapt, const char[]);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptChoose(TSGLLEAdapt, PetscInt, const PetscInt[], const PetscReal[], const PetscReal[], PetscInt, PetscReal, PetscReal, PetscInt *, PetscReal *, PetscBool *);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptView(TSGLLEAdapt, PetscViewer);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptSetFromOptions(TSGLLEAdapt, PetscOptionItems);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptDestroy(TSGLLEAdapt *);

PETSC_EXTERN PetscErrorCode TSGLLEAcceptRegister(const char[], TSGLLEAcceptFn *);

PETSC_EXTERN PetscErrorCode TSGLLERegister(const char[], PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSGLLEInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSGLLEFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLLESetType(TS, TSGLLEType);
PETSC_EXTERN PetscErrorCode TSGLLEGetAdapt(TS, TSGLLEAdapt *);
PETSC_EXTERN PetscErrorCode TSGLLESetAcceptType(TS, TSGLLEAcceptType);

PETSC_EXTERN PetscErrorCode TSEIMEXSetMaxRows(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSEIMEXSetRowCol(TS, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TSEIMEXSetOrdAdapt(TS, PetscBool);

PETSC_EXTERN PetscErrorCode TSRKGetOrder(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSRKGetType(TS, TSRKType *);
PETSC_EXTERN PetscErrorCode TSRKSetType(TS, TSRKType);
PETSC_EXTERN PetscErrorCode TSRKGetTableau(TS, PetscInt *, const PetscReal *[], const PetscReal *[], const PetscReal *[], const PetscReal *[], PetscInt *, const PetscReal *[], PetscBool *);
PETSC_EXTERN PetscErrorCode TSRKSetMultirate(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSRKGetMultirate(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSRKRegister(TSRKType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSRKInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSRKFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSRKRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TSMPRKGetType(TS, TSMPRKType *);
PETSC_EXTERN PetscErrorCode TSMPRKSetType(TS, TSMPRKType);
PETSC_EXTERN PetscErrorCode TSMPRKRegister(TSMPRKType, PetscInt, PetscInt, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscInt[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscInt[], const PetscReal[], const PetscReal[], const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSMPRKInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSMPRKFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSMPRKRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TSIRKGetType(TS, TSIRKType *);
PETSC_EXTERN PetscErrorCode TSIRKSetType(TS, TSIRKType);
PETSC_EXTERN PetscErrorCode TSIRKGetNumStages(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSIRKSetNumStages(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSIRKRegister(const char[], PetscErrorCode (*function)(TS));
PETSC_EXTERN PetscErrorCode TSIRKTableauCreate(TS, PetscInt, const PetscReal *, const PetscReal *, const PetscReal *, const PetscReal *, const PetscScalar *, const PetscScalar *, const PetscScalar *);
PETSC_EXTERN PetscErrorCode TSIRKInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSIRKFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSIRKRegisterDestroy(void);

/*J
   TSGLEEMode - String with the mode of error estimation for a General Linear with Error Estimation `TSGLEE` type

   Level: beginner

.seealso: [](ch_ts), `TSGLEESetMode()`, `TS`, `TSGLEE`, `TSGLEERegister()`
J*/
PETSC_EXTERN PetscErrorCode TSGLEEGetType(TS, TSGLEEType *);
PETSC_EXTERN PetscErrorCode TSGLEESetType(TS, TSGLEEType);
PETSC_EXTERN PetscErrorCode TSGLEERegister(TSGLEEType, PetscInt, PetscInt, PetscInt, PetscReal, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSGLEERegisterAll(void);
PETSC_EXTERN PetscErrorCode TSGLEEFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLEEInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSGLEERegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TSARKIMEXGetType(TS, TSARKIMEXType *);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetType(TS, TSARKIMEXType);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetFullyImplicit(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSARKIMEXGetFullyImplicit(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetFastSlowSplit(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSARKIMEXGetFastSlowSplit(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegister(TSARKIMEXType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[], const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSARKIMEXInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TSDIRKGetType(TS, TSDIRKType *);
PETSC_EXTERN PetscErrorCode TSDIRKSetType(TS, TSDIRKType);
PETSC_EXTERN PetscErrorCode TSDIRKRegister(TSDIRKType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);

PETSC_EXTERN PetscErrorCode TSRosWGetType(TS, TSRosWType *);
PETSC_EXTERN PetscErrorCode TSRosWSetType(TS, TSRosWType);
PETSC_EXTERN PetscErrorCode TSRosWSetRecomputeJacobian(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSRosWRegister(TSRosWType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSRosWRegisterRos4(TSRosWType, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSRosWInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSRosWFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSRosWRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TSBDFSetOrder(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSBDFGetOrder(TS, PetscInt *);

PETSC_EXTERN PetscErrorCode TSBasicSymplecticSetType(TS, TSBasicSymplecticType);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticGetType(TS, TSBasicSymplecticType *);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticRegister(TSBasicSymplecticType, PetscInt, PetscInt, PetscReal[], PetscReal[]);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TSDiscGradSetFormulation(TS, PetscErrorCode (*)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (*)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (*)(TS, PetscReal, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TSDiscGradGetFormulation(TS, PetscErrorCode (**)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (**)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (**)(TS, PetscReal, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TSDiscGradSetImplicitFormulation(TS, PetscErrorCode (*)(TS, PetscReal, Vec, Vec, Vec, void *), PetscErrorCode (*)(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *));
PETSC_EXTERN PetscErrorCode TSDiscGradSetType(TS, TSDGType);
PETSC_EXTERN PetscErrorCode TSDiscGradGetType(TS, TSDGType *);
PETSC_EXTERN PetscErrorCode TSDiscGradGetX0AndXdot(TS, DM, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TSDiscGradRestoreX0AndXdot(TS, DM, Vec *, Vec *);

/*
       PETSc interface to Sundials
*/
#if PetscDefined(HAVE_SUNDIALS2)
PETSC_EXTERN PetscErrorCode TSSundialsSetType(TS, TSSundialsLmmType);
PETSC_EXTERN PetscErrorCode TSSundialsGetPC(TS, PC *);
PETSC_EXTERN PetscErrorCode TSSundialsSetTolerance(TS, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsSetMinTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsSetMaxTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsGetIterations(TS, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSundialsSetGramSchmidtType(TS, TSSundialsGramSchmidtType);
PETSC_EXTERN PetscErrorCode TSSundialsSetLinearTolerance(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsMonitorInternalSteps(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSSundialsSetMaxl(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSSundialsSetMaxord(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSSundialsSetUseDense(TS, PetscBool);
#endif

PETSC_EXTERN PetscErrorCode TSThetaSetTheta(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSThetaGetTheta(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSThetaGetEndpoint(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSThetaSetEndpoint(TS, PetscBool);

PETSC_EXTERN PetscErrorCode TSAlphaSetRadius(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSAlphaSetParams(TS, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAlphaGetParams(TS, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode TSAlpha2SetRadius(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSAlpha2SetParams(TS, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAlpha2GetParams(TS, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode TSAlpha2SetPredictor(TS, TSAlpha2PredictorFn *, void *);

PETSC_EXTERN PetscErrorCode TSSetDM(TS, DM);
PETSC_EXTERN PetscErrorCode TSGetDM(TS, DM *);

PETSC_EXTERN PetscErrorCode SNESTSFormFunction(SNES, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode SNESTSFormJacobian(SNES, Vec, Mat, Mat, void *);

PETSC_EXTERN PetscErrorCode TSRHSJacobianTest(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSRHSJacobianTestTranspose(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode TSGetComputeInitialCondition(TS, PetscErrorCode (**)(TS, Vec));
PETSC_EXTERN PetscErrorCode TSSetComputeInitialCondition(TS, PetscErrorCode (*)(TS, Vec));
PETSC_EXTERN PetscErrorCode TSComputeInitialCondition(TS, Vec);
PETSC_EXTERN PetscErrorCode TSGetComputeExactError(TS, PetscErrorCode (**)(TS, Vec, Vec));
PETSC_EXTERN PetscErrorCode TSSetComputeExactError(TS, PetscErrorCode (*)(TS, Vec, Vec));
PETSC_EXTERN PetscErrorCode TSComputeExactError(TS, Vec, Vec);
PETSC_EXTERN PetscErrorCode PetscConvEstUseTS(PetscConvEst, PetscBool);

PETSC_EXTERN PetscErrorCode TSSetMatStructure(TS, MatStructure);
