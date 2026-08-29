#pragma once

#include <petscsnes.h>
#include <petsctaotypes.h>
#include <petsctaoterm.h>

/* SUBMANSEC = Tao */

PETSC_EXTERN PetscErrorCode MatDSFischer(Mat, Vec, Vec, Vec, Vec, PetscReal, Vec, Vec, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoSoftThreshold(Vec, PetscReal, PetscReal, Vec);

PETSC_EXTERN PetscClassId      TAO_CLASSID;
PETSC_EXTERN PetscFunctionList TaoList;

PETSC_EXTERN PetscErrorCode TaoInitializePackage(void);
PETSC_EXTERN PetscErrorCode TaoFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TaoCreate(MPI_Comm, Tao *);
PETSC_EXTERN PetscErrorCode TaoSetFromOptions(Tao);
PETSC_EXTERN PetscErrorCode TaoSetUp(Tao);
PETSC_EXTERN PetscErrorCode TaoSetType(Tao, TaoType);
PETSC_EXTERN PetscErrorCode TaoGetType(Tao, TaoType *);
PETSC_EXTERN PetscErrorCode TaoSetApplicationContext(Tao, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetApplicationContext(Tao, PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoDestroy(Tao *);
PETSC_EXTERN PetscErrorCode TaoParametersInitialize(Tao);

PETSC_EXTERN PetscErrorCode TaoSetOptionsPrefix(Tao, const char[]);
PETSC_EXTERN PetscErrorCode TaoView(Tao, PetscViewer);
PETSC_EXTERN PetscErrorCode TaoViewFromOptions(Tao, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode TaoSolve(Tao);

PETSC_EXTERN PetscErrorCode TaoRegister(const char[], PetscErrorCode (*)(Tao));
PETSC_EXTERN PetscErrorCode TaoRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TaoGetConvergedReason(Tao, TaoConvergedReason *);
PETSC_EXTERN PetscErrorCode TaoGetConvergedReasonString(Tao, const char *[]);
PETSC_EXTERN PetscErrorCode TaoGetSolutionStatus(Tao, PetscInt *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, TaoConvergedReason *);
PETSC_EXTERN PetscErrorCode TaoSetConvergedReason(Tao, TaoConvergedReason);
PETSC_EXTERN PetscErrorCode TaoSetSolution(Tao, Vec);
PETSC_EXTERN PetscErrorCode TaoGetSolution(Tao, Vec *);

PETSC_EXTERN PetscErrorCode TaoSetObjective(Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetObjective(Tao, PetscErrorCode (**)(Tao, Vec, PetscReal *, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoSetGradient(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetGradient(Tao, Vec *, PetscErrorCode (**)(Tao, Vec, Vec, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoSetObjectiveAndGradient(Tao, Vec, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetObjectiveAndGradient(Tao, Vec *, PetscErrorCode (**)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoSetHessian(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetHessian(Tao, Mat *, Mat *, PetscErrorCode (**)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoGetHessianMatrices(Tao, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode TaoGetDM(Tao, DM *);
PETSC_EXTERN PetscErrorCode TaoSetDM(Tao, DM);

PETSC_EXTERN PetscErrorCode TaoSetGradientNorm(Tao, Mat);
PETSC_EXTERN PetscErrorCode TaoGetGradientNorm(Tao, Mat *);
PETSC_EXTERN PetscErrorCode TaoSetLMVMMatrix(Tao, Mat);
PETSC_EXTERN PetscErrorCode TaoGetLMVMMatrix(Tao, Mat *);
PETSC_EXTERN PetscErrorCode TaoSetRecycleHistory(Tao, PetscBool);
PETSC_EXTERN PetscErrorCode TaoGetRecycleHistory(Tao, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoLMVMSetH0(Tao, Mat);
PETSC_EXTERN PetscErrorCode TaoLMVMGetH0(Tao, Mat *);
PETSC_EXTERN PetscErrorCode TaoLMVMGetH0KSP(Tao, KSP *);
PETSC_EXTERN PetscErrorCode TaoLMVMRecycle(Tao, PetscBool);
PETSC_EXTERN PetscErrorCode TaoSetResidualRoutine(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoSetResidualWeights(Tao, Vec, PetscInt, PetscInt *, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoSetConstraintsRoutine(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoSetInequalityConstraintsRoutine(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetInequalityConstraintsRoutine(Tao, Vec *, PetscErrorCode (**)(Tao, Vec, Vec, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoSetEqualityConstraintsRoutine(Tao, Vec, PetscErrorCode (*)(Tao, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetEqualityConstraintsRoutine(Tao, Vec *, PetscErrorCode (**)(Tao, Vec, Vec, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoSetJacobianResidualRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoSetJacobianRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoSetJacobianStateRoutine(Tao, Mat, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoSetJacobianDesignRoutine(Tao, Mat, PetscErrorCode (*)(Tao, Vec, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoSetJacobianInequalityRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetJacobianInequalityRoutine(Tao, Mat *, Mat *, PetscErrorCode (**)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtxRt);
PETSC_EXTERN PetscErrorCode TaoSetJacobianEqualityRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoGetJacobianEqualityRoutine(Tao, Mat *, Mat *, PetscErrorCode (**)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtxRt);

PETSC_EXTERN PetscErrorCode TaoPythonSetType(Tao, const char[]);
PETSC_EXTERN PetscErrorCode TaoPythonGetType(Tao, const char *[]);

PETSC_EXTERN PetscErrorCode TaoShellSetSolve(Tao, PetscErrorCode (*)(Tao));
PETSC_EXTERN PetscErrorCode TaoShellSetContext(Tao, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoShellGetContext(Tao, PetscCtxRt);

PETSC_EXTERN PetscErrorCode TaoSetStateDesignIS(Tao, IS, IS);

PETSC_EXTERN PetscErrorCode TaoComputeObjective(Tao, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoComputeResidual(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTestGradient(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeGradient(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeObjectiveAndGradient(Tao, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeConstraints(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeInequalityConstraints(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeEqualityConstraints(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoDefaultComputeGradient(Tao, Vec, Vec, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoIsObjectiveDefined(Tao, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoIsGradientDefined(Tao, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoIsObjectiveAndGradientDefined(Tao, PetscBool *);

PETSC_EXTERN PetscErrorCode TaoTestHessian(Tao);
PETSC_EXTERN PetscErrorCode TaoComputeHessian(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeResidualJacobian(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobian(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianState(Tao, Vec, Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianEquality(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianInequality(Tao, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianDesign(Tao, Vec, Mat);

PETSC_EXTERN PetscErrorCode TaoDefaultComputeHessian(Tao, Vec, Mat, Mat, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoDefaultComputeHessianColor(Tao, Vec, Mat, Mat, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoDefaultComputeHessianMFFD(Tao, Vec, Mat, Mat, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoComputeDualVariables(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoSetVariableBounds(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoGetVariableBounds(Tao, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoGetDualVariables(Tao, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoSetInequalityBounds(Tao, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoGetInequalityBounds(Tao, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoSetVariableBoundsRoutine(Tao, PetscErrorCode (*)(Tao, Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoComputeVariableBounds(Tao);

PETSC_EXTERN PetscErrorCode TaoGetTolerances(Tao, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoSetTolerances(Tao, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoGetConstraintTolerances(Tao, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoSetConstraintTolerances(Tao, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetFunctionLowerBound(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetInitialTrustRegionRadius(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetMaximumIterations(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoSetMaximumFunctionEvaluations(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoGetFunctionLowerBound(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoGetInitialTrustRegionRadius(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoGetCurrentTrustRegionRadius(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoGetMaximumIterations(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoGetCurrentFunctionEvaluations(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoGetMaximumFunctionEvaluations(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoGetIterationNumber(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoSetIterationNumber(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoGetTotalIterationNumber(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoSetTotalIterationNumber(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoGetResidualNorm(Tao, PetscReal *);

PETSC_EXTERN PetscErrorCode TaoAppendOptionsPrefix(Tao, const char[]);
PETSC_EXTERN PetscErrorCode TaoGetOptionsPrefix(Tao, const char *[]);
PETSC_EXTERN PetscErrorCode TaoResetStatistics(Tao);
PETSC_EXTERN PetscErrorCode TaoSetUpdate(Tao, PetscErrorCode (*)(Tao, PetscInt, PetscCtx), PetscCtx);

PETSC_EXTERN PetscErrorCode TaoGetKSP(Tao, KSP *);
PETSC_EXTERN PetscErrorCode TaoGetLinearSolveIterations(Tao, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoKSPSetUseEW(Tao, PetscBool);

#include <petsctaolinesearch.h>

PETSC_EXTERN PetscErrorCode TaoGetLineSearch(Tao, TaoLineSearch *);

PETSC_EXTERN PetscErrorCode TaoSetConvergenceHistory(Tao, PetscReal *, PetscReal *, PetscReal *, PetscInt *, PetscInt, PetscBool);
PETSC_EXTERN PetscErrorCode TaoGetConvergenceHistory(Tao, PetscReal **, PetscReal **, PetscReal **, PetscInt **, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoMonitorSet(Tao, PetscErrorCode (*)(Tao, PetscCtx), PetscCtx, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode TaoMonitorSetFromOptions(Tao, const char[], const char[], const char[], PetscErrorCode (*)(Tao, PetscViewerAndFormat *));
PETSC_EXTERN PetscErrorCode TaoMonitorCancel(Tao);
PETSC_EXTERN PetscErrorCode TaoMonitorDefault(Tao, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TaoMonitorGlobalization(Tao, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TaoMonitorConstraintNorm(Tao, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TaoMonitorSolution(Tao, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TaoMonitorResidual(Tao, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TaoMonitorGradient(Tao, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TaoMonitorStep(Tao, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TaoMonitorSolutionDraw(Tao, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoMonitorStepDraw(Tao, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoMonitorGradientDraw(Tao, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoAddLineSearchCounts(Tao);

PETSC_EXTERN PetscErrorCode TaoDefaultConvergenceTest(Tao, PetscCtx);
PETSC_EXTERN PetscErrorCode TaoSetConvergenceTest(Tao, PetscErrorCode (*)(Tao, PetscCtx), PetscCtx);

PETSC_EXTERN PetscErrorCode TaoMonitor(Tao, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoMonitorDrawCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TaoMonitorDrawCtx *);
PETSC_EXTERN PetscErrorCode TaoMonitorDrawCtxDestroy(TaoMonitorDrawCtx *);

PETSC_EXTERN PetscErrorCode TaoBRGNGetSubsolver(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoBRGNGetRegularizationType(Tao, TaoBRGNRegularizationType *);
PETSC_EXTERN PetscErrorCode TaoBRGNSetRegularizationType(Tao, TaoBRGNRegularizationType);
PETSC_EXTERN PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine(Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoBRGNSetRegularizerHessianRoutine(Tao, Mat, PetscErrorCode (*)(Tao, Vec, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoBRGNSetRegularizerWeight(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoBRGNSetL1SmoothEpsilon(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoBRGNSetDictionaryMatrix(Tao, Mat);
PETSC_EXTERN PetscErrorCode TaoBRGNGetDampingVector(Tao, Vec *);

PETSC_EXTERN PetscErrorCode TaoBNCGSetType(Tao, TaoBNCGType);
PETSC_EXTERN PetscErrorCode TaoBNCGGetType(Tao, TaoBNCGType *);

PETSC_EXTERN PetscErrorCode TaoADMMGetMisfitSubsolver(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoADMMGetRegularizationSubsolver(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoADMMGetDualVector(Tao, Vec *);
PETSC_EXTERN PetscErrorCode TaoADMMGetSpectralPenalty(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoADMMSetSpectralPenalty(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoGetADMMParentTao(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoADMMSetConstraintVectorRHS(Tao, Vec);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerCoefficient(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoADMMGetRegularizerCoefficient(Tao, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoADMMSetMisfitConstraintJacobian(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerConstraintJacobian(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerHessianRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerObjectiveAndGradientRoutine(Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoADMMSetMisfitHessianRoutine(Tao, Mat, Mat, PetscErrorCode (*)(Tao, Vec, Mat, Mat, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoADMMSetMisfitObjectiveAndGradientRoutine(Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode TaoADMMSetMisfitHessianChangeStatus(Tao, PetscBool);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegHessianChangeStatus(Tao, PetscBool);
PETSC_EXTERN PetscErrorCode TaoADMMSetMinimumSpectralPenalty(Tao, PetscReal);
PETSC_EXTERN PetscErrorCode TaoADMMSetRegularizerType(Tao, TaoADMMRegularizerType);
PETSC_EXTERN PetscErrorCode TaoADMMGetRegularizerType(Tao, TaoADMMRegularizerType *);
PETSC_EXTERN PetscErrorCode TaoADMMSetUpdateType(Tao, TaoADMMUpdateType);
PETSC_EXTERN PetscErrorCode TaoADMMGetUpdateType(Tao, TaoADMMUpdateType *);

PETSC_EXTERN PetscErrorCode TaoALMMGetType(Tao, TaoALMMType *);
PETSC_EXTERN PetscErrorCode TaoALMMSetType(Tao, TaoALMMType);
PETSC_EXTERN PetscErrorCode TaoALMMGetSubsolver(Tao, Tao *);
PETSC_EXTERN PetscErrorCode TaoALMMSetSubsolver(Tao, Tao);
PETSC_EXTERN PetscErrorCode TaoALMMGetMultipliers(Tao, Vec *);
PETSC_EXTERN PetscErrorCode TaoALMMSetMultipliers(Tao, Vec);
PETSC_EXTERN PetscErrorCode TaoALMMGetPrimalIS(Tao, IS *, IS *);
PETSC_EXTERN PetscErrorCode TaoALMMGetDualIS(Tao, IS *, IS *);

PETSC_EXTERN PetscErrorCode TaoVecGetSubVec(Vec, IS, TaoSubsetType, PetscReal, Vec *);
PETSC_EXTERN PetscErrorCode TaoMatGetSubMat(Mat, IS, Vec, TaoSubsetType, Mat *);
PETSC_EXTERN PetscErrorCode TaoGradientNorm(Tao, Vec, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoEstimateActiveBounds(Vec, Vec, Vec, Vec, Vec, Vec, PetscReal, PetscReal *, IS *, IS *, IS *, IS *, IS *);
PETSC_EXTERN PetscErrorCode TaoBoundStep(Vec, Vec, Vec, IS, IS, IS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TaoBoundSolution(Vec, Vec, Vec, PetscReal, PetscInt *, Vec);

PETSC_EXTERN PetscErrorCode MatCreateSubMatrixFree(Mat, IS, IS, Mat *);

PETSC_EXTERN PetscErrorCode TaoGetTerm(Tao, PetscReal *, TaoTerm *, Vec *, Mat *);
PETSC_EXTERN PetscErrorCode TaoAddTerm(Tao, const char[], PetscReal, TaoTerm, Vec, Mat);

#include <petsctao_deprecations.h>
