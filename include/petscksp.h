/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#pragma once

#include <petscpc.h>
#include <petscksptypes.h>

/* SUBMANSEC = KSP */

PETSC_EXTERN PetscErrorCode KSPInitializePackage(void);
PETSC_EXTERN PetscErrorCode KSPFinalizePackage(void);

/* Logging support */
PETSC_EXTERN PetscClassId KSP_CLASSID;
PETSC_EXTERN PetscClassId KSPGUESS_CLASSID;
PETSC_EXTERN PetscClassId DMKSP_CLASSID;

PETSC_EXTERN PetscErrorCode KSPCreate(MPI_Comm, KSP *);
PETSC_EXTERN PetscErrorCode KSPSetType(KSP, KSPType);
PETSC_EXTERN PetscErrorCode KSPGetType(KSP, KSPType *);
PETSC_EXTERN PetscErrorCode KSPSetUp(KSP);
PETSC_EXTERN PetscErrorCode KSPSetUpOnBlocks(KSP);
PETSC_EXTERN PetscErrorCode KSPSolve(KSP, Vec, Vec);
PETSC_EXTERN PetscErrorCode KSPSolveTranspose(KSP, Vec, Vec);
PETSC_EXTERN PetscErrorCode KSPSetUseExplicitTranspose(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPMatSolve(KSP, Mat, Mat);
PETSC_EXTERN PetscErrorCode KSPMatSolveTranspose(KSP, Mat, Mat);
PETSC_EXTERN PetscErrorCode KSPSetMatSolveBatchSize(KSP, PetscInt);
PETSC_DEPRECATED_FUNCTION(3, 15, 0, "KSPSetMatSolveBatchSize()", ) static inline PetscErrorCode KSPSetMatSolveBlockSize(KSP ksp, PetscInt n)
{
  return KSPSetMatSolveBatchSize(ksp, n);
}
PETSC_EXTERN PetscErrorCode KSPGetMatSolveBatchSize(KSP, PetscInt *);
PETSC_DEPRECATED_FUNCTION(3, 15, 0, "KSPGetMatSolveBatchSize()", ) static inline PetscErrorCode KSPGetMatSolveBlockSize(KSP ksp, PetscInt *n)
{
  return KSPGetMatSolveBatchSize(ksp, n);
}
PETSC_EXTERN PetscErrorCode KSPReset(KSP);
PETSC_EXTERN PetscErrorCode KSPResetViewers(KSP);
PETSC_EXTERN PetscErrorCode KSPDestroy(KSP *);
PETSC_EXTERN PetscErrorCode KSPSetReusePreconditioner(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetReusePreconditioner(KSP, PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetSkipPCSetFromOptions(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPCheckSolve(KSP, PC, Vec);

PETSC_EXTERN PetscFunctionList KSPList;
PETSC_EXTERN PetscFunctionList KSPGuessList;
PETSC_EXTERN PetscFunctionList KSPMonitorList;
PETSC_EXTERN PetscFunctionList KSPMonitorCreateList;
PETSC_EXTERN PetscFunctionList KSPMonitorDestroyList;
PETSC_EXTERN PetscErrorCode    KSPRegister(const char[], PetscErrorCode (*)(KSP));

PETSC_EXTERN PetscErrorCode KSPMonitorRegister(const char[], PetscViewerType, PetscViewerFormat, KSPMonitorRegisterFn *, KSPMonitorRegisterCreateFn *, KSPMonitorRegisterDestroyFn *);

PETSC_EXTERN PetscErrorCode KSPSetPCSide(KSP, PCSide);
PETSC_EXTERN PetscErrorCode KSPGetPCSide(KSP, PCSide *);
PETSC_EXTERN PetscErrorCode KSPSetTolerances(KSP, PetscReal, PetscReal, PetscReal, PetscInt);
PETSC_EXTERN PetscErrorCode KSPGetTolerances(KSP, PetscReal *, PetscReal *, PetscReal *, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPSetMinimumIterations(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPGetMinimumIterations(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPSetInitialGuessNonzero(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetInitialGuessNonzero(KSP, PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetErrorIfNotConverged(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetErrorIfNotConverged(KSP, PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetComputeEigenvalues(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPSetComputeRitz(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetComputeEigenvalues(KSP, PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetComputeSingularValues(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetComputeSingularValues(KSP, PetscBool *);
PETSC_EXTERN PetscErrorCode KSPGetRhs(KSP, Vec *);
PETSC_EXTERN PetscErrorCode KSPGetSolution(KSP, Vec *);
PETSC_EXTERN PetscErrorCode KSPGetResidualNorm(KSP, PetscReal *);
PETSC_EXTERN PetscErrorCode KSPGetIterationNumber(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPGetTotalIterations(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPCreateVecs(KSP, PetscInt, Vec **, PetscInt, Vec **);
PETSC_DEPRECATED_FUNCTION(3, 6, 0, "KSPCreateVecs()", ) static inline PetscErrorCode KSPGetVecs(KSP ksp, PetscInt n, Vec **x, PetscInt m, Vec **y)
{
  return KSPCreateVecs(ksp, n, x, m, y);
}

PETSC_EXTERN PetscErrorCode KSPSetPreSolve(KSP, KSPPSolveFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode KSPSetPostSolve(KSP, KSPPSolveFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode KSPPreSolve(KSP, Vec, Vec);
PETSC_EXTERN PetscErrorCode KSPPostSolve(KSP, Vec, Vec);

PETSC_EXTERN PetscErrorCode KSPSetPC(KSP, PC);
PETSC_EXTERN PetscErrorCode KSPGetPC(KSP, PC *);
PETSC_EXTERN PetscErrorCode KSPSetNestLevel(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPGetNestLevel(KSP, PetscInt *);

PETSC_EXTERN PetscErrorCode KSPMonitor(KSP, PetscInt, PetscReal);
PETSC_EXTERN PetscErrorCode KSPMonitorSet(KSP, KSPMonitorFn *, PetscCtx, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode KSPMonitorCancel(KSP);
PETSC_EXTERN PetscErrorCode KSPGetMonitorContext(KSP, PetscCtxRt);
PETSC_EXTERN PetscErrorCode KSPGetResidualHistory(KSP, const PetscReal *[], PetscInt *);
PETSC_EXTERN PetscErrorCode KSPSetResidualHistory(KSP, PetscReal[], PetscCount, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetErrorHistory(KSP, const PetscReal *[], PetscInt *);
PETSC_EXTERN PetscErrorCode KSPSetErrorHistory(KSP, PetscReal[], PetscCount, PetscBool);

PETSC_EXTERN PetscErrorCode KSPBuildSolutionDefault(KSP, Vec, Vec *);
PETSC_EXTERN PetscErrorCode KSPBuildResidualDefault(KSP, Vec, Vec, Vec *);
PETSC_EXTERN PetscErrorCode KSPDestroyDefault(KSP);
PETSC_EXTERN PetscErrorCode KSPSetWorkVecs(KSP, PetscInt);

PETSC_EXTERN PetscErrorCode PCKSPGetKSP(PC, KSP *);
PETSC_EXTERN PetscErrorCode PCKSPSetKSP(PC, KSP);
PETSC_EXTERN PetscErrorCode PCBJacobiGetSubKSP(PC, PetscInt *, PetscInt *, KSP *[]);
PETSC_EXTERN PetscErrorCode PCASMGetSubKSP(PC, PetscInt *, PetscInt *, KSP *[]);
PETSC_EXTERN PetscErrorCode PCGASMGetSubKSP(PC, PetscInt *, PetscInt *, KSP *[]);
PETSC_EXTERN PetscErrorCode PCPatchGetSubKSP(PC, PetscInt *, KSP *[]);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetSubKSP(PC, PetscInt *, KSP *[]);
PETSC_EXTERN PetscErrorCode PCFieldSplitSchurGetSubKSP(PC, PetscInt *, KSP *[]);
PETSC_EXTERN PetscErrorCode PCMGGetSmoother(PC, PetscInt, KSP *);
PETSC_EXTERN PetscErrorCode PCMGGetSmootherDown(PC, PetscInt, KSP *);
PETSC_EXTERN PetscErrorCode PCMGGetSmootherUp(PC, PetscInt, KSP *);
PETSC_EXTERN PetscErrorCode PCMGGetCoarseSolve(PC, KSP *);
PETSC_EXTERN PetscErrorCode PCGalerkinGetKSP(PC, KSP *);
PETSC_EXTERN PetscErrorCode PCDeflationGetCoarseKSP(PC, KSP *);

PETSC_EXTERN PetscFunctionList PCMGCoarseList;
PETSC_EXTERN PetscErrorCode    PCMGRegisterCoarseSpaceConstructor(const char[], PCMGCoarseSpaceConstructorFn *);
PETSC_EXTERN PetscErrorCode    PCMGGetCoarseSpaceConstructor(const char[], PCMGCoarseSpaceConstructorFn **);

PETSC_EXTERN PetscErrorCode KSPBuildSolution(KSP, Vec, Vec *);
PETSC_EXTERN PetscErrorCode KSPBuildResidual(KSP, Vec, Vec, Vec *);

PETSC_EXTERN PetscErrorCode KSPRichardsonSetScale(KSP, PetscReal);
PETSC_EXTERN PetscErrorCode KSPRichardsonSetSelfScale(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPChebyshevSetEigenvalues(KSP, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode KSPChebyshevEstEigSet(KSP, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode KSPChebyshevEstEigSetUseNoisy(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPChebyshevSetKind(KSP, KSPChebyshevKind);
PETSC_EXTERN PetscErrorCode KSPChebyshevGetKind(KSP, KSPChebyshevKind *);
PETSC_EXTERN PetscErrorCode KSPChebyshevEstEigGetKSP(KSP, KSP *);
PETSC_EXTERN PetscErrorCode KSPComputeExtremeSingularValues(KSP, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode KSPComputeEigenvalues(KSP, PetscInt, PetscReal[], PetscReal[], PetscInt *);
PETSC_EXTERN PetscErrorCode KSPComputeEigenvaluesExplicitly(KSP, PetscInt, PetscReal[], PetscReal[]);
PETSC_EXTERN PetscErrorCode KSPComputeRitz(KSP, PetscBool, PetscBool, PetscInt *, Vec[], PetscReal[], PetscReal[]);

PETSC_EXTERN PetscErrorCode KSPFCGSetMmax(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPFCGGetMmax(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPFCGSetNprealloc(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPFCGGetNprealloc(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPFCGSetTruncationType(KSP, KSPFCDTruncationType);
PETSC_EXTERN PetscErrorCode KSPFCGGetTruncationType(KSP, KSPFCDTruncationType *);

PETSC_EXTERN PetscErrorCode KSPPIPEFCGSetMmax(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPPIPEFCGGetMmax(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPPIPEFCGSetNprealloc(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPPIPEFCGGetNprealloc(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPPIPEFCGSetTruncationType(KSP, KSPFCDTruncationType);
PETSC_EXTERN PetscErrorCode KSPPIPEFCGGetTruncationType(KSP, KSPFCDTruncationType *);

PETSC_EXTERN PetscErrorCode KSPPIPEGCRSetMmax(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPPIPEGCRGetMmax(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPPIPEGCRSetNprealloc(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPPIPEGCRGetNprealloc(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPPIPEGCRSetTruncationType(KSP, KSPFCDTruncationType);
PETSC_EXTERN PetscErrorCode KSPPIPEGCRGetTruncationType(KSP, KSPFCDTruncationType *);
PETSC_EXTERN PetscErrorCode KSPPIPEGCRSetUnrollW(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPPIPEGCRGetUnrollW(KSP, PetscBool *);

PETSC_EXTERN PetscErrorCode KSPFlexibleSetModifyPC(KSP, KSPFlexibleModifyPCFn *, PetscCtx, PetscCtxDestroyFn *);

PETSC_DEPRECATED_FUNCTION(3, 25, 0, "KSPFlexibleSetModifyPC()", ) static inline PetscErrorCode KSPPIPEGCRSetModifyPC(KSP ksp, KSPFlexibleModifyPCFn *fun, PetscCtx ctx, PetscCtxDestroyFn *dfun)
{
  return KSPFlexibleSetModifyPC(ksp, fun, ctx, dfun);
}

PETSC_EXTERN PetscErrorCode KSPGMRESSetRestart(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPGMRESGetRestart(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPGMRESSetHapTol(KSP, PetscReal);
PETSC_EXTERN PetscErrorCode KSPGMRESSetBreakdownTolerance(KSP, PetscReal);

PETSC_EXTERN PetscErrorCode KSPGMRESSetPreAllocateVectors(KSP);
PETSC_EXTERN PetscErrorCode KSPGMRESSetOrthogonalization(KSP, PetscErrorCode (*)(KSP, PetscInt));
PETSC_EXTERN PetscErrorCode KSPGMRESGetOrthogonalization(KSP, PetscErrorCode (**)(KSP, PetscInt));
PETSC_EXTERN PetscErrorCode KSPGMRESModifiedGramSchmidtOrthogonalization(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPGMRESClassicalGramSchmidtOrthogonalization(KSP, PetscInt);

PETSC_EXTERN PetscErrorCode KSPLGMRESSetAugDim(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPLGMRESSetConstant(KSP);

PETSC_EXTERN PetscErrorCode KSPPIPEFGMRESSetShift(KSP, PetscScalar);

PETSC_EXTERN PetscErrorCode KSPGCRSetRestart(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPGCRGetRestart(KSP, PetscInt *);

PETSC_EXTERN PetscErrorCode KSPIDRSetS(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPIDRGetS(KSP, PetscInt *);
PETSC_EXTERN PetscErrorCode KSPIDRSetCosine(KSP, PetscReal);
PETSC_EXTERN PetscErrorCode KSPIDRGetCosine(KSP, PetscReal *);
PETSC_EXTERN PetscErrorCode KSPIDRSetRandom(KSP, PetscRandom);
PETSC_EXTERN PetscErrorCode KSPIDRGetRandom(KSP, PetscRandom *);

PETSC_DEPRECATED_FUNCTION(3, 25, 0, "KSPFlexibleSetModifyPC()", ) static inline PetscErrorCode KSPGCRSetModifyPC(KSP ksp, KSPFlexibleModifyPCFn *fun, PetscCtx ctx, PetscCtxDestroyFn *dfun)
{
  return KSPFlexibleSetModifyPC(ksp, fun, ctx, dfun);
}

PETSC_EXTERN PetscErrorCode KSPMINRESSetRadius(KSP, PetscReal);
PETSC_EXTERN PetscErrorCode KSPMINRESGetUseQLP(KSP, PetscBool *);
PETSC_EXTERN PetscErrorCode KSPMINRESSetUseQLP(KSP, PetscBool);

PETSC_EXTERN PetscErrorCode KSPFETIDPGetInnerBDDC(KSP, PC *);
PETSC_EXTERN PetscErrorCode KSPFETIDPSetInnerBDDC(KSP, PC);
PETSC_EXTERN PetscErrorCode KSPFETIDPGetInnerKSP(KSP, KSP *);
PETSC_EXTERN PetscErrorCode KSPFETIDPSetPressureOperator(KSP, Mat);

PETSC_EXTERN PetscErrorCode KSPHPDDMSetDeflationMat(KSP, Mat);
PETSC_EXTERN PetscErrorCode KSPHPDDMGetDeflationMat(KSP, Mat *);
#if PetscDefined(HAVE_HPDDM)
PETSC_DEPRECATED_FUNCTION(3, 18, 0, "KSPHPDDMSetDeflationMat()", ) static inline PetscErrorCode KSPHPDDMSetDeflationSpace(KSP ksp, Mat U)
{
  return KSPHPDDMSetDeflationMat(ksp, U);
}
PETSC_DEPRECATED_FUNCTION(3, 18, 0, "KSPHPDDMGetDeflationMat()", ) static inline PetscErrorCode KSPHPDDMGetDeflationSpace(KSP ksp, Mat *U)
{
  return KSPHPDDMGetDeflationMat(ksp, U);
}
#endif
PETSC_DEPRECATED_FUNCTION(3, 14, 0, "KSPMatSolve()", ) static inline PetscErrorCode KSPHPDDMMatSolve(KSP ksp, Mat B, Mat X)
{
  return KSPMatSolve(ksp, B, X);
}

PETSC_EXTERN PetscErrorCode KSPHPDDMSetType(KSP, KSPHPDDMType);
PETSC_EXTERN PetscErrorCode KSPHPDDMGetType(KSP, KSPHPDDMType *);

/*MC
   KSP_GMRES_CGS_REFINE_NEVER - Do the classical (unmodified) Gram-Schmidt process

   Level: advanced

   Note:
   Possibly unstable, but the fastest to compute

.seealso: [](ch_ksp), `KSPGMRES`, `KSPGMRESCGSRefinementType`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESSetOrthogonalization()`,
          `KSP`, `KSPGMRESGetOrthogonalization()`,
          `KSPGMRESSetCGSRefinementType()`, `KSPGMRESGetCGSRefinementType()`, `KSP_GMRES_CGS_REFINE_IFNEEDED`, `KSP_GMRES_CGS_REFINE_ALWAYS`,
          `KSPGMRESModifiedGramSchmidtOrthogonalization()`
M*/

/*MC
    KSP_GMRES_CGS_REFINE_IFNEEDED - Do the classical (unmodified) Gram-Schmidt process and one step of
          iterative refinement if an estimate of the orthogonality of the resulting vectors indicates
          poor orthogonality.

   Level: advanced

   Note:
   This is slower than `KSP_GMRES_CGS_REFINE_NEVER` because it requires an extra norm computation to
   estimate the orthogonality but is more stable.

.seealso: [](ch_ksp), `KSPGMRES`, `KSPGMRESCGSRefinementType`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESSetOrthogonalization()`,
          `KSP`, `KSPGMRESGetOrthogonalization()`,
          `KSPGMRESSetCGSRefinementType()`, `KSPGMRESGetCGSRefinementType()`, `KSP_GMRES_CGS_REFINE_NEVER`, `KSP_GMRES_CGS_REFINE_ALWAYS`,
          `KSPGMRESModifiedGramSchmidtOrthogonalization()`
M*/

/*MC
   KSP_GMRES_CGS_REFINE_ALWAYS - Do two steps of the classical (unmodified) Gram-Schmidt process.

   Level: advanced

   Notes:
   This is roughly twice the cost of `KSP_GMRES_CGS_REFINE_NEVER` because it performs the process twice
   but it saves the extra norm calculation needed by `KSP_GMRES_CGS_REFINE_IFNEEDED`.

   You should only use this if you absolutely know that the iterative refinement is needed.

.seealso: [](ch_ksp), `KSPGMRES`, `KSPGMRESCGSRefinementType`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESSetOrthogonalization()`,
          `KSP`, `KSPGMRESGetOrthogonalization()`,
          `KSPGMRESSetCGSRefinementType()`, `KSPGMRESGetCGSRefinementType()`, `KSP_GMRES_CGS_REFINE_IFNEEDED`, `KSP_GMRES_CGS_REFINE_ALWAYS`,
          `KSPGMRESModifiedGramSchmidtOrthogonalization()`
M*/

PETSC_EXTERN PetscErrorCode KSPGMRESSetCGSRefinementType(KSP, KSPGMRESCGSRefinementType);
PETSC_EXTERN PetscErrorCode KSPGMRESGetCGSRefinementType(KSP, KSPGMRESCGSRefinementType *);

PETSC_EXTERN KSPFlexibleModifyPCFn KSPFlexibleModifyPCNoChange;
PETSC_EXTERN KSPFlexibleModifyPCFn KSPFlexibleModifyPCKSP;

PETSC_DEPRECATED_FUNCTION(3, 25, 0, "KSPFlexibleModifyPCNoChange()", ) static inline PetscErrorCode KSPFGMRESModifyPCNoChange(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, PetscCtx ctx)
{
  return KSPFlexibleModifyPCNoChange(ksp, total_its, loc_its, res_norm, ctx);
}

PETSC_DEPRECATED_FUNCTION(3, 25, 0, "KSPFlexibleModifyPCKSP()", ) static inline PetscErrorCode KSPFGMRESModifyPCKSP(KSP ksp, PetscInt total_its, PetscInt loc_its, PetscReal res_norm, PetscCtx ctx)
{
  return KSPFlexibleModifyPCKSP(ksp, total_its, loc_its, res_norm, ctx);
}

PETSC_EXTERN PetscErrorCode KSPQCGSetTrustRegionRadius(KSP, PetscReal);
PETSC_EXTERN PetscErrorCode KSPQCGGetQuadratic(KSP, PetscReal *);
PETSC_EXTERN PetscErrorCode KSPQCGGetTrialStepNorm(KSP, PetscReal *);

PETSC_EXTERN PetscErrorCode KSPBCGSLSetXRes(KSP, PetscReal);
PETSC_EXTERN PetscErrorCode KSPBCGSLSetPol(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPBCGSLSetEll(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPBCGSLSetUsePseudoinverse(KSP, PetscBool);

PETSC_EXTERN PetscErrorCode KSPSetFromOptions(KSP);
PETSC_EXTERN PetscErrorCode KSPResetFromOptions(KSP);

PETSC_EXTERN PetscErrorCode       KSPMonitorSetFromOptions(KSP, const char[], const char[], PetscCtx);
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorResidual;
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorResidualView;
PETSC_DEPRECATED_FUNCTION(3, 23, 0, "KSPMonitorResidualDraw()", ) static inline PetscErrorCode KSPMonitorResidualDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  return KSPMonitorResidualView(ksp, n, rnorm, vf);
}
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorResidualDrawLG;
PETSC_EXTERN PetscErrorCode       KSPMonitorResidualDrawLGCreate(PetscViewer, PetscViewerFormat, PetscCtx, PetscViewerAndFormat **);
PETSC_DEPRECATED_FUNCTION(3, 26, 0, "KSPMonitorResidual()", ) static inline PetscErrorCode KSPMonitorResidualShort(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  return KSPMonitorResidual(ksp, n, rnorm, vf);
}
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorResidualRange;
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorTrueResidual;
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorTrueResidualView;
PETSC_DEPRECATED_FUNCTION(3, 23, 0, "KSPMonitorTrueResidualDraw()", ) static inline PetscErrorCode KSPMonitorTrueResidualDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  return KSPMonitorTrueResidualView(ksp, n, rnorm, vf);
}
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorTrueResidualDrawLG;
PETSC_EXTERN PetscErrorCode       KSPMonitorTrueResidualDrawLGCreate(PetscViewer, PetscViewerFormat, PetscCtx, PetscViewerAndFormat **);
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorTrueResidualMax;
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorError;
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorErrorDraw;
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorErrorDrawLG;
PETSC_EXTERN PetscErrorCode       KSPMonitorErrorDrawLGCreate(PetscViewer, PetscViewerFormat, PetscCtx, PetscViewerAndFormat **);
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorSolution;
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorSolutionDraw;
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorSolutionDrawLG;
PETSC_EXTERN PetscErrorCode       KSPMonitorSolutionDrawLGCreate(PetscViewer, PetscViewerFormat, PetscCtx, PetscViewerAndFormat **);
PETSC_EXTERN KSPMonitorRegisterFn KSPMonitorSingularValue;
PETSC_EXTERN PetscErrorCode       KSPMonitorSingularValueCreate(PetscViewer, PetscViewerFormat, PetscCtx, PetscViewerAndFormat **);
PETSC_DEPRECATED_FUNCTION(3, 15, 0, "KSPMonitorResidual()", ) static inline PetscErrorCode KSPMonitorDefault(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  return KSPMonitorResidual(ksp, n, rnorm, vf);
}
PETSC_DEPRECATED_FUNCTION(3, 15, 0, "KSPMonitorTrueResidual()", ) static inline PetscErrorCode KSPMonitorTrueResidualNorm(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  return KSPMonitorTrueResidual(ksp, n, rnorm, vf);
}
PETSC_DEPRECATED_FUNCTION(3, 15, 0, "KSPMonitorTrueResidualMax()", ) static inline PetscErrorCode KSPMonitorTrueResidualMaxNorm(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  return KSPMonitorTrueResidualMax(ksp, n, rnorm, vf);
}

PETSC_EXTERN PetscErrorCode KSPGMRESMonitorKrylov(KSP, PetscInt, PetscReal, void *);
PETSC_EXTERN PetscErrorCode KSPMonitorDynamicTolerance(KSP, PetscInt, PetscReal, PetscCtx);
PETSC_EXTERN PetscErrorCode KSPMonitorDynamicToleranceDestroy(PetscCtxRt);
PETSC_EXTERN PetscErrorCode KSPMonitorDynamicToleranceCreate(void *);
PETSC_EXTERN PetscErrorCode KSPMonitorDynamicToleranceSetCoefficient(void *, PetscReal);
PETSC_EXTERN PetscErrorCode KSPMonitorSAWs(KSP, PetscInt, PetscReal, void *);
PETSC_EXTERN PetscErrorCode KSPMonitorSAWsCreate(KSP, void **);
PETSC_EXTERN PetscErrorCode KSPMonitorSAWsDestroy(PetscCtxRt);

PETSC_EXTERN PetscErrorCode KSPUnwindPreconditioner(KSP, Vec, Vec);
PETSC_EXTERN PetscErrorCode KSPInitialResidual(KSP, Vec, Vec, Vec, Vec, Vec);

PETSC_EXTERN PetscErrorCode KSPSetOperators(KSP, Mat, Mat);
PETSC_EXTERN PetscErrorCode KSPGetOperators(KSP, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode KSPGetOperatorsSet(KSP, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetOptionsPrefix(KSP, const char[]);
PETSC_EXTERN PetscErrorCode KSPAppendOptionsPrefix(KSP, const char[]);
PETSC_EXTERN PetscErrorCode KSPGetOptionsPrefix(KSP, const char *[]);

PETSC_EXTERN PetscErrorCode KSPSetDiagonalScale(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetDiagonalScale(KSP, PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetDiagonalScaleFix(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetDiagonalScaleFix(KSP, PetscBool *);

PETSC_EXTERN PetscErrorCode KSPView(KSP, PetscViewer);
PETSC_EXTERN PetscErrorCode KSPLoad(KSP, PetscViewer);
PETSC_EXTERN PetscErrorCode KSPViewFromOptions(KSP, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode KSPConvergedReasonView(KSP, PetscViewer);
PETSC_EXTERN PetscErrorCode KSPConvergedReasonViewSet(KSP, KSPConvergedReasonViewFn *, void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode KSPConvergedReasonViewFromOptions(KSP);
PETSC_EXTERN PetscErrorCode KSPConvergedReasonViewCancel(KSP);
PETSC_EXTERN PetscErrorCode KSPConvergedRateView(KSP, PetscViewer);

PETSC_DEPRECATED_FUNCTION(3, 14, 0, "KSPConvergedReasonView()", ) static inline PetscErrorCode KSPReasonView(KSP ksp, PetscViewer v)
{
  return KSPConvergedReasonView(ksp, v);
}
PETSC_DEPRECATED_FUNCTION(3, 14, 0, "KSPConvergedReasonViewFromOptions()", ) static inline PetscErrorCode KSPReasonViewFromOptions(KSP ksp)
{
  return KSPConvergedReasonViewFromOptions(ksp);
}

#define KSP_FILE_CLASSID 1211223

PETSC_EXTERN PetscErrorCode       KSPLSQRSetExactMatNorm(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode       KSPLSQRSetComputeStandardErrorVec(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode       KSPLSQRGetStandardErrorVec(KSP, Vec *);
PETSC_EXTERN PetscErrorCode       KSPLSQRGetNorms(KSP, PetscReal *, PetscReal *);
PETSC_EXTERN KSPMonitorRegisterFn KSPLSQRMonitorResidual;
PETSC_EXTERN KSPMonitorRegisterFn KSPLSQRMonitorResidualDrawLG;
PETSC_EXTERN PetscErrorCode       KSPLSQRMonitorResidualDrawLGCreate(PetscViewer, PetscViewerFormat, void *, PetscViewerAndFormat **);

PETSC_EXTERN PetscErrorCode PCRedundantGetKSP(PC, KSP *);
PETSC_EXTERN PetscErrorCode PCRedistributeGetKSP(PC, KSP *);
PETSC_EXTERN PetscErrorCode PCTelescopeGetKSP(PC, KSP *);
PETSC_EXTERN PetscErrorCode PCMPIGetKSP(PC, KSP *);

#define KSP_NORM_MAX (KSP_NORM_NATURAL + 1)

/*MC
   KSP_NORM_NONE - Do not compute a norm during the Krylov process. This will
   possibly save some computation but means the convergence test cannot
   be based on a norm of a residual etc.

   Level: advanced

   Note:
   Some Krylov methods need to compute a residual norm (such as `KPSGMRES`) and then this option is ignored

.seealso: [](ch_ksp), `KSPNormType`, `KSP`, `KSPSetNormType()`, `KSP_NORM_PRECONDITIONED`, `KSP_NORM_UNPRECONDITIONED`, `KSP_NORM_NATURAL`
M*/

/*MC
   KSP_NORM_PRECONDITIONED - Compute the norm of the preconditioned residual B*(b - A*x), if left preconditioning, and pass that to the
   convergence test routine.

   Level: advanced

.seealso: [](ch_ksp), `KSPNormType`, `KSP`, `KSPSetNormType()`, `KSP_NORM_NONE`, `KSP_NORM_UNPRECONDITIONED`, `KSP_NORM_NATURAL`, `KSPSetConvergenceTest()`
M*/

/*MC
   KSP_NORM_UNPRECONDITIONED - Compute the norm of the true residual (b - A*x) and pass that to the
   convergence test routine.

   Level: advanced

.seealso: [](ch_ksp), `KSPNormType`, `KSP`, `KSPSetNormType()`, `KSP_NORM_NONE`, `KSP_NORM_PRECONDITIONED`, `KSP_NORM_NATURAL`, `KSPSetConvergenceTest()`
M*/

/*MC
   KSP_NORM_NATURAL - Compute the 'natural norm' of residual sqrt((b - A*x)*B*(b - A*x)) and pass that to the
   convergence test routine. This is only supported by  `KSPCG`, `KSPCR`, `KSPCGNE`, `KSPCGS`, `KSPFCG`, `KSPPIPEFCG`, `KSPPIPEGCR`

   Level: advanced

.seealso: [](ch_ksp), `KSPNormType`, `KSP`, `KSPSetNormType()`, `KSP_NORM_NONE`, `KSP_NORM_PRECONDITIONED`, `KSP_NORM_UNPRECONDITIONED`, `KSPSetConvergenceTest()`
M*/

PETSC_EXTERN PetscErrorCode KSPSetNormType(KSP, KSPNormType);
PETSC_EXTERN PetscErrorCode KSPGetNormType(KSP, KSPNormType *);
PETSC_EXTERN PetscErrorCode KSPSetSupportedNorm(KSP, KSPNormType, PCSide, PetscInt);
PETSC_EXTERN PetscErrorCode KSPSetCheckNormIteration(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPSetLagNorm(KSP, PetscBool);

/*MC
   KSP_CONVERGED_RTOL - $||r|| \le rtol*||b||$ or $rtol*||b - A*x_0||$ if `KSPConvergedDefaultSetUIRNorm()` was called

   Level: beginner

   Notes:
   See `KSPNormType` and `KSPSetNormType()` for possible norms that may be used. By default
   for left preconditioning it is the 2-norm of the preconditioned residual, and the
   2-norm of the residual for right preconditioning

   See also `KSP_CONVERGED_ATOL` which may apply before this tolerance.

.seealso: [](ch_ksp), `KSPNormType`, `KSP_CONVERGED_ATOL`, `KSP_DIVERGED_DTOL`, `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_CONVERGED_ATOL - $||r|| \le atol$

   Level: beginner

   Notes:
   See `KSPNormType` and `KSPSetNormType()` for possible norms that may be used. By default
   for left preconditioning it is the 2-norm of the preconditioned residual, and the
   2-norm of the residual for right preconditioning

   See also `KSP_CONVERGED_RTOL` which may apply before this tolerance.

.seealso: [](ch_ksp), `KSPNormType`, `KSP_CONVERGED_RTOL`, `KSP_DIVERGED_DTOL`, `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_DIVERGED_DTOL - $||r|| \ge dtol*||b||$

   Level: beginner

   Note:
   See `KSPNormType` and `KSPSetNormType()` for possible norms that may be used. By default
   for left preconditioning it is the 2-norm of the preconditioned residual, and the
   2-norm of the residual for right preconditioning

.seealso: [](ch_ksp), `KSPNormType`, `KSP_CONVERGED_ATOL`, `KSP_CONVERGED_RTOL`, `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_DIVERGED_ITS - Ran out of iterations before any convergence criteria was
   reached

   Level: beginner

.seealso: [](ch_ksp), `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_CONVERGED_ITS - Used by the `KSPPREONLY` solver after the single iteration of
   the preconditioner is applied. Also used when the `KSPConvergedSkip()` convergence
   test routine is set in `KSP`.

   Level: beginner

.seealso: [](ch_ksp), `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_DIVERGED_BREAKDOWN - A breakdown in the Krylov method was detected so the
   method could not continue to enlarge the Krylov space. Could be due to a singular matrix or
   preconditioner. In `KSPHPDDM`, this is also returned when some search directions within a block
   are collinear.

   Level: beginner

.seealso: [](ch_ksp), `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_DIVERGED_BREAKDOWN_BICG - A breakdown in the `KSPBICG` method was detected so the
   method could not continue to enlarge the Krylov space.

   Level: beginner

.seealso: [](ch_ksp), `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_DIVERGED_NONSYMMETRIC - It appears the operator or preconditioner is not
   symmetric and this Krylov method (`KSPCG`, `KSPMINRES`, `KSPCR`) requires symmetry

   Level: beginner

.seealso: [](ch_ksp), `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_DIVERGED_INDEFINITE_PC - It appears the preconditioner is indefinite (has both
   positive and negative eigenvalues) and this Krylov method (`KSPCG`) requires it to
   be symmetric positive definite (SPD).

   Level: beginner

   Note:
   This can happen with the `PCICC` preconditioner, use the options database option `-pc_factor_shift_positive_definite` to force
   the `PCICC` preconditioner to generate a positive definite preconditioner

.seealso: [](ch_ksp), `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_DIVERGED_PC_FAILED - It was not possible to build or use the requested preconditioner. This is usually due to a
   zero pivot in a factorization. It can also result from a failure in a subpreconditioner inside a nested preconditioner
   such as `PCFIELDSPLIT`.

   Level: beginner

   Note:
   Run with `-ksp_error_if_not_converged` to stop the program when the error is detected and print an error message with details.

.seealso: [](ch_ksp), `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

/*MC
   KSP_CONVERGED_ITERATING - This flag is returned if `KSPGetConvergedReason()` is called
   while `KSPSolve()` is still running.

   Level: beginner

.seealso: [](ch_ksp), `KSPSolve()`, `KSPGetConvergedReason()`, `KSPConvergedReason`, `KSPSetTolerances()`
M*/

PETSC_EXTERN PetscErrorCode       KSPSetConvergenceTest(KSP, KSPConvergenceTestFn *, void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode       KSPGetConvergenceTest(KSP, KSPConvergenceTestFn **, PetscCtxRt, PetscCtxDestroyFn **);
PETSC_EXTERN PetscErrorCode       KSPGetAndClearConvergenceTest(KSP, KSPConvergenceTestFn **, PetscCtxRt, PetscCtxDestroyFn **);
PETSC_EXTERN PetscErrorCode       KSPGetConvergenceContext(KSP, PetscCtxRt);
PETSC_EXTERN KSPConvergenceTestFn KSPConvergedDefault;
PETSC_EXTERN KSPConvergenceTestFn KSPLSQRConvergedDefault;
PETSC_EXTERN PetscCtxDestroyFn    KSPConvergedDefaultDestroy;
PETSC_EXTERN PetscErrorCode       KSPConvergedDefaultCreate(void **);
PETSC_EXTERN PetscErrorCode       KSPConvergedDefaultSetUIRNorm(KSP);
PETSC_EXTERN PetscErrorCode       KSPConvergedDefaultSetUMIRNorm(KSP);
PETSC_EXTERN PetscErrorCode       KSPConvergedDefaultSetConvergedMaxits(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode       KSPConvergedSkip(KSP, PetscInt, PetscReal, KSPConvergedReason *, void *);
PETSC_EXTERN PetscErrorCode       KSPGetConvergedReason(KSP, KSPConvergedReason *);
PETSC_EXTERN PetscErrorCode       KSPGetConvergedReasonString(KSP, const char *[]);
PETSC_EXTERN PetscErrorCode       KSPComputeConvergenceRate(KSP, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode       KSPSetConvergedNegativeCurvature(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode       KSPGetConvergedNegativeCurvature(KSP, PetscBool *);

PETSC_DEPRECATED_FUNCTION(3, 5, 0, "KSPConvergedDefault()", ) static inline void KSPDefaultConverged(void)
{ /* never called */
}
#define KSPDefaultConverged (KSPDefaultConverged, KSPConvergedDefault)
PETSC_DEPRECATED_FUNCTION(3, 5, 0, "KSPConvergedDefaultDestroy()", ) static inline void KSPDefaultConvergedDestroy(void)
{ /* never called */
}
#define KSPDefaultConvergedDestroy (KSPDefaultConvergedDestroy, KSPConvergedDefaultDestroy)
PETSC_DEPRECATED_FUNCTION(3, 5, 0, "KSPConvergedDefaultCreate()", ) static inline void KSPDefaultConvergedCreate(void)
{ /* never called */
}
#define KSPDefaultConvergedCreate (KSPDefaultConvergedCreate, KSPConvergedDefaultCreate)
PETSC_DEPRECATED_FUNCTION(3, 5, 0, "KSPConvergedDefaultSetUIRNorm()", ) static inline void KSPDefaultConvergedSetUIRNorm(void)
{ /* never called */
}
#define KSPDefaultConvergedSetUIRNorm (KSPDefaultConvergedSetUIRNorm, KSPConvergedDefaultSetUIRNorm)
PETSC_DEPRECATED_FUNCTION(3, 5, 0, "KSPConvergedDefaultSetUMIRNorm()", ) static inline void KSPDefaultConvergedSetUMIRNorm(void)
{ /* never called */
}
#define KSPDefaultConvergedSetUMIRNorm (KSPDefaultConvergedSetUMIRNorm, KSPConvergedDefaultSetUMIRNorm)
PETSC_DEPRECATED_FUNCTION(3, 5, 0, "KSPConvergedSkip()", ) static inline void KSPSkipConverged(void)
{ /* never called */
}
#define KSPSkipConverged (KSPSkipConverged, KSPConvergedSkip)

PETSC_EXTERN PetscErrorCode KSPComputeOperator(KSP, MatType, Mat *);
PETSC_DEPRECATED_FUNCTION(3, 12, 0, "KSPComputeOperator()", ) static inline PetscErrorCode KSPComputeExplicitOperator(KSP A, Mat *B)
{
  return KSPComputeOperator(A, PETSC_NULLPTR, B);
}

PETSC_EXTERN PetscErrorCode KSPCGSetType(KSP, KSPCGType);
PETSC_EXTERN PetscErrorCode KSPCGUseSingleReduction(KSP, PetscBool);

PETSC_EXTERN PetscErrorCode KSPCGSetRadius(KSP, PetscReal);
PETSC_EXTERN PetscErrorCode KSPCGSetObjectiveTarget(KSP, PetscReal);
PETSC_EXTERN PetscErrorCode KSPCGGetNormD(KSP, PetscReal *);
PETSC_EXTERN PetscErrorCode KSPCGGetObjFcn(KSP, PetscReal *);

PETSC_EXTERN PetscErrorCode KSPGLTRGetMinEig(KSP, PetscReal *);
PETSC_EXTERN PetscErrorCode KSPGLTRGetLambda(KSP, PetscReal *);
PETSC_DEPRECATED_FUNCTION(3, 12, 0, "KSPGLTRGetMinEig()", ) static inline PetscErrorCode KSPCGGLTRGetMinEig(KSP ksp, PetscReal *x)
{
  return KSPGLTRGetMinEig(ksp, x);
}
PETSC_DEPRECATED_FUNCTION(3, 12, 0, "KSPGLTRGetLambda()", ) static inline PetscErrorCode KSPCGGLTRGetLambda(KSP ksp, PetscReal *x)
{
  return KSPGLTRGetLambda(ksp, x);
}

PETSC_EXTERN PetscErrorCode KSPPythonSetType(KSP, const char[]);
PETSC_EXTERN PetscErrorCode KSPPythonGetType(KSP, const char *[]);

PETSC_EXTERN PetscErrorCode PCPreSolve(PC, KSP);
PETSC_EXTERN PetscErrorCode PCPostSolve(PC, KSP);

PETSC_EXTERN PetscErrorCode KSPMonitorLGRange(KSP, PetscInt, PetscReal, void *);

PETSC_EXTERN PetscErrorCode PCShellSetPreSolve(PC, PCShellPSolveFn *);
PETSC_EXTERN PetscErrorCode PCShellSetPostSolve(PC, PCShellPSolveFn *);

PETSC_EXTERN PetscErrorCode KSPGuessRegister(const char[], PetscErrorCode (*)(KSPGuess));
PETSC_EXTERN PetscErrorCode KSPSetGuess(KSP, KSPGuess);
PETSC_EXTERN PetscErrorCode KSPGetGuess(KSP, KSPGuess *);
PETSC_EXTERN PetscErrorCode KSPGuessView(KSPGuess, PetscViewer);
PETSC_EXTERN PetscErrorCode KSPGuessDestroy(KSPGuess *);
PETSC_EXTERN PetscErrorCode KSPGuessCreate(MPI_Comm, KSPGuess *);
PETSC_EXTERN PetscErrorCode KSPGuessSetType(KSPGuess, KSPGuessType);
PETSC_EXTERN PetscErrorCode KSPGuessGetType(KSPGuess, KSPGuessType *);
PETSC_EXTERN PetscErrorCode KSPGuessSetTolerance(KSPGuess, PetscReal);
PETSC_EXTERN PetscErrorCode KSPGuessSetUp(KSPGuess);
PETSC_EXTERN PetscErrorCode KSPGuessUpdate(KSPGuess, Vec, Vec);
PETSC_EXTERN PetscErrorCode KSPGuessFormGuess(KSPGuess, Vec, Vec);
PETSC_EXTERN PetscErrorCode KSPGuessSetFromOptions(KSPGuess);
PETSC_EXTERN PetscErrorCode KSPGuessFischerSetModel(KSPGuess, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode KSPSetUseFischerGuess(KSP, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode KSPSetInitialGuessKnoll(KSP, PetscBool);
PETSC_EXTERN PetscErrorCode KSPGetInitialGuessKnoll(KSP, PetscBool *);

PETSC_EXTERN PetscErrorCode MatCreateSchurComplement(Mat, Mat, Mat, Mat, Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatSchurComplementGetKSP(Mat, KSP *);
PETSC_EXTERN PetscErrorCode MatSchurComplementSetKSP(Mat, KSP);
PETSC_EXTERN PetscErrorCode MatSchurComplementSetSubMatrices(Mat, Mat, Mat, Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatSchurComplementUpdateSubMatrices(Mat, Mat, Mat, Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode MatSchurComplementGetSubMatrices(Mat, Mat *, Mat *, Mat *, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode MatSchurComplementSetAinvType(Mat, MatSchurComplementAinvType);
PETSC_EXTERN PetscErrorCode MatSchurComplementGetAinvType(Mat, MatSchurComplementAinvType *);
PETSC_EXTERN PetscErrorCode MatSchurComplementGetPmat(Mat, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatSchurComplementComputeExplicitOperator(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatGetSchurComplement(Mat, IS, IS, IS, IS, MatReuse, Mat *, MatSchurComplementAinvType, MatReuse, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateSchurComplementPmat(Mat, Mat, Mat, Mat, MatSchurComplementAinvType, MatReuse, Mat *);

PETSC_EXTERN PetscErrorCode MatCreateLMVMDFP(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMBFGS(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMDBFGS(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMDDFP(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMDQN(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMSR1(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMBroyden(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMBadBroyden(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMSymBroyden(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMSymBadBroyden(MPI_Comm, PetscInt, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode MatCreateLMVMDiagBroyden(MPI_Comm, PetscInt, PetscInt, Mat *);

PETSC_EXTERN PetscErrorCode MatLMVMUpdate(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatLMVMIsAllocated(Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatLMVMAllocate(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatLMVMReset(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatLMVMResetShift(Mat);
PETSC_EXTERN PetscErrorCode MatLMVMClearJ0(Mat);
PETSC_EXTERN PetscErrorCode MatLMVMSetJ0(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatLMVMSetJ0Scale(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatLMVMSetJ0Diag(Mat, Vec);
PETSC_EXTERN PetscErrorCode MatLMVMSetJ0PC(Mat, PC);
PETSC_EXTERN PetscErrorCode MatLMVMSetJ0KSP(Mat, KSP);
PETSC_EXTERN PetscErrorCode MatLMVMApplyJ0Fwd(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatLMVMApplyJ0Inv(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatLMVMGetLastUpdate(Mat, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode MatLMVMGetJ0(Mat, Mat *);
PETSC_EXTERN PetscErrorCode MatLMVMGetJ0PC(Mat, PC *);
PETSC_EXTERN PetscErrorCode MatLMVMGetJ0KSP(Mat, KSP *);
PETSC_EXTERN PetscErrorCode MatLMVMSetHistorySize(Mat, PetscInt);
PETSC_EXTERN PetscErrorCode MatLMVMGetHistorySize(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatLMVMGetUpdateCount(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatLMVMGetRejectCount(Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode MatLMVMSymBroydenSetDelta(Mat, PetscScalar);

PETSC_EXTERN PetscErrorCode MatLMVMSetMultAlgorithm(Mat, MatLMVMMultAlgorithm);
PETSC_EXTERN PetscErrorCode MatLMVMGetMultAlgorithm(Mat, MatLMVMMultAlgorithm *);

PETSC_EXTERN PetscErrorCode MatLMVMSymBroydenSetScaleType(Mat, MatLMVMSymBroydenScaleType);
PETSC_EXTERN PetscErrorCode MatLMVMSymBroydenGetPhi(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatLMVMSymBroydenSetPhi(Mat, PetscReal);
PETSC_EXTERN PetscErrorCode MatLMVMSymBadBroydenGetPsi(Mat, PetscReal *);
PETSC_EXTERN PetscErrorCode MatLMVMSymBadBroydenSetPsi(Mat, PetscReal);

PETSC_EXTERN PetscErrorCode MatLMVMDenseSetType(Mat, MatLMVMDenseType);

PETSC_EXTERN PetscErrorCode KSPSetDM(KSP, DM);
PETSC_EXTERN PetscErrorCode KSPGetDM(KSP, DM *);
PETSC_EXTERN PetscErrorCode KSPSetDMActive(KSP, KSPDMActive, PetscBool);
PETSC_EXTERN PetscErrorCode KSPSetApplicationContext(KSP, PetscCtx);
PETSC_EXTERN PetscErrorCode KSPGetApplicationContext(KSP, PetscCtxRt);

PETSC_EXTERN PetscErrorCode KSPSetComputeRHS(KSP, KSPComputeRHSFn *, void *);
PETSC_EXTERN PetscErrorCode KSPSetComputeOperators(KSP, KSPComputeOperatorsFn, void *);
PETSC_EXTERN PetscErrorCode KSPSetComputeInitialGuess(KSP, KSPComputeInitialGuessFn *, void *);
PETSC_EXTERN PetscErrorCode DMKSPSetComputeOperators(DM, KSPComputeOperatorsFn *, void *);
PETSC_EXTERN PetscErrorCode DMKSPGetComputeOperators(DM, KSPComputeOperatorsFn **, void *);
PETSC_EXTERN PetscErrorCode DMKSPSetCreateOperators(DM, KSPCreateOperatorsFn *, void *);
PETSC_EXTERN PetscErrorCode DMKSPGetCreateOperators(DM, KSPCreateOperatorsFn **, void *);
PETSC_EXTERN PetscErrorCode DMKSPSetComputeRHS(DM, KSPComputeRHSFn *, void *);
PETSC_EXTERN PetscErrorCode DMKSPGetComputeRHS(DM, KSPComputeRHSFn **, void *);
PETSC_EXTERN PetscErrorCode DMKSPSetComputeInitialGuess(DM, KSPComputeInitialGuessFn *, void *);
PETSC_EXTERN PetscErrorCode DMKSPGetComputeInitialGuess(DM, KSPComputeInitialGuessFn **, void *);

PETSC_EXTERN PetscErrorCode DMGlobalToLocalSolve(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMSwarmProjectFields(DM, DM, PetscInt, const char *[], Vec[], ScatterMode);
PETSC_EXTERN PetscErrorCode DMSwarmProjectGradientFields(DM, DM, PetscInt, const char *[], Vec[], ScatterMode);

PETSC_EXTERN PetscErrorCode DMAdaptInterpolator(DM, DM, Mat, KSP, Mat, Mat, Mat *, void *);
PETSC_EXTERN PetscErrorCode DMCheckInterpolator(DM, Mat, Mat, Mat, PetscReal);

PETSC_EXTERN PetscErrorCode PCBJKOKKOSSetKSP(PC, KSP);
PETSC_EXTERN PetscErrorCode PCBJKOKKOSGetKSP(PC, KSP *);

PETSC_EXTERN PetscErrorCode DMCopyDMKSP(DM, DM);

#include <petscdstypes.h>
PETSC_EXTERN PetscErrorCode DMProjectField(DM, PetscReal, Vec, PetscPointFn **, InsertMode, Vec);
