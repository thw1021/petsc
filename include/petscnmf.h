#pragma once

#include <petsctao.h>

/* SUBMANSEC = NMF */

/*E
   PetscNMFInitializeType - Determine initialization type for `PetscNMF`.
   TODO NOte that I am copying scikit-learn API

   Values:
+  `PETSCNMF_INIT_NONE`     - ‘nndsvda’ if rank <= min(n_samples, n_features), otherwise random.
.  `PETSCNMF_INIT_RANDOM`   - non-negative random matrices, scaled with: sqrt(X.mean() / rank)
.  `PETSCNMF_INIT_NNDSVD`   - Nonnegative Double Singular Value Decomposition (NNDSVD) initialization (better for sparseness)
.  `PETSCNMF_INIT_NNDSVDA`  - NNDSVD with zeros filled with the average of X (better when sparsity is not desired)
.  `PETSCNMF_INIT_NNDSVDAR` - NNDSVD with zeros filled with small random values (generally faster, less accurate alternative to NNDSVDa for when sparsity is not desired)
-  `PETSCNMF_INIT_CUSTOM`   - Use custom matrices W and H which must both be provided.

  Level: advanced

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFSetInitializationType()`
E*/
typedef enum {
  PETSCNMF_INIT_NONE,
  PETSCNMF_INIT_RANDOM,
  PETSCNMF_INIT_NNDSVD,
  PETSCNMF_INIT_NNDSVDA,
  PETSCNMF_INIT_NNDSVDAR,
  PETSCNMF_INIT_CUSTOM
} PetscNMFInitType;
PETSC_EXTERN const char *const PetscNMFInitTypes[];

/*E
   PetscNMFTermParameterType - Determine parameter type for `TaoTerm` used in `PetscNMF`.
   TODO TERM_PARAM?

   Values:
+  `PETSCNMF_PARAM_NONE` - Parameter is NULL
.  `PETSCNMF_PARAM_X`    - Parameter is Vec(X)
.  `PETSCNMF_PARAM_W`    - Parameter is Vec(W)
.  `PETSCNMF_PARAM_H`    - Parameter is Vec(H)
-  `PETSCNMF_PARAM_WH`   - Parameter is Vec(WH)

  Level: advanced

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFAddObjectiveTerm()`
E*/
typedef enum {
  PETSCNMF_PARAM_NONE,
  PETSCNMF_PARAM_X,
  PETSCNMF_PARAM_W,
  PETSCNMF_PARAM_H,
  PETSCNMF_PARAM_WH,
} PetscNMFTermParameterType;
PETSC_EXTERN const char *const PetscNMFTermParameterTypes[];

/*S
   PetscNMF  - Abstract PETSc object that manages non-negative matrix factorization problems

   Level: beginner

.seealso: [](ch_nmf), `PetscNMFCreate()`, `PetscNMFDestroy()`, `PetscNMFSetType()`, `PetscNMFType`
S*/
typedef struct _p_PetscNMF *PetscNMF;

/*J
  PetscNMFType - String with the name of a `PetscNMF` method

  Values:
+ `PETSCNMFAOADMM` - alternating objective ADMM
- `PETSCNMFHALS`   - hierarchical alternating least squares

  Level: beginner

.seealso: [](ch_nmf), `PetscNMFCreate()`, `PetscNMFDestroy()`, `PetscNMFSetType()`
J*/
typedef const char *PetscNMFType;
#define PETSCNMFAOADMM "aoadmm"
#define PETSCNMFHALS   "hals"

PETSC_EXTERN PetscClassId      PETSCNMF_CLASSID;
PETSC_EXTERN PetscFunctionList PetscNMFList;

//TODO NMF Converged Reason?

PETSC_EXTERN PetscErrorCode PetscNMFInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscNMFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscNMFCreate(MPI_Comm, PetscNMF *);
PETSC_EXTERN PetscErrorCode PetscNMFSetFromOptions(PetscNMF);
PETSC_EXTERN PetscErrorCode PetscNMFSetUp(PetscNMF);
PETSC_EXTERN PetscErrorCode PetscNMFSetType(PetscNMF, PetscNMFType);
PETSC_EXTERN PetscErrorCode PetscNMFGetType(PetscNMF, PetscNMFType *);
PETSC_EXTERN PetscErrorCode PetscNMFSetApplicationContext(PetscNMF, void *);
PETSC_EXTERN PetscErrorCode PetscNMFGetApplicationContext(PetscNMF, void *);
PETSC_EXTERN PetscErrorCode PetscNMFDestroy(PetscNMF *);
PETSC_EXTERN PetscErrorCode PetscNMFParametersInitialize(PetscNMF);

PETSC_EXTERN PetscErrorCode PetscNMFSetOptionsPrefix(PetscNMF, const char[]);
PETSC_EXTERN PetscErrorCode PetscNMFAppendOptionsPrefix(PetscNMF, const char[]);
PETSC_EXTERN PetscErrorCode PetscNMFGetOptionsPrefix(PetscNMF, const char *[]);
PETSC_EXTERN PetscErrorCode PetscNMFView(PetscNMF, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscNMFViewFromOptions(PetscNMF, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode PetscNMFSetMaximumIterations(PetscNMF, PetscInt);
PETSC_EXTERN PetscErrorCode PetscNMFGetMaximumIterations(PetscNMF, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscNMFGetIterationNumber(PetscNMF, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscNMFSetIterationNumber(PetscNMF, PetscInt);
PETSC_EXTERN PetscErrorCode PetscNMFGetTolerances(PetscNMF, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscNMFSetTolerances(PetscNMF, PetscReal);

PETSC_EXTERN PetscErrorCode PetscNMFSetInitializationType(PetscNMF, PetscNMFInitType);
PETSC_EXTERN PetscErrorCode PetscNMFGetInitializationType(PetscNMF, PetscNMFInitType *);

PETSC_EXTERN PetscErrorCode PetscNMFSetRandomContext(PetscNMF, PetscRandom);
PETSC_EXTERN PetscErrorCode PetscNMFGetRandomContext(PetscNMF, PetscRandom *);

// TaoTerm: HALFL2SQUARED, KL, IS, or beta-div (todo)
PETSC_EXTERN PetscErrorCode PetscNMFSetLossTerm(PetscNMF, TaoTerm);
PETSC_EXTERN PetscErrorCode PetscNMFGetLossTerm(PetscNMF, TaoTerm);
//TODO mat currently does nothing? Is there a case where Mat would be needed/useful?
//                                                                                      f(Ax;p)  x type                     p type
PETSC_EXTERN PetscErrorCode PetscNMFAddObjectiveTerm(PetscNMF, const char *, PetscReal, TaoTerm, PetscNMFTermParameterType, PetscNMFTermParameterType, Mat);

PETSC_EXTERN PetscErrorCode PetscNMFSetRank(PetscNMF, PetscInt);
PETSC_EXTERN PetscErrorCode PetscNMFGetRank(PetscNMF, PetscInt *);

//TODO are these needed for ADMM? or just CD?
//TODO want
// -nmf_w_scale
// -nmf_h_scale
// TAOTERML1
// Q: should these be automatic, or manual?
PETSC_EXTERN PetscErrorCode PetscNMFSetWScale(PetscNMF, PetscReal);
PETSC_EXTERN PetscErrorCode PetscNMFGetWScale(PetscNMF, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscNMFSetHScale(PetscNMF, PetscReal);
PETSC_EXTERN PetscErrorCode PetscNMFGetHScale(PetscNMF, PetscReal *);
// -nmf_l1_ratio
// TAOTERMHALFL2SQUARED
PETSC_EXTERN PetscErrorCode PetscNMFSetL1Ratio(PetscNMF, PetscReal);
PETSC_EXTERN PetscErrorCode PetscNMFGetL1Ratio(PetscNMF, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscNMFFit(PetscNMF);

PETSC_EXTERN PetscErrorCode PetscNMFSetTrainingMat(PetscNMF, Mat);
PETSC_EXTERN PetscErrorCode PetscNMFSetTransformMat(PetscNMF, Mat);
PETSC_EXTERN PetscErrorCode PetscNMFSetComponentsMat(PetscNMF, Mat);

//TODO W,H, mats - Mat or list of Vec?
//TODO do I need Set Components? prob some kind of warm-start init stuff
//Geting H matrix
PETSC_EXTERN PetscErrorCode PetscNMFGetComponentsMat(PetscNMF, Mat *);

PETSC_EXTERN PetscErrorCode PetscNMFTransform(PetscNMF);
PETSC_EXTERN PetscErrorCode PetscNMFGetTransformMat(PetscNMF, Mat *);

//TODO monitor stuff
//TODO history stuff

PETSC_EXTERN PetscErrorCode PetscNMFRegister(const char[], PetscErrorCode (*)(PetscNMF));
PETSC_EXTERN PetscErrorCode PetscNMFRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscNMFSetSolution(PetscNMF, Mat);
PETSC_EXTERN PetscErrorCode PetscNMFGetSolution(PetscNMF, Mat *);
