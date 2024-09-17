#pragma once

#include <petscmat.h>
#include <petsctaotypes.h>

/*J
  TaoTermType - String with the name of a `TaoTerm` method

  Values:
+ `TAOTERMTAOCALLBACKS`    - uses the callback functions set in `TaoSetObjective()`, `TaoSetGradient()`, etc.
+ `TAOTERMBRGNREGULARIZER` - uses the callback functions set in `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()`, etc.
+ `TAOTERMADMMREGULARIZER` - uses the callback functions set in `TaoADMMSetRegularizerObjectiveAndGradientRoutine()`, etc.
+ `TAOTERMADMMMISFIT`      - uses the callback functions set in `TaoADMMSetMisfitObjectiveAndGradientRoutine()`, etc.
. `TAOTERMSHELL`           - a container for arbitrary user-defined callbacks
. `TAOTERMSUM`             - a sum of multiple other `TaoTerm`s
. `TAOTERMDM`              - a term whose value and operations are determined by a `DM`
. `TAOTERML1`              - $\|x - \theta\|_1$, where $\theta$ are the `TaoTerm` parameters
. `TAOTERMLINF`            - $\|x - \theta\|_\infty$
. `TAOTERML2SQUARED`       - $\|x - \theta\|_2^2$
. `TAOTERMQUADRATIC`       - a quadratic form $(x - \theta)^T Q (x - \theta)$ for a matrix $Q$
- `TAOTERMKL`              - the KL divergence $D_{KL}(x \| \theta)$

  Level: advanced

.seealso: [](doc_taosolve), [](ch_tao), `TaoTerm`, `TaoTermCreate()`, `TaoTermSetType()`
J*/
typedef const char *TaoTermType;
#define TAOTERMTAOCALLBACKS    "taocallbacks"
#define TAOTERMBRGNREGULARIZER "brgnregularizer"
#define TAOTERMADMMREGULARIZER "admmregularizer"
#define TAOTERMADMMMISFIT      "admmmisfit"
#define TAOTERMSHELL           "shell"
#define TAOTERMSUM             "sum"
#define TAOTERMDM              "dm"
#define TAOTERML1              "l1"
#define TAOTERMLINF            "linf"
#define TAOTERMHALFL2SQUARED   "halfl2squared"
#define TAOTERMQUADRATIC       "quadratic"
#define TAOTERMKL              "kl"

PETSC_EXTERN PetscErrorCode TaoTermRegister(const char[], PetscErrorCode (*)(TaoTerm));

PETSC_EXTERN PetscClassId      TAOTERM_CLASSID;
PETSC_EXTERN PetscFunctionList TaoTermList;

PETSC_EXTERN PetscErrorCode TaoTermCreate(MPI_Comm, TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermDestroy(TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermView(TaoTerm, PetscViewer);
PETSC_EXTERN PetscErrorCode TaoTermSetUp(TaoTerm);
PETSC_EXTERN PetscErrorCode TaoTermSetType(TaoTerm, TaoTermType);
PETSC_EXTERN PetscErrorCode TaoTermSetFromOptions(TaoTerm);

PETSC_EXTERN PetscErrorCode TaoTermSetSolutionTemplate(TaoTerm, Vec);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersTemplate(TaoTerm, Vec);
PETSC_EXTERN PetscErrorCode TaoTermGetVecTypes(TaoTerm, VecType *, VecType *);
PETSC_EXTERN PetscErrorCode TaoTermGetLayouts(TaoTerm, PetscLayout *, PetscLayout *);
PETSC_EXTERN PetscErrorCode TaoTermCreateVecs(TaoTerm, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMatrices(TaoTerm, Mat *, Mat *);

PETSC_EXTERN PetscErrorCode TaoTermObjective(TaoTerm, Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoTermGradient(TaoTerm, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermObjectiveAndGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoTermHessian(TaoTerm, Vec, Vec, Mat, Mat);
//PETSC_EXTERN PetscErrorCode TaoTermProximalMap(TaoTerm, Vec, PetscReal, TaoTerm, Vec, PetscReal, Vec);

PETSC_EXTERN PetscErrorCode TaoTermCreateShell(MPI_Comm, void *, PetscErrorCode (*)(void *), TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetContext(TaoTerm, void *);
PETSC_EXTERN PetscErrorCode TaoTermShellGetContext(TaoTerm, void *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetContextDestroy(TaoTerm, PetscErrorCode (*)(void *));
PETSC_EXTERN PetscErrorCode TaoTermShellSetObjective(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, PetscReal *));
PETSC_EXTERN PetscErrorCode TaoTermShellSetGradient(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, Vec));
PETSC_EXTERN PetscErrorCode TaoTermShellSetObjectiveAndGradient(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, PetscReal *, Vec));
PETSC_EXTERN PetscErrorCode TaoTermShellSetHessian(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, Mat, Mat));
PETSC_EXTERN PetscErrorCode TaoTermShellSetProximalMap(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, PetscReal, TaoTerm, Vec, PetscReal, Vec));
PETSC_EXTERN PetscErrorCode TaoTermShellSetView(TaoTerm, PetscErrorCode (*)(TaoTerm, PetscViewer));
PETSC_EXTERN PetscErrorCode TaoTermShellSetCreateVecs(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec *, Vec *));
PETSC_EXTERN PetscErrorCode TaoTermShellSetCreateHessianMatrices(TaoTerm, PetscErrorCode (*)(TaoTerm, Mat *, Mat *));

PETSC_EXTERN PetscErrorCode TaoTermSumSetNumSubterms(TaoTerm, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermSumGetNumSubterms(TaoTerm, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSumSetSubterm(TaoTerm, PetscInt, const char[], PetscReal, TaoTerm, Mat);
PETSC_EXTERN PetscErrorCode TaoTermSumGetSubterm(TaoTerm, PetscInt, const char **, PetscReal *, TaoTerm *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermSumAddSubterm(TaoTerm, const char[], PetscReal, TaoTerm, Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSumConcatenateParameters(TaoTerm, Vec[], Vec *);
PETSC_EXTERN PetscErrorCode TaoTermSumGetParameters(TaoTerm, Vec, Vec **);
PETSC_EXTERN PetscErrorCode TaoTermSumRestoreParameters(TaoTerm, Vec, Vec **);

PETSC_EXTERN PetscErrorCode TaoTermIsObjectiveDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsGradientDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsObjectiveAndGradientDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsHessianDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsCreateHessianMatricesDefined(TaoTerm, PetscBool *);
