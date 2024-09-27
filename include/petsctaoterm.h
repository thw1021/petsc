#pragma once

#include <petscmat.h>
#include <petsctaotypes.h>

/*J
  TaoTermType - String with the name of a `TaoTerm` method

  Values:
+ `TAOTERMTAOCALLBACKS`    - uses the callback functions set in `TaoSetObjective()`, `TaoSetGradient()`, etc.
+ `TAOTERMBRGNREGULARIZER` - uses the callback functions set in `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()`, etc.
. `TAOTERMADMMREGULARIZER` - uses the callback functions set in `TaoADMMSetRegularizerObjectiveAndGradientRoutine()`, etc.
. `TAOTERMADMMMISFIT`      - uses the callback functions set in `TaoADMMSetMisfitObjectiveAndGradientRoutine()`, etc.
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
PETSC_EXTERN PetscErrorCode TaoTermGetType(TaoTerm, TaoTermType *);
PETSC_EXTERN PetscErrorCode TaoTermSetType(TaoTerm, TaoTermType);
PETSC_EXTERN PetscErrorCode TaoTermSetFromOptions(TaoTerm);

/*E
    TaoTermParametersType - Ways a `TaoTerm` can accept parameter vectors in `TaoTermObjective()` and related functions

   Values:
+  `TAOTERM_PARAMETERS_OPTIONAL` - the term has default parameters that will be used if parameters are omitted
.  `TAOTERM_PARAMETERS_NONE`     - the term is not parametric, passing parameters is an error
-  `TAOTERM_PARAMETERS_REQUIRED` - the term requires parameters, omitting parameters is an error

   Level: intermediate

.seealso: `TaoTerm`, `TaoTermGetParametersType()`
E*/
typedef enum {
  TAOTERM_PARAMETERS_OPTIONAL,
  TAOTERM_PARAMETERS_NONE,
  TAOTERM_PARAMETERS_REQUIRED
} TaoTermParametersType;
PETSC_EXTERN const char *const TaoTermParametersTypes[];

PETSC_EXTERN PetscErrorCode TaoTermSetParametersType(TaoTerm, TaoTermParametersType);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersType(TaoTerm, TaoTermParametersType *);

/*E
    TaoTermDuplicateOption - Aspects to preserve when duplicating a `TaoTerm`

   Values:
+  `TAOTERM_DUPLICATE_SIZEONLY` - size of the solution space only, user must call `TaoTermSetType()`
-  `TAOTERM_DUPLICATE_TYPE`     - `TaoTermType` preserved

   Level: intermediate

.seealso: `TaoTerm`, `TaoTermDuplicate()`
E*/
typedef enum {
  TAOTERM_DUPLICATE_SIZEONLY,
  TAOTERM_DUPLICATE_TYPE,
} TaoTermDuplicateOption;

PETSC_EXTERN PetscErrorCode TaoTermDuplicate(TaoTerm, TaoTermDuplicateOption, TaoTerm *);

PETSC_EXTERN PetscErrorCode TaoTermSetSolutionSizes(TaoTerm, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermGetSolutionSizes(TaoTerm, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSetSolutionTemplate(TaoTerm, Vec);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersSizes(TaoTerm, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersSizes(TaoTerm, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersTemplate(TaoTerm, Vec);
PETSC_EXTERN PetscErrorCode TaoTermSetVecTypes(TaoTerm, VecType, VecType);
PETSC_EXTERN PetscErrorCode TaoTermGetVecTypes(TaoTerm, VecType *, VecType *);
PETSC_EXTERN PetscErrorCode TaoTermSetLayouts(TaoTerm, PetscLayout, PetscLayout);
PETSC_EXTERN PetscErrorCode TaoTermGetLayouts(TaoTerm, PetscLayout *, PetscLayout *);
PETSC_EXTERN PetscErrorCode TaoTermCreateVecs(TaoTerm, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMatrices(TaoTerm, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMatricesDefault(TaoTerm, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermSetCreateHessianMode(TaoTerm, PetscBool, MatType, MatType);
PETSC_EXTERN PetscErrorCode TaoTermGetCreateHessianMode(TaoTerm, PetscBool *, MatType *, MatType *);

PETSC_EXTERN PetscErrorCode TaoTermObjective(TaoTerm, Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoTermGradient(TaoTerm, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermObjectiveAndGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoTermHessian(TaoTerm, Vec, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoTermHessianSingle(TaoTerm, Vec, Vec, Mat, Mat, PetscErrorCode (*)(TaoTerm, Vec, Vec, Mat), MatStructure);
PETSC_EXTERN PetscErrorCode TaoTermHessianMult(TaoTerm, Vec, Vec, Vec, Vec);

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
PETSC_EXTERN PetscErrorCode TaoTermSumSetSubtermHessianMatrices(TaoTerm, PetscInt, Mat, Mat, Mat, Mat);

PETSC_EXTERN PetscErrorCode TaoTermL1SetEpsilon(TaoTerm, PetscReal);
PETSC_EXTERN PetscErrorCode TaoTermL1GetEpsilon(TaoTerm, PetscReal *);

PETSC_EXTERN PetscErrorCode TaoTermIsObjectiveDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsGradientDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsObjectiveAndGradientDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsHessianDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsCreateHessianMatricesDefined(TaoTerm, PetscBool *);

PETSC_EXTERN PetscErrorCode TaoTermCreateHessianShell(TaoTerm, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermUpdateHessianShell(TaoTerm, Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermUpdateHessianShells(TaoTerm, Vec, Vec, Mat *, Mat *);
