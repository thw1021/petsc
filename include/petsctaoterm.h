#pragma once

#include <petscmat.h>
#include <petsctaotypes.h>

/* MANSEC = Tao */
/* SUBMANSEC = TaoTerm */

/*S
  TaoTerm - Abstract PETSc object for a parametric real-valued function that can be (optionally)
            combined with other terms to define the complete `Tao` objective function.

  Level: beginner

  Note:
  User can combine a user defined `TaoTerm` and a built-in types to define the objective function.

.seealso: [](ch_tao), [](sec_tao_term),
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermView()`,
          `TaoTermObjective()`,
          `TaoTermGradient()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessian()`,
          `TaoTermHessianMult()`,
          `TaoTermDestroy()`,
S*/
typedef struct _p_TaoTerm *TaoTerm;

/*J
  TaoTermType - String with the name of a `TaoTerm` method

  Values:
+ `TAOTERMTAOCALLBACKS`    - uses the callback functions set in `TaoSetObjective()`, `TaoSetGradient()`, etc.
. `TAOTERMBRGNREGULARIZER` - uses the callback functions set in `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()`, etc.
. `TAOTERMADMMREGULARIZER` - uses the callback functions set in `TaoADMMSetRegularizerObjectiveAndGradientRoutine()`, etc.
. `TAOTERMADMMMISFIT`      - uses the callback functions set in `TaoADMMSetMisfitObjectiveAndGradientRoutine()`, etc.
. `TAOTERMSHELL`           - a container for arbitrary user-defined callbacks
. `TAOTERMSUM`             - a sum of multiple other `TaoTerm`s
. `TAOTERMHALFL2SQUARED`   - $\tfrac{1}{2}\|x - p\|_2^2$
- `TAOTERML1`              - $\|x - p\|_1$

  Level: intermediate

.seealso: [](ch_tao), [](sec_tao_term), `TaoTerm`, `TaoTermCreate()`, `TaoTermSetType()`
J*/
typedef const char *TaoTermType;
#define TAOTERMTAOCALLBACKS    "taocallbacks"
#define TAOTERMBRGNREGULARIZER "brgnregularizer"
#define TAOTERMADMMREGULARIZER "admmregularizer"
#define TAOTERMADMMMISFIT      "admmmisfit"
#define TAOTERMSHELL           "shell"
#define TAOTERMSUM             "sum"
#define TAOTERMHALFL2SQUARED   "halfl2squared"
#define TAOTERML1              "l1"

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
  TaoTermParametersMode - Ways a `TaoTerm` can accept parameter vectors in `TaoTermObjective()` and related functions

  Values:
+ `TAOTERM_PARAMETERS_OPTIONAL` - the term has default parameters that will be used if parameters are omitted
. `TAOTERM_PARAMETERS_NONE`     - the term is not parametric, passing parameters is an error
- `TAOTERM_PARAMETERS_REQUIRED` - the term requires parameters, omitting parameters is an error

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermGetParametersMode()`, `TaoTermSetParametersMode()`
E*/
typedef enum {
  TAOTERM_PARAMETERS_OPTIONAL,
  TAOTERM_PARAMETERS_NONE,
  TAOTERM_PARAMETERS_REQUIRED
} TaoTermParametersMode;
PETSC_EXTERN const char *const TaoTermParametersModes[];

/*E
  TaoTermMask - Determine which evaluation operations are masked.

  Values:
+ `TAOTERM_MASK_NONE`      - do not mask any evaluation routines
. `TAOTERM_MASK_OBJECTIVE` - override the term's objective function and return 0 instead
. `TAOTERM_MASK_GRADIENT`  - override the term's gradient and return a zero vector instead
- `TAOTERM_MASK_HESSIAN`   - override the term's Hessian and return a zero matrix instead

  Level: advanced

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermSumSetSubtermMask()`, `TaoTermSumGetSubtermMask()`
E*/
typedef enum {
  TAOTERM_MASK_NONE      = 0, /* 0x0 */
  TAOTERM_MASK_OBJECTIVE = 1, /* 0x1 */
  TAOTERM_MASK_GRADIENT  = 2, /* 0x2 */
  TAOTERM_MASK_HESSIAN   = 4  /* 0x4 */
} TaoTermMask;

PETSC_EXTERN PetscErrorCode TaoTermSetParametersMode(TaoTerm, TaoTermParametersMode);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersMode(TaoTerm, TaoTermParametersMode *);

/*E
  TaoTermDuplicateOption - Aspects to preserve when duplicating a `TaoTerm`

  Values:
+ `TAOTERM_DUPLICATE_SIZEONLY` - size of the solution space only, user must call `TaoTermSetType()`
- `TAOTERM_DUPLICATE_TYPE`     - `TaoTermType` preserved

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

PETSC_EXTERN PetscErrorCode TaoTermCreateShell(MPI_Comm, void *, PetscCtxDestroyFn *, TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetContext(TaoTerm, void *);
PETSC_EXTERN PetscErrorCode TaoTermShellGetContext(TaoTerm, void *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetContextDestroy(TaoTerm, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode TaoTermShellSetObjective(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, PetscReal *));
PETSC_EXTERN PetscErrorCode TaoTermShellSetGradient(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, Vec));
PETSC_EXTERN PetscErrorCode TaoTermShellSetObjectiveAndGradient(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, PetscReal *, Vec));
PETSC_EXTERN PetscErrorCode TaoTermShellSetHessian(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, Mat, Mat));
PETSC_EXTERN PetscErrorCode TaoTermShellSetHessianMult(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec, Vec, Vec, Vec));
PETSC_EXTERN PetscErrorCode TaoTermShellSetView(TaoTerm, PetscErrorCode (*)(TaoTerm, PetscViewer));
PETSC_EXTERN PetscErrorCode TaoTermShellSetCreateVecs(TaoTerm, PetscErrorCode (*)(TaoTerm, Vec *, Vec *));
PETSC_EXTERN PetscErrorCode TaoTermShellSetCreateHessianMatrices(TaoTerm, PetscErrorCode (*)(TaoTerm, Mat *, Mat *));

PETSC_EXTERN PetscErrorCode TaoTermSumSetNumSubterms(TaoTerm, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermSumGetNumSubterms(TaoTerm, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSumSetSubterm(TaoTerm, PetscInt, const char[], PetscReal, TaoTerm, Mat);
PETSC_EXTERN PetscErrorCode TaoTermSumGetSubterm(TaoTerm, PetscInt, const char **, PetscReal *, TaoTerm *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermSumAddSubterm(TaoTerm, const char[], PetscReal, TaoTerm, Mat, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSumParametersPack(TaoTerm, Vec[], Vec *);
PETSC_EXTERN PetscErrorCode TaoTermSumParametersUnpack(TaoTerm, Vec *, Vec[]);
PETSC_EXTERN PetscErrorCode VecNestGetTaoTermSumSubParameters(Vec, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode TaoTermSumSetSubtermHessianMatrices(TaoTerm, PetscInt, Mat, Mat, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoTermSumGetSubtermMask(TaoTerm, PetscInt, TaoTermMask *);
PETSC_EXTERN PetscErrorCode TaoTermSumSetSubtermMask(TaoTerm, PetscInt, TaoTermMask);

PETSC_EXTERN PetscErrorCode TaoTermCreateHalfL2Squared(MPI_Comm, PetscInt, PetscInt, TaoTerm *);

PETSC_EXTERN PetscErrorCode TaoTermCreateL1(MPI_Comm, PetscInt, PetscInt, PetscReal, TaoTerm *);
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

PETSC_EXTERN PetscErrorCode TaoTermGetFDDelta(TaoTerm, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoTermSetFDDelta(TaoTerm, PetscReal);
PETSC_EXTERN PetscErrorCode TaoTermGradientFD(TaoTerm, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermGradientUseFDPush(TaoTerm);
PETSC_EXTERN PetscErrorCode TaoTermGradientUseFDPop(TaoTerm);
PETSC_EXTERN PetscErrorCode TaoTermSetHessianColoring(TaoTerm, ISColoring);
PETSC_EXTERN PetscErrorCode TaoTermGetHessianColoring(TaoTerm, ISColoring *);
PETSC_EXTERN PetscErrorCode TaoTermHessianSingleFD(TaoTerm, Vec, Vec, Mat);
PETSC_EXTERN PetscErrorCode TaoTermHessianFD(TaoTerm, Vec, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoTermHessianMultFD(TaoTerm, Vec, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermHessianUseFDPush(TaoTerm);
PETSC_EXTERN PetscErrorCode TaoTermHessianUseFDPop(TaoTerm);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMFFD(TaoTerm, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermUpdateHessianMFFD(TaoTerm, Mat, Vec, Vec);
