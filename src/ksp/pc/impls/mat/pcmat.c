
#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef struct _PCMAT {
  MatOperation apply;
} PC_Mat;

static PetscErrorCode PCApply_Mat(PC pc, Vec x, Vec y)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;

  PetscFunctionBegin;
  switch (pcmat->apply) {
  case MATOP_MULT:
    PetscCall(MatMult(pc->pmat, x, y));
    break;
  case MATOP_MULT_TRANSPOSE:
    PetscCall(MatMultTranspose(pc->pmat, x, y));
    break;
  case MATOP_SOLVE:
    PetscCall(MatSolve(pc->pmat, x, y));
    break;
  case MATOP_SOLVE_TRANSPOSE:
    PetscCall(MatSolveTranspose(pc->pmat, x, y));
    break;
  case MATOP_MULT_HERMITIAN_TRANSPOSE:
    PetscCall(MatMultHermitianTranspose(pc->pmat, x, y));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Operation cannot be PCApply operation for PCMAT");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApply_Mat(PC pc, Mat X, Mat Y)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;

  PetscFunctionBegin;
  switch (pcmat->apply) {
  case MATOP_MULT:
    PetscCall(MatMatMult(pc->pmat, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    break;
  case MATOP_MULT_TRANSPOSE:
    PetscCall(MatTransposeMatMult(pc->pmat, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    break;
  case MATOP_SOLVE:
    PetscCall(MatMatSolve(pc->pmat, X, Y));
    break;
  case MATOP_SOLVE_TRANSPOSE:
    PetscCall(MatMatSolveTranspose(pc->pmat, X, Y));
    break;
  case MATOP_MULT_HERMITIAN_TRANSPOSE: {
    Mat W;

    PetscCall(MatDuplicate(X, MAT_COPY_VALUES, &W));
    PetscCall(MatConjugate(W));
    PetscCall(MatTransposeMatMult(pc->pmat, W, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    PetscCall(MatConjugate(Y));
    PetscCall(MatDestroy(&W));
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Operation cannot be PCMatApply operation for PCMAT");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_Mat(PC pc, Vec x, Vec y)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;

  PetscFunctionBegin;
  switch (pcmat->apply) {
  case MATOP_MULT:
    PetscCall(MatMultTranspose(pc->pmat, x, y));
    break;
  case MATOP_MULT_TRANSPOSE:
    PetscCall(MatMult(pc->pmat, x, y));
    break;
  case MATOP_SOLVE:
    PetscCall(MatSolveTranspose(pc->pmat, x, y));
    break;
  case MATOP_SOLVE_TRANSPOSE:
    PetscCall(MatSolve(pc->pmat, x, y));
    break;
  case MATOP_MULT_HERMITIAN_TRANSPOSE: {
    Vec w;

    PetscCall(VecDuplicate(x, &w));
    PetscCall(VecCopy(x, w));
    PetscCall(VecConjugate(w));
    PetscCall(MatMult(pc->pmat, w, y));
    PetscCall(VecConjugate(y));
    PetscCall(VecDestroy(&w));
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Operation cannot be PCApplyTranspose operation for PCMAT");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Mat(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatSetApplyOperation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatGetApplyOperation_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMatSetApplyOperation - Set which matrix operation of the preconditioning matrix implements `PCApply()` for `PCMAT`.

  Logically collective

  Input Parameters:
+ pc - An instance of `PCMAT`
- matop - The selected `MatOperation`

  Level: intermediate

  Note:
  If you have a matrix type that implements an exact inverse that isn't a factorization,
  you can use `PCMatSetApplyOperation(pc, MATOP_SOLVE)`.

.seealso: `PCMAT`, `PCMatGetApplyOperation()`, `PCApply()`, `MatOperation`
@*/
PetscErrorCode PCMatSetApplyOperation(PC pc, MatOperation matop)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod((PetscObject)pc, "PCMatSetApplyOperation_C", (PC, MatOperation), (pc, matop));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMatGetApplyOperation - Get which matrix operation of the preconditioning matrix implements PCApply() for PCMAT.

  Logically collective

  Input Parameter:
. pc - An instance of `PCMAT`

  Output Parameter:
. matop - The `MatOperation`

  Level: intermediate

.seealso: `PCMAT`, `PCMatSetApplyOperation()`, `PCApply()`, `MatOperation`
@*/
PetscErrorCode PCMatGetApplyOperation(PC pc, MatOperation *matop)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod((PetscObject)pc, "PCMatGetApplyOperation_C", (PC, MatOperation *), (pc, matop));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatSetApplyOperation_Mat(PC pc, MatOperation matop)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;

  PetscFunctionBegin;
  switch (matop) {
  case MATOP_MULT:
  case MATOP_MULT_TRANSPOSE:
  case MATOP_SOLVE:
  case MATOP_SOLVE_TRANSPOSE:
  case MATOP_MULT_HERMITIAN_TRANSPOSE:
    pcmat->apply = matop;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_INCOMP, "Operation cannot be PCApply operation for PCMAT");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatGetApplyOperation_Mat(PC pc, MatOperation *matop)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;

  PetscFunctionBegin;
  *matop = pcmat->apply;
  PetscFunctionReturn(PETSC_SUCCESS);
}

enum {
  PCMAT_MULT,
  PCMAT_MULT_TRANSPOSE,
  PCMAT_MULT_HERMITIAN_TRANSPOSE,
  PCMAT_SOLVE,
  PCMAT_SOLVE_TRANSPOSE,
  PCMAT_UNSUPPORTED
};
const char *const PCMatOpTypes[] = {"Mult", "MultTranspose", "MultHermitianTranspose", "Solve", "SolveTranspose", "Unsupported"};

static PetscErrorCode PCView_Mat(PC pc, PetscViewer viewer)
{
  PC_Mat   *pcmat = (PC_Mat *)pc->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    size_t op = PCMAT_UNSUPPORTED;

#define MATOP_TO_PCMAT_CASE(var, OP) \
  case MATOP_##OP: \
    (var) = PCMAT_##OP; \
    break
    switch (pcmat->apply) {
      MATOP_TO_PCMAT_CASE(op, MULT);
      MATOP_TO_PCMAT_CASE(op, MULT_TRANSPOSE);
      MATOP_TO_PCMAT_CASE(op, MULT_HERMITIAN_TRANSPOSE);
      MATOP_TO_PCMAT_CASE(op, SOLVE);
      MATOP_TO_PCMAT_CASE(op, SOLVE_TRANSPOSE);
    default:
      SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "PCMat apply set to unsupported MatOperation %d", (int)pcmat->apply);
    }

    PetscCall(PetscViewerASCIIPrintf(viewer, "PCApply() == Mat%s()\n", PCMatOpTypes[op]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCMAT - A preconditioner obtained by applying an operation of the preconditioner matrix supplied
             in `PCSetOperators()` or `KSPSetOperators()`.  By default, the operation is `MATOP_MULT`,
             meaning that the preconditioning matrix implements an approximate inverse of the system matrix.
             If some other operation of preconditioner matrix implements the approximate inverse,
             use `PCMatSetApplyOperation()` to select that operation.

   Level: intermediate

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCSHELL`, `MatOperation`, `PCMatSetApplyOperation()`, `PCMatGetApplyOperation()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Mat(PC pc)
{
  PC_Mat *data;

  PetscFunctionBegin;
  PetscCall(PetscNew(&data));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatSetApplyOperation_C", PCMatSetApplyOperation_Mat));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatGetApplyOperation_C", PCMatGetApplyOperation_Mat));
  data->apply                  = MATOP_MULT;
  pc->data                     = data;
  pc->ops->apply               = PCApply_Mat;
  pc->ops->matapply            = PCMatApply_Mat;
  pc->ops->applytranspose      = PCApplyTranspose_Mat;
  pc->ops->setup               = NULL;
  pc->ops->destroy             = PCDestroy_Mat;
  pc->ops->setfromoptions      = NULL;
  pc->ops->view                = PCView_Mat;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
