#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef struct {
  PetscBool allocated;
  PetscBool scalediag;
  KSP       kspL;
  Vec       scale;
  Vec       x0, y0, x1;
  Mat       L;         /* keep a copy to reuse when obtained with L = A10*A01 */
  Mat       CAdiaginv; /* The product of C and the inverse of the diagional of A */
  Mat       A, B, C;
} PC_LSC;

static PetscErrorCode PCLSCAllocate_Private(PC pc)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;
  Mat     A;

  PetscFunctionBegin;
  if (lsc->allocated) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &lsc->kspL));
  PetscCall(KSPSetErrorIfNotConverged(lsc->kspL, pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)lsc->kspL, (PetscObject)pc, 1));
  PetscCall(KSPSetType(lsc->kspL, KSPPREONLY));
  PetscCall(KSPSetOptionsPrefix(lsc->kspL, ((PetscObject)pc)->prefix));
  PetscCall(KSPAppendOptionsPrefix(lsc->kspL, "lsc_"));
  PetscCall(MatSchurComplementGetSubMatrices(pc->mat, &A, NULL, NULL, NULL, NULL));
  PetscCall(MatCreateVecs(A, &lsc->x0, &lsc->y0));
  PetscCall(MatCreateVecs(pc->pmat, &lsc->x1, NULL));
  if (lsc->scalediag) PetscCall(VecDuplicate(lsc->x0, &lsc->scale));
  lsc->allocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_LSC(PC pc)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;
  Mat     L = NULL, Lp = NULL;

  PetscFunctionBegin;
  PetscCall(PCLSCAllocate_Private(pc));

  /* Query for L operators, e.b. C * Qvdiag^-1 * B */
  PetscCall(PetscObjectQuery((PetscObject)pc->mat, "LSC_L", (PetscObject *)&L));
  if (!L) PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_L", (PetscObject *)&L));
  PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_Lp", (PetscObject *)&Lp));
  if (!Lp) PetscCall(PetscObjectQuery((PetscObject)pc->mat, "LSC_Lp", (PetscObject *)&Lp));

  /* Query for user provided matrices. These may be different than those from the system matrix in the case that the user has constraints like Dirichlet boundary conditions encoded in the system matrix */
  PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_A", (PetscObject *)&lsc->A));
  if (!lsc->A) PetscCall(MatSchurComplementGetSubMatrices(pc->mat, NULL, &lsc->A, NULL, NULL, NULL));
  PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_B", (PetscObject *)&lsc->B));
  if (!lsc->B) PetscCall(MatSchurComplementGetSubMatrices(pc->mat, NULL, NULL, &lsc->B, NULL, NULL));
  PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_C", (PetscObject *)&lsc->C));
  if (!lsc->C) PetscCall(MatSchurComplementGetSubMatrices(pc->mat, NULL, NULL, NULL, &lsc->C, NULL));

  if (lsc->scale) {
    Mat Q_scale;
    PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_Q_scale", (PetscObject *)&Q_scale));
    if (!Q_scale) PetscCall(MatSchurComplementGetSubMatrices(pc->mat, NULL, &Q_scale, NULL, NULL, NULL));
    PetscCall(MatGetDiagonal(Q_scale, lsc->scale));
    PetscCall(VecReciprocal(lsc->scale));
  }
  if (!L) {
    if (lsc->scale) {
      if (!lsc->CAdiaginv) {
        PetscCall(MatConvert(lsc->C, MATSAME, MAT_INITIAL_MATRIX, &lsc->CAdiaginv));
      } else {
        PetscCall(MatCopy(lsc->C, lsc->CAdiaginv, SAME_NONZERO_PATTERN));
      }
      PetscCall(MatDiagonalScale(lsc->CAdiaginv, NULL, lsc->scale));
      if (!lsc->L) {
        PetscCall(MatMatMult(lsc->CAdiaginv, lsc->B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lsc->L));
      } else {
        PetscCall(MatMatMult(lsc->CAdiaginv, lsc->B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &lsc->L));
      }
    } else {
      if (!lsc->L) {
        PetscCall(MatMatMult(lsc->C, lsc->B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lsc->L));
      } else {
        PetscCall(MatMatMult(lsc->C, lsc->B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &lsc->L));
      }
    }
    Lp = L = lsc->L;
  }
  PetscCall(KSPSetOperators(lsc->kspL, L, Lp));
  PetscCall(KSPSetFromOptions(lsc->kspL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_LSC(PC pc, Vec x, Vec y)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPSolve(lsc->kspL, x, lsc->x1));
  PetscCall(KSPCheckSolve(lsc->kspL, pc, lsc->x1));
  PetscCall(MatMult(lsc->B, lsc->x1, lsc->x0));
  if (lsc->scale) PetscCall(VecPointwiseMult(lsc->x0, lsc->x0, lsc->scale));
  PetscCall(MatMult(lsc->A, lsc->x0, lsc->y0));
  if (lsc->scale) PetscCall(VecPointwiseMult(lsc->y0, lsc->y0, lsc->scale));
  PetscCall(MatMult(lsc->C, lsc->y0, lsc->x1));
  PetscCall(KSPSolve(lsc->kspL, lsc->x1, y));
  PetscCall(KSPCheckSolve(lsc->kspL, pc, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_LSC(PC pc)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&lsc->x0));
  PetscCall(VecDestroy(&lsc->y0));
  PetscCall(VecDestroy(&lsc->x1));
  PetscCall(VecDestroy(&lsc->scale));
  PetscCall(KSPDestroy(&lsc->kspL));
  PetscCall(MatDestroy(&lsc->L));
  PetscCall(MatDestroy(&lsc->CAdiaginv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_LSC(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_LSC(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_LSC(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "LSC options");
  {
    PetscCall(PetscOptionsBool("-pc_lsc_scale_diag", "Use diagonal of velocity block (A) for scaling", "None", lsc->scalediag, &lsc->scalediag, NULL));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_LSC(PC pc, PetscViewer viewer)
{
  PC_LSC   *jac = (PC_LSC *)pc->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (jac->kspL) {
      PetscCall(KSPView(jac->kspL, viewer));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "PCLSC KSP object not yet created, hence cannot display"));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCLSC - Preconditioning for Schur complements, based on Least Squares Commutators

   Options Database Key:
.    -pc_lsc_scale_diag - Use the diagonal of A for scaling

   Level: intermediate

   Notes:
   This preconditioner will normally be used with `PCFIELDSPLIT` to precondition the Schur complement, but
   it can be used for any Schur complement system.  Consider the Schur complement

.vb
   S = A11 - A10 inv(A00) A01
.ve

   `PCLSC` currently doesn't do anything with A11, so let's assume it is 0.  The idea is that a good approximation to
   inv(S) is given by

.vb
   inv(A10 A01) A10 A00 A01 inv(A10 A01)
.ve

   The product A10 A01 can be computed for you, but you can provide it (this is
   usually more efficient anyway).  In the case of incompressible flow, A10 A01 is a Laplacian; call it L.  The current
   interface is to hang L and a preconditioning matrix Lp on the preconditioning matrix.

   If you had called `KSPSetOperators`(ksp,S,Sp), S should have type `MATSCHURCOMPLEMENT` and Sp can be any type you
   like (`PCLSC` doesn't use it directly) but should have matrices composed with it, under the names "LSC_L" and "LSC_Lp".
   For example, you might have setup code like this

.vb
   PetscObjectCompose((PetscObject)Sp,"LSC_L",(PetscObject)L);
   PetscObjectCompose((PetscObject)Sp,"LSC_Lp",(PetscObject)Lp);
.ve

   And then your Jacobian assembly would look like

.vb
   PetscObjectQuery((PetscObject)Sp,"LSC_L",(PetscObject*)&L);
   PetscObjectQuery((PetscObject)Sp,"LSC_Lp",(PetscObject*)&Lp);
   if (L) { assembly L }
   if (Lp) { assemble Lp }
.ve

   With this, you should be able to choose LSC preconditioning, using e.g. ML's algebraic multigrid to solve with L

.vb
   -fieldsplit_1_pc_type lsc -fieldsplit_1_lsc_pc_type ml
.ve

   Since we do not use the values in Sp, you can still put an assembled matrix there to use normal preconditioners.

   References:
+  * - Elman, Howle, Shadid, Shuttleworth, and Tuminaro, Block preconditioners based on approximate commutators, 2006.
-  * - Silvester, Elman, Kay, Wathen, Efficient preconditioning of the linearized Navier Stokes equations for incompressible flow, 2001.

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `Block_Preconditioners`, `PCFIELDSPLIT`,
          `PCFieldSplitGetSubKSP()`, `PCFieldSplitSetFields()`, `PCFieldSplitSetType()`, `PCFieldSplitSetIS()`, `PCFieldSplitSetSchurPre()`,
          `MatCreateSchurComplement()`, `MatCreateSchurComplement()`, `MatSchurComplementSetSubMatrices()`, `MatSchurComplementUpdateSubMatrices()`,
          `MatSchurComplementSetAinvType()`, `MatGetSchurComplement()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_LSC(PC pc)
{
  PC_LSC *lsc;

  PetscFunctionBegin;
  PetscCall(PetscNew(&lsc));
  pc->data = (void *)lsc;

  pc->ops->apply           = PCApply_LSC;
  pc->ops->applytranspose  = NULL;
  pc->ops->setup           = PCSetUp_LSC;
  pc->ops->reset           = PCReset_LSC;
  pc->ops->destroy         = PCDestroy_LSC;
  pc->ops->setfromoptions  = PCSetFromOptions_LSC;
  pc->ops->view            = PCView_LSC;
  pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
