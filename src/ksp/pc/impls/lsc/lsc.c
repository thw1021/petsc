#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef struct {
  PetscBool allocated;
  KSP       kspL, kspMass;
  Vec       x0, x1, y1;
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
  PetscCall(MatCreateVecs(pc->pmat, &lsc->x0, NULL));
  PetscCall(MatCreateVecs(A, &lsc->x1, &lsc->y1));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &lsc->kspMass));
  PetscCall(KSPSetErrorIfNotConverged(lsc->kspMass, pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)lsc->kspMass, (PetscObject)pc, 1));
  PetscCall(KSPSetType(lsc->kspMass, KSPPREONLY));
  PetscCall(KSPSetOptionsPrefix(lsc->kspMass, ((PetscObject)pc)->prefix));
  PetscCall(KSPAppendOptionsPrefix(lsc->kspMass, "lsc_mass"));

  lsc->allocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_LSC(PC pc)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;
  Mat     L, Lp, Qscale;

  PetscFunctionBegin;
  PetscCall(PCLSCAllocate_Private(pc));

  /* Query for L operators */
  PetscCall(PetscObjectQuery((PetscObject)pc->mat, "LSC_L", (PetscObject *)&L));
  if (!L) PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_L", (PetscObject *)&L));
  PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_Lp", (PetscObject *)&Lp));
  if (!Lp) PetscCall(PetscObjectQuery((PetscObject)pc->mat, "LSC_Lp", (PetscObject *)&Lp));
  PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_Qscale", (PetscObject *)&Qscale));
  if (!Qscale) PetscCall(PetscObjectQuery((PetscObject)pc->mat, "LSC_Qscale", (PetscObject *)&Qscale));
  PetscCheck(L || Lp, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "The user must provide an L operator for LSC preconditioning");
  PetscCheck(Qscale, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "The user must provide a Q matrix for LSC preconditioning");
  if (!L && Lp)
    L = Lp;
  else if (L && !Lp)
    Lp = L;
  PetscCall(KSPSetOperators(lsc->kspL, L, Lp));
  PetscCall(KSPSetFromOptions(lsc->kspL));
  PetscCall(KSPSetOperators(lsc->kspMass, Qscale, Qscale));
  PetscCall(KSPSetFromOptions(lsc->kspMass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_LSC(PC pc, Vec x, Vec y)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;
  Mat     A, B, C;

  PetscFunctionBegin;
  PetscCall(MatSchurComplementGetSubMatrices(pc->mat, &A, NULL, &B, &C, NULL));
  PetscCall(KSPSolve(lsc->kspMass, x, lsc->x0));
  PetscCall(KSPCheckSolve(lsc->kspMass, pc, lsc->x0));
  PetscCall(MatMult(B, lsc->x0, lsc->x1));
  PetscCall(KSPSolve(lsc->kspL, lsc->x1, lsc->y1));
  PetscCall(KSPCheckSolve(lsc->kspL, pc, lsc->y1));
  PetscCall(MatMult(A, lsc->y1, lsc->x1));
  PetscCall(KSPSolve(lsc->kspL, lsc->x1, lsc->y1));
  PetscCall(KSPCheckSolve(lsc->kspL, pc, lsc->y1));
  PetscCall(MatMult(C, lsc->y1, lsc->x0));
  PetscCall(KSPSolve(lsc->kspMass, lsc->x0, y));
  PetscCall(KSPCheckSolve(lsc->kspMass, pc, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_LSC(PC pc)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&lsc->x0));
  PetscCall(VecDestroy(&lsc->x1));
  PetscCall(VecDestroy(&lsc->y1));
  PetscCall(KSPDestroy(&lsc->kspL));
  PetscCall(KSPDestroy(&lsc->kspMass));
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
  PetscFunctionBegin;
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
    if (jac->kspMass) {
      PetscCall(KSPView(jac->kspMass, viewer));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "PCLSC Mass KSP object not yet created, hence cannot display"));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCLSC - Preconditioning for Schur complements, based on Least Squares Commutators

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
