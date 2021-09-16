
/*  --------------------------------------------------------------------

     This file implements a Jacobi preconditioner in PETSc as part of PC.
     You can use this as a starting point for implementing your own
     preconditioner that is not provided with PETSc. (You might also consider
     just using PCSHELL)

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _Jacobi (e.g., PCCreate_Jacobi, PCApply_Jacobi).
     These routines are actually called via the common user interface
     routines PCCreate(), PCSetFromOptions(), PCApply(), and PCDestroy(),
     so the application code interface remains identical for all
     preconditioners.

     Another key routine is:
          PCSetUp_XXX()           - Prepares for the use of a preconditioner
     by setting data structures and options.   The interface routine PCSetUp()
     is not usually called directly by the user, but instead is called by
     PCApply() if necessary.

     Additional basic routines are:
          PCView_XXX()            - Prints details of runtime options that
                                    have actually been used.
     These are called by application codes via the interface routines
     PCView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  One exception is
     that the analogues of PCApply() for these components are KSPSolve(),
     SNESSolve(), and TSSolve().

     Additional optional functionality unique to preconditioners is left and
     right symmetric preconditioner application via PCApplySymmetricLeft()
     and PCApplySymmetricRight().  The Jacobi implementation is
     PCApplySymmetricLeftOrRight_Jacobi().

    -------------------------------------------------------------------- */

/*
   Include files needed for the AMGx preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

/*
   Private context (data structure) for the AMGx preconditioner.
*/
typedef struct {
  Vec diag;                      /* vector containing the reciprocals of the diagonal elements of the preconditioner matrix */
  Vec diagsqrt;                  /* vector containing the reciprocals of the square roots of
                                    the diagonal elements of the preconditioner matrix (used
                                    only for symmetric preconditioner application) */
  PetscBool userowmax;           /* set with PCAMGxSetType() */
  PetscBool userowsum;
  PetscBool useabs;              /* use the absolute values of the diagonal entries */
  PetscBool fixdiag;             /* fix zero diagonal terms */
} PC_AMGx;

static PetscErrorCode  PCAMGxSetUseAbs_AMGx(PC pc,PetscBool flg)
{
  PC_AMGx *j = (PC_AMGx*)pc->data;

  PetscFunctionBegin;
  j->useabs = flg;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCAMGxGetUseAbs_AMGx(PC pc,PetscBool *flg)
{
  PC_AMGx *j = (PC_AMGx*)pc->data;

  PetscFunctionBegin;
  *flg = j->useabs;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCAMGxSetFixDiagonal_AMGx(PC pc,PetscBool flg)
{
  PC_AMGx *j = (PC_AMGx*)pc->data;

  PetscFunctionBegin;
  j->fixdiag = flg;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCAMGxGetFixDiagonal_AMGx(PC pc,PetscBool *flg)
{
  PC_AMGx *j = (PC_AMGx*)pc->data;

  PetscFunctionBegin;
  *flg = j->fixdiag;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_AMGx - Prepares for the use of the AMGx preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_AMGx(PC pc)
{
  PC_AMGx      *jac = (PC_AMGx*)pc->data;
  Vec            diag,diagsqrt;
  PetscErrorCode ierr;
  PetscInt       n,i;
  PetscScalar    *x;
  PetscBool      zeroflag = PETSC_FALSE;

  PetscFunctionBegin;
  /*
       For most preconditioners the code would begin here something like

  if (pc->setupcalled == 0) { allocate space the first time this is ever called
    ierr = MatCreateVecs(pc->mat,&jac->diag);CHKERRQ(ierr);
    PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->diag);
  }

    But for this preconditioner we want to support use of both the matrix' diagonal
    elements (for left or right preconditioning) and square root of diagonal elements
    (for symmetric preconditioning).  Hence we do not allocate space here, since we
    don't know at this point which will be needed (diag and/or diagsqrt) until the user
    applies the preconditioner, and we don't want to allocate BOTH unless we need
    them both.  Thus, the diag and diagsqrt are allocated in PCSetUp_AMGx_NonSymmetric()
    and PCSetUp_AMGx_Symmetric(), respectively.
  */

  /*
    Here we set up the preconditioner; that is, we copy the diagonal values from
    the matrix and put them into a format to make them quick to apply as a preconditioner.
  */
  diag     = jac->diag;
  diagsqrt = jac->diagsqrt;

  if (diag) {
    PetscBool isspd;

    if (jac->userowmax) {
      ierr = MatGetRowMaxAbs(pc->pmat,diag,NULL);CHKERRQ(ierr);
    } else if (jac->userowsum) {
      ierr = MatGetRowSum(pc->pmat,diag);CHKERRQ(ierr);
    } else {
      ierr = MatGetDiagonal(pc->pmat,diag);CHKERRQ(ierr);
    }
    ierr = VecReciprocal(diag);CHKERRQ(ierr);
    if (jac->useabs) {
      ierr = VecAbs(diag);CHKERRQ(ierr);
    }
    ierr = MatGetOption(pc->pmat,MAT_SPD,&isspd);CHKERRQ(ierr);
    if (jac->fixdiag && !isspd) {
      ierr = VecGetLocalSize(diag,&n);CHKERRQ(ierr);
      ierr = VecGetArray(diag,&x);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        if (x[i] == 0.0) {
          x[i]     = 1.0;
          zeroflag = PETSC_TRUE;
        }
      }
      ierr = VecRestoreArray(diag,&x);CHKERRQ(ierr);
    }
  }
  if (diagsqrt) {
    if (jac->userowmax) {
      ierr = MatGetRowMaxAbs(pc->pmat,diagsqrt,NULL);CHKERRQ(ierr);
    } else if (jac->userowsum) {
      ierr = MatGetRowSum(pc->pmat,diagsqrt);CHKERRQ(ierr);
    } else {
      ierr = MatGetDiagonal(pc->pmat,diagsqrt);CHKERRQ(ierr);
    }
    ierr = VecGetLocalSize(diagsqrt,&n);CHKERRQ(ierr);
    ierr = VecGetArray(diagsqrt,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (x[i] != 0.0) x[i] = 1.0/PetscSqrtReal(PetscAbsScalar(x[i]));
      else {
        x[i]     = 1.0;
        zeroflag = PETSC_TRUE;
      }
    }
    ierr = VecRestoreArray(diagsqrt,&x);CHKERRQ(ierr);
  }
  if (zeroflag) {
    ierr = PetscInfo(pc,"Zero detected in diagonal of matrix, using 1 at those locations\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCSetUp_AMGx_Symmetric - Allocates the vector needed to store the
   inverse of the square root of the diagonal entries of the matrix.  This
   is used for symmetric application of the AMGx preconditioner.

   Input Parameter:
.  pc - the preconditioner context
*/
static PetscErrorCode PCSetUp_AMGx_Symmetric(PC pc)
{
  PetscErrorCode ierr;
  PC_AMGx      *jac = (PC_AMGx*)pc->data;

  PetscFunctionBegin;
  ierr = MatCreateVecs(pc->pmat,&jac->diagsqrt,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->diagsqrt);CHKERRQ(ierr);
  ierr = PCSetUp_AMGx(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCSetUp_AMGx_NonSymmetric - Allocates the vector needed to store the
   inverse of the diagonal entries of the matrix.  This is used for left of
   right application of the AMGx preconditioner.

   Input Parameter:
.  pc - the preconditioner context
*/
static PetscErrorCode PCSetUp_AMGx_NonSymmetric(PC pc)
{
  PetscErrorCode ierr;
  PC_AMGx      *jac = (PC_AMGx*)pc->data;

  PetscFunctionBegin;
  ierr = MatCreateVecs(pc->pmat,&jac->diag,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->diag);CHKERRQ(ierr);
  ierr = PCSetUp_AMGx(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCApply_AMGx - Applies the AMGx preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
static PetscErrorCode PCApply_AMGx(PC pc,Vec x,Vec y)
{
  PC_AMGx      *jac = (PC_AMGx*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac->diag) {
    ierr = PCSetUp_AMGx_NonSymmetric(pc);CHKERRQ(ierr);
  }
  ierr = VecPointwiseMult(y,x,jac->diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCApplySymmetricLeftOrRight_AMGx - Applies the left or right part of a
   symmetric preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routines: PCApplySymmetricLeft(), PCApplySymmetricRight()
*/
static PetscErrorCode PCApplySymmetricLeftOrRight_AMGx(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_AMGx      *jac = (PC_AMGx*)pc->data;

  PetscFunctionBegin;
  if (!jac->diagsqrt) {
    ierr = PCSetUp_AMGx_Symmetric(pc);CHKERRQ(ierr);
  }
  ierr = VecPointwiseMult(y,x,jac->diagsqrt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
static PetscErrorCode PCReset_AMGx(PC pc)
{
  PC_AMGx      *jac = (PC_AMGx*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&jac->diag);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->diagsqrt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   PCDestroy_AMGx - Destroys the private context for the AMGx preconditioner
   that was created with PCCreate_AMGx().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_AMGx(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_AMGx(pc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxSetUseAbs_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxGetUseAbs_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxSetFixDiagonal_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxGetFixDiagonal_C",NULL);CHKERRQ(ierr);

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_AMGx(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_AMGx      *jac = (PC_AMGx*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"AMGx options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_amgx_abs","Use absolute values of diagonal entries","PCAMGxSetUseAbs",jac->useabs,&jac->useabs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_amgx_fixdiagonal","Fix null terms on diagonal","PCAMGxSetFixDiagonal",jac->fixdiag,&jac->fixdiag,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_AMGx(PC pc, PetscViewer viewer)
{
  PC_AMGx     *jac = (PC_AMGx *) pc->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscBool         useAbs,fixdiag;
    PetscViewerFormat format;

    ierr = PCAMGxGetUseAbs(pc, &useAbs);CHKERRQ(ierr);
    ierr = PCAMGxGetFixDiagonal(pc, &fixdiag);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, useAbs ? ", using absolute value of entries" : "", !fixdiag ? ", not checking null diagonal entries" : "");CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = VecView(jac->diag, viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_AMGx - Creates a AMGx preconditioner context, PC_AMGx,
   and sets this as the private data within the generic preconditioning
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCAMGX - AMGx (i.e. diagonal scaling preconditioning)

   Options Database Key:
+    -pc_amgx_type <diagonal,rowmax,rowsum> - approach for forming the preconditioner
.    -pc_amgx_abs - use the absolute value of the diagonal entry
-    -pc_amgx_fixdiag - fix for zero diagonal terms

   Level: beginner

  Notes:
    By using KSPSetPCSide(ksp,PC_SYMMETRIC) or -ksp_pc_side symmetric
         can scale each side of the matrix by the square root of the diagonal entries.

         Zero entries along the diagonal are replaced with the value 1.0

         See PCPBAMGX for a point-block AMGx preconditioner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCAMGxSetType(), PCAMGxSetUseAbs(), PCAMGxGetUseAbs(),
           PCAMGxSetFixDiagonal(), PCAMGxGetFixDiagonal(), PCPBAMGX
M*/

PETSC_EXTERN PetscErrorCode PCCreate_AMGx(PC pc)
{
  PC_AMGx      *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr     = PetscNewLog(pc,&jac);CHKERRQ(ierr);
  pc->data = (void*)jac;

  /*
     Initialize the pointers to vectors to ZERO; these will be used to store
     diagonal entries of the matrix for fast preconditioner application.
  */
  jac->diag      = NULL;
  jac->diagsqrt  = NULL;
  jac->userowmax = PETSC_FALSE;
  jac->userowsum = PETSC_FALSE;
  jac->useabs    = PETSC_FALSE;
  jac->fixdiag   = PETSC_TRUE;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_AMGx;
  pc->ops->applytranspose      = PCApply_AMGx;
  pc->ops->setup               = PCSetUp_AMGx;
  pc->ops->reset               = PCReset_AMGx;
  pc->ops->destroy             = PCDestroy_AMGx;
  pc->ops->setfromoptions      = PCSetFromOptions_AMGx;
  pc->ops->view                = PCView_AMGx;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeftOrRight_AMGx;
  pc->ops->applysymmetricright = PCApplySymmetricLeftOrRight_AMGx;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxSetUseAbs_C",PCAMGxSetUseAbs_AMGx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxGetUseAbs_C",PCAMGxGetUseAbs_AMGx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxSetFixDiagonal_C",PCAMGxSetFixDiagonal_AMGx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxGetFixDiagonal_C",PCAMGxGetFixDiagonal_AMGx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCAMGxSetUseAbs - Causes the AMGx preconditioner to use the
      absolute values of the diagonal divisors in the preconditioner

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flg - whether to use absolute values or not

   Options Database Key:
.  -pc_amgx_abs

   Notes:
    This takes affect at the next construction of the preconditioner

   Level: intermediate

.seealso: PCAMGxaSetType(), PCAMGxGetUseAbs()

@*/
PetscErrorCode  PCAMGxSetUseAbs(PC pc,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCAMGxSetUseAbs_C",(PC,PetscBool),(pc,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCAMGxGetUseAbs - Determines if the AMGx preconditioner uses the
      absolute values of the diagonal divisors in the preconditioner

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  flg - whether to use absolute values or not

   Options Database Key:
.  -pc_amgx_abs

   Level: intermediate

.seealso: PCAMGxaSetType(), PCAMGxSetUseAbs(), PCAMGxGetType()

@*/
PetscErrorCode  PCAMGxGetUseAbs(PC pc,PetscBool *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCAMGxGetUseAbs_C",(PC,PetscBool*),(pc,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCAMGxSetFixDiagonal - Do not check for zero values on diagonal

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flg - the boolean flag

   Options Database Key:
.  -pc_amgx_fixdiagonal

   Notes:
    This takes affect at the next construction of the preconditioner

   Level: intermediate

.seealso: PCAMGxSetType(), PCAMGxGetFixDiagonal()

@*/
PetscErrorCode  PCAMGxSetFixDiagonal(PC pc,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCAMGxSetFixDiagonal_C",(PC,PetscBool),(pc,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCAMGxGetFixDiagonal - Determines if the AMGx preconditioner checks for zero diagonal terms

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  flg - the boolean flag

   Options Database Key:
.  -pc_amgx_fixdiagonal

   Level: intermediate

.seealso: PCAMGxSetType(), PCAMGxSetFixDiagonal()

@*/
PetscErrorCode  PCAMGxGetFixDiagonal(PC pc,PetscBool *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCAMGxGetFixDiagonal_C",(PC,PetscBool*),(pc,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
