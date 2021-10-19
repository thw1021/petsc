
/*  --------------------------------------------------------------------

     This file implements a MAGx preconditioner in PETSc as part of PC.

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
  PetscBool fixdiag;             /* fix zero diagonal terms */
} PC_AMGx;

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

  if (!diag) {
    ierr = PetscInfo1(pc,"diag=NULL diagsqrt=%p\n",diagsqrt);CHKERRQ(ierr);
  }
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
    ierr = PCSetUp_AMGx(pc);CHKERRQ(ierr);
  } else {
    ierr = VecPointwiseMult(y,x,jac->diag);CHKERRQ(ierr);
  }
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
    PetscBool         fixdiag;
    PetscViewerFormat format;

    ierr = PCAMGxGetFixDiagonal(pc, &fixdiag);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, !fixdiag ? "not checking null diagonal entries" : "");CHKERRQ(ierr);
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
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxSetFixDiagonal_C",PCAMGxSetFixDiagonal_AMGx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCAMGxGetFixDiagonal_C",PCAMGxGetFixDiagonal_AMGx);CHKERRQ(ierr);
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
