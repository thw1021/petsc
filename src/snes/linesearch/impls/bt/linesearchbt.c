#include <petsc/private/linesearchimpl.h> /*I  "petscsnes.h"  I*/
#include <petsc/private/snesimpl.h>

typedef struct {
  PetscReal alpha;        /* sufficient decrease parameter */
} SNESLineSearch_BT;

/*@
   SNESLineSearchBTSetAlpha - Sets the descent parameter, alpha, in the BT linesearch variant.

   Input Parameters:
+  linesearch - linesearch context
-  alpha - The descent parameter

   Level: intermediate

.seealso: SNESLineSearchSetLambda(), SNESLineSearchGetTolerances() SNESLINESEARCHBT
@*/
PetscErrorCode SNESLineSearchBTSetAlpha(SNESLineSearch linesearch, PetscReal alpha)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT*)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  bt->alpha = alpha;
  PetscFunctionReturn(0);
}

/*@
   SNESLineSearchBTGetAlpha - Gets the descent parameter, alpha, in the BT linesearch variant.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  alpha - The descent parameter

   Level: intermediate

.seealso: SNESLineSearchGetLambda(), SNESLineSearchGetTolerances() SNESLINESEARCHBT
@*/
PetscErrorCode SNESLineSearchBTGetAlpha(SNESLineSearch linesearch, PetscReal *alpha)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT*)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  *alpha = bt->alpha;
  PetscFunctionReturn(0);
}

static PetscErrorCode  SNESLineSearchApply_BT(SNESLineSearch linesearch)
{
  PetscBool         changed_y,changed_w;
  PetscErrorCode    ierr;
  Vec               X,F,Y,W,G,GradF,WN,GN,WC,GC,YCtmp,YNtmp;
  SNES              snes;
  PetscReal         fnorm, xnorm, ynorm, gnorm, ynnorm, gTBg, auk, gfnorm, gcnorm, gnnorm, gn, gc;
  PetscReal         lambda,lambdatemp,lambdaprev,minlambda,maxstep,initslope,alpha,stol;
  PetscReal         t1,t2,a,b,d;
  PetscReal         f;
  PetscReal         g,gprev;
  PetscViewer       monitor;
  PetscInt          max_its,count;
  SNESLineSearch_BT *bt = (SNESLineSearch_BT*)linesearch->data;
  Mat               jac;
  PetscErrorCode    (*objective)(SNES,Vec,PetscReal*,void*);

  PetscFunctionBegin;
  ierr = SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchGetDefaultMonitor(linesearch, &monitor);CHKERRQ(ierr);
  ierr = SNESLineSearchGetTolerances(linesearch,&minlambda,&maxstep,NULL,NULL,NULL,&max_its);CHKERRQ(ierr);
  ierr = SNESGetTolerances(snes,NULL,NULL,&stol,NULL,NULL);CHKERRQ(ierr);
  ierr = SNESGetObjective(snes,&objective,NULL);CHKERRQ(ierr);
  alpha = bt->alpha;

  ierr = SNESGetJacobian(snes, &jac, NULL, NULL, NULL);CHKERRQ(ierr);
  if (!jac && !objective) SETERRQ(PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "SNESLineSearchBT requires a Jacobian matrix");

  ierr = SNESLineSearchPreCheck(linesearch,X,Y,&changed_y);CHKERRQ(ierr);
  ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED);CHKERRQ(ierr);

  ierr = VecNormBegin(Y, NORM_2, &ynnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(X, NORM_2, &xnorm);CHKERRQ(ierr);
  ierr = VecNormEnd(Y, NORM_2, &ynnorm);CHKERRQ(ierr);
  ierr = VecNormEnd(X, NORM_2, &xnorm);CHKERRQ(ierr);

  ierr = VecDuplicate(Y,&GradF);CHKERRQ(ierr);
  ierr = VecDuplicate(W,&WN);CHKERRQ(ierr);
  ierr = VecDuplicate(G,&GN);CHKERRQ(ierr);
  ierr = VecDuplicate(W,&WC);CHKERRQ(ierr);
  ierr = VecDuplicate(G,&GC);CHKERRQ(ierr);
  ierr = VecDuplicate(Y,&YCtmp);CHKERRQ(ierr);
  ierr = VecDuplicate(Y,&YNtmp);CHKERRQ(ierr);

  ierr = VecCopy(Y,YNtmp);CHKERRQ(ierr);  /* copy Newton Solution */

  /* Cauchy (Steepest Descent) Solution */
  ierr = MatMultTranspose(jac,F,GradF);CHKERRQ(ierr);  /* GradF = grad f = J^T F */
  ierr = MatMult(jac,GradF,W);CHKERRQ(ierr);
  ierr = VecDotRealPart(W,W,&gTBg);CHKERRQ(ierr);  /* completes GradF^T J^T J GradF */
  ierr = VecNorm(GradF,NORM_2,&gfnorm);CHKERRQ(ierr);  /* grad f norm <- || grad f || */
  if (gTBg <= 0.0) {
    auk = 1.0E20;
  } else {
    auk = PetscSqr(gfnorm)/gTBg;
  }
  auk = PetscMin(lambda*xnorm/gfnorm,auk);
  ierr = VecCopy(GradF, YCtmp);CHKERRQ(ierr);
  ierr = VecScale(YCtmp, auk);CHKERRQ(ierr);

  /* GradF^T GradF / ||GradF|| */
  initslope = -gfnorm;

  f = 0.5*PetscSqr(fnorm);

  while (PETSC_TRUE) {
    ierr = VecWAXPY(WC,-lambda,YCtmp,X);CHKERRQ(ierr);
    ierr = (*linesearch->ops->snesfunc)(snes,WC,GC);CHKERRQ(ierr);
    ierr = VecNorm(GC,NORM_2,&gcnorm);CHKERRQ(ierr);
    gc = 0.5*PetscSqr(gcnorm);
    if (!PetscIsInfOrNanReal(gc)) break;
    if (lambda <= minlambda) {
      SNESCheckFunctionNorm(snes,gc);
    }
    lambda = .5*lambda;
  }

  /* Line search backtracking quadratic using Cauchy Solution */
  if (gc >= f + lambda*alpha*initslope) { /* insufficient reduction or step tolerance convergence */
      /* Fit points with quadratic */
      lambdatemp = -initslope/(2.0*(gc - f - lambda*initslope));
      lambdaprev = lambda;
      gprev      = gc;
      if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
      if (lambdatemp <= .1*lambda) lambda = .1*lambda;
      else                         lambda = lambdatemp;

      ierr  = VecWAXPY(WC,-lambda,YCtmp,X);CHKERRQ(ierr);
      ierr = (*linesearch->ops->snesfunc)(snes,WC,GC);CHKERRQ(ierr);
      gc = 0.5*PetscSqr(gnorm);
      if (PetscIsInfOrNanReal(gc)) {
        ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF);CHKERRQ(ierr);
        ierr = PetscInfo(snes,"Aborted due to Nan or Inf in function evaluation\n");CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
   }

  if (gc > f + lambda*alpha*initslope) { /* sufficient reduction */
    for (count = 0; count < max_its; count++) {
      if (lambda <= minlambda) {
        ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      /* quadratic fit */
      lambdatemp = -initslope/(2.0*(gc - f - initslope));
      if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
      if (lambdatemp <= .1*lambda) lambda     = .1*lambda;
      else                         lambda     = lambdatemp;
      
      ierr = VecWAXPY(WC,-lambda,YCtmp,X);CHKERRQ(ierr);
      
      if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
        ierr = PetscInfo1(snes,"Exceeded maximum function evaluations, while looking for good step length! %D \n",count);CHKERRQ(ierr);
        if (!objective) {
          ierr = PetscInfo5(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",
                            (double)fnorm,(double)gnorm,(double)ynorm,(double)lambda,(double)initslope);CHKERRQ(ierr);
        }
        ierr         = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
        PetscFunctionReturn(0);
      }
      ierr = (*linesearch->ops->snesfunc)(snes,WC,GC);CHKERRQ(ierr);
      ierr = VecNorm(GC,NORM_2,&gcnorm);CHKERRQ(ierr);
      gc = 0.5*PetscSqr(gcnorm);

      if (PetscIsInfOrNanReal(gc)) {
          ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF);CHKERRQ(ierr);
          ierr = PetscInfo(snes,"Aborted due to Nan or Inf in function evaluation\n");CHKERRQ(ierr);
          PetscFunctionReturn(0);
        }
      
      if (gc < f + lambda*alpha*initslope) {
        break;
      }
    }
  }

  ierr = VecWAXPY(WN,-1.0,YNtmp,X);CHKERRQ(ierr);
  ierr = (*linesearch->ops->snesfunc)(snes,WN,GN);CHKERRQ(ierr);
  ierr = VecNorm(GN,NORM_2,&gnnorm);CHKERRQ(ierr);
  gn = 0.5*PetscSqr(gnnorm);

  if (gn < gc) {
    ierr = VecCopy(YNtmp,Y);CHKERRQ(ierr);  /* copy Newton Solution */
    ierr = VecCopy(WN,W);CHKERRQ(ierr);
    ierr = VecCopy(GN,G);CHKERRQ(ierr);
    gnorm = gnnorm;
    lambda = 1.0;
  } else {
    ierr = VecCopy(YCtmp,Y);CHKERRQ(ierr);  /* copy Cauchy linesearch Solution */
    ierr = VecCopy(WC,W);CHKERRQ(ierr);
    ierr = VecCopy(GC,G);CHKERRQ(ierr);
    gnorm = gcnorm;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Cauchy accepted lambda: %14.12e, g: %14.12e\n", (double)lambda, (double)gn);CHKERRQ(ierr);    
  }
  
  /* postcheck */
  /* update Y to lambda*Y so that W is consistent with  X - lambda*Y */
  ierr = VecScale(Y,lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchPostCheck(linesearch,X,Y,W,&changed_y,&changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(W,-1.0,Y,X);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
    }
  }
  if (changed_y || changed_w || objective) { /* recompute the function norm if the step has changed or the objective isn't the norm */
    ierr = (*linesearch->ops->snesfunc)(snes,W,G);CHKERRQ(ierr);
    if (linesearch->ops->vinorm) {
      gnorm = fnorm;
      ierr  = (*linesearch->ops->vinorm)(snes, G, W, &gnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
    }
    ierr = VecNorm(Y,NORM_2,&ynorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(gnorm)) {
      ierr = SNESLineSearchSetReason(linesearch,SNES_LINESEARCH_FAILED_NANORINF);CHKERRQ(ierr);
      ierr = PetscInfo(snes,"Aborted due to Nan or Inf in function evaluation\n");CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  /* copy the solution over */
  ierr = VecCopy(W, X);CHKERRQ(ierr);
  ierr = VecCopy(G, F);CHKERRQ(ierr);
  ierr = VecNorm(X, NORM_2, &xnorm);CHKERRQ(ierr);
  ierr = SNESLineSearchSetLambda(linesearch, lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchSetNorms(linesearch, xnorm, gnorm, ynorm);CHKERRQ(ierr);

  ierr = VecDestroy(&GradF);CHKERRQ(ierr);
  ierr = VecDestroy(&WN);CHKERRQ(ierr);
  ierr = VecDestroy(&GN);CHKERRQ(ierr);
  ierr = VecDestroy(&WC);CHKERRQ(ierr);
  ierr = VecDestroy(&GC);CHKERRQ(ierr);
  ierr = VecDestroy(&YNtmp);CHKERRQ(ierr);
  ierr = VecDestroy(&YCtmp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode SNESLineSearchView_BT(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  SNESLineSearch_BT *bt = (SNESLineSearch_BT*)linesearch->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
      ierr = PetscViewerASCIIPrintf(viewer, "  interpolation: cubic\n");CHKERRQ(ierr);
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      ierr = PetscViewerASCIIPrintf(viewer, "  interpolation: quadratic\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "  alpha=%e\n", (double)bt->alpha);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESLineSearchDestroy_BT(SNESLineSearch linesearch)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(linesearch->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESLineSearchSetFromOptions_BT(PetscOptionItems *PetscOptionsObject,SNESLineSearch linesearch)
{
  PetscErrorCode    ierr;
  SNESLineSearch_BT *bt = (SNESLineSearch_BT*)linesearch->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SNESLineSearch BT options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_alpha",   "Descent tolerance",        "SNESLineSearchBT", bt->alpha, &bt->alpha, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   SNESLINESEARCHBT - Backtracking line search.

   This line search finds the minimum of a polynomial fitting of the L2 norm of the
   function or the objective function if it is provided with SNESSetObjective(). If this fit does not satisfy the conditions for progress, the interval shrinks
   and the fit is reattempted at most max_it times or until lambda is below minlambda.

   Options Database Keys:
+  -snes_linesearch_alpha <1e\-4> - slope descent parameter
.  -snes_linesearch_damping <1.0> - initial step length
.  -snes_linesearch_maxstep <length> - if the length the initial step is larger than this then the
                                       step is scaled back to be of this length at the beginning of the line search
.  -snes_linesearch_max_it <40> - maximum number of shrinking step
.  -snes_linesearch_minlambda <1e\-12> - minimum step length allowed
-  -snes_linesearch_order <cubic,quadratic> - order of the approximation

   Level: advanced

   Notes:
   This line search is taken from "Numerical Methods for Unconstrained
   Optimization and Nonlinear Equations" by Dennis and Schnabel, page 325.

   This line search will always produce a step that is less than or equal to, in length, the full step size.

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType()
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_BT(SNESLineSearch linesearch)
{

  SNESLineSearch_BT *bt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_BT;
  linesearch->ops->destroy        = SNESLineSearchDestroy_BT;
  linesearch->ops->setfromoptions = SNESLineSearchSetFromOptions_BT;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = SNESLineSearchView_BT;
  linesearch->ops->setup          = NULL;

  ierr = PetscNewLog(linesearch,&bt);CHKERRQ(ierr);

  linesearch->data    = (void*)bt;
  linesearch->max_its = 40;
  linesearch->order   = SNES_LINESEARCH_ORDER_CUBIC;
  bt->alpha           = 1e-4;
  PetscFunctionReturn(0);
}
