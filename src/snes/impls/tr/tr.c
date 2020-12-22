
#include <../src/snes/impls/tr/trimpl.h>                /*I   "petscsnes.h"   I*/

typedef struct {
  SNES           snes;
  /*  Information on the regular SNES convergence test; which may have been user provided */
  PetscErrorCode (*convtest)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);
  PetscErrorCode (*convdestroy)(void*);
  void           *convctx;
} SNES_TR_KSPConverged_Ctx;

static PetscErrorCode SNESTR_KSPConverged_Private(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *cctx)
{
  SNES_TR_KSPConverged_Ctx *ctx = (SNES_TR_KSPConverged_Ctx*)cctx;
  SNES                     snes = ctx->snes;
  SNES_NEWTONTR            *neP = (SNES_NEWTONTR*)snes->data;
  Vec                      x;
  PetscReal                nrm;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = (*ctx->convtest)(ksp,n,rnorm,reason,ctx->convctx);CHKERRQ(ierr);
  if (*reason) {
    ierr = PetscInfo2(snes,"Default or user provided convergence test KSP iterations=%D, rnorm=%g\n",n,(double)rnorm);CHKERRQ(ierr);
  }
  /* Determine norm of solution */
  ierr = KSPBuildSolution(ksp,NULL,&x);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm >= neP->delta) {
    ierr    = PetscInfo2(snes,"Ending linear iteration early, delta=%g, length=%g\n",(double)neP->delta,(double)nrm);CHKERRQ(ierr);
    *reason = KSP_CONVERGED_STEP_LENGTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTR_KSPConverged_Destroy(void *cctx)
{
  SNES_TR_KSPConverged_Ctx *ctx = (SNES_TR_KSPConverged_Ctx*)cctx;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = (*ctx->convdestroy)(ctx->convctx);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
/*
   SNESTR_Converged_Private -test convergence JUST for
   the trust region tolerance.

*/
static PetscErrorCode SNESTR_Converged_Private(SNES snes,PetscInt it,PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  SNES_NEWTONTR  *neP = (SNES_NEWTONTR*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *reason = SNES_CONVERGED_ITERATING;
  if (neP->delta < xnorm * snes->deltatol) {
    ierr    = PetscInfo3(snes,"Converged due to trust region param %g<%g*%g\n",(double)neP->delta,(double)xnorm,(double)snes->deltatol);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_TR_DELTA;
  } else if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    ierr    = PetscInfo1(snes,"Exceeded maximum number of function evaluations: %D\n",snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }
  PetscFunctionReturn(0);
}


PetscErrorCode  SNESNewtonTRGetRhoFlag(SNES snes, PetscBool *rho_flag)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidBoolPointer(rho_flag,2);
  *rho_flag = tr->rho_satisfied;
  PetscFunctionReturn(0);
}


/*@C
   SNESNewtonTRSetPreCheck - Sets a user function that is called before the search step has been determined.
       Allows the user a chance to change or override the decision of the line search routine.

   Logically Collective on snes

   Input Parameters:
+  snes - the nonlinear solver object
.  func - [optional] function evaluation routine, see SNESNewtonTRPreCheck()  for the calling sequence
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

   Note: This function is called BEFORE the function evaluation within the SNESNEWTONTR solver.

.seealso: SNESNewtonTRPreCheck(), SNESNewtonTRGetPreCheck(), SNESNewtonTRSetPostCheck(), SNESNewtonTRGetPostCheck()
@*/
PetscErrorCode  SNESNewtonTRSetPreCheck(SNES snes, PetscErrorCode (*func)(SNES,Vec,Vec,PetscBool*,void*),void *ctx)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) tr->precheck    = func;
  if (ctx)  tr->precheckctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRGetPreCheck - Gets the pre-check function

   Not collective

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  func - [optional] function evaluation routine, see for the calling sequence SNESNewtonTRPreCheck()
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

.seealso: SNESNewtonTRSetPreCheck(), SNESNewtonTRPreCheck()
@*/
PetscErrorCode  SNESNewtonTRGetPreCheck(SNES snes, PetscErrorCode (**func)(SNES,Vec,Vec,PetscBool*,void*),void **ctx)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) *func = tr->precheck;
  if (ctx)  *ctx  = tr->precheckctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRSetPostCheck - Sets a user function that is called after the search step has been determined but before the next
       function evaluation. Allows the user a chance to change or override the decision of the line search routine

   Logically Collective on snes

   Input Parameters:
+  snes - the nonlinear solver object
.  func - [optional] function evaluation routine, see SNESNewtonTRPostCheck()  for the calling sequence
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

   Note: This function is called BEFORE the function evaluation within the SNESNEWTONTR solver while the function set in
   SNESLineSearchSetPostCheck() is called AFTER the function evaluation.

.seealso: SNESNewtonTRPostCheck(), SNESNewtonTRGetPostCheck()
@*/
PetscErrorCode  SNESNewtonTRSetPostCheck(SNES snes, PetscErrorCode (*func)(SNES,Vec,Vec,Vec,PetscBool*,PetscBool*,void*),void *ctx)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) tr->postcheck    = func;
  if (ctx)  tr->postcheckctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRGetPostCheck - Gets the post-check function

   Not collective

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  func - [optional] function evaluation routine, see for the calling sequence SNESNewtonTRPostCheck()
-  ctx  - [optional] user-defined context for private data for the function evaluation routine (may be NULL)

   Level: intermediate

.seealso: SNESNewtonTRSetPostCheck(), SNESNewtonTRPostCheck()
@*/
PetscErrorCode  SNESNewtonTRGetPostCheck(SNES snes, PetscErrorCode (**func)(SNES,Vec,Vec,Vec,PetscBool*,PetscBool*,void*),void **ctx)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (func) *func = tr->postcheck;
  if (ctx)  *ctx  = tr->postcheckctx;
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRPreCheck - Called before the step has been determined in SNESNEWTONTR

   Logically Collective on snes

   Input Parameters:
+  snes - the solver
.  X - The last solution
-  Y - The step direction

   Output Parameters:
.  changed_Y - Indicator that the step direction Y has been changed.

   Level: developer

.seealso: SNESNewtonTRSetPreCheck(), SNESNewtonTRGetPreCheck()
@*/
static PetscErrorCode SNESNewtonTRPreCheck(SNES snes,Vec X,Vec Y,PetscBool *changed_Y)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  if (tr->precheck) {
    ierr = (*tr->precheck)(snes,X,Y,changed_Y,tr->precheckctx);CHKERRQ(ierr);
    PetscValidLogicalCollectiveBool(snes,*changed_Y,4);
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESNewtonTRPostCheck - Called after the step has been determined in SNESNEWTONTR but before the function evaluation

   Logically Collective on snes

   Input Parameters:
+  snes - the solver
.  X - The last solution
.  Y - The full step direction
-  W - The updated solution, W = X - Y

   Output Parameters:
+  changed_Y - indicator if step has been changed
-  changed_W - Indicator if the new candidate solution W has been changed.

   Notes:
     If Y is changed then W is recomputed as X - Y

   Level: developer

.seealso: SNESNewtonTRSetPostCheck(), SNESNewtonTRGetPostCheck()
@*/
static PetscErrorCode SNESNewtonTRPostCheck(SNES snes,Vec X,Vec Y,Vec W,PetscBool *changed_Y,PetscBool *changed_W)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *changed_Y = PETSC_FALSE;
  *changed_W = PETSC_FALSE;
  if (tr->postcheck) {
    ierr = (*tr->postcheck)(snes,X,Y,W,changed_Y,changed_W,tr->postcheckctx);CHKERRQ(ierr);
    PetscValidLogicalCollectiveBool(snes,*changed_Y,5);
    PetscValidLogicalCollectiveBool(snes,*changed_W,6);
  }
  PetscFunctionReturn(0);
}

/*
   SNESSolve_NEWTONTR - Implements Newton's Method with a very simple trust
   region approach for solving systems of nonlinear equations.

*/
static PetscErrorCode SNESSolve_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR            *neP = (SNES_NEWTONTR*)snes->data;
  Vec                      X,F,Y,G,Ytmp,W,GradF,YNtmp;
  PetscScalar    *X_mon, *Y_mon, *F_mon, *W_mon, *G_mon, *YN_mon, *YC_mon; /* debug */
  Vec                      Diag;
  Vec                      YCtmp;
//  PetscReal                *inorms;
  Mat                      jac;
  PetscErrorCode           ierr;
  PetscInt                 maxits,i,j,lits,bs,inner_count;
  PetscReal                rho,fnorm,gnorm,xnorm=0,delta,ynorm,temp_xnorm,temp_ynorm;
  PetscReal                deltaM,ynnorm,f0,mp,gTy,g,nscale;
  PetscReal                auk,gfnorm,ycnorm,c0,c1,c2,tau,tau_pos,tau_neg,gTBg;
  PetscReal                yTHy;
  KSP                      ksp;
  SNESConvergedReason      reason = SNES_CONVERGED_ITERATING;
  PetscBool                breakout = PETSC_FALSE;
  SNES_TR_KSPConverged_Ctx *ctx;
  PetscErrorCode           (*convtest)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),(*convdestroy)(void*);
  void                     *convctx;

  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* update vector */
  G      = snes->work[1];               /* updated residual */
  W      = snes->work[2];               /* temporary vector */
  GradF  = snes->work[3];               /* grad f = J^T F */
  YNtmp  = snes->work[4];               /* Newton solution */
  YCtmp  = snes->work[5];               /* Cauchy solution */

  /* for multiphase scaling */
  ierr = VecGetBlockSize(YNtmp,&bs);
  PetscReal inorms[bs];    
  neP->rho_satisfied = PETSC_FALSE;
    
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  /* Set the linear stopping criteria to use the More' trick. */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetConvergenceTest(ksp,&convtest,&convctx,&convdestroy);CHKERRQ(ierr);
  if (convtest != SNESTR_KSPConverged_Private) {
    ierr                  = PetscNew(&ctx);CHKERRQ(ierr);
    ctx->snes             = snes;
    ierr                  = KSPGetAndClearConvergenceTest(ksp,&ctx->convtest,&ctx->convctx,&ctx->convdestroy);CHKERRQ(ierr);
    ierr                  = KSPSetConvergenceTest(ksp,SNESTR_KSPConverged_Private,ctx,SNESTR_KSPConverged_Destroy);CHKERRQ(ierr);
    ierr                  = PetscInfo(snes,"Using Krylov convergence test SNESTR_KSPConverged_Private\n");CHKERRQ(ierr);
  }

  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);          /* F(X) */
  } else snes->vec_func_init_set = PETSC_FALSE;

  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);             /* fnorm <- || F || */
  SNESCheckFunctionNorm(snes,fnorm);
  ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);             /* xnorm <- || X || */
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  delta      = xnorm ? neP->delta0*xnorm : neP->delta0;  /* initial trust region size scaled by xnorm */
  deltaM     = xnorm ? neP->deltaM*xnorm : neP->deltaM;  /* maximum trust region size scaled by xnorm */
  neP->delta = delta;
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {
    PetscBool changed_y;
    PetscBool changed_w;

     /* dogleg method */
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    SNESCheckJacobianDomainerror(snes);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian);CHKERRQ(ierr);
    ierr = KSPSolve(snes->ksp,F,YNtmp);CHKERRQ(ierr);   /* Quasi Newton Solution */
    SNESCheckKSPSolve(snes);  /* this is necessary but old tr.c did not have it either*/
    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes, &jac, NULL, NULL, NULL);CHKERRQ(ierr);
    
    /* rescale Jacobian, Newton solution update, and re-calculate delta for multiphase */
    if (bs > 1 && neP->auto_scale_multiphase) {
      ierr = VecStrideNormAll(YNtmp,NORM_INFINITY,inorms);CHKERRQ(ierr);
      for (j=0; j<bs; j++) {
//        if (inorms[j] < 1.0e-5) {
//          inorms[j] = 1.0e-5;
//        }
        ierr = VecStrideSet(W, j, inorms[j]);CHKERRQ(ierr);
        ierr = VecStrideScale(YNtmp, j, 1.0/inorms[j]);
        ierr = VecStrideScale(X, j, 1.0/inorms[j]);
      }
      ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);
      if (i==0) {
        delta = neP->delta0*xnorm;
      } else {
        delta = neP->delta*xnorm;
      }
      deltaM = neP->deltaM*xnorm;
      ierr = MatDiagonalScale(jac,PETSC_NULL,W);CHKERRQ(ierr);
    }

    /* monitors for debugging */
    VecGetArrayRead(X,(const PetscScalar**)&X_mon);CHKERRQ(ierr);
    VecGetArrayRead(YNtmp,(const PetscScalar**)&YN_mon);CHKERRQ(ierr);
    VecGetArrayRead(F,(const PetscScalar**)&F_mon);CHKERRQ(ierr);
    VecGetArrayRead(W,(const PetscScalar**)&W_mon);CHKERRQ(ierr);
    
    VecRestoreArrayRead(X,(const PetscScalar**)&X_mon);CHKERRQ(ierr);
    VecRestoreArrayRead(YNtmp,(const PetscScalar**)&YN_mon);CHKERRQ(ierr);
    VecRestoreArrayRead(F,(const PetscScalar**)&F_mon);CHKERRQ(ierr);
    VecRestoreArrayRead(W,(const PetscScalar**)&W_mon);CHKERRQ(ierr);

    /* calculating GradF of minimization function */
    ierr = MatMultTranspose(jac,F,GradF);CHKERRQ(ierr);  /* grad f = J^T F */
    ierr = VecNorm(YNtmp,NORM_2,&ynnorm);CHKERRQ(ierr);  /* ynnorm <- || Y_newton || */
    
    inner_count = 0;
    neP->rho_satisfied = PETSC_FALSE;
    while (1) {

      if (ynnorm <= delta) {  /* see if the Newton solution is with in the trust region */
        ierr = VecCopy(YNtmp, Y);CHKERRQ(ierr);
      } else if (neP->use_cauchy) { /* use Cauchy direction if enabled */
        ierr = MatMult(jac,GradF,W);CHKERRQ(ierr); 
        ierr = VecDotRealPart(W,W,&gTBg);CHKERRQ(ierr);  /* completes GradF^T J^T J GradF */
        ierr = VecNorm(GradF,NORM_2,&gfnorm);CHKERRQ(ierr);  /* grad f norm <- || grad f || */
        if (gTBg <= 0.0) {
          auk = 1.0E20;
        } else {
          auk = PetscSqr(gfnorm)/gTBg;
        }
        auk = PetscMin(delta/gfnorm,auk);
        ierr = VecCopy(GradF, YCtmp);CHKERRQ(ierr); /*improve*/
        ierr = VecScale(YCtmp, auk);CHKERRQ(ierr);  /* YCtmp, Cauchy solution */
        ierr = VecNorm(YCtmp,NORM_2,&ycnorm);CHKERRQ(ierr);  /* ycnorm <- || Y_cauchy || */

        /* monitors for debugging */
        VecGetArrayRead(X,(const PetscScalar**)&X_mon);CHKERRQ(ierr);
        VecGetArrayRead(YNtmp,(const PetscScalar**)&YN_mon);CHKERRQ(ierr);
        VecGetArrayRead(YCtmp,(const PetscScalar**)&YC_mon);CHKERRQ(ierr);
        VecGetArrayRead(F,(const PetscScalar**)&F_mon);CHKERRQ(ierr);

        VecRestoreArrayRead(X,(const PetscScalar**)&X_mon);CHKERRQ(ierr);
        VecRestoreArrayRead(YNtmp,(const PetscScalar**)&YN_mon);CHKERRQ(ierr);
        VecRestoreArrayRead(YCtmp,(const PetscScalar**)&YC_mon);CHKERRQ(ierr);
        VecRestoreArrayRead(F,(const PetscScalar**)&F_mon);CHKERRQ(ierr);
        
        if (ycnorm >= delta) {  /* see if the Cauchy solution meets the criteria */
            ierr = VecCopy(YCtmp, Y);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD, "DL evaluated. delta: %8.4e, ynnorm: %8.4e, ycnorm: %8.4e\n", (double)delta, (double)ynnorm, (double)ycnorm);CHKERRQ(ierr);
        } else {  /* take ratio, tau, of Cauchy and Newton direction and step */
          ierr = VecAXPY(YNtmp,-1.0,YCtmp);CHKERRQ(ierr);  /* YCtmp = A, YNtmp = B */
          ierr = VecNorm(YNtmp,NORM_2,&c0);CHKERRQ(ierr); /*improve*/
          c0 = PetscSqr(c0);
          ierr = VecDotRealPart(YCtmp,YNtmp,&c1);CHKERRQ(ierr);
          c1 = 2.0*c1;
          ierr = VecNorm(YCtmp,NORM_2,&c2);CHKERRQ(ierr); /*improve*/
          c2 = PetscSqr(c2) - PetscSqr(delta);
          tau_pos = (c1 + PetscSqrtReal(PetscSqr(c1) - 4.*c0*c2)) / (2.*c0); /* quadratic formula */
          tau_neg = (c1 - PetscSqrtReal(PetscSqr(c1) - 4.*c0*c2)) / (2.*c0);
          tau = PetscMax(tau_pos, tau_neg);  /* can tau_neg > tau_pos? I don't think so, but just in case. */
          ierr = PetscPrintf(PETSC_COMM_WORLD, "DL evaluated. tau: %8.4e, ynnorm: %8.4e, ycnorm: %8.4e\n", (double)tau, (double)ynnorm, (double)ycnorm);CHKERRQ(ierr);
          ierr = VecWAXPY(W,tau,YNtmp,YCtmp);CHKERRQ(ierr);
          ierr = VecCopy(W, Y);CHKERRQ(ierr); /*improve*/
        }
      } else {
        /* if Cauchy is disabled, only use Newton direction */
        auk = delta/ynnorm;
        ierr = VecScale(YNtmp, auk);CHKERRQ(ierr);
        ierr = VecCopy(YNtmp, Y);CHKERRQ(ierr); /*improve*/
      }

      /* monitors for debugging */
      VecGetArrayRead(X,(const PetscScalar**)&X_mon);CHKERRQ(ierr);
      VecGetArrayRead(Y,(const PetscScalar**)&Y_mon);CHKERRQ(ierr);
      VecGetArrayRead(YNtmp,(const PetscScalar**)&YN_mon);CHKERRQ(ierr);
      VecGetArrayRead(YCtmp,(const PetscScalar**)&YC_mon);CHKERRQ(ierr);
      VecGetArrayRead(F,(const PetscScalar**)&F_mon);CHKERRQ(ierr);

      ierr = VecNorm(Y,NORM_2,&ynorm);CHKERRQ(ierr);  /* compute the final ynorm  */
      f0 = 0.5*PetscSqr(fnorm);  /* minimizing function f(X) */
      ierr = MatMult(jac,Y,W);CHKERRQ(ierr);  
      ierr = VecDotRealPart(W,W,&yTHy);CHKERRQ(ierr);  /* completes GradY^T J^T J GradY */
      ierr = VecDotRealPart(GradF,Y,&gTy);CHKERRQ(ierr);
      mp = f0 - gTy + 0.5*yTHy;  /* quadratic model to satisfy, -gTy because our update is X-Y*/

      /* scale back solution update */
      if (bs > 1 && neP->auto_scale_multiphase) {
        for (j=0; j<bs; j++) {
          ierr = VecStrideScale(Y, j, inorms[j]);
          if (inner_count == 0) {
            /* TR inner algorithm does not need scaled X after calculating delta in outer iteration */
            /* need to scale back X to match Y and provide proper update to the external code */
            ierr = VecStrideScale(X, j, inorms[j]);
          }
        }
        if (inner_count == 0) ierr = VecNorm(X,NORM_2,&temp_xnorm);CHKERRQ(ierr);
        ierr = VecNorm(Y,NORM_2,&temp_ynorm);CHKERRQ(ierr);
      } else {
        temp_xnorm = xnorm;
        temp_ynorm = ynorm;
      }
      inner_count++;

      /* Evaluate the solution to meet the improvement ratio criteria */
      ierr = SNESNewtonTRPreCheck(snes,X,Y,&changed_y);CHKERRQ(ierr);
      ierr = VecWAXPY(W,-1.0,Y,X);CHKERRQ(ierr);
      ierr = SNESNewtonTRPostCheck(snes,X,Y,W,&changed_y,&changed_w);CHKERRQ(ierr);
      if (changed_y) ierr = VecWAXPY(W,-1.0,Y,X);CHKERRQ(ierr);
      ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr); /*  F(X-Y) = G */
      ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);      /* gnorm <- || g || */
      SNESCheckFunctionNorm(snes,gnorm);
      g = 0.5*PetscSqr(gnorm); /* minimizing function g(W) */
      if (f0 == mp) rho = 0.0;
      else rho = (f0 - g)/(f0 - mp);  /* actual improvement over predicted improvement */

      /* monitors for debugging */
      VecGetArrayRead(W,(const PetscScalar**)&W_mon);CHKERRQ(ierr);
      VecGetArrayRead(G,(const PetscScalar**)&G_mon);CHKERRQ(ierr);
  
      if (rho < neP->eta2) {
        delta *= neP->t1;  /* shrink the region */
      } else if (rho > neP->eta3) {
        delta = PetscMin(neP->t2*delta,deltaM); /* expand the region, but not greater than deltaM */
      }

      /* monitors for debugging */ 
      VecRestoreArrayRead(X,(const PetscScalar**)&X_mon);CHKERRQ(ierr);
      VecRestoreArrayRead(Y,(const PetscScalar**)&Y_mon);CHKERRQ(ierr);
      VecRestoreArrayRead(YNtmp,(const PetscScalar**)&YN_mon);CHKERRQ(ierr);
      VecRestoreArrayRead(YCtmp,(const PetscScalar**)&YC_mon);CHKERRQ(ierr);
      VecRestoreArrayRead(F,(const PetscScalar**)&F_mon);CHKERRQ(ierr);
      VecRestoreArrayRead(W,(const PetscScalar**)&W_mon);CHKERRQ(ierr);
      VecRestoreArrayRead(G,(const PetscScalar**)&G_mon);CHKERRQ(ierr);

      neP->delta = delta;
      if (rho >= neP->eta1) {
        /* unscale delta and xnorm before going to the next outer iteration */
        if (bs > 1 && neP->auto_scale_multiphase) {
          neP->delta = delta/xnorm;
          xnorm = temp_xnorm;
          ynorm = temp_ynorm;
        }
        neP->rho_satisfied = PETSC_TRUE;
        break;  /* the improvement ratio is satisfactory */
      }
      ierr = PetscInfo(snes,"Trying again in smaller region\n");CHKERRQ(ierr);
  
      /* check to see if progress is hopeless */
      neP->itflag = PETSC_FALSE;
      /* both delta, ynorm, and xnorm are either scaled or unscaled */
      ierr        = SNESTR_Converged_Private(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      if (!reason) {
         /* temp_xnorm, temp_ynorm is always unscaled */
         /* also the inner iteration already calculated the Jacobian and solved the matrix */
         /* therefore, it should be passing iteration number of 1 instead of 0 in the first iteration */
         ierr = (*snes->ops->converged)(snes,snes->iter+1,temp_xnorm,temp_ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      }
      /* if multiphase state changes, break out inner iteration */
      if (reason == SNES_BREAKOUT_INNER_ITER) {
        if (bs > 1 && neP->auto_scale_multiphase) {
          /* unscale delta and xnorm before going to the next outer iteration */
          neP->delta = delta/xnorm;
          xnorm = temp_xnorm;
          ynorm = temp_ynorm;
        }
        reason = SNES_CONVERGED_ITERATING; 
        break;  
      }
      if (reason == SNES_CONVERGED_SNORM_RELATIVE) reason = SNES_DIVERGED_INNER;
      if (reason) {
        if (reason < 0) {
            /* We're not progressing, so return with the current iterate */
            ierr     = SNESMonitor(snes,i+1,fnorm);CHKERRQ(ierr);
            breakout = PETSC_TRUE;
            break;
        } else if (reason > 0) {
            /* We're converged, so return with the current iterate and update solution */
            ierr     = SNESMonitor(snes,i+1,fnorm);CHKERRQ(ierr);
            breakout = PETSC_FALSE;
            break;
        }
      }
      snes->numFailures++;
    }
    if (!breakout) {
      /* Update function and solution vectors */
      fnorm = gnorm;
      ierr  = VecCopy(G,F);CHKERRQ(ierr);
      ierr  = VecCopy(W,X);CHKERRQ(ierr);
      /* Monitor convergence */
      ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
      snes->iter = i+1;
      snes->norm = fnorm;
      snes->xnorm = xnorm;
      snes->ynorm = ynorm;
      ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
      ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
      ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
      /* Test for convergence, xnorm = || X || */
      neP->itflag = PETSC_TRUE;
      if (snes->ops->converged != SNESConvergedSkip) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
      ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      if (reason) break;
    } else break;
  }

  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!reason) reason = SNES_DIVERGED_MAX_IT;
  }
  ierr         = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->reason = reason;
  ierr         = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  if (convtest != SNESTR_KSPConverged_Private) {
    ierr       = KSPGetAndClearConvergenceTest(ksp,&ctx->convtest,&ctx->convctx,&ctx->convdestroy);CHKERRQ(ierr);
    ierr       = PetscFree(ctx);CHKERRQ(ierr);
    ierr       = KSPSetConvergenceTest(ksp,convtest,convctx,convdestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode SNESSetUp_NEWTONTR(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSetWorkVecs(snes,6);CHKERRQ(ierr);
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESReset_NEWTONTR(SNES snes)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_NEWTONTR(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NEWTONTR(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode SNESSetFromOptions_NEWTONTR(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_NEWTONTR  *ctx = (SNES_NEWTONTR*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SNES trust region options for nonlinear equations");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_trtol","Trust region tolerance","SNESSetTrustRegionTolerance",snes->deltatol,&snes->deltatol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_eta1","eta1","None",ctx->eta1,&ctx->eta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_eta2","eta2","None",ctx->eta2,&ctx->eta2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_eta3","eta3","None",ctx->eta3,&ctx->eta3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_t1","t1","None",ctx->t1,&ctx->t1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_t2","t2","None",ctx->t2,&ctx->t2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_deltaM","deltaM","None",ctx->deltaM,&ctx->deltaM,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_tr_delta0","delta0","None",ctx->delta0,&ctx->delta0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_use_cauchy","use_cauchy","use Cauchy step and direction",ctx->use_cauchy,&ctx->use_cauchy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_auto_scale_multiphase","auto_scale_multiphase","Auto scaling for proper cauchy direction",ctx->auto_scale_multiphase,&ctx->auto_scale_multiphase,NULL);CHKERRQ(ierr);  
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_NEWTONTR(SNES snes,PetscViewer viewer)
{
  SNES_NEWTONTR  *tr = (SNES_NEWTONTR*)snes->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Trust region tolerance (-snes_trtol)\n",(double)snes->deltatol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  eta1=%g, eta2=%g, eta3=%g\n",(double)tr->eta1,(double)tr->eta2,(double)tr->eta3);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  delta0=%g, t1=%g, t2=%g, deltaM=%g\n",(double)tr->delta0,(double)tr->t1,(double)tr->t2,(double)tr->deltaM);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
/*MC
      SNESNEWTONTR - Newton based nonlinear solver that uses a trust region

   Options Database:
+    -snes_trtol <tol> - trust region tolerance
.    -snes_tr_mu <mu> - trust region parameter
.    -snes_tr_eta <eta> - trust region parameter
.    -snes_tr_sigma <sigma> - trust region parameter
.    -snes_tr_delta0 <delta0> -  initial size of the trust region is delta0*norm2(x)
.    -snes_tr_delta1 <delta1> - trust region parameter
.    -snes_tr_delta2 <delta2> - trust region parameter
-    -snes_tr_delta3 <delta3> - trust region parameter

   The basic algorithm is taken from "The Minpack Project", by More',
   Sorensen, Garbow, Hillstrom, pages 88-111 of "Sources and Development
   of Mathematical Software", Wayne Cowell, editor.

   Level: intermediate

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESSetTrustRegionTolerance()

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONTR(SNES snes)
{
  SNES_NEWTONTR  *neP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONTR;
  snes->ops->solve          = SNESSolve_NEWTONTR;
  snes->ops->destroy        = SNESDestroy_NEWTONTR;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONTR;
  snes->ops->view           = SNESView_NEWTONTR;
  snes->ops->reset          = SNESReset_NEWTONTR;

  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  ierr        = PetscNewLog(snes,&neP);CHKERRQ(ierr);
  snes->data  = (void*)neP;
  neP->delta  = 0.0;
  neP->delta0 = 0.1;
  neP->eta1   = 0.001;
  neP->eta2   = 0.25;
  neP->eta3   = 0.75;
  neP->t1     = 0.25;
  neP->t2     = 2.0;
  neP->deltaM = 0.5;
  neP->sigma  = 0.0001;
  neP->itflag = PETSC_FALSE;
  neP->rnorm0 = 0.0;
  neP->ttol   = 0.0;
  neP->use_cauchy = PETSC_TRUE;
  neP->auto_scale_multiphase = PETSC_FALSE;
  neP->rho_satisfied = PETSC_FALSE;
  snes->deltatol = 1.e-8;
  PetscFunctionReturn(0);
}
