
/*
    This file implements QMRCGS (QMRCGStab).
    Only allow right preconditioning.

    Contributed by: Xiangmin Jiao (xiangmin.jiao@stonybrook.edu)

    References: Chan, Gallopoulos, Simoncini, Szeto, and Tong (SISC 1994),
                Ghai, Lu, and Jiao (NLAA 2019)
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

static PetscErrorCode KSPSetUp_QMRCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,14);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Only need a few hacks from KSPSolve_BCGS */

static PetscErrorCode  KSPSolve_QMRCGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    tau,theta,eta,rho1,rho2,alpha,theta2,final,c,tau2,eta2,omega,beta,cf,cf1,uu,vv,F = 0.0,NV,nrmb;
  Vec            X,B,R,P,PH,V,D2,X2,S,SH,T,D,S2,RP,AX,Z;
  PetscReal      dp    = 0.0;
  KSP_BCGS       *bcgs = (KSP_BCGS*)ksp->data;
  PC             pc;
  Mat            mat;

  PetscFunctionBegin;
  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  P  = ksp->work[1];
  PH = ksp->work[2];
  V  = ksp->work[3];
  D2 = ksp->work[4];
  X2 = ksp->work[5];
  S  = ksp->work[6];
  SH = ksp->work[7];
  T  = ksp->work[8];
  D  = ksp->work[9];
  S2 = ksp->work[10];
  RP = ksp->work[11];
  AX = ksp->work[12];
  Z  = ksp->work[13];
  /*  Only supports right preconditioning */
  if (ksp->pc_side != PC_RIGHT) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP qmrcgs does not support %s",PCSides[ksp->pc_side]);
  if (!ksp->guess_zero) {
    if (!bcgs->guess) {
      ierr = VecDuplicate(X,&bcgs->guess);CHKERRQ(ierr);
    }
    ierr = VecCopy(X,bcgs->guess);CHKERRQ(ierr);
  } else {
    ierr = VecSet(X,0.0);CHKERRQ(ierr);
  }

  /* Compute initial residual */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetUp(pc);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&mat,NULL);CHKERRQ(ierr);
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,mat,X,S2);CHKERRQ(ierr);
    ierr = VecCopy(B,R);CHKERRQ(ierr);
    ierr = VecAXPY(R,-1.0,S2);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);
  }

  ierr = VecNorm(B,NORM_2,&nrmb);CHKERRQ(ierr);

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  }
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP);CHKERRQ(ierr);

  eta   = 1.0;
  theta = 1.0;
  if (dp == 0.0) {
    ierr = VecNorm(R,NORM_2,&tau);CHKERRQ(ierr);
  }
  else {
    tau = dp;
  }

  ierr = VecDot(RP,RP,&rho1);CHKERRQ(ierr);
  ierr = VecCopy(R,P);CHKERRQ(ierr);

  i=0;
  do {
    ierr = KSP_PCApply(ksp,P,PH);CHKERRQ(ierr); /*  ph <- K p */
    ierr = KSP_MatMult(ksp,mat,PH,V);CHKERRQ(ierr); /* v <- A ph */

    ierr = VecDot(V,RP,&rho2);CHKERRQ(ierr); /* rho2 <- (v,rp) */
    if (rho2 == 0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Divide by zero");

    if (rho1 == 0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN; /* Stagnation */
      break;
    }

    alpha = rho1 / rho2; /* alpha <- rho1/rho2; */
    ierr  = VecWAXPY(S,-alpha,V,R);CHKERRQ(ierr); /* s <- r - alpha v */

    /* First Quasi-Minimization Step */
    ierr   =  VecNorm(S,NORM_2,&F);CHKERRQ(ierr); /* f <- norm(s) */
    theta2 =  F / tau;   /* theta2 <- norm(s)/tau ;*/

    c = 1.0 / PetscSqrtReal(1.0 + theta2 * theta2);

    tau2 = tau * theta2 * c;
    eta2 = c * c * alpha;
    cf  = theta * theta * eta / alpha;
    ierr = VecWAXPY(D2,cf,D,PH);CHKERRQ(ierr);        /* d2 <- ph + cf d */
    ierr = VecWAXPY(X2,eta2,D2,X);CHKERRQ(ierr);      /* x2 <- x + eta2 d2 */

    /* Apply the right preconditioner again */
    ierr = KSP_PCApply(ksp,S,SH);CHKERRQ(ierr); /*  sh <- K s */
    ierr = KSP_MatMult(ksp,mat,SH,T);CHKERRQ(ierr); /* t <- A sh */

    ierr = VecDotNorm2(S,T,&uu,&vv);CHKERRQ(ierr);
    if (vv == 0.0) {
      ierr = VecDot(S,S,&uu);CHKERRQ(ierr);
      if (uu != 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      ierr = VecAXPY(X,alpha,SH);CHKERRQ(ierr);   /* x <- x + alpha sh */
      ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i+1,0.0);CHKERRQ(ierr);
      break;
    }
    ierr   =  VecNorm(V,NORM_2,&NV);CHKERRQ(ierr); /* nv <- norm(v) */

    if (NV == 0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Matrix is singular, breakdown because of the singularity");

    omega = uu / vv; /* omega <- uu/vv; */
    if (omega == 0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      break;
    }

    /* Computing the residual */
    ierr = VecWAXPY(R,-omega,T,S);CHKERRQ(ierr);  /* r <- s - omega t */

    /* Second Quasi-Minimization Step */
    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    }

    if (tau2 == 0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Divide by zero");
    theta = dp / tau2;
    c = 1.0 / PetscSqrtReal(1.0 + theta * theta);
    if (dp == 0.0) {
      ierr = VecNorm(R,NORM_2,&tau);CHKERRQ(ierr);
    }
    else {
      tau = dp;
    }
    tau = tau * c;
    eta = c * c * omega;

    cf1  = theta2 * theta2 * eta2 / omega;
    ierr = VecWAXPY(D,cf1,D2,SH);CHKERRQ(ierr);     /* d <- sh + cf1 d2 */
    ierr = VecWAXPY(X,eta,D,X2);CHKERRQ(ierr);      /* x <- x2 + eta d */

    ierr =  VecDot(R,RP,&rho2);
    if (rho2 == 0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      break;
    }
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);

    beta = (alpha*rho2)/ (omega*rho1);
    ierr = VecAXPBYPCZ(P,1.0,-omega*beta,beta,R,V);CHKERRQ(ierr); /* p <- r - omega * beta* v + beta * p */
    rho1 = rho2;
    ierr = KSP_MatMult(ksp,mat,X,AX);CHKERRQ(ierr); /* Ax <- A x */
    ierr = VecWAXPY(Z,-1.0,AX,B);CHKERRQ(ierr);  /* r <- s - omega t */
    ierr = VecNorm(Z,NORM_2,&final);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    i++;
  } while (i<ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPQMRCGS - Implements QMRCGStab method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:
    Only allow right preconditioning

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBICGS, KSPFBCGSL, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_QMRCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&bcgs);CHKERRQ(ierr);

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_QMRCGS;
  ksp->ops->solve          = KSPSolve_QMRCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;
  ksp->pc_side             = PC_RIGHT;  /* set default PC side */

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
