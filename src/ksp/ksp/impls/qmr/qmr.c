#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_QMR(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,10);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_QMR(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    delta,eta,epsilon,beta;
  PetscReal      dp,ksi,rorig,sabs,rho,gamma,theta,gammaold,thetaold,rhoold;
  Vec            X,B,V,P,R,Q,D,Y,Z,W,P1,S;
  Mat            Amat, Pmat;

  PetscFunctionBegin;
  X    = ksp->vec_sol;
  B    = ksp->vec_rhs;
  V    = ksp->work[0];
  P    = ksp->work[1];
  R    = ksp->work[2];
  Q    = ksp->work[3];
  D    = ksp->work[4];
  Y    = ksp->work[5];
  Z    = ksp->work[6];
  W    = ksp->work[7];
  P1   = ksp->work[8];
  S    = ksp->work[9];

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  /* Compute initial unpreconditioned residual */
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);
  }

  /* Test for nothing to do */
  ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  KSPCheckNorm(ksp,dp);
  ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = dp;
  else ksp->rnorm = 0.0;
  ksp->its = 0;
  ierr     = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr     = KSPMonitor(ksp,0,ksp->rnorm);CHKERRQ(ierr);
  ierr     = (*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Set the initial conditions */

  ierr = VecCopy(R,V);CHKERRQ(ierr);
  ierr = VecCopy(V,Y);CHKERRQ(ierr);
  ierr = VecCopy(R,W);CHKERRQ(ierr);              /* W chosen as R */
  if (ksp->pc_side == PC_RIGHT) {
    ierr = VecCopy(V,Y);CHKERRQ(ierr);
  } else if (ksp->pc_side == PC_LEFT) {
    ierr = KSP_PCApply(ksp,V,Y);CHKERRQ(ierr);
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    ierr = PCApplySymmetricLeft(ksp->pc,V,Y);CHKERRQ(ierr);
  }

  if (ksp->pc_side == PC_RIGHT) {
    ierr = VecConjugate(W);CHKERRQ(ierr);
    /* To replace with KSP_PC_ApplyHermitianTranspose, this is a workaround to obtain the solution */
    ierr = KSP_PCApplyTranspose(ksp,W,Z);CHKERRQ(ierr);     /* Solve M_2^H Z = W */
    ierr = VecConjugate(Z);CHKERRQ(ierr);
    ierr = VecConjugate(W);CHKERRQ(ierr);
  } else if (ksp->pc_side == PC_LEFT) {
    ierr = VecCopy(W,Z);CHKERRQ(ierr);
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    ierr = VecConjugate(W);CHKERRQ(ierr);
    ierr = PCApplySymmetricLeft(ksp->pc,W,Z);CHKERRQ(ierr);
    ierr = VecConjugate(Z);CHKERRQ(ierr);
    ierr = VecConjugate(W);CHKERRQ(ierr);
  }

  ierr = VecNorm(Z,NORM_2,&ksi);CHKERRQ(ierr);
  KSPCheckNorm(ksp,ksi);

  ierr = VecNorm(Y,NORM_2,&rhoold);CHKERRQ(ierr);
  KSPCheckNorm(ksp,rhoold);

  gammaold   = 1.0;
  eta        = -1.0;
  thetaold   = 0.0;
  i          = 0;
  rorig      = dp;
  sabs       = 1.0;
  epsilon    = 0.0;

  do {
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

    if ((rhoold == 0) || (ksi == 0)) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;  /* Method fails */
      break;
    }

    ierr = VecScale(V,1/rhoold);CHKERRQ(ierr);
    ierr = VecScale(Y,1/rhoold);CHKERRQ(ierr);
    ierr = VecScale(W,1/ksi);CHKERRQ(ierr);
    ierr = VecScale(Z,1/ksi);CHKERRQ(ierr);

    ierr = VecDot(Y,Z,&delta);CHKERRQ(ierr);    /* delta <- Z'*Y */

    if (ksp->pc_side == PC_RIGHT) {
      ierr = KSP_PCApply(ksp,Y,P1);CHKERRQ(ierr);     /* Solve M_2 Y1 = Y */
      ierr = VecCopy(P1,Y);CHKERRQ(ierr);
      ierr = VecCopy(Z,P1);CHKERRQ(ierr);
    } else if (ksp->pc_side == PC_LEFT) {
      ierr = VecConjugate(Z);
      /* To replace with KSP_PC_ApplyHermitianTranspose, this is a workaround to obtain the solution */
      ierr = KSP_PCApplyTranspose(ksp,Z,P1);CHKERRQ(ierr);     /* Solve M_1^H P1 = Z*/
      ierr = VecConjugate(P1);
    }
    else if (ksp->pc_side == PC_SYMMETRIC) {
      ierr = PCApplySymmetricRight(ksp->pc,Y,P1);CHKERRQ(ierr);
      ierr = VecCopy(P1,Y);CHKERRQ(ierr);
      ierr = VecConjugate(Z);
      ierr = PCApplySymmetricRight(ksp->pc,Z,P1);CHKERRQ(ierr);
      ierr = VecConjugate(P1);
    }


    if (i == 0) {
      ierr = VecCopy(Y,P);CHKERRQ(ierr);
      ierr = VecCopy(P1,Q);CHKERRQ(ierr);
    } else {
      ierr = VecAYPX(P,-(ksi*delta)/epsilon,Y);CHKERRQ(ierr);
      ierr = VecAYPX(Q,-rho*PetscConj(delta/epsilon),P1);CHKERRQ(ierr);
    }

    ierr = KSP_MatMult(ksp,Amat,P,P1);CHKERRQ(ierr);

    ierr = VecDot(P1, Q, &epsilon);CHKERRQ(ierr);
    if (epsilon == 0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;  /* Method fails */
      break;
    }
    beta = epsilon/delta;
    if (beta == 0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;  /* Method fails */
      break;
    }

    ierr = VecAYPX(V,-beta,P1);CHKERRQ(ierr);
    if (ksp->pc_side == PC_RIGHT) {
      ierr = VecCopy(V,Y);CHKERRQ(ierr);
    } else if (ksp->pc_side == PC_LEFT) {
      ierr = KSP_PCApply(ksp,V,Y);CHKERRQ(ierr);
    }
    else if (ksp->pc_side == PC_SYMMETRIC) {
      ierr = PCApplySymmetricLeft(ksp->pc,V,Y);CHKERRQ(ierr);
    }

    ierr = VecNorm(Y,NORM_2,&rho);CHKERRQ(ierr);
    KSPCheckNorm(ksp,rho);
    ierr = VecScale(W,-PetscConj(beta));CHKERRQ(ierr);
    ierr = MatMultHermitianTransposeAdd(Amat,Q,W,W);CHKERRQ(ierr);

    if (ksp->pc_side == PC_RIGHT) {
      ierr = VecConjugate(W);CHKERRQ(ierr);
      /* To replace with KSP_PC_ApplyHermitianTranspose, this is a workaround to obtain the solution */
      ierr = KSP_PCApplyTranspose(ksp,W,Z);CHKERRQ(ierr);
      ierr = VecConjugate(Z);CHKERRQ(ierr);
      ierr = VecConjugate(W);CHKERRQ(ierr);
    } else if (ksp->pc_side == PC_LEFT) {
      ierr = VecCopy(W,Z);CHKERRQ(ierr);
    }
    else if (ksp->pc_side == PC_SYMMETRIC) {
      ierr = VecConjugate(W);CHKERRQ(ierr);
      ierr = PCApplySymmetricLeft(ksp->pc,W,Z);CHKERRQ(ierr);
      ierr = VecConjugate(Z);CHKERRQ(ierr);
      ierr = VecConjugate(W);CHKERRQ(ierr);
    }

    ierr = VecNorm(Z,NORM_2,&ksi);CHKERRQ(ierr);
    KSPCheckNorm(ksp,ksi);CHKERRQ(ierr);

    theta = rho/(gammaold*PetscAbsScalar(beta));
    gamma = 1/(PetscSqrtScalar(1.0 + theta*theta));
    if (gamma==0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;  /* Method fails */
      break;
    }

    sabs *= PetscAbsReal(gamma*theta);

    eta = -eta*rhoold*gamma*gamma/(beta*gammaold*gammaold);

    if (i == 0) {
      ierr = VecCopy(P,D);CHKERRQ(ierr);
      ierr = VecScale(D,eta);CHKERRQ(ierr);
      if (ksp->pc_side != PC_RIGHT) {
        ierr = VecCopy(P1,S);CHKERRQ(ierr);
        ierr = VecScale(S,eta);CHKERRQ(ierr);
      }
    } else {
      ierr = VecAXPBY(D,eta,thetaold*thetaold*gamma*gamma,P);CHKERRQ(ierr);
      if (ksp->pc_side != PC_RIGHT) {
        ierr = VecAXPBY(S,eta,thetaold*thetaold*gamma*gamma,P1);CHKERRQ(ierr);
      }
    }

    ierr = VecAYPX(X,1.0,D);CHKERRQ(ierr);
    if (ksp->pc_side != PC_RIGHT) {
      ierr = VecAXPY(R,-1.0,S);CHKERRQ(ierr);
    }

    /* Check convergence */
    if (ksp->pc_side == PC_RIGHT) {
      dp = rorig * PetscSqrtReal((PetscReal) i+2) * sabs;
    } else {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
      KSPCheckNorm(ksp,dp);
    }

    ierr  = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = dp;
    else ksp->rnorm = 0.0;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,ksp->rnorm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;


    thetaold = theta;
    rhoold   = rho;
    gammaold = gamma;
    i++;
  } while (i < ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode KSPBuildResidual_QMR(KSP ksp,Vec t,Vec v,Vec *V)
{
  PetscErrorCode ierr;
  Mat            Amat,Pmat;
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  ierr = KSP_MatMult(ksp,Amat,ksp->vec_sol,v);CHKERRQ(ierr);
  ierr = VecAYPX(v,-1.0,ksp->vec_rhs);CHKERRQ(ierr);
  *V   = v;
  PetscFunctionReturn(0);
}

/*MC
     KSPQMR - QMR (quasi minimal residual),

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:
    This is the QMR solver without look-ahead.
    Originally, QMR is designed for symmetric preconditioners M = M1*M2. However, KSPQMR supports
    left preconditioning (in this case M2 = I) and right preconditioning (in this case M1 = I).
    The symmetric preconditioning is only supported with Jacobi and ICC preconditioners.
    An upper bound is used for the convergence criterion with right preconditioning (cf. Freund and
    Nachtigal, proposition 4.1).


   References:
.   1. -  R. Barrett, 1994, 'Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods'
.   2. -  R. W. Freund, N. M. Nachtigal, 1991, 'QMR: a quasi-minimal residual method for non-Hermitian linear systems'

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPTCQMR, KSPTFQMR
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_QMR(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,4);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_RIGHT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_SYMMETRIC,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_SYMMETRIC,1);CHKERRQ(ierr);

  ksp->data                = (void*)0;
  ksp->ops->setup          = KSPSetUp_QMR;
  ksp->ops->solve          = KSPSolve_QMR;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_QMR;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
