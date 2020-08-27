#include <petsc/private/kspimpl.h>
/*   Include the auxillary functions needed by pipecg2.c    */
#include "auxillaryfunctions.c"

/*
     KSPSetUp_PIPECG2 - Sets up the workspace needed by the PIPECG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
  PetscErrorCode KSPSetUp_PIPECG2(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get work vectors needed by PIPECG2 */
  ierr = KSPSetWorkVecs(ksp,20);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 KSPSolve_PIPECG2 - This routine actually applies the PIPECG2 method
*/
static PetscErrorCode  KSPSolve_PIPECG2(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  PetscScalar    alpha[2],beta[2],gamma[2],delta[2],lambda[10];
  PetscReal      dp    = 0.0;
  Vec            X,B,Z,P,W,Q,U,M,N,R,S,C,D,E,F,G[2],H[2],A1,B1;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;
  MPI_Comm       pcomm;
  MPI_Request    req;
  MPI_Status     stat;

  PetscFunctionBegin;
  pcomm = PetscObjectComm((PetscObject)ksp);
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X = ksp->vec_sol;
  B = ksp->vec_rhs;
  M = ksp->work[0];
  Z = ksp->work[1];
  P = ksp->work[2];
  N = ksp->work[3];
  W = ksp->work[4];
  Q = ksp->work[5];
  U = ksp->work[6];
  R = ksp->work[7];
  S = ksp->work[8];
  C = ksp->work[9];
  D = ksp->work[10];
  E = ksp->work[11];
  F = ksp->work[12];
  G[0]  = ksp->work[13];
  H[0] = ksp->work[14];
  G[1]  = ksp->work[15];
  H[1] = ksp->work[16];
  A1 = ksp->work[17];
  B1 = ksp->work[18];

  ierr = PetscArrayzero(alpha,2*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscArrayzero(beta,2*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscArrayzero(gamma,2*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscArrayzero(delta,2*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscArrayzero(lambda,2*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = VecGetLocalSize(B,&n);CHKERRQ(ierr);
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);		/*  r <- b - Ax  */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);			
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);				/*  r <- b (x is 0) */
  }

  ierr = KSP_PCApply(ksp,R,U);CHKERRQ(ierr);     		/*  u <- Br  */
  ierr = KSP_MatMult(ksp,Amat,U,W);CHKERRQ(ierr);               /*  w <- Au  */ 

  ierr =VecMergedDot(U,W,R,ksp->normtype,&gamma[0],&delta[0],&dp);CHKERRQ(ierr);	/*  gamma  <- r'*u , delta <- w'*u , dp <- u'*u or r'*r or r'*u                                                                                            depending on ksp_norm_type  */    
  lambda[7]= gamma[0];
  lambda[8]= delta[0];
  lambda[9] = dp;

  ierr = MPI_Iallreduce(MPI_IN_PLACE,&lambda[7],3,MPI_DOUBLE,MPI_SUM,pcomm,&req);CHKERRQ(ierr);
  
  ierr = KSP_PCApply(ksp,W,M);CHKERRQ(ierr);			/*  m <- Bw  */
  ierr = KSP_MatMult(ksp,Amat,M,N);CHKERRQ(ierr);		/*  n <- Am  */
  
  ierr = KSP_PCApply(ksp,N,G[0]);CHKERRQ(ierr);			/*  g <- Bn  */
  ierr = KSP_MatMult(ksp,Amat,G[0],H[0]);CHKERRQ(ierr);		/*  h <- Ag  */
    
  ierr = KSP_PCApply(ksp,H[0],E);CHKERRQ(ierr);        		/*  e <- Bh  */ 
  ierr = KSP_MatMult(ksp,Amat,E,F);CHKERRQ(ierr);		/*  f <- Ae  */

  ierr = MPI_Wait(&req,&stat);CHKERRQ(ierr);

  gamma[0] =lambda[7];
  delta[0] = lambda[8];
  dp    = PetscSqrtReal(PetscAbsScalar(lambda[9]));
 
  VecMergedDot2(N,M,W,&lambda[1],&lambda[4]);			/*  lambda_1 <- w'*m , lambda_4 <- n'*m  */
  ierr = MPI_Allreduce(MPI_IN_PLACE,&lambda[1],1,MPI_DOUBLE,MPI_SUM,pcomm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&lambda[4],1,MPI_DOUBLE,MPI_SUM,pcomm);CHKERRQ(ierr);


  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;


  ierr       = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  double alphaold=0.0;
  for(i =0; i<ksp->max_it; i+=2  ) {    
    if (i == 0) {
      beta[0] = 0;
      alpha[0] = gamma[0] / delta[0];

      gamma[1] = gamma[0] -2*alpha[0] *( delta[0]) + alpha[0] * alpha[0] *(lambda[1] );
      delta[1] = delta[0] - alpha[0] * (lambda[1]) - alpha[0] *(lambda[1]) + alpha[0]*alpha[0]*(lambda[4] );
      
      beta[1]  = gamma[1] / gamma[0];
      alpha[1] = gamma[1] / (delta[1] - beta[1] / alpha[0] * gamma[1]);
      
      ierr= VecMergedOpsShort(X,R,Z,W,P,Q,C,D,G[0],H[0],G[1],H[1],S,A1,B1,E,F,M,N,U,ksp->normtype,beta[0],alpha[0], beta[1],alpha[1],lambda);
      CHKERRQ(ierr);  
    } else {
        beta[0]  = gamma[1] / gamma[0];
        alpha[0] = gamma[1] / (delta[1] - beta[0] / alpha[1] * gamma[1]);
 
        gamma[0] = gamma[1];
        delta[0] = delta[1];

        gamma[1] = gamma[0] -2*alpha[0] *( delta[0] + beta[0] * lambda[0]) + //
		   alpha[0] * alpha[0] * (lambda[1] + beta[0] * lambda[2] + beta[0] * lambda[2] + beta[0] * beta[0] * lambda[3]);

        delta[1] = delta[0] - alpha[0] * (lambda[1] + beta[0]* lambda[2]) - //
	           alpha[0] *(lambda[1] + beta[0] * lambda[2]) + //
		   alpha[0]*alpha[0] * (lambda[4] + beta[0] * lambda[5] + beta[0] *lambda[5] + beta[0] * beta[0] * lambda[6]);

        beta[1]  = gamma[1] / gamma[0];
        alpha[1] = gamma[1] / (delta[1] - beta[1] / alpha[0] * gamma[1]);

        ierr=  VecMergedOps(X,R,Z,W,P,Q,C,D,G[0],H[0],G[1],H[1],S,A1,B1,E,F,M,N,U,ksp->normtype,beta[0],alpha[0],beta[1],alpha[1],lambda,alphaold);
	CHKERRQ(ierr);
    }

    gamma[0] = gamma[1];
    delta[0] = delta[1];
 
    ierr = MPI_Iallreduce(MPI_IN_PLACE,lambda,10,MPI_DOUBLE,MPI_SUM,pcomm,&req);CHKERRQ(ierr);  /* Calculating the lambdas, gamma, delta and dp */

    ierr = KSP_PCApply(ksp,N,G[0]);CHKERRQ(ierr);			/*  g <- Bn  */
    ierr = KSP_MatMult(ksp,Amat,G[0],H[0]);CHKERRQ(ierr);		/*  h <- Ag  */      

    ierr = KSP_PCApply(ksp,H[0],E);CHKERRQ(ierr);      		/*  e <- Bh  */
    ierr = KSP_MatMult(ksp,Amat,E,F);CHKERRQ(ierr); 		/*  f <- Ae */

    ierr = MPI_Wait(&req,&stat);CHKERRQ(ierr);

    gamma[1] =lambda[7];
    delta[1] = lambda[8];
    dp    = PetscSqrtReal(PetscAbsScalar(lambda[9]));

    alphaold = alpha[1];
    ksp->its = i;

    if (i > 0) {
      if (ksp->normtype == KSP_NORM_NATURAL) dp = PetscSqrtReal(PetscAbsScalar(gamma[0]));
      else if (ksp->normtype == KSP_NORM_NONE) dp = 0.0;

      ksp->rnorm = dp;
      ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,i,dp);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }
  }

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
   KSPPIPECG2 - Pipelined conjugate gradient method with a single non-blocking allreduce per two iterations.

   This method has only a single non-blocking reduction per two iterations, compared to 2 blocking for standard CG.  The
   non-blocking reduction is overlapped by two matrix-vector products and two preconditioner applications.

   Level: intermediate

   Notes:
   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See the FAQ on the PETSc website for details.

   Contributed by:
   Manasi Tiwari, Computational and Data Sciences, Indian Institute of Science, Bangalore

   Reference:
   Manasi Tiwari and Sathish Vadhiyar, "Pipelined Conjugate Gradient Methods for Distributed Memory Systems",
   Submitted to International Conference on High Performance Computing, Data and Analytics 2020.

.seealso: KSPCreate(), KSPSetType(). 
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECG2(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_PIPECG2;
  ksp->ops->solve          = KSPSolve_PIPECG2;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = 0;
  ksp->ops->setfromoptions = 0;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
