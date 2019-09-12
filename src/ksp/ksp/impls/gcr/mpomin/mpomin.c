/*
  This file defines the multipreconditioned orthomin solver.

  It is adapted to SPD matrices (or even more general thanks to full orthogonalization), 
  and any preconditioner which provides the PCApplyMP method
*/
#include <petsc/private/kspimpl.h>
#include <../src/mat/impls/dense/seq/dense.h>
#include <petscblaslapack.h>
#include <petscmat.h>

typedef struct {
  PetscReal eigtrunc;       /*Threshold for eigenvalue-based pseudo inversion*/
} KSP_MPOMIN;

/*
     KSPSetUp_MPOMIN - Sets up the workspace needed by the MPOMIN method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_MPOMIN(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get work vectors needed by MPOMIN */
  ierr = KSPSetWorkVecs(ksp, 1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 KSPSolve_MPOMIN - This routine actually applies the multipreconditioned orthomin method

 Input Parameter:
 .     ksp - the Krylov space object that was set to use multipreconditioned orthomin, by, for
             example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPMPOMIN);
*/
static PetscErrorCode KSPSolve_MPOMIN(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscMPIInt    comm_size;
  PetscScalar    beta;
  Vec            Beta,Beta2;
  Mat            Gamma, Delta, Delta_R, Delta_RD;
  PetscReal      dp = 0.0;
  Vec            X, B, R;
  Mat            Z, W, P, Ptemp, Wtemp, Ztest;

  Mat            *Pstor, *Wstor;

  Mat            Amat, Pmat;
  PetscBool      diagonalscale;

  PetscInt       n_sd,n_coarse;
  PetscInt       VecLocSize, VecGloSize;
  PetscInt       nkk,ngood,ii,jj;

  MatScalar      *vals;
  Mat_SeqDense   *mat;
  Mat            Vmat;
  PetscScalar    *myrows;

  IS             rows,cols;
  const PetscInt *irows,*icols;

  Vec            Vsubdiag;
  PetscReal      *thediag;
  PetscBLASInt   info, n;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
  Mat            WW;
#endif
  PetscScalar    *theZdiag;


  PetscFunctionBegin;

  ierr = PCGetDiagonalScale(ksp->pc, &diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  X = ksp->vec_sol;
  B = ksp->vec_rhs;
  R = ksp->work[0];

  ierr = PetscMalloc2(ksp->max_it, &Pstor,ksp->max_it, &Wstor);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)X)->comm, &comm_size);CHKERRQ(ierr);

  /* Ugly reverse communication to get info from the multipreconditioner */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, 1, 3, NULL, &Ztest);CHKERRQ(ierr);
  ierr = KSP_PCApplyMP(ksp, R, Ztest);CHKERRQ(ierr); 
  ierr = MatDenseGetArray(Ztest, &vals);CHKERRQ(ierr);
  n_sd = (PetscInt)vals[0];
  n_coarse = (PetscInt)vals[2];
  ierr = MatDenseRestoreArray(Ztest, &vals);CHKERRQ(ierr);
  ierr = MatDestroy(&Ztest);CHKERRQ(ierr);
  /* End of ugly reverse communication to get info from the multipreconditioner */

  ierr = PetscMalloc1(n_sd + n_coarse, &thediag);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(n_sd + n_coarse, &theZdiag);
#else
  theZdiag = thediag;
#endif
  ierr = VecGetLocalSize(R, &VecLocSize);CHKERRQ(ierr);
  ierr = VecGetSize(R, &VecGloSize);CHKERRQ(ierr);
  n_sd += n_coarse; 
  ierr = MatCreateAIJ(((PetscObject)X)->comm, VecLocSize, PETSC_DECIDE, VecGloSize, n_sd, n_sd, NULL, n_sd, NULL, &Z);CHKERRQ(ierr);
  ierr = VecCreateMPI(((PetscObject)X)->comm, PETSC_DECIDE, n_sd, &Beta);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, n_sd, n_sd, NULL, &Delta_RD);CHKERRQ(ierr);
  ierr = PCGetOperators(ksp->pc, &Amat, &Pmat);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(3*n_sd-2, &rwork);CHKERRQ(ierr);
#endif


  ksp->its = 0;
  if (!ksp->guess_zero){
    ierr = KSP_MatMult(ksp, Amat, X, R);CHKERRQ(ierr); /*    r <- b - Ax                       */
    ierr = VecAYPX(R, -1.0, B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B, R);CHKERRQ(ierr); /*    r <- b (x is 0)                   */
  }

  switch (ksp->normtype) {
    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r                        */
      break;
    case KSP_NORM_NONE:
      dp = 0.0;
    default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }

  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);     /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  ierr = KSP_PCApplyMP(ksp, R, Z);CHKERRQ(ierr); /*    Z <- [B1 r, B2 r ... ]          */
  ierr = MatDuplicate(Z, MAT_COPY_VALUES, &P);CHKERRQ(ierr); /*     P <- Z                           */
  ierr = KSP_MatMatMult(ksp, Amat, P, &W, MAT_INITIAL_MATRIX);CHKERRQ(ierr); /* W <- A*P */

  i = 0;
  do {
    ksp->its = i+1;  
    ierr = MatMultHermitianTranspose(W, R, Beta);CHKERRQ(ierr); /* Beta = W'*r */
    ierr = VecSum(Beta, &beta);CHKERRQ(ierr);

    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
    }
#if defined(PETSC_USE_COMPLEX)
    ierr = MatDuplicate(W,MAT_COPY_VALUES,&WW);CHKERRQ(ierr);
    ierr = MatConjugate(WW);
    ierr = MatTransposeMatMult(WW, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Delta);CHKERRQ(ierr); /*     Delta <- W' * W                       */
    ierr = MatDestroy(&WW);CHKERRQ(ierr);
#else
    ierr = MatTransposeMatMult(W, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Delta);CHKERRQ(ierr); /*     Delta <- W' * W                       */
#endif
    /*  This version makes use of dense algebra should be improved for larger blocks */
    ierr = MatCreateRedundantMatrix(Delta, comm_size, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &Delta_R);CHKERRQ(ierr);
    ierr = MatDestroy(&Delta);CHKERRQ(ierr);
    ierr = MatConvert(Delta_R, MATSEQDENSE, MAT_REUSE_MATRIX, &Delta_RD);CHKERRQ(ierr);
    ierr = MatDestroy(&Delta_R);CHKERRQ(ierr);
    mat = (Mat_SeqDense *)Delta_RD->data;

    ierr = PetscBLASIntCast(Delta_RD->cmap->n, &n);CHKERRQ(ierr);

    if (!mat->fwork) {
      PetscScalar dummy;
      mat->lfwork = -1;
#if !defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &n, mat->v, &mat->lda, thediag, &dummy, &mat->lfwork, &info));CHKERRQ(ierr);
#else
      PetscStackCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &n, mat->v, &mat->lda, thediag, &dummy, &mat->lfwork, rwork, &info));CHKERRQ(ierr);
#endif
      mat->lfwork = (PetscInt)PetscRealPart(dummy);
      ierr = PetscMalloc1(mat->lfwork, &mat->fwork);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)Delta_RD, mat->lfwork * sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &n, mat->v, &mat->lda, thediag, mat->fwork, &mat->lfwork, &info));CHKERRQ(ierr);
#else
      PetscStackCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &n, mat->v, &mat->lda, thediag, mat->fwork, &mat->lfwork, rwork, &info));CHKERRQ(ierr);
#endif

    if (info < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad parameter %D", (PetscInt)info - 1);

    /* Now we analyze the eigenvalues, they are in ascending order, they also should be positive*/

    nkk = 0;
    for (jj = 0; jj < n; ++jj) {
      if (thediag[jj] < thediag[n - 1] * ((KSP_MPOMIN *)ksp->data)->eigtrunc) {
        nkk = jj + 1;
        thediag[jj] = 0.;
      } else {
        thediag[jj] = 1. / sqrt(thediag[jj]);
#if defined(PETSC_USE_COMPLEX)
        theZdiag[jj] = thediag[jj]; /*real to complex conversion*/
#endif
      }
    }
    ngood = n - nkk;
    /* Values from nkk to n should be kept */

    ierr = ISCreateStride(PetscObjectComm((PetscObject)ksp), n_sd, 0, 1, &rows);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)ksp), ngood, 0, 1, &cols);CHKERRQ(ierr);
    ierr = ISGetIndices(rows,&irows);CHKERRQ(ierr);
    ierr = ISGetIndices(cols,&icols);CHKERRQ(ierr);
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)ksp), PETSC_DECIDE, ngood,&Vsubdiag);CHKERRQ(ierr);
    ierr = VecSetValues(Vsubdiag, ngood, icols, &theZdiag[nkk], INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(Vsubdiag);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Vsubdiag);CHKERRQ(ierr);
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)ksp), PETSC_DECIDE, ngood, &Beta2);CHKERRQ(ierr);
 
    PetscMalloc1(ngood*n_sd,&myrows);CHKERRQ(ierr);
    for (ii=0;ii<n;++ii){
      for (jj=0;jj<ngood;++jj){
        myrows[jj+ii*ngood]=mat->v[(nkk+jj)*n+ii];
      }
    }
    ierr = MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, ngood, NULL, &Vmat);CHKERRQ(ierr);
    ierr = MatSetValues(Vmat,n_sd,irows,ngood,icols,myrows,INSERT_VALUES);CHKERRQ(ierr);
    PetscFree(myrows);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Vmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Vmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = ISRestoreIndices(rows,&irows);CHKERRQ(ierr);
    ierr = ISRestoreIndices(cols,&icols);CHKERRQ(ierr);
    ierr = ISDestroy(&rows);CHKERRQ(ierr);
    ierr = ISDestroy(&cols);CHKERRQ(ierr);

    ierr = MatConvert(Vmat, MATAIJ, MAT_INPLACE_MATRIX, &Vmat);CHKERRQ(ierr);
    ierr = MatMatMult(P, Vmat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Pstor[i]);CHKERRQ(ierr);
    ierr = MatMatMult(W, Vmat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Wstor[i]);CHKERRQ(ierr);
    ierr = MatDiagonalScale(Pstor[i], NULL, Vsubdiag);CHKERRQ(ierr);
    ierr = MatDiagonalScale(Wstor[i], NULL, Vsubdiag);CHKERRQ(ierr);
    ierr = MatMultHermitianTranspose(Vmat, Beta, Beta2);CHKERRQ(ierr);
    ierr = VecPointwiseMult(Beta2, Vsubdiag, Beta2);CHKERRQ(ierr);
    ierr = VecDestroy(&Vsubdiag);CHKERRQ(ierr);
    ierr = MatDestroy(&Vmat);CHKERRQ(ierr);
    ierr = MatMultAdd(Pstor[i], Beta2, X, X);CHKERRQ(ierr);
    ierr = VecScale(Beta2, -1.);CHKERRQ(ierr);
    ierr = MatMultAdd(Wstor[i], Beta2, R, R);CHKERRQ(ierr);
 
    ierr = MatConvert(Pstor[i], MATAIJ, MAT_INPLACE_MATRIX, &Pstor[i]);CHKERRQ(ierr);
    ierr = MatConvert(Wstor[i], MATAIJ, MAT_INPLACE_MATRIX, &Wstor[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&W);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = VecDestroy(&Beta2);CHKERRQ(ierr);
    
    ierr = KSP_PCApplyMP(ksp, R, Z);CHKERRQ(ierr); /*    Z <- [B1 r, B2 r ... ]          */
    ierr = MatDuplicate(Z, MAT_COPY_VALUES, &P);CHKERRQ(ierr); /* P <- Z*/
    ierr = KSP_MatMatMult(ksp, Amat, P, &W, MAT_INITIAL_MATRIX);CHKERRQ(ierr); /* W <- A*P */

    for (jj = 0; jj < i+1; jj++) { /* Full block orthogonalization */
#if defined(PETSC_USE_COMPLEX)
      ierr = MatConjugate(Wstor[jj]);CHKERRQ(ierr); /* Ugly hack, better MatHermitianTransposeMatMult should be coded */
#endif
      ierr = MatTransposeMatMult(Wstor[jj], W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gamma);CHKERRQ(ierr);
 #if defined(PETSC_USE_COMPLEX)
      ierr = MatConjugate(Wstor[jj]);CHKERRQ(ierr); /* Ugly hack, better MatHermitianTransposeMatMult should be coded */
#endif
      ierr = MatMatMult(Pstor[jj], Gamma, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Ptemp);CHKERRQ(ierr);
      ierr = MatMatMult(Wstor[jj], Gamma, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Wtemp);CHKERRQ(ierr);
      ierr = MatDestroy(&Gamma);CHKERRQ(ierr);
      ierr = MatAXPY(P, -1., Ptemp, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(W, -1., Wtemp, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&Ptemp);CHKERRQ(ierr);
      ierr = MatDestroy(&Wtemp);CHKERRQ(ierr);
    }

    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r                        */
    }
    
    ksp->rnorm = dp;
    ierr = KSPLogResidualHistory(ksp, dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp, i+1, dp);CHKERRQ(ierr);

    ierr = (*ksp->converged)(ksp, i+1, dp, &ksp->reason, ksp->cnvP);CHKERRQ(ierr);

    if (ksp->reason) { 
      break;
    }
    i++;
  } while (i < ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  ierr = PetscFree(thediag);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(theZdiag);CHKERRQ(ierr);
#endif
  ierr = VecDestroy(&Beta);CHKERRQ(ierr);
  ierr = MatDestroy(&Delta_RD);CHKERRQ(ierr);
  ierr = MatDestroy(&Delta_R);CHKERRQ(ierr);
  ierr = MatDestroy(&Delta);CHKERRQ(ierr);
  ierr = MatDestroy(&Gamma);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = MatDestroy(&W);CHKERRQ(ierr);
  ierr = MatDestroy(&Z);CHKERRQ(ierr);
  
  i = PetscMin(i+1,ksp->max_it);
  for (jj=0;jj<i;jj++){
    ierr = MatDestroy(&Pstor[jj]);CHKERRQ(ierr);
    ierr = MatDestroy(&Wstor[jj]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(Pstor,Wstor);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_MPOMIN - Checks the options database for options related to the
                           multipreconditioned conjugate gradient method.
*/
PetscErrorCode KSPSetFromOptions_MPOMIN(PetscOptionItems *PetscOptionsObject, KSP ksp)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  KSP_MPOMIN     *cg = (KSP_MPOMIN *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "KSP MPOMIN options");CHKERRQ(ierr);
  cg->eigtrunc = 1.e-12;
  ierr = PetscOptionsReal("-ksp_mp_eigtrunc", "Truncation for pseudoinversion based on eigenvalue", "KSPMPC", cg->eigtrunc, &cg->eigtrunc, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   KSPMPOMIN - Multipreconditioned orthomin method.

   At each iteration this method searches the approximation in a large subspace which is defined by the preconditioner.
   The preconditioner must return a Matrix instead of a Vector.

   See also KSPMPCG

   Level: intermediate

   Notes: This is a simple implementation of multipreconditioned orthomin. 
   It is adapted to any preconditioner providing the PCApplyMP method.
   The computation of the pseudo inverse is based on the computation of eigenvalues, as proposed for mpcg in [Molina and Roux. doi:10.1002/nme.6024].
   This implementation is based on dense algebra for the pseudo-inversion, it is general but can not scale well with the number of columns.
   Depending on the PC context, optimization could be considered by the exploitation of the sparsity [Molina and Roux. doi:10.1002/nme.6024]

   Contributed by: P. Gosselet and N. Tardieu

   Example usage:
    [*] KSP ex71, restricted ASM:
	    ./ex71 -pde_type Elasticity -cells 10,10,10 -dim 3 -dirichlet -pc_type asm -mat_type seqaij  
      -ksp_type mpomin -ksp_norm_type unpreconditioned -pc_asm_type restrict -pc_asm_overlap 1 -pc_asm_blocks 8 
      -sub_ksp_type preonly -sub_pc_type lu -sub_pc_factor_mat_solver_type mumps  -ksp_monitor -ksp_view


   Reference: [Bovet et al. doi:j.crma.2017.01.010]
 
.seealso: KSPCreate(), KSPSetType(), 
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_MPOMIN(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_MPOMIN     *mpomin;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1);CHKERRQ(ierr);
  ierr = PetscNewLog(ksp, &mpomin);CHKERRQ(ierr);
  ksp->data = (void *)mpomin;

  ksp->ops->setup = KSPSetUp_MPOMIN;
  ksp->ops->solve = KSPSolve_MPOMIN;
  ksp->ops->destroy = KSPDestroyDefault;
  ksp->ops->view = 0;
  ksp->ops->setfromoptions = KSPSetFromOptions_MPOMIN;
  ksp->ops->buildsolution = KSPBuildSolutionDefault;
  ksp->ops->buildresidual = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
