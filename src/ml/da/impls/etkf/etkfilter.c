#include <petscda.h>
#include <petsc/private/daimpl.h>
#include <petscblaslapack.h>

extern PetscLogEvent PetscDA_Analysis;

typedef struct {
  PetscDA_Ensemble en;
  Vec              mean;
  Vec              y_mean;
  Vec              delta_scaled;
  Vec              w;
  Vec              r_inv_sqrt;
  Mat              Z;
  Mat              S;
  Mat              T_sqrt;
  Mat              w_ones;
} PetscDA_ETKF;

/*  T-Matrix Factorization and Application Methods [Alg 6.4 line 7] */

/*
   Tolerance for matrix square root verification in debug mode
   Use a more relaxed tolerance to account for accumulated floating-point errors
   in multiple matrix operations (Y^T * T * Y involves 3 matrix multiplications).
   A tolerance of 1e-2 (1%) is reasonable for numerical verification. */
#define MATRIX_SQRT_TOLERANCE_FACTOR 1.0e-2

/*
  PetscDAEnsembleTFactor_Cholesky - Computes Cholesky factorization of T

  Input Parameters:
+ da - the PetscDA context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  Computes the lower triangular Cholesky factor L such that T = L * L^T.
  Then zeros out the upper triangular part to ensure L is strictly lower triangular.
*/
static PetscErrorCode PetscDAEnsembleTFactor_Cholesky(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscBLASInt      n, lda, info;
  PetscScalar      *a_array;
  PetscInt          m_T, N_T, i, j;

  PetscFunctionBegin;
  /* Initialize or update L_cholesky matrix */
  if (!en->L_cholesky) {
    PetscCall(MatDuplicate(en->I_StS, MAT_COPY_VALUES, &en->L_cholesky));
  } else {
    PetscCall(MatCopy(en->I_StS, en->L_cholesky, SAME_NONZERO_PATTERN));
  }

  /* Get matrix dimensions and convert to BLAS int */
  PetscCall(MatGetSize(en->L_cholesky, &m_T, &N_T));
  PetscCheck(m_T == N_T, PetscObjectComm((PetscObject)en->L_cholesky), PETSC_ERR_ARG_WRONG, "Matrix must be square for Cholesky");
  PetscCall(PetscBLASIntCast(N_T, &n));
  lda = n;

  /* Get array from dense matrix */
  PetscCall(MatDenseGetArray(en->L_cholesky, &a_array));

  /* Compute Cholesky factorization: A = L * L^T (lower triangular) */
  LAPACKpotrf_("L", &n, a_array, &lda, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky factorization (xPOTRF): info=%" PetscInt_FMT ". Matrix T is not positive definite.", (PetscInt)info);

  /* Zero out upper triangular part (LAPACK leaves it unchanged) */
  for (j = 0; j < n; j++) {
    for (i = 0; i < j; i++) a_array[i + j * lda] = 0.0;
  }

  /* Restore array and finalize matrix */
  PetscCall(MatDenseRestoreArray(en->L_cholesky, &a_array));
  PetscCall(MatAssemblyBegin(en->L_cholesky, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(en->L_cholesky, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAEnsembleTFactor_Eigen - Computes Eigendecomposition of T

  Input Parameters:
+ da - the PetscDA context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  Computes eigenvectors V and eigenvalues D such that T = V * D * V^T.
*/
static PetscErrorCode PetscDAEnsembleTFactor_Eigen(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscBLASInt      n, lda, lwork, info;
  PetscScalar      *a_array, *work, *eig_array;
  PetscInt          m_V, N_V;
#if defined(PETSC_USE_COMPLEX)
  PetscReal *rwork = NULL;
#endif

  PetscFunctionBegin;
  /* Initialize or update V matrix */
  if (!en->V) {
    PetscCall(MatDuplicate(en->I_StS, MAT_COPY_VALUES, &en->V));
  } else {
    PetscCall(MatCopy(en->I_StS, en->V, SAME_NONZERO_PATTERN));
  }

  /* Initialize or update eigenvalue vector */
  if (!en->sqrt_eigen_vals) PetscCall(MatCreateVecs(en->I_StS, &en->sqrt_eigen_vals, NULL));

  /* Get matrix dimensions */
  PetscCall(MatGetSize(en->V, &m_V, &N_V));
  PetscCheck(m_V == N_V, PetscObjectComm((PetscObject)en->V), PETSC_ERR_ARG_WRONG, "Matrix must be square");
  PetscCall(PetscBLASIntCast(N_V, &n));
  lda = n;

  /* Get arrays */
  PetscCall(MatDenseGetArray(en->V, &a_array));
  PetscCall(VecGetArray(en->sqrt_eigen_vals, &eig_array));

  /* Query optimal workspace size */
  lwork = -1;
  PetscCall(PetscMalloc1(1, &work));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(PetscMax(1, 3 * n - 2), &rwork));
  LAPACKsyev_("V", "U", &n, a_array, &lda, (PetscReal *)eig_array, work, &lwork, rwork, &info);
#else
  LAPACKsyev_("V", "U", &n, a_array, &lda, eig_array, work, &lwork, &info);
#endif
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine xSYEV work query: info=%" PetscInt_FMT, (PetscInt)info);

  /* Allocate workspace */
  lwork = (PetscBLASInt)PetscRealPart(work[0]);
  PetscCall(PetscFree(work));
  PetscCall(PetscMalloc1(lwork, &work));

  /* Compute eigendecomposition */
#if defined(PETSC_USE_COMPLEX)
  LAPACKsyev_("V", "U", &n, a_array, &lda, (PetscReal *)eig_array, work, &lwork, rwork, &info);
  PetscCall(PetscFree(rwork));
#else
  LAPACKsyev_("V", "U", &n, a_array, &lda, eig_array, work, &lwork, &info);
#endif
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine xSYEV: info=%" PetscInt_FMT, (PetscInt)info);

  /* Cleanup */
  PetscCall(PetscFree(work));
  PetscCall(VecRestoreArray(en->sqrt_eigen_vals, &eig_array));
  PetscCall(MatDenseRestoreArray(en->V, &a_array));

  PetscCall(MatAssemblyBegin(en->V, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(en->V, MAT_FINAL_ASSEMBLY));

  /* Compute sqrt(eigenvalues) */
  PetscCall(VecSqrtAbs(en->sqrt_eigen_vals));

  /* Debug verification: Ensure V * D * V^T == T */
  if (PetscDefined(USE_DEBUG)) {
    PetscReal norm_T, norm_diff, relative_error;
    Mat       V_D, VDVt;

    /* Compute D * V^T by scaling rows */
    PetscCall(MatDuplicate(en->V, MAT_COPY_VALUES, &V_D));

    /* Restore D for verification (since sqrt_eigen_vals currently holds sqrt(D)) */
    PetscCall(VecPointwiseMult(en->sqrt_eigen_vals, en->sqrt_eigen_vals, en->sqrt_eigen_vals));

    PetscCall(MatDiagonalScale(V_D, NULL, en->sqrt_eigen_vals));

    /* Compute V * D * V^T */
    PetscCall(MatMatTransposeMult(V_D, en->V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VDVt));

    /* Compute ||V*D*V^T - T|| / ||T|| */
    PetscCall(MatAXPY(VDVt, -1.0, en->I_StS, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(en->I_StS, NORM_FROBENIUS, &norm_T));
    PetscCall(MatNorm(VDVt, NORM_FROBENIUS, &norm_diff));

    PetscCheck(norm_T > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "T = 0");
    relative_error = norm_diff / norm_T;
    PetscCheck(relative_error < MATRIX_SQRT_TOLERANCE_FACTOR, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "Eigendecomposition verification failed: ||V*D*V^T - T||/||T|| = %g", (double)relative_error);

    /* Restore sqrt(D) back to sqrt_eigen_vals */
    PetscCall(VecSqrtAbs(en->sqrt_eigen_vals));

    /* Cleanup debug matrices */
    PetscCall(MatDestroy(&V_D));
    PetscCall(MatDestroy(&VDVt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleTFactor - Compute and store factorization of T matrix

  Collective

  Input Parameters:
+ da - the `PetscDA` context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  This function computes T = I + S^T * S and stores its factorization based on
  the selected sqrt_type.

  - For CHOLESKY mode: computes the lower triangular Cholesky factor L such that T = L * L^T.
  - For EIGEN mode: computes eigenvectors V and eigenvalues D such that T = V * D * V^T.

  The implementation uses matrix reuse (MAT_REUSE_MATRIX) to minimize memory allocation
  overhead when the ensemble size remains constant across analysis cycles.

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAEnsembleApplyTInverse()`, `PetscDAEnsembleApplySqrtTInverse()`
@*/
PetscErrorCode PetscDAEnsembleTFactor(PetscDA da, Mat S)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscInt          m, s_rows, s_cols;
  MatReuse          scall      = MAT_INITIAL_MATRIX;
  PetscBool         reallocate = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(S, MAT_CLASSID, 2);

  /* 1. Validate Matrix Dimensions */
  PetscCall(MatGetSize(S, &s_rows, &s_cols));
  m = s_cols; /* Ensemble size */

  PetscCheck(m > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Innovation matrix S must have positive columns, got %" PetscInt_FMT, m);
  PetscCheck(m == en->ensemble_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "S matrix columns (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ") defined in PetscDA", m, en->ensemble_size);

  /* 2. Manage Resource Reuse */
  /* Check if we can reuse the T matrix (I_StS) and dependent factors */
  if (en->I_StS) {
    PetscInt t_rows, t_cols;
    PetscCall(MatGetSize(en->I_StS, &t_rows, &t_cols));

    /* If dimensions have changed, we must fully reallocate */
    if (t_rows != m || t_cols != m) {
      reallocate = PETSC_TRUE;
      PetscCall(PetscInfo(da, "Ensemble size changed (old: %" PetscInt_FMT ", new: %" PetscInt_FMT "), reallocating T matrix and factors\n", t_rows, m));
    } else {
      scall = MAT_REUSE_MATRIX;
    }
  }

  if (reallocate && en->I_StS) {
    PetscCall(MatDestroy(&en->I_StS));
    PetscCall(MatDestroy(&en->V));
    PetscCall(MatDestroy(&en->L_cholesky));
    PetscCall(VecDestroy(&en->sqrt_eigen_vals));
    scall = MAT_INITIAL_MATRIX;
  }

  /* 3. Compute T = I + S^T * S */
  /*
     MatTransposeMatMult computes C = A^T * B (here C = S^T * S).
     When using MAT_REUSE_MATRIX, the existing C is overwritten with the new result.
  */
  PetscCall(MatTransposeMatMult(S, S, scall, PETSC_DEFAULT, &en->I_StS));

  /* Add Identity: T = (1/rho)I + S^T*S */
  PetscCall(MatShift(en->I_StS, 1.0 / en->inflation));

  /* 4. Compute Factorization based on strategy */
  switch (en->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(PetscDAEnsembleTFactor_Cholesky(da));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(PetscDAEnsembleTFactor_Eigen(da));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)en->sqrt_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplyTInverse_Cholesky - Helper for Cholesky solver path
*/
static PetscErrorCode ApplyTInverse_Cholesky(PetscDA da, Vec sdel, Vec w)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscBLASInt      n, lda, nrhs, info;
  PetscScalar      *a_array, *b_array;
  PetscInt          m_L, N_L;

  PetscFunctionBegin;
  PetscCheck(en->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

  /* Get dimensions */
  PetscCall(MatGetSize(en->L_cholesky, &m_L, &N_L));
  PetscCall(PetscBLASIntCast(N_L, &n));
  lda  = n;
  nrhs = 1;

  /* Copy sdel to w for in-place solve */
  PetscCall(VecCopy(sdel, w));

  /* Get arrays */
  PetscCall(MatDenseGetArrayRead(en->L_cholesky, (const PetscScalar **)&a_array));
  PetscCall(VecGetArray(w, &b_array));

  /* Solve L * L^T * w = sdel using LAPACK's Cholesky solve (xPOTRS) */
  /* Note: POTRS expects the input B (w) to contain the RHS, and overwrites it with the solution */
  LAPACKpotrs_("L", &n, &nrhs, a_array, &lda, b_array, &n, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky solve (xPOTRS): info=%" PetscInt_FMT, (PetscInt)info);

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayRead(en->L_cholesky, (const PetscScalar **)&a_array));
  PetscCall(VecRestoreArray(w, &b_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplyTInverse_Eigen - Helper for Eigendecomposition solver path
*/
static PetscErrorCode ApplyTInverse_Eigen(PetscDA da, Vec sdel, Vec w)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  Vec               temp;

  PetscFunctionBegin;
  PetscCheck(en->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
  PetscCheck(en->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

  /* Allocate temporary vector for projection */
  PetscCall(VecDuplicate(sdel, &temp));

  /* 1. Project onto eigenvectors: temp = V^T * sdel */
  PetscCall(MatMultTranspose(en->V, sdel, temp));

  /* 2. Scale by inverse eigenvalues: temp = D^{-1} * temp */
  /* We store sqrt(D), so divide twice: temp = (temp / sqrt(D)) / sqrt(D) */
  PetscCall(VecPointwiseDivide(temp, temp, en->sqrt_eigen_vals));
  PetscCall(VecPointwiseDivide(temp, temp, en->sqrt_eigen_vals));

  /* 3. Map back to standard basis: w = V * temp */
  PetscCall(MatMult(en->V, temp, w));

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleApplyTInverse - Apply T^{-1} to a vector [Alg 6.4 line 8]

  Collective

  Input Parameters:
+ da   - the `PetscDA` context
- sdel - input vector S^T-delta

  Output Parameter:
. w - output vector w = T^{-1} * sdel

  Notes:
  This function applies the inverse of T = I + S^T S using the stored
  factorization. For CHOLESKY mode, it uses triangular solves. For EIGEN mode,
  it uses the eigendecomposition (T^{-1} = V D^{-1} V^T).

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAEnsembleTFactor()`, `PetscDAEnsembleApplySqrtTInverse()`
@*/
PetscErrorCode PetscDAEnsembleApplyTInverse(PetscDA da, Vec sdel, Vec w)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(sdel, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 3);

  PetscCheck(en->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "T matrix not factored. Call PetscDAEnsembleTFactor first");

  switch (en->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(ApplyTInverse_Cholesky(da, sdel, w));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(ApplyTInverse_Eigen(da, sdel, w));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)en->sqrt_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplySqrtTInverse_Cholesky - Computes Y = L^{-T} * U using Cholesky factorization

  Notes:
  For T = L * L^T (Cholesky factorization), this computes the ASYMMETRIC square root
  T^{-1/2} = L^{-T} (upper triangular).

  This satisfies the product property:
    T^{-1/2} * (T^{-1/2})^T = L^{-T} * L^{-1} = (L * L^T)^{-1} = T^{-1}

  WARNING: L^{-T} is upper triangular and NOT symmetric. This is valid for ETKF where
  the global ensemble transform W = X_a * T^{-1/2} does not require symmetry. However,
  LETKF requires a SYMMETRIC square root T^{-1/2} = V * D^{-1/2} * V^T for the local
  ensemble perturbation update. Use PETSCDA_SQRT_EIGEN for LETKF.

  This requires solving L^T * Y = U for Y.
*/
static PetscErrorCode ApplySqrtTInverse_Cholesky(PetscDA da, Mat U, Mat Y)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscBLASInt       n, lda, nrhs, info;
  const PetscScalar *l_array;
  PetscScalar       *y_array;
  PetscInt           m_L, N_L, m_U, N_U;
  Mat                U_identity = NULL;

  PetscFunctionBegin;
  PetscCheck(en->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

  /* Get dimensions and validate compatibility */
  PetscCall(MatGetSize(en->L_cholesky, &m_L, &N_L));

  /* Handle NULL U (identity matrix case) */
  if (!U) {
    /* Create identity matrix of size m_L x m_L */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)en->L_cholesky), PETSC_DECIDE, PETSC_DECIDE, m_L, m_L, NULL, &U_identity));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)U_identity, "dense_"));
    PetscCall(MatSetFromOptions(U_identity));
    PetscCall(MatSetUp(U_identity));
    PetscCall(MatShift(U_identity, 1.0)); /* Set diagonal to 1 */
    U = U_identity;
  }

  PetscCall(MatGetSize(U, &m_U, &N_U));
  PetscCheck(m_L == m_U, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Cholesky factor rows (%" PetscInt_FMT ") must match U rows (%" PetscInt_FMT ")", m_L, m_U);

  PetscCall(PetscBLASIntCast(N_L, &n));
  PetscCall(PetscBLASIntCast(N_U, &nrhs));
  lda = n;

  /* Initialize Y with U for in-place solve */
  PetscCall(MatCopy(U, Y, SAME_NONZERO_PATTERN));

  /* Get direct array access */
  PetscCall(MatDenseGetArrayRead(en->L_cholesky, &l_array));
  PetscCall(MatDenseGetArray(Y, &y_array));

  /* Solve L^T * Y = U using LAPACK triangular solve (L is lower, so L^T is upper)
     TRTRS args: UPLO='L', TRANS='T', DIAG='N' */
  LAPACKtrtrs_("L", "T", "N", &n, &nrhs, (PetscScalar *)l_array, &lda, y_array, &n, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK triangular solve (xTRTRS): info=%" PetscInt_FMT, (PetscInt)info);

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayRead(en->L_cholesky, &l_array));
  PetscCall(MatDenseRestoreArray(Y, &y_array));

  PetscCall(MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY));

  /* Cleanup temporary identity matrix if created */
  if (U_identity) PetscCall(MatDestroy(&U_identity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplySqrtTInverse_Eigen - Computes Y = V * D^{-1/2} * V^T * U.

  Notes:
  This computes the symmetric square root T^{-1/2} = V * D^{-1/2} * V^T.
  The operation is performed as Y = V * (D^{-1/2} * (V^T * U)) to strictly follow
  linear algebra operations for general matrix U.
*/
static PetscErrorCode ApplySqrtTInverse_Eigen(PetscDA da, Mat U, Mat Y)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  Mat               W;
  Vec               diag_inv;

  PetscFunctionBegin;
  PetscCheck(en->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
  PetscCheck(en->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

  /* Prepare inverse sqrt eigenvalues: D^{-1/2}
     Note: en->sqrt_eigen_vals currently stores sqrt(D) */
  PetscCall(VecDuplicate(en->sqrt_eigen_vals, &diag_inv));
  PetscCall(VecCopy(en->sqrt_eigen_vals, diag_inv));
  PetscCall(VecReciprocal(diag_inv)); /* Now diag_inv contains 1/sqrt(D) = D^{-1/2} */

  if (U) {
    /* General case: Compute Y = V * D^{-1/2} * V^T * U */
    /* Step 1: Compute W = V^T * U (Project U onto eigenbasis) */
    PetscCall(MatTransposeMatMult(en->V, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &W));

    /* Step 2: Scale rows of W by D^{-1/2}: W <- D^{-1/2} * W */
    PetscCall(MatDiagonalScale(W, diag_inv, NULL));

    /* Step 3: Compute Y = V * W (Project back to standard basis)
       Y = V * (D^{-1/2} * V^T * U) */
    {
      Mat Y_temp;
      PetscCall(MatMatMult(en->V, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y_temp));
      PetscCall(MatCopy(Y_temp, Y, SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&Y_temp));
    }

    /* Cleanup */
    PetscCall(MatDestroy(&W));
  } else {
    /* U is NULL (identity): Compute Y = V * D^{-1/2} * V^T directly */
    /* Step 1: Compute W = V * D^{-1/2} (scale columns of V) */
    PetscCall(MatDuplicate(en->V, MAT_COPY_VALUES, &W));
    PetscCall(MatDiagonalScale(W, NULL, diag_inv));

    /* Step 2: Compute Y = W * V^T = V * D^{-1/2} * V^T */
    {
      Mat Y_temp;
      PetscCall(MatMatTransposeMult(W, en->V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y_temp));
      PetscCall(MatCopy(Y_temp, Y, SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&Y_temp));
    }

    /* Cleanup */
    PetscCall(MatDestroy(&W));
  }

  PetscCall(VecDestroy(&diag_inv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleApplySqrtTInverse - Apply T^{-1/2} to a matrix U [Alg 6.4 line 9]

  Collective

  Input Parameters:
+ da - the `PetscDA` context
- U  - input matrix (usually Identity, but can be general)

  Output Parameter:
. Y - output matrix Y = T^{-1/2} * U

  Notes:
  This function applies the inverse square root of T = I + S^T * S using the
  stored factorization.

  - For CHOLESKY mode: Computes Y = L^{-T} U
  - For EIGEN mode: Computes Y = V D^{-1/2} V^T U

  Both results satisfy Y^T * T * Y = U^T * U, preserving the metric.

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAEnsembleTFactor()`, `PetscDAEnsembleApplyTInverse()`
@*/
PetscErrorCode PetscDAEnsembleApplySqrtTInverse(PetscDA da, Mat U, Mat Y)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (U) PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(Y, MAT_CLASSID, 3);

  PetscCheck(en->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "I_StS matrix not created. Call PetscDAEnsembleTFactor first");

  switch (en->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(ApplySqrtTInverse_Cholesky(da, U, Y));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(ApplySqrtTInverse_Eigen(da, U, Y));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)en->sqrt_type);
  }

  /* Debugging verification: Check that metric is preserved
     Verify that Y^T * T * Y = U^T * U (or Y^T * T * Y = I if U is NULL) */
  if (PetscDefined(USE_DEBUG) && U) {
    Mat       YtTY, UtU, T_Y;
    PetscReal norm_ref, norm_diff;

    /* Compute LHS: Y^T * T * Y */
    PetscCall(MatMatMult(en->I_StS, Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_Y));     /* T * Y */
    PetscCall(MatTransposeMatMult(Y, T_Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &YtTY)); /* Y^T * (T * Y) */

    /* Compute RHS: U^T * U */
    PetscCall(MatTransposeMatMult(U, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UtU));

    /* Compute difference: Diff = LHS - RHS */
    PetscCall(MatAXPY(YtTY, -1.0, UtU, SAME_NONZERO_PATTERN));

    /* Check norms */
    PetscCall(MatNorm(UtU, NORM_FROBENIUS, &norm_ref));
    PetscCall(MatNorm(YtTY, NORM_FROBENIUS, &norm_diff));

    if (norm_ref > 0.0) PetscCheck(norm_diff / norm_ref < MATRIX_SQRT_TOLERANCE_FACTOR, PETSC_COMM_SELF, PETSC_ERR_PLIB, "T^{-1/2} verification failed. ||Y^T*T*Y - U^T*U||/||U^T*U|| = %g", (double)(norm_diff / norm_ref));

    /* Cleanup debug matrices */
    PetscCall(MatDestroy(&T_Y));
    PetscCall(MatDestroy(&YtTY));
    PetscCall(MatDestroy(&UtU));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleSetSqrtType - Selects the reduced-space square-root algorithm used during analysis.

  Logically Collective

  Input Parameters:
+ da   - the `PetscDA` object
- type - either `PETSCDA_SQRT_CHOLESKY` or `PETSCDA_SQRT_EIGEN`

  Options Database Key:
. -petscda_ensemble_sqrt_type <cholesky or eigen> - set the `PetscDASqrtType`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDASqrtType`, `PetscDAEnsembleGetSqrtType()`
@*/
PetscErrorCode PetscDAEnsembleSetSqrtType(PetscDA da, PetscDASqrtType type)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCheck(type == PETSCDA_SQRT_CHOLESKY || type == PETSCDA_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDA square-root type %" PetscInt_FMT, (PetscInt)type);

  en->sqrt_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleGetSqrtType - Retrieves the current square-root implementation configured for analysis.

  Not Collective

  Input Parameters:
. da - the `PetscDA` object

  Output Parameter:
. type - on output, the configured `PetscDASqrtType`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAEnsembleSetSqrtType()`
@*/
PetscErrorCode PetscDAEnsembleGetSqrtType(PetscDA da, PetscDASqrtType *type)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = en->sqrt_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleSetInflation - Sets the inflation factor for the data assimilation method.

  Logically Collective

  Input Parameters:
+ da        - the `PetscDA` context
- inflation - the inflation factor (must be >= 1.0)

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleGetInflation()`
@*/
PetscErrorCode PetscDAEnsembleSetInflation(PetscDA da, PetscReal inflation)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveReal(da, inflation, 2);
  PetscCheck(inflation >= 1.0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Inflation factor must be >= 1.0, got %g", (double)inflation);
  en->inflation = inflation;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleGetInflation - Gets the inflation factor for the data assimilation method.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. inflation - the inflation factor

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleSetInflation()`
@*/
PetscErrorCode PetscDAEnsembleGetInflation(PetscDA da, PetscReal *inflation)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(inflation, 2);
  *inflation = en->inflation;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleGetMember - Returns a read-only view of an ensemble member stored in the `PetscDA`.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
- member_idx - index of the requested member (0 <= idx < ensemble_size)

  Output Parameter:
. member - read-only vector view; call `PetscDAEnsembleRestoreMember()` when done

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleRestoreMember()`, `PetscDAEnsembleSetMember()`
@*/
PetscErrorCode PetscDAEnsembleGetMember(PetscDA da, PetscInt member_idx, Vec *member)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(member, 3);
  PetscCheck(en->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before accessing ensemble members");
  PetscCheck(member_idx >= 0 && member_idx < en->ensemble_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, en->ensemble_size);

  PetscCall(MatDenseGetColumnVecRead(en->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleRestoreMember - Returns a column view obtained with `PetscDAGetEnsembleMember()`.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
. member_idx - index that was previously requested
- member     - location that holds the view to restore

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleGetMember()`
@*/
PetscErrorCode PetscDAEnsembleRestoreMember(PetscDA da, PetscInt member_idx, Vec *member)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(member, 3);

  PetscCall(MatDenseRestoreColumnVecRead(en->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleSetMember - Overwrites an ensemble member with user-provided state data.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
. member_idx - index of the entry to modify
- member     - vector containing the new state values

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleGetMember()`
@*/
PetscErrorCode PetscDAEnsembleSetMember(PetscDA da, PetscInt member_idx, Vec member)
{
  Vec               col;
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(member, VEC_CLASSID, 3);
  PetscCheck(en->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before setting ensemble members");
  PetscCheck(member_idx >= 0 && member_idx < en->ensemble_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, en->ensemble_size);

  PetscCall(MatDenseGetColumnVecWrite(en->ensemble, member_idx, &col));
  PetscCall(VecCopy(member, col));
  PetscCall(MatDenseRestoreColumnVecWrite(en->ensemble, member_idx, &col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleComputeMean - Computes ensemble mean for a `PetscDA`

  Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. mean - vector that will hold the ensemble mean

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleComputeAnomalies()`
@*/
PetscErrorCode PetscDAEnsembleComputeMean(PetscDA da, Vec mean)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscCall((*en->computemean)(da, mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAEnsembleInitialize - Initialize ensemble members with Gaussian perturbations

  Input Parameters:
+ da            - PetscDA context
. x0            - Background state
. obs_error_std - Standard deviation for perturbations
- rng           - Random number generator

  Notes:
  Each ensemble member is initialized as x0 + Gaussian(0, obs_error_std)
*/
PetscErrorCode PetscDAEnsembleInitialize(PetscDA da, Vec x0, PetscReal obs_error_std, PetscRandom rng)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  Vec       member, col, x_mean;
  PetscInt  i;
  PetscReal scale;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rng, PETSC_RANDOM_CLASSID, 5);
  PetscCall(VecDuplicate(x0, &member));
  PetscCall(VecDuplicate(x0, &x_mean));

  /*
     Scale factor to maintain consistent ensemble spread across different ensemble sizes.
     After removing the sample mean, the ensemble variance is approximately:
       Var_final ~= Var_initial * (m-1)/m
     To maintain consistent initial spread regardless of m, we scale by sqrt(m/(m-1)).
     This ensures the final ensemble spread is approximately obs_error_std^2. */
  scale = PetscSqrtReal((PetscReal)en->ensemble_size / (PetscReal)(en->ensemble_size - 1));

  /* Populate the Gaussian draws with scaled standard deviation */
  for (i = 0; i < en->ensemble_size; i++) {
    PetscCall(VecSetRandomGaussian(member, rng, 0.0, obs_error_std * scale));
    PetscCall(PetscDAEnsembleSetMember(da, i, member));
  }
  /* get mean of perturbations */
  PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
  /* remove mean and add x0 */
  for (i = 0; i < en->ensemble_size; i++) {
    PetscCall(MatDenseGetColumnVecWrite(en->ensemble, i, &col));
    PetscCall(VecAXPY(col, -1.0, x_mean));
    PetscCall(VecAXPY(col, 1.0, x0));
    PetscCall(MatDenseRestoreColumnVecWrite(en->ensemble, i, &col));
  }

  PetscCall(VecDestroy(&member));
  PetscCall(VecDestroy(&x_mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleComputeAnomalies - Forms the state-space anomalies matrix for a `PetscDA`.

  Collective

  Input Parameters:
+ da   - the `PetscDA` context
- mean - optional mean state vector (pass `NULL` to compute internally)

  Output Parameter:
. anomalies - location to store the newly created anomalies matrix

  Notes:
  If `mean` is `NULL`, the function will create a temporary vector and compute
  the ensemble mean using `PetscDAComputeEnsembleMean()`. If `mean` is provided,
  it will be used directly, which can improve performance when the mean has
  already been computed.

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleComputeMean()`
@*/
PetscErrorCode PetscDAEnsembleComputeAnomalies(PetscDA da, Vec mean, Mat *anomalies)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (mean) PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscAssertPointer(anomalies, 3);
  PetscCall((*en->computeanomalies)(da, mean, anomalies));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleAnalysis - Executes the analysis (update) step using sparse observation matrix H

  Collective

  Input Parameters:
+ da          - the `PetscDA` context
. observation - observation vector y in R^P
- H           - observation operator matrix (P x N), sparse AIJ format

  Notes:
  The observation matrix H maps from state space (N dimensions) to observation
  space (P dimensions): y = H*x + noise

  H must be a sparse AIJ matrix

  For identity observations (observe entire state), use an identity matrix for H.
  For partial observations, set appropriate rows and columns to observe
  specific state components.

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleForecast()`, `PetscDASetObsErrorVariance()`
@*/
PetscErrorCode PetscDAEnsembleAnalysis(PetscDA da, Vec observation, Mat H)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscInt          h_rows, h_cols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 3);

  /* Validate H dimensions match PetscDA configuration */
  PetscCall(MatGetSize(H, &h_rows, &h_cols));
  PetscCheck(h_rows == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "H matrix rows (%" PetscInt_FMT ") must match obs_size (%" PetscInt_FMT ")", h_rows, da->obs_size);
  PetscCheck(h_cols == da->state_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "H matrix cols (%" PetscInt_FMT ") must match state_size (%" PetscInt_FMT ")", h_cols, da->state_size);
  PetscCall(VecGetSize(observation, &h_rows));
  PetscCheck(h_rows == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "observation vector size (%" PetscInt_FMT ") must match obs_size (%" PetscInt_FMT ")", h_rows, da->obs_size);

  PetscCall(PetscLogEventBegin(PetscDA_Analysis, (PetscObject)da, 0, 0, 0));
  PetscCall((*en->analysis)(da, observation, H));
  PetscCall(PetscLogEventEnd(PetscDA_Analysis, (PetscObject)da, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDAEnsembleForecast - Advances every ensemble member through the user-supplied forecast model.

  Collective

  Input Parameters:
+ da    - the `PetscDA` context
. model - routine that evaluates the model map `f(input, output; ctx)`
- ctx   - optional context for `model`

  Level: intermediate

.seealso: [](ch_da), `PetscDAEnsembleAnalysis()`
@*/
PetscErrorCode PetscDAEnsembleForecast(PetscDA da, PetscErrorCode (*model)(Vec, Vec, PetscCtx), PetscCtx ctx)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCall((*en->forecast)(da, model, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDAView_Ensemble(PetscDA da, PetscViewer viewer)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Ensemble size: %" PetscInt_FMT "\n", en->ensemble_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Assembled: %s\n", en->assembled ? "true" : "false"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Inflation: %g\n", (double)en->inflation));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Square root type: %s\n", (en->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDASetUp_Ensemble(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  MPI_Comm          comm;

  PetscFunctionBegin;
  if (en->assembled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set state size before calling PetscDASetUp()");
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set observation size before calling PetscDASetUp()");
  PetscCheck(en->ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set ensemble size before calling PetscDASetUp()");

  comm = PetscObjectComm((PetscObject)da);
  if (!en->ensemble) {
    PetscCall(MatCreateDense(comm, da->local_state_size, PETSC_DECIDE, da->state_size, en->ensemble_size, NULL, &en->ensemble));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)en->ensemble, "dense_"));
    PetscCall(MatSetFromOptions(en->ensemble));
    PetscCall(MatSetUp(en->ensemble));
  }
  en->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleSetSize - Sets the ensemble dimensions used by a `PetscDA`.

  Collective

  Input Parameters:
+ da            - the `PetscDA` context
- ensemble_size - number of ensemble members

  Options Database Key:
. -petscda_ensemble_size <size> - number of ensemble members

  Level: beginner

.seealso: [](ch_da), `PetscDAGetSizes()`, `PetscDASetSizes()`, `PetscDASetUp()`
@*/
PetscErrorCode PetscDAEnsembleSetSize(PetscDA da, PetscInt ensemble_size)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, ensemble_size, 4);
  PetscCheck(!en->assembled, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Cannot change sizes after PetscDASetUp() has been called");
  en->ensemble_size = ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleGetSize - Retrieves the dimension of the ensemble in a `PetscDA`.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameters:
. ensemble_size - number of ensemble members

  Level: beginner

.seealso: [](ch_da), `PetscDASetSizes()`, `PetscDAGetSizes()`
@*/
PetscErrorCode PetscDAEnsembleGetSize(PetscDA da, PetscInt *ensemble_size)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  *ensemble_size = en->ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDASetFromOptions_Ensemble(PetscDA da, PetscOptionItems *PetscOptionsObjectPtr)
{
  PetscDA_Ensemble *en                 = (PetscDA_Ensemble *)da->data;
  PetscOptionItems  PetscOptionsObject = *PetscOptionsObjectPtr;
  char              sqrt_type_name[256];
  PetscBool         sqrt_set = PETSC_FALSE;
  const char       *sqrt_default;
  PetscDASqrtType   sqrt_type;
  PetscReal         inflation_val = en->inflation;
  PetscBool         inflation_set;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscDA Ensemble Options");

  PetscCall(PetscOptionsReal("-petscda_ensemble_inflation", "Inflation factor", "PetscDASetInflation", en->inflation, &inflation_val, &inflation_set));
  if (inflation_set) PetscCall(PetscDAEnsembleSetInflation(da, inflation_val));

  sqrt_default = (en->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky";
  PetscCall(PetscOptionsString("-petscda_ensemble_sqrt_type", "Matrix square root factorization", "PetscDASetSqrtType", sqrt_default, sqrt_type_name, sizeof(sqrt_type_name), &sqrt_set));
  if (sqrt_set) {
    PetscBool match_cholesky, match_eigen;
    PetscCall(PetscStrcmp(sqrt_type_name, "cholesky", &match_cholesky));
    PetscCall(PetscStrcmp(sqrt_type_name, "eigen", &match_eigen));
    if (match_cholesky) {
      sqrt_type = PETSCDA_SQRT_CHOLESKY;
    } else if (match_eigen) {
      sqrt_type = PETSCDA_SQRT_EIGEN;
    } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDA square-root type \"%s\"", sqrt_type_name);
    PetscCall(PetscDAEnsembleSetSqrtType(da, sqrt_type));
  }
  PetscCall(PetscOptionsInt("-petscda_ensemble_size", "Number of ensemble members", "PetscDAEnsembleSetSize", en->ensemble_size, &en->ensemble_size, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDADestroy_Ensemble(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&en->ensemble));
  PetscCall(VecDestroy(&da->obs_error_var));
  PetscCall(MatDestroy(&da->R));

  /* Destroy T-matrix factorization data */
  PetscCall(MatDestroy(&en->V));
  PetscCall(MatDestroy(&en->L_cholesky));
  PetscCall(VecDestroy(&en->sqrt_eigen_vals));
  PetscCall(MatDestroy(&en->I_StS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute mean of ensemble */
PETSC_INTERN PetscErrorCode PetscDAEnsembleComputeMean_Default(PetscDA da, Vec mean)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscScalar       inv_m;
  PetscInt          m;

  PetscFunctionBegin;
  PetscCheck(en->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before computing the ensemble mean");
  PetscCheck(en->ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Ensemble size must be positive");

  m     = en->ensemble_size;
  inv_m = 1.0 / (PetscScalar)m;
  /* Compute observation mean mean = (1/m) * sum(E_i) */
  PetscCall(MatGetRowSum(en->ensemble, mean));
  PetscCall(VecScale(mean, inv_m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Alg 6.4 line 2: Compute anomalies X = (E - mean) / sqrt(m-1)
PETSC_INTERN PetscErrorCode PetscDAEnsembleComputeAnomalies_Default(PetscDA da, Vec mean_in, Mat *anomalies_out)
{
  PetscDA_Ensemble *en   = (PetscDA_Ensemble *)da->data;
  Vec               mean = NULL;
  Vec               col_in, col_out;
  Mat               anomalies;
  MPI_Comm          comm;
  PetscReal         scale;
  PetscInt          ensemble_size;
  PetscInt          j;
  PetscBool         mean_created = PETSC_FALSE;

  PetscFunctionBegin;
  PetscAssertPointer(anomalies_out, 3);
  PetscCheck(en->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before computing anomalies");
  PetscCheck(en->ensemble_size > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least 2 to form anomalies");
  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "State size must be positive");

  /* Cache frequently-used values for clarity and efficiency */
  ensemble_size = en->ensemble_size;
  comm          = PetscObjectComm((PetscObject)en->ensemble);

  /*
    Compute normalization scale for anomalies.
    Alg 6.4 line 2: anomalies are normalized by 1/sqrt(m-1) so that
    the anomalies matrix X satisfies X*X^T = ensemble covariance matrix.
    This ensures proper statistical properties for ensemble-based methods.
  */
  scale = 1.0 / PetscSqrtReal((PetscReal)(ensemble_size - 1));

  /* Allocate anomalies matrix (state_size x ensemble_size) */
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, da->state_size, ensemble_size, NULL, &anomalies));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)anomalies, "dense_"));
  PetscCall(MatSetFromOptions(anomalies));
  PetscCall(MatSetUp(anomalies));

  /* Use provided mean or create and compute it */
  if (mean_in) {
    mean = mean_in;
  } else {
    /* Create and compute ensemble mean vector */
    PetscCall(MatCreateVecs(anomalies, NULL, &mean));
    PetscCall(VecSetFromOptions(mean));
    mean_created = PETSC_TRUE;

    /* Alg 6.4 line 1: \bar{x} = (1/m)\sum_j x^{(j)} */
    PetscCall(PetscDAEnsembleComputeMean(da, mean));
  }

  /*
    Form anomalies by subtracting mean from each ensemble member and scaling.
    For each column j: anomaly_j = (ensemble_j - mean) / sqrt(m-1)
  */
  for (j = 0; j < ensemble_size; ++j) { // should be locals only
    PetscCall(MatDenseGetColumnVecRead(en->ensemble, j, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(anomalies, j, &col_out));

    /* Alg 6.4 line 2: subtract the mean column-wise to form x^{(j)} - \bar{x} */
    PetscCall(VecWAXPY(col_out, -1.0, mean, col_in));
    /* Alg 6.4 line 2: scale anomalies by 1/\sqrt{m-1} */
    PetscCall(VecScale(col_out, scale));

    PetscCall(MatDenseRestoreColumnVecWrite(anomalies, j, &col_out));
    PetscCall(MatDenseRestoreColumnVecRead(en->ensemble, j, &col_in));
  }

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(anomalies, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(anomalies, MAT_FINAL_ASSEMBLY));

  /* Transfer ownership to output and clean up temporary resources */
  *anomalies_out = anomalies;
  if (mean_created) PetscCall(VecDestroy(&mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDACreate_Ensemble(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  en->computemean      = PetscDAEnsembleComputeMean_Default;
  en->computeanomalies = PetscDAEnsembleComputeAnomalies_Default;

  en->ensemble_size = 0;
  en->ensemble      = NULL;
  en->assembled     = PETSC_FALSE;
  en->inflation     = 1.0;

  /* Initialize T-matrix factorization fields */
  en->sqrt_type       = PETSCDA_SQRT_EIGEN;
  en->V               = NULL;
  en->L_cholesky      = NULL;
  en->sqrt_eigen_vals = NULL;
  en->I_StS           = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeNormalizedInnovationMatrix - Computes S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1) [Alg 6.4 line 5]

  Input Parameters:
+ Z          - observation ensemble matrix
. y_mean     - mean of observations
. r_inv_sqrt - R^{-1/2}
. m          - ensemble size
- scale      - 1/sqrt(m-1)

  Output Parameter:
. S - normalized innovation matrix
*/
static PetscErrorCode ComputeNormalizedInnovationMatrix(Mat Z, Vec y_mean, Vec r_inv_sqrt, PetscInt m, PetscScalar scale, Mat S)
{
  const PetscScalar *z_array, *y_array, *r_array;
  PetscScalar       *s_array;
  PetscInt           obs_size, obs_size_local, z_cols, i, j;
  PetscInt           y_local_size, r_local_size;
  PetscInt           lda_z, lda_s;

  PetscFunctionBegin;
  PetscCheck(m > 0, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive, got %" PetscInt_FMT, m);

  /* Get observation size from input matrix Z and validate dimensions */
  PetscCall(MatGetSize(Z, &obs_size, &z_cols));
  PetscCall(MatGetLocalSize(Z, &obs_size_local, NULL));
  PetscCheck(z_cols == m, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Matrix Z has %" PetscInt_FMT " columns but ensemble size is %" PetscInt_FMT, z_cols, m);

  /* Verify vector dimensions match observation size (both global and local) */
  PetscCall(VecGetLocalSize(y_mean, &y_local_size));
  PetscCall(VecGetLocalSize(r_inv_sqrt, &r_local_size));
  PetscCheck(y_local_size == obs_size_local, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Vector y_mean local size %" PetscInt_FMT " does not match matrix local rows %" PetscInt_FMT, y_local_size, obs_size_local);
  PetscCheck(r_local_size == obs_size_local, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Vector r_inv_sqrt local size %" PetscInt_FMT " does not match matrix local rows %" PetscInt_FMT, r_local_size, obs_size_local);

  /* Get direct access to arrays for performance */
  PetscCall(MatDenseGetArrayRead(Z, &z_array));
  PetscCall(MatDenseGetArrayWrite(S, &s_array));
  PetscCall(VecGetArrayRead(y_mean, &y_array));
  PetscCall(VecGetArrayRead(r_inv_sqrt, &r_array));

  /* Get Leading Dimension (LDA) to handle padding/strides correctly */
  PetscCall(MatDenseGetLDA(Z, &lda_z));
  PetscCall(MatDenseGetLDA(S, &lda_s));

  /* Compute normalized innovation: S_ij = (Z_ij - y_mean_i) * scale * r_inv_sqrt_i
     Iterate column-wise (j) then row-wise (i) for optimal cache access with column-major storage */
  for (j = 0; j < m; j++) {
    const PetscScalar *z_col = z_array + j * lda_z;
    PetscScalar       *s_col = s_array + j * lda_s;

    for (i = 0; i < obs_size_local; i++) s_col[i] = (z_col[i] - y_array[i]) * scale * r_array[i];
  }

  /* Restore arrays */
  PetscCall(VecRestoreArrayRead(r_inv_sqrt, &r_array));
  PetscCall(VecRestoreArrayRead(y_mean, &y_array));
  PetscCall(MatDenseRestoreArrayWrite(S, &s_array));
  PetscCall(MatDenseRestoreArrayRead(Z, &z_array));

  /* Finalize assembly */
  PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  BroadcastWeightVector - Creates matrix with weight vector replicated across all columns

  Input Parameters:
+ w - weight vector of size m (analysis weights from ETKF update)
- m - ensemble size (number of columns to replicate, must equal vector size)

  Output Parameter:
. w_ones - m x m dense matrix where each column is a copy of w (i.e., w * 1^T)

  Notes:
  This function constructs the broadcast matrix w * 1^T, where w is the m-dimensional
  weight vector and 1 is an m-dimensional vector of ones. This matrix is a fundamental
  component in the ETKF transform: G = w * 1^T + sqrt(m-1) * T^{1/2} * U.

  The implementation uses direct array access for performance, avoiding the overhead of
  repeated vector wrapping and copying. This is particularly efficient for dense matrices
  where memory is contiguous column-wise.

  Complexity: O(m^2) time and memory.

  Level: developer

*/
static PetscErrorCode BroadcastWeightVector(Vec w, PetscInt m, Mat w_ones)
{
  const PetscScalar *w_array;
  PetscScalar       *mat_array;
  PetscInt           w_size, w_size_local, mat_rows_local, mat_cols_local;
  PetscInt           i, lda;

  PetscFunctionBegin;
  PetscCheck(m > 0, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive for broadcasting, got %" PetscInt_FMT, m);
  /* Check for potential overflow in matrix size calculation */
  PetscCheck(m <= PETSC_MAX_INT / m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m = %" PetscInt_FMT " too large", m);

  /* Verify dimensions */
  PetscCall(VecGetSize(w, &w_size));
  PetscCall(VecGetLocalSize(w, &w_size_local));
  PetscCheck(w_size == m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_INCOMP, "Weight vector global size (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ")", w_size, m);

  /* Verify consistent parallel layout between vector and matrix */
  PetscCall(MatGetLocalSize(w_ones, &mat_rows_local, &mat_cols_local));
  PetscCheck(mat_rows_local == w_size_local, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix row distribution (%" PetscInt_FMT ") inconsistent with vector distribution (%" PetscInt_FMT ")", mat_rows_local, w_size_local);
  PetscCheck(mat_cols_local == m, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix local columns (%" PetscInt_FMT ") must equal global columns m (%" PetscInt_FMT ") for MPIDense", mat_cols_local, m);

  /* Access raw arrays for efficient broadcasting */
  PetscCall(VecGetArrayRead(w, &w_array));
  PetscCall(MatDenseGetArrayWrite(w_ones, &mat_array));
  PetscCall(MatDenseGetLDA(w_ones, &lda));

  /* Copy w to each column of w_ones */
  /* Note: MatDense uses column-major storage. We copy the vector w into each column. */
  for (i = 0; i < m; i++) PetscCall(PetscArraycpy(mat_array + i * lda, w_array, w_size_local));

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayWrite(w_ones, &mat_array));
  PetscCall(VecRestoreArrayRead(w, &w_array));

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(w_ones, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(w_ones, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  UpdateEnsembleWithTransform - Updates ensemble via ETKF transform: E = mean * 1' + X * G [Alg 6.4 line 9]

  Input Parameters:
+ mean     - ensemble mean vector (size state_size), must be initialized
. X        - scaled anomaly matrix (state_size x ensemble_size), X = (E - mean*1')/sqrt(m-1)
. G        - ETKF transform matrix (ensemble_size x ensemble_size), G = w*1' + sqrt(m-1)*T^{1/2}*U
. m        - ensemble size (number of columns in ensemble), must be > 0
- ensemble - ensemble matrix to update in-place (state_size x ensemble_size)

  Notes:
  This function performs the final step (Step 10) of the ETKF analysis algorithm from
  Asch, M., Bocquet, M., and Nodet, M., transforming the forecast ensemble into the analysis ensemble.
  The operation E^a = mean + X * G is computed using matrix-matrix multiplication followed
  by column-wise addition to efficiently handle large state spaces.

  Error Handling:
  - Validates all input dimensions for consistency
  - Checks for positive ensemble size
  - Ensures proper matrix/vector initialization
  - Handles parallel assembly correctly

  Performance Considerations:
  - Memory: Creates one temporary matrix X_G of size (state_size x m)
  - Time complexity: O(state_size * m^2) for matrix multiply + O(state_size * m) for additions
  - Optimization: Uses direct array access for dense matrices to avoid Vec overhead
  - Parallel: Fully parallelizable across both matrix multiply and column updates

  Level: developer

*/
static PetscErrorCode UpdateEnsembleWithTransform(Vec mean, Mat X, Mat G, PetscInt m, Mat ensemble)
{
  Mat                X_G;
  const PetscScalar *xg_array, *mean_array;
  PetscScalar       *ens_array;
  PetscInt           x_rows, x_cols, g_rows, g_cols, ens_rows, ens_cols;
  PetscInt           n_local_ens, n_local_xg, mean_local_size;
  PetscInt           lda_ens, lda_xg;
  PetscInt           mean_size, i, j;

  PetscFunctionBegin;
  /* Validate input parameters for correct types and null pointers */
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(G, MAT_CLASSID, 3);
  PetscValidLogicalCollectiveInt(X, m, 4);
  PetscValidHeaderSpecific(ensemble, MAT_CLASSID, 5);

  /* Retrieve and validate matrix dimensions for compatibility */
  PetscCall(MatGetSize(X, &x_rows, &x_cols));
  PetscCall(MatGetSize(G, &g_rows, &g_cols));
  PetscCall(MatGetSize(ensemble, &ens_rows, &ens_cols));
  PetscCall(VecGetSize(mean, &mean_size));

  /* Verify dimension consistency across all inputs */
  PetscCheck(x_cols == m, PetscObjectComm((PetscObject)X), PETSC_ERR_ARG_INCOMP, "Anomaly matrix X columns (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", x_cols, m);
  PetscCheck(g_rows == m, PetscObjectComm((PetscObject)G), PETSC_ERR_ARG_INCOMP, "Transform matrix G rows (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", g_rows, m);
  PetscCheck(g_cols == m, PetscObjectComm((PetscObject)G), PETSC_ERR_ARG_INCOMP, "Transform matrix G must be square, got %" PetscInt_FMT " x %" PetscInt_FMT, g_rows, g_cols);
  PetscCheck(ens_rows == x_rows, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_INCOMP, "Ensemble rows (%" PetscInt_FMT ") must match anomaly matrix X rows (%" PetscInt_FMT ")", ens_rows, x_rows);
  PetscCheck(ens_cols == m, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_INCOMP, "Ensemble columns (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", ens_cols, m);
  PetscCheck(mean_size == x_rows, PetscObjectComm((PetscObject)mean), PETSC_ERR_ARG_INCOMP, "Mean vector size (%" PetscInt_FMT ") must match state size (%" PetscInt_FMT ")", mean_size, x_rows);

  /* Compute transformed anomaly matrix: X_G = X * G (state_size x m) */
  PetscCall(MatMatMult(X, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X_G));

  /* Access underlying data arrays for direct performance access
     This avoids creating/destroying m Vec objects and calling VecWAXPY m times. */
  PetscCall(MatDenseGetArrayRead(X_G, &xg_array));
  PetscCall(MatDenseGetArrayWrite(ensemble, &ens_array));
  PetscCall(VecGetArrayRead(mean, &mean_array));

  /* Get local dimensions and strides for array traversal */
  PetscCall(MatGetLocalSize(ensemble, &n_local_ens, NULL));
  PetscCall(MatGetLocalSize(X_G, &n_local_xg, NULL));
  PetscCall(VecGetLocalSize(mean, &mean_local_size));
  PetscCall(MatDenseGetLDA(ensemble, &lda_ens));
  PetscCall(MatDenseGetLDA(X_G, &lda_xg));

  /* Verify local dimensions match before direct array access */
  PetscCheck(n_local_ens == n_local_xg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local row size mismatch: ensemble (%" PetscInt_FMT ") vs X_G (%" PetscInt_FMT ")", n_local_ens, n_local_xg);
  PetscCheck(n_local_ens == mean_local_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local row size mismatch: ensemble (%" PetscInt_FMT ") vs mean (%" PetscInt_FMT ")", n_local_ens, mean_local_size);

  /* Update each ensemble member: E_ij = (XG)_ij + mean_i
     Loop over columns (j) and rows (i) of the local data block */
  for (j = 0; j < m; j++) {
    const PetscScalar *xg_col  = xg_array + j * lda_xg;
    PetscScalar       *ens_col = ens_array + j * lda_ens;
    for (i = 0; i < n_local_ens; i++) ens_col[i] = xg_col[i] + mean_array[i];
  }

  /* Restore arrays and finalize assembly */
  PetscCall(VecRestoreArrayRead(mean, &mean_array));
  PetscCall(MatDenseRestoreArrayWrite(ensemble, &ens_array));
  PetscCall(MatDenseRestoreArrayRead(X_G, &xg_array));

  PetscCall(MatAssemblyBegin(ensemble, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ensemble, MAT_FINAL_ASSEMBLY));

  /* Clean up temporary transformed anomaly matrix */
  PetscCall(MatDestroy(&X_G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDADestroy_ETKF(PetscDA da)
{
  PetscDA_ETKF *impl = (PetscDA_ETKF *)da->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&impl->mean));
  PetscCall(VecDestroy(&impl->y_mean));
  PetscCall(VecDestroy(&impl->delta_scaled));
  PetscCall(VecDestroy(&impl->w));
  PetscCall(VecDestroy(&impl->r_inv_sqrt));
  PetscCall(MatDestroy(&impl->Z));
  PetscCall(MatDestroy(&impl->S));
  PetscCall(MatDestroy(&impl->T_sqrt));
  PetscCall(MatDestroy(&impl->w_ones));
  PetscCall(PetscDADestroy_Ensemble(da));
  PetscCall(PetscFree(da->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDAEnsembleAnalysis_ETKF(PetscDA da, Vec observation, Mat H)
{
  PetscDA_ETKF *impl = (PetscDA_ETKF *)da->data;
  Mat           X;
  PetscInt      m;
  PetscScalar   scale, sqrt_m_minus_1;
  PetscBool     reallocate = PETSC_FALSE;

  PetscFunctionBegin;
  /* Validate ensemble size */
  m = impl->en.ensemble_size;
  PetscCheck(m > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be > 1, got %" PetscInt_FMT, m);

  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  PetscCall(PetscInfo(da, "square root type %s, %" PetscInt_FMT " ensembles\n", (impl->en.sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky", m));

  /* Check for reallocation needs */
  if (impl->mean) {
    PetscInt mean_size;
    PetscCall(VecGetSize(impl->mean, &mean_size));
    if (mean_size != da->state_size) reallocate = PETSC_TRUE;
  }
  if (impl->Z) {
    PetscInt z_rows, z_cols;
    PetscCall(MatGetSize(impl->Z, &z_rows, &z_cols));
    if (z_rows != da->obs_size || z_cols != m) reallocate = PETSC_TRUE;
  }
  if (impl->w) {
    PetscInt w_size;
    PetscCall(VecGetSize(impl->w, &w_size));
    if (w_size != m) reallocate = PETSC_TRUE;
  }

  /* Initialize or reallocate persistent work objects */
  if (!impl->mean || reallocate) {
    PetscCall(VecDestroy(&impl->mean));
    PetscCall(VecDestroy(&impl->y_mean));
    PetscCall(VecDestroy(&impl->delta_scaled));
    PetscCall(VecDestroy(&impl->w));
    PetscCall(VecDestroy(&impl->r_inv_sqrt));
    PetscCall(MatDestroy(&impl->Z));
    PetscCall(MatDestroy(&impl->S));
    PetscCall(MatDestroy(&impl->T_sqrt));
    PetscCall(MatDestroy(&impl->w_ones));

    /* Create mean vector from ensemble matrix (right vector = state space) */
    PetscCall(MatCreateVecs(impl->en.ensemble, NULL, &impl->mean));

    /* Create Z matrix (obs_size x m) */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)impl->en.ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, &impl->Z));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->Z, "dense_"));
    PetscCall(MatSetFromOptions(impl->Z));
    PetscCall(MatSetUp(impl->Z));

    /* Create observation space vectors from Z matrix (left vector = observation space) */
    PetscCall(MatCreateVecs(impl->Z, NULL, &impl->y_mean));
    PetscCall(VecDuplicate(impl->y_mean, &impl->delta_scaled));
    PetscCall(VecDuplicate(da->obs_error_var, &impl->r_inv_sqrt));

    /* Create w vector (size m) for analysis weights */
    PetscCall(MatCreateVecs(impl->Z, &impl->w, NULL));

    /* Create S matrix (same layout as Z) */
    PetscCall(MatDuplicate(impl->Z, MAT_DO_NOT_COPY_VALUES, &impl->S));

    /* Create T_sqrt matrix (m x m) - usually small */
    /* T_sqrt will hold the result of applying T^{-1/2} to identity matrix */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)impl->en.ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &impl->T_sqrt));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->T_sqrt, "dense_"));
    PetscCall(MatSetFromOptions(impl->T_sqrt));
    PetscCall(MatSetUp(impl->T_sqrt));

    /* Create w_ones matrix (m x m) */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)impl->en.ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &impl->w_ones));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->w_ones, "dense_"));
    PetscCall(MatSetFromOptions(impl->w_ones));
    PetscCall(MatSetUp(impl->w_ones));
  }

  /* Alg 6.4 line 1-2: Compute ensemble mean and scaled anomalies */
  PetscCall(PetscDAEnsembleComputeMean(da, impl->mean));

  /* X = (E - x_mean * 1') / sqrt(m - 1) */
  /* Note: PetscDAComputeAnomalies creates a NEW matrix X every time.
     We should probably optimize this too in the future, but for now we follow the API. */
  PetscCall(PetscDAEnsembleComputeAnomalies(da, impl->mean, &X));

  /* Alg 6.4 line 3-4: Compute observation ensemble Z = H * E */
  {
    MatReuse scall = MAT_INITIAL_MATRIX;
    if (impl->Z) {
      PetscInt z_rows, z_cols;
      PetscCall(MatGetSize(impl->Z, &z_rows, &z_cols));
      if (z_rows == da->obs_size && z_cols == m) scall = MAT_REUSE_MATRIX;
      else {
        PetscCall(MatDestroy(&impl->Z));
        scall = MAT_INITIAL_MATRIX;
      }
    }
    PetscCall(MatMatMult(H, impl->en.ensemble, scall, PETSC_DEFAULT, &impl->Z));
  }

  /* Compute observation mean y_mean = H * x_mean */
  PetscCall(MatMult(H, impl->mean, impl->y_mean));

  /* Alg 6.4 line 5-6: Build normalized innovation statistics */
  PetscCall(VecCopy(da->obs_error_var, impl->r_inv_sqrt));
  PetscCall(VecSqrtAbs(impl->r_inv_sqrt));
  PetscCall(VecReciprocal(impl->r_inv_sqrt));

  /* S = R^{-1/2} * (Z - y_mean * 1') / sqrt(m - 1) */
  PetscCall(ComputeNormalizedInnovationMatrix(impl->Z, impl->y_mean, impl->r_inv_sqrt, m, scale, impl->S));

  /* delta_scaled = R^{-1/2} * (y^o - y_mean) [Alg 6.4 line 6] */
  PetscCall(VecWAXPY(impl->delta_scaled, -1.0, impl->y_mean, observation));
  PetscCall(VecPointwiseMult(impl->delta_scaled, impl->delta_scaled, impl->r_inv_sqrt));

  /* Alg 6.4 line 7: Factor T = (I + S^T S) and store factorization */
  /* Note: Inflation is handled inside PetscDAEnsembleTFactor by shifting the diagonal of T */
  PetscCall(PetscDAEnsembleTFactor(da, impl->S));

  /* Alg 6.4 line 8: Compute analysis weights w = T^{-1} * S^T * delta_scaled */
  {
    Vec s_transpose_delta;
    /* Create temporary vector for S^T * delta_scaled */
    PetscCall(MatCreateVecs(impl->Z, &s_transpose_delta, NULL));
    PetscCall(MatMultTranspose(impl->S, impl->delta_scaled, s_transpose_delta));

    PetscCall(PetscDAEnsembleApplyTInverse(da, s_transpose_delta, impl->w));
    PetscCall(VecDestroy(&s_transpose_delta));
  }

  /* Alg 6.4 line 9: Compute square-root transform T^{-1/2} */
  PetscCall(PetscDAEnsembleApplySqrtTInverse(da, NULL, impl->T_sqrt));

  /* Alg 6.4 line 9: Form transform G = w * 1' + sqrt(m - 1) * T^{1/2} * U */
  {
    Mat T_sqrt_scaled;
    PetscCall(MatDuplicate(impl->T_sqrt, MAT_COPY_VALUES, &T_sqrt_scaled));
    PetscCall(MatScale(T_sqrt_scaled, sqrt_m_minus_1));

    /* w_ones = w * 1' (broadcast weight vector to all columns) */
    PetscCall(BroadcastWeightVector(impl->w, m, impl->w_ones));

    /* G = w_ones + sqrt(m-1)*T_sqrt
     Accumulate the scaled T_sqrt into w_ones to form the transform matrix G */
    PetscCall(MatAXPY(impl->w_ones, 1.0, T_sqrt_scaled, SAME_NONZERO_PATTERN));

    PetscCall(MatDestroy(&T_sqrt_scaled));
  }

  /* Alg 6.4 line 9: Update ensemble E = x_mean * 1' + X * G */
  PetscCall(UpdateEnsembleWithTransform(impl->mean, X, impl->w_ones, m, impl->en.ensemble));

  /* Cleanup temporary X matrix */
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDAEnsembleForecast_Ensemble(PetscDA da, PetscErrorCode (*model)(Vec, Vec, PetscCtx), PetscCtx ctx)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  Vec               col_in, col_out, temp;
  PetscInt          i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  /* Create temp vector from ensemble matrix (right vector = state space) */
  PetscCall(MatCreateVecs(en->ensemble, NULL, &temp));

  for (i = 0; i < en->ensemble_size; i++) {
    PetscCall(MatDenseGetColumnVecRead(en->ensemble, i, &col_in));
    PetscCall(model(col_in, temp, ctx));
    PetscCall(MatDenseRestoreColumnVecRead(en->ensemble, i, &col_in));

    PetscCall(MatDenseGetColumnVecWrite(en->ensemble, i, &col_out));
    PetscCall(VecCopy(temp, col_out));
    PetscCall(MatDenseRestoreColumnVecWrite(en->ensemble, i, &col_out));
  }

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCDAETKF - Ensemble transform Kalman filter data assimilation using a deterministic square-root update that avoids stochastic perturbations.

   Options Database Keys:
+  -petscda_type etkf                            - set the `PetscDAType` to `PETSCDALETKF`
.  -petscda_ensemble_size <size>                 - number of ensemble members
-  -petscda_ensemble_sqrt_type <cholesky, eigen> - the square root of the matrix to use

   Level: beginner

   Note:
   The ETKF algorithm is based on Algorithm 6.4 in {cite}`da2016`

.seealso: [](ch_da), `PetscDA`, `PetscDACreate()`, `PETSCDALETKF`, `PetscDAEnsembleSetSize()`, `PetscDASetSizes()`, `PetscDAEnsembleSetSqrtType()`,
          `PetscDAEnsembleSetInflation()`,
          `PetscDAEnsembleComputeMean()`, `PetscDAEnsembleComputeAnomalies()`, `PetscDAEnsembleAnalysis()`, `PetscDAEnsembleForecast()`
M*/
PETSC_INTERN PetscErrorCode PetscDACreate_ETKF(PetscDA da)
{
  PetscDA_ETKF *impl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&impl));
  da->data = impl;
  PetscCall(PetscDACreate_Ensemble(da));
  da->ops->setup            = PetscDASetUp_Ensemble;
  da->ops->destroy          = PetscDADestroy_ETKF;
  da->ops->view             = PetscDAView_Ensemble;
  da->ops->setfromoptions   = PetscDASetFromOptions_Ensemble;
  impl->en.analysis         = PetscDAEnsembleAnalysis_ETKF;
  impl->en.forecast         = PetscDAEnsembleForecast_Ensemble;
  impl->en.computemean      = PetscDAEnsembleComputeMean_Default;
  impl->en.computeanomalies = PetscDAEnsembleComputeAnomalies_Default;
  PetscFunctionReturn(PETSC_SUCCESS);
}
