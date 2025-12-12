#include "petscda.h"
#include <petsc/private/daimpl.h>
#include <petscblaslapack.h>
#include <Kokkos_Core.hpp> // Disabled for now

typedef struct {
  PetscDASqrtType sqrt_type;
  Mat             Q;
} PetscDALETKFData;

static PetscFunctionList PetscDALETKFSqrtList           = NULL;
static PetscBool         PetscDALETKFPackageInitialized = PETSC_FALSE;

static PetscErrorCode PetscDALETKFSetSqrt_Cholesky(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscDALETKFSetSqrtType(da, PETSCDA_SQRT_CHOLESKY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetSqrt_Eigen(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscDALETKFSetSqrtType(da, PETSCDA_SQRT_EIGEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFDestroy(PetscDA da)
{
  PetscDALETKFData *impl;

  PetscFunctionBegin;
  if (da->data) {
    impl = (PetscDALETKFData *)da->data;
    PetscCall(MatDestroy(&impl->Q));
    PetscCall(PetscFree(da->data));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASetFromOptions_DALETKF(PetscDA da, PetscOptionItems *PetscOptions)
{
  PetscDALETKFData *impl;
  PetscOptionItems  PetscOptionsObject;
  const char       *defaultType;
  char              typeName[256];
  PetscBool         set             = PETSC_FALSE;
  PetscErrorCode (*setter)(PetscDA) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  impl               = (PetscDALETKFData *)da->data;
  PetscOptionsObject = PetscOptions ? *PetscOptions : NULL;

  defaultType = (impl->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky";
  PetscCall(PetscStrncpy(typeName, defaultType, sizeof(typeName)));
  PetscCall(PetscOptionsFList("-da_letkf_sqrt_type", "Matrix square root factorization", "PetscDALETKFSetSqrtType", PetscDALETKFSqrtList, defaultType, typeName, sizeof(typeName), &set));
  if (set) {
    PetscCall(PetscFunctionListFind(PetscDALETKFSqrtList, typeName, &setter));
    PetscCheck(setter, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDALETKF square-root type \"%s\"", typeName);
    PetscCall((*setter)(da));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFView(PetscDA da, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscDALETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  impl = (PetscDALETKFData *)da->data;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDALETKF Object:\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Square root type: %s\n", (impl->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetLocalization_C(PetscDA da, Mat Q)
{
  PetscDALETKFData *impl = (PetscDALETKFData *)da->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)Q));
  PetscCall(MatDestroy(&impl->Q));
  impl->Q = Q;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Helper to compute observation ensemble Z = H(E)
static PetscErrorCode ComputeObservationEnsemble(PetscDA da, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx, Mat Z)
{
  Vec      col_in, col_out;
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < da->ensemble_size; i++) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, i, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(Z, i, &col_out));
    PetscCall(observation_operator(col_in, col_out, obs_ctx));
    PetscCall(MatDenseRestoreColumnVecWrite(Z, i, &col_out));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, i, &col_in));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFAnalysis(PetscDA da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  PetscDALETKFData  *impl = (PetscDALETKFData *)da->data;
  Mat                X, Z, Y;
  Vec                mean, y_mean;
  PetscInt           m, n, p;
  PetscReal          scale;
  const PetscScalar *X_arr, *Y_arr, *y_obs_arr, *y_mean_arr, *R_arr;
  PetscScalar       *E_arr;
  PetscInt           lda_X, lda_Y, lda_E;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);

  m     = da->ensemble_size;
  n     = da->state_size;
  p     = da->obs_size;
  scale = 1.0 / PetscSqrtReal((PetscReal)(m - 1));

  // 1. Global Computations
  // Alg 6.4 line 1: Compute ensemble mean
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(mean));
  PetscCall(PetscDAComputeEnsembleMean(da, mean));

  // Alg 6.4 line 2: Compute anomalies X = (E - mean) / sqrt(m-1)
  PetscCall(PetscDAComputeAnomalies(da, mean, &X));

  // Alg 6.4 line 3-4: Compute observation ensemble Z = H(E)
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, p, m, NULL, &Z));
  PetscCall(MatSetUp(Z));
  PetscCall(ComputeObservationEnsemble(da, observation_operator, obs_ctx, Z));

  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &y_mean));
  PetscCall(VecSetSizes(y_mean, PETSC_DECIDE, p));
  PetscCall(VecSetFromOptions(y_mean));
  PetscCall(MatGetRowSum(Z, y_mean));
  PetscCall(VecScale(y_mean, 1.0 / m));

  // Alg 6.4 line 5 (partial): Y = (Z - y_mean) / sqrt(m-1)
  PetscCall(MatDuplicate(Z, MAT_COPY_VALUES, &Y));
  {
    Vec col;
    for (PetscInt i = 0; i < m; i++) {
      PetscCall(MatDenseGetColumnVecWrite(Y, i, &col));
      PetscCall(VecAXPY(col, -1.0, y_mean));
      PetscCall(VecScale(col, scale));
      PetscCall(MatDenseRestoreColumnVecWrite(Y, i, &col));
    }
  }

  // Get data pointers for Kokkos
  const PetscScalar *mean_arr;
  PetscCall(MatDenseGetArrayRead(X, &X_arr));
  PetscCall(MatDenseGetArrayRead(Y, &Y_arr));
  PetscCall(VecGetArrayRead(observation, &y_obs_arr));
  PetscCall(VecGetArrayRead(y_mean, &y_mean_arr));
  PetscCall(VecGetArrayRead(mean, &mean_arr));
  PetscCall(VecGetArrayRead(da->obs_error_var, &R_arr));
  PetscCall(MatDenseGetArray(da->ensemble, &E_arr));

  PetscCall(MatDenseGetLDA(X, &lda_X));
  PetscCall(MatDenseGetLDA(Y, &lda_Y));
  PetscCall(MatDenseGetLDA(da->ensemble, &lda_E));

  // Localization matrix Q
  PetscCheck(impl->Q, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Localization matrix Q not set");
  PetscInt Istart, Iend;
  PetscCall(MatGetOwnershipRange(impl->Q, &Istart, &Iend));

  // Allocate workspace for local analysis
  PetscScalar  *S_local, *T_local, *w_local, *delta_local, *work;
  PetscBLASInt *piv;
  PetscBLASInt  info, m_blas, ncols_blas, one = 1;

  // Additional workspace for Eigen
  PetscScalar *work_syev  = NULL;
  PetscReal   *eig_vals   = NULL;
  PetscBLASInt lwork_syev = 0;
#if defined(PETSC_USE_COMPLEX)
  PetscReal *rwork_syev = NULL;
#endif

  PetscCall(PetscBLASIntCast(m, &m_blas));
  PetscCall(PetscBLASIntCast(NUM_OBSERVATIONS_VERTEX, &ncols_blas));

  if (impl->sqrt_type == PETSCDA_SQRT_EIGEN) {
    PetscScalar  dummy_work;
    PetscBLASInt dummy_lwork = -1;
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscMalloc1(PetscMax(1, 3 * m_blas - 2), &rwork_syev));
    LAPACKsyev_("V", "U", &m_blas, NULL, &m_blas, NULL, &dummy_work, &dummy_lwork, rwork_syev, &info);
#else
    LAPACKsyev_("V", "U", &m_blas, NULL, &m_blas, NULL, &dummy_work, &dummy_lwork, &info);
#endif
    lwork_syev = (PetscBLASInt)PetscRealPart(dummy_work);
    PetscCall(PetscMalloc1(lwork_syev, &work_syev));
    PetscCall(PetscMalloc1(m, &eig_vals));
  }

  // Use fixed size for local arrays based on NUM_OBSERVATIONS_VERTEX
  PetscCall(PetscMalloc5(NUM_OBSERVATIONS_VERTEX * m, &S_local, m * m, &T_local, m, &w_local, NUM_OBSERVATIONS_VERTEX, &delta_local, m, &piv));
  PetscCall(PetscMalloc1(m * m, &work)); // Workspace for inversion or T^{-1/2}

  // Access Q data directly assuming SeqAIJ
  for (PetscInt i = Istart; i < Iend; i++) {
    PetscInt           ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    PetscCall(MatGetRow(impl->Q, i, &ncols, &cols, &vals));
    PetscCheck(ncols == NUM_OBSERVATIONS_VERTEX, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "ncols %d != NUM_OBSERVATIONS_VERTEX", (int)ncols);

    PetscBLASInt actual_ncols_blas;
    PetscCall(PetscBLASIntCast(ncols, &actual_ncols_blas));

    // Alg 6.4 line 5: Form local S = R^{-1/2} Y_local
    for (PetscInt k = 0; k < ncols; k++) {
      PetscInt    obs_idx    = cols[k];
      PetscScalar weight     = PetscSqrtScalar(vals[k]); // Tapering
      PetscScalar r_inv_sqrt = 1.0 / PetscSqrtScalar(R_arr[obs_idx]);

      for (PetscInt j = 0; j < m; j++) {
        // Y is column-major in PETSc MatDense
        S_local[j * ncols + k] = Y_arr[j * lda_Y + obs_idx] * weight * r_inv_sqrt;
      }
    }
    // Alg 6.4 line 6: Form local delta = R^{-1/2} (y^o - y_mean)
    for (PetscInt k = 0; k < ncols; k++) {
      PetscInt    obs_idx    = cols[k];
      PetscScalar weight     = PetscSqrtScalar(vals[k]); // Tapering
      PetscScalar r_inv_sqrt = 1.0 / PetscSqrtScalar(R_arr[obs_idx]);
      delta_local[k]         = (y_obs_arr[obs_idx] - y_mean_arr[obs_idx]) * weight * r_inv_sqrt;
    }

    // Alg 6.4 line 7: Compute T = (I + S^T S)^{-1}
    // Note: Here we compute T^{-1} first (which is (m-1) * P^a_ens in state space theory, but strictly T in Alg 6.4 is the inverse)
    // Actually Alg 6.4 says T = (I + S^T S)^{-1}.
    // We add inflation here as (1/rho)*I.
    PetscScalar diag_val = 1.0 / da->inflation;
    for (PetscInt j = 0; j < m * m; j++) T_local[j] = 0.0;
    for (PetscInt j = 0; j < m; j++) T_local[j * m + j] = diag_val;

    // T += S^T S
    // Refill S_local correctly for column-major (ncols rows, m cols) for BLAS
    // Note: We refill because previous loop filled it row-major w.r.t (m x ncols) logic but stored linearly?
    // Wait, earlier loop: S_local[j * ncols + k].
    // For BLASsyrk with "T" (C = alpha A^T A), A should be k x n. Here C is m x m. So A is (ncols x m).
    // If we use TRANS='T', we compute A^T * A. If S_local is (m x ncols) in storage?
    // Re-reading loop: j from 0 to m. k from 0 to ncols.
    // S_local[j * ncols + k]. This looks like row-major for (m x ncols) matrix?
    // BLAS expects column-major.
    // So S_local as stored is actually Transpose(S_local_col_major) if we consider it m x ncols.
    // Let's stick to the existing logic which seems to refill it:
    for (PetscInt k = 0; k < ncols; k++) {
      PetscInt    obs_idx    = cols[k];
      PetscScalar weight     = PetscSqrtScalar(vals[k]);
      PetscScalar r_inv_sqrt = 1.0 / PetscSqrtScalar(R_arr[obs_idx]);
      // Here: S_local[k + j * ncols] -> This is column major for (ncols x m) matrix OR row major for (m x ncols) transposed?
      // Let's assume the previous logic was correct for Cholesky.
      for (PetscInt j = 0; j < m; j++) { S_local[k + j * ncols] = Y_arr[j * lda_Y + obs_idx] * weight * r_inv_sqrt; }
    }
    // With S_local[k + j*ncols]:
    // j=0..m-1 (cols of result?), k=0..ncols-1 (inner dim)
    // This storage is column-major for a matrix of size (ncols x m).
    // We want T = S^T * S.
    // If S_local represents S (obs x m), then we want S^T S.
    // The matrix filled is size (ncols x m). Let's call it A.
    // We want T = A^T * A.
    // BLASsyrk("U", "T", n=m, k=ncols, A=S_local, lda=ncols, C=T_local, ldc=m)
    // This computes C = A^T * A. Correct.

    PetscScalar alpha = 1.0, beta = 1.0;
    PetscCallBLAS("BLASsyrk", BLASsyrk_("U", "T", &m_blas, &actual_ncols_blas, &alpha, S_local, &actual_ncols_blas, &beta, T_local, &m_blas));

    if (impl->sqrt_type == PETSCDA_SQRT_CHOLESKY) {
      // Alg 6.4 line 7 (cont): Invert T using Cholesky to get T proper
      PetscCallBLAS("LAPACKpotrf", LAPACKpotrf_("U", &m_blas, T_local, &m_blas, &info));
      if (info != 0) {
        for (PetscInt j = 0; j < m; j++) w_local[j] = 0.0;
        for (PetscInt j = 0; j < m * m; j++) work[j] = 0.0;
        for (PetscInt j = 0; j < m; j++) work[j * m + j] = 1.0;
      } else {
        PetscCallBLAS("LAPACKpotri", LAPACKpotri_("U", &m_blas, T_local, &m_blas, &info));
        // POTRI only computes upper triangle. Fill lower.
        for (PetscInt j = 0; j < m; j++) {
          for (PetscInt k = j + 1; k < m; k++) { T_local[k + j * m] = T_local[j + k * m]; }
        }

        // Alg 6.4 line 8: Compute w = T S^T delta
        PetscScalar *temp_w = work; // Reuse work
        alpha               = 1.0;
        beta                = 0.0;
        // S_local is (ncols x m). S^T is (m x ncols).
        // We want S^T * delta.
        // GEMV T: y = alpha * A^T * x + beta * y.
        // A is S_local (ncols x m). A^T is (m x ncols).
        // Wait, GEMV 'T' on A(ncols, m) gives result size m.
        // Correct.
        PetscCallBLAS("BLASgemv", BLASgemv_("T", &actual_ncols_blas, &m_blas, &alpha, S_local, &actual_ncols_blas, delta_local, &one, &beta, temp_w, &one));

        // w = T * temp_w
        PetscCallBLAS("BLASgemv", BLASgemv_("N", &m_blas, &m_blas, &alpha, T_local, &m_blas, temp_w, &one, &beta, w_local, &one));

        // Alg 6.4 line 9 (part 2): Compute T^{1/2} (actually need T^{1/2} for update)
        // We need U where T = U U^T (or L L^T).
        // But wait, the update is X * (w*1^T + sqrt(m-1)*T^{1/2}*U_arb)
        // In Cholesky, T^{1/2} corresponds to the upper triangular factor U from T = U^T U?
        // Or is it L from T = L L^T?
        // T is symmetric positive definite.
        // If we want T^{1/2} such that T^{1/2} (T^{1/2})^T = T.
        // Re-do inversion of (I + S^T S) to get T^{-1} (inverse of T in code), then Cholesky?
        // No, T_local currently holds T = (I + S^T S)^{-1}.
        // We want T^{1/2}.
        // Cholesky of T: T = U^T U. Then U is a square root.
        PetscCallBLAS("LAPACKpotrf", LAPACKpotrf_("U", &m_blas, T_local, &m_blas, &info));
        // T_local now contains U (upper triangular).
        // We need to zero out the lower triangle for correctness if we use it as a full matrix.
        for (PetscInt j = 0; j < m; j++) {
           for (PetscInt k = j + 1; k < m; k++) { T_local[k + j * m] = 0.0; } // Zero lower (cols k > j, so row k > row j... wait)
           // Storage is col-major. T_local[k + j*m] is row k, col j.
           // Upper triangle means k <= j.
           // We want to zero k > j.
           for (PetscInt k = j + 1; k < m; k++) { T_local[k + j * m] = 0.0; } // This zeros lower part of column j
        }
        // But wait, we need to handle the previous block logic which was doing T^{-1/2} = L^{-T}.
        // Let's stick to what was there but rename.
        // Previous logic:
        // Re-do T construction for L
        // ...
        // Compute T^{-1/2} = L^{-T}.
        // Alg 6.4 line 9 calls for T^{1/2}.
        // If T_local holds T, we want T^{1/2}.
        // If we re-compute (I+S^T S) and Cholesky it -> L L^T. Then T = (L L^T)^{-1} = L^{-T} L^{-1}.
        // So T^{1/2} could be L^{-T}.
        // Let's restore the logic but fix comments.

        for (PetscInt j = 0; j < m * m; j++) T_local[j] = 0.0;
        for (PetscInt j = 0; j < m; j++) T_local[j * m + j] = diag_val;
        PetscCallBLAS("BLASsyrk", BLASsyrk_("L", "T", &m_blas, &actual_ncols_blas, &alpha, S_local, &actual_ncols_blas, &beta, T_local, &m_blas));
        PetscCallBLAS("LAPACKpotrf", LAPACKpotrf_("L", &m_blas, T_local, &m_blas, &info));

        // Compute T^{1/2} = L^{-T}.
        for (PetscInt j = 0; j < m; j++) {
          for (PetscInt k = j; k < m; k++) { // Lower
            work[k + j * m] = T_local[k + j * m];
            work[j + k * m] = 0.0; // Zero upper
          }
        }
        PetscCallBLAS("LAPACKtrtri", LAPACKtrtri_("L", "N", &m_blas, work, &m_blas, &info));
      }
    } else {
      // PETSCDA_SQRT_EIGEN
      // T is in T_local (upper triangle filled) - this is (I + S^T S)
      // Eigendecomposition: (I + S^T S) = V D V^T
      // Then T = (I + S^T S)^{-1} = V D^{-1} V^T
      // And T^{1/2} = V D^{-1/2} V^T
      // LAPACKsyev overwrites T_local with eigenvectors V, and eig_vals with eigenvalues D.
#if defined(PETSC_USE_COMPLEX)
      LAPACKsyev_("V", "U", &m_blas, T_local, &m_blas, eig_vals, work_syev, &lwork_syev, rwork_syev, &info);
#else
      LAPACKsyev_("V", "U", &m_blas, T_local, &m_blas, eig_vals, work_syev, &lwork_syev, &info);
#endif
      if (info != 0) {
        // Fallback
        for (PetscInt j = 0; j < m; j++) w_local[j] = 0.0;
        for (PetscInt j = 0; j < m * m; j++) work[j] = 0.0;
        for (PetscInt j = 0; j < m; j++) work[j * m + j] = 1.0;
      } else {
        // Alg 6.4 line 8: Compute w = T S^T delta
        // w = V * D^{-1} * V^T * S^T * delta
        // First, S^T * delta
        PetscScalar *temp_w = work; // Reuse work buffer
        alpha               = 1.0;
        beta                = 0.0;
        PetscCallBLAS("BLASgemv", BLASgemv_("T", &actual_ncols_blas, &m_blas, &alpha, S_local, &actual_ncols_blas, delta_local, &one, &beta, temp_w, &one));

        // Project onto V: y = V^T * (S^T * delta)
        // V is in T_local.
        PetscScalar *y_vec = w_local; // Use w_local as temp
        PetscCallBLAS("BLASgemv", BLASgemv_("T", &m_blas, &m_blas, &alpha, T_local, &m_blas, temp_w, &one, &beta, y_vec, &one));

        // Scale by D^{-1}: y = D^{-1} * y
        for (PetscInt j = 0; j < m; j++) { y_vec[j] /= eig_vals[j]; }

        // Rotate back: w = V * y
        PetscCallBLAS("BLASgemv", BLASgemv_("N", &m_blas, &m_blas, &alpha, T_local, &m_blas, y_vec, &one, &beta, w_local, &one)); // Output to w_local, using y_vec(w_local) as input?
        // WARN: BLASgemv cannot alias x and y.
        // We need a temp buffer.
        // temp_w is available. Copy y_vec to temp_w.
        for (PetscInt j = 0; j < m; j++) temp_w[j] = y_vec[j];
        PetscCallBLAS("BLASgemv", BLASgemv_("N", &m_blas, &m_blas, &alpha, T_local, &m_blas, temp_w, &one, &beta, w_local, &one));

        // Alg 6.4 line 9 (part 2): Compute T^{1/2} = V * D^{-1/2} * V^T
        // Store result in 'work' (m x m)
        // First compute W = V * D^{-1/2}
        // We can do this in place in T_local? No, we need V for the second multiply.
        // Let's use 'work' to store V * D^{-1/2}.
        for (PetscInt j = 0; j < m; j++) { // Columns of V
          PetscScalar scale = 1.0 / PetscSqrtScalar(eig_vals[j]);
          for (PetscInt k = 0; k < m; k++) { // Rows of V
            work[k + j * m] = T_local[k + j * m] * scale;
          }
        }
        // Now work = V * D^{-1/2}.
        // We need work * V^T.
        // T^{1/2} = work * T_local^T
        // gemm("N", "T", work, T_local)
        // Output to... we need another buffer?
        // We can overwrite T_local if we are done with V, but we want result in 'work' for the UpdateEnsemble step.
        // The update step expects T^{1/2} in 'work'.
        // So we need to compute C = A * B^T and put in 'work'.
        // A = work (current content), B = T_local (V).
        // We cannot overwrite A during GEMM.
        // We need a temp matrix buffer.
        // We can use S_local? It is ncols * m. If ncols >= m, yes.
        // NUM_OBSERVATIONS_VERTEX is 40. m (ensemble size) can be anything.
        // Safer to allocate a temp matrix or re-use something else.
        // Let's allocate T_sqrt_temp.
        // Or, we can compute T^{1/2} * I.
        // Actually, UpdateEnsemble uses 'work' as T^{1/2} (or L^{-T}).
        // Let's allocate a small temp buffer for the result, then copy back to work.
        PetscScalar *T_sqrt_temp;
        PetscCall(PetscMalloc1(m * m, &T_sqrt_temp));
        PetscCallBLAS("BLASgemm", BLASgemm_("N", "T", &m_blas, &m_blas, &m_blas, &alpha, work, &m_blas, T_local, &m_blas, &beta, T_sqrt_temp, &m_blas));
        // Copy back to work
        PetscCall(PetscMemcpy(work, T_sqrt_temp, m * m * sizeof(PetscScalar)));
        PetscCall(PetscFree(T_sqrt_temp));
     }
   }

   // Alg 6.4 line 9: Form G = w * 1' + sqrt(m-1) * T^{1/2} * U
   PetscScalar sqrt_m_minus_1 = PetscSqrtScalar(m - 1);

    // Alg 6.4 line 9: Update ensemble E = mean + X * G
    for (PetscInt ens_j = 0; ens_j < m; ens_j++) {
      PetscScalar val = 0.0;
      // X_i * G[:, ens_j]
      for (PetscInt k = 0; k < m; k++) {
        PetscScalar G_kj = w_local[k] + sqrt_m_minus_1 * work[ens_j + k * m];
        val += X_arr[i + k * lda_X] * G_kj;
      }
      E_arr[i + ens_j * lda_E] = mean_arr[i] + val;
    }

    PetscCall(MatRestoreRow(impl->Q, i, &ncols, &cols, &vals));
  }

  PetscCall(PetscFree5(S_local, T_local, w_local, delta_local, piv));
  PetscCall(PetscFree(work));
  if (impl->sqrt_type == PETSCDA_SQRT_EIGEN) {
    PetscCall(PetscFree(work_syev));
    PetscCall(PetscFree(eig_vals));
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscFree(rwork_syev));
#endif
  }

  // Restore arrays
  PetscCall(MatDenseRestoreArrayRead(X, &X_arr));
  PetscCall(MatDenseRestoreArrayRead(Y, &Y_arr));
  PetscCall(VecRestoreArrayRead(observation, &y_obs_arr));
  PetscCall(VecRestoreArrayRead(y_mean, &y_mean_arr));
  PetscCall(VecRestoreArrayRead(mean, &mean_arr));
  PetscCall(VecRestoreArrayRead(da->obs_error_var, &R_arr));
  PetscCall(MatDenseRestoreArray(da->ensemble, &E_arr));

  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Z));
  PetscCall(MatDestroy(&Y));
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&y_mean));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFApplyModel(PetscDA da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
{
  Vec      col_in, col_out, temp;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &temp));
  PetscCall(VecSetSizes(temp, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(temp));

  for (i = 0; i < da->ensemble_size; i++) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, i, &col_in));
    PetscCall(model(col_in, temp, model_ctx));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, i, &col_in));

    PetscCall(MatDenseGetColumnVecWrite(da->ensemble, i, &col_out));
    PetscCall(VecCopy(temp, col_out));
    PetscCall(MatDenseRestoreColumnVecWrite(da->ensemble, i, &col_out));
  }

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDALETKFInitialize(PetscDA da)
{
  PetscDALETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscCall(PetscNew(&impl));
  impl->Q         = NULL;
  impl->sqrt_type = PETSCDA_SQRT_EIGEN;

  da->data                  = impl;
  da->ops->analysis         = PetscDALETKFAnalysis;
  da->ops->applymodel       = PetscDALETKFApplyModel;
  da->ops->computemean      = NULL;
  da->ops->computeanomalies = NULL;
  da->ops->destroy          = PetscDALETKFDestroy;
  da->ops->view             = PetscDALETKFView;
  da->ops->setfromoptions   = PetscDASetFromOptions_DALETKF;

  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalization_C", PetscDALETKFSetLocalization_C));

  PetscCall(PetscDALETKFInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDALETKFSetSqrtType(PetscDA da, PetscDASqrtType type)
{
  PetscDALETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA data structure not initialized");
  PetscCheck(type == PETSCDA_SQRT_CHOLESKY || type == PETSCDA_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDA square-root type %d", (int)type);

  impl            = (PetscDALETKFData *)da->data;
  impl->sqrt_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDALETKFGetSqrtType(PetscDA da, PetscDASqrtType *type)
{
  PetscDALETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA data structure not initialized");

  impl  = (PetscDALETKFData *)da->data;
  *type = impl->sqrt_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDALETKFInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDALETKFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDALETKFPackageInitialized = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&PetscDALETKFSqrtList, "cholesky", PetscDALETKFSetSqrt_Cholesky));
  PetscCall(PetscFunctionListAdd(&PetscDALETKFSqrtList, "eigen", PetscDALETKFSetSqrt_Eigen));
  PetscCall(PetscRegisterFinalize(PetscDALETKFFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDALETKFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscDALETKFPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&PetscDALETKFSqrtList));
  PetscFunctionReturn(PETSC_SUCCESS);
}
