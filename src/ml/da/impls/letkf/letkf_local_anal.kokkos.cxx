#include "letkf_impl.h"
#include <petscblaslapack.h>
#include <Kokkos_Core.hpp>
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBatched_SVD_Serial_Impl.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
  #include <cusolverDn.h>
  #include <cuda_runtime.h>
#endif

/* ========================================================================== */
/*                    Batched Eigendecomposition for LETKF                    */
/* ========================================================================== */

/*
  BatchedEigenSolve_Host - Compute eigendecomposition for a batch of symmetric matrices (CPU version)

  Input Parameters:
+ T_batch      - batch of symmetric matrices (n_batch x n_size x n_size)
. n_batch      - number of matrices in the batch
- n_size       - size of each matrix (m x m)

  Output Parameters:
+ Lambda_batch - eigenvalues for each matrix (n_batch x n_size)
- V_batch      - eigenvectors for each matrix (n_batch x n_size x n_size)

  Notes:
  Uses LAPACK's syev routine to compute eigendecomposition sequentially on host.
*/
#if !defined(KOKKOS_ENABLE_CUDA)
static PetscErrorCode BatchedEigenSolve_Host(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size)
{
  PetscFunctionBegin;

  /* Create host mirrors */
  auto T_host      = Kokkos::create_mirror_view(T_batch);
  auto Lambda_host = Kokkos::create_mirror_view(Lambda_batch);
  auto V_host      = Kokkos::create_mirror_view(V_batch);

  /* Copy T to host */
  Kokkos::deep_copy(T_host, T_batch);

  /* Allocate contiguous buffers for LAPACK */
  PetscScalar *v_contiguous, *lambda_contiguous;
  PetscCall(PetscMalloc2(n_size * n_size, &v_contiguous, n_size, &lambda_contiguous));

  /* Process each matrix sequentially on host using LAPACK */
  for (PetscInt i = 0; i < n_batch; i++) {
    PetscBLASInt n, lda, lwork, info;
    PetscScalar *work;

    PetscCall(PetscBLASIntCast(n_size, &n));
    lda = n;

    /* Copy T_host(i, :, :) to v_contiguous (column-major) */
    for (PetscInt j = 0; j < n_size; j++) {
      for (PetscInt k = 0; k < n_size; k++) { v_contiguous[k + j * n_size] = T_host(i, k, j); }
    }

    /* Query workspace size */
    lwork = -1;
    PetscCall(PetscMalloc1(1, &work));
    LAPACKsyev_("V", "U", &n, v_contiguous, &lda, lambda_contiguous, work, &lwork, &info);
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK workspace query failed: info=%" PetscBLASInt_FMT, info);

    /* Allocate workspace */
    lwork = (PetscBLASInt)PetscRealPart(work[0]);
    PetscCall(PetscFree(work));
    PetscCall(PetscMalloc1(lwork, &work));

    /* Compute eigendecomposition: T = V * Lambda * V^T */
    LAPACKsyev_("V", "U", &n, v_contiguous, &lda, lambda_contiguous, work, &lwork, &info);
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK eigendecomposition failed for matrix %" PetscInt_FMT ": info=%" PetscBLASInt_FMT, i, info);

    PetscCall(PetscFree(work));

    /* Copy results back to host views */
    for (PetscInt j = 0; j < n_size; j++) {
      Lambda_host(i, j) = lambda_contiguous[j];
      for (PetscInt k = 0; k < n_size; k++) { V_host(i, k, j) = v_contiguous[k + j * n_size]; }
    }
  }

  PetscCall(PetscFree2(v_contiguous, lambda_contiguous));

  /* Copy results back to device */
  Kokkos::deep_copy(Lambda_batch, Lambda_host);
  Kokkos::deep_copy(V_batch, V_host);

  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  BatchedEigenSolve_CUDA - Compute eigendecomposition for a batch of symmetric matrices (GPU version)

  Input Parameters:
+ T_batch      - batch of symmetric matrices (n_batch x n_size x n_size)
. n_batch      - number of matrices in the batch
- n_size       - size of each matrix (m x m)

  Output Parameters:
+ Lambda_batch - eigenvalues for each matrix (n_batch x n_size)
- V_batch      - eigenvectors for each matrix (n_batch x n_size x n_size)

  Notes:
  Uses cuSOLVER's syevd routine to compute eigendecomposition on GPU.
*/
#if defined(KOKKOS_ENABLE_CUDA)
static PetscErrorCode BatchedEigenSolve_CUDA(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size)
{
  cusolverDnHandle_t cusolverH;
  cusolverStatus_t   cusolver_status;
  int                lwork = 0;
  PetscScalar       *d_work;
  int               *d_info;

  PetscFunctionBegin;

  /* Create cuSOLVER handle */
  cusolver_status = cusolverDnCreate(&cusolverH);
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreate failed");

  /* Query workspace size */
  #if defined(PETSC_USE_REAL_SINGLE)
  cusolver_status = cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, NULL, n_size, NULL, &lwork);
  #else
  cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, NULL, n_size, NULL, &lwork);
  #endif
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDn*syevd_bufferSize failed");

  /* Allocate workspace on device */
  cudaMalloc(&d_work, sizeof(PetscScalar) * lwork);
  cudaMalloc(&d_info, sizeof(int));

  /* Process each matrix in the batch */
  for (PetscInt i = 0; i < n_batch; i++) {
    /* Get subviews for this matrix */
    auto T_i      = Kokkos::subview(T_batch, i, Kokkos::ALL(), Kokkos::ALL());
    auto Lambda_i = Kokkos::subview(Lambda_batch, i, Kokkos::ALL());
    auto V_i      = Kokkos::subview(V_batch, i, Kokkos::ALL(), Kokkos::ALL());

    /* Copy T to V (cuSOLVER overwrites input) */
    Kokkos::deep_copy(V_i, T_i);

    /* Solve eigendecomposition on device */
  #if defined(PETSC_USE_REAL_SINGLE)
    cusolver_status = cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, V_i.data(), n_size, Lambda_i.data(), d_work, lwork, d_info);
  #else
    cusolver_status = cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, V_i.data(), n_size, Lambda_i.data(), d_work, lwork, d_info);
  #endif
    PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDn*syevd failed for matrix %" PetscInt_FMT, i);

    /* Check for errors (copy d_info to host) */
    int h_info;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    PetscCheck(h_info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuSOLVER eigendecomposition failed for matrix %" PetscInt_FMT ": info=%d", i, h_info);
  }

  /* Cleanup */
  cudaFree(d_work);
  cudaFree(d_info);
  cusolverDnDestroy(cusolverH);

  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  BatchedEigenSolve - Compute eigendecomposition for a batch of symmetric matrices

  Input Parameters:
+ T_batch      - batch of symmetric matrices (n_batch x n_size x n_size)
. n_batch      - number of matrices in the batch
- n_size       - size of each matrix (m x m)

  Output Parameters:
+ Lambda_batch - eigenvalues for each matrix (n_batch x n_size)
- V_batch      - eigenvectors for each matrix (n_batch x n_size x n_size)

  Notes:
  Dispatcher function that calls the appropriate backend (CPU or GPU).
*/
static PetscErrorCode BatchedEigenSolve(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size)
{
  PetscFunctionBegin;

#if defined(KOKKOS_ENABLE_CUDA)
  PetscCall(BatchedEigenSolve_CUDA(T_batch, Lambda_batch, V_batch, n_batch, n_size));
#else
  PetscCall(BatchedEigenSolve_Host(T_batch, Lambda_batch, V_batch, n_batch, n_size));
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                    LETKF Local Analysis (Main Function)                    */
/* ========================================================================== */

/*
  PetscDALETKFLocalAnalysis_GPU - Performs local LETKF analysis for all grid points (Kokkos version)

  Input Parameters:
+ da             - the PetscDA context
. impl           - LETKF implementation data
. m              - ensemble size
. n_vertices     - number of grid points
. X              - global anomaly matrix (state_size x m)
. observation    - observation vector
. Z_global       - global observation ensemble (obs_size x m)
. y_mean_global  - global observation mean
- r_inv_sqrt_global - global R^{-1/2}

  Output:
. da->ensemble - updated with analysis ensemble

  Notes:
  This function performs the local analysis loop for LETKF, processing each grid point
  independently using its local observations defined by the localization matrix Q.
  This is the CPU version that does not use Kokkos acceleration.

  All local analysis workspace objects (Z_local, S_local, T_sqrt_local, G_local, y_local,
  y_mean_local, delta_scaled_local, r_inv_sqrt_local, w_local, s_transpose_delta) are
  created with PETSC_COMM_SELF because the analysis at each vertex is serial and independent.
*/
PetscErrorCode PetscDALETKFLocalAnalysis_GPU(PetscDA da, PetscDALETKFData *impl, PetscInt m, PetscInt n_vertices, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global)
{
  PetscInt  ndof;
  PetscReal sqrt_m_minus_1, scale, inflation_inv;

  PetscFunctionBegin;
  ndof           = da->ndof;
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  inflation_inv  = 1.0 / da->inflation; /* (1/ρ) for T matrix: T = (1/ρ)I + S^T*S */

  /* ===================================================================== */
  /* Step 2.1.1: Create batched workspace for ALL grid points            */
  /* ===================================================================== */
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_3d    = Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, exec_space>;
  using view_2d    = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>;

  /* Batched workspace for ALL grid points (device memory) */
  view_3d Z_batch("Z_batch", n_vertices, impl->p_local, m);                // (n_vertices, p_local, m)
  view_3d S_batch("S_batch", n_vertices, impl->p_local, m);                // (n_vertices, p_local, m)
  view_3d T_batch("T_batch", n_vertices, m, m);                            // (n_vertices, m, m)
  view_3d V_batch("V_batch", n_vertices, m, m);                            // (n_vertices, m, m)
  view_2d Lambda_batch("Lambda_batch", n_vertices, m);                     // (n_vertices, m)
  view_3d T_sqrt_batch("T_sqrt_batch", n_vertices, m, m);                  // (n_vertices, m, m)
  view_3d G_batch("G_batch", n_vertices, m, m);                            // (n_vertices, m, m)
  view_2d w_batch("w_batch", n_vertices, m);                               // (n_vertices, m)
  view_2d delta_batch("delta_batch", n_vertices, impl->p_local);           // (n_vertices, p_local)
  view_2d y_batch("y_batch", n_vertices, impl->p_local);                   // (n_vertices, p_local)
  view_2d y_mean_batch("y_mean_batch", n_vertices, impl->p_local);         // (n_vertices, p_local)
  view_2d r_inv_sqrt_batch("r_inv_sqrt_batch", n_vertices, impl->p_local); // (n_vertices, p_local)

  /* ===================================================================== */
  /* Step 2.1.2a: Pre-extract Q matrix CSR data for device access        */
  /* ===================================================================== */
  /* Get direct access to Q's CSR arrays (zero-copy) */
  const PetscInt *Q_i, *Q_j;
  PetscScalar    *Q_a;
  PetscMemType    Q_memtype;

  PetscCall(MatSeqAIJGetCSRAndMemType(impl->Q, &Q_i, &Q_j, &Q_a, &Q_memtype));

  /* Create unmanaged Kokkos views wrapping the CSR arrays */
  using view_1d_int_const    = Kokkos::View<const PetscInt *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_scalar_const = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  view_1d_int_const    Q_i_view(Q_i, n_vertices + 1);
  view_1d_int_const    Q_j_view(Q_j, n_vertices * Q_NUM_LOCAL_OBSERVATIONS_MAX);
  view_1d_scalar_const Q_a_view(Q_a, n_vertices * Q_NUM_LOCAL_OBSERVATIONS_MAX);

  /* Get global observation data arrays */
  const PetscScalar *z_global_array, *y_global_array, *y_mean_global_array, *r_inv_sqrt_global_array;
  PetscInt           lda_z_global;

  PetscCall(MatDenseGetArrayRead(Z_global, &z_global_array));
  PetscCall(VecGetArrayRead(observation, &y_global_array));
  PetscCall(VecGetArrayRead(y_mean_global, &y_mean_global_array));
  PetscCall(VecGetArrayRead(r_inv_sqrt_global, &r_inv_sqrt_global_array));
  PetscCall(MatDenseGetLDA(Z_global, &lda_z_global));

  /* Create unmanaged Kokkos views for global observation data */
  using view_2d_unmanaged = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_unmanaged = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  view_2d_unmanaged Z_global_view(z_global_array, lda_z_global, m);
  view_1d_unmanaged y_global_view(y_global_array, lda_z_global);
  view_1d_unmanaged y_mean_global_view(y_mean_global_array, lda_z_global);
  view_1d_unmanaged r_inv_sqrt_global_view(r_inv_sqrt_global_array, lda_z_global);

  /* ===================================================================== */
  /* Step 2.1.2: Parallelize observation extraction for all grid points  */
  /* ===================================================================== */
  Kokkos::parallel_for(
    "ExtractAllLocalObservations", Kokkos::RangePolicy<exec_space>(0, n_vertices), KOKKOS_LAMBDA(const int i_grid_point) {
      /* Get Q row for this grid point using CSR format */
      PetscInt row_start = Q_i_view(i_grid_point);
      PetscInt row_end   = Q_i_view(i_grid_point + 1);
      PetscInt ncols     = row_end - row_start;

      /* Extract observations for this grid point */
      for (PetscInt k = 0; k < ncols; k++) {
        PetscInt    obs_idx = Q_j_view(row_start + k);
        PetscScalar weight  = Q_a_view(row_start + k);

        /* Extract observation vectors */
        y_batch(i_grid_point, k)          = y_global_view(obs_idx);
        y_mean_batch(i_grid_point, k)     = y_mean_global_view(obs_idx);
        r_inv_sqrt_batch(i_grid_point, k) = r_inv_sqrt_global_view(obs_idx) * Kokkos::sqrt(weight);

        /* Extract Z matrix columns for this observation */
        for (int j = 0; j < m; j++) { Z_batch(i_grid_point, k, j) = Z_global_view(obs_idx, j); }
      }
    });
  Kokkos::fence();

  /* ===================================================================== */
  /* Step 2.1.3: Parallelize normalized innovation computation            */
  /* ===================================================================== */
  /* Compute S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1) for all grid points */
  /* Also compute delta = R^{-1/2}(y - y_mean) for all grid points */
  Kokkos::parallel_for(
    "ComputeAllNormalizedInnovations", Kokkos::MDRangePolicy<Kokkos::Rank<3>, exec_space>({0, 0, 0}, {n_vertices, impl->p_local, m}), KOKKOS_LAMBDA(const int i, const int k, const int j) {
      /* S_batch(i, k, j) = (Z_batch(i, k, j) - y_mean_batch(i, k)) * scale * r_inv_sqrt_batch(i, k) */
      S_batch(i, k, j) = (Z_batch(i, k, j) - y_mean_batch(i, k)) * scale * r_inv_sqrt_batch(i, k);
    });

  /* Compute delta = R^{-1/2}(y - y_mean) for all grid points */
  Kokkos::parallel_for(
    "ComputeAllDeltas", Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>({0, 0}, {n_vertices, impl->p_local}), KOKKOS_LAMBDA(const int i, const int k) { delta_batch(i, k) = (y_batch(i, k) - y_mean_batch(i, k)) * r_inv_sqrt_batch(i, k); });
  Kokkos::fence();

  /* ===================================================================== */
  /* Step 2.1.4: Parallelize T matrix formation (T = (1/ρ)I + S^T * S)  */
  /* ===================================================================== */
  /* Compute T_i = (1/ρ)I + S_i^T * S_i for all grid points */
  /* This is the Gram matrix C = S^T * S plus scaled identity */
  Kokkos::parallel_for(
    "ComputeAllTMatrices", Kokkos::RangePolicy<exec_space>(0, n_vertices), KOKKOS_LAMBDA(const int i) {
      /* For grid point i, compute T_i = (1/ρ)I + S_i^T * S_i */
      /* T_i(j,k) = (1/ρ)*delta_{jk} + sum_p S_i(p,j) * S_i(p,k) */
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < m; k++) {
          PetscScalar sum = (j == k) ? inflation_inv : 0.0; /* Add scaled identity: (1/ρ)*I_{jk} */
          for (int p = 0; p < impl->p_local; p++) { sum += S_batch(i, p, j) * S_batch(i, p, k); }
          T_batch(i, j, k) = sum;
        }
      }
    });
  Kokkos::fence();

  /* ===================================================================== */
  /* Step 3.1.1: Batched eigendecomposition for all grid points          */
  /* ===================================================================== */
  /* Compute T_i = V_i * Lambda_i * V_i^T for all grid points */
  PetscCall(BatchedEigenSolve(T_batch, Lambda_batch, V_batch, n_vertices, m));

  /* ===================================================================== */
  /* Step 3.1.2: Batched T^{-1} application for all grid points          */
  /* ===================================================================== */
  /* Compute w_i = T_i^{-1} * (S_i^T * delta_i) for all grid points */
  /* Using efficient factorization: w_i = V_i * (Lambda_i^{-1} * (V_i^T * S_i^T * delta_i)) */
  Kokkos::parallel_for(
    "BatchedApplyTInverse", Kokkos::RangePolicy<exec_space>(0, n_vertices), KOKKOS_LAMBDA(const int i) {
      /* Step 1: Compute S_i^T * delta_i -> temp1 */
      PetscScalar temp1[200]; /* Assuming m <= 200 */
      for (int j = 0; j < m; j++) {
        temp1[j] = 0.0;
        for (int k = 0; k < impl->p_local; k++) { temp1[j] += S_batch(i, k, j) * delta_batch(i, k); }
      }

      /* Step 2: Compute V_i^T * temp1 -> temp2 */
      PetscScalar temp2[200];
      for (int j = 0; j < m; j++) {
        temp2[j] = 0.0;
        for (int k = 0; k < m; k++) { temp2[j] += V_batch(i, k, j) * temp1[k]; }
      }

      /* Step 3: Scale by Lambda_i^{-1} */
      for (int j = 0; j < m; j++) { temp2[j] /= Lambda_batch(i, j); }

      /* Step 4: Compute V_i * temp2 -> w_i */
      for (int j = 0; j < m; j++) {
        w_batch(i, j) = 0.0;
        for (int k = 0; k < m; k++) { w_batch(i, j) += V_batch(i, j, k) * temp2[k]; }
      }
    });
  Kokkos::fence();

  /* ===================================================================== */
  /* Step 3.1.3: Batched T^{-1/2} application for all grid points        */
  /* ===================================================================== */
  /* Compute T_sqrt_i = V_i * Lambda_i^{-1/2} * V_i^T for all grid points */
  Kokkos::parallel_for(
    "BatchedApplySqrtTInverse", Kokkos::RangePolicy<exec_space>(0, n_vertices), KOKKOS_LAMBDA(const int i) {
      /* Step 1: Compute V_i * Lambda_i^{-1/2} -> V_scaled */
      PetscScalar V_scaled[200][200]; /* Assuming m <= 200 */
      for (int j = 0; j < m; j++) {
        PetscScalar scale = 1.0 / Kokkos::sqrt(Lambda_batch(i, j));
        for (int k = 0; k < m; k++) { V_scaled[k][j] = V_batch(i, k, j) * scale; }
      }

      /* Step 2: Compute (V_i * Lambda_i^{-1/2}) * V_i^T -> T_sqrt_i */
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < m; k++) {
          PetscScalar sum = 0.0;
          for (int p = 0; p < m; p++) { sum += V_scaled[j][p] * V_batch(i, k, p); }
          T_sqrt_batch(i, j, k) = sum;
        }
      }
    });
  Kokkos::fence();

  /* ===================================================================== */
  /* Step 3.1.4: Parallelize G matrix formation for all grid points      */
  /* ===================================================================== */
  /* Compute G_i = w_i * 1^T + sqrt(m-1) * T_sqrt_i for all grid points */
  Kokkos::parallel_for(
    "BatchedFormGMatrices", Kokkos::RangePolicy<exec_space>(0, n_vertices), KOKKOS_LAMBDA(const int i) {
      /* G_i = sqrt(m-1) * T_sqrt_i + w_i * 1^T */
      /* w_i * 1^T means: row j gets w_i(j) added to all columns */
      /* In column-major storage: G(j,k) is at position [j + k*m] */
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < m; k++) {
          /* G_i(j,k) = sqrt(m-1) * T_sqrt_i(j,k) + w_i(j) */
          /* Note: w_i(j) is added to row j (all columns k in row j get w_i(j)) */
          G_batch(i, j, k) = sqrt_m_minus_1 * T_sqrt_batch(i, j, k) + w_batch(i, j);
        }
      }
    });
  Kokkos::fence();

  /* ===================================================================== */
  /* Step 3.1.5: Parallelize ensemble update for all grid points         */
  /* ===================================================================== */
  /* Compute E[i,:] = mean[i] + X[i,:] * G_i for all grid points */
  /* Get access to global X matrix and mean vector */
  const PetscScalar *x_array, *mean_array;
  PetscScalar       *e_array;
  PetscInt           lda_x, lda_e;
  PetscCall(MatDenseGetArrayRead(X, &x_array));
  PetscCall(VecGetArrayRead(impl->mean, &mean_array));
  PetscCall(MatDenseGetArrayWrite(da->ensemble, &e_array));
  PetscCall(MatDenseGetLDA(X, &lda_x));
  PetscCall(MatDenseGetLDA(da->ensemble, &lda_e));

  /* Create unmanaged Kokkos views for global data */
  using view_2d_unmanaged_write = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  view_2d_unmanaged       X_view(const_cast<PetscScalar *>(x_array), lda_x, m);
  view_1d_unmanaged       mean_view(mean_array, lda_x);
  view_2d_unmanaged_write E_view(e_array, lda_e, m);

  /* Batched ensemble update: E[i*ndof:(i+1)*ndof, :] = mean[i*ndof:(i+1)*ndof] + X[i*ndof:(i+1)*ndof, :] * G_i */
  Kokkos::parallel_for(
    "BatchedEnsembleUpdate", Kokkos::RangePolicy<exec_space>(0, n_vertices), KOKKOS_LAMBDA(const int i) {
      /* For each grid point i, compute E_i = mean_i + X_i * G_i */
      /* E_i is ndof x m, X_i is ndof x m, G_i is m x m */

      /* Compute X_i * G_i -> E_i using matrix multiplication */
      for (int row = 0; row < ndof; row++) {
        for (int col = 0; col < m; col++) {
          PetscScalar sum = 0.0;
          /* Matrix multiply: E_i(row, col) = sum_k X_i(row, k) * G_i(k, col) */
          for (int k = 0; k < m; k++) { sum += X_view(i * ndof + row, k) * G_batch(i, k, col); }
          /* Add mean and store: E[i*ndof + row, col] = mean[i*ndof + row] + sum */
          E_view(i * ndof + row, col) = mean_view(i * ndof + row) + sum;
        }
      }
    });
  Kokkos::fence();

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayWrite(da->ensemble, &e_array));
  PetscCall(VecRestoreArrayRead(impl->mean, &mean_array));
  PetscCall(MatDenseRestoreArrayRead(X, &x_array));

  /* Restore global observation arrays */
  PetscCall(VecRestoreArrayRead(r_inv_sqrt_global, &r_inv_sqrt_global_array));
  PetscCall(VecRestoreArrayRead(y_mean_global, &y_mean_global_array));
  PetscCall(VecRestoreArrayRead(observation, &y_global_array));
  PetscCall(MatDenseRestoreArrayRead(Z_global, &z_global_array));

  /* Ensemble has been updated in batched form above */
  PetscCall(MatAssemblyBegin(da->ensemble, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(da->ensemble, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
