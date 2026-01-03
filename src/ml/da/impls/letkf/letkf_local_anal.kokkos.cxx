#include "letkf_impl.h"
#include <petscblaslapack.h>
#include <Kokkos_Core.hpp>
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBatched_SVD_Serial_Impl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_Util.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
  #include <cusolverDn.h>
  #include <cuda_runtime.h>
  #include <petscdevice_cuda.h>
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

  /* Allocate contiguous buffers for LAPACK for ALL matrices */
  PetscScalar *all_v, *all_lambda, *all_work;
  PetscBLASInt lwork_query = -1, lwork;
  PetscScalar  work_query;
  PetscBLASInt n_blas;
  PetscCall(PetscBLASIntCast(n_size, &n_blas));

  /* Query workspace size once */
  {
    PetscBLASInt info;
    LAPACKsyev_("V", "U", &n_blas, &work_query, &n_blas, &work_query, &work_query, &lwork_query, &info);
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK workspace query failed");
    lwork = (PetscBLASInt)PetscRealPart(work_query);
  }

  PetscCall(PetscMalloc3(n_batch * n_size * n_size, &all_v, n_batch * n_size, &all_lambda, n_batch * lwork, &all_work));

  /* Process each matrix in parallel on host using LAPACK */
  Kokkos::parallel_for(
    "BatchedEigenSolve_Host", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      PetscBLASInt n   = n_blas;
      PetscBLASInt lda = n;
      PetscBLASInt info;
      PetscBLASInt lw = lwork;

      /* Pointers for this matrix */
      PetscScalar *v_ptr      = all_v + i * n_size * n_size;
      PetscScalar *lambda_ptr = all_lambda + i * n_size;
      PetscScalar *work_ptr   = all_work + i * lwork;

      /* Copy T_host(i, :, :) to v_ptr (column-major) */
      for (PetscInt j = 0; j < n_size; j++) {
        for (PetscInt k = 0; k < n_size; k++) { v_ptr[k + j * n_size] = T_host(i, k, j); }
      }

      /* Compute eigendecomposition: T = V * Lambda * V^T */
      LAPACKsyev_("V", "U", &n, v_ptr, &lda, lambda_ptr, work_ptr, &lw, &info);

      if (info != 0) {
        /* We cannot return error code from lambda, so we just abort or ignore. 
           In production code, we should use a reduction to report errors. */
        Kokkos::abort("LAPACK eigendecomposition failed in parallel region");
      }

      /* Copy results back to host views */
      for (PetscInt j = 0; j < n_size; j++) {
        Lambda_host(i, j) = lambda_ptr[j];
        for (PetscInt k = 0; k < n_size; k++) { V_host(i, k, j) = v_ptr[k + j * n_size]; }
      }
    });

  PetscCall(PetscFree3(all_v, all_lambda, all_work));

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
  syevjInfo_t        syevj_params = NULL;
  PetscScalar       *d_work;
  int               *d_info;
  PetscScalar       *d_A, *d_W;
  int                lwork = 0;

  PetscFunctionBegin;

  /* Create cuSOLVER handle */
  cusolver_status = cusolverDnCreate(&cusolverH);
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreate failed");

  /* Create syevj params */
  cusolver_status = cusolverDnCreateSyevjInfo(&syevj_params);
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreateSyevjInfo failed");

  /* Set default params */
  cusolverDnXsyevjSetTolerance(syevj_params, 1e-7);
  cusolverDnXsyevjSetMaxSweeps(syevj_params, 100);
  cusolverDnXsyevjSetSortEig(syevj_params, 1); /* Sort eigenvalues */

  /* Allocate device memory for ALL matrices */
  PetscCallCUDA(cudaMalloc(&d_A, sizeof(PetscScalar) * n_batch * n_size * n_size));
  PetscCallCUDA(cudaMalloc(&d_W, sizeof(PetscScalar) * n_batch * n_size));
  PetscCallCUDA(cudaMalloc(&d_info, sizeof(int) * n_batch));

  /* Copy T_batch (interleaved) to d_A (contiguous) */
  /* T_batch(i, j, k) -> d_A[i * n*n + k*n + j] (Column-Major) */
  Kokkos::parallel_for(
    "CopyTToContiguous", Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>({0, 0, 0}, {n_batch, n_size, n_size}),
    KOKKOS_LAMBDA(const int i, const int j, const int k) { d_A[i * n_size * n_size + k * n_size + j] = T_batch(i, j, k); });
  Kokkos::fence();

  /* Query workspace size */
  #if defined(PETSC_USE_REAL_SINGLE)
  cusolver_status = cusolverDnSsyevjBatched_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A, n_size, d_W, &lwork, syevj_params, n_batch);
  #else
  cusolver_status = cusolverDnDsyevjBatched_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A, n_size, d_W, &lwork, syevj_params, n_batch);
  #endif
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDn*syevjBatched_bufferSize failed");

  /* Allocate workspace */
  cudaMalloc(&d_work, sizeof(PetscScalar) * lwork);

  /* Solve batched eigendecomposition */
  #if defined(PETSC_USE_REAL_SINGLE)
  cusolver_status = cusolverDnSsyevjBatched(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A, n_size, d_W, d_work, lwork, d_info, syevj_params, n_batch);
  #else
  cusolver_status = cusolverDnDsyevjBatched(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A, n_size, d_W, d_work, lwork, d_info, syevj_params, n_batch);
  #endif
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDn*syevjBatched failed");

  /* Check info */
  int *h_info;
  PetscCall(PetscMalloc1(n_batch, &h_info));
  PetscCallCUDA(cudaMemcpy(h_info, d_info, sizeof(int) * n_batch, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n_batch; i++) {
    if (h_info[i] != 0) { PetscCheck(h_info[i] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuSOLVER eigendecomposition failed for matrix %" PetscInt_FMT ": info=%d", i, h_info[i]); }
  }
  PetscCall(PetscFree(h_info));

  /* Copy results back */
  /* d_A (eigenvectors) -> V_batch */
  /* d_W (eigenvalues) -> Lambda_batch */
  Kokkos::parallel_for(
    "CopyResultsBack", Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>({0, 0, 0}, {n_batch, n_size, n_size}), KOKKOS_LAMBDA(const int i, const int j, const int k) {
      V_batch(i, j, k) = d_A[i * n_size * n_size + k * n_size + j];
      if (k == 0) { /* Only need to copy eigenvalues once per row */
        Lambda_batch(i, j) = d_W[i * n_size + j];
      }
    });
  Kokkos::fence();

  /* Cleanup */
  PetscCallCUDA(cudaFree(d_A));
  PetscCallCUDA(cudaFree(d_W));
  PetscCallCUDA(cudaFree(d_work));
  PetscCallCUDA(cudaFree(d_info));
  cusolverDnDestroySyevjInfo(syevj_params);
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
  /* 
     NOTE ON PARALLELISM STRATEGY:
     We use Kokkos::RangePolicy over grid points (n_vertices) combined with KokkosBatched::Serial kernels.
     Since the data layout is LayoutLeft (Column-Major) to match PETSc/LAPACK, the index 'i' (grid point)
     is the fastest varying index (stride 1).

     RangePolicy maps consecutive threads to consecutive 'i', ensuring perfect memory coalescing 
     when accessing arrays like S_batch(i, p, j).

     Using TeamPolicy/TeamVectorRange to parallelize inner loops (m or p) would assign a team to 'i',
     causing threads within the team to access S_batch with stride 'n_vertices', which leads to 
     uncoalesced memory access and poor performance on GPUs.

     Therefore, RangePolicy + SerialGemm is the optimal strategy for this data layout.
  */
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_3d    = Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, exec_space>;
  using view_2d    = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>;

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

  /* Determine chunk size to avoid OOM on large grids */
  /* Target ~2GB workspace. Approx memory per point: m*m*8 (T) + p*m*8 (Z) */
  /* With reuse: m*m*8 + p*m*8 */
  PetscInt mem_per_point = sizeof(PetscScalar) * (m * m + impl->p_local * m);
  PetscInt chunk_size    = (PetscInt)(2.0 * 1024 * 1024 * 1024 / mem_per_point);

  /* Allow override via command line */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-da_letkf_chunk_size", &chunk_size, NULL));

  if (chunk_size < 1) chunk_size = 1;
  if (chunk_size > n_vertices) chunk_size = n_vertices;

  /* Loop over chunks */
  for (PetscInt chunk_start = 0; chunk_start < n_vertices; chunk_start += chunk_size) {
    PetscInt chunk_end       = (chunk_start + chunk_size > n_vertices) ? n_vertices : chunk_start + chunk_size;
    PetscInt n_batch_current = chunk_end - chunk_start;

    /* Batched workspace for CURRENT chunk (device memory) */
    view_3d Z_batch("Z_batch", n_batch_current, impl->p_local, m);                // (n_batch_current, p_local, m)
    view_3d S_batch = Z_batch;                                                    // Reuse Z memory for S
    view_3d T_batch("T_batch", n_batch_current, m, m);                            // (n_batch_current, m, m)
    view_3d V_batch = T_batch;                                                    // Reuse T memory for V
    view_2d Lambda_batch("Lambda_batch", n_batch_current, m);                     // (n_batch_current, m)
    view_3d T_sqrt_batch("T_sqrt_batch", n_batch_current, m, m);                  // (n_batch_current, m, m)
    view_3d G_batch("G_batch", n_batch_current, m, m);                            // (n_batch_current, m, m)
    view_2d w_batch("w_batch", n_batch_current, m);                               // (n_batch_current, m)
    view_2d delta_batch("delta_batch", n_batch_current, impl->p_local);           // (n_batch_current, p_local)
    view_2d y_batch("y_batch", n_batch_current, impl->p_local);                   // (n_batch_current, p_local)
    view_2d y_mean_batch("y_mean_batch", n_batch_current, impl->p_local);         // (n_batch_current, p_local)
    view_2d r_inv_sqrt_batch("r_inv_sqrt_batch", n_batch_current, impl->p_local); // (n_batch_current, p_local)
    view_2d temp1_batch("temp1_batch", n_batch_current, m);                       // (n_batch_current, m) - Workspace
    view_2d temp2_batch("temp2_batch", n_batch_current, m);                       // (n_batch_current, m) - Workspace

    /* ===================================================================== */
    /* Step 2.1.2: Parallelize observation extraction for current chunk      */
    /* ===================================================================== */
    Kokkos::parallel_for(
      "ExtractAllLocalObservations", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i_local) {
        PetscInt i_global = chunk_start + i_local;
        /* Get Q row for this grid point using CSR format */
        PetscInt row_start = Q_i_view(i_global);
        PetscInt row_end   = Q_i_view(i_global + 1);
        PetscInt ncols     = row_end - row_start;

        /* Extract observations for this grid point */
        for (PetscInt k = 0; k < ncols; k++) {
          PetscInt    obs_idx = Q_j_view(row_start + k);
          PetscScalar weight  = Q_a_view(row_start + k);

          /* Extract observation vectors */
          y_batch(i_local, k)          = y_global_view(obs_idx);
          y_mean_batch(i_local, k)     = y_mean_global_view(obs_idx);
          r_inv_sqrt_batch(i_local, k) = r_inv_sqrt_global_view(obs_idx) * Kokkos::sqrt(weight);

          /* Extract Z matrix columns for this observation */
          for (int j = 0; j < m; j++) { Z_batch(i_local, k, j) = Z_global_view(obs_idx, j); }
        }
      });
    Kokkos::fence();

    /* ===================================================================== */
    /* Step 2.1.3: Parallelize normalized innovation computation            */
    /* ===================================================================== */
    /* Compute S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1) for current chunk */
    /* Also compute delta = R^{-1/2}(y - y_mean) for current chunk */
    Kokkos::parallel_for(
      "ComputeSAndDelta", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i) {
        for (int k = 0; k < impl->p_local; k++) {
          /* Compute delta */
          delta_batch(i, k) = (y_batch(i, k) - y_mean_batch(i, k)) * r_inv_sqrt_batch(i, k);

          /* Compute S row */
          PetscScalar scale_factor = scale * r_inv_sqrt_batch(i, k);
          PetscScalar mean_val     = y_mean_batch(i, k);
          for (int j = 0; j < m; j++) { S_batch(i, k, j) = (Z_batch(i, k, j) - mean_val) * scale_factor; }
        }
      });
    Kokkos::fence();

    /* ===================================================================== */
    /* Step 2.1.4: Parallelize T matrix formation (T = (1/ρ)I + S^T * S)  */
    /* ===================================================================== */
    /* Compute T_i = (1/ρ)I + S_i^T * S_i for current chunk */
    /* This is the Gram matrix C = S^T * S plus scaled identity */
    Kokkos::parallel_for(
      "ComputeAllTMatrices", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i) {
        auto S_i = Kokkos::subview(S_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto T_i = Kokkos::subview(T_batch, i, Kokkos::ALL(), Kokkos::ALL());

        /* Init T_i = (1/ρ)I */
        for (int j = 0; j < m; j++) {
          for (int k = 0; k < m; k++) { T_i(j, k) = 0.0; }
          T_i(j, j) = inflation_inv;
        }

        /* T_i += S_i^T * S_i */
        KokkosBatched::SerialGemm<KokkosBatched::Trans::Transpose, KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Unblocked>::invoke(1.0, S_i, S_i, 1.0, T_i);
      });
    Kokkos::fence();

    /* ===================================================================== */
    /* Step 3.1.1: Batched eigendecomposition for current chunk            */
    /* ===================================================================== */
    /* Compute T_i = V_i * Lambda_i * V_i^T for current chunk */
    PetscCall(BatchedEigenSolve(T_batch, Lambda_batch, V_batch, n_batch_current, m));

    /* ===================================================================== */
    /* Step 3.1.2: Batched G matrix formation (Fused)                      */
    /* ===================================================================== */
    /* Compute w_i = T_i^{-1} * (S_i^T * delta_i) */
    /* Compute T_sqrt_i = T_i^{-1/2} */
    /* Compute G_i = w_i * 1^T + sqrt(m-1) * T_sqrt_i */
    Kokkos::parallel_for(
      "ComputeGMatrices", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i) {
        auto S_i      = Kokkos::subview(S_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto V_i      = Kokkos::subview(V_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto Lambda_i = Kokkos::subview(Lambda_batch, i, Kokkos::ALL());
        auto delta_i  = Kokkos::subview(delta_batch, i, Kokkos::ALL());
        auto w_i      = Kokkos::subview(w_batch, i, Kokkos::ALL());
        auto T_sqrt_i = Kokkos::subview(T_sqrt_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto G_i      = Kokkos::subview(G_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto temp1    = Kokkos::subview(temp1_batch, i, Kokkos::ALL());
        auto temp2    = Kokkos::subview(temp2_batch, i, Kokkos::ALL());

        /* 1. Compute w_i = V * L^-1 * V^T * S^T * delta */
        /* temp1 = S^T * delta */
        for (int j = 0; j < m; j++) {
          PetscScalar sum = 0.0;
          for (int k = 0; k < impl->p_local; k++) { sum += S_i(k, j) * delta_i(k); }
          temp1(j) = sum;
        }

        /* temp2 = V^T * temp1 */
        for (int j = 0; j < m; j++) {
          PetscScalar sum = 0.0;
          for (int k = 0; k < m; k++) { sum += V_i(k, j) * temp1(k); }
          temp2(j) = sum;
        }

        /* temp2 = temp2 / Lambda */
        for (int j = 0; j < m; j++) { temp2(j) /= Lambda_i(j); }

        /* w = V * temp2 */
        for (int j = 0; j < m; j++) {
          PetscScalar sum = 0.0;
          for (int k = 0; k < m; k++) { sum += V_i(j, k) * temp2(k); }
          w_i(j) = sum;
        }

        /* 2. Compute T_sqrt = V * L^-1/2 * V^T */
        /* Pre-compute inverse square roots in temp1 (reused) */
        for (int p = 0; p < m; p++) { temp1(p) = 1.0 / Kokkos::sqrt(Lambda_i(p)); }

        /* T_sqrt(j, k) = sum_p V(j, p) * V(k, p) * inv_sqrt_lambda(p) */
        /* 3. Compute G = w * 1^T + sqrt(m-1) * T_sqrt (Fused) */
        for (int j = 0; j < m; j++) {
          PetscScalar w_val = w_i(j);
          for (int k = 0; k < m; k++) {
            PetscScalar sum = 0.0;
            for (int p = 0; p < m; p++) { sum += V_i(j, p) * V_i(k, p) * temp1(p); }
            T_sqrt_i(j, k) = sum;
            G_i(j, k)      = sqrt_m_minus_1 * sum + w_val;
          }
        }
      });
    Kokkos::fence();

    /* ===================================================================== */
    /* Step 3.1.5: Parallelize ensemble update for current chunk             */
    /* ===================================================================== */
    /* Compute E[i,:] = mean[i] + X[i,:] * G_i for current chunk */
    Kokkos::parallel_for(
      "BatchedEnsembleUpdate", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i_local) {
        PetscInt i_global = chunk_start + i_local;
        /* For each grid point i, compute E_i = mean_i + X_i * G_i */
        auto X_i    = Kokkos::subview(X_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof), Kokkos::ALL());
        auto E_i    = Kokkos::subview(E_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof), Kokkos::ALL());
        auto G_i    = Kokkos::subview(G_batch, i_local, Kokkos::ALL(), Kokkos::ALL());
        auto mean_i = Kokkos::subview(mean_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof));

        /* Init E_i with mean */
        for (int row = 0; row < ndof; row++) {
          PetscScalar m_val = mean_i(row);
          for (int col = 0; col < m; col++) { E_i(row, col) = m_val; }
        }

        /* E_i += X_i * G_i */
        KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Unblocked>::invoke(1.0, X_i, G_i, 1.0, E_i);
      });
    Kokkos::fence();
  }

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
