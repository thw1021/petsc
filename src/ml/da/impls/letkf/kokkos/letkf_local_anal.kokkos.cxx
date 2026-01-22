#include "../letkf_impl.h"
#include <petscblaslapack.h>
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
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
  PetscScalar *all_v, *all_work;
  PetscReal   *all_lambda;
  #if defined(PETSC_USE_COMPLEX)
  PetscReal *all_rwork;
  #endif
  PetscBLASInt lwork_query = -1, lwork;
  PetscScalar  work_query;
  PetscBLASInt n_blas;
  PetscCall(PetscBLASIntCast(n_size, &n_blas));

  /* Query workspace size once */
  {
    PetscBLASInt info;
  #if defined(PETSC_USE_COMPLEX)
    PetscReal rwork_query;
    LAPACKsyev_("V", "U", &n_blas, &work_query, &n_blas, &rwork_query, &work_query, &lwork_query, &rwork_query, &info);
  #else
    LAPACKsyev_("V", "U", &n_blas, &work_query, &n_blas, &work_query, &work_query, &lwork_query, &info);
  #endif
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK workspace query failed");
    lwork = (PetscBLASInt)PetscRealPart(work_query);
  }

  #if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc4(n_batch * n_size * n_size, &all_v, n_batch * n_size, &all_lambda, n_batch * lwork, &all_work, n_batch * (3 * n_size - 2), &all_rwork));
  #else
  PetscCall(PetscMalloc3(n_batch * n_size * n_size, &all_v, n_batch * n_size, &all_lambda, n_batch * lwork, &all_work));
  #endif

  /* Process each matrix in parallel on host using LAPACK */
  Kokkos::parallel_for(
    "BatchedEigenSolve_Host", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      PetscBLASInt n   = n_blas;
      PetscBLASInt lda = n;
      PetscBLASInt info;
      PetscBLASInt lw = lwork;

      /* Pointers for this matrix */
      PetscScalar *v_ptr      = all_v + i * n_size * n_size;
      PetscReal   *lambda_ptr = all_lambda + i * n_size;
      PetscScalar *work_ptr   = all_work + i * lwork;
  #if defined(PETSC_USE_COMPLEX)
      PetscReal *rwork_ptr = all_rwork + i * (3 * n_size - 2);
  #endif

      /* Copy T_host(i, :, :) to v_ptr (column-major) */
      for (PetscInt j = 0; j < n_size; j++) {
        for (PetscInt k = 0; k < n_size; k++) v_ptr[k + j * n_size] = T_host(i, k, j);
      }

    /* Compute eigendecomposition: T = V * Lambda * V^T */
  #if defined(PETSC_USE_COMPLEX)
      LAPACKsyev_("V", "U", &n, v_ptr, &lda, lambda_ptr, work_ptr, &lw, rwork_ptr, &info);
  #else
      LAPACKsyev_("V", "U", &n, v_ptr, &lda, lambda_ptr, work_ptr, &lw, &info);
  #endif

      if (info != 0) {
        /* We cannot return error code from lambda, so we just abort or ignore.
           In production code, we should use a reduction to report errors. */
        Kokkos::abort("LAPACK eigendecomposition failed in parallel region");
      }

      /* Copy results back to host views */
      for (PetscInt j = 0; j < n_size; j++) {
        Lambda_host(i, j) = (PetscScalar)lambda_ptr[j];
        for (PetscInt k = 0; k < n_size; k++) V_host(i, k, j) = v_ptr[k + j * n_size];
      }
    });

  #if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree4(all_v, all_lambda, all_work, all_rwork));
  #else
  PetscCall(PetscFree3(all_v, all_lambda, all_work));
  #endif

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
static PetscErrorCode BatchedEigenSolve_CUDA(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, cusolverDnHandle_t cusolverH)
{
  cusolverStatus_t cusolver_status;
  syevjInfo_t      syevj_params = NULL;
  PetscScalar     *d_work;
  int             *d_info;
  int              lwork = 0;

  PetscFunctionBegin;
  /* Create syevj params */
  cusolver_status = cusolverDnCreateSyevjInfo(&syevj_params);
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreateSyevjInfo failed");

  /* Set default params */
  cusolverDnXsyevjSetTolerance(syevj_params, 1e-7);
  cusolverDnXsyevjSetMaxSweeps(syevj_params, 100);
  cusolverDnXsyevjSetSortEig(syevj_params, 1); /* Sort eigenvalues */

  /* Get raw pointers from Kokkos views - zero-copy access */
  PetscScalar *d_A = T_batch.data();
  PetscScalar *d_W = Lambda_batch.data();

  /* Allocate device memory for info array */
  PetscCallCUDA(cudaMalloc(&d_info, sizeof(int) * n_batch));

  /*
     OPTIMIZATION: Use strided batched API to avoid memory copies
     cuSOLVER expects contiguous layout: [matrix0][matrix1]...[matrixN]
     But T_batch has LayoutLeft with batch index first: T_batch(i,j,k)

     We need to check if the data is already in the right layout.
     For LayoutLeft with dimensions (n_batch, n_size, n_size):
     - stride[0] = 1 (batch index varies fastest)
     - stride[1] = n_batch (row index)
     - stride[2] = n_batch * n_size (column index)

     cuSOLVER needs: matrix i at offset i*n_size*n_size
     Our layout: element (i,j,k) at offset i + j*n_batch + k*n_batch*n_size

     These don't match, so we need to use strided access or reorganize.
     For now, we'll use the batched API with proper stride handling.
  */

  /* Query workspace size - use first matrix pointer with stride */
  #if defined(PETSC_USE_REAL_SINGLE)
  cusolver_status = cusolverDnSsyevjBatched_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A, n_size, d_W, &lwork, syevj_params, n_batch);
  #else
  cusolver_status = cusolverDnDsyevjBatched_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A, n_size, d_W, &lwork, syevj_params, n_batch);
  #endif
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDn*syevjBatched_bufferSize failed");

  /* Allocate workspace */
  PetscCallCUDA(cudaMalloc(&d_work, sizeof(PetscScalar) * lwork));

  /*
     NOTE: cuSOLVER batched API requires contiguous matrices.
     Since our LayoutLeft has batch index first (stride-1), we need to reorganize.
     We'll do an in-place transpose-like operation using a temporary buffer.
  */
  PetscScalar *d_A_contig;
  PetscCallCUDA(cudaMalloc(&d_A_contig, sizeof(PetscScalar) * n_batch * n_size * n_size));

  /* Copy T_batch to contiguous layout for cuSOLVER */
  Kokkos::parallel_for(
    "ReorganizeForCuSOLVER", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        for (int k = 0; k < n_size; k++) d_A_contig[i * n_size * n_size + k * n_size + j] = T_batch(i, j, k);
      }
    });
  Kokkos::fence();

  /* Solve batched eigendecomposition */
  #if defined(PETSC_USE_REAL_SINGLE)
  cusolver_status = cusolverDnSsyevjBatched(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A_contig, n_size, d_W, d_work, lwork, d_info, syevj_params, n_batch);
  #else
  cusolver_status = cusolverDnDsyevjBatched(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A_contig, n_size, d_W, d_work, lwork, d_info, syevj_params, n_batch);
  #endif
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDn*syevjBatched failed");

  /* Check info */
  int *h_info;
  PetscCall(PetscMalloc1(n_batch, &h_info));
  PetscCallCUDA(cudaMemcpy(h_info, d_info, sizeof(int) * n_batch, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n_batch; i++) {
    if (h_info[i] != 0) PetscCheck(h_info[i] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuSOLVER eigendecomposition failed for matrix %" PetscInt_FMT ": info=%d", i, h_info[i]);
  }
  PetscCall(PetscFree(h_info));

  /* Copy results back from contiguous layout to V_batch */
  Kokkos::parallel_for(
    "CopyResultsBack", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        Lambda_batch(i, j) = d_W[i * n_size + j];
        for (int k = 0; k < n_size; k++) V_batch(i, j, k) = d_A_contig[i * n_size * n_size + k * n_size + j];
      }
    });
  Kokkos::fence();

  /* Cleanup */
  PetscCallCUDA(cudaFree(d_A_contig));
  PetscCallCUDA(cudaFree(d_work));
  PetscCallCUDA(cudaFree(d_info));
  cusolverDnDestroySyevjInfo(syevj_params);
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
#if defined(KOKKOS_ENABLE_CUDA)
static PetscErrorCode BatchedEigenSolve(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, cusolverDnHandle_t cusolverH)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_CUDA(T_batch, Lambda_batch, V_batch, n_batch, n_size, cusolverH));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#else
static PetscErrorCode BatchedEigenSolve(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Host(T_batch, Lambda_batch, V_batch, n_batch, n_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  PetscDALETKFSetupLocalization_Kokkos - Prepares device views for localization matrix Q
*/
PetscErrorCode PetscDALETKFSetupLocalization_Kokkos(PetscDALETKFData *impl)
{
  PetscInt        nrows;

  PetscFunctionBegin;
  if (!impl->Q) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscKokkosInitializeCheck());

  /* Get CSR data */
  PetscInt rstart, rend, i, nnz;
  PetscCall(MatGetOwnershipRange(impl->Q, &rstart, &rend));
  nrows = rend - rstart;

  /* Define View types */
  using view_1d_int    = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
  using view_1d_scalar = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft>;

  /* Allocate device views */
  view_1d_int    *d_Q_i = new view_1d_int("Q_i", nrows + 1);
  view_1d_int    *d_Q_j = new view_1d_int("Q_j", nrows * impl->n_obs_vertex);
  view_1d_scalar *d_Q_a = new view_1d_scalar("Q_a", nrows * impl->n_obs_vertex);

  /* Create host mirrors */
  auto h_Q_i = Kokkos::create_mirror_view(*d_Q_i);
  auto h_Q_j = Kokkos::create_mirror_view(*d_Q_j);
  auto h_Q_a = Kokkos::create_mirror_view(*d_Q_a);

  /* Fill host mirrors */
  h_Q_i(0) = 0;
  for (i = 0; i < nrows; i++) {
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(impl->Q, rstart + i, &nnz, &cols, &vals));
    h_Q_i(i + 1) = h_Q_i(i) + nnz;
    for (PetscInt k = 0; k < nnz; k++) {
      h_Q_j(h_Q_i(i) + k) = cols[k];
      h_Q_a(h_Q_i(i) + k) = vals[k];
    }
    PetscCall(MatRestoreRow(impl->Q, rstart + i, &nnz, &cols, &vals));
  }

  /* Copy to device */
  Kokkos::deep_copy(*d_Q_i, h_Q_i);
  Kokkos::deep_copy(*d_Q_j, h_Q_j);
  Kokkos::deep_copy(*d_Q_a, h_Q_a);

  /* Store in impl */
  impl->Q_device_i = static_cast<void *>(d_Q_i);
  impl->Q_device_j = static_cast<void *>(d_Q_j);
  impl->Q_device_a = static_cast<void *>(d_Q_a);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDALETKFDestroyLocalization_Kokkos(PetscDALETKFData *impl)
{
  PetscFunctionBegin;
  if (impl->Q_device_i) {
    using view_1d_int = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
    delete static_cast<view_1d_int *>(impl->Q_device_i);
    impl->Q_device_i = NULL;
  }
  if (impl->Q_device_j) {
    using view_1d_int = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
    delete static_cast<view_1d_int *>(impl->Q_device_j);
    impl->Q_device_j = NULL;
  }
  if (impl->Q_device_a) {
    using view_1d_scalar = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft>;
    delete static_cast<view_1d_scalar *>(impl->Q_device_a);
    impl->Q_device_a = NULL;
  }
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
  inflation_inv  = 1.0 / da->inflation; /* (1/rho) for T matrix: T = (1/rho)I + S^T*S */

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
  using view_1d_int_const    = Kokkos::View<const PetscInt *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_scalar_const = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_int          = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
  using view_1d_scalar       = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft>;

  view_1d_int_const    Q_i_view;
  view_1d_int_const    Q_j_view;
  view_1d_scalar_const Q_a_view;

  if (impl->Q_device_i) {
    /* Use pre-allocated device views */
    view_1d_int    *d_Q_i = static_cast<view_1d_int *>(impl->Q_device_i);
    view_1d_int    *d_Q_j = static_cast<view_1d_int *>(impl->Q_device_j);
    view_1d_scalar *d_Q_a = static_cast<view_1d_scalar *>(impl->Q_device_a);

    Q_i_view = view_1d_int_const(d_Q_i->data(), d_Q_i->extent(0));
    Q_j_view = view_1d_int_const(d_Q_j->data(), d_Q_j->extent(0));
    Q_a_view = view_1d_scalar_const(d_Q_a->data(), d_Q_a->extent(0));
  } else {
    /* Fallback to host pointers (unsafe if not UVM) */
    PetscCheck(PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Q matrix must be setup with PetscDALETKFSetupLocalization_Kokkos");
  }

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
  PetscInt chunk_size;
  if (impl->batch_size > 0) {
    chunk_size = impl->batch_size;
  } else {
    /* Target ~2GB workspace. Approx memory per point: m*m*8 (T) + p*m*8 (Z) */
    /* With reuse: m*m*8 + p*m*8 */
    PetscInt mem_per_point = sizeof(PetscScalar) * (m * m + impl->n_obs_vertex * m);
    chunk_size             = (PetscInt)(2.0 * 1024 * 1024 * 1024 / mem_per_point);
    /* Clamp to reasonable max to avoid huge allocations even if memory allows */
    if (chunk_size > 32768) chunk_size = 32768;
  }

  if (chunk_size < 1) chunk_size = 1;
  if (chunk_size > n_vertices) chunk_size = n_vertices;

  /* OPTIMIZATION: Create cuSOLVER handle once, reuse across chunks */
#if defined(KOKKOS_ENABLE_CUDA)
  cusolverDnHandle_t cusolverH = nullptr;
  cusolverStatus_t   cusolver_status;
  cusolver_status = cusolverDnCreate(&cusolverH);
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreate failed");
#endif

  /* Loop over chunks */
  for (PetscInt chunk_start = 0; chunk_start < n_vertices; chunk_start += chunk_size) {
    PetscInt chunk_end       = (chunk_start + chunk_size > n_vertices) ? n_vertices : chunk_start + chunk_size;
    PetscInt n_batch_current = chunk_end - chunk_start;

    /* Copy host variables to local scope to avoid capturing host pointers in device kernels */
    PetscInt n_obs_vertex_copy = impl->n_obs_vertex;

    /* Batched workspace for CURRENT chunk (device memory) */
    /* NOTE: G_batch eliminated - computed on-the-fly during ensemble update */
    view_3d Z_batch("Z_batch", n_batch_current, n_obs_vertex_copy, m);                // (n_batch_current, n_obs_vertex, m)
    view_3d S_batch = Z_batch;                                                        // Reuse Z memory for S
    view_3d T_batch("T_batch", n_batch_current, m, m);                                // (n_batch_current, m, m)
    view_3d V_batch = T_batch;                                                        // Reuse T memory for V
    view_2d Lambda_batch("Lambda_batch", n_batch_current, m);                         // (n_batch_current, m)
    view_3d T_sqrt_batch("T_sqrt_batch", n_batch_current, m, m);                      // (n_batch_current, m, m)
    view_2d w_batch("w_batch", n_batch_current, m);                                   // (n_batch_current, m)
    view_2d delta_batch("delta_batch", n_batch_current, n_obs_vertex_copy);           // (n_batch_current, n_obs_vertex)
    view_2d y_batch("y_batch", n_batch_current, n_obs_vertex_copy);                   // (n_batch_current, n_obs_vertex)
    view_2d y_mean_batch("y_mean_batch", n_batch_current, n_obs_vertex_copy);         // (n_batch_current, n_obs_vertex)
    view_2d r_inv_sqrt_batch("r_inv_sqrt_batch", n_batch_current, n_obs_vertex_copy); // (n_batch_current, n_obs_vertex)
    view_2d temp1_batch("temp1_batch", n_batch_current, m);                           // (n_batch_current, m) - Workspace
    view_2d temp2_batch("temp2_batch", n_batch_current, m);                           // (n_batch_current, m) - Workspace
    view_2d inv_sqrt_lambda_batch("inv_sqrt_lambda_batch", n_batch_current, m);       // (n_batch_current, m) - Precomputed 1/sqrt(Lambda)

    /* ===================================================================== */
    /* Step 2.1.2: Fused observation extraction and S/Delta computation     */
    /* ===================================================================== */
    /* Extract local observations and immediately compute S and delta       */
    /* This fusion eliminates one kernel launch and improves cache locality */
    Kokkos::parallel_for(
      "ExtractAndComputeSAndDelta", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i_local) {
        PetscInt i_global = chunk_start + i_local;
        /* Get Q row for this grid point using CSR format */
        PetscInt row_start = Q_i_view(i_global);
        PetscInt row_end   = Q_i_view(i_global + 1);
        PetscInt ncols     = row_end - row_start;

        /* Extract observations and compute S/delta for this grid point */
        for (PetscInt k = 0; k < ncols; k++) {
          PetscInt    obs_idx = Q_j_view(row_start + k);
          PetscScalar weight  = Q_a_view(row_start + k);

          /* Extract observation vectors */
          PetscScalar y_val      = y_global_view(obs_idx);
          PetscScalar y_mean_val = y_mean_global_view(obs_idx);
          PetscScalar r_inv_sqrt = r_inv_sqrt_global_view(obs_idx) * Kokkos::sqrt(PetscRealPart(weight));

          /* Store for later use if needed */
          y_batch(i_local, k)          = y_val;
          y_mean_batch(i_local, k)     = y_mean_val;
          r_inv_sqrt_batch(i_local, k) = r_inv_sqrt;

          /* Compute delta immediately: delta = R^{-1/2}(y - y_mean) */
          delta_batch(i_local, k) = (y_val - y_mean_val) * r_inv_sqrt;

          /* Compute S row: S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1) */
          PetscScalar scale_factor = scale * r_inv_sqrt;
          for (int j = 0; j < m; j++) {
            PetscScalar z_val      = Z_global_view(obs_idx, j);
            Z_batch(i_local, k, j) = z_val; /* Store Z for potential later use */
            S_batch(i_local, k, j) = (z_val - y_mean_val) * scale_factor;
          }
        }
      });
    Kokkos::fence();

    /* ===================================================================== */
    /* Step 2.1.4: Optimized T matrix formation (T = (1/rho)I + S^T * S)    */
    /* ===================================================================== */
    /* Compute T_i = (1/rho)I + S_i^T * S_i for current chunk */
    /* Exploit symmetry: only compute upper triangle, then copy to lower */
    /* This reduces operations by ~50% */
    Kokkos::parallel_for(
      "ComputeAllTMatrices", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i) {
        auto S_i = Kokkos::subview(S_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto T_i = Kokkos::subview(T_batch, i, Kokkos::ALL(), Kokkos::ALL());

        /* Compute upper triangle of T_i = (1/rho)I + S_i^T * S_i */
        /* T_i(j,k) = (1/rho)*delta_jk + sum_p S_i(p,j) * S_i(p,k) for j <= k */
        for (int j = 0; j < m; j++) {
          for (int k = j; k < m; k++) {
            PetscScalar sum = (j == k) ? inflation_inv : 0.0;
            for (int p = 0; p < n_obs_vertex_copy; p++) sum += S_i(p, j) * S_i(p, k);
            T_i(j, k) = sum;
          }
        }

        /* Copy upper triangle to lower triangle (T is symmetric) */
        for (int j = 0; j < m; j++) {
          for (int k = 0; k < j; k++) T_i(j, k) = T_i(k, j);
        }
      });
    Kokkos::fence();

    /* ===================================================================== */
    /* Step 3.1.1: Batched eigendecomposition for current chunk            */
    /* ===================================================================== */
    /* Compute T_i = V_i * Lambda_i * V_i^T for current chunk */
#if defined(KOKKOS_ENABLE_CUDA)
    PetscCall(BatchedEigenSolve(T_batch, Lambda_batch, V_batch, n_batch_current, m, cusolverH));
#else
    PetscCall(BatchedEigenSolve(T_batch, Lambda_batch, V_batch, n_batch_current, m));
#endif

    /* ===================================================================== */
    /* Step 3.1.2: Precompute w and inv_sqrt_lambda for ensemble update    */
    /* ===================================================================== */
    /* Compute w_i = T_i^{-1} * (S_i^T * delta_i) using eigendecomposition */
    /* Precompute 1/sqrt(Lambda) for use in ensemble update */
    Kokkos::parallel_for(
      "ComputeWeightsAndInvSqrtLambda", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i) {
        auto S_i               = Kokkos::subview(S_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto V_i               = Kokkos::subview(V_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto Lambda_i          = Kokkos::subview(Lambda_batch, i, Kokkos::ALL());
        auto delta_i           = Kokkos::subview(delta_batch, i, Kokkos::ALL());
        auto w_i               = Kokkos::subview(w_batch, i, Kokkos::ALL());
        auto inv_sqrt_lambda_i = Kokkos::subview(inv_sqrt_lambda_batch, i, Kokkos::ALL());
        auto temp1             = Kokkos::subview(temp1_batch, i, Kokkos::ALL());
        auto temp2             = Kokkos::subview(temp2_batch, i, Kokkos::ALL());

        /* 1. Compute w_i = V * L^-1 * V^T * S^T * delta */
        /* Step 1a: temp1 = S^T * delta using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::Transpose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, S_i, delta_i, 0.0, temp1);

        /* Step 1b: temp2 = V^T * temp1 using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::Transpose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, V_i, temp1, 0.0, temp2);

        /* Step 1c: temp2 = temp2 / Lambda */
        for (int j = 0; j < m; j++) temp2(j) /= (Lambda_i(j) + 1.0e-14);

        /* Step 1d: w = V * temp2 using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, V_i, temp2, 0.0, w_i);

        /* 2. Precompute 1/sqrt(Lambda) for ensemble update */
        for (int p = 0; p < m; p++) inv_sqrt_lambda_i(p) = 1.0 / Kokkos::sqrt(PetscRealPart(Lambda_i(p)) + 1.0e-14);
      });
    Kokkos::fence();

    /* ===================================================================== */
    /* Step 3.1.3: Fused G computation and ensemble update                  */
    /* ===================================================================== */
    /* Compute E[i,:] = mean[i] + X[i,:] * G_i on-the-fly */
    /* G_i is computed column-by-column and immediately applied */
    /* This eliminates the need to store G_batch, saving m*m*n_batch memory */
    Kokkos::parallel_for(
      "FusedGComputeAndEnsembleUpdate", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i_local) {
        PetscInt i_global = chunk_start + i_local;

        auto X_i    = Kokkos::subview(X_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof), Kokkos::ALL());
        auto E_i    = Kokkos::subview(E_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof), Kokkos::ALL());
        auto mean_i = Kokkos::subview(mean_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof));

        auto V_i               = Kokkos::subview(V_batch, i_local, Kokkos::ALL(), Kokkos::ALL());
        auto w_i               = Kokkos::subview(w_batch, i_local, Kokkos::ALL());
        auto inv_sqrt_lambda_i = Kokkos::subview(inv_sqrt_lambda_batch, i_local, Kokkos::ALL());
        auto T_sqrt_i          = Kokkos::subview(T_sqrt_batch, i_local, Kokkos::ALL(), Kokkos::ALL());

        /* Initialize E_i with mean */
        for (int row = 0; row < ndof; row++) {
          PetscScalar m_val = mean_i(row);
          for (int col = 0; col < m; col++) E_i(row, col) = m_val;
        }

        /* Compute T_sqrt = V * diag(1/sqrt(Lambda)) * V^T */
        /* Optimized: Exploit symmetry - only compute upper triangle, then copy to lower */
        /* T_sqrt(j,k) = sum_p V(j,p) * V(k,p) / sqrt(Lambda(p)) for j <= k */
        for (int j = 0; j < m; j++) {
          for (int k = j; k < m; k++) {
            PetscScalar sum = 0.0;
            for (int p = 0; p < m; p++) sum += V_i(j, p) * V_i(k, p) * inv_sqrt_lambda_i(p);
            T_sqrt_i(j, k) = sum;
          }
        }
        /* Copy upper triangle to lower triangle (T_sqrt is symmetric) */
        for (int j = 0; j < m; j++) {
          for (int k = 0; k < j; k++) T_sqrt_i(j, k) = T_sqrt_i(k, j);
        }

        /* Compute E_i += X_i * G_i column-by-column */
        /* G_i(:,k) = w_i + sqrt(m-1) * T_sqrt_i(:,k) */
        for (int k = 0; k < m; k++) {
          /* Compute column k of G on-the-fly */
          for (int row = 0; row < ndof; row++) {
            PetscScalar sum = 0.0;
            for (int j = 0; j < m; j++) {
              /* G_i(j,k) = w_i(j) + sqrt(m-1) * T_sqrt_i(j,k) */
              PetscScalar G_jk = w_i(j) + sqrt_m_minus_1 * T_sqrt_i(j, k);
              sum += X_i(row, j) * G_jk;
            }
            E_i(row, k) += sum;
          }
        }
      });
    Kokkos::fence();
  }

  /* OPTIMIZATION: Destroy cuSOLVER handle after all chunks */
#if defined(KOKKOS_ENABLE_CUDA)
  if (cusolverH) cusolverDnDestroy(cusolverH);
#endif

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

  {
    MatInfo   info;
    PetscReal flops = 0.0;
    PetscReal n_obs_total;

    if (impl->Q) {
      PetscCall(MatGetInfo(impl->Q, MAT_LOCAL, &info));
      n_obs_total = info.nz_used;
    } else {
      n_obs_total = 0.0;
    }

    /* Step 2.1.2: Fused observation extraction and S/Delta computation */
    flops += n_obs_total * (2.0 + 2.0 * m);

    /* Step 2.1.4: Optimized T matrix formation */
    flops += (PetscReal)n_vertices * m * (m + 1) * impl->n_obs_vertex;

    /* Step 3.1.2: Precompute w and inv_sqrt_lambda */
    flops += (PetscReal)n_vertices * (2.0 * m * impl->n_obs_vertex + 4.0 * m * m + 3.0 * m);

    /* Step 3.1.3: Fused G computation and ensemble update */
    /* T_sqrt: 1.5*m^3 + 1.5*m^2 */
    flops += (PetscReal)n_vertices * (1.5 * m * m * m + 1.5 * m * m);
    /* E update: ndof * m * (4*m + 1) */
    /* Note: G_jk computation (2 flops) is inside the inner loop, so it's 2*m*ndof*m */
    /* Matrix product X*G (2 flops) is also 2*m*ndof*m */
    flops += (PetscReal)n_vertices * ndof * m * (4.0 * m + 1.0);

    PetscCall(PetscLogGpuFlops(flops));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
