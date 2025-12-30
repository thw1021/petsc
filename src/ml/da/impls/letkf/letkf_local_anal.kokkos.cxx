#include "letkf_impl.h"
#include <petscblaslapack.h>
#include <Kokkos_Core.hpp>
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBatched_SVD_Serial_Impl.hpp>

/*
  PetscDALETKFLocalAnalysis_GPU - GPU-accelerated local LETKF analysis using Kokkos batched operations

  This is a complete GPU implementation that processes all vertices in parallel.
  Currently implements Phase 1: Getting CSR arrays from Q matrix.

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
  The local analysis at each vertex is serial and independent. This GPU implementation
  uses Kokkos device views instead of PETSc objects for the local analysis workspace,
  avoiding the need for PETSC_COMM_SELF objects. All local computations are performed
  directly on device memory using Kokkos parallel_for kernels.
*/
PetscErrorCode PetscDALETKFLocalAnalysis_GPU(PetscDA da, PetscDALETKFData *impl, PetscInt m, PetscInt n_vertices, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global)
{
  PetscFunctionBegin;
  /* Phase 1: Get Q matrix CSR arrays with memory type detection */
  const PetscInt    *Q_i = NULL, *Q_j = NULL; // CSR row pointers and column indices
  const PetscScalar *Q_v = NULL;              // CSR values
  PetscMemType       Q_memtype;
  PetscInt           p_local = impl->p_local; // Q_NUM_LOCAL_OBSERVATIONS_MAX

  PetscCall(MatSeqAIJGetCSRAndMemType(impl->Q, &Q_i, &Q_j, (PetscScalar **)&Q_v, &Q_memtype));

  /* Verify that p_local matches Q_NUM_LOCAL_OBSERVATIONS_MAX */
  PetscCheck(p_local == Q_NUM_LOCAL_OBSERVATIONS_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "p_local (%" PetscInt_FMT ") must equal Q_NUM_LOCAL_OBSERVATIONS_MAX (%d) for stack-allocated arrays", p_local, Q_NUM_LOCAL_OBSERVATIONS_MAX);

  /* Verify that ensemble size does not exceed ENSEMBLE_SIZE_MAX */
  PetscCheck(m <= ENSEMBLE_SIZE_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m (%" PetscInt_FMT ") exceeds ENSEMBLE_SIZE_MAX (%d)", m, ENSEMBLE_SIZE_MAX);

  /* Phase 2: Get dense matrix/vector arrays with memory type detection */
  const PetscScalar *Z_global_array          = NULL;
  const PetscScalar *y_global_array          = NULL;
  const PetscScalar *y_mean_global_array     = NULL;
  const PetscScalar *r_inv_sqrt_global_array = NULL;
  const PetscScalar *X_array                 = NULL;
  const PetscScalar *mean_array              = NULL;
  PetscScalar       *ensemble_array          = NULL;

  PetscMemType Z_memtype, y_memtype, y_mean_memtype, r_inv_sqrt_memtype;
  PetscMemType X_memtype, mean_memtype, ensemble_memtype;

  PetscInt obs_size_global, lda_z_global, lda_x, lda_ensemble;

  /* Get array pointers and memory types for all dense matrices and vectors */
  PetscCall(MatDenseGetArrayReadAndMemType(Z_global, &Z_global_array, &Z_memtype));
  PetscCall(VecGetArrayReadAndMemType(observation, &y_global_array, &y_memtype));
  PetscCall(VecGetArrayReadAndMemType(y_mean_global, &y_mean_global_array, &y_mean_memtype));
  PetscCall(VecGetArrayReadAndMemType(r_inv_sqrt_global, &r_inv_sqrt_global_array, &r_inv_sqrt_memtype));
  PetscCall(MatDenseGetArrayReadAndMemType(X, &X_array, &X_memtype));
  PetscCall(VecGetArrayReadAndMemType(impl->mean, &mean_array, &mean_memtype));
  PetscCall(MatDenseGetArrayWriteAndMemType(da->ensemble, &ensemble_array, &ensemble_memtype));

  /* Get dimensions and leading dimensions */
  PetscCall(VecGetSize(observation, &obs_size_global));
  PetscCall(MatDenseGetLDA(Z_global, &lda_z_global));
  PetscCall(MatDenseGetLDA(X, &lda_x));
  PetscCall(MatDenseGetLDA(da->ensemble, &lda_ensemble));

  /* Phase 3: Parallel extraction kernel - extract local obs for all vertices */
  PetscCall(PetscKokkosInitializeCheck());

  /* Create device views for global data (copy from host if needed) */
  Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>    dQ_i("Q_i", n_vertices + 1);
  Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>    dQ_j("Q_j", n_vertices * p_local);
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> dQ_v("Q_v", n_vertices * p_local);
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> dZ_global("Z_global", lda_z_global * m);
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> dy_global("y_global", obs_size_global);
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> dy_mean_global("y_mean_global", obs_size_global);
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> dr_inv_sqrt_global("r_inv_sqrt_global", obs_size_global);

  /* Copy Q CSR arrays to device */
  {
    Kokkos::View<const PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>    hQ_i(Q_i, n_vertices + 1);
    Kokkos::View<const PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>    hQ_j(Q_j, n_vertices * p_local);
    Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hQ_v(Q_v, n_vertices * p_local);
    Kokkos::deep_copy(dQ_i, hQ_i);
    Kokkos::deep_copy(dQ_j, hQ_j);
    Kokkos::deep_copy(dQ_v, hQ_v);
  }

  /* Copy dense arrays to device (or wrap if already on device) */
  {
    Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hZ_global(Z_global_array, lda_z_global * m);
    Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hy_global(y_global_array, obs_size_global);
    Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hy_mean_global(y_mean_global_array, obs_size_global);
    Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hr_inv_sqrt_global(r_inv_sqrt_global_array, obs_size_global);
    Kokkos::deep_copy(dZ_global, hZ_global);
    Kokkos::deep_copy(dy_global, hy_global);
    Kokkos::deep_copy(dy_mean_global, hy_mean_global);
    Kokkos::deep_copy(dr_inv_sqrt_global, hr_inv_sqrt_global);
  }

  /* Phase 3+4 FUSED: Compute S directly from global arrays
     PHASE 4 FUSION: Eliminated ExtractLocalObservations + ComputeS kernels
     This eliminates 4 intermediate arrays: dZ_local_all, dy_local_all, dy_mean_local_all, dr_inv_sqrt_local_all
     S is computed directly from global arrays using Q matrix for indirection */

  PetscReal scale = 1.0 / PetscSqrtReal((PetscReal)(m - 1));

  /* Allocate device view for S matrices (all vertices) - still needed for SVD */
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dS_all("S_all", n_vertices, p_local * m);

  /* Allocate temporary arrays for local observations (needed by ComputeWeightsFused)
     These are much smaller than the original 4 arrays since they're only used in one kernel */
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dy_local_all("y_local_all", n_vertices, p_local);
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dy_mean_local_all("y_mean_local_all", n_vertices, p_local);
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dr_inv_sqrt_local_all("r_inv_sqrt_local_all", n_vertices, p_local);

  /* Compute S directly from global arrays with inline extraction
     For each vertex, extract local observations and immediately compute S */
  Kokkos::parallel_for(
    "ComputeSWithInlineExtraction", n_vertices, KOKKOS_LAMBDA(const int i_vertex) {
      /* For each vertex, extract its local observations using Q row */
      PetscInt row_start = dQ_i(i_vertex);
      PetscInt row_end   = dQ_i(i_vertex + 1);
      PetscInt ncols     = row_end - row_start;

      /* Extract and compute S in one pass */
      for (int k = 0; k < ncols; k++) {
        PetscInt    obs_idx = dQ_j(row_start + k);
        PetscScalar weight  = dQ_v(row_start + k);

        /* Extract observation values (needed for weight computation later) */
        PetscScalar y_local          = dy_global(obs_idx);
        PetscScalar y_mean_local     = dy_mean_global(obs_idx);
        PetscScalar r_inv_sqrt_local = dr_inv_sqrt_global(obs_idx) * PetscSqrtScalar(weight);

        /* Store for later use in ComputeWeightsFused */
        dy_local_all(i_vertex, k)          = y_local;
        dy_mean_local_all(i_vertex, k)     = y_mean_local;
        dr_inv_sqrt_local_all(i_vertex, k) = r_inv_sqrt_local;

        /* Compute S directly: S(k,j) = (Z(k,j) - y_mean(k)) * scale * r_inv_sqrt(k) */
        for (int j = 0; j < m; j++) {
          PetscScalar Z_local               = dZ_global(obs_idx + j * lda_z_global);
          dS_all(i_vertex, k + j * p_local) = (Z_local - y_mean_local) * scale * r_inv_sqrt_local;
        }
      }
    });

  Kokkos::fence();

  /* Phase 5: Batched SVD on GPU - compute SVD of all S matrices */
  /* Allocate device views for SVD outputs */
  PetscInt                                                     min_dim = (p_local < m) ? p_local : m;
  Kokkos::View<PetscScalar ***, Kokkos::DefaultExecutionSpace> dU_all("U_all", n_vertices, p_local, p_local);
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace>  dSigma_all("Sigma_all", n_vertices, min_dim);
  Kokkos::View<PetscScalar ***, Kokkos::DefaultExecutionSpace> dVt_all("Vt_all", n_vertices, m, m);

  /* Compute batched SVD for all vertices in parallel
     PHASE 2 FUSION: Eliminated ReshapeS kernel by creating temporary row-major copy inside SVD kernel
     This eliminates the dS_all_3d intermediate array and associated kernel launch */
  Kokkos::parallel_for(
    "BatchedSVDWithInlineReshape", n_vertices, KOKKOS_LAMBDA(const int i_vertex) {
      /* Create temporary row-major copy of S matrix for this vertex
         S is stored in dS_all as column-major: S(i,j) = dS_all(i_vertex, i + j*p_local)
         SVD needs row-major: S_temp(i,j) stored contiguously */
      PetscScalar S_temp[Q_NUM_LOCAL_OBSERVATIONS_MAX * ENSEMBLE_SIZE_MAX]; // [p_local x m]
      for (int i = 0; i < p_local; i++) {
        for (int j = 0; j < m; j++) S_temp[i * m + j] = dS_all(i_vertex, i + j * p_local);
      }

      /* Create unmanaged view wrapping S_temp with LayoutRight (row-major) */
      Kokkos::View<PetscScalar **, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> S_i(S_temp, p_local, m);

      /* Get subviews for this vertex's output matrices */
      auto U_i     = Kokkos::subview(dU_all, i_vertex, Kokkos::ALL, Kokkos::ALL);
      auto Sigma_i = Kokkos::subview(dSigma_all, i_vertex, Kokkos::ALL);
      auto Vt_i    = Kokkos::subview(dVt_all, i_vertex, Kokkos::ALL, Kokkos::ALL);

      /* Allocate workspace for SVD on stack (must be contiguous, not LayoutStride)
         Work size is max(p_local, m) as per Kokkos SVD requirements */
      PetscInt    max_dim = (p_local > m) ? p_local : m;
      PetscScalar Work_temp[(Q_NUM_LOCAL_OBSERVATIONS_MAX > ENSEMBLE_SIZE_MAX) ? Q_NUM_LOCAL_OBSERVATIONS_MAX : ENSEMBLE_SIZE_MAX];

      /* Create unmanaged contiguous view wrapping Work_temp - use default layout to ensure contiguity */
      Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Work_i(Work_temp, max_dim);

      /* Compute SVD: S = U * Sigma * V^T using Kokkos Kernels batched SVD */
      /* Use SVD_USV_Tag for full SVD with U, Sigma, and V^T */
      PetscScalar tol       = 1.0e-10;
      int         max_iters = 100;
      KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_USV_Tag(), S_i, U_i, Sigma_i, Vt_i, Work_i, tol, max_iters);
    });

  Kokkos::fence();

  /* Phase 6+7 FUSED: Compute weights and T_sqrt in single kernel
     PHASE 5 FUSION: Fused ComputeWeights + ComputeTSqrt kernels
     Both kernels consume SVD outputs and can share computation of (I + Sigma^2)^{-1/2}
     This eliminates 1 synchronization point and improves data locality */

  /* Get U matrix from PetscDA (needed for T_sqrt computation) */
  const PetscScalar *U_array = NULL;
  PetscMemType       U_memtype;
  PetscInt           lda_u;
  PetscCall(MatDenseGetArrayReadAndMemType(da->U, &U_array, &U_memtype));
  PetscCall(MatDenseGetLDA(da->U, &lda_u));

  /* Copy U to device */
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> dU_matrix("U_matrix", lda_u * m);
  {
    Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hU_matrix(U_array, lda_u * m);
    Kokkos::deep_copy(dU_matrix, hU_matrix);
  }

  /* Allocate device views for weights and T_sqrt (all vertices) */
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace>  dw_all("w_all", n_vertices, m);
  Kokkos::View<PetscScalar ***, Kokkos::DefaultExecutionSpace> dT_sqrt_all("T_sqrt_all", n_vertices, m, m);

  /* Compute both weights and T_sqrt in single fused kernel
     w = V (I + Sigma^2)^{-1} Sigma U^T delta
     T_sqrt = V (I + Sigma^2)^{-1/2} V^T U
     Both share computation of (I + Sigma^2)^{-1/2} */
  Kokkos::parallel_for(
    "ComputeWeightsAndTSqrtFused", n_vertices, KOKKOS_LAMBDA(const int i_vertex) {
      /* Shared computation: Compute (I + Sigma^2)^{-1/2} once for both w and T_sqrt */
      PetscScalar inv_sqrt_factor[40]; // max(min_dim) = min(40, m)
      for (int i = 0; i < min_dim; i++) {
        PetscScalar sigma    = dSigma_all(i_vertex, i);
        PetscScalar sigma_sq = sigma * sigma;
        inv_sqrt_factor[i]   = 1.0 / PetscSqrtScalar(1.0 + sigma_sq);
      }

      /* ===== Part 1: Compute weights w ===== */
      /* Step 1: Compute U^T delta (result is min_dim vector)
         Compute delta_scaled inline: delta_scaled[k] = (y_local[k] - y_mean_local[k]) * r_inv_sqrt_local[k] */
      PetscScalar UtDelta[40];
      for (int i = 0; i < min_dim; i++) {
        UtDelta[i] = 0.0;
        for (int k = 0; k < p_local; k++) {
          /* Compute delta_scaled on-the-fly */
          PetscScalar delta_scaled = (dy_local_all(i_vertex, k) - dy_mean_local_all(i_vertex, k)) * dr_inv_sqrt_local_all(i_vertex, k);
          UtDelta[i] += dU_all(i_vertex, k, i) * delta_scaled;
        }
      }

      /* Step 2: Compute (I + Sigma^2)^{-1} Sigma U^T delta (reuse inv_sqrt_factor) */
      PetscScalar temp_w[40];
      for (int i = 0; i < min_dim; i++) {
        PetscScalar sigma = dSigma_all(i_vertex, i);
        temp_w[i]         = (sigma * inv_sqrt_factor[i] * inv_sqrt_factor[i]) * UtDelta[i]; // (sigma / (1 + sigma^2)) = sigma * inv_sqrt^2
      }

      /* Step 3: Compute w = V temp_w (V is m x m, stored as Vt transposed) */
      for (int j = 0; j < m; j++) {
        dw_all(i_vertex, j) = 0.0;
        for (int i = 0; i < min_dim; i++) dw_all(i_vertex, j) += dVt_all(i_vertex, i, j) * temp_w[i];
      }

      /* ===== Part 2: Compute T_sqrt ===== */
      /* Step 1: Compute V^T U (result is m x m, but only first min_dim rows are non-zero) */
      PetscScalar VtU[40][50]; // [min_dim x m]
      for (int i = 0; i < min_dim; i++) {
        for (int j = 0; j < m; j++) {
          VtU[i][j] = 0.0;
          for (int k = 0; k < m; k++) VtU[i][j] += dVt_all(i_vertex, i, k) * dU_matrix(k + j * lda_u);
          /* Step 2: Scale by (I + Sigma^2)^{-1/2} (reuse inv_sqrt_factor) */
          VtU[i][j] *= inv_sqrt_factor[i];
        }
      }

      /* Step 3: Compute T_sqrt = V * scaled_VtU */
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          dT_sqrt_all(i_vertex, i, j) = 0.0;
          for (int k = 0; k < min_dim; k++) dT_sqrt_all(i_vertex, i, j) += dVt_all(i_vertex, k, i) * VtU[k][j];
        }
      }
    });

  Kokkos::fence();

  /* Phase 8+9 FUSED: Parallel ensemble update with inline G computation */
  /* E_a[i,:] = x_bar_f[i] + X_f[i,:] * G[i]
     where G[i,k,j] = w[i,k] + sqrt(m-1) * T_sqrt[i,k,j]
     PHASE 3 FUSION: Eliminated FormG kernel by computing G elements on-the-fly
     This eliminates the dG_all intermediate array (LARGEST intermediate!) */

  PetscReal sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));

  /* Copy X and mean to device */
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> dX("X", lda_x * m);
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> dmean("mean", n_vertices);
  {
    Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hX(X_array, lda_x * m);
    Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hmean(mean_array, n_vertices);
    Kokkos::deep_copy(dX, hX);
    Kokkos::deep_copy(dmean, hmean);
  }

  /* Allocate device view for ensemble output */
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> densemble("ensemble", lda_ensemble * m);

  /* Update ensemble with inline G computation: E_a[i,j] = mean[i] + sum_k X[i,k] * G[i,k,j]
     where G[i,k,j] is computed on-the-fly instead of being stored */
  Kokkos::parallel_for(
    "UpdateEnsembleWithInlineG", Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>({0, 0}, {n_vertices, m}), KOKKOS_LAMBDA(const int i_vertex, const int j) {
      PetscScalar sum = 0.0;
      for (int k = 0; k < m; k++) {
        /* Compute G[i_vertex,k,j] on-the-fly: G = w * 1' + sqrt(m-1) * T_sqrt */
        PetscScalar G_kj = dw_all(i_vertex, k) + sqrt_m_minus_1 * dT_sqrt_all(i_vertex, k, j);
        sum += dX(i_vertex + k * lda_x) * G_kj;
      }
      densemble(i_vertex + j * lda_ensemble) = dmean(i_vertex) + sum;
    });

  Kokkos::fence();

  /* Phase 10: Copy ensemble back to host and restore arrays */
  {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hensemble(ensemble_array, lda_ensemble * m);
    Kokkos::deep_copy(hensemble, densemble);
  }

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayReadAndMemType(da->U, &U_array));
  PetscCall(MatDenseRestoreArrayWriteAndMemType(da->ensemble, &ensemble_array));
  PetscCall(VecRestoreArrayReadAndMemType(impl->mean, &mean_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(X, &X_array));
  PetscCall(VecRestoreArrayReadAndMemType(r_inv_sqrt_global, &r_inv_sqrt_global_array));
  PetscCall(VecRestoreArrayReadAndMemType(y_mean_global, &y_mean_global_array));
  PetscCall(VecRestoreArrayReadAndMemType(observation, &y_global_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(Z_global, &Z_global_array));

  /* Finalize ensemble assembly */
  PetscCall(MatAssemblyBegin(da->ensemble, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(da->ensemble, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
