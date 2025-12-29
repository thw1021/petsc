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

  /* Allocate device views for local data (all vertices) */
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dZ_local_all("Z_local_all", n_vertices, p_local * m);
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dy_local_all("y_local_all", n_vertices, p_local);
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dy_mean_local_all("y_mean_local_all", n_vertices, p_local);
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dr_inv_sqrt_local_all("r_inv_sqrt_local_all", n_vertices, p_local);

  /* Extract local observations for all vertices in parallel */
  Kokkos::parallel_for(
    "ExtractLocalObservations", n_vertices, KOKKOS_LAMBDA(const int i_vertex) {
      /* For each vertex, extract its local observations using Q row */
      PetscInt row_start = dQ_i(i_vertex);
      PetscInt row_end   = dQ_i(i_vertex + 1);
      PetscInt ncols     = row_end - row_start;

      /* Extract vectors */
      for (int k = 0; k < ncols; k++) {
        PetscInt    obs_idx = dQ_j(row_start + k);
        PetscScalar weight  = dQ_v(row_start + k);

        dy_local_all(i_vertex, k)          = dy_global(obs_idx);
        dy_mean_local_all(i_vertex, k)     = dy_mean_global(obs_idx);
        dr_inv_sqrt_local_all(i_vertex, k) = dr_inv_sqrt_global(obs_idx) * PetscSqrtScalar(weight);

        /* Extract Z matrix row (column-major layout) */
        for (int j = 0; j < m; j++) { dZ_local_all(i_vertex, k + j * p_local) = dZ_global(obs_idx + j * lda_z_global); }
      }
    });

  Kokkos::fence();

  /* Phase 4: Parallel S computation kernel - compute S for all vertices */
  PetscReal scale = 1.0 / PetscSqrtReal((PetscReal)(m - 1));

  /* Allocate device view for S matrices (all vertices) */
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dS_all("S_all", n_vertices, p_local * m);

  /* Compute normalized innovation matrix S for all vertices in parallel */
  Kokkos::parallel_for(
    "ComputeS", Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>({0, 0, 0}, {n_vertices, p_local, m}), KOKKOS_LAMBDA(const int i_vertex, const int i, const int j) {
      /* S[i_vertex](i,j) = (Z_local[i_vertex](i,j) - y_mean_local[i_vertex](i)) * scale * r_inv_sqrt_local[i_vertex](i) */
      dS_all(i_vertex, i + j * p_local) = (dZ_local_all(i_vertex, i + j * p_local) - dy_mean_local_all(i_vertex, i)) * scale * dr_inv_sqrt_local_all(i_vertex, i);
    });

  Kokkos::fence();

  /* Phase 5: Batched SVD on GPU - compute SVD of all S matrices */
  /* Allocate device views for SVD outputs */
  PetscInt                                                     min_dim = (p_local < m) ? p_local : m;
  Kokkos::View<PetscScalar ***, Kokkos::DefaultExecutionSpace> dU_all("U_all", n_vertices, p_local, p_local);
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace>  dSigma_all("Sigma_all", n_vertices, min_dim);
  Kokkos::View<PetscScalar ***, Kokkos::DefaultExecutionSpace> dVt_all("Vt_all", n_vertices, m, m);

  /* Reshape S_all to 3D for batched SVD: [n_vertices x p_local x m] */
  Kokkos::View<PetscScalar ***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> dS_all_3d("S_all_3d", n_vertices, p_local, m);

  /* Copy S from 2D to 3D layout */
  Kokkos::parallel_for(
    "ReshapeS", Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>({0, 0, 0}, {n_vertices, p_local, m}), KOKKOS_LAMBDA(const int i_vertex, const int i, const int j) { dS_all_3d(i_vertex, i, j) = dS_all(i_vertex, i + j * p_local); });

  /* Allocate workspace for SVD (size depends on algorithm) */
  PetscInt                                                    work_size = 5 * (p_local + m); // Conservative estimate for workspace
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dWork_all("Work_all", n_vertices, work_size);

  /* Compute batched SVD for all vertices in parallel */
  Kokkos::parallel_for(
    "BatchedSVD", n_vertices, KOKKOS_LAMBDA(const int i_vertex) {
      /* Get subviews for this vertex's matrices */
      auto S_i     = Kokkos::subview(dS_all_3d, i_vertex, Kokkos::ALL, Kokkos::ALL);
      auto U_i     = Kokkos::subview(dU_all, i_vertex, Kokkos::ALL, Kokkos::ALL);
      auto Sigma_i = Kokkos::subview(dSigma_all, i_vertex, Kokkos::ALL);
      auto Vt_i    = Kokkos::subview(dVt_all, i_vertex, Kokkos::ALL, Kokkos::ALL);
      auto Work_i  = Kokkos::subview(dWork_all, i_vertex, Kokkos::ALL);

      /* Compute SVD: S = U * Sigma * V^T using Kokkos Kernels batched SVD */
      /* Use SVD_USV_Tag for full SVD with U, Sigma, and V^T */
      PetscScalar tol       = 1.0e-10;
      int         max_iters = 100;
      KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_USV_Tag(), S_i, U_i, Sigma_i, Vt_i, Work_i, tol, max_iters);
    });

  Kokkos::fence();

  /* Phase 6: Parallel weight computation - compute w using SVD results */
  /* w = T^{-1} S^T delta where T^{-1} = V (I + Sigma^2)^{-1} V^T
     Given SVD: S = U Sigma V^T, we have:
     w = V (I + Sigma^2)^{-1} V^T V Sigma U^T delta = V (I + Sigma^2)^{-1} Sigma U^T delta */

  /* Allocate device views for delta_scaled (all vertices) */
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> ddelta_scaled_all("delta_scaled_all", n_vertices, p_local);

  /* Compute delta_scaled = y_local - y_mean_local (element-wise, already scaled by r_inv_sqrt) */
  Kokkos::parallel_for(
    "ComputeDeltaScaled", Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>({0, 0}, {n_vertices, p_local}),
    KOKKOS_LAMBDA(const int i_vertex, const int i) { ddelta_scaled_all(i_vertex, i) = (dy_local_all(i_vertex, i) - dy_mean_local_all(i_vertex, i)) * dr_inv_sqrt_local_all(i_vertex, i); });

  /* Allocate device views for weights (all vertices) */
  Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> dw_all("w_all", n_vertices, m);

  /* Compute weights: w = V (I + Sigma^2)^{-1} Sigma U^T delta */
  Kokkos::parallel_for(
    "ComputeWeights", n_vertices, KOKKOS_LAMBDA(const int i_vertex) {
      /* Step 1: Compute U^T delta (result is min_dim vector) */
      PetscScalar UtDelta[40]; // max(min_dim) = min(40, m)
      for (int i = 0; i < min_dim; i++) {
        UtDelta[i] = 0.0;
        for (int k = 0; k < p_local; k++) { UtDelta[i] += dU_all(i_vertex, k, i) * ddelta_scaled_all(i_vertex, k); }
      }

      /* Step 2: Compute (I + Sigma^2)^{-1} Sigma U^T delta */
      PetscScalar temp[40];
      for (int i = 0; i < min_dim; i++) {
        PetscScalar sigma    = dSigma_all(i_vertex, i);
        PetscScalar sigma_sq = sigma * sigma;
        temp[i]              = (sigma / (1.0 + sigma_sq)) * UtDelta[i];
      }

      /* Step 3: Compute w = V temp (V is m x m, stored as Vt transposed) */
      for (int j = 0; j < m; j++) {
        dw_all(i_vertex, j) = 0.0;
        for (int i = 0; i < min_dim; i++) { dw_all(i_vertex, j) += dVt_all(i_vertex, i, j) * temp[i]; }
      }
    });

  Kokkos::fence();

  /* Phase 7: Parallel T_sqrt computation - compute T^{-1/2} using SVD */
  /* T^{-1/2} = V (I + Sigma^2)^{-1/2} V^T
     We need to apply this to U (the random orthogonal matrix from PetscDA) */

  /* Get U matrix from PetscDA */
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

  /* Allocate device views for T_sqrt (all vertices) */
  Kokkos::View<PetscScalar ***, Kokkos::DefaultExecutionSpace> dT_sqrt_all("T_sqrt_all", n_vertices, m, m);

  /* Compute T_sqrt = T^{-1/2} U = V (I + Sigma^2)^{-1/2} V^T U */
  Kokkos::parallel_for(
    "ComputeTSqrt", n_vertices, KOKKOS_LAMBDA(const int i_vertex) {
      /* Step 1: Compute V^T U (result is m x m, but only first min_dim rows are non-zero) */
      PetscScalar VtU[40][50]; // [min_dim x m]
      for (int i = 0; i < min_dim; i++) {
        for (int j = 0; j < m; j++) {
          VtU[i][j] = 0.0;
          for (int k = 0; k < m; k++) { VtU[i][j] += dVt_all(i_vertex, i, k) * dU_matrix(k + j * lda_u); }
        }
      }

      /* Step 2: Scale by (I + Sigma^2)^{-1/2} */
      for (int i = 0; i < min_dim; i++) {
        PetscScalar sigma    = dSigma_all(i_vertex, i);
        PetscScalar sigma_sq = sigma * sigma;
        PetscScalar scale    = 1.0 / PetscSqrtScalar(1.0 + sigma_sq);
        for (int j = 0; j < m; j++) { VtU[i][j] *= scale; }
      }

      /* Step 3: Compute T_sqrt = V * scaled_VtU */
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          dT_sqrt_all(i_vertex, i, j) = 0.0;
          for (int k = 0; k < min_dim; k++) { dT_sqrt_all(i_vertex, i, j) += dVt_all(i_vertex, k, i) * VtU[k][j]; }
        }
      }
    });

  Kokkos::fence();

  /* Phase 8: Parallel G formation - form transform matrices */
  /* G = w * 1' + sqrt(m-1) * T_sqrt
     where w * 1' means adding w to each column of G */

  PetscReal sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));

  /* Allocate device views for G (all vertices) */
  Kokkos::View<PetscScalar ***, Kokkos::DefaultExecutionSpace> dG_all("G_all", n_vertices, m, m);

  /* Form G matrices */
  Kokkos::parallel_for(
    "FormG", Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>({0, 0, 0}, {n_vertices, m, m}),
    KOKKOS_LAMBDA(const int i_vertex, const int i, const int j) { dG_all(i_vertex, i, j) = dw_all(i_vertex, i) + sqrt_m_minus_1 * dT_sqrt_all(i_vertex, i, j); });

  Kokkos::fence();

  /* Phase 9: Parallel ensemble update - update all ensemble rows */
  /* E_a[i,:] = x_bar_f[i] + X_f[i,:] * G[i]
     where E_a is the analysis ensemble, x_bar_f is the forecast mean,
     X_f is the forecast anomaly, and G is the transform matrix */

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

  /* Update ensemble: E_a[i,j] = mean[i] + sum_k X[i,k] * G[i,k,j] */
  Kokkos::parallel_for(
    "UpdateEnsemble", Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>({0, 0}, {n_vertices, m}), KOKKOS_LAMBDA(const int i_vertex, const int j) {
      PetscScalar sum = 0.0;
      for (int k = 0; k < m; k++) { sum += dX(i_vertex + k * lda_x) * dG_all(i_vertex, k, j); }
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
