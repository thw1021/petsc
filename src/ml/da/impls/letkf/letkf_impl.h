#ifndef PETSC_LETKF_IMPL_H
#define PETSC_LETKF_IMPL_H

#include "petscda.h"
#include <petsc/private/daimpl.h>

typedef struct {
  /* Persistent work vectors and matrices to avoid repeated allocation */
  Vec      mean;
  Vec      y_mean;
  Vec      delta_scaled;
  Vec      w;
  Vec      r_inv_sqrt;
  Mat      Z;
  Mat      S;
  Mat      T_sqrt;
  Mat      w_ones;
  Mat      Q;       // Localization matrix (n_grid x n_observations_total) Each row has exactly Q_NUM_LOCAL_OBSERVATIONS_MAX non-zeros
  PetscInt p_local; // = Q_NUM_LOCAL_OBSERVATIONS_MAX (number of local observations per grid point)
  PetscInt n_grid;  // Number of grid points (n_grid = state_size / da->ndof)
} PetscDALETKFData;

/* Function declarations */
PetscErrorCode ComputeNormalizedInnovationMatrix(Mat Z, Vec y_mean, Vec r_inv_sqrt, PetscInt m, PetscScalar scale, Mat S);
PetscErrorCode ExtractLocalObservations(Mat Q, PetscInt vertex_idx, Mat Z_global, Vec y_global, Vec y_mean_global, Vec r_inv_sqrt_global, PetscInt m, Mat Z_local, Vec y_local, Vec y_mean_local, Vec r_inv_sqrt_local, PetscInt *local_obs_indices);
PetscErrorCode PetscDALETKFLocalAnalysis(PetscDA da, PetscDALETKFData *impl, PetscInt m, PetscInt n_vertices, PetscScalar scale, PetscScalar sqrt_m_minus_1, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global);

#endif /* PETSC_LETKF_IMPL_H */
