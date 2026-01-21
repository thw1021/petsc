#pragma once

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
  Mat      Q;            // Localization matrix (n_grid x n_observations_total) Each row has exactly n_obs_vertex non-zeros
  PetscInt n_obs_vertex; // = number of local observations per grid point (const now)
  PetscInt n_grid;       // Number of grid points (n_grid = state_size / da->ndof)
  PetscInt batch_size;   // Batch size for GPU processing
  /* Device pointers for Q (Kokkos views cast to void*) */
  void *Q_device_i;
  void *Q_device_j;
  void *Q_device_a;
} PetscDALETKFData;

#if defined(PETSC_HAVE_KOKKOS)
/* Function declarations */
PETSC_EXTERN PetscErrorCode PetscDALETKFLocalAnalysis(PetscDA, PetscDALETKFData *, PetscInt, PetscInt, Mat, Vec, Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode PetscDALETKFLocalAnalysis_GPU(PetscDA, PetscDALETKFData *, PetscInt, PetscInt, Mat, Vec, Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode PetscDALETKFSetupLocalization_Kokkos(PetscDALETKFData *);
PETSC_EXTERN PetscErrorCode PetscDALETKFDestroyLocalization_Kokkos(PetscDALETKFData *);
#endif
