#pragma once

/*
  petscfekokkosmaps_impl.h -- Private declarations for PetscFEKokkosMaps lifecycle functions.

  This header is intentionally placed in include/petsc/private/ so that the PETSc
  Fortran binding generator (getAPI.py / generatefortranbindings.py) does NOT scan it.
  The generator only greps top-level include headers for PETSC_EXTERN PetscErrorCode declarations.
  These functions are Kokkos-only and have no Fortran bindings.

  This header is included at the end of petscfekokkos.h (inside #if PETSC_HAVE_KOKKOS),
  after PetscFEKokkosMaps is fully defined.
*/

#if defined(PETSC_HAVE_KOKKOS)

PETSC_EXTERN PetscErrorCode PetscFEKokkosCreateMaps(DM dm, PetscFEKokkosMaps *maps);
PETSC_EXTERN PetscErrorCode PetscFEKokkosStageMaps(PetscFEKokkosMaps *maps, DM dm);
PETSC_EXTERN PetscErrorCode PetscFEKokkosEnsureDynamicViews(PetscFEKokkosMaps *maps, PetscInt Ne, PetscInt Nq, PetscInt dE, PetscInt totDim, PetscInt numConstants);
PETSC_EXTERN PetscErrorCode PetscFEKokkosPreallocateCOO(PetscFEKokkosMaps *maps, Mat J);
PETSC_EXTERN PetscErrorCode PetscFEKokkosResetGeometry(PetscFEKokkosMaps *maps);
PETSC_EXTERN PetscErrorCode PetscFEKokkosMapsDestroy(PetscFEKokkosMaps *maps);
PETSC_EXTERN PetscErrorCode PetscFEKokkosSetUpGeometry(DM dm, PetscFEKokkosMaps *maps);
PETSC_EXTERN PetscErrorCode PetscFEKokkosSetUp(DM dm, PetscFEKokkosMaps *maps, Mat J);

#endif /* PETSC_HAVE_KOKKOS */
