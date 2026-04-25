/*
  Private header for PetscFEKokkosMaps lifecycle function declarations.

  These PETSC_EXTERN declarations live here (not in petscfekokkos.h) so that
  the Fortran binding generator -- which greps top-level include headers for
  PETSC_EXTERN -- does not create Fortran stubs for them.  petscfekokkos.h
  includes this file inside #if defined(PETSC_HAVE_KOKKOS), after
  PetscFEKokkosMaps is defined.

  This header is NOT standalone; do not include it directly.
*/
#pragma once

PETSC_EXTERN PetscErrorCode PetscFEKokkosCreateMaps(DM, PetscFEKokkosMaps *);
PETSC_EXTERN PetscErrorCode PetscFEKokkosStageMaps(PetscFEKokkosMaps *, DM);
PETSC_EXTERN PetscErrorCode PetscFEKokkosEnsureDynamicViews(PetscFEKokkosMaps *, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscFEKokkosPreallocateCOO(PetscFEKokkosMaps *, Mat);
PETSC_EXTERN PetscErrorCode PetscFEKokkosResetGeometry(PetscFEKokkosMaps *);
PETSC_EXTERN PetscErrorCode PetscFEKokkosMapsDestroy(PetscFEKokkosMaps **);
PETSC_EXTERN PetscErrorCode PetscFEKokkosSetUpGeometry(DM, PetscFEKokkosMaps *);
PETSC_EXTERN PetscErrorCode PetscFEKokkosSetUp(DM, PetscFEKokkosMaps *, Mat);
