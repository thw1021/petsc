#pragma once

#include <petscfekokkos.h>

/* Private declarations for PetscFEKokkosMaps lifecycle functions.
   These are C++ only (require Kokkos) and have no Fortran bindings.
   Only include this header from .kokkos.cxx translation units. */

PETSC_EXTERN PetscErrorCode PetscFEKokkosMapsCreate(PetscFEKokkosMaps **);
PETSC_INTERN PetscErrorCode PetscFEKokkosCreateMaps(DM, PetscFEKokkosMaps *);
PETSC_INTERN PetscErrorCode PetscFEKokkosStageMaps(PetscFEKokkosMaps *, DM);
PETSC_EXTERN PetscErrorCode PetscFEKokkosEnsureDynamicViews(PetscFEKokkosMaps *, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode PetscFEKokkosPreallocateCOO(PetscFEKokkosMaps *, Mat);
PETSC_INTERN PetscErrorCode PetscFEKokkosResetGeometry(PetscFEKokkosMaps *);
PETSC_EXTERN PetscErrorCode PetscFEKokkosMapsDestroy(PetscFEKokkosMaps **);
PETSC_INTERN PetscErrorCode PetscFEKokkosSetUpGeometry(DM, PetscFEKokkosMaps *);
PETSC_EXTERN PetscErrorCode PetscFEKokkosSetUp(DM, PetscFEKokkosMaps *, Mat);
