#pragma once

#include <petscfekokkos.h>

/* Private declarations for PetscFEKokkosMaps lifecycle functions.
   These are C++ only (require Kokkos) and have no Fortran bindings.
   Only include this header from .kokkos.cxx translation units. */

PETSC_INTERN PetscErrorCode PetscFEKokkosMapsCreate(PetscFEKokkosMaps **);
PETSC_INTERN PetscErrorCode PetscFEKokkosCreateMaps(DM, PetscFEKokkosMaps *);
PETSC_INTERN PetscErrorCode PetscFEKokkosStageMaps(PetscFEKokkosMaps *, DM);
PETSC_INTERN PetscErrorCode PetscFEKokkosEnsureDynamicViews(PetscFEKokkosMaps *, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode PetscFEKokkosPreallocateCOO(PetscFEKokkosMaps *, Mat);
PETSC_INTERN PetscErrorCode PetscFEKokkosResetGeometry(PetscFEKokkosMaps *);
PETSC_INTERN PetscErrorCode PetscFEKokkosMapsDestroy(PetscFEKokkosMaps **);
PETSC_INTERN PetscErrorCode PetscFEKokkosSetUpGeometry(DM, PetscFEKokkosMaps *);
PETSC_INTERN PetscErrorCode PetscFEKokkosSetUp(DM, PetscFEKokkosMaps *, Mat);
