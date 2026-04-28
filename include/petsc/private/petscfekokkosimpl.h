#pragma once

#include <petscfekokkos.h>

/* Private declarations for PetscFEKokkosMaps lifecycle functions.
   These are C++ only (require Kokkos) and have no Fortran bindings.
   Only include this header from .kokkos.cxx translation units.

   PETSC_EXTERN declarations for PetscFEKokkosMapsCreate, PetscFEKokkosMapsDestroy,
   PetscFEKokkosSetUp, and PetscFEKokkosEnsureDynamicViews live in the public header
   <petscfekokkos.h> (included above) and are not repeated here. */

PETSC_INTERN PetscErrorCode PetscFEKokkosCreateMaps(DM, PetscFEKokkosMaps *);
PETSC_INTERN PetscErrorCode PetscFEKokkosStageMaps(PetscFEKokkosMaps *, DM);
PETSC_INTERN PetscErrorCode PetscFEKokkosPreallocateCOO(PetscFEKokkosMaps *, Mat);
PETSC_INTERN PetscErrorCode PetscFEKokkosResetGeometry(PetscFEKokkosMaps *);
PETSC_INTERN PetscErrorCode PetscFEKokkosSetUpGeometry(DM, PetscFEKokkosMaps *);
