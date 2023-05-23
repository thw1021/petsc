#ifndef PETSCDEVICEBLAS_H
#define PETSCDEVICEBLAS_H

#include <petscmacros.h>
#include <petscdevice.h>

PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private(PetscDeviceContext, PetscMemType, PetscMemType, char, char, PetscInt, PetscInt, PetscInt, const PetscScalar *, const PetscScalar[], PetscInt, const PetscScalar[], PetscInt, const PetscScalar *, PetscScalar C[], PetscInt);

PETSC_INTERN PetscErrorCode PetscDeviceGEMV_Private(PetscDeviceContext, PetscMemType, PetscMemType, char, PetscInt, PetscInt, const PetscScalar *, const PetscScalar[], PetscInt, const PetscScalar[], PetscInt, const PetscScalar *, PetscScalar C[], PetscInt);

#endif
