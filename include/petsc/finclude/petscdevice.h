#ifndef PETSC_FINCLUDE_PETSCDEVICE_H
#define PETSC_FINCLUDE_PETSCDEVICE_H
!
! Include file for Fortran use of the PetscDevice package in PETSc
!

#include "petsc/finclude/petscsys.h"

! PetscDevice
#define PetscDevice type(tPetscDevice)

#define PetscDeviceInitType  PetscEnum
#define PetscDeviceType      PetscEnum
#define PetscDeviceAttribute PetscEnum

! PetscDeviceContext
#define PetscDeviceContext type(tPetscDeviceContext)

#define PetscStreamType            PetscEnum
#define PetscDeviceContextJoinMode PetscEnum
#define PetscDeviceCopyMode        PetscEnum
#define PetscMemoryAccessMode      PetscEnum

#endif /* PETSC_FINCLUDE_PETSCDEVICE_H */
