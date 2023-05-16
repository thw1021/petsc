!
!   Example of using PetscOptionsBegin in Fortran

#include "petsc/finclude/petsc.h"
      use petsc
      implicit none

      PetscErrorCode                            :: ierr
      PetscBool                                 :: set = PETSC_FALSE
      PetscInt                                  :: value = 2

      PetscCallA(PetscInitialize(ierr))

      PetscCall(PetscOptionsBegin(PETSC_COMM_WORLD,'prefix_','Setting options for my application','Section 1',ierr))
      PetscCall(PetscOptionsInt('-int','Get an application int','Man page',value,value,set,ierr))
      PetscCall(PetscOptionsEnd(ierr))
      if (set .eqv. PETSC_TRUE) then
         PetscCall(PetscPrintf(PETSC_COMM_WORLD,'The integer value was set\n',ierr))
      endif
      PetscCallA(PetscFinalize(ierr))
      end

!
!/*TEST
!
!   build:
!      requires: defined(PETSC_USING_F2003) defined(PETSC_USING_F90FREEFORM)
!
!   test:
!
!   test:
!      suffix: 2
!      args: -prefix_int 22
!
!TEST*/
