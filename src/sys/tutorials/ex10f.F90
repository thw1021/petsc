! Demonstrates PetscViewerASCIIOpenWithFileUnit()

program main
#include <petsc/finclude/petscsys.h>
      use petscmpi  ! or mpi or mpi_f08
      use petscsys

      implicit none
      PetscErrorCode :: ierr
      PetscViewer    :: viewer

      ! Every PETSc program should begin with the PetscInitialize() routine.
      PetscCallA(PetscInitialize(ierr))

      PetscCall(PetscViewerASCIIOpenWithFileUnit(PETSC_COMM_WORLD,6,viewer,ierr))
      PetscCall(PetscOptionsView(PETSC_NULL_OPTIONS,viewer,ierr))
      PetscCall(PetscViewerDestroy(viewer,ierr))
      PetscCallA(PetscFinalize(ierr))
end program main

!/*TEST
!
!   test:
!     args: -options_view -options_left no
!
!TEST*/
