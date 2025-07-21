! Demonstrates PetscViewerASCIIOpenWithFileUnit()

#include <petsc/finclude/petscsys.h>
program ex10f
      use petscmpi  ! or mpi or mpi_f08
      use petscsys

      implicit none
      PetscErrorCode :: ierr
      PetscViewer    :: viewer
      integer, parameter :: unit = 6

      ! Every PETSc program should begin with the PetscInitialize() routine.
      PetscCallA(PetscInitialize(ierr))

      PetscCallA(PetscViewerASCIIOpenWithFileUnit(PETSC_COMM_WORLD,unit,viewer,ierr))
      PetscCallA(PetscOptionsView(PETSC_NULL_OPTIONS,viewer,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
      PetscCallA(PetscFinalize(ierr))
end program ex10f

!/*TEST
!
!   test:
!     args: -options_view -options_left no
!
!TEST*/
