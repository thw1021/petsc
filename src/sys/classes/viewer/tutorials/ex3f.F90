      program ex1f90

#include <petsc/finclude/petscsys.h>
      use petscsys
      implicit none

      PetscErrorCode ierr
      PetscCallA(PetscInitialize(ierr))

      call PetscViewerASCIIStdoutSetFileUnit(22,ierr)
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!     test:
!       requires: defined(PETSC_USE_LOG)
!       args: -log_view
!
!TEST*/
