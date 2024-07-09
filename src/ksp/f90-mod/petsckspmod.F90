#include "petsc/finclude/petscpc.h"
#include "petsc/finclude/petscksp.h"
        module petscksp
        use petsckspdef
        use petscpc
#include <../src/ksp/f90-mod/petscksp.h90>
        interface
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscksp.h90>
        end interface

        contains

!     deprecated API

        subroutine KSPGetResidualHistoryF90(ksp,r,na,ierr)
          KSP ksp
          PetscInt na
          PetscReal, pointer :: r(:)
          PetscErrorCode, intent(out) :: ierr
          call KSPGetResidualHistory(ksp,r,na,ierr)
        end subroutine

        end module petscksp
