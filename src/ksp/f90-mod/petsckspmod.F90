        module petsckspdef
        use petscdmdef

#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscall.h>
        end module petsckspdef

!     ----------------------------------------------

        module petscksp
        use petscdm
        use petsckspdef

#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscall.h90>

        contains

#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscall.hf90>

!     deprecated API

        subroutine KSPGetResidualHistoryF90(ksp,r,na,ierr)
          KSP ksp
          PetscInt na
          PetscReal, pointer :: r(:)
          PetscErrorCode, intent(out) :: ierr
          call KSPGetResidualHistory(ksp,r,na,ierr)
        end subroutine

        end module petscksp

