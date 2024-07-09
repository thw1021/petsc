        module petsckspdef
        use petscdmdef

#include "petsc/finclude/petscpc.h"
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscpc.h>
#include "petsc/finclude/petscksp.h"
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscksp.h>
        end module petsckspdef

!     ----------------------------------------------

        module petscksp
        use petscdm
        use petsckspdef
#include <../src/ksp/f90-mod/petscpc.h90>
#include <../src/ksp/f90-mod/petscksp.h90>
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscpc.h90>
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscksp.h90>

!   Possible arguments to KSPMonitorSet()
!
      external KSPCONVERGEDDEFAULT
      external KSPMONITORRESIDUAL
      external KSPMONITORTRUERESIDUAL
      external KSPMONITORSOLUTION
      external KSPMONITORSINGULARVALUE
      external KSPGMRESMONITORKRYLOV
      external KSPGMRESCLASSICALGRAMSCHMIDTORTHOGONALIZATION
      external KSPGMRESMODIFIEDGRAMSCHMIDTORTHOGONALIZATION

        contains

#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscpc.hf90>
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscksp.hf90>

!     deprecated API

        subroutine KSPGetResidualHistoryF90(ksp,r,na,ierr)
          KSP ksp
          PetscInt na
          PetscReal, pointer :: r(:)
          PetscErrorCode, intent(out) :: ierr
          call KSPGetResidualHistory(ksp,r,na,ierr)
        end subroutine

        end module petscksp

