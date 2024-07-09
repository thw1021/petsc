        module petscsnesdef
        use petsckspdef

#include "petsc/finclude/petscsnes.h"
#include "petsc/finclude/petscconvest.h"
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscsnes.h>
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscconvest.h>
        end module petscsnesdef

        module petscsnes
        use petscksp
        use petscsnesdef

#include <../src/snes/f90-mod/petscsnes.h90>
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscconvest.h90>
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscsnes.h90>
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscsnesfas.h90>

!  Some PETSc Fortran functions that the user might pass as arguments
!
      external SNESCOMPUTEJACOBIANDEFAULT
      external MATMFFDCOMPUTEJACOBIAN
      external SNESCOMPUTEJACOBIANDEFAULTCOLOR
      external SNESMONITORDEFAULT
      external SNESMONITORSOLUTION
      external SNESMONITORSOLUTIONUPDATE

      external SNESCONVERGEDDEFAULT
      external SNESCONVERGEDSKIPx

        contains

#include <../src/snes/f90-mod/ftn-auto-interfaces/petscsnes.hf90>
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscconvest.hf90>

!       deprecated API

        subroutine SNESGetConvergenceHistoryF90(snes,r,its,na,ierr)
          SNES snes
          PetscInt na
          PetscReal, pointer :: r(:)
          PetscInt, pointer :: its(:)
          PetscErrorCode, intent(out) :: ierr
          call SNESGetConvergenceHistory(snes,r,its,na,ierr)
        end subroutine

      end module
