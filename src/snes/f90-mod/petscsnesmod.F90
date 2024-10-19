        module petscsnesdef
        use petsckspdef

#include <../src/snes/f90-mod/ftn-auto-interfaces/petscall.h>
#include "petsc/finclude/petscconvest.h"
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscconvest.h>
        end module petscsnesdef

        module petscsnes
        use petscksp
        use petscsnesdef

#include <../src/snes/f90-mod/petscsnes.h90>
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscall.h90>
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscconvest.h90>

!  Some PETSc Fortran functions that the user might pass as arguments
!
      external SNESCOMPUTEJACOBIANDEFAULT
      external MATMFFDCOMPUTEJACOBIAN
      external SNESCOMPUTEJACOBIANDEFAULTCOLOR

      external SNESCONVERGEDDEFAULT
      external SNESCONVERGEDSKIP

        contains

#include <../src/snes/f90-mod/ftn-auto-interfaces/petscall.hf90>
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscconvest.hf90>

      end module
