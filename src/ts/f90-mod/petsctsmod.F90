        module petsctsdef
        use petscsnesdef
#include "petsc/finclude/petscts.h"
#include <../src/ts/f90-mod/ftn-auto-interfaces/petscts.h>
        end module petsctsdef

        module petscts
        use petscsnes
        use petsctsdef

#include <../src/ts/f90-mod/petscts.h90>
#include <../src/ts/f90-mod/ftn-auto-interfaces/petscts.h90>
#include <../src/ts/f90-mod/ftn-auto-interfaces/petscsensitivity.h90>

!
!  Some PETSc Fortran functions that the user might pass as arguments
!
      external TSCOMPUTERHSFUNCTIONLINEAR
      external TSCOMPUTERHSJACOBIANCONSTANT
      external TSCOMPUTEIFUNCTIONLINEAR
      external TSCOMPUTEIJACOBIANCONSTANT

      contains

#include <../src/ts/f90-mod/ftn-auto-interfaces/petscts.hf90>

        end module

!     ----------------------------------------------

        module  petsccharacteristic
        use petscvec
#include <petsc/finclude/petsccharacteristic.h>
#include <../src/ts/f90-mod/ftn-auto-interfaces/petsccharacteristic.h>
#include <../src/ts/f90-mod/ftn-auto-interfaces/petsccharacteristic.h90>
        contains
#include <../src/ts/f90-mod/ftn-auto-interfaces/petsccharacteristic.hf90>
        end module

!     ----------------------------------------------

        module  petsclandau
        use petscts
#include <petsc/finclude/petsclandau.h>
#include <../src/ts/f90-mod/ftn-auto-interfaces/petsclandau.h>
#include <../src/ts/f90-mod/ftn-auto-interfaces/petsclandau.h90>
        end module

