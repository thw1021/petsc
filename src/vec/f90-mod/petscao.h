!
!  Used by petscvecmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscao.h"

!  cannot use tAO because that type matches the variable tao used in tao examples
      type, extends(tPetscObject) :: tPetscAO
      end type tPetscAO
      AO, parameter :: PETSC_NULL_AO = tPetscAO(0)

