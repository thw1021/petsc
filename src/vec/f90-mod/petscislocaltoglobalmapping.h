!
!  Used by petscvecmod.F90 to create Fortran module file
!
      type, extends(tPetscObject) :: tISLocalToGlobalMapping
      end type tISLocalToGlobalMapping

      ISLocalToGlobalMapping, parameter :: PETSC_NULL_ISLOCALTOGLOBALMAPPING = tISLocalToGlobalMapping(0)

