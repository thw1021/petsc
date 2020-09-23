!
!  Test Fortran binding of sort routines
!
module UserModule
#include "petsc/finclude/petsc.h"
  use petsc
  type User
  end type User
end module UserModule

subroutine CompareIntegers(a,b,ctx,res)
  use UserModule
  implicit none

  PetscInt, intent(in)  :: a,b
  PetscInt, intent(out) :: res
  type(User)            :: ctx

  if (a .lt. b) then
     res = -1
  else if (a .eq. b) then
     res = 0
  else
     res = 1
  end if
end subroutine CompareIntegers

program main

  use UserModule
  implicit none

  PetscErrorCode          ierr
  PetscInt,parameter::    N=3
  PetscMPIInt,parameter:: mN=3
  PetscInt                x(N),x1(N),y(N),z(N)
  PetscMPIInt             mx(N),my(N),mz(N)
  PetscScalar             s(N)
  PetscReal               r(N)
  PetscMPIInt,parameter:: two=2, five=5, seven=7
  type(User)              ctx
  external                CompareIntegers

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  x  = [3, 2, 1]
  x1 = [3, 2, 1]
  y  = [6, 5, 4]
  z  = [3, 5, 2]
  mx = [five, seven, two]
  my = [five, seven, two]
  mz = [five, seven, two]
  s  = [1.0, 2.0, 3.0]
  r  = [1.0, 2.0, 3.0]

  call PetscSortInt(N,x,ierr)
  call PetscTimSort(N,x1,PetscSizeT,CompareIntegers,ctx,ierr)
  call PetscSortIntWithArray(N,y,x,ierr)
  call PetscSortIntWithArrayPair(N,x,y,z,ierr)

  call PetscSortMPIInt(N,mx,ierr)
  call PetscSortMPIIntWithArray(mN,mx,my,ierr)
  call PetscSortMPIIntWithIntArray(mN,mx,y,ierr)

  call PetscSortIntWithScalarArray(N,x,s,ierr)

  call PetscSortReal(N,r,ierr)
  call PetscSortRealWithArrayInt(N,r,x,ierr)

  call PetscFinalize(ierr)
end program main

!/*TEST
!
!   test:
!
!TEST*/
