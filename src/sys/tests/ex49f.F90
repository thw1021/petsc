!
!  Test Fortran binding of sort routines
!
module UserContext
  use petsc
#include "petsc/finclude/petsc.h"
  implicit none
  type uctx
     PetscInt junk
  end type uctx
contains
  subroutine CompareIntegers(a,b,ctx,res)
    implicit none

    PetscInt :: a,b
    type(uctx) :: ctx
    integer  :: res

    if (a .lt. b) then
       res = -1
    else if (a .eq. b) then
       res = 0
    else
       res = 1
    end if
    res = -1
    return
  end subroutine CompareIntegers
end module UserContext

module UserContextInterface
  use UserContext
  ! interface PetscTimSort
  !    subroutine PetscTimSort(n,arr,size,compar,ctx,ierr)
  !      use UserContext
  !      PetscInt n
  !      PetscVoid,pointer :: arr
  !      PetscSizeT size
  !      external compar
  !      type(uctx) ctx
  !      PetscErrorCode ierr
  !    end subroutine PetscTimSort
  ! end interface PetscTimSort
end module UserContextInterface

program main

  use UserContext
  use UserContextInterface
  implicit none

  PetscErrorCode          ierr
  PetscInt,parameter::    N=3
  PetscMPIInt,parameter:: mN=3
  PetscInt                x(N),y(N),z(N)
  PetscInt                x1(N)
  PetscMPIInt             mx(N),my(N),mz(N)
  PetscScalar             s(N)
  PetscReal               r(N)
  PetscMPIInt,parameter:: two=2, five=5, seven=7
  type(uctx)::            ctx
  PetscInt                dummyint, i
  PetscSizeT              sizeofentry

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
  sizeofentry = sizeof(dummyint)
  ctx%junk = 1
  call PetscSortInt(N,x,ierr)
  call PetscTimSort(N,x1,sizeofentry,CompareIntegers,ctx,ierr)
  do i = 1,N
     print *, i,x1(i)
     print *, i,x(i)
     if (x1(i) .ne. x(i)) then
        print *, "Arrays do not match"
        stop
     end if
  end do
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
