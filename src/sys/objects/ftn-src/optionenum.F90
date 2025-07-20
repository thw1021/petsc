#include "petsc/finclude/petscsys.h"

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscOptionsGetEnum
!DEC$ ATTRIBUTES DLLEXPORT::PetscOptionsEnum
#endif

subroutine PetscOptionsGetEnum(po,pre,name,FArray,opt,set,ierr)
  use,intrinsic :: ISO_C_binding
  use petscsysdef
  implicit none

  character(*)                pre,name
  character(*)                FArray(*)
  PetscEnum, intent(out)      :: opt
  PetscBool, intent(out)      :: set
  PetscOptions                :: po
  PetscErrorCode,intent(out)  :: ierr

  type(C_Ptr),dimension(:),pointer :: CArray
  character(kind=c_char),pointer   :: nullc => null()
  PetscInt   :: i,Len
  character(kind=C_char,len=99),dimension(:),pointer::list1

  Len=0
  do i=1,100
    if (len_trim(Farray(i)) == 0) then
      Len = i-1
      exit
    endif
  enddo

  allocate(list1(Len),stat=ierr)
  if (ierr /= 0) return
  allocate(CArray(Len+1),stat=ierr)
  if (ierr /= 0) return
  do i=1,Len
      list1(i) = trim(FArray(i))//C_NULL_CHAR
      CArray(i) = c_loc(list1(i))
  enddo

  CArray(Len+1) = c_loc(nullc)
  call PetscOptionsGetEnumPrivate(po,pre,name,CArray,opt,set,ierr)
  deallocate(CArray)
  deallocate(list1)
end subroutine

subroutine PetscOptionsEnum(opt,text,man,Flist,curr,ivalue,set,ierr)
  use,intrinsic :: ISO_C_binding
  use petscsysdef
  implicit none

  character(*)                opt,text,man
  character(*)                Flist(*)
  PetscEnum                    :: curr
  PetscEnum, intent(out)       :: ivalue
  PetscBool, intent(out)       :: set
  PetscErrorCode,intent(out)   :: ierr

  type(C_Ptr),dimension(:),pointer :: CArray
  character(kind=c_char),pointer   :: nullc => null()
  PetscInt   :: i,Len
  character(kind=C_char,len=99),dimension(:),pointer::list1

  Len=0
  do i=1,100
    if (len_trim(Flist(i)) == 0) then
      Len = i-1
      exit
    endif
  enddo

  allocate(list1(Len),stat=ierr)
  if (ierr /= 0) return
  allocate(CArray(Len+1),stat=ierr)
  if (ierr /= 0) return
  do i=1,Len
      list1(i) = trim(Flist(i))//C_NULL_CHAR
      CArray(i) = c_loc(list1(i))
  enddo

  CArray(Len+1) = c_loc(nullc)
  call PetscOptionsEnumPrivate(opt,text,man,CArray,curr,ivalue,set,ierr)

  deallocate(CArray)
  deallocate(list1)
end subroutine PetscOptionsEnum
