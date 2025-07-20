#include "petsc/finclude/petscbag.h"

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscBagRegisterEnum
#endif
      Subroutine PetscBagRegisterEnum(bag,addr,FArray,def,n,h,ierr)
      use,intrinsic :: iso_c_binding
      use petscbag
      implicit none

      PetscBag   bag
      character(*)                n,h
      character(*)                FArray(*)
      PetscEnum                   :: def
      PetscErrorCode,intent(out)  :: ierr
      PetscReal addr(*)

      type(C_Ptr),dimension(:),pointer :: CArray
      character(kind=c_char),pointer   :: nullc => null()
      PetscInt   :: i,length
      character(kind=C_char,len=256),dimension(:),pointer::list1

      do i=1,256
        if (len_trim(Farray(i)) == 0) then
          length = i-1
          goto 100
        endif
        if (len_trim(Farray(i)) > 255) then
          ierr = PETSC_ERR_ARG_OUTOFRANGE
          return
        endif
      enddo
      ierr = PETSC_ERR_ARG_OUTOFRANGE
      return

 100  continue

      allocate(list1(length),stat=ierr)
      if (ierr /= 0) return
      allocate(CArray(length+1),stat=ierr)
      if (ierr /= 0) return

      do i=1,length
         list1(i) = trim(FArray(i))//C_NULL_CHAR
         CArray(i) = c_loc(list1(i))
      enddo

      CArray(length+1) = c_loc(nullc)
      call PetscBagRegisterEnumPrivate(bag,addr,CArray,def,n,h,ierr)
      deallocate(CArray)
      deallocate(list1)
      end subroutine PetscBagRegisterEnum
