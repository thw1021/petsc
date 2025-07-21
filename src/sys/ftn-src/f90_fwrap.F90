!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#include <petsc/finclude/petscsys.h>
      subroutine F90Array1dCreateScalar(array,start,len1,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start,len1
      PetscScalar, target :: array(start:start+len1-1)
      PetscScalar, pointer :: ptr(:)

      ptr => array
      end subroutine F90Array1dCreateScalar

      subroutine F90Array1dCreateReal(array,start,len1,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start,len1
      PetscReal, target :: array(start:start+len1-1)
      PetscReal, pointer :: ptr(:)

      ptr => array
      end subroutine F90Array1dCreateReal

      subroutine F90Array1dCreateInt(array,start,len1,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start,len1
      PetscInt, target :: array(start:start+len1-1)
      PetscInt, pointer :: ptr(:)

      ptr => array
      end subroutine F90Array1dCreateInt

      subroutine F90Array1dCreateMPIInt(array,start,len1,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start,len1
      PetscMPIInt, target :: array(start:start+len1-1)
      PetscMPIInt, pointer :: ptr(:)

      ptr => array
      end subroutine F90Array1dCreateMPIInt

      subroutine F90Array1dCreateFortranAddr(array,start,len1,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start,len1
      PetscFortranAddr, target :: array(start:start+len1-1)
      PetscFortranAddr, pointer :: ptr(:)

      ptr => array
      end subroutine F90Array1dCreateFortranAddr

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array1dAccessScalar(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start

      if (.not. associated(ptr)) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrScalar(ptr(start),address)
      endif
      end subroutine F90Array1dAccessScalar

      subroutine F90Array1dAccessReal(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscReal, pointer :: ptr(:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start

      if (.not. associated(ptr)) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrReal(ptr(start),address)
      endif
      end subroutine F90Array1dAccessReal

      subroutine F90Array1dAccessInt(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, pointer :: ptr(:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start

      if (.not. associated(ptr)) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrInt(ptr(start),address)
      endif
      end subroutine F90Array1dAccessInt

      subroutine F90Array1dAccessMPIInt(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscMPIInt, pointer :: ptr(:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start

      if (.not. associated(ptr)) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrMPIInt(ptr(start),address)
      endif
      end subroutine F90Array1dAccessMPIInt

      subroutine F90Array1dAccessFortranAddr(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscFortranAddr, pointer :: ptr(:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start

      if (.not. associated(ptr)) then
        address = 0
      else
        start = lbound(ptr,1)
        call F90Array1dGetAddrFortranAddr(ptr(start),address)
      endif
      end subroutine F90Array1dAccessFortranAddr

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array1dDestroyScalar(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:)

      nullify(ptr)
      end subroutine F90Array1dDestroyScalar

      subroutine F90Array1dDestroyReal(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscReal, pointer :: ptr(:)

      nullify(ptr)
      end subroutine F90Array1dDestroyReal

      subroutine F90Array1dDestroyInt(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, pointer :: ptr(:)

      nullify(ptr)
      end subroutine F90Array1dDestroyInt

      subroutine F90Array1dDestroyMPIInt(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscMPIInt, pointer :: ptr(:)

      nullify(ptr)
      end subroutine F90Array1dDestroyMPIInt

      subroutine F90Array1dDestroyFortranAddr(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscFortranAddr, pointer :: ptr(:)

      nullify(ptr)
      end subroutine F90Array1dDestroyFortranAddr
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array2dCreateScalar(array,start1,len1,start2,len2,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscScalar, target :: array(start1:start1+len1-1,start2:start2+len2-1)
      PetscScalar, pointer :: ptr(:,:)

      ptr => array
      end subroutine F90Array2dCreateScalar

      subroutine F90Array2dCreateReal(array,start1,len1,start2,len2,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscReal, target :: array(start1:start1+len1-1,start2:start2+len2-1)
      PetscReal, pointer :: ptr(:,:)

      ptr => array
      end subroutine F90Array2dCreateReal

      subroutine F90Array2dCreateInt(array,start1,len1,start2,len2,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, target :: array(start1:start1+len1-1,start2:start2+len2-1)
      PetscInt, pointer :: ptr(:,:)

      ptr => array
      end subroutine F90Array2dCreateInt

      subroutine F90Array2dCreateFortranAddr(array,start1,len1,start2,len2,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscFortranAddr, target :: array(start1:start1+len1-1,start2:start2+len2-1)
      PetscFortranAddr, pointer :: ptr(:,:)

      ptr => array
      end subroutine F90Array2dCreateFortranAddr

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array2dAccessScalar(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrScalar(ptr(start1,start2),address)
      end subroutine F90Array2dAccessScalar

      subroutine F90Array2dAccessReal(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscReal, pointer :: ptr(:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrReal(ptr(start1,start2),address)
      end subroutine F90Array2dAccessReal

      subroutine F90Array2dAccessInt(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, pointer :: ptr(:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrInt(ptr(start1,start2),address)
      end subroutine F90Array2dAccessInt

      subroutine F90Array2dAccessFortranAddr(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start1,start2

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      call F90Array2dGetAddrFortranAddr(ptr(start1,start2),address)
      end subroutine F90Array2dAccessFortranAddr

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array2dDestroyScalar(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine F90Array2dDestroyScalar

      subroutine F90Array2dDestroyReal(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscReal, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine F90Array2dDestroyReal

      subroutine F90Array2dDestroyInt(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine F90Array2dDestroyInt

      subroutine F90Array2dDestroyFortranAddr(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:)

      nullify(ptr)
      end subroutine F90Array2dDestroyFortranAddr
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array3dCreateScalar(array,start1,len1,start2,len2,start3,len3,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, intent(in) :: start3,len3
      PetscScalar, target ::                                                      &
           array(start1:start1+len1-1,start2:start2+len2-1,start3:start3+len3-1)
      PetscScalar, pointer :: ptr(:,:,:)

      ptr => array
      end subroutine F90Array3dCreateScalar

      subroutine F90Array3dCreateReal(array,start1,len1,start2,len2,start3,len3,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, intent(in) :: start3,len3
      PetscReal, target ::                                                        &
           array(start1:start1+len1-1,start2:start2+len2-1,start3:start3+len3-1)
      PetscReal, pointer :: ptr(:,:,:)

      ptr => array
      end subroutine F90Array3dCreateReal

      subroutine F90Array3dCreateInt(array,start1,len1,start2,len2,start3,len3,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, intent(in) :: start3,len3
      PetscInt, target ::                                                         &
           array(start1:start1+len1-1,start2:start2+len2-1,start3:start3+len3-1)
      PetscInt, pointer :: ptr(:,:,:)

      ptr => array
      end subroutine F90Array3dCreateInt

      subroutine F90Array3dCreateFortranAddr(array,start1,len1,start2,len2,start3,len3,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, intent(in) :: start3,len3
      PetscFortranAddr, target ::                                                 &
           array(start1:start1+len1-1,start2:start2+len2-1,start3:start3+len3-1)
      PetscFortranAddr, pointer :: ptr(:,:,:)

      ptr => array
      end subroutine F90Array3dCreateFortranAddr

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array3dAccessScalar(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:,:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start1,start2,start3

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      call F90Array3dGetAddrScalar(ptr(start1,start2,start3),address)
      end subroutine F90Array3dAccessScalar

      subroutine F90Array3dAccessReal(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscReal, pointer :: ptr(:,:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start1,start2,start3

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      call F90Array3dGetAddrReal(ptr(start1,start2,start3),address)
      end subroutine F90Array3dAccessReal

      subroutine F90Array3dAccessInt(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, pointer :: ptr(:,:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start1,start2,start3

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      call F90Array3dGetAddrInt(ptr(start1,start2,start3),address)
      end subroutine F90Array3dAccessInt

      subroutine F90Array3dAccessFortranAddr(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt start1,start2,start3

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      call F90Array3dGetAddrFortranAddr(ptr(start1,start2,start3),address)
      end subroutine F90Array3dAccessFortranAddr

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array3dDestroyScalar(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:,:,:)

      nullify(ptr)
      end subroutine F90Array3dDestroyScalar

      subroutine F90Array3dDestroyReal(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscReal, pointer :: ptr(:,:,:)

      nullify(ptr)
      end subroutine F90Array3dDestroyReal

      subroutine F90Array3dDestroyInt(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, pointer :: ptr(:,:,:)

      nullify(ptr)
      end subroutine F90Array3dDestroyInt

      subroutine F90Array3dDestroyFortranAddr(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:,:)

      nullify(ptr)
      end subroutine F90Array3dDestroyFortranAddr

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine F90Array4dCreateScalar(array,start1,len1,start2,len2,start3,len3,start4,len4,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, intent(in) :: start3,len3
      PetscInt, intent(in) :: start4,len4
      PetscScalar, target ::                                                      &
           array(start1:start1+len1-1,start2:start2+len2-1,                       &
                 start3:start3+len3-1,start4:start4+len4-1)
      PetscScalar, pointer :: ptr(:,:,:,:)

      ptr => array
      end subroutine F90Array4dCreateScalar

      subroutine F90Array4dCreateReal(array,start1,len1,start2,len2,start3,len3,start4,len4,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, intent(in) :: start3,len3
      PetscInt, intent(in) :: start4,len4
      PetscReal, target ::                                                        &
           array(start1:start1+len1-1,start2:start2+len2-1,                       &
                 start3:start3+len3-1,start4:start4+len4-1)
      PetscReal, pointer :: ptr(:,:,:,:)

      ptr => array
      end subroutine F90Array4dCreateReal

      subroutine F90Array4dCreateInt(array,start1,len1,start2,len2,start3,len3,start4,len4,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, intent(in) :: start3,len3
      PetscInt, intent(in) :: start4,len4
      PetscInt, target ::                                                         &
           array(start1:start1+len1-1,start2:start2+len2-1,                       &
                 start3:start3+len3-1,start4:start4+len4-1)
      PetscInt, pointer :: ptr(:,:,:,:)

      ptr => array
      end subroutine F90Array4dCreateInt

      subroutine F90Array4dCreateFortranAddr(array,start1,len1,start2,len2,start3,len3,start4,len4,ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, intent(in) :: start1,len1
      PetscInt, intent(in) :: start2,len2
      PetscInt, intent(in) :: start3,len3
      PetscInt, intent(in) :: start4,len4
      PetscFortranAddr, target ::                                                 &
           array(start1:start1+len1-1,start2:start2+len2-1,                       &
                 start3:start3+len3-1,start4:start4+len4-1)
      PetscFortranAddr, pointer :: ptr(:,:,:,:)

      ptr => array
      end subroutine F90Array4dCreateFortranAddr

      subroutine F90Array4dAccessScalar(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:,:,:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt :: start1,start2,start3,start4

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      start4 = lbound(ptr,4)
      call F90Array4dGetAddrScalar(ptr(start1,start2,start3,start4),address)
      end subroutine F90Array4dAccessScalar

      subroutine F90Array4dAccessReal(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscReal, pointer :: ptr(:,:,:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt :: start1,start2,start3,start4

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      start4 = lbound(ptr,4)
      call F90Array4dGetAddrReal(ptr(start1,start2,start3,start4),address)
      end subroutine F90Array4dAccessReal

      subroutine F90Array4dAccessInt(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, pointer :: ptr(:,:,:,:)
      PetscFortranAddr, intent(out) :: address
      PetscInt :: start1,start2,start3,start4

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      start4 = lbound(ptr,4)
      call F90Array4dGetAddrInt(ptr(start1,start2,start3,start4),address)
      end subroutine F90Array4dAccessInt

      subroutine F90Array4dAccessFortranAddr(ptr,address)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:,:,:,:)
      PetscFortranAddr, intent(out) :: address
      PetscFortranAddr :: start1,start2,start3,start4

      start1 = lbound(ptr,1)
      start2 = lbound(ptr,2)
      start3 = lbound(ptr,3)
      start4 = lbound(ptr,4)
      call F90Array4dGetAddrFortranAddr(ptr(start1,start2,start3,start4),address)
      end subroutine F90Array4dAccessFortranAddr

      subroutine F90Array4dDestroyScalar(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscScalar, pointer :: ptr(:,:,:,:)

      nullify(ptr)
      end subroutine F90Array4dDestroyScalar

      subroutine F90Array4dDestroyReal(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscReal, pointer :: ptr(:,:,:,:)

      nullify(ptr)
      end subroutine F90Array4dDestroyReal

      subroutine F90Array4dDestroyInt(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscInt, pointer :: ptr(:,:,:,:)

      nullify(ptr)
      end subroutine F90Array4dDestroyInt

      subroutine F90Array4dDestroyFortranAddr(ptr)
      use, intrinsic :: ISO_C_binding
      implicit none
      PetscFortranAddr, pointer :: ptr(:,:,:,:)

      nullify(ptr)
      end subroutine F90Array4dDestroyFortranAddr

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
