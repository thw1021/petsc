!
!
!  This example demonstrates use of PetscDrawZoom()
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdraw.h>
      module ex4f_mod
      use petscsys
      use petscdraw
      implicit none
      contains
!
!      This function is called repeatedly by PetscDrawZoom() to
!      redraw the figure
!
      subroutine zoomfunction(draw,dummy,ierr)

      PetscReal, parameter :: zero = 0., one = 1., max = 256.0
      PetscReal value
      PetscDraw draw
      integer dummy
      PetscErrorCode ierr

      PetscInt32 i

      do i=0,255
        value = i/max
        PetscCall(PetscDrawLine(draw,zero,value,one,value,i,ierr))
      end do
      end subroutine zoomfunction
      end module ex4f_mod

      program ex4f
      use petscsys
      use petscdraw
      use ex4f_mod
      implicit none

      PetscDraw draw
      PetscErrorCode ierr
      integer4, parameter :: x = 0, y = 0, width = 256, height = 256

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,'Title',x,y,width,height,draw,ierr))
      PetscCallA(PetscDrawSetFromOptions(draw,ierr))
      PetscCallA(PetscDrawZoom(draw,zoomfunction,PETSC_NULL_INTEGER,ierr))
      PetscCallA(PetscDrawDestroy(draw,ierr))
      PetscCallA(PetscFinalize(ierr))

      end program ex4f

!/*TEST
!
!   build:
!     requires: x
!
!   test:
!     output_file: output/empty.out
!
!TEST*/
