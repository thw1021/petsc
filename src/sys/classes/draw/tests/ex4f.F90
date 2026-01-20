!
!
!  This example demonstrates use of PetscDrawZoom()
!
!          This function is called repeatedly by PetscDrawZoom() to
!      redraw the figure
!
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdraw.h>
      subroutine zoomfunction(draw, dummy, ierr)
        use petscsys
        use petscdraw
        implicit none

        PetscReal value
        PetscReal, parameter :: max = 256.0
        PetscDraw draw
        integer dummy
        PetscErrorCode, intent(out) :: ierr
        PetscInt32 i

        do i = 0, 255
          value = real(i, PETSC_REAL_KIND)/max
          PetscCall(PetscDrawLine(draw, 0.0_PETSC_REAL_KIND, value, 1.0_PETSC_REAL_KIND, value, i, ierr))
        end do
        ierr = 0
      end

      program main
        use petscsys
        use petscdraw
        implicit none

        PetscDraw draw
        PetscErrorCode ierr
        integer4, parameter :: x = 0, y = 0, width = 256, height = 256
        external zoomfunction

        PetscCallA(PetscInitialize(ierr))
        PetscCallA(PetscDrawCreate(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, 'Title', x, y, width, height, draw, ierr))
        PetscCallA(PetscDrawSetFromOptions(draw, ierr))
        PetscCallA(PetscDrawZoom(draw, zoomfunction, PETSC_NULL_INTEGER, ierr))
        PetscCallA(PetscDrawDestroy(draw, ierr))
        PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   build:
!     requires: x
!
!   test:
!     output_file: output/empty.out
!
!TEST*/
