!
!
!   This program demonstrates use of PETSc dense matrices.
!
#include <petsc/finclude/petscmat.h>
module ex36fmodule
  use petscmat
  implicit none
! -----------------------------------------------------------------
!
!  Demo1 -  This subroutine demonstrates the use of PETSc-allocated dense
!  matrix storage.  Here MatDenseGetArray() is used for direct access to the
!  array that stores the dense matrix.
!
!  Note the use of PETSC_NULL_SCALAR_ARRAY in MatCreateSeqDense() to indicate that no
!  storage is being provided by the user. (Do NOT pass a zero in that
!  location.)
!
contains
  subroutine Demo1(m,n)

    PetscInt, intent(in) :: m, n
    Mat A
    PetscScalar, pointer :: aa(:, :)
    PetscErrorCode ierr

    ! Create matrix
    PetscCall(MatCreate(PETSC_COMM_SELF, A, ierr))
    PetscCall(MatSetSizes(A, m, n, m, n, ierr))
    PetscCall(MatSetType(A, MATSEQDENSE, ierr))
    PetscCall(MatSetUp(A, ierr))

    ! Access array storage
    PetscCall(MatDenseGetArray(A, aa, ierr))

    ! Set matrix values directly
    PetscCall(FillUpMatrix(m, n, aa))

    PetscCall(MatDenseRestoreArray(A, aa, ierr))

    ! Finalize matrix assembly
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

    ! View matrix
    PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF, ierr))

    ! Clean up
    PetscCall(MatDestroy(A, ierr))
  end subroutine Demo1

! -----------------------------------------------------------------
!
!  Demo2 -  This subroutine demonstrates the use of user-allocated dense
!  matrix storage.
!
  subroutine Demo2(m,n)

    PetscInt, intent(in) :: m, n
    Mat A
    PetscScalar, pointer :: aa(:, :)
    PetscErrorCode ierr

    allocate (aa(m, n))

    ! Create matrix
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, n, aa, A, ierr))

    ! Set matrix values directly
    PetscCall(FillUpMatrix(m, n, aa))

    ! Finalize matrix assembly
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

    ! View matrix
    PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF, ierr))

    ! Clean up
    PetscCall(MatDestroy(A, ierr))
  end subroutine Demo2

! -----------------------------------------------------------------

  subroutine FillUpMatrix(m, n, X)
    PetscInt, intent(in) :: m, n
    PetscScalar, intent(out) :: X(m, n)
    PetscInt i, j

    do j = 1, n
      do i = 1, m
        X(i, j) = 1.0/real(i + j - 1)
      end do
    end do
  end subroutine FillUpMatrix
end module ex36fmodule

program main
  use ex36fmodule
  implicit none

  PetscErrorCode ierr
  PetscInt, parameter :: m = 5, n = 4

  PetscCallA(PetscInitialize(ierr))

  ! Demo of PETSc-allocated dense matrix storage
  call Demo1(m,n)

  ! Demo of user-allocated dense matrix storage
  call Demo2(m,n)

  PetscCallA(PetscFinalize(ierr))
end program main

!/*TEST
!
!   test:
!      nsize: 1
!      output_file: output/ex36f.out
!
!TEST*/
