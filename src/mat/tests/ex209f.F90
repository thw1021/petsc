!
!
!
#include <petsc/finclude/petscmat.h>
program main
  use petscmat
  implicit none

  Mat A
  PetscErrorCode ierr
  PetscScalar, pointer :: km(:, :)
  PetscInt, parameter :: three = 3, one = 1, idxm(1) = 0
  PetscInt i, j
  PetscScalar v(1)

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetSizes(A, three, three, three, three, ierr))
  PetscCallA(MatSetBlockSize(A, three, ierr))
  PetscCallA(MatSetType(A, MATSEQBAIJ, ierr))
  PetscCallA(MatSetUp(A, ierr))

  allocate (km(three, three))
  do i = 1, 3
    do j = 1, 3
      km(i, j) = i + j
    end do
  end do

  PetscCallA(MatSetValuesBlocked(A, one, idxm, one, idxm, reshape(km, [three**2]), ADD_VALUES, ierr))
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatView(A, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatGetValues(A, one, [0_PETSC_INT_KIND], one, [0_PETSC_INT_KIND], v, ierr))

  PetscCallA(MatDestroy(A, ierr))

  deallocate (km)
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!     test:
!       requires: double !complex
!
!TEST*/
