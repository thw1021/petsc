!!! This program Test of SNESVICreateActiveSetIS and SNESVIGetInactiveSetIS

#include <petsc/finclude/petscsnes.h>
module ex22f_module
  use petscsnes
  implicit none
contains
  subroutine computeFunction(snes, X, F, ctx, ierr)
    SNES           :: snes
    Vec            :: X, F
    PetscInt       :: ctx
    PetscErrorCode :: ierr

    PetscInt       :: n, i
    PetscReal, pointer :: XX(:), FF(:)

    PetscCall(VecGetSize(X, n, ierr))
    PetscCall(VecGetArrayRead(X, XX, ierr))
    PetscCall(VecGetArrayWrite(F, FF, ierr))
    FF = XX - [(i, i=0, n - 1)]
    PetscCall(VecRestoreArrayWrite(F, FF, ierr))
    PetscCall(VecRestoreArrayRead(X, XX, ierr))
  end subroutine computeFunction

  subroutine computeJacobian(snes, X, J, P, ctx, ierr)
    SNES           :: snes
    Vec            :: X
    Mat            :: J, P
    PetscInt       :: ctx
    PetscErrorCode :: ierr

    PetscInt       :: n, i
    PetscReal      :: one = 1.0

    PetscCall(MatGetSize(J, n, PETSC_NULL_INTEGER, ierr))
    do i = 0, n - 1
      PetscCall(MatSetValue(J, i, i, one, INSERT_VALUES, ierr))
    end do
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY, ierr))
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY, ierr))
  end subroutine computeJacobian
end module ex22f_module

program ex22f
  use ex22f_module
  implicit none

  SNES           :: snes
  Vec            :: X, F, Xl, Xu
  Mat            :: A
  PetscReal      :: lb = -1, ub = 6
  PetscInt       :: n = 5
  IS             :: iA
  PetscReal      :: zero = 0.0
  PetscErrorCode :: ierr

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', N, PETSC_NULL_BOOL, ierr))
  PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-lb', lb, PETSC_NULL_BOOL, ierr))
  PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-ub', ub, PETSC_NULL_BOOL, ierr))

  PetscCall(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n, ierr))
  PetscCall(MatSetType(A, MATAIJ, ierr))
  PetscCall(MatSetUp(A, ierr))
  PetscCall(MatSetFromOptions(A, ierr))

  PetscCallA(VecCreate(PETSC_COMM_WORLD, X, ierr))
  PetscCallA(VecSetSizes(X, PETSC_DECIDE, n, ierr))
  PetscCallA(VecSetType(X, VECMPI, ierr))
  PetscCallA(VecSet(X, zero, ierr))
  PetscCallA(PetscObjectSetName(X, "X", ierr))
  PetscCallA(VecDuplicate(X, F, ierr))

  PetscCallA(SNESCreate(PETSC_COMM_WORLD, snes, ierr))
  PetscCallA(SNESSetType(snes, "vinewtonrsls", ierr))
  PetscCallA(SNESSetFunction(snes, F, computeFunction, PETSC_NULL_INTEGER, ierr))
  PetscCallA(SNESSetJacobian(snes, A, A, computeJacobian, PETSC_NULL_INTEGER, ierr))
  PetscCallA(SNESSetFromOptions(snes, ierr))

  PetscCallA(VecDuplicate(X, Xl, ierr))
  PetscCallA(VecDuplicate(X, Xu, ierr))
  PetscCallA(VecSet(Xl, lb, ierr))
  PetscCallA(VecSet(Xu, ub, ierr))
  PetscCallA(SNESVISetVariableBounds(snes, Xl, Xu, ierr))

  PetscCallA(SNESSolve(snes, PETSC_NULL_VEC, X, ierr))
  PetscCallA(VecView(X, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(SNESVICreateActiveSetIS(snes, X, F, iA, ierr))
  PetscCallA(ISView(iA, PETSC_VIEWER_STDOUT_SELF, ierr))
  PetscCallA(ISDestroy(iA, ierr))

  PetscCallA(SNESVIGetInactiveSet(snes, iA, ierr))
  PetscCallA(ISView(iA, PETSC_VIEWER_STDOUT_SELF, ierr))

  PetscCallA(VecDestroy(Xu, ierr))
  PetscCallA(VecDestroy(Xl, ierr))
  PetscCallA(SNESDestroy(snes, ierr))
  PetscCallA(VecDestroy(F, ierr))
  PetscCallA(VecDestroy(X, ierr))
  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(PetscFinalize(ierr))
end program ex22f

!/*TEST
!
!test:
!  args:-lb 2.5
!
!TEST*/
