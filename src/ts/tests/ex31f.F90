!
!   Tests TSRKGetTableau()
!
!23456789012345678901234567890123456789012345678901234567890123456789012
#include <petsc/finclude/petscts.h>

program main
  use petscts
  implicit none
!
  TS ts
  PetscInt s, p
  PetscBool FSAL
  PetscReal, pointer :: A(:), b(:), c(:), bembed(:), binterp(:)
  PetscErrorCode ierr
  Vec x

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(TSCreate(PETSC_COMM_WORLD, ts, ierr))
  PetscCallA(TSSetType(ts, TSRK, ierr))
  PetscCallA(TSSetFromOptions(ts, ierr))
  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, 1_PETSC_INT_KIND, x, ierr))
  PetscCallA(TSSetSolution(ts, x, ierr))

  PetscCallA(TSSetUp(ts, ierr))
  PetscCallA(TSRKGetTableau(ts, s, A, b, c, bembed, p, binterp, FSAL, ierr))
  print *, A
  print *, b, c, bembed
  print *, p, binterp
  print *, FSAL
  PetscCallA(TSRKRestoreTableau(ts, s, A, b, c, bembed, p, binterp, FSAL, ierr))

  PetscCallA(TSRKSetType(ts, "5bs", ierr))
  PetscCallA(TSSetUp(ts, ierr))
  PetscCallA(TSRKGetTableau(ts, s, A, PETSC_NULL_REAL_POINTER, c, bembed, p, binterp, PETSC_NULL_BOOL, ierr))
  print *, A
  print *, c, bembed
  print *, p, binterp
  PetscCallA(TSRKRestoreTableau(ts, s, A, PETSC_NULL_REAL_POINTER, c, bembed, p, binterp, PETSC_NULL_BOOL, ierr))

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(TSDestroy(ts, ierr))
  PetscCallA(PetscFinalize(ierr))
end
!/*TEST
!
!    test:
!
!TEST*/
