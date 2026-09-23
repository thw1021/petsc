!
!  Description: Tests VecCUDAGetArray() and friends along with OpenMP target offload
!
#include <petsc/finclude/petscvec.h>
program main
  use petscvec
  implicit none

  PetscInt, parameter :: n = 4
  PetscScalar, pointer, dimension(:) :: a
  PetscInt :: i
  PetscErrorCode :: ierr
  Vec :: x

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(VecCreate(PETSC_COMM_WORLD, x, ierr))
  PetscCallA(VecSetSizes(x, PETSC_DECIDE, n, ierr))
  PetscCallA(VecSetType(x, 'cuda', ierr))

!  Write access hands back uninitialized device memory; fill it on the GPU.

  PetscCallA(VecCUDAGetArrayWrite(x, a, ierr))
  !$omp target teams distribute parallel do is_device_ptr(a)
  do i = 1, n
    a(i) = 2.0
  end do
  !$omp end target teams distribute parallel do
  PetscCallA(VecCUDARestoreArrayWrite(x, a, ierr))

!  Read-write access: update the buffer in place on the GPU.

  PetscCallA(VecCUDAGetArray(x, a, ierr))
  !$omp target teams distribute parallel do is_device_ptr(a)
  do i = 1, n
    a(i) = a(i) + 1.0
  end do
  !$omp end target teams distribute parallel do
  PetscCallA(VecCUDARestoreArray(x, a, ierr))

!  Read-only access round-trip.

  PetscCallA(VecCUDAGetArrayRead(x, a, ierr))
  PetscCallA(VecCUDARestoreArrayRead(x, a, ierr))

  PetscCallA(PetscObjectSetName(x, 'x', ierr))
  PetscCallA(VecView(x, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!
!/*TEST
!
!   test:
!     requires: cuda defined(PETSC_HAVE_OPENMP_TARGET_OFFLOAD_FC)
!
!TEST*/
