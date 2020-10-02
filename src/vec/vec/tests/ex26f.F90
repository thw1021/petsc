!
!
!  Test VecGetSubVector()
!  Contributed-by: Adrian Croucher <gitlab@mg.gitlab.com>

program main
#include <petsc/finclude/petsc.h>

  use petsc
  implicit none
  PetscMPIInt :: rank
  PetscErrorCode :: ierr
  PetscInt :: num_cells, subsize, i
  PetscInt, parameter :: blocksize = 3, field = 0
  Vec :: v, subv
  IS :: index_set
  PetscInt, allocatable :: subindices(:)

  call PetscInitialize(PETSC_NULL_CHARACTER, ierr); CHKERRQ(ierr)
  call MPI_COMM_RANK(PETSC_COMM_WORLD, rank, ierr)

  if (rank == 0) then
     num_cells = 1
  else
     num_cells = 0
  end if

  call VecCreate(PETSC_COMM_WORLD, v, ierr); CHKERRQ(ierr)
  call VecSetSizes(v, num_cells * blocksize, PETSC_DECIDE, ierr); CHKERRQ(ierr)
  call VecSetBlockSize(v, blocksize, ierr); CHKERRQ(ierr)
  call VecSetFromOptions(v, ierr); CHKERRQ(ierr)

  subsize = num_cells
  allocate(subindices(0: subsize - 1))
  subindices = [(i, i = 0, subsize - 1)] * blocksize + field
  call ISCreateGeneral(PETSC_COMM_WORLD, subsize, subindices, &
       PETSC_COPY_VALUES, index_set, ierr); CHKERRQ(ierr)
  deallocate(subindices)

  call VecGetSubVector(v, index_set, subv, ierr); CHKERRQ(ierr)
  call VecRestoreSubVector(v, index_set, subv, ierr); CHKERRQ(ierr)
  call ISDestroy(index_set, ierr); CHKERRQ(ierr)

  call VecDestroy(v, ierr); CHKERRQ(ierr)
  call PetscFinalize(ierr); CHKERRQ(ierr)
  end


!/*TEST
!
!   test:
!      nsize: 2
!      filter: sort -b
!      filter_output: sort -b
!
!TEST*/
