!
#include <petsc/finclude/petscvec.h>
program main
  use petscvec
  implicit none

!
!      This example demonstrates writing an array to a file in binary
!      format that may be read in by PETSc's VecLoad() routine.
!
  PetscInt i
  PetscErrorCode ierr
  integer4 fd
  PetscInt vecclassid(1)
  PetscScalar array(5)
  Vec x
  PetscViewer v

  vecclassid(1) = 1211211 + 3

  PetscCallA(PetscInitialize(ierr))

  array = [(real(i), i=1, 5)]

!      Open binary file for writing
  PetscCallA(PetscBinaryOpen('testfile', FILE_MODE_WRITE, fd, ierr))
!      Write the Vec header
  PetscCallA(PetscBinaryWrite(fd, vecclassid, 1_PETSC_INT_KIND, PETSC_INT, ierr))
!      Write the array length
  PetscCallA(PetscBinaryWrite(fd, 5_PETSC_INT_KIND, 1_PETSC_INT_KIND, PETSC_INT, ierr))
!      Write the array
  PetscCallA(PetscBinaryWrite(fd, array, 5_PETSC_INT_KIND, PETSC_SCALAR, ierr))
!      Close the file
  PetscCallA(PetscBinaryClose(fd, ierr))

!
!      Open the file for reading by PETSc
!
  PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF, 'testfile', FILE_MODE_READ, v, ierr))
!
!      Load the vector
!
  PetscCallA(VecCreate(PETSC_COMM_WORLD, x, ierr))
  PetscCallA(VecLoad(x, v, ierr))
  PetscCallA(PetscViewerDestroy(v, ierr))
!
!      Print the vector
!
  PetscCallA(VecView(x, PETSC_VIEWER_STDOUT_SELF, ierr))
!

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!     test:
!
!TEST*/
