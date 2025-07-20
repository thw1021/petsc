!
!  Simple PETSc Program written in Fortran
!
#include <petsc/finclude/petscsys.h>
       program ex38f
       use petscmpi  ! or mpi or mpi_f08
       use petscsys
       implicit none

       PetscErrorCode  ierr
       PetscInt f(1)
       PetscCallA(PetscInitialize(ierr))
       f(1) = 1
       PetscCallMPIA(MPI_Allreduce(MPI_IN_PLACE,f,1,MPIU_INTEGER,MPI_MIN,PETSC_COMM_WORLD,ierr))
       PetscCallA(PetscFinalize(ierr))
       end program ex38f

!/*TEST
!
!   test:
!      output_file: output/empty.out
!
!TEST*/
