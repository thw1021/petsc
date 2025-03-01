!
!
!  Tests MatGetOption() for correct return flag with Intel Fortran compilers
!

      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat A
      PetscErrorCode ierr
      PetscInt i,j,m,n,iar(1),jar(1)
      PetscInt one
      PetscScalar  v(1)
      PetscBool sym,known

      PetscCallA(PetscInitialize(ierr))
      m = 3
      n = 2
      one = 1
!
!      Create a parallel dense matrix shared by all processors
!
      PetscCallA(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_NULL_SCALAR_ARRAY,A,ierr))

!
!     Set values into the matrix. All processors set all values.
!
      do 10, i=0,m-1
        iar(1) = i
        do 20, j=0,n-1
          jar(1) = j
          v(1)   = 9.0/real(i+j+1)
          PetscCallA(MatSetValues(A,one,iar,one,jar,v,INSERT_VALUES,ierr))
 20     continue
 10   continue

      PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

      PetscCallA(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE,ierr))
      PetscCallA(MatIsSymmetricKnown(A,known,sym,ierr))
      PetscCheckA(known,PETSC_COMM_WORLD,PETSC_ERR_PLIB,'Symmetric boolean not correctly returned')
      PetscCheckA(sym,PETSC_COMM_WORLD,PETSC_ERR_PLIB,'Symmetric boolean not correctly returned')
!
!      Free the space used by the matrix
!
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      nsize: 2
!
!TEST*/
