    program main
#include "petsc/finclude/petsc.h"
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>
      use petsc
      use petscvec
      use petscmat
      use petscvecdef
      use petscmatdef
      implicit none

      Vec:: x,y
      Mat:: A
      PetscInt::N=500,i,j,NN=5
      PetscReal::alpha=2.0,xx=4.0,v=2.0
      PetscErrorCode ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call VecCreate(PETSC_COMM_WORLD,x,ierr)
      call VecSetSizes(x,PETSC_DECIDE,N,ierr)
      call VecSetFromOptions(x,ierr)
      call VecDuplicate(x,y,ierr)

      call VecSet(x,xx,ierr)

      call MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,NN,PETSC_NULL_INTEGER,A,ierr)
      call MatSetFromOptions(A,ierr)

      call MatSetValues(A,1,0,1,0,v,INSERT_VALUES,ierr)

      do i=0,N-1
          v=2.0
          do j=i-2,i+2
              if (j>=0.and.j<N) then
                  call MatSetValues(A,1,i,1,j,v,INSERT_VALUES,ierr)
              endif
          enddo
      enddo

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)


      do i=0,100
          call VecAXPY(y,alpha,x,ierr)
          call MatMult(A,x,y,ierr)
      enddo
      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call MatDestroy(A,ierr)
      call PetscFinalize(ierr);

    end program main
