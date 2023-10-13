!
!     Demonstrates use of MatDuplicate() for a shell matrix with a context
!
     MODULE solver_context
#include "petsc/finclude/petscmat.h"
       USE petscmat
       IMPLICIT NONE
       TYPE :: MatCtx
         PetscReal :: lambda
       END TYPE MatCtx
     END MODULE solver_context

     MODULE solver_context_interfaces
       USE solver_context
       IMPLICIT NONE

       INTERFACE MatCreateShell
         SUBROUTINE MatCreateShell(comm,mloc,nloc,m,n,ctx,mat,ierr)
           USE solver_context
           MPI_Comm :: comm
           PetscInt :: mloc,nloc,m,n
           TYPE(MatCtx) :: ctx
           Mat :: mat
           PetscErrorCode :: ierr
         END SUBROUTINE MatCreateShell
       END INTERFACE MatCreateShell

       INTERFACE MatShellSetContext
         SUBROUTINE MatShellSetContext(mat,ctx,ierr)
           USE solver_context
           Mat :: mat
           TYPE(MatCtx) :: ctx
           PetscErrorCode :: ierr
         END SUBROUTINE MatShellSetContext
       END INTERFACE MatShellSetContext

       INTERFACE MatShellGetContext
         SUBROUTINE MatShellGetContext(mat,ctx,ierr)
           USE solver_context
           Mat :: mat
           TYPE(MatCtx), POINTER :: ctx
           PetscErrorCode :: ierr
         END SUBROUTINE MatShellGetContext
       END INTERFACE MatShellGetContext

     END MODULE solver_context_interfaces

! ----------------------------------------------------
!                    main program
! ----------------------------------------------------
     PROGRAM main
#include "petsc/finclude/petscmat.h"
       USE solver_context_interfaces
       IMPLICIT NONE
       Mat :: F, Fcopy
       TYPE(MatCtx) :: ctxF
       TYPE(MatCtx),POINTER :: ctxF_pt, ctxFcopy_pt
       PetscErrorCode :: ierr
       PetscInt :: n=128

       PetscCallA(PetscInitialize(ierr))
       ctxF%lambda = 3.14d0
       PetscCallA(MatCreateShell(PETSC_COMM_WORLD,n,n,PETSC_DETERMINE,PETSC_DETERMINE,ctxF,F,ierr))
       PetscCallA(MatShellSetOperation(F,MATOP_DUPLICATE,MatDuplicate_F,ierr))
       PRINT*,'ctxF%lambda = ',ctxF%lambda

       PetscCallA(MatShellGetContext(F,ctxF_pt,ierr))
       PRINT*,'ctxF_pt%lambda = ',ctxF_pt%lambda

       PetscCallA(MatDuplicate(F,MAT_DO_NOT_COPY_VALUES,Fcopy,ierr))
       PetscCallA(MatShellGetContext(Fcopy,ctxFcopy_pt,ierr))
       PRINT*,'ctxFcopy_pt%lambda = ',ctxFcopy_pt%lambda

       PetscCallA(MatDestroy(F,ierr))
       PetscCallA(MatDestroy(Fcopy,ierr))
       PetscCallA(PetscFinalize(ierr))

     CONTAINS

       SUBROUTINE MatDuplicate_F(F, opt, M, ierr)

         Mat                      :: F, M
         MatDuplicateOption       :: opt
         PetscErrorCode           :: ierr
         PetscInt                 :: Mg,Ng,ml,nl
         TYPE(MatCtx),POINTER     :: ctxM,ctxF_pt

         PetscCall(MatGetSize(F,Mg,Ng,ierr));
         PetscCall(MatGetLocalSize(F,ml,nl,ierr));
         PetscCall(MatShellGetContext(F,ctxF_pt,ierr))
         allocate(ctxM)
         ctxM%lambda = ctxF_pt%lambda
         PetscCall(MatCreateShell(PETSC_COMM_WORLD,Mg,Ng,ml,nl,ctxM,M,ierr))
!        PetscCall(MatShellSetOperation(M,MATOP_DUPLICATE,MatDuplicate_F,ierr))

       END SUBROUTINE MatDuplicate_F

     END PROGRAM main

!/*TEST
!
!     build:
!       requires: double
!
!     test:
!
!TEST*/
