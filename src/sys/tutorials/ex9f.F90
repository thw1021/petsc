!
!   Example of using PetscOptionsBegin in Fortran
program ex9f
#include "petsc/finclude/petsc.h"
      use petsc
      implicit none

      PetscReal,Parameter                       :: PReal = 1.0
      Integer,Parameter                         :: Pr = Selected_Real_Kind(Precision(PReal))
                                            
      PetscInt,Parameter                        :: PInt = 1
      Integer,Parameter                         :: Pi = kind(PInt)

      PetscErrorCode                            :: ierr
      PetscBool                                 :: set = PETSC_FALSE
      PetscInt                                  :: value = 2_Pi

      Character(len=256)                        :: IOBuffer

      PetscCallA(PetscInitialize(ierr))

      PetscCallA(PetscOptionsBegin(PETSC_COMM_WORLD,'prefix_','Setting options for my application','Section 1',ierr))
      PetscCallA(PetscOptionsInt('-int','Get an application int','Man page',2_Pi,value,set,ierr))
      PetscCallA(PetscOptionsEnd(ierr))
      if (set) then
         write(IOBuffer,'("The integer value was set to ",I3,"\n")') value
         PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
      endif
      PetscCallA(PetscFinalize(ierr))
end program ex9f

!
!/*TEST
!
!   build:
!      requires: defined(PETSC_USING_F2003) defined(PETSC_USING_F90FREEFORM)
!
!   test:
!
!   test:
!      suffix: 2
!      args: -prefix_int 22
!
!TEST*/
