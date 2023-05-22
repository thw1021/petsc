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
    PetscBool                                 :: setb,sete,seti,setia,setr,setra,sets
    PetscBool                                 :: bvalue = PETSC_TRUE
    PetscInt                                  :: nopt = 3_Pi
    PetscInt                                  :: ivalue = 2_Pi
    PetscInt,dimension(:),pointer             :: iarray
    PetscReal                                 :: rvalue = 1.23_Pr
    PetscReal,dimension(:),pointer            :: rarray
    PetscScalar                               :: svalue = -4.56_Pr
    character(len=256)                        :: IOBuffer
    character(len=256)                        :: list(6)
    PetscEnum                                 :: evalue = 2

    PetscCallA(PetscInitialize(ierr))
    ! list(1) = 'a123   '
    ! list(2) = 'b456   '
    ! list(3) = 'c789   '
    ! list(4) = 'list   '
    ! list(5) = 'prefix_'
    ! list(6) = ''


    Allocate(iarray(nopt),source=-1_Pi)
    Allocate(rarray(nopt),source=-99.0_pr)

    PetscCallA(PetscOptionsBegin(PETSC_COMM_WORLD,'prefix_','Setting options for my application','Section 1',ierr))
    PetscCallA(PetscOptionsBool('-bool','Get an application bool','Man page',bvalue,bvalue,setb,ierr))
    PetscCallA(PetscOptionsInt('-int','Get an application int','Man page',ivalue,ivalue,seti,ierr))
    PetscCallA(PetscOptionsIntArray('-intarray','Get an application int array','Man page',iarray,nopt,setia,ierr))
    ! PetscCallA(PetscOptionsEnum('-enum','Get an application enum','Man page',evalue,evalue,sete,ierr))
    PetscCallA(PetscOptionsReal('-real','Get an application real','Man page',rvalue,rvalue,setr,ierr))
    PetscCallA(PetscOptionsRealArray('-realarray','Get an application real array','Man page',rarray,nopt,setra,ierr))
    PetscCallA(PetscOptionsScalar('-scalar','Get an application scalar','Man page',svalue,svalue,sets,ierr))
    PetscCallA(PetscOptionsEnd(ierr))

    if (setb) then
        write(IOBuffer,'("The bool value was set to ",L0,"\n")') bvalue
        PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
    endif
    if (seti) then
        write(IOBuffer,'("The integer value was set to ",I0,"\n")') ivalue
        PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
    endif
    if (setia) then
        write(IOBuffer, '("The integer array was set to ",*(i0," "))') iarray
        PetscCallA(PetscPrintf(PETSC_COMM_WORLD,trim(IOBuffer)//"\n",ierr))
    endif
    if (setr) then
        write(IOBuffer,'("The real value was set to ",ES12.5,"\n")') rvalue
        PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))
    endif
    if (setra) then
        write(IOBuffer,'("The real array was set to ",*(ES12.5," "))') rarray
        PetscCallA(PetscPrintf(PETSC_COMM_WORLD,trim(IOBuffer)//"\n",ierr))
    endif

    deallocate(iarray)
    deallocate(rarray)
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
!      args: -prefix_bool no -prefix_int 22 -prefix_intarray 2-5 -prefix_real 2.34 -prefix_realarray -3,-4,5.5 -prefix_scalar 7.89
!
!TEST*/
