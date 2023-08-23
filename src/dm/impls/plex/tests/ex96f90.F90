program ex98f90
#include "petsc/finclude/petsc.h"
    use petsc
    implicit none

    ! Get the fortran kind associated with PetscInt and PetscReal so that we can use literal constants.
    PetscInt                           :: dummyPetscInt
    PetscReal                          :: dummyPetscreal
    integer,parameter                  :: kPI = kind(dummyPetscInt)
    integer,parameter                  :: kPR = Selected_Real_Kind(Precision(dummyPetscreal))

    type(tDM)                          :: dm,pdm
    type(tPetscSection)                :: section
    character(len=PETSC_MAX_PATH_LEN)  :: ifilename,iobuffer
    PetscInt                           :: pStart,pEnd
    PetscErrorCode                     :: ierr
    PetscBool                          :: flg
    PetscMPIInt                        :: numProc
    MPI_Comm                           :: comm
    PetscReal,Dimension(:),Pointer     :: cVal
    PetscInt                           :: clSize
    Type(tVec)                         :: v

    PetscCallA(PetscInitialize(ierr))
    PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,numProc,ierr))

    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-i",ifilename,flg,ierr))
    if (.not. flg) then
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"missing input file name -i <input file name>")
    end if

    PetscCallA(DMPlexCreateFromFile(PETSC_COMM_WORLD,ifilename,PETSC_NULL_CHARACTER,PETSC_TRUE,dm,ierr))
    PetscCallA(DMPlexDistributeSetDefault(dm,PETSC_FALSE,ierr))
    PetscCallA(DMSetFromOptions(dm,ierr))

    if (numproc > 1) then
        PetscCallA(DMPlexDistribute(dm,0_kPI,PETSC_NULL_SF,pdm,ierr))
        PetscCallA(DMDestroy(dm,ierr))
        dm = pdm
    end if
    PetscCallA(DMViewFromOptions(dm,PETSC_NULL_OPTIONS,"-dm_view",ierr))

    PetscCallA(PetscObjectGetComm(dm,comm,ierr))
    PetscCallA(PetscSectionCreate(comm,section,ierr))
    PetscCallA(DMPlexGetChart(dm,pStart,pEnd,ierr))
    PetscCallA(PetscSectionSetChart(section,pStart,pEnd,ierr))

    PetscCallA(PetscSectionSetDof(section,pStart,1_kPI,ierr))

    PetscCallA(PetscSectionSetUp(section,ierr))
    PetscCallA(DMSetLocalSection(dm,section,ierr))
    PetscCallA(PetscObjectViewFromOptions(section,PETSC_NULL_OPTIONS,"-dm_section_view",ierr))

    PetscCallA(DMGetLocalVector(dm,v,ierr))
    PetscCallA(VecSet(v,-1.0_kPR,ierr))
    PetscCallA(VecViewFromOptions(v,PETSC_NULL_OPTIONS,"-dm_vec_view",ierr))

    Write(iobuffer,"('Point: ',i0'\n')") pStart
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr));
    PetscCallA(DMPlexVecGetClosure(dm,section,v,pStart,cval,ierr))
    Write(iobuffer,"('size(cval): ',i0'\n')") size(cval)
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr));
    PetscCallA(PetscRealView(size(cval),cval,PETSC_VIEWER_STDOUT_SELF,ierr))
    PetscCallA(DMPlexVecRestoreClosure(dm,section,v,pStart,cval,ierr))

    Write(iobuffer,"('Point: ',i0'\n')") pStart+1
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr));
    PetscCallA(DMPlexVecGetClosure(dm,section,v,pStart+1,cval,ierr))
    Write(iobuffer,"('size(cval): ',i0'\n')") size(cval)
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr));
    PetscCallA(PetscRealView(size(cval),cval,PETSC_VIEWER_STDOUT_SELF,ierr))
    PetscCallA(DMPlexVecRestoreClosure(dm,section,v,pStart+1,cval,ierr))

    PetscCallA(PetscSectionDestroy(section,ierr))
    PetscCallA(DMRestoreLocalVector(dm,v,ierr))
    PetscCallA(DMDestroy(dm,ierr))

    PetscCallA(VecCreate(PETSC_COMM_WORLD,v,ierr))
    PetscCallA(VecSetSizes(v,PETSC_DECIDE,0,ierr))
    PetscCallA(PetscObjectSetName(v, "U",ierr))
    PetscCallA(VecSetFromOptions(v,ierr))
    PetscCallA(VecViewFromOptions(v,PETSC_NULL_OPTIONS,"-dm_vec_view",ierr))
    PetscCallA(VecGetSize(v,clSize,ierr))
    Write(iobuffer,"('Size: ',i0'\n')") clSize
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr));
    PetscCallA(VecGetArrayF90(v,cval,ierr))
    PetscCallA(PetscRealView(clSize,cval,PETSC_VIEWER_STDOUT_SELF,ierr))
    PetscCallA(VecRestoreArrayF90(v,cval,ierr))

    PetscCallA(PetscFinalize(ierr))
end program ex98f90

! /*TEST
!   build:
!     requires: exodusii pnetcdf !complex
!   testset:
!     args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/SquareFaceSet.exo -dm_view -dm_section_view -dm_vec_view
!     nsize: 1
!     test:
!       suffix: 0
!       args:
! TEST*/
