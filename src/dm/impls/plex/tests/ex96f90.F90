program ex96f90
#include "petsc/finclude/petsc.h"
    use petsc
    implicit none
#include "exodusII.inc"

    ! Get the Fortran kind associated with PetscInt and PetscReal so that we can use literal constants.
    PetscInt                           :: dummyPetscInt
    PetscReal                          :: dummyPetscreal
    integer,parameter                  :: kPI = kind(dummyPetscInt)
    integer,parameter                  :: kPR = kind(dummyPetscReal)

    PetscErrorCode                     :: ierr
    type(tDM)                          :: dm
    character(len=PETSC_MAX_PATH_LEN)  :: ifilename,ofilename,IOBuffer
    PetscInt                           :: numVar
    PetscMPIInt                        :: rank,numProc
    PetscInt                           :: order = 1
    PetscBool                          :: flg
    type(tPetscViewer)                 :: viewer


    PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER,ierr))
    if (ierr /= 0) then
      print*,'Unable to initialize PETSc'
      stop
    endif

    PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
    PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,numProc,ierr))
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-i',ifilename,flg,ierr))
    PetscCheckA(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,'missing input file name -i <input file name>')
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-o',ofilename,flg,ierr))
    PetscCheckA(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,'missing output file name -o <output file name>')

    ! Read the mesh in any supported format
    PetscCallA(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename,PETSC_NULL_CHARACTER,PETSC_TRUE,dm,ierr))
    PetscCallA(DMSetFromOptions(dm,ierr));
    PetscCallA(PetscObjectSetName(dm, "ex96f90", ierr));
    PetscCallA(DMViewFromOptions(dm, PETSC_NULL_OPTIONS,'-dm_view',ierr));

    ! enable exodus debugging information
    PetscCallA(exopts(EXVRBS+EXDEBG,ierr))

    ! Create the exodus file
    PetscCallA(PetscViewerExodusIIOpen(PETSC_COMM_WORLD,ofilename,FILE_MODE_WRITE,viewer,ierr))
    PetscCallA(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD,ierr))

    PetscCallA(PetscViewerExodusIISetOrder(viewer,order,ierr))
    ! Save the geometry to the file, erasing all previous content
    PetscCallA(DMView(dm,viewer,ierr))
    PetscCallA(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD,ierr))
    PetscCall(PetscViewerFlush(viewer,ierr))

    numVar = 3
    PetscCall(PetscViewerExodusIISetZonalVariable(viewer, numVar,ierr))

    numVar = -1
    PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, numVar,ierr))
    write(IOBuffer,'("Number of zonal variables", I3, "\n")') numVar
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))

    numVar = 2
    PetscCall(PetscViewerExodusIISetNodalVariable(viewer, numVar,ierr))

    numVar = -1
    PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, numVar,ierr))
    write(IOBuffer,'("Number of nodal variables", I3, "\n")') numVar
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,IOBuffer,ierr))

    PetscCallA(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD,ierr))

    !!! This call should lead to an error
    numVar = 2
    PetscCall(PetscViewerExodusIISetNodalVariable(viewer, numVar,ierr))

    PetscCallA(PetscViewerDestroy(viewer, ierr))
    PetscCallA(PetscFinalize(ierr))
end program ex96f90