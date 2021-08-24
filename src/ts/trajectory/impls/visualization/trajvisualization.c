#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscviewerhdf5.h>

static PetscErrorCode TSTrajectorySet_Visualization(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  char           filename[PETSC_MAX_PATH_LEN];
  PetscReal      tprev;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  if (tj->storageviewertype == TJ_HDF5) {
    ierr = PetscSNPrintf(filename,sizeof(filename),"Visualization-data/SA-%06d.h5",stepnum);CHKERRQ(ierr);
  }
#endif
  if (tj->storageviewertype == TJ_BINARY) {
    ierr = PetscSNPrintf(filename,sizeof(filename),"Visualization-data/SA-%06d.bin",stepnum);CHKERRQ(ierr);
  }
  ierr = PetscViewerFileSetName(tj->outputviewer,filename);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(tj->outputviewer);CHKERRQ(ierr);
  if (stepnum == 0) {
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    if (!rank) {
      ierr = PetscRMTree("Visualization-data");CHKERRQ(ierr);
      ierr = PetscMkdir("Visualization-data");CHKERRQ(ierr);
    }
    if (tj->names) {
      PetscViewer bnames;

#if defined(PETSC_HAVE_HDF5)
      if (tj->storageviewertype == TJ_HDF5) {
        ierr = PetscViewerHDF5Open(comm,"Visualization-data/variablenames",FILE_MODE_WRITE,&bnames);CHKERRQ(ierr);
        ierr = PetscViewerHDF5WriteAttribute(bnames,NULL,"variablenames",PETSC_STRING,tj->names);CHKERRQ(ierr);
      }
#endif
      if (tj->storageviewertype == TJ_BINARY) {
        ierr = PetscViewerBinaryOpen(comm,"Visualization-data/variablenames",FILE_MODE_WRITE,&bnames);CHKERRQ(ierr);
        ierr = PetscViewerBinaryWriteStringArray(bnames,(const char *const *)tj->names);CHKERRQ(ierr);
      }
      ierr = PetscViewerDestroy(&bnames);CHKERRQ(ierr);
    }
  }
  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  if (tj->storageviewertype == TJ_HDF5) {
    ierr = PetscViewerHDF5WriteAttribute(tj->outputviewer,NULL,"time",PETSC_REAL,(void*)&time);CHKERRQ(ierr);
    if (stepnum != 0) {
      ierr = PetscViewerHDF5WriteAttribute(tj->outputviewer,NULL,"previous_time",PETSC_REAL,(void*)&tprev);CHKERRQ(ierr);
    }
  }
#endif
  if (!tj->transform) {
    ierr = VecView(X,tj->outputviewer);CHKERRQ(ierr);
  } else {
    Vec XX;
    ierr = (*tj->transform)(tj->transformctx,X,&XX);CHKERRQ(ierr);
    ierr = VecView(XX,tj->outputviewer);CHKERRQ(ierr);
    ierr = VecDestroy(&XX);CHKERRQ(ierr);
  }
  if (tj->storageviewertype == TJ_BINARY) {
    ierr = PetscViewerBinaryWrite(tj->outputviewer,&time,1,PETSC_REAL);CHKERRQ(ierr);
    if (stepnum != 0) {
      ierr = PetscViewerBinaryWrite(tj->outputviewer,&tprev,1,PETSC_REAL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYVISUALIZATION - Stores each solution of the ODE/DAE in a file

      Saves each timestep into a separate file in Visualization-data/SA-%06d.bin

      This version saves only the solutions at each timestep, it does not save the solution at each stage,
      see TSTRAJECTORYBASIC that saves all stages

      $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m and $PETSC_DIR/lib/petsc/bin/PetscBinaryIOTrajectory.py
      can read in files created with this format into MATLAB and Python.

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType(), TSTrajectoryType, TSTrajectorySetVariableNames()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Visualization(TSTrajectory tj,TS ts)
{
  PetscFunctionBegin;
  tj->ops->set    = TSTrajectorySet_Visualization;
  tj->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}
