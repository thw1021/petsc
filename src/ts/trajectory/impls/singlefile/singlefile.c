#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscviewerhdf5.h>

static PetscErrorCode TSTrajectorySet_Singlefile(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscErrorCode ierr;
  const char     *filename;

  PetscFunctionBegin;
  if (stepnum == 0) {
    ierr = PetscObjectGetName((PetscObject)tj,&filename);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(tj->outputviewer,filename);CHKERRQ(ierr);
    ierr = PetscViewerSetUp(tj->outputviewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
    if (tj->storageviewertype == TJ_HDF5) {
      ierr = PetscViewerHDF5PushTimestepping(tj->outputviewer);CHKERRQ(ierr);
    }
#endif
  }
#if defined(PETSC_HAVE_HDF5)
  if (tj->storageviewertype == TJ_HDF5) {
    ierr = PetscViewerHDF5SetTimestep(tj->outputviewer,stepnum);CHKERRQ(ierr);
  }
#endif
  ierr = VecView(X,tj->outputviewer);CHKERRQ(ierr);
  if (tj->storageviewertype == TJ_BINARY) {
    ierr = PetscViewerBinaryWrite(tj->outputviewer,&time,1,PETSC_REAL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYSINGLEFILE - Stores all solutions of the ODE/ADE into a single file followed by each timestep. Does not save the intermediate stages in a multistage method

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory tj,TS ts)
{
  PetscFunctionBegin;
  tj->ops->set     = TSTrajectorySet_Singlefile;
  tj->ops->get     = NULL;
  ts->setupcalled  = PETSC_TRUE;
  PetscFunctionReturn(0);
}
