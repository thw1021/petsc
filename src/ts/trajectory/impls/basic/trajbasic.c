#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscviewerhdf5.h>

/*
  For n-th time step, TSTrajectorySet_Basic always saves the solution X(t_n) and the current time t_n,
  and optionally saves the stage values Y[] between t_{n-1} and t_n, the previous time t_{n-1}, and
  forward stage sensitivities S[] = dY[]/dp.
*/
static PetscErrorCode TSTrajectorySet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  char           filename[PETSC_MAX_PATH_LEN];
  PetscInt       ns,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof(filename),tj->dirfiletemplate,stepnum);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(tj->outputviewer,filename);CHKERRQ(ierr); /* this triggers PetscViewer to be set up again */
  ierr = PetscViewerSetUp(tj->outputviewer);CHKERRQ(ierr);
  ierr = VecView(X,tj->outputviewer);CHKERRQ(ierr);

#if defined(PETSC_HAVE_HDF5)
  if (tj->storageviewertype == TJ_HDF5) {
    ierr = PetscViewerHDF5WriteAttribute(tj->outputviewer,NULL,"time",PETSC_REAL,(void *)&time);CHKERRQ(ierr);
  }
#endif
  if (tj->storageviewertype == TJ_BINARY) {
    ierr = PetscViewerBinaryWrite(tj->outputviewer,&time,1,PETSC_REAL);CHKERRQ(ierr);
  }
  if (stepnum && !tj->solution_only) {
    Vec       *Y;
    PetscReal tprev;

    ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
    if (tj->storageviewertype == TJ_HDF5) {
      ierr = PetscViewerHDF5WriteAttribute(tj->outputviewer,NULL,"previous_time",PETSC_REAL,(void *)&tprev);CHKERRQ(ierr); /* this does not work if called after VecView */
    }
#endif
    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    for (i=0; i<ns; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be saved again. */
      if (ts->stifflyaccurate && i == ns-1) continue;
      ierr = VecView(Y[i],tj->outputviewer);CHKERRQ(ierr);
    }
    ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
    if (tj->storageviewertype == TJ_BINARY) {
      ierr = PetscViewerBinaryWrite(tj->outputviewer,&tprev,1,PETSC_REAL);CHKERRQ(ierr);
    }
  }
  /* Tangent linear sensitivities needed by second-order adjoint */
  if (ts->forward_solve) {
    Mat A,*S;

    ierr = TSForwardGetSensitivities(ts,NULL,&A);CHKERRQ(ierr);
    ierr = MatView(A,tj->outputviewer);CHKERRQ(ierr);
    if (stepnum) {
      ierr = TSForwardGetStages(ts,&ns,&S);CHKERRQ(ierr);
      for (i=0; i<ns; i++) {
        ierr = MatView(S[i],tj->outputviewer);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectorySetFromOptions_Basic(PetscOptionItems *PetscOptionsObject,TSTrajectory tj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"TS trajectory options for Basic type");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTrajectoryGet_Basic(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  Vec            Sol;
  PetscInt       ns,i;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filename,sizeof(filename),tj->dirfiletemplate,stepnum);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(tj->inputviewer,filename);CHKERRQ(ierr); /* this triggers PetscViewer to be set up again */
  ierr = PetscViewerSetUp(tj->inputviewer);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&Sol);CHKERRQ(ierr);
  ierr = VecLoad(Sol,tj->inputviewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  if (tj->storageviewertype == TJ_HDF5) {
    ierr = PetscViewerHDF5ReadAttribute(tj->inputviewer,NULL,"time",PETSC_REAL,NULL,t);CHKERRQ(ierr);
  }
#endif
  if (tj->storageviewertype == TJ_BINARY) {
    ierr = PetscViewerBinaryRead(tj->inputviewer,t,1,NULL,PETSC_REAL);CHKERRQ(ierr);
  }
  if (stepnum && !tj->solution_only) {
    Vec       *Y;
    PetscReal timepre;
    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    for (i=0; i<ns; i++) {
      /* For stiffly accurate TS methods, the last stage Y[ns-1] is the same as the solution X, thus does not need to be loaded again. */
      if (ts->stifflyaccurate && i == ns-1) continue;
      ierr = VecLoad(Y[i],tj->inputviewer);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_HDF5)
    if (tj->storageviewertype == TJ_HDF5) {
      ierr = PetscViewerHDF5ReadAttribute(tj->inputviewer,NULL,"previous_time",PETSC_REAL,NULL,&timepre);CHKERRQ(ierr);
    }
#endif
    if (tj->storageviewertype == TJ_BINARY) {
      ierr = PetscViewerBinaryRead(tj->inputviewer,&timepre,1,NULL,PETSC_REAL);CHKERRQ(ierr);
    }
    if (tj->adjoint_solve_mode) {
      ierr = TSSetTimeStep(ts,-(*t)+timepre);CHKERRQ(ierr);
    }
  }
  /* Tangent linear sensitivities needed by second-order adjoint */
  if (ts->forward_solve) {
    if (!ts->stifflyaccurate) {
      Mat A;
      ierr = TSForwardGetSensitivities(ts,NULL,&A);CHKERRQ(ierr);
      ierr = MatLoad(A,tj->inputviewer);CHKERRQ(ierr);
    }
    if (stepnum) {
      Mat *S;
      ierr = TSForwardGetStages(ts,&ns,&S);CHKERRQ(ierr);
      for (i=0; i<ns; i++) {
        ierr = MatLoad(S[i],tj->inputviewer);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSTrajectorySetUp_Basic(TSTrajectory tj,TS ts)
{
  MPI_Comm       comm;
  PetscMPIInt    rank;
  char           dtempname[16] = "TS-data-XXXXXX";
  char           *dir = tj->dirname;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tj,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  if (!dir) {
    if (!rank) {
      ierr = PetscMkdtemp(dtempname);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(dtempname,16,MPI_CHAR,0,comm);CHKERRMPI(ierr);
    ierr = PetscStrallocpy(dtempname,&tj->dirname);CHKERRQ(ierr);
  } else {
    if (!rank) {
      PetscBool flg;
      ierr = PetscTestDirectory(dir,'w',&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscTestFile(dir,'r',&flg);CHKERRQ(ierr);
        if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Specified path is a file - not a dir: %s",dir);
        ierr = PetscMkdir(dir);CHKERRQ(ierr);
      } else SETERRQ1(comm,PETSC_ERR_SUP,"Directory %s not empty",tj->dirname);
    }
  }
  ierr = PetscBarrier((PetscObject)tj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYBASIC - Stores each solution of the ODE/DAE in a file

      Saves each timestep into a separate file named TS-data-XXXXXX/TS-%06d.bin. The file name can be changed.

      This version saves the solutions at all the stages

      $PETSC_DIR/share/petsc/matlab/PetscReadBinaryTrajectory.m can read in files created with this format

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType(), TSTrajectorySetDirname(), TSTrajectorySetFile()

M*/
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory tj,TS ts)
{
  PetscFunctionBegin;
  tj->ops->set            = TSTrajectorySet_Basic;
  tj->ops->get            = TSTrajectoryGet_Basic;
  tj->ops->setup          = TSTrajectorySetUp_Basic;
  tj->ops->setfromoptions = TSTrajectorySetFromOptions_Basic;
  PetscFunctionReturn(0);
}
