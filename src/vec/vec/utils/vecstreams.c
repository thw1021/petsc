
#include <petscsys.h>
#include <petscvec.h>         /*I  "petscvec.h"  I*/

#if defined(PETSC_HAVE_SETJMP_H)
#include <setjmp.h>
static jmp_buf PetscSegvJumpBuf;

static PetscErrorCode PetscSignalHandler(int sig,void *ptr)
{
  longjmp(PetscSegvJumpBuf,1);
}
#endif

#define CHKERR(ierr)  do {if (ierr) {PetscPopErrorHandler(); PetscFunctionReturn(0);}} while (0)

static PetscErrorCode VecStreamsView_Private(MPI_Comm comm,VecType vtype,PetscLogDouble *rate,PetscBool *success)
{
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  Vec               x,y,w;
  PetscInt          N = 4000000,n=4;
  PetscLogDouble    t = 0, tr = PETSC_MAX_REAL;
  MPI_Comm          ncomm;

  PetscFunctionBegin;
  *success = PETSC_FALSE;
  /* Do not terminate the code upon errors; simply return with a success of false */
  ierr = PetscPushErrorHandler(PetscReturnErrorHandler,NULL);CHKERRQ(ierr);

  /* This catches signals in the block of code below and converts them to a harmless return */
  /* Note this will not work if the OS sends a terminate signal due to excessive memory usage */
#if defined(PETSC_HAVE_SETJMP_H)
  ierr = PetscPushSignalHandler(PetscSignalHandler,NULL);CHKERR(ierr);
  if (setjmp(PetscSegvJumpBuf)) {
    ierr = PetscPopSignalHandler();CHKERR(ierr);
    ierr = PetscPopErrorHandler();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED,0,0,&ncomm);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERR(ierr);
  ierr = VecCreate(ncomm,&x);CHKERR(ierr);
  ierr = VecSetSizes(x,N,PETSC_DECIDE);CHKERR(ierr);
  ierr = VecSetType(x,vtype);CHKERR(ierr);
  ierr = VecSetUp(x);CHKERR(ierr);
  ierr = VecDuplicate(x,&y);CHKERR(ierr);
  ierr = VecDuplicate(x,&w);CHKERR(ierr);
  ierr = VecSetRandom(x,NULL);CHKERR(ierr);
  ierr = VecSetRandom(y,NULL);CHKERR(ierr);

  ierr = VecWAXPY(w,3.0,x,y);CHKERR(ierr);

  for (PetscInt i=0; i<n; i++) {
    ierr = PetscTimeSubtract(&t);CHKERR(ierr);
    ierr = MPI_Barrier(ncomm);CHKERR(ierr);
    ierr = VecWAXPY(w,3.0,x,y);CHKERR(ierr);
    ierr = MPI_Barrier(ncomm);CHKERR(ierr);
    ierr = PetscTimeAdd(&t);CHKERR(ierr);
    tr = PetscMin(t,tr);
  }
  t = 1.e-6*3*N*sizeof(PetscScalar)/tr;
  ierr = MPI_Allreduce(&t,rate,1,MPI_DOUBLE,MPI_SUM,comm);CHKERR(ierr);
  ierr = VecDestroy(&x);CHKERR(ierr);
  ierr = VecDestroy(&y);CHKERR(ierr);
  ierr = VecDestroy(&w);CHKERR(ierr);
  ierr = MPI_Comm_free(&ncomm);CHKERRMPI(ierr);
#if defined(PETSC_HAVE_SETJMP_H)
  ierr = PetscPopSignalHandler();CHKERR(ierr);
#endif
  ierr = PetscPopErrorHandler();CHKERRQ(ierr);
  *success = PETSC_TRUE;
  PetscFunctionReturn(0);
}
PetscErrorCode VecStreamsView_System(PetscViewer viewer,VecType vtype)
{
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  MPI_Comm          comm;
  PetscBool         success;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (!(size % 2)) {
    PetscSubcomm   sub;
    MPI_Comm       subcomm;
    PetscLogDouble rate[2];
    PetscMPIInt    rank;

    ierr = VecStreamsView_Private(comm,vtype,&rate[0],&success);CHKERRQ(ierr);
    if (!success) PetscFunctionReturn(0);

    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = PetscSubcommCreate(comm,&sub);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(sub,2);CHKERRQ(ierr);
    ierr = PetscSubcommSetType(sub,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
    subcomm = PetscSubcommChild(sub);CHKERRQ(ierr);
    if (!(rank % 2)) {
      ierr = VecStreamsView_Private(subcomm,vtype,&rate[1],&success);CHKERRQ(ierr);
    }
    ierr = PetscSubcommDestroy(&sub);CHKERRQ(ierr);
    if (!success) PetscFunctionReturn(0);

    if (rank == 0 /* && rate[0] < 1.7*rate[1] */) {
      ierr = PetscViewerASCIIPrintf(viewer,"**************************************************************************************************************************************\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"The %s streams rate %g (MB/s) on all (%d) MPI ranks is not nearly two times the rate on half of the ranks %g.\n",vtype,rate[0],size,rate[1]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"This indicates the memory bandwidth does not scale with the number of MPI ranks. You might be using more ranks than optimal.\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"See: https://petsc.org/release/faq/#what-kind-of-parallel-computers-or-clusters-are-needed-to-use-petsc-or-why-do-i-get-little-speedup\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"**************************************************************************************************************************************\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C

     VecStreamsView - run a streams-like benchmark on an MPI communicator and half of that communicator obtained from a viewer and report the results to the viewer

  Input Parameter:
.   viewer - the ASCII viewer

@*/
PetscErrorCode VecStreamsView(PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscLogStage     stage;

  PetscFunctionBegin;
  ierr = PetscLogStageRegister("VecStreams Micro-benchmark",&stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = VecStreamsView_System(viewer,VECSTANDARD);CHKERRQ(ierr);
  if (0 /* && PetscDeviceInitialized(PETSC_DEVICE_CUDA)*/) {
    ierr = VecStreamsView_System(viewer,VECCUDA);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
