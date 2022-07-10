
#include <petscsys.h>
#include <petsc/private/benchimpl.h>     /*I  "petscbench.h"   I*/
#include <petscvec.h>

typedef struct {
  Vec vec;
} PetscBench_VecStreams;


#if defined(PETSC_HAVE_SETJMP_H)
#include <setjmp.h>
static jmp_buf PetscSegvJumpBuf;

static PetscErrorCode PetscSignalHandler(int sig,void *ptr)
{
  longjmp(PetscSegvJumpBuf,1);
}
#endif

/* Ignore errors while running and checking the benchmark */
/*#define PetscCallIgnore(...) do {                                     \
    PetscErrorCode ierr_q_ = __VA_ARGS__;                                                      \
    if (PetscUnlikely(ierr_q_)) {PetscPopErrorHandler(); PetscFunctionReturn(0);} \
 } while (0) */
#define PetscCallIgnore(...) PetscCall(__VA_ARGS__)

static PetscErrorCode VecStreamsView_Private(MPI_Comm comm,VecType vtype,PetscLogDouble *rate,PetscBool *success)
{
  PetscMPIInt       size;
  Vec               x,y,w;
  PetscInt          N = 4000000,n=4;
  PetscLogDouble    t = 0, tr = PETSC_MAX_REAL;
  MPI_Comm          ncomm;
  MPI_Info          info;

  PetscFunctionBegin;
  *success = PETSC_FALSE;
  /* Do not terminate the code upon errors; simply return with a success of false */
  PetscCallIgnore(PetscPushErrorHandler(PetscReturnErrorHandler,NULL));

  /* This catches signals in the block of code below and converts them to a harmless return */
  /* Note this will not work if the OS sends a terminate signal due to excessive memory usage */
#if defined(PETSC_HAVE_SETJMP_H)
  PetscCallIgnore(PetscPushSignalHandler(PetscSignalHandler,NULL));
  if (setjmp(PetscSegvJumpBuf)) {
    PetscCallIgnore(PetscPopSignalHandler());
    PetscCallIgnore(PetscPopErrorHandler());
    PetscFunctionReturn(0);
  }
#endif

  MPI_Info_create(&info);
  PetscCallIgnore(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED,0,info,&ncomm));
  MPI_Info_free(&info);
  PetscCallIgnore(MPI_Comm_size(comm,&size));
  PetscCallIgnore(VecCreate(ncomm,&x));
  PetscCallIgnore(VecSetSizes(x,N,PETSC_DECIDE));
  PetscCallIgnore(VecSetType(x,vtype));
  PetscCallIgnore(VecSetUp(x));
  PetscCallIgnore(VecDuplicate(x,&y));
  PetscCallIgnore(VecDuplicate(x,&w));
  PetscCallIgnore(VecSetRandom(x,NULL));
  PetscCallIgnore(VecSetRandom(y,NULL));

  PetscCallIgnore(VecWAXPY(w,3.0,x,y));

  for (PetscInt i=0; i<n; i++) {
    PetscCallIgnore(PetscTimeSubtract(&t));
    PetscCallIgnore(MPI_Barrier(ncomm));
    PetscCallIgnore(VecWAXPY(w,3.0,x,y));
    PetscCallIgnore(MPI_Barrier(ncomm));
    PetscCallIgnore(PetscTimeAdd(&t));
    tr = PetscMin(t,tr);
  }
  t = 1.e-6*3*N*sizeof(PetscScalar)/tr;
  PetscCallIgnore(MPI_Allreduce(&t,rate,1,MPI_DOUBLE,MPI_SUM,comm));
  PetscCallIgnore(VecDestroy(&x));
  PetscCallIgnore(VecDestroy(&y));
  PetscCallIgnore(VecDestroy(&w));
  PetscCallIgnore(MPI_Comm_free(&ncomm));
#if defined(PETSC_HAVE_SETJMP_H)
  PetscCallIgnore(PetscPopSignalHandler());
#endif
  PetscCallIgnore(PetscPopErrorHandler());
  *success = PETSC_TRUE;
  PetscFunctionReturn(0);
}
PetscErrorCode VecStreamsView_System(PetscViewer viewer,VecType vtype)
{
  PetscMPIInt       size;
  MPI_Comm          comm;
  PetscBool         success;

  PetscFunctionBegin;
  PetscCallIgnore(PetscObjectGetComm((PetscObject)viewer,&comm));
  PetscCallIgnore(MPI_Comm_size(comm,&size));
  if (!(size % 2)) {
    PetscSubcomm   sub;
    MPI_Comm       subcomm;
    PetscLogDouble rate[2];
    PetscMPIInt    rank;

    PetscCallIgnore(VecStreamsView_Private(comm,vtype,&rate[0],&success));
    if (!success) PetscFunctionReturn(0);

    PetscCallIgnore(MPI_Comm_rank(comm,&rank));
    PetscCallIgnore(PetscSubcommCreate(comm,&sub));
    PetscCallIgnore(PetscSubcommSetNumber(sub,2));
    PetscCallIgnore(PetscSubcommSetType(sub,PETSC_SUBCOMM_INTERLACED));
    subcomm = PetscSubcommChild(sub);
    if (!(rank % 2)) PetscCallIgnore(VecStreamsView_Private(subcomm,vtype,&rate[1],&success));
    PetscCallIgnore(PetscSubcommDestroy(&sub));
    if (!success) PetscFunctionReturn(0);

    if (rank == 0 /* && rate[0] < 1.7*rate[1] */) {
      PetscCallIgnore(PetscViewerASCIIPrintf(viewer,"**************************************************************************************************************************************\n"));
      PetscCallIgnore(PetscViewerASCIIPrintf(viewer,"The %s streams rate %g (MB/s) on all (%d) MPI ranks is not nearly two times the rate on half of the ranks %g.\n",vtype,rate[0],size,rate[1]));
      PetscCallIgnore(PetscViewerASCIIPrintf(viewer,"This indicates the memory bandwidth does not scale with the number of MPI ranks. You might be using more ranks than optimal.\n"));
      PetscCallIgnore(PetscViewerASCIIPrintf(viewer,"See: https://petsc.org/release/faq/#what-kind-of-parallel-computers-or-clusters-are-needed-to-use-petsc-or-why-do-i-get-little-speedup\n"));
      PetscCallIgnore(PetscViewerASCIIPrintf(viewer,"**************************************************************************************************************************************\n"));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscBenchRun_VecStreams(PetscBench ben)
{
  PetscViewer viewer = PETSC_VIEWER_STDOUT_WORLD;

  PetscFunctionBegin;
  PetscCallIgnore(VecStreamsView_System(viewer,VECSTANDARD));
  if (0 /* && PetscDeviceInitialized(PETSC_DEVICE_CUDA)*/) {
    PetscCallIgnore(VecStreamsView_System(viewer,VECCUDA));
  }
  PetscFunctionReturn(0);
}

/*@C

     PetscBenchVecStreamsCreate - create a benchmark object for a PETSc implementation of the streams benchmark

  Input Parameter:
.   comm - the communicator on which the benchmark would be run

  Output Parameter:
.   ben - the benchmark object

  Level: advanced

  References:
. * - McCalpin, John D. "Memory bandwidth and machine balance in current high performance computers." IEEE computer society technical committee on
  computer architecture (TCCA) newsletter 2, no. 19-25 (1995).

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchDestroy()`, `PetscBenchDestroy()`, `PetscBenchCompute()` `PetscBenchCreateVecStreams()`
@*/
PetscErrorCode PetscBenchVecStreamsCreate(MPI_Comm comm,PetscBench *ben)
{
  PetscBench_VecStreams *str;

  PetscFunctionBegin;
  PetscCall(PetscBenchCreate(comm,ben));
  PetscCall(PetscNewLog(*ben,&str));
  (*ben)->data = (void*)str;
  (*ben)->ops->run = PetscBenchRun_VecStreams;
  PetscFunctionReturn(0);
}
