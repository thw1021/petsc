
#include <petscsys.h>
#include <petscvec.h>         /*I  "petscvec.h"  I*/

static PetscErrorCode VecStreamsView_Private(MPI_Comm comm,VecType vtype,PetscLogDouble *rate)
{
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  Vec               x,y,w;
  PetscInt          N = 40000000,n=4;
  PetscLogDouble    t = 0, tr = PETSC_MAX_REAL;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,N,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(x,vtype);CHKERRQ(ierr);
  ierr = VecSetUp(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
  ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
  ierr = VecSetRandom(y,NULL);CHKERRQ(ierr);

  ierr = VecWAXPY(w,3.0,x,y);CHKERRQ(ierr);

  for (PetscInt i=0; i<n; i++) {
    ierr = MPI_Barrier(comm);CHKERRMPI(ierr);
    ierr = PetscTimeSubtract(&t);CHKERRQ(ierr);
    ierr = VecWAXPY(w,3.0,x,y);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&t);CHKERRQ(ierr);
    tr = PetscMin(t,tr);
  }
  t = 1.e-6*3*N*sizeof(PetscScalar)/tr;
  ierr = MPI_Allreduce(&t,rate,1,MPI_DOUBLE,MPI_SUM,comm);CHKERRMPI(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecStreamsView_System(PetscViewer viewer,VecType vtype)
{
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  MPI_Comm          comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (!(size % 2)) {
    PetscSubcomm   sub;
    MPI_Comm       subcomm;
    PetscLogDouble rate[2];
    PetscMPIInt    rank;
    PetscBool      cpu = PETSC_TRUE;

    ierr = VecStreamsView_Private(comm,vtype,&rate[0]);CHKERRQ(ierr);

    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = PetscSubcommCreate(comm,&sub);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(sub,2);CHKERRQ(ierr);
    ierr = PetscSubcommSetType(sub,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
    subcomm = PetscSubcommChild(sub);CHKERRQ(ierr);
    if (!(rank % 2)) {
      ierr = VecStreamsView_Private(subcomm,vtype,&rate[1]);CHKERRQ(ierr);
    }
    ierr = PetscSubcommDestroy(&sub);CHKERRQ(ierr);

    if (rank == 0 && rate[0] < 1.7*rate[1]) {
      ierr = PetscStrcmp(vtype,"standard",&cpu);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"**************************************************************************************************************************************\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"The %s streams rate %g (MB/s) on all (%d) MPI ranks is not nearly two times the rate on half of the ranks %g.\n",cpu ? "CPU" : "GPU",rate[0],size,rate[1]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"This indicates the memory bandwidth does not scale with the number of MPI ranks. You might be using more ranks than optimal.\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"See: https://petsc.org/release/faq/#what-kind-of-parallel-computers-or-clusters-are-needed-to-use-petsc-or-why-do-i-get-little-speedup\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"**************************************************************************************************************************************\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C

     VecStreamsView - run a streams-like benchmark on am MPI communicator and half of that communicator obtained from a viewer and report the results to the viewer

  Input Parameter:
.   viewer - the ASCII viewer

@*/
PetscErrorCode VecStreamsView(PetscViewer viewer)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecStreamsView_System(viewer,VECSTANDARD);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  ierr = VecStreamsView_System(viewer,VECCUDA);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
