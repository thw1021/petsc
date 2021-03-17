static char help[] = "Tests asynchronous vector operations\n";

#include <petscvec.h>

PETSC_STATIC_INLINE PetscErrorCode VecViewFromOptionsSynchronized(MPI_Comm comm, Vec v, PetscObject obj, const char name[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size,sizeWorld;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&sizeWorld);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == sizeWorld) {
    ierr = VecViewFromOptions(v,obj,name);CHKERRQ(ierr);
  } else {
    PetscMPIInt rank;

    /* Force each rank to print one at a time */
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    for (PetscMPIInt i = 0; i < size; ++i) {
      ierr = MPI_Barrier(comm);CHKERRMPI(ierr);
      if (rank == i) {ierr = VecViewFromOptions(v,obj,name);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode VecClose(Vec vref, Vec vtest)
{
  PetscErrorCode    ierr;
  PetscBool         equal;
  PetscInt          n;
  const PetscScalar *arrRef, *arrTest;

  PetscFunctionBegin;
  ierr = VecEqual(vref,vtest,&equal);CHKERRQ(ierr);
  if (equal) PetscFunctionReturn(0);
  ierr = VecGetLocalSize(vref,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vref,&arrRef);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vtest,&arrTest);CHKERRQ(ierr);
  for (PetscInt i = 0; i < n; ++i) {
    const PetscReal realRef = PetscRealPart(arrRef[i]), realTest = PetscRealPart(arrTest[i]);
    if (!PetscIsCloseAtTol(realRef,realTest,1e-7,1e-7)) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Vectors don't match. refVector[%D]: %.10g != testVector[%D]: %.10g",i,(double)realRef,i,(double)realTest);
    }
  }
  ierr = VecRestoreArrayRead(vref,&arrRef);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vtest,&arrTest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Because apparently const PetscInt n=3;PetscInt arr[n]; is not a constant enough expression for mswin cl to be able to
 set array size statically...
*/
#if !defined(THREE) && !defined(TWO)
#define THREE 3
#define TWO   2
#else
#error "THREE and TWO are already defined"
#endif

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  const PetscInt    nscal=TWO,nstream=THREE,nloop=9;
  PetscInt          n=50;
  PetscStreamType   stype=PETSCSTREAMCUDA;
  VecType           vtype=VECCUDA;
  PetscStream       pstream[THREE];
  PetscStreamScalar pscal[TWO];
  const PetscScalar one=1.0;
  PetscMPIInt       rank,size;
  Vec               seq1,seq2,seq1Async,seq2Async,seqReset;
  Vec               mpi1,mpi2,mpi1Async,mpi2Async,mpiReset;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* Create seq vector */
  ierr = VecCreate(PETSC_COMM_SELF,&seq1);CHKERRQ(ierr);
  ierr = VecSetSizes(seq1,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(seq1,vtype);CHKERRQ(ierr);
  ierr = VecSetRandom(seq1,NULL);CHKERRQ(ierr);
  ierr = VecSetFromOptions(seq1);CHKERRQ(ierr);

  /* Create mpi vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&mpi1);CHKERRQ(ierr);
  ierr = VecSetSizes(mpi1,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(mpi1,vtype);CHKERRQ(ierr);
  ierr = VecSetRandom(mpi1,NULL);CHKERRQ(ierr);
  ierr = VecSetFromOptions(mpi1);CHKERRQ(ierr);

  /* Assmeble sequential vectors */
  ierr = VecAssemblyBegin(seq1);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(seq1);CHKERRQ(ierr);
  ierr = VecDuplicate(seq1,&seqReset);CHKERRQ(ierr);
  ierr = VecDuplicate(seq1,&seq2);CHKERRQ(ierr);
  ierr = VecDuplicate(seq1,&seq1Async);CHKERRQ(ierr);
  ierr = VecDuplicate(seq1,&seq2Async);CHKERRQ(ierr);
  ierr = VecCopy(seq1,seqReset);CHKERRQ(ierr);
  ierr = VecCopy(seq1,seq2);CHKERRQ(ierr);
  ierr = VecCopy(seq1,seq1Async);CHKERRQ(ierr);
  ierr = VecCopy(seq1,seq2Async);CHKERRQ(ierr);

  /* Assemble parallel vectors */
  ierr = VecAssemblyBegin(mpi1);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(mpi1);CHKERRQ(ierr);
  ierr = VecDuplicate(mpi1,&mpiReset);CHKERRQ(ierr);
  ierr = VecDuplicate(mpi1,&mpi2);CHKERRQ(ierr);
  ierr = VecDuplicate(mpi1,&mpi1Async);CHKERRQ(ierr);
  ierr = VecDuplicate(mpi1,&mpi2Async);CHKERRQ(ierr);
  ierr = VecCopy(mpi1,mpiReset);CHKERRQ(ierr);
  ierr = VecCopy(mpi1,mpi2);CHKERRQ(ierr);
  ierr = VecCopy(mpi1,mpi1Async);CHKERRQ(ierr);
  ierr = VecCopy(mpi1,mpi2Async);CHKERRQ(ierr);

  /* Create stream objects */
  ierr = PetscStreamCreate(&pstream[0]);CHKERRQ(ierr);
  ierr = PetscStreamSetType(pstream[0],stype);CHKERRQ(ierr);
  ierr = PetscStreamSetMode(pstream[0],PETSC_STREAM_DEFAULT_BLOCKING);CHKERRQ(ierr);
  ierr = PetscStreamSetUp(pstream[0]);CHKERRQ(ierr);
  ierr = PetscStreamDuplicate(pstream[0],&(pstream[1]));CHKERRQ(ierr);
  ierr = PetscStreamDuplicate(pstream[0],&(pstream[2]));CHKERRQ(ierr);

  /* Create stream scalar objects */
  ierr = PetscStreamScalarCreate(&pscal[0]);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetType(pscal[0],stype);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetUp(pscal[0]);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscal[0],&one,PETSC_MEMTYPE_HOST,pstream[0]);CHKERRQ(ierr);

  /* Initialize the host versions */
  for (PetscInt i = 0; i < nloop; ++i) {
    ierr = VecAXPY(seq1,1.0,seq2);CHKERRQ(ierr);
    ierr = VecAXPY(mpi1,1.0,mpi2);CHKERRQ(ierr);
  }

  /* Test asynchronous AXPY on same stream */
  for (PetscInt i = 0; i < nloop; ++i) {
    ierr = VecAXPYAsync(seq1Async,pscal[0],seq2Async,pstream[0]);CHKERRQ(ierr);
    ierr = VecAXPYAsync(mpi1Async,pscal[0],mpi2Async,pstream[1]);CHKERRQ(ierr);
  }
  ierr = PetscStreamSynchronize(pstream[0]);CHKERRQ(ierr);
  ierr = PetscStreamSynchronize(pstream[1]);CHKERRQ(ierr);

  ierr = VecClose(seq1,seq1Async);CHKERRQ(ierr);
  ierr = VecClose(mpi1,mpi1Async);CHKERRQ(ierr);

  /* Reset some vectors, no need to redo host */
  ierr = VecCopy(seqReset,seq1Async);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi1Async);CHKERRQ(ierr);

  /* Test serialization of AXPY on different streams */
  for (PetscInt i = 0; i < nloop; ++i) {
    const PetscInt stri = i%nstream;
    /* Don't need to wait on first loop */
    if (!i) {
      PetscEvent event;
      ierr = VecGetEvent(seq1Async,&event);CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(pstream[stri],event);CHKERRQ(ierr);
      ierr = VecRestoreEvent(seq1Async,&event);CHKERRQ(ierr);
    }
    ierr = VecAXPYAsync(seq1Async,pscal[0],seq2Async,pstream[stri]);CHKERRQ(ierr);
    if (!i) {
      PetscEvent event;
      ierr = VecGetEvent(mpi1Async,&event);CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(pstream[stri],event);CHKERRQ(ierr);
      ierr = VecRestoreEvent(mpi1Async,&event);CHKERRQ(ierr);
    }
    ierr = VecAXPYAsync(mpi1Async,pscal[0],mpi2Async,pstream[stri]);CHKERRQ(ierr);
    /* i is loop-only variable, so do sync here */
    if (i == nloop-1) {ierr = PetscStreamSynchronize(pstream[stri]);CHKERRQ(ierr);}
  }

  ierr = VecClose(seq1,seq1Async);CHKERRQ(ierr);
  ierr = VecClose(mpi1,mpi1Async);CHKERRQ(ierr);

  /* Reset all the vectors */
  ierr = VecCopy(seqReset,seq1);CHKERRQ(ierr);
  ierr = VecCopy(seqReset,seq1Async);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi1);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi1Async);CHKERRQ(ierr);
  ierr = PetscStreamScalarDuplicate(pscal[0],&pscal[1]);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscal[1],&one,PETSC_MEMTYPE_HOST,pstream[1]);CHKERRQ(ierr);

  {
    PetscScalar seqVal=1.0,mpiVal=1.0;
    PetscReal   seqNorm=1.0,mpiNorm=1.0;

    /* Initialize the host versions, only need to do it once */
    for (PetscInt i = 0; i < nloop; ++i) {
      ierr = VecNorm(seq1,NORM_2,&seqNorm);CHKERRQ(ierr);
      seqVal = (PetscScalar)(1.0/seqNorm);
      ierr = VecScale(seq1,seqVal);CHKERRQ(ierr);
      ierr = VecDot(seq1,seq2,&seqVal);CHKERRQ(ierr);
      seqVal = -seqVal;
      ierr = VecAXPY(seq2,seqVal,seq1);CHKERRQ(ierr);
      ierr = VecNorm(seq2,NORM_2,&seqNorm);CHKERRQ(ierr);
      seqVal = (PetscScalar)(1.0/seqNorm);
      ierr = VecScale(seq2,seqVal);CHKERRQ(ierr);

      ierr = VecNorm(mpi1,NORM_2,&mpiNorm);CHKERRQ(ierr);
      mpiVal = (PetscScalar)(1.0/mpiNorm);
      ierr = VecScale(mpi1,mpiVal);CHKERRQ(ierr);
      ierr = VecDot(mpi1,mpi2,&mpiVal);CHKERRQ(ierr);
      mpiVal = -mpiVal;
      ierr = VecAXPY(mpi2,mpiVal,mpi1);CHKERRQ(ierr);
      ierr = VecNorm(mpi2,NORM_2,&mpiNorm);CHKERRQ(ierr);
      mpiVal = (PetscScalar)(1.0/mpiNorm);
      ierr = VecScale(mpi2,mpiVal);CHKERRQ(ierr);
    }
  }

  /* Test serialization of various vector routines to orthogonalize vectors on same streams. The loop is done here to
   try and get the streams to overlap as much as possible. */
  for (PetscInt i = 0; i < nloop; ++i) {
    ierr = VecNormAsync(seq1Async,NORM_2,&pscal[0],pstream[0]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAYDX(1.0,NULL,pscal[0],pstream[0]);CHKERRQ(ierr);
    ierr = VecScaleAsync(seq1Async,pscal[0],pstream[0]);CHKERRQ(ierr);
    ierr = VecDotAsync(seq1Async,seq2Async,pscal[0],pstream[0]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAXTY(-1.0,pscal[0],NULL,pstream[0]);CHKERRQ(ierr);
    ierr = VecAXPYAsync(seq2Async,pscal[0],seq1Async,pstream[0]);CHKERRQ(ierr);
    ierr = VecNormAsync(seq2Async,NORM_2,&pscal[0],pstream[0]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAYDX(1.0,NULL,pscal[0],pstream[0]);CHKERRQ(ierr);
    ierr = VecScaleAsync(seq2Async,pscal[0],pstream[0]);CHKERRQ(ierr);

    ierr = VecNormAsync(mpi1Async,NORM_2,&pscal[1],pstream[1]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAYDX(1.0,NULL,pscal[1],pstream[1]);CHKERRQ(ierr);
    ierr = VecScaleAsync(mpi1Async,pscal[1],pstream[1]);CHKERRQ(ierr);
    ierr = VecDotAsync(mpi1Async,mpi2Async,pscal[1],pstream[1]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAXTY(-1.0,pscal[1],NULL,pstream[1]);CHKERRQ(ierr);
    ierr = VecAXPYAsync(mpi2Async,pscal[1],mpi1Async,pstream[1]);CHKERRQ(ierr);
    ierr = VecNormAsync(mpi2Async,NORM_2,&pscal[1],pstream[1]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAYDX(1.0,NULL,pscal[1],pstream[1]);CHKERRQ(ierr);
    ierr = VecScaleAsync(mpi2Async,pscal[1],pstream[1]);CHKERRQ(ierr);
  }
  ierr = PetscStreamSynchronize(pstream[0]);CHKERRQ(ierr);
  ierr = PetscStreamSynchronize(pstream[1]);CHKERRQ(ierr);

  ierr = VecClose(seq1,seq1Async);CHKERRQ(ierr);
  ierr = VecClose(seq2,seq2Async);CHKERRQ(ierr);
  ierr = VecClose(mpi1,mpi1Async);CHKERRQ(ierr);
  ierr = VecClose(mpi2,mpi2Async);CHKERRQ(ierr);

  /* Test serialization of various vector routines to orthogonalize vectors on multiple streams */
  for (PetscInt i = 0; i < nloop; ++i) {
    const PetscInt stri = i%nstream;

    if (!i) {
      PetscEvent event;
      ierr = VecGetEvent(seq2Async,&event);CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(pstream[stri],event);CHKERRQ(ierr);
      ierr = VecRestoreEvent(seq2Async,&event);CHKERRQ(ierr);
      ierr = VecCopyAsync(seqReset,seq1Async,pstream[stri]);CHKERRQ(ierr);
      ierr = VecCopyAsync(seqReset,seq2Async,pstream[stri]);CHKERRQ(ierr);
    }
    ierr = VecNormAsync(seq1Async,NORM_2,&pscal[0],pstream[stri]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAYDX(1.0,NULL,pscal[0],pstream[stri]);CHKERRQ(ierr);
    ierr = VecScaleAsync(seq1Async,pscal[0],pstream[stri]);CHKERRQ(ierr);
    ierr = VecDotAsync(seq1Async,seq2Async,pscal[0],pstream[stri]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAXTY(-1.0,pscal[0],NULL,pstream[stri]);CHKERRQ(ierr);
    ierr = VecAXPYAsync(seq2Async,pscal[0],seq1Async,pstream[stri]);CHKERRQ(ierr);
    ierr = VecNormAsync(seq2Async,NORM_2,&pscal[0],pstream[stri]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAYDX(1.0,NULL,pscal[0],pstream[stri]);CHKERRQ(ierr);
    ierr = VecScaleAsync(seq2Async,pscal[0],pstream[stri]);CHKERRQ(ierr);

    if (!i) {
      PetscEvent event;
      ierr = VecGetEvent(mpi2Async,&event);CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(pstream[stri],event);CHKERRQ(ierr);
      ierr = VecRestoreEvent(mpi2Async,&event);CHKERRQ(ierr);
      ierr = VecCopyAsync(mpiReset,mpi1Async,pstream[stri]);CHKERRQ(ierr);
      ierr = VecCopyAsync(mpiReset,mpi2Async,pstream[stri]);CHKERRQ(ierr);
    }
    ierr = VecNormAsync(mpi1Async,NORM_2,&pscal[1],pstream[stri]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAYDX(1.0,NULL,pscal[1],pstream[stri]);CHKERRQ(ierr);
    ierr = VecScaleAsync(mpi1Async,pscal[1],pstream[stri]);CHKERRQ(ierr);
    ierr = VecDotAsync(mpi1Async,mpi2Async,pscal[1],pstream[stri]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAXTY(-1.0,pscal[1],NULL,pstream[stri]);CHKERRQ(ierr);
    ierr = VecAXPYAsync(mpi2Async,pscal[1],mpi1Async,pstream[stri]);CHKERRQ(ierr);
    ierr = VecNormAsync(mpi2Async,NORM_2,&pscal[1],pstream[stri]);CHKERRQ(ierr);
    ierr = PetscStreamScalarAYDX(1.0,NULL,pscal[1],pstream[stri]);CHKERRQ(ierr);
    ierr = VecScaleAsync(mpi2Async,pscal[1],pstream[stri]);CHKERRQ(ierr);
    if (i == nloop-1) {ierr = PetscStreamSynchronize(pstream[stri]);CHKERRQ(ierr);}
  }

  ierr = VecClose(seq1,seq1Async);CHKERRQ(ierr);
  ierr = VecClose(seq2,seq2Async);CHKERRQ(ierr);
  ierr = VecClose(mpi1,mpi1Async);CHKERRQ(ierr);
  ierr = VecClose(mpi2,mpi2Async);CHKERRQ(ierr);

  ierr = VecViewFromOptionsSynchronized(PETSC_COMM_WORLD,seq1Async,NULL,"-vec_seq_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(mpi1Async,NULL,"-vec_mpi_view");CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"All operations completed sucessfully\n");CHKERRQ(ierr);
  for (PetscInt i = 0; i < nscal; ++i) {ierr = PetscStreamScalarDestroy(&pscal[i]);CHKERRQ(ierr);}
  for (PetscInt i = 0; i < nstream; ++i) {ierr = PetscStreamDestroy(&pstream[i]);CHKERRQ(ierr);}
  ierr = VecDestroy(&seq1);CHKERRQ(ierr);
  ierr = VecDestroy(&seq2);CHKERRQ(ierr);
  ierr = VecDestroy(&seq1Async);CHKERRQ(ierr);
  ierr = VecDestroy(&seq2Async);CHKERRQ(ierr);
  ierr = VecDestroy(&seqReset);CHKERRQ(ierr);
  ierr = VecDestroy(&mpi1);CHKERRQ(ierr);
  ierr = VecDestroy(&mpi2);CHKERRQ(ierr);
  ierr = VecDestroy(&mpi1Async);CHKERRQ(ierr);
  ierr = VecDestroy(&mpi2Async);CHKERRQ(ierr);
  ierr = VecDestroy(&mpiReset);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

#undef THREE
#undef TWO

/*TEST

 testset:
   requires: cuda
   nsize: {{1 2}}
   suffix: cuda
   args: -vec_type cuda
   test:
     args: -n {{50 10000}}

TEST*/
