static char help[] = "Tests asynchronous vector operations\n";

#include <petscvec.h>

static PetscErrorCode VecViewFromOptionsSynchronized(MPI_Comm comm, Vec v, PetscObject obj, const char name[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  /* Force each rank to print one at a time */
  for (PetscMPIInt i = 0; i < size; ++i) {
    ierr = MPI_Barrier(comm);CHKERRMPI(ierr);
    if (rank == i) {ierr = VecViewFromOptions(v,obj,name);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestVecEqual(Vec vref, Vec vtest, PetscInt N)
{
  PetscErrorCode    ierr;
  const PetscScalar *arrRef, *arrTest;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(vref,&arrRef);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vtest,&arrTest);CHKERRQ(ierr);
  for (PetscInt i = 0; i < N; ++i) {
    if (arrTest[i] != arrRef[i]) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Vectors don't match. testVector[%D]: %g != refVector[%D]: %g",i,(double)arrTest[i],i,(double)arrRef[i]);
    }
  }
  ierr = VecRestoreArrayRead(vref,&arrRef);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vtest,&arrTest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;
  PetscStreamType   stype=PETSCSTREAMCUDA;
  VecType           vtype=VECCUDA;
  PetscStream       pstream1,pstream2,pstream3;
  PetscStreamScalar pscal,pscal2;
  PetscInt          n=50;
  PetscInt          *idx;
  const PetscScalar one=1.0,four=4.0;
  PetscScalar       *arr;
  PetscMPIInt       rank,size;
  PetscBool         equal;
  Vec               seq1,seq2,seq1Async,seq2Async,mpi1,mpi2,mpi1Async,mpi2Async,seqReset,mpiReset;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Create seq vector */
  ierr = VecCreate(PETSC_COMM_SELF,&seq1);CHKERRQ(ierr);
  ierr = VecSetSizes(seq1,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(seq1,vtype);CHKERRQ(ierr);
  ierr = VecSetFromOptions(seq1);CHKERRQ(ierr);

  /* Create mpi vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&mpi1);CHKERRQ(ierr);
  ierr = VecSetSizes(mpi1,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(mpi1,vtype);CHKERRQ(ierr);
  ierr = VecSetFromOptions(mpi1);CHKERRQ(ierr);

  ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&arr);CHKERRQ(ierr);

  for (PetscInt i = 0; i < n; ++i) {
    idx[i] = i;
    arr[i] = (PetscScalar)(rank*n+i);
  }

  /* Initialize vectors */
  ierr = VecSetValues(seq1,n,idx,arr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(seq1);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(seq1);CHKERRQ(ierr);
  ierr = VecDuplicate(seq1,&seqReset);CHKERRQ(ierr);
  ierr = VecDuplicate(seq1,&seq2);CHKERRQ(ierr);
  ierr = VecDuplicate(seq1,&seq1Async);CHKERRQ(ierr);
  ierr = VecDuplicate(seq1,&seq2Async);CHKERRQ(ierr);
  ierr = VecCopy(seq1,seqReset);CHKERRQ(ierr);
  ierr = VecCopy(seqReset,seq2);CHKERRQ(ierr);
  ierr = VecCopy(seqReset,seq1Async);CHKERRQ(ierr);
  ierr = VecCopy(seqReset,seq2Async);CHKERRQ(ierr);

  for (PetscInt i = 0; i < n; ++i) idx[i] = rank*n+i;

  ierr = VecSetValues(mpi1,n,idx,arr,INSERT_VALUES);CHKERRQ(ierr);
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
  ierr = VecViewFromOptionsSynchronized(MPI_COMM_WORLD,seq1,NULL,"-vec_seq_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(mpi1,NULL,"-vec_mpi_view");CHKERRQ(ierr);

  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = PetscFree(arr);CHKERRQ(ierr);

  /* Create stream objects */
  ierr = PetscStreamCreate(&pstream1);CHKERRQ(ierr);
  ierr = PetscStreamSetType(pstream1,stype);CHKERRQ(ierr);
  ierr = PetscStreamSetMode(pstream1,PETSC_STREAM_DEFAULT_BLOCKING);CHKERRQ(ierr);
  ierr = PetscStreamSetUp(pstream1);CHKERRQ(ierr);
  ierr = PetscStreamDuplicate(pstream1,&pstream2);CHKERRQ(ierr);
  ierr = PetscStreamDuplicate(pstream1,&pstream3);CHKERRQ(ierr);

  ierr = PetscStreamScalarCreate(&pscal);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetType(pscal,stype);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetUp(pscal);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscal,&one,PETSC_MEMTYPE_HOST,pstream1);CHKERRQ(ierr);

  /* Initialize the host versions */
  for (PetscInt i = 0; i < 10; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      ierr = VecAXPY(seq1,1.0,seq2);CHKERRQ(ierr);
      ierr = VecAXPY(mpi1,1.0,mpi2);CHKERRQ(ierr);
    }
  }

  /* Test serialization of AXPY on different streams */
  for (PetscInt i = 0; i < 10; ++i) {
    ierr = VecAXPYAsync(seq1Async,pscal,seq2Async,pstream1);CHKERRQ(ierr);
    ierr = VecAXPYAsync(seq1Async,pscal,seq2Async,pstream2);CHKERRQ(ierr);
    ierr = VecAXPYAsync(seq1Async,pscal,seq2Async,pstream3);CHKERRQ(ierr);

    ierr = VecAXPYAsync(mpi1Async,pscal,mpi2Async,pstream1);CHKERRQ(ierr);
    ierr = VecAXPYAsync(mpi1Async,pscal,mpi2Async,pstream2);CHKERRQ(ierr);
    ierr = VecAXPYAsync(mpi1Async,pscal,mpi2Async,pstream3);CHKERRQ(ierr);
  }

  ierr = VecEqual(seq1,seq1Async,&equal);CHKERRQ(ierr);
  if (!equal) {ierr = TestVecEqual(seq1,seq1Async,n);CHKERRQ(ierr);}
  ierr = VecEqual(mpi1,mpi1Async,&equal);CHKERRQ(ierr);
  if (!equal) {ierr = TestVecEqual(mpi1,mpi1Async,n);CHKERRQ(ierr);}
  ierr = VecViewFromOptionsSynchronized(MPI_COMM_WORLD,seq1Async,NULL,"-vec_seq_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(mpi1Async,NULL,"-vec_mpi_view");CHKERRQ(ierr);

  /* Reset all the vectors */
  ierr = VecCopy(seqReset,seq1);CHKERRQ(ierr);
  ierr = VecCopy(seqReset,seq1Async);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi1);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi1Async);CHKERRQ(ierr);
  ierr = PetscStreamScalarDuplicate(pscal,&pscal2);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscal2,&one,PETSC_MEMTYPE_HOST,pstream1);CHKERRQ(ierr);

  {
    PetscScalar seqVal=1.0,mpiVal=1.0;
    /* Initialize the host versions */
    for (PetscInt i = 0; i < 10; ++i) {
      ierr = VecNorm(seq2,NORM_2,&seqVal);CHKERRQ(ierr);
      ierr = VecScale(seq2,seqVal);CHKERRQ(ierr);
      ierr = VecDot(seq1,seq2,&seqVal);CHKERRQ(ierr);
      ierr = VecAXPY(seq1,seqVal,seq2);CHKERRQ(ierr);
      ierr = VecAYPX(seq1,seqVal,seq2);CHKERRQ(ierr);

      ierr = VecNorm(mpi2,NORM_2,&mpiVal);CHKERRQ(ierr);
      ierr = VecScale(mpi2,mpiVal);CHKERRQ(ierr);
      ierr = VecDot(mpi1,mpi2,&mpiVal);CHKERRQ(ierr);
      ierr = VecAXPY(mpi1,mpiVal,mpi2);CHKERRQ(ierr);
      ierr = VecAYPX(mpi1,mpiVal,mpi2);CHKERRQ(ierr);
    }
  }

  /* Test serialization of various vector routines on different streams */
  for (PetscInt i = 0; i < 10; ++i) {
    ierr = VecNormAsync(seq2Async,NORM_2,&pscal,pstream1);CHKERRQ(ierr);
    ierr = VecScaleAsync(seq2Async,pscal,pstream2);CHKERRQ(ierr);
    ierr = VecDotAsync(seq1Async,seq2Async,pscal,pstream1);CHKERRQ(ierr);
    ierr = VecAXPYAsync(seq1Async,pscal,seq2Async,pstream3);CHKERRQ(ierr);
    ierr = VecAYPXAsync(seq1Async,pscal,seq2Async,pstream2);CHKERRQ(ierr);

    ierr = VecNormAsync(mpi2Async,NORM_2,&pscal2,pstream1);CHKERRQ(ierr);
    ierr = VecScaleAsync(mpi2Async,pscal2,pstream2);CHKERRQ(ierr);
    ierr = VecDotAsync(mpi1Async,mpi2Async,pscal2,pstream1);CHKERRQ(ierr);
    ierr = VecAXPYAsync(mpi1Async,pscal2,mpi2Async,pstream3);CHKERRQ(ierr);
    ierr = VecAYPXAsync(mpi1Async,pscal2,mpi2Async,pstream2);CHKERRQ(ierr);
  }


  ierr = VecEqual(seq1,seq1Async,&equal);CHKERRQ(ierr);
  if (!equal) {ierr = TestVecEqual(seq1,seq1Async,n);CHKERRQ(ierr);}
  ierr = VecEqual(seq2,seq2Async,&equal);CHKERRQ(ierr);
  if (!equal) {ierr = TestVecEqual(seq2,seq2Async,n);CHKERRQ(ierr);}
  ierr = VecEqual(mpi1,mpi1Async,&equal);CHKERRQ(ierr);
  if (!equal) {ierr = TestVecEqual(mpi1,mpi1Async,n);CHKERRQ(ierr);}
  ierr = VecEqual(mpi2,mpi2Async,&equal);CHKERRQ(ierr);
  if (!equal) {ierr = TestVecEqual(mpi2,mpi2Async,n);CHKERRQ(ierr);}
  ierr = VecViewFromOptionsSynchronized(MPI_COMM_WORLD,seq1Async,NULL,"-vec_seq_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(mpi1Async,NULL,"-vec_mpi_view");CHKERRQ(ierr);

  /* Reset all the vectors */
  ierr = VecCopy(seqReset,seq1);CHKERRQ(ierr);
  ierr = VecCopy(seqReset,seq2);CHKERRQ(ierr);
  ierr = VecCopy(seqReset,seq1Async);CHKERRQ(ierr);
  ierr = VecCopy(seqReset,seq2Async);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi1);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi2);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi1Async);CHKERRQ(ierr);
  ierr = VecCopy(mpiReset,mpi2Async);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscal,&four,PETSC_MEMTYPE_HOST,pstream1);CHKERRQ(ierr);

  /* Initialize the host versions */
  for (PetscInt i = 0; i < 10; ++i) {
    for (PetscInt j = 0; j < 4; ++j) {
      ierr = VecAXPY(seq1,4.0,seq2);CHKERRQ(ierr);
      ierr = VecAXPY(mpi1,4.0,mpi2);CHKERRQ(ierr);
    }
  }

  /* Test interleaving of regular and asynchronous AXPY */
  for (PetscInt i = 0; i < 10; ++i) {
    ierr = VecAXPYAsync(seq1Async,pscal,seq2Async,pstream1);CHKERRQ(ierr);
    ierr = VecAXPYAsync(seq1Async,pscal,seq2Async,pstream2);CHKERRQ(ierr);
    ierr = VecAXPY(seq1Async,four,seq2Async);CHKERRQ(ierr);
    ierr = VecAXPYAsync(seq1Async,pscal,seq2Async,pstream3);CHKERRQ(ierr);

    ierr = VecAXPYAsync(mpi1Async,pscal,mpi2Async,pstream1);CHKERRQ(ierr);
    ierr = VecAXPYAsync(mpi1Async,pscal,mpi2Async,pstream2);CHKERRQ(ierr);
    ierr = VecAXPY(mpi1Async,four,mpi2Async);CHKERRQ(ierr);
    ierr = VecAXPYAsync(mpi1Async,pscal,mpi2Async,pstream3);CHKERRQ(ierr);
  }

  ierr = VecEqual(seq1,seq1Async,&equal);CHKERRQ(ierr);
  if (!equal) {ierr = TestVecEqual(seq1,seq1Async,n);CHKERRQ(ierr);}
  ierr = VecEqual(mpi1,mpi1Async,&equal);CHKERRQ(ierr);
  if (!equal) {ierr = TestVecEqual(mpi1,mpi1Async,n);CHKERRQ(ierr);}
  ierr = VecViewFromOptionsSynchronized(MPI_COMM_WORLD,seq1Async,NULL,"-vec_seq_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(mpi1Async,NULL,"-vec_mpi_view");CHKERRQ(ierr);

  ierr = PetscStreamScalarDestroy(&pscal);CHKERRQ(ierr);
  ierr = PetscStreamScalarDestroy(&pscal2);CHKERRQ(ierr);
  ierr = PetscStreamDestroy(&pstream1);CHKERRQ(ierr);
  ierr = PetscStreamDestroy(&pstream2);CHKERRQ(ierr);
  ierr = PetscStreamDestroy(&pstream3);CHKERRQ(ierr);
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
