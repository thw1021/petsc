
static char help[] = "Tests MatInvertVariableBlockEnvelope()\n\n";

#include <petscmat.h>
extern PetscErrorCode MatIsDiagonal(Mat);
extern PetscErrorCode BuildMatrix(const PetscInt*, PetscInt,const PetscInt *,Mat*);

int main(int argc,char **argv)
{
  Mat            A,C,D,F;
  PetscErrorCode ierr;
  PetscInt       i,j,rows[2],*parts,cnt, N = 21,nblocks,*blocksizes;
  PetscScalar    values[2][2],rand;
  PetscRandom    rctx;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_DENSE);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,6,18);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  values[0][0] = 2; values[0][1] = 1;
  values[1][0] = 1; values[1][1] = 2;
  for (i=0;i<3;i++){
    rows[0] = 2*i; rows[1] = 2*i + 1;
    ierr = MatSetValues(C,2,rows,2,rows,(PetscScalar*)values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatMatTransposeMult(C,C,MAT_INITIAL_MATRIX,PETSC_DETERMINE,&A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatInvertVariableBlockEnvelope(A,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatMatMult(A,D,MAT_INITIAL_MATRIX,1.0,&F);CHKERRQ(ierr);
  ierr = MatView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatIsDiagonal(F);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rctx);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscMalloc1(size,&parts);CHKERRQ(ierr);

  for (j=0; j<1024; j++) {
    cnt = 0;
    for (i=0; i<size-1; i++) {
      ierr = PetscRandomGetValue(rctx,&rand);CHKERRQ(ierr);
      parts[i] = (PetscInt) N*rand;
      parts[i] = PetscMin(parts[i],N-cnt);
      cnt      += parts[i];
    }
    parts[size-1] = N - cnt;

    ierr = PetscRandomGetValue(rctx,&rand);CHKERRQ(ierr);
    nblocks = rand*10;
    nblocks = PetscMax(nblocks,2);
    cnt = 0;
    ierr = PetscMalloc1(nblocks,&blocksizes);CHKERRQ(ierr);
    for (i=0; i<nblocks-1; i++) {
      ierr = PetscRandomGetValue(rctx,&rand);CHKERRQ(ierr);
      blocksizes[i] = PetscMax(1,(PetscInt) N*rand);
      blocksizes[i] = PetscMin(blocksizes[i],N-cnt);
      cnt      += blocksizes[i];
      if (cnt == N) {
        nblocks = i + 1;
        break;
      }
    }
    if (cnt < N) {
      blocksizes[nblocks-1] = N - cnt;
    }

    ierr = BuildMatrix(parts,nblocks,blocksizes,&A);CHKERRQ(ierr);
    ierr = PetscFree(blocksizes);CHKERRQ(ierr);
    // ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = MatInvertVariableBlockEnvelope(A,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    // ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = MatMatMult(A,D,MAT_INITIAL_MATRIX,1.0,&F);CHKERRQ(ierr);
    // ierr = MatView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatIsDiagonal(F);CHKERRQ(ierr);

    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
  }
  ierr = PetscFree(parts);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode MatIsDiagonal(Mat A)
{
  PetscErrorCode    ierr;
  PetscInt          ncols,i,j,rstart,rend;
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscBool         founddiag;

  PetscFunctionBeginUser;
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    founddiag = PETSC_FALSE;
    ierr = MatGetRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++) {
      if (cols[j] == i){
        if (PetscAbsScalar(vals[j] - 1) > PETSC_SMALL) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Row %D does not have 1 on the diagonal, it has %g",i,vals[j]);
        founddiag = PETSC_TRUE;
      } else {
        if (PetscAbsScalar(vals[j]) > PETSC_SMALL) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Row %D has off-diagonal value %g at %D",i,vals[j],cols[j]);
      }
    }
    ierr = MatRestoreRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    if (!founddiag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Row %D does not have diagonal entrie",i);
  }
  PetscFunctionReturn(0);
}

/*
    All processes receive all the block information
*/
PetscErrorCode BuildMatrix(const PetscInt *parts, PetscInt nblocks, const PetscInt *blocksizes,Mat *A)
{
  PetscErrorCode    ierr;
  PetscInt          i,cnt = 0;
  PetscMPIInt       rank;

  PetscFunctionBeginUser;
  // ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of blocks %D: blocks:",nblocks);CHKERRQ(ierr);
  // for (i=0; i<nblocks; i++)   ierr = PetscPrintf(PETSC_COMM_WORLD," %D",blocksizes[i]);CHKERRQ(ierr);
  // ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,parts[rank],parts[rank],PETSC_DETERMINE,PETSC_DETERMINE,0,NULL,0,NULL,A);CHKERRQ(ierr);
  ierr = MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
  if (!rank) {
    for (i=0; i<nblocks; i++) {
      ierr = MatSetValue(*A,cnt,cnt+blocksizes[i]-1,1.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(*A,cnt+blocksizes[i]-1,cnt,1.0,INSERT_VALUES);CHKERRQ(ierr);
      cnt += blocksizes[i];
    }
  }
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(*A,10);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:

   test:
     suffix: 2
     nsize: 2

   test:
     suffix: 5
     nsize: 5

TEST*/
