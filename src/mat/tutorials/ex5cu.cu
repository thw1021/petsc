static char help[] = "Serial test of Cuda matrix assemble with 1D Laplacian.\n\n";

#include <petscconf.h>
#include <petscmat.h>
#include <petscaijdevice.h>
#include <petsccublas.h>


__global__
void assemble_device(PetscSplitCSRDataStructure *d_mat, PetscInt start, PetscInt end, PetscInt N, PetscMPIInt rank)
{
  const PetscInt  inc = blockDim.x, my0 = threadIdx.x;
  PetscInt        i;
  PetscScalar     values[] = {-1,2.1,-1};
  PetscErrorCode  ierr;
  for (i=start+my0; i<end; i+=inc) {
    if (i==0) {
      PetscInt js[] = {0, 1};
      MatSetValuesDevice(d_mat,1,&i,2,js,&values[1],INSERT_VALUES,&ierr);
      if (ierr) return;
    } else if (i==N-1) {
      PetscInt js[] = {i-1, i};
      MatSetValuesDevice(d_mat,1,&i,2,js,values,INSERT_VALUES,&ierr);
      if (ierr) return;
    } else {
      PetscInt js[] = {i-1, i, i+1};
      MatSetValuesDevice(d_mat,1,&i,3,js,values,INSERT_VALUES,&ierr);
      if (ierr) return;
    }
  }
}


void assemble_mat(Mat A, PetscInt start, PetscInt end, PetscInt N, PetscMPIInt rank)
{
  PetscInt        i;
  PetscScalar     values[] = {-1,2.1,-1};
  PetscErrorCode  ierr;
  for (i=start; i<end; i++) {
    if (i==0) {
      PetscInt js[] = {0, 1};
      ierr = MatSetValues(A,1,&i,2,js,&values[1],INSERT_VALUES);
      if (ierr) return;
    } else if (i==N-1) {
      PetscInt js[] = {i-1, i};
      ierr = MatSetValues(A,1,&i,2,js,values,INSERT_VALUES);
      if (ierr) return;
    } else {
      PetscInt js[] = {i-1, i, i+1};
      ierr = MatSetValues(A,1,&i,3,js,values,INSERT_VALUES);
      if (ierr) return;
    }
  }
}

int main(int argc,char **args)
{
  PetscErrorCode               ierr;
  Mat                          A;
  PetscInt                     N=3, nz=3, Istart, Iend;
  PetscSplitCSRDataStructure   *d_mat;
  PetscLogEvent                event;
  Vec                          x,y;
  cudaError_t                  cerr;
  PetscMPIInt                  rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-nz_row", &nz, NULL);CHKERRQ(ierr); // does not work?
  ierr = PetscOptionsGetInt(NULL,NULL, "-n", &N, NULL);CHKERRQ(ierr);
  if (nz>N) {
    PetscPrintf(PETSC_COMM_WORLD,"warning decreasing nz\n");
    nz=N;
  }
  ierr = PetscLogEventRegister("GPU operator", MAT_CLASSID, &event);CHKERRQ(ierr);
  ierr = MatCreateAIJCUSPARSE(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,nz,NULL,nz-1,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);

  // assemble end on CPU
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  assemble_mat(A, Istart, Iend, N, rank);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // test cusparse
  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = VecViewFromOptions(y,NULL,"-vec_view");CHKERRQ(ierr);

  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  ierr = MatCUSPARSEGetDeviceMatWrite(A,&d_mat);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  assemble_device<<<1,1>>>(d_mat, Istart, Iend, N, rank);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  fflush(stdout);
  // assemble end on CPU for now
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = VecViewFromOptions(y,NULL,"-vec_view");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: cuda !define(PETSC_USE_CTABLE)

   test:
      suffix: 0
      args: -n 11 -mat_view -vec_view
      nsize:  2

TEST*/
