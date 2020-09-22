static char help[] = "Serial test of Cuda matrix assemble with 1D Laplacian.\n\n";

#include <petscmat.h>
#include <petscaijdevice.h>
#include <petsccublas.h>

__global__
void assemble(PetscInt start, PetscInt end, PetscInt N, PetscMPIInt rank, PetscSplitCSRDataStructure *d_mat)
{
  const PetscInt  inc = blockDim.x, my0 = threadIdx.x;
  PetscInt        i;
  PetscScalar     values[] = {-1,2,-1};
  PetscErrorCode  ierr;
  printf("[%d.%d] start=%d end=%d N=%d eq=%d\n",rank,my0,start,end,N,start+my0);
  for (i=start+my0; i<end; i+=inc) {
    if (i==0) {
      PetscInt js[] = {0, 1};
      MatSetValues_AIJ_device(d_mat,1,&i,2,js,&values[1],INSERT_VALUES,&ierr);
      printf("\tb[%d.%d] inc=%d. add row %d. (%d)\n",rank,my0,inc,i,ierr);
      if (ierr) return;
    } else if (i==N-1) {
      PetscInt js[] = {i-1, i};
      MatSetValues_AIJ_device(d_mat,1,&i,2,js,values,INSERT_VALUES,&ierr);
      printf("\t\t\te[%d.%d]add row %d. (%d)\n",rank,my0,i,ierr);
      if (ierr) return;
    } else {
      PetscInt js[] = {i-1, i, i+1};
      MatSetValues_AIJ_device(d_mat,1,&i,3,js,values,INSERT_VALUES,&ierr);
      printf("\t\t[%d.%d]add row %d. (%d)\n",rank,my0,i,ierr);
      if (ierr) return;
    }
  }
}

int main(int argc,char **args)
{
  PetscErrorCode               ierr;
  Mat                          A;
  PetscInt                     N=3, n, nz=3, Istart, Iend;
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
  ierr = MatCreateAIJCUSPARSE(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,nz,NULL,0,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatCUSPARSEGetDeviceMatWrite(A,&d_mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  n = Iend - Istart;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  assemble<<<1,n>>>(Istart, Iend, N, rank, d_mat);
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
  fflush(stdout);
  PetscSleep(1);
  // assemble end on CPU for now
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  VecViewFromOptions(y,NULL,"-vec_view");

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: cuda

   testset:
      args: -n 11 -mat_view -vec_view -info :mat
      output_file: output/ex5cu_1.out
      test:
        suffix: 1
        nsize:  1

TEST*/
