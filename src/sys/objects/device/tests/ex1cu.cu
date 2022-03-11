static char help[] = "Benchmarking CUDA kernel launch time\n";
/*
  Running example on Summit at OLCF:
  # run with total 1 resource set (RS) (-n1), 1 RS per node (-r1), 1 MPI rank (-a1), 7 cores (-c7) and 1 GPU (-g1) per RS
  $ jsrun -n1 -a1 -c7 -g1 -r1  ./ex1cu
  Average asynchronous CUDA kernel launch time = 4.78 microseconds
  Average synchronous  CUDA kernel launch time = 12.95 microseconds
*/
#include <petscsys.h>
#include <petscdevice.h>

__global__ void NullKernel(){}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n=100000;
  cudaError_t    cerr;
  cudaEvent_t    start, stop;
  float          ms = 0; /* in milliseconds with a resolution of around 0.5 microseconds */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  cerr = cudaEventCreate(&start);CHKERRCUDA(cerr);
  cerr = cudaEventCreate(&stop);CHKERRCUDA(cerr);

  /* Launch a sequence of kernels asynchronously. Previous launched kernels do not need to be completed before launching a new one */
  cerr = cudaEventRecord(start);CHKERRCUDA(cerr);
  for (i=0; i<n; i++) {NullKernel<<<1,1,0,NULL>>>();}
  cerr = cudaEventRecord(stop);CHKERRCUDA(cerr);
  cerr = cudaEventSynchronize(stop);CHKERRCUDA(cerr); /* Wait for the 'stop' event to complete */
  cerr = cudaEventElapsedTime(&ms,start,stop);CHKERRCUDA(cerr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average asynchronous CUDA kernel launch time = %.2f microseconds\n",ms/n*1000);CHKERRQ(ierr);

  /* Launch a sequence of kernels synchronously. Only launch a new kernel after the one before it has been completed */
  cerr = cudaEventRecord(start);CHKERRCUDA(cerr);
  for (i=0; i<n; i++) {
    NullKernel<<<1,1,0,NULL>>>();
    cerr = cudaStreamSynchronize(NULL);CHKERRCUDA(cerr);
  }
  cerr = cudaEventRecord(stop);CHKERRCUDA(cerr);
  cerr = cudaEventSynchronize(stop);CHKERRCUDA(cerr);
  cerr = cudaEventElapsedTime(&ms,start,stop);CHKERRCUDA(cerr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average synchronous  CUDA kernel launch time = %.2f microseconds\n",ms/n*1000);CHKERRQ(ierr);

  cerr = cudaEventDestroy(start);CHKERRCUDA(cerr);
  cerr = cudaEventDestroy(stop);CHKERRCUDA(cerr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  build:
    requires: cuda

  test:
    requires: cuda
    args: -n 2
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

TEST*/
