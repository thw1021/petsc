static char help[] = "Benchmarking HIP kernel launch time\n";
/*
  Running example on Crusher at OLCF:
  # run with 1 mpi rank (-n1), 32 CPUs (-c32), and map the process to CPU 0 and GPU 0
  $ srun -n1 -c32 --cpu-bind=map_cpu:0 --gpus-per-node=8 --gpu-bind=map_gpu:0 ./ex1hip
  Average asynchronous HIP kernel launch time = 1.36 microseconds
  Average synchronous  HIP kernel launch time = 6.69 microseconds
*/
#include <petscsys.h>
#include <petscdevice.h>

__global__ void NullKernel(){}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n=100000;
  hipError_t     cerr;
  hipEvent_t     start,stop;
  float          ms = 0; /* in milliseconds with a resolution of around 0.5 microseconds */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  cerr = hipEventCreate(&start);CHKERRHIP(cerr);
  cerr = hipEventCreate(&stop);CHKERRHIP(cerr);

  /* Launch a sequence of kernels asynchronously. Previous launched kernels do not need to be completed before launching a new one */
  cerr = hipEventRecord(start);CHKERRHIP(cerr);
  for (i=0; i<n; i++) {NullKernel<<<1,1>>>();}
  cerr = hipEventRecord(stop);CHKERRHIP(cerr);
  cerr = hipEventSynchronize(stop);CHKERRHIP(cerr); /* Wait for the 'stop' event to complete */
  cerr = hipEventElapsedTime(&ms,start,stop);CHKERRHIP(cerr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average asynchronous HIP kernel launch time = %.2f microseconds\n",ms/n*1000);CHKERRQ(ierr);

  /* Launch a sequence of kernels synchronously. Only launch a new kernel after the one before it has been completed */
  cerr = hipEventRecord(start);CHKERRHIP(cerr);
  for (i=0; i<n; i++) {
    NullKernel<<<1,1>>>();
    cerr = hipDeviceSynchronize();CHKERRHIP(cerr);
  }
  cerr = hipEventRecord(stop);CHKERRHIP(cerr);
  cerr = hipEventSynchronize(stop);CHKERRHIP(cerr);
  cerr = hipEventElapsedTime(&ms,start,stop);CHKERRHIP(cerr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average synchronous  HIP kernel launch time = %.2f microseconds\n",ms/n*1000);CHKERRQ(ierr);

  cerr = hipEventDestroy(start);CHKERRHIP(cerr);
  cerr = hipEventDestroy(stop);CHKERRHIP(cerr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  build:
    requires: hip

  test:
    requires: hip
    args: -n 2
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

TEST*/
