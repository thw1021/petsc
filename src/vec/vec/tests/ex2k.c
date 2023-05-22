static char help[] = "Benchmarking VecMDot()\n";
/*
  Usage:
   mpirun -n <np> ./ex1k -vec_type <vector type>
     -n  <n>  # number of data points of vector sizes from 128, 256, 512 and up. Maxima and default is 23.
     -m  <m>  # run each VecMDot() m times to get the average time, default is 1000.

  Example:

  Running on Crusher at OLCF:
  # run with 1 mpi rank (-n1), 32 CPUs (-c32), and map the process to CPU 0 and GPU 0
  $ srun -n1 -c32 --cpu-bind=map_cpu:0 --gpus-per-node=8 --gpu-bind=map_gpu:0 ./ex2k -vec_type kokkos
*/

#include <petscvec.h>
#include <petscdevice.h>

int main(int argc, char **argv)
{
  PetscInt           i, j, k, N, n, m = 128, nsamples, nvs;
  PetscLogDouble     tstart, tend, times[8];
  Vec                x, *ys;
  PetscScalar       *vals;
  PetscMPIInt        size;
  PetscInt           Ns[]  = {// Use explicit sizes so that one can add sizes very close to 2^31
                   128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912};
  PetscInt           NVs[] = {1, 3, 8, 30}; // try these numbers of ys in VecMDot
  PetscDeviceContext dctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  n = nsamples = sizeof(Ns) / sizeof(Ns[0]);
  nvs          = sizeof(NVs) / sizeof(NVs[0]);

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL)); // Up to vectors of local size 2^{n+6}
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL)); // Run each VecMDot() m times
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscMalloc1(NVs[nvs - 1], &vals));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Vector(N)   "));
  for (j = 0; j < nvs; j++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecMDot(nv=%" PetscInt_FMT ") ", NVs[j]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " (us)\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------------------------------\n"));

  nsamples = PetscMin(nsamples, n);
  for (k = 0; k < nsamples; k++) { // for each vector sizes in Ns[]
    N = Ns[k];
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSetSizes(x, N, PETSC_DECIDE));
    PetscCall(VecSetUp(x));
    PetscCall(VecDuplicateVecs(x, NVs[nvs - 1], &ys));
    PetscCall(VecSet(x, 2.5));
    for (i = 0; i < NVs[nvs - 1]; i++) PetscCall(VecSet(ys[i], 4.0));

    for (j = 0; j < nvs; j++) { // for each nv in NVs[]
      // Warm-up
      for (i = 0; i < 2; i++) PetscCall(VecMDot(x, NVs[j], ys, vals));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

      PetscCall(PetscTime(&tstart));
      for (i = 0; i < m; i++) PetscCall(VecMDot(x, NVs[j], ys, vals));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&tend));
      times[j] = (tend - tstart) * 1e6 / m;
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12" PetscInt_FMT, N));
    for (j = 0; j < nvs; j++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12.1f ", times[j]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroyVecs(NVs[nvs - 1], &ys));
  }

  PetscCall(PetscFree(vals));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    args: -n 2 -m 2
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

    test:
      suffix: standard

    test:
      requires: kokkos_kernels
      suffix: kok
      args: -vec_type kokkos

TEST*/
