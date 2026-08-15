#include <petscvec.h>
#include <cuda_runtime.h>
int main(int argc, char **argv)
{
  PetscMPIInt      rank;
  Vec              v;
  int              dev = -1;
  struct cudaDeviceProp prop;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(VecSetSizes(v, 1024, PETSC_DECIDE));
  PetscCall(VecSetType(v, VECCUDA));
  PetscCall(VecSet(v, 1.0));
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank %d -> cuda dev %d  bus 0x%02x\n", rank, dev, prop.pciBusID));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscFinalize());
  return 0;
}
