static const char help[] = "Test reuse of CUDA/HIP SF links after changing the current device context.\n";

#include <petscsf.h>
#include <petscdevice.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
#endif

int main(int argc, char **argv)
{
  PetscSF            sf;
  PetscSFNode        remote[2];
  PetscDeviceContext saved, dctx[2], current;
  PetscDevice        device;
  PetscDeviceType    type;
  PetscMemType       mtype;
  PetscInt           local[] = {0, 2};
  PetscInt          *root, *leaf, *input, *output;
  PetscMPIInt        rank, size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  mtype = type == PETSC_DEVICE_CUDA ? PETSC_MEMTYPE_CUDA : PETSC_MEMTYPE_HIP;
  for (PetscInt i = 0; i < 2; ++i) {
    PetscCall(PetscDeviceContextDuplicate(saved, &dctx[i]));
    PetscCall(PetscDeviceContextSetStreamType(dctx[i], PETSC_STREAM_NONBLOCKING));
    PetscCall(PetscDeviceContextSetUp(dctx[i]));
  }
  PetscCall(PetscDeviceContextSetCurrentContext(dctx[0]));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_DEVICE, 4, PETSC_DECIDE, &root));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_DEVICE, 4, PETSC_DECIDE, &leaf));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_HOST, 4, PETSC_DECIDE, &input));
  PetscCall(PetscDeviceMalloc(dctx[0], PETSC_MEMTYPE_HOST, 4, PETSC_DECIDE, &output));
  PetscCall(PetscDeviceContextSynchronize(dctx[0]));
  remote[0].rank = remote[1].rank = (rank + 1) % size;
  remote[0].index                 = 3;
  remote[1].index                 = 1;
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetGraph(sf, 4, 2, local, PETSC_COPY_VALUES, remote, PETSC_COPY_VALUES));
  PetscCall(PetscSFSetUp(sf));

  // Initialize the link on A, then reuse it on B and A. Complete each pass before switching contexts.
  for (PetscInt pass = 0; pass < 3; ++pass) {
    current = dctx[pass % 2];
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    for (PetscInt i = 0; i < 4; ++i) input[i] = 100 * pass + 4 * rank + i;
    if (pass) PetscCall(PetscDeviceContextDelay(current, 0.05));
    PetscCall(PetscDeviceMemcpy(current, root, input, 4 * sizeof(*root)));
    PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_INT, mtype, root, mtype, leaf, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, root, leaf, MPI_REPLACE));
    PetscCall(PetscDeviceMemcpy(current, output, leaf, 4 * sizeof(*leaf)));
    PetscCall(PetscDeviceContextSynchronize(current));
    for (PetscInt i = 0; i < 2; ++i)
      PetscCheck(output[local[i]] == 100 * pass + 4 * remote[i].rank + remote[i].index, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pass %" PetscInt_FMT ", leaf %" PetscInt_FMT ": got %" PetscInt_FMT ", expected %" PetscInt_FMT, pass, local[i], output[local[i]],
                 100 * pass + 4 * remote[i].rank + remote[i].index);
  }

  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscDeviceFree(dctx[0], root));
  PetscCall(PetscDeviceFree(dctx[0], leaf));
  PetscCall(PetscDeviceFree(dctx[0], input));
  PetscCall(PetscDeviceFree(dctx[0], output));
  PetscCall(PetscDeviceContextSynchronize(dctx[0]));
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  for (PetscInt i = 0; i < 2; ++i) PetscCall(PetscDeviceContextDestroy(&dctx[i]));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: {{1 2}}
    output_file: output/empty.out
    args: -sf_type {{basic neighbor}} -use_gpu_aware_mpi 0

    test:
      suffix: cuda
      requires: cuda
      args: -sf_backend cuda

    test:
      suffix: hip
      requires: hip
      args: -sf_backend hip

  test:
    suffix: cuda_nvshmem
    requires: cuda nvshmem
    nsize: 2
    args: -sf_backend cuda -use_nvshmem -use_nvshmem_get {{0 1}}
    output_file: output/empty.out

TEST*/
