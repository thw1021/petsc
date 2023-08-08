#include <HYPRE_utilities.h>
#include <petscdevice_cuda.h>
#include <../src/mat/impls/hypre/mhypre_kernels.hpp>

PetscErrorCode MatZeroRows_CUDA(PetscInt n, const PetscInt rows[], const HYPRE_Int i[], const HYPRE_Int j[], HYPRE_Complex a[], HYPRE_Complex diag)
{
  const PetscInt     blkDimX = 16, blkDimY = 32;
  PetscInt           gridDimX = (n + blkDimX - 1) / blkDimX;
  PetscDeviceContext dctx;
  PetscDeviceType    type;
  void              *handle;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &handle));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &type));
  PetscCheck(type == PETSC_DEVICE_CUDA, PETSC_COMM_SELF, PETSC_ERR_GPU, "expect a CUDA device");
  ZeroRows<<<dim3(gridDimX, 1), dim3(blkDimX, blkDimY), 0, *(cudaStream_t *)handle>>>(n, rows, i, j, a, diag);
  PetscCallCUDA(cudaGetLastError());
  PetscFunctionReturn(PETSC_SUCCESS);
}
