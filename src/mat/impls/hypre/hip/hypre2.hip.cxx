#include <petsc/private/petschypre.h>
#include <petscdevice_hip.h>
#include <../src/mat/impls/hypre/mhypre_kernels.hpp>

PetscErrorCode MatZeroRows_HIP(PetscInt n, const PetscInt rows[], const HYPRE_Int i[], const HYPRE_Int j[], HYPRE_Complex a[], HYPRE_Complex diag)
{
  const PetscInt blkDimX = 16, blkDimY = 32;
  PetscInt       gridDimX = (n + blkDimX - 1) / blkDimX;
  hipStream_t    stream;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscGetCurrentHIPStream(&stream));
  hipLaunchKernelGGL(ZeroRows, dim3(gridDimX, 1), dim3(blkDimX, blkDimY), 0, stream, n, rows, i, j, a, diag);
  PetscCallHIP(hipGetLastError());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscHypreIntCastArray_HIP(size_t n, const PetscInt *a, HYPRE_Int *b)
{
  hipStream_t stream;

  PetscFunctionBegin;
  if (n) {
    PetscCall(PetscGetCurrentHIPStream(&stream));
    hipLaunchKernelGGL(CastArray, dim3((n + 255) / 256), dim3(256), 0, stream, n, a, b);
    PetscCallCUDA(hipGetLastError());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscMalloc_HIP(size_t size, void **ptr)
{
  PetscFunctionBegin;
  PetscCallCUDA(hipMalloc(ptr, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFree_HIP(void *a)
{
  PetscFunctionBegin;
  if (a) PetscCallCUDA(hipFree(a));
  PetscFunctionReturn(PETSC_SUCCESS);
}
