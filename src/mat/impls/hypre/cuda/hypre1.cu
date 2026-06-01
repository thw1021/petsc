#include <petsc/private/petschypre.h>
#include <petscdevice_cuda.h>
#include <../src/mat/impls/hypre/mhypre_kernels.hpp>
#include <../src/mat/impls/hypre/mhypre.h>

PetscErrorCode MatZeroRows_CUDA(PetscInt n, const PetscInt rows[], const HYPRE_Int i[], const HYPRE_Int j[], HYPRE_Complex a[], HYPRE_Complex diag)
{
  const PetscInt blkDimX = 16, blkDimY = 32;
  PetscInt       gridDimX = (n + blkDimX - 1) / blkDimX;
  cudaStream_t   stream;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  ZeroRows<<<dim3(gridDimX, 1), dim3(blkDimX, blkDimY), 0, stream>>>(n, rows, i, j, a, diag);
  PetscCallCUDA(cudaGetLastError());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscHypreIntCastArray_CUDA(PetscInt n, const PetscInt *a, HYPRE_Int *b)
{
  PetscFunctionBegin;
  if (n) {
    CastArray<<<(n + 255) / 256, 256>>>(n, a, b);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscMalloc_CUDA(size_t size, void **ptr)
{
  PetscFunctionBegin;
  PetscCallCUDA(cudaMalloc(ptr, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFree_CUDA(void *a)
{
  PetscFunctionBegin;
  PetscCallCUDA(cudaFree(a));
  PetscFunctionReturn(PETSC_SUCCESS);
}
