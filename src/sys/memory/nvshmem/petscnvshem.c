#include <petsc/private/petscnvshmemimpl.h>

#if defined(HAVE_NVSHMEM)
PETSC_INTERN PetscErrorCode PetscNvshmemInitializeCheck(void)
{
  PetscFunctionBegin;
  if (!PetscNvshmemInitialized) { /* Note NVSHMEM does not provide a routine to check whether it is initialized */
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &PETSC_COMM_WORLD;
    PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
    PetscCall(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
    PetscNvshmemInitialized = PETSC_TRUE;
    PetscBeganNvshmem       = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscNvshmemMalloc(size_t size, void **ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscNvshmemInitializeCheck());
  *ptr = nvshmem_malloc(size);
  PetscCheck(*ptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "nvshmem_malloc() failed to allocate %zu bytes", size);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscNvshmemCalloc(size_t size, void **ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscNvshmemInitializeCheck());
  *ptr = nvshmem_calloc(size, 1);
  PetscCheck(*ptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "nvshmem_calloc() failed to allocate %zu bytes", size);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscNvshmemFree_Private(void *ptr)
{
  PetscFunctionBegin;
  nvshmem_free(ptr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscNvshmemFinalize(void)
{
  PetscFunctionBegin;
  nvshmem_finalize();
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
