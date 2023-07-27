
#include <petsc/private/vecimpl.h> /*I  "petscvec.h"   I*/

static PetscErrorCode VecCreate_Common_Private(Vec v)
{
  PetscFunctionBegin;
  v->array_gotten = PETSC_FALSE;
  v->petscnative  = PETSC_FALSE;
  v->offloadmask  = PETSC_OFFLOAD_UNALLOCATED;
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  v->minimum_bytes_pinned_memory = 0;
  v->pinned_memory               = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_DEVICE)
  v->boundtocpu = PETSC_TRUE;
#endif
  PetscCall(PetscStrallocpy(PETSCRANDER48, &v->defaultrandtype));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecCreate - Creates an empty vector object. The type can then be set with `VecSetType()`,
  or `VecSetFromOptions().`

  Collective

  Input Parameter:
. comm - The communicator for the vector object

  Output Parameter:
. vec - The vector object

  Level: beginner

  Notes:
  If you never  call `VecSetType()` or `VecSetFromOptions()` it will generate an
  error when you try to use the vector.

.seealso: [](ch_vectors), `Vec`, `VecSetType()`, `VecSetSizes()`, `VecCreateMPIWithArray()`, `VecCreateMPI()`, `VecDuplicate()`,
          `VecDuplicateVecs()`, `VecCreateGhost()`, `VecCreateSeq()`, `VecPlaceArray()`
@*/
PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec)
{
  Vec v;

  PetscFunctionBegin;
  PetscAssertPointer(vec, 2);
  *vec = NULL;
  PetscCall(VecInitializePackage());
  PetscCall(PetscHeaderCreate(v, VEC_CLASSID, "Vec", "Vector", "Vec", comm, VecDestroy, VecView));
  PetscCall(PetscLayoutCreate(comm, &v->map));
  PetscCall(VecCreate_Common_Private(v));
  *vec = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create a vector with the given layout.  The reference count of the input layout will be increased by 1 */
PetscErrorCode VecCreateWithLayout_Private(PetscLayout map, Vec *vec)
{
  Vec v;

  PetscFunctionBegin;
  PetscAssertPointer(vec, 2);
  *vec = NULL;
  PetscCall(VecInitializePackage());
  PetscCall(PetscHeaderCreate(v, VEC_CLASSID, "Vec", "Vector", "Vec", map->comm, VecDestroy, VecView));
  v->map = map;
  map->refcnt++;
  PetscCall(VecCreate_Common_Private(v));
  v->bstash.bs = map->bs;
  *vec         = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  VecCreateWithArraysAndMemType - Creates a petsc device vector with a user provided host array and/or device array (of the given memory type)

  Collective

  Input Parameters:
+ comm - The communicator for the vector object.  If its size is one, then the vector will be a sequential vector; otherwise, it will be an MPI parallel vector
. bs - the block size of the vector
. n  - the vector's local size, cannot be `PETSC_DECIDE`
. N  - the vector's global size (or `PETSC_DECIDE` to have it calculated)
. harray - the host array; could be NULL, then petsc will allocate it
. mtype - memory type of darray; if mtype is PETSC_MEMTYPE_HOST, harray and darray must be the same if both are not NULL.
- darray - the device array; could be NULL, then petsc will allocate it

  Output Parameter:
. vec - The vector object

  Level: beginner

  Notes:
   If both harray and darray are provided, we assume they contain the same data, i.e., host and device are synchronized.
   Users need to free the given arrays after destroying the vector.

  .seealso: `VecCreateMPIWithArray()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`, `VecCreateGhost()`, `VecCreateSeq()`, `VecPlaceArray()`
*/
PetscErrorCode VecCreateWithArraysAndMemType(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar harray[], PetscMemType mtype, const PetscScalar darray[], Vec *vec)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidPointer(vec, 8);
  PetscCall(MPI_Comm_size(comm, &size));
  PetscCheck(n != PETSC_DECIDE, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must set local size of vector");

  if (size == 1) {
    PetscCheck(N == PETSC_DECIDE || n == N, comm, PETSC_ERR_ARG_INCOMP, "global size and local size do not match");
    if (PetscMemTypeCUDA(mtype)) {
#if defined(PETSC_HAVE_KOKKOS) && defined(PETSC_HAVE_CUDA)
      PetscCall(VecCreateSeqKokkosWithArrays_Private(comm, bs, n, harray, darray, vec));
#elif defined(PETSC_HAVE_CUDA)
      PetscCall(VecCreateSeqCUDAWithArrays(comm, bs, n, harray, darray, vec));
#else
      SETERRQ(comm, PETSC_ERR_SUP, "Requesting a CUDA vector without CUDA support");
#endif
    } else if (PetscMemTypeHIP(mtype)) {
#if defined(PETSC_HAVE_KOKKOS) && defined(PETSC_HAVE_HIP)
      PetscCall(VecCreateSeqKokkosWithArrays_Private(comm, bs, n, harray, darray, vec));
#elif defined(PETSC_HAVE_HIP)
      PetscCall(VecCreateSeqHIPWithArrays(comm, bs, n, harray, darray, vec));
#else
      SETERRQ(comm, PETSC_ERR_SUP, "Requesting a HIP vector without HIP support");
#endif
    } else if (PetscMemTypeSYCL(mtype)) {
#if defined(PETSC_HAVE_KOKKOS) && defined(PETSC_HAVE_SYCL)
      PetscCall(VecCreateSeqKokkosWithArrays_Private(comm, bs, n, harray, darray, vec));
#else
      SETERRQ(comm, PETSC_ERR_SUP, "Requesting a SYCL vector without SYCL support");
#endif
    } else if (PetscMemTypeHost(mtype)) {
      if (n) {
        PetscCheck(harray || darray, comm, PETSC_ERR_ARG_INCOMP, "harray and darray could not be both NULL");
        if (harray && darray) PetscCheck(harray == darray, comm, PETSC_ERR_ARG_INCOMP, "harray and darray must be the same to create a host vector");
      }
      PetscCall(VecCreateSeqWithArray(comm, bs, n, harray ? harray : darray, vec));
    } else SETERRQ(comm, PETSC_ERR_SUP, "Unsupported PetscMemType");
  } else {
    if (PetscMemTypeCUDA(mtype)) {
#if defined(PETSC_HAVE_KOKKOS) && defined(PETSC_HAVE_CUDA)
      PetscCall(VecCreateMPIKokkosWithArrays_Private(comm, bs, n, N, harray, darray, vec));
#elif defined(PETSC_HAVE_CUDA)
      PetscCall(VecCreateMPICUDAWithArrays(comm, bs, n, N, harray, darray, vec));
#else
      SETERRQ(comm, PETSC_ERR_SUP, "Requesting a CUDA vector without CUDA support");
#endif
    } else if (PetscMemTypeHIP(mtype)) {
#if defined(PETSC_HAVE_KOKKOS) && defined(PETSC_HAVE_HIP)
      PetscCall(VecCreateMPIKokkosWithArrays_Private(comm, bs, n, N, harray, darray, vec));
#elif defined(PETSC_HAVE_HIP)
      PetscCall(VecCreateMPIHIPWithArrays(comm, bs, n, N, harray, darray, vec));
#else
      SETERRQ(comm, PETSC_ERR_SUP, "Requesting a HIP vector without HIP support");
#endif
    } else if (PetscMemTypeSYCL(mtype)) {
#if defined(PETSC_HAVE_KOKKOS) && defined(PETSC_HAVE_SYCL)
      PetscCall(VecCreateMPIKokkosWithArrays_Private(comm, bs, n, N, harray, darray, vec));
#else
      SETERRQ(comm, PETSC_ERR_SUP, "Requesting a SYCL vector without SYCL support");
#endif
    } else if (PetscMemTypeHost(mtype)) {
      if (n) {
        PetscCheck(harray || darray, comm, PETSC_ERR_ARG_INCOMP, "harray and darray could not be both NULL");
        if (harray && darray) PetscCheck(harray == darray, comm, PETSC_ERR_ARG_INCOMP, "harray and darray must be the same to create a host vector");
      }
      PetscCall(VecCreateMPIWithArray(comm, bs, n, N, harray ? harray : darray, vec));
    } else SETERRQ(comm, PETSC_ERR_SUP, "Unsupported PetscMemType");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
