#include <petscvec.h> /*I "petscvec.h" I*/
#include <petsc/private/petscimpl.h>

/*@
  VecCreateSeqWithArrayAndMemType - Creates a sequential array-style vector using a user-provided array with the specified memory type

  Collective

  Input Parameters:
+ comm  - the communicator, should be `PETSC_COMM_SELF`
. mtype - the memory type of `array`, which must be the same on all processes in `comm`
. bs    - the block size
. n     - the vector length
- array - memory where the vector elements are to be stored

  Output Parameter:
. V - the vector

  Level: intermediate

  Notes:
  `mtype` determines whether the resulting vector is a standard, CUDA, or HIP vector. PETSc does not free `array` when the vector is destroyed via `VecDestroy()`.

.seealso: `VecCreateSeqWithArray()`, `VecCreateMPIWithArrayAndMemType()`, `VecCreateSeqCUDAWithArray()`, `VecCreateSeqHIPWithArray()`, `PetscMemType`
@*/
PetscErrorCode VecCreateSeqWithArrayAndMemType(MPI_Comm comm, PetscMemType mtype, PetscInt bs, PetscInt n, const PetscScalar array[], Vec *V)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(comm, (PetscInt)mtype, 2);
  if (PetscMemTypeHost(mtype)) PetscCall(VecCreateSeqWithArray(comm, bs, n, array, V));
  else if (PetscMemTypeCUDA(mtype)) PetscCall(VecCreateSeqCUDAWithArray(comm, bs, n, array, V));
  else if (PetscMemTypeHIP(mtype)) PetscCall(VecCreateSeqHIPWithArray(comm, bs, n, array, V));
  else SETERRQ(comm, PETSC_ERR_SUP, "Not for PetscMemType %s", PetscMemTypeToString(mtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecCreateMPIWithArrayAndMemType - Creates a parallel array-style vector using a user-provided array with the specified memory type

  Collective

  Input Parameters:
+ comm  - the MPI communicator to use
. mtype - the memory type of `array`, which must be the same on all processes in `comm`
. bs    - block size, same meaning as `VecSetBlockSize()`
. n     - local vector length, cannot be `PETSC_DECIDE`
. N     - global vector length (or `PETSC_DETERMINE` to have calculated)
- array - memory where the vector elements are to be stored

  Output Parameter:
. V - the vector

  Level: intermediate

  Notes:
  `mtype` determines whether the resulting vector is a standard, CUDA, or HIP vector. PETSc does not free `array` when the vector is destroyed via `VecDestroy()`.

.seealso: `VecCreateMPIWithArray()`, `VecCreateSeqWithArrayAndMemType()`, `VecCreateMPICUDAWithArray()`, `VecCreateMPIHIPWithArray()`, `PetscMemType`
@*/
PetscErrorCode VecCreateMPIWithArrayAndMemType(MPI_Comm comm, PetscMemType mtype, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar array[], Vec *V)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(comm, (PetscInt)mtype, 2);
  if (PetscMemTypeHost(mtype)) PetscCall(VecCreateMPIWithArray(comm, bs, n, N, array, V));
  else if (PetscMemTypeCUDA(mtype)) PetscCall(VecCreateMPICUDAWithArray(comm, bs, n, N, array, V));
  else if (PetscMemTypeHIP(mtype)) PetscCall(VecCreateMPIHIPWithArray(comm, bs, n, N, array, V));
  else SETERRQ(comm, PETSC_ERR_SUP, "Not for PetscMemType %s", PetscMemTypeToString(mtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}
