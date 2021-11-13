#include  "../veccupm.hpp"

using namespace Petsc;

static constexpr auto VecSeqCUDA = Impl::VecSeq_CUPM<CUPMDeviceType::CUDA>();

PetscErrorCode VecCreate_SeqCUDA(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeqCUDA.create_async(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  VecCreateSeqCUDA - Creates a standard, sequential array-style vector.

  Collective, Possibly Synchronous

  Input Parameter:
+ comm - the communicator, must be PETSC_COMM_SELF
- n    - the vector length

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  This function may initialize CUDA, which may incur a device synchronization.

  Level: intermediate

.seealso: VecCreate(), VecCreateSeq(), VecCreateSeqCUDAWithArray(), VecCreateMPI(), VecCreateMPICUDA(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
@*/
PetscErrorCode VecCreateSeqCUDA(MPI_Comm comm, PetscInt n, Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(v,3);
  ierr = VecSeqCUDA.createseqcupm_async(comm,n,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCreateSeqCUDAWithArray - Creates a CUDA sequential array-style vector,
  where the user provides the array space to store the vector values. The array
  provided must be a GPU array.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm  - the communicator, must be PETSC_COMM_SELF
. bs    - the block size
. n     - the vector length
- array - GPU memory where the vector elements are to be stored.

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  If the user-provided array is NULL, then VecCUDAPlaceArray() can be used at a later stage to
  SET the array for storing the vector values.

  PETSc does NOT free the array when the vector is destroyed via VecDestroy(). The user should
  not free the array until the vector is destroyed.

  This function may initialize CUDA, which may incur a device synchronization.

  Level: intermediate

.seealso: VecCreate(), VecCreateSeq(), VecCreateSeqWithArray(), VecCreateSeqCUDA(),
VecCreateMPICUDAWithArray(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
VecCUDAPlaceArray(), VecCreateMPIWithArray()
@*/
PetscErrorCode  VecCreateSeqCUDAWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar array[], Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n) PetscValidScalarPointer(array,4);
  PetscValidPointer(v,5);
  ierr = VecSeqCUDA.createwitharray_async(comm,bs,n,array,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCreateSeqCUDAWithArrays - Creates a CUDA sequential array-style vector, where the user
  provides the array space to store the vector values.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm     - the communicator, must be PETSC_COMM_SELF
. bs       - the block size
. n        - the local vector length
. cpuarray - CPU memory where the vector elements are to be stored.
- gpuarray - GPU memory where the vector elements are to be stored.

  Output Parameter:
. v - the vector

  Notes:
  If both cpuarray and gpuarray are provided, the provided arrays must have identical
  values. This is checked on debug builds but blindly assumed when debugging is disabled.

  PETSc does NOT free the provided arrays when the vector is destroyed via VecDestroy(). The
  user is instead responsible for freeing them, although they should not do so before calling
  VecDestroy() on the vector.

  This function may initialize CUDA, which may incur a device synchronization.

  Level: intermediate

.seealso: VecCreate(), VecCreateSeqWithArray(), VecCreateSeqCUDA(),
VecCreateSeqCUDAWithArray(), VecCreateMPICUDA(), VecCreateMPICUDAWithArray(),
VecCreateMPICUDAWithArrays(), VecCUDAPlaceArray()
@*/
PetscErrorCode  VecCreateSeqCUDAWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n) PetscValidScalarPointer(cpuarray,4);
  PetscValidPointer(v,6);
  ierr = VecSeqCUDA.createwithbotharrays_async(comm,bs,n,cpuarray,gpuarray,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAGetArray - Provides access to the CUDA buffer inside a vector.

  Asynchronous

  Input Parameter:
. v - the vector

  Output Parameter:
. a - the CUDA device pointer

  Notes:
  This function has semantics similar to VecGetArray(); the pointer returned by this function
  points to a consistent view of the vector data. This may involve a copy operation of data
  from the host to the device if the data on the device is out of date.

  It is assumed that the user will modify the returned array; and any data on the host is
  immediately marked as out of date. This is similar to intent(inout) in fortran.

  Fortran note:
  This function is not currently available from Fortran.

  Developer Notes:
  If the device memory hasn't been allocated previously it will be allocated as part of this
  function call.

  Level: intermediate

.seealso: VecCUDARestoreArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDAGetArray(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  ierr = VecSeqCUDA.getarray_async(v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCUDARestoreArray - Restore a CUDA device pointer previously acquired with VecCUDAGetArray().

  Asynchronous

  Input Parameters:
+ v - the vector
- a - the CUDA device pointer

  Notes:
  The restored pointer is invalid after this function returns. This function also marks the
  host data as out of date. Subsequent access to the vector data on the host side via
  VecGetArray() will incur a (synchronous) data transfer.

  Fortran note:
  This function is not currently available from Fortran.

  Level: intermediate

.seealso: VecCUDAGetArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDARestoreArray(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  ierr = VecSeqCUDA.restorearray_async(v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAGetArrayRead - Provides read access to the CUDA buffer inside a vector.

  Asynchronous

  Input Parameter:
. v - the vector

  Output Parameter:
. a - the CUDA pointer.

  Notes:
  See VecCUDAGetArray() for data movement semantics of this function.

  This function assumes that the user will not modify the vector data. This is analgogous to
  intent(in) in Fortran.

  The device pointer must be restored by calling VecCUDARestoreArrayRead(). If the data on the
  host side was previously up to date it will remain so, i.e. data on both the device and the
  host is up to date. Accessing data on the host side does not incur a device to host data
  transfer.

  Fortran note:
  This function is not currently available from Fortran.

  Level: intermediate

.seealso: VecCUDARestoreArrayRead(), VecCUDAGetArray(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDAGetArrayRead(Vec v,const PetscScalar** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAGetArray(v,const_cast<PetscScalar**>(a));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
