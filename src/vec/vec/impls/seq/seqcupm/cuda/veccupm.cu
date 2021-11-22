#include  "../veccupm.hpp" /*I "petscvec.h" I*/

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

  This function may initialize PetscDevice, which may incur a device synchronization.

  Level: intermediate

.seealso: PetscDeviceInitialize(), VecCreate(), VecCreateSeq(), VecCreateSeqCUDAWithArray(),
VecCreateMPI(), VecCreateMPICUDA(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
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
+ comm     - the communicator, must be PETSC_COMM_SELF
. bs       - the block size
. n        - the vector length
- gpuarray - GPU memory where the vector elements are to be stored.

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  If the user-provided array is NULL, then VecCUDAPlaceArray() can be used at a later stage to
  SET the array for storing the vector values.

  PETSc does NOT free the array when the vector is destroyed via VecDestroy(). The user should
  not free the array until the vector is destroyed.

  This function may initialize PetscDevice, which may incur a device synchronization.

  Level: intermediate

.seealso: PetscDeviceInitialize(), VecCreate(), VecCreateSeq(), VecCreateSeqWithArray(),
VecCreateSeqCUDA(), VecCreateMPICUDAWithArray(), VecDuplicate(), VecDuplicateVecs(),
VecCreateGhost(), VecCUDAPlaceArray(), VecCreateMPIWithArray()
@*/
PetscErrorCode VecCreateSeqCUDAWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar gpuarray[], Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeqCUDAWithArrays(comm,bs,n,nullptr,gpuarray,v);CHKERRQ(ierr);
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

  This function may initialize PetscDevice, which may incur a device synchronization.

  Level: intermediate

.seealso: PetscDeviceInitialize(), VecCreate(), VecCreateSeqWithArray(), VecCreateSeqCUDA(),
VecCreateSeqCUDAWithArray(), VecCreateMPICUDA(), VecCreateMPICUDAWithArray(),
VecCreateMPICUDAWithArrays(), VecCUDAPlaceArray()
@*/
PetscErrorCode VecCreateSeqCUDAWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v)
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
  using Impl::detail::MemoryAccess;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  ierr = VecSeqCUDA.getarray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ_WRITE>(v,a);CHKERRQ(ierr);
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
  using Impl::detail::MemoryAccess;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  ierr = VecSeqCUDA.restorearray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ_WRITE>(v,a);CHKERRQ(ierr);
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
PetscErrorCode VecCUDAGetArrayRead(Vec v, const PetscScalar **a)
{
  using Impl::detail::MemoryAccess;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeqCUDA.getarray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ>(v,const_cast<PetscScalar**>(a));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCUDARestoreArrayRead - Restore a CUDA device pointer previously acquired with VecCUDAGetArrayRead().

  Input Parameters:
+ v - the vector
- a - the CUDA device pointer

  Notes:
  The pointer is invalid after this function returns.

  If the data on the host side was previously up to date it will remain so, i.e. data on both
  the device and the host is up to date. Accessing data on the host side e.g. with
  VecGetArray() does not incur a device to host data transfer.

  Fortran note:
  This function is not currently available from Fortran.

  Level: intermediate

.seealso: VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecCUDAGetArray(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDARestoreArrayRead(Vec v, const PetscScalar **a)
{
  using Impl::detail::MemoryAccess;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  ierr = VecSeqCUDA.restorearray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ>(v,const_cast<PetscScalar**>(a));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAGetArrayWrite - Provides write access to the CUDA buffer inside a vector.

  Input Parameter:
. v - the vector

  Output Parameter:
. a - the CUDA pointer

  Notes:
  The data pointed to by the device pointer is uninitialized. The user may not read from this
  data.  Furthermore, the entire array needs to be filled by the user to obtain well-defined
  behaviour. The device memory will be allocated by this function if it hasn't been allocated
  previously. This is analogous to intent(out) in Fortran.

  The device pointer needs to be released with VecCUDARestoreArrayWrite(). When the pointer is
  released the host data of the vector is marked as out of data. Subsequent access of the host
  data with e.g. VecGetArray() incurs a device to host data transfer.

  Fortran note:
  This function is not currently available from Fortran.

  Level: advanced

.seealso: VecCUDARestoreArrayWrite(), VecCUDAGetArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDAGetArrayWrite(Vec v, PetscScalar **a)
{
  using Impl::detail::MemoryAccess;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  ierr = VecSeqCUDA.getarray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::WRITE>(v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
 VecCUDARestoreArrayWrite - Restore a CUDA device pointer previously acquired with
 VecCUDAGetArrayWrite().

  Input Parameters:
+ v - the vector
- a - the CUDA device pointer.  This pointer is invalid after VecCUDARestoreArrayWrite() returns.

  Notes:
  Data on the host will be marked as out of date. Subsequent access of the data on the host
  side e.g. with VecGetArray() will incur a device to host data transfer.

  Fortran note:
  This function is not currently available from Fortran.

  Level: intermediate

.seealso: VecCUDAGetArrayWrite(), VecCUDAGetArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDARestoreArrayWrite(Vec v, PetscScalar **a)
{
  using Impl::detail::MemoryAccess;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  ierr = VecSeqCUDA.restorearray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::WRITE>(v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAPlaceArray - Allows one to replace the GPU array in a vector with a
   GPU array provided by the user. This is useful to avoid copying an
   array into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the GPU array

   Notes:
   You can return to the original GPU array with a call to VecCUDAResetArray()
   It is not possible to use VecCUDAPlaceArray() and VecPlaceArray() at the
   same time on the same vector.

   Level: developer

.seealso: VecPlaceArray(), VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecResetArray(), VecCUDAResetArray(), VecCUDAReplaceArray()

@*/
PetscErrorCode VecCUDAPlaceArray(Vec vin, const PetscScalar a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  ierr = VecSeqCUDA.placearray_async<PETSC_MEMTYPE_DEVICE>(vin,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAReplaceArray - Allows one to replace the GPU array in a vector
   with a GPU array provided by the user. This is useful to avoid copying
   a GPU array into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the GPU array

   Notes:
   This permanently replaces the GPU array and frees the memory associated
   with the old GPU array.

   The memory passed in CANNOT be freed by the user. It will be freed
   when the vector is destroyed.

   Not supported from Fortran

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecPlaceArray(), VecResetArray(), VecCUDAResetArray(), VecCUDAPlaceArray(), VecReplaceArray()

@*/
PetscErrorCode VecCUDAReplaceArray(Vec vin, const PetscScalar a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  ierr = VecSeqCUDA.replacearray_async<PETSC_MEMTYPE_DEVICE>(vin,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAResetArray - Resets a vector to use its default memory. Call this
   after the use of VecCUDAPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray(), VecResetArray(), VecCUDAPlaceArray(), VecCUDAReplaceArray()
@*/
PetscErrorCode VecCUDAResetArray(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  ierr = VecSeqCUDA.resetarray_async<PETSC_MEMTYPE_DEVICE>(vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
