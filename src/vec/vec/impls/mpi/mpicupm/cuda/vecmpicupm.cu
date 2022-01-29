#include "../vecmpicupm.hpp" /*I <petscvec.h> I*/

using namespace Petsc::Vector::CUPM::Impl;

using VecMPI_CUDA = VecMPI_CUPM<Petsc::Device::CUPM::DeviceType::CUDA>;

/*MC
  VECCUDA - VECCUDA = "cuda" - A VECSEQCUDA on a single-process communicator, and VECMPICUDA
  otherwise.

  Options Database Keys:
. -vec_type cuda - sets the vector type to VECCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECSEQCUDA,
VECMPICUDA, VECSTANDARD, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

/*MC
  VECMPICUDA - VECMPICUDA = "mpicuda" - The basic parallel vector, modified to use CUDA

  Options Database Keys:
. -vec_type mpicuda - sets the vector type to VECMPICUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI,
VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecCreate_CUDA(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecMPI_CUDA::Create_CUPM_(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPICUDA(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecMPI_CUDA::create_async(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  VecCreateMPICUDA - Creates a standard, parallel array-style vector for CUDA devices.

  Collective

  Input Parameters:
+ comm - the MPI communicator to use
. n    - local vector length (or PETSC_DECIDE to have calculated if N is given)
- N    - global vector length (or PETSC_DETERMINE to have calculated if n is given)

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  Level: intermediate

.seealso: VecCreateMPICUDAWithArray(), VecCreateMPICUDAWithArrays(), VecCreateSeqCUDA(),
VecCreateSeq(), VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(),
VecCreateGhost(), VecCreateMPIWithArray(), VecCreateGhostWithArray(), VecMPISetGhost()
@*/
PetscErrorCode VecCreateMPICUDA(MPI_Comm comm, PetscInt n, PetscInt N, Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(v,4);
  ierr = VecMPI_CUDA::creatempicupm_async(comm,0,n,N,v,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCreateMPICUDAWithArrays - Creates a parallel, array-style vector, where the user provides
  the CPU and GPU array space to store the vector values.

  Collective

  Input Parameters:
+ comm  - the MPI communicator to use
. bs    - block size, same meaning as VecSetBlockSize()
. n     - local vector length, cannot be PETSC_DECIDE
. N     - global vector length (or PETSC_DECIDE to have calculated)
. cpuarray - the user provided CPU array to store the vector values
- gpuarray - the user provided GPU array to store the vector values

  Output Parameter:
. v - the vector

  Notes:
  If both cpuarray and gpuarray are provided, the caller must ensure that the provided arrays
  have identical values.

  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  If the user-provided arrays are NULL, then VecCUDAPlaceArray() can be used at a later stage to
  SET the array for storing the vector values.

  PETSc does NOT free the array when the vector is destroyed via VecDestroy(). The user should
  not free the array until the vector is destroyed.

  Level: intermediate

.seealso: VecCreateMPICUDA(), VecCreateSeqCUDAWithArray(), VecCreateMPIWithArray(),
VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()
@*/
PetscErrorCode VecCreateMPICUDAWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n) PetscValidScalarPointer(cpuarray,5);
  PetscValidPointer(v,7);
  ierr = VecMPI_CUDA::creatempicupmwitharrays_async(comm,bs,n,N,cpuarray,gpuarray,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCreateMPICUDAWithArray - Creates a parallel, array-style vector, where the user provides
  the GPU array space to store the vector values.

  Collective

  Input Parameters:
+ comm  - the MPI communicator to use
. bs    - block size, same meaning as VecSetBlockSize()
. n     - local vector length, cannot be PETSC_DECIDE
. N     - global vector length (or PETSC_DECIDE to have calculated)
- gpuarray - the user provided GPU array to store the vector values

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  If the user-provided array is NULL, then VecCUDAPlaceArray() can be used at a later stage to
  SET the array for storing the vector values.

  PETSc does NOT free the array when the vector is destroyed via VecDestroy(). The user should
  not free the array until the vector is destroyed.

  Level: intermediate

.seealso: VecCreateMPICUDA(), VecCreateSeqCUDAWithArray(), VecCreateMPIWithArray(),
VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()
@*/
PetscErrorCode VecCreateMPICUDAWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar gpuarray[], Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPICUDAWithArrays(comm,bs,n,N,nullptr,gpuarray,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
