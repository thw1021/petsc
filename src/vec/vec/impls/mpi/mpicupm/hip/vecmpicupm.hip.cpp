#include "../vecmpicupm.hpp"

using namespace Petsc::Vector::CUPM::Impl;

using VecMPI_HIP = VecMPI_CUPM<Petsc::Device::CUPM::DeviceType::HIP>;

/*MC
  VECHIP - VECHIP = "hip" - A VECSEQHIP on a single-process communicator, and VECMPIHIP
  otherwise.

  Options Database Keys:
. -vec_type hip - sets the vector type to VECHIP during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECSEQHIP,
VECMPIHIP, VECSTANDARD, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

/*MC
  VECMPIHIP - VECMPIHIP = "mpihip" - The basic parallel vector, modified to use HIP

  Options Database Keys:
. -vec_type mpihip - sets the vector type to VECMPIHIP during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI,
VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecCreate_HIP(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecMPI_HIP::Create_CUPM_(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPIHIP(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecMPI_HIP::create_async(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  VecCreateMPIHIP - Creates a standard, parallel array-style vector for HIP devices.

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

.seealso: VecCreateMPIHIPWithArray(), VecCreateMPIHIPWithArrays(), VecCreateSeqHIP(),
VecCreateSeq(), VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(),
VecCreateGhost(), VecCreateMPIWithArray(), VecCreateGhostWithArray(), VecMPISetGhost()
@*/
PetscErrorCode VecCreateMPIHIP(MPI_Comm comm, PetscInt n, PetscInt N, Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(v,4);
  ierr = VecMPI_HIP::creatempicupm_async(comm,0,n,N,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCreateMPIHIPWithArrays - Creates a parallel, array-style vector, where the user provides
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

  If the user-provided arrays are NULL, then VecHIPPlaceArray() can be used at a later stage to
  SET the array for storing the vector values.

  PETSc does NOT free the array when the vector is destroyed via VecDestroy(). The user should
  not free the array until the vector is destroyed.

  Level: intermediate

.seealso: VecCreateMPIHIP(), VecCreateSeqHIPWithArray(), VecCreateMPIWithArray(),
VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()
@*/
PetscErrorCode VecCreateMPIHIPWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n) PetscValidScalarPointer(cpuarray,5);
  PetscValidPointer(v,7);
  ierr = VecMPI_HIP::creatempicupmwitharrays_async(comm,bs,n,N,cpuarray,gpuarray,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecCreateMPIHIPWithArray - Creates a parallel, array-style vector, where the user provides
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

  If the user-provided array is NULL, then VecHIPPlaceArray() can be used at a later stage to
  SET the array for storing the vector values.

  PETSc does NOT free the array when the vector is destroyed via VecDestroy(). The user should
  not free the array until the vector is destroyed.

  Level: intermediate

.seealso: VecCreateMPIHIP(), VecCreateSeqHIPWithArray(), VecCreateMPIWithArray(),
VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()
@*/
PetscErrorCode VecCreateMPIHIPWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar gpuarray[], Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPIHIPWithArrays(comm,bs,n,N,nullptr,gpuarray,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
