#include  "../veccupm.hpp"

using namespace Petsc;

PetscErrorCode VecCreateSeqCUDA(MPI_Comm comm, PetscInt n, Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,detail::CUPMVecSubTypeName<CUPMDeviceType::CUDA>());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
