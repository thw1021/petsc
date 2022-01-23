#include "../veccupm.hpp"

using namespace Petsc::Vec::CUPM::Impl;

static constexpr auto VecSeqHIP = VecSeq_CUPM<Petsc::Device::CUPM::DeviceType::HIP>();

PetscErrorCode VecCreate_SeqHIP(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeqHIP.create_async(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
