#include "../vecseqcupm.hpp"

using namespace Petsc::Vector::CUPM::Impl;

template struct VecSeq_CUPM<Petsc::Device::CUPM::DeviceType::HIP>;

static constexpr auto VecSeq_HIP = VecSeq_CUPM<Petsc::Device::CUPM::DeviceType::HIP>{};

PetscErrorCode VecCreate_SeqHIP(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_HIP.create_async(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
