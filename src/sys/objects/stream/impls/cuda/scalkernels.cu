#include "streamcuda.h"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

PetscErrorCode PetscStreamScalarAXTY_CUDA_Kernel(PetscScalar a, PetscStreamScalar pscalx, PetscStreamScalar pscaly, PetscStream pstream)
{
  PetscErrorCode                  ierr;
  cudaStream_t                    cstream;
  PetscScalar                     *dx;
  thrust::device_ptr<PetscScalar> dptrx;

  PetscFunctionBegin;
  ierr = PetscStreamScalarGetDeviceWrite(pscalx,&dx,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  if (pscalx == pscaly) {
    try {
      using namespace thrust::placeholders;
      dptrx = thrust::device_pointer_cast(dx);
      thrust::transform(thrust::cuda::par.on(cstream),dptrx,dptrx+1,dptrx,a*_1*_1);
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  } else if (!pscaly) {
    try {
      using namespace thrust::placeholders;
      dptrx = thrust::device_pointer_cast(dx);
      thrust::transform(thrust::cuda::par.on(cstream),dptrx,dptrx+1,dptrx,a*_1);
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  } else {
    const PetscScalar                     *dy;
    thrust::device_ptr<const PetscScalar> dptry;
    ierr = PetscStreamScalarGetDeviceRead(pscaly,&dy,pstream);CHKERRQ(ierr);
    try {
      using namespace thrust::placeholders;
      dptrx = thrust::device_pointer_cast(dx);
      dptry = thrust::device_pointer_cast(dy);
      thrust::transform(thrust::cuda::par.on(cstream),dptrx,dptrx+1,dptry,dptrx,a*_1*_2);
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  }
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarRestoreDeviceWrite(pscalx,&dx,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarAYDX_CUDA_Kernel(PetscScalar a, PetscStreamScalar pscalx, PetscStreamScalar pscaly, PetscStream pstream)
{
  PetscErrorCode                  ierr;
  cudaStream_t                    cstream;
  PetscScalar                     *dx;
  thrust::device_ptr<PetscScalar> dptrx;

  PetscFunctionBegin;
  ierr = PetscStreamScalarGetDeviceWrite(pscalx,&dx,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  dptrx = thrust::device_pointer_cast(dx);
  if (!pscaly) {
    try {
      using namespace thrust::placeholders;
      thrust::transform(thrust::cuda::par.on(cstream),dptrx,dptrx+1,dptrx,a/_1);
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  } else {
    const PetscScalar                     *dy;
    thrust::device_ptr<const PetscScalar> dptry;
    ierr = PetscStreamScalarGetDeviceRead(pscaly,&dy,pstream);CHKERRQ(ierr);
    dptry = thrust::device_pointer_cast(dy);
    try {
      using namespace thrust::placeholders;
      thrust::transform(thrust::cuda::par.on(cstream),dptrx,dptrx+1,dptry,dptrx,a*_2/_1);
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  }
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarRestoreDeviceWrite(pscalx,&dx,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
