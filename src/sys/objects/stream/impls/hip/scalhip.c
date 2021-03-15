#include "streamhip.h"

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarDestroy_HIP(PetscStreamScalar pscal)
{
  hipError_t cerr;

  PetscFunctionBegin;
  cerr = hipHostFree(pscal->host);CHKERRHIP(cerr);
  cerr = hipFree(pscal->device);CHKERRHIP(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSetup_HIP(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet supported");
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSetValue_HIP(PetscStreamScalar pscal, const PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;
  hipError_t     herr;
  hipStream_t    hstream;

  PetscFunctionBegin;
  ierr = PetscStreamGetStream(pstream,&hstream);CHKERRQ(ierr);
  if (val) {
    if (PetscMemTypeHost(mtype)) {
      herr = hipMemcpyAsync(pscal->device,val,sizeof(PetscScalar),hipMemcpyHostToDevice,hstream);CHKERRHIP(herr);
      herr = hipMemcpyAsync(pscal->host,val,sizeof(PetscScalar),hipMemcpyHostToHost,hstream);CHKERRHIP(herr);
      pscal->omask = PETSC_OFFLOAD_BOTH;
    } else {
      herr = hipMemcpyAsync(pscal->device,val,sizeof(PetscScalar),hipMemcpyDeviceToDevice,hstream);CHKERRHIP(herr);
      pscal->omask = PETSC_OFFLOAD_GPU;
    }
  } else {
    herr = hipMemsetAsync(pscal->device,0,sizeof(PetscScalar),hstream);CHKERRHIP(herr);
    herr = hipMemsetAsync(pscal->host,0,sizeof(PetscScalar),hstream);CHKERRHIP(herr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  }
  ierr = PetscStreamRestoreStream(pstream,&hstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream,pscal->event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarAwait_HIP(PetscStreamScalar pscal, PetscScalar *val, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pscal->omask == PETSC_OFFLOAD_GPU) {
    hipStream_t hstream;
    hipError_t  herr;

    ierr = PetscStreamGetStream(pstream,&hstream);CHKERRQ(ierr);
    herr = hipMemcpyAsync(pscal->host,pscal->device,sizeof(PetscScalar),hipMemcpyDeviceToHost,hstream);CHKERRHIP(herr);
    ierr = PetscStreamRestoreStream(pstream,&hstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream,pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  }
  ierr = PetscEventSynchronize(pscal->event);CHKERRQ(ierr);
  *val = *pscal->host;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarRestoreHost_HIP(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscErrorCode ierr;
  hipError_t    cerr;
  hipStream_t   cstream;

  PetscFunctionBegin;
  /* Assumption is that a host-side write is due to performing an operation not possible on device, but that device will
   soon use result. So we immediately pipe value to device */
  ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  cerr = hipMemcpyAsync(pscal->device, pscal->host, sizeof(PetscScalar), hipMemcpyHostToDevice, cstream);CHKERRHIP(cerr);
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
  pscal->omask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarGetDevice_HIP(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscFunctionBegin;
  if (pscal->omask == PETSC_OFFLOAD_CPU) {
    PetscErrorCode ierr;
    hipStream_t   cstream;
    hipError_t    cerr;

    ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    cerr = hipMemcpyAsync(pscal->device, pscal->host, sizeof(PetscScalar), hipMemcpyHostToDevice, cstream);CHKERRHIP(cerr);
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  }
  *val = pscal->device;
  PetscFunctionReturn(0);
}

static const struct _ScalOps scalhipops = {
  PetscStreamScalarDestroy_HIP,
  PetscStreamScalarSetup_HIP,
  PetscStreamScalarSetValue_HIP,
  PetscStreamScalarAwait_HIP,
  PetscStreamScalarGetDevice_HIP,
  NULL,
  NULL,
  NULL
};

PetscErrorCode PetscStreamScalarCreate_HIP(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"NOT FULLY IMPLEMENTED");
  {
    PetscErrorCode ierr;
    hipError_t     herr;

    herr = hipHostMalloc((void **)&pscal->host,sizeof(PetscScalar));CHKERRHIP(herr);
    herr = hipMalloc((void **)&pscal->device,sizeof(PetscScalar));CHKERRHIP(herr);
    ierr = PetscMemcpy(pscal->ops,&scalhipops,sizeof(scalhipops));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
