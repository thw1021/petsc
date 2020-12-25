#include "streamhip.h"

#if PetscDefined(HAVE_HIP)
PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarDestroy_HIP(PetscStreamScalar pscal)
{
  hipError_t cerr;

  PetscFunctionBegin;
  cerr = hipFreeHost(pscal->host);CHKERRHIP(cerr);
  cerr = hipFree(pscal->device);CHKERRHIP(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSetup_HIP(PetscStreamScalar pscal, PetscStream pstream)
{
  PetscErrorCode ierr;
  hipError_t    cerr;
  hipStream_t   cstream;

  PetscFunctionBegin;
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  cerr = hipMemsetAsync(pscal->device, 0, sizeof(PetscScalar), cstream);CHKERRHIP(cerr);
  *pscal->host = (PetscScalar)0.0;
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
  pscal->omask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSetValue_HIP(PetscStreamScalar pscal, const PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;
  hipError_t    cerr;
  hipStream_t   cstream;

  PetscFunctionBegin;
  ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  if (val) {
    if (PetscMemTypeHost(mtype)) {
      *pscal->host = *val;
      cerr = hipMemcpyAsync(pscal->device, val, sizeof(PetscScalar), hipMemcpyHostToDevice, cstream);CHKERRHIP(cerr);
    } else {
      cerr = hipMemcpyAsync(pscal->device, val, sizeof(PetscScalar), hipMemcpyDeviceToDevice, cstream);CHKERRHIP(cerr);
      cerr = hipMemcpyAsync(pscal->host, val, sizeof(PetscScalar), hipMemcpyDeviceToHost, cstream);CHKERRHIP(cerr);
    }
  } else {
    *pscal->host = (PetscScalar)0.0;
    cerr = hipMemsetAsync(pscal->device, 0, sizeof(PetscScalar), cstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
  pscal->omask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarAwait_HIP(PetscStreamScalar pscal, PetscScalar *val, PetscStream pstream)
{
  PetscFunctionBegin;
  /* Sometimes we just want the host pointer, such as during writes */
  if (pscal->omask == PETSC_OFFLOAD_GPU) {
    PetscErrorCode ierr;
    hipStream_t    hstream;
    hipError_t     herr;

    ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &hstream);CHKERRQ(ierr);
    herr = hipMemcpyAsync(pscal->host, pscal->device, sizeof(PetscScalar), hipMemcpyDeviceToHost, cstream);CHKERRHIP(herr);
    ierr = PetscStreamRestoreStream(pstream, &hstream);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
    ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
  }
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
  PetscStreamScalarCreate_HIP,
  PetscStreamScalarDestroy_HIP,
  PetscStreamScalarSetup_HIP,
  PetscStreamScalarSetValue_HIP,
  PetscStreamScalarAwait_HIP,
  PetscStreamScalarGetDevice_HIP,
  NULL,
  NULL,
  NULL
};
#endif /* HAVE_HIP */

PetscErrorCode PetscStreamScalarCreate_HIP(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_HIP)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"NOT FULLY IMPLEMENTED");
  {
    PetscErrorCode ierr;
    hipError_t     herr;

    herr = hipMallocHost((void **) &pscal->host, sizeof(PetscScalar));CHKERRHIP(herr);
    herr = hipMalloc((void **) &pscal->device, sizeof(PetscScalar));CHKERRHIP(herr);
    ierr = PetscMemcpy(pscal->ops, &scalhipops, sizeof(scalcuops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with HIP support");
#endif
  PetscFunctionReturn(0);
}
