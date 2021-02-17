#include "streamhip.h"

#if PetscDefined(HAVE_HIP)
static PetscErrorCode PetscStreamScalarDestroy_HIP(PetscStreamScalar pscal)
{
  hipError_t cerr;

  PetscFunctionBegin;
  cerr = hipFree(pscal->host);CHKERRHIP(cerr);
  cerr = hipFree(pscal->device);CHKERRHIP(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarSetup_HIP(PetscStreamScalar pscal, PetscStream pstream)
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

static PetscErrorCode PetscStreamScalarSetValue_HIP(PetscStreamScalar pscal, const PetscScalar *val, PetscMemType mtype, PetscStream pstream)
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

static PetscErrorCode PetscStreamScalarGetHost_HIP(PetscStreamScalar pscal, PetscScalar **val, PetscBool update, PetscStream pstream)
{
  PetscFunctionBegin;
  /* Sometimes we just want the host pointer, such as during writes */
  if (update && pscal->omask == PETSC_OFFLOAD_GPU) {
    PetscErrorCode ierr;
    hipStream_t   cstream;
    hipError_t    cerr;

    ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    cerr = hipMemcpyAsync(pscal->host, pscal->device, sizeof(PetscScalar), hipMemcpyDeviceToHost, cstream);CHKERRHIP(cerr);
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
    ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
  }
  *val = pscal->host;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarRestoreHost_HIP(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
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

static PetscErrorCode PetscStreamScalarGetDevice_HIP(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
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

PetscErrorCode PetscStreamScalarAccumOpDispatch_HIP(PetscStreamScalar pscalret, PetscInt n, PetscStreamScalar pscal[], PetscStreamComputeOp epiop, PetscStreamComputeOp accop, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarAccumOpDispatch_Internal(pscalret, n, pscal, epiop, accop, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _ScalOps scalhipops = {
  PetscStreamScalarCreate_HIP,
  PetscStreamScalarDestroy_HIP,
  PetscStreamScalarSetup_HIP,
  PetscStreamScalarSetValue_HIP,
  PetscStreamScalarGetHost_HIP,
  PetscStreamScalarRestoreHost_HIP,
  PetscStreamScalarGetDevice_HIP,
  NULL,
  PetscStreamScalarAccumOpDispatch_HIP
};
#endif /* HAVE_HIP */

PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate_HIP(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_HIP)
  {
    PetscErrorCode ierr;
    hipError_t    cerr;

    cerr = hipMallocHost((void **) &pscal->host, sizeof(PetscScalar));CHKERRHIP(cerr);
    cerr = hipMalloc((void **) &pscal->device, sizeof(PetscScalar));CHKERRHIP(cerr);
    ierr = PetscMemcpy(pscal->ops, &scalhipops, sizeof(scalcuops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with HIP support\n");
#endif
  PetscFunctionReturn(0);
}
