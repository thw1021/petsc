#include "streamcuda.h"

#if PetscDefined(HAVE_CUDA)
static PetscErrorCode PetscStreamScalarDestroy_CUDA(PetscStreamScalar pscal)
{
  cudaError_t cerr;

  PetscFunctionBegin;
  cerr = cudaFree(pscal->host);CHKERRCUDA(cerr);
  cerr = cudaFree(pscal->device);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarSetup_CUDA(PetscStreamScalar pscal, PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  cudaStream_t   cstream;

  PetscFunctionBegin;
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  if (PetscMemTypeHost(mtype)) {
    *pscal->host = *val;
    cerr = cudaMemcpyAsync(pscal->device, pscal->host, sizeof(PetscScalar), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
  } else {
    cerr = cudaMemcpyAsync(pscal->device, val, sizeof(PetscScalar), cudaMemcpyDeviceToDevice, cstream);CHKERRCUDA(cerr);
    cerr = cudaMemcpyAsync(pscal->host, val, sizeof(PetscScalar), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
    ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
  pscal->isZero = (PetscBool)(*pscal->host == (PetscScalar)0.0);
  pscal->isOne  = (PetscBool)(*pscal->host == (PetscScalar)1.0);
  pscal->omask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarGetHost_CUDA(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscFunctionBegin;
  if (pscal->omask == PETSC_OFFLOAD_GPU) {
    PetscErrorCode ierr;
    cudaStream_t   cstream;
    cudaError_t    cerr;

    ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(pscal->host, pscal->device, sizeof(PetscScalar), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  }
  *val = pscal->host;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarRestoreHost_CUDA(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  cudaStream_t   cstream;

  PetscFunctionBegin;
  /* Assumption is that a host-side write is due to performing an operation not possible on device, but that device will
   soon use result. So we immediately pipe value to device */
  ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  cerr = cudaMemcpyAsync(pscal->device, pscal->host, sizeof(PetscScalar), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
  pscal->omask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarGetDevice_CUDA(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscFunctionBegin;
  if (pscal->omask == PETSC_OFFLOAD_CPU) {
    PetscErrorCode ierr;
    cudaStream_t   cstream;
    cudaError_t    cerr;

    ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(pscal->device, pscal->host, sizeof(PetscScalar), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  }
  *val = pscal->device;
  PetscFunctionReturn(0);
}

static struct _ScalOps scalcuops = {
  PetscStreamScalarCreate_CUDA,
  PetscStreamScalarDestroy_CUDA,
  PetscStreamScalarSetup_CUDA,
  PetscStreamScalarGetHost_CUDA,
  PetscStreamScalarRestoreHost_CUDA,
  PetscStreamScalarGetDevice_CUDA,
  NULL
};
#endif /* HAVE_CUDA */

PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate_CUDA(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_CUDA)
  {
    PetscErrorCode ierr;
    cudaError_t    cerr;

    cerr = cudaMallocHost((void **) &pscal->host, sizeof(PetscScalar));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **) &pscal->device, sizeof(PetscScalar));CHKERRCUDA(cerr);
    ierr = PetscMemcpy(pscal->ops, &scalcuops, sizeof(scalcuops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with CUDA support\n");
#endif
  PetscFunctionReturn(0);
}
