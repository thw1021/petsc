#include "streamcuda.h"

#if PetscDefined(HAVE_CUDA)
static PetscErrorCode PetscStreamScalarDestroy_CUDA(PetscStreamScalar pscal)
{
  cudaError_t cerr;

  PetscFunctionBegin;
  cerr = cudaFreeHost(pscal->host);CHKERRCUDA(cerr);
  cerr = cudaFree(pscal->device);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarSetup_CUDA(PetscStreamScalar pscal, PetscStream pstream)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  cudaStream_t   cstream;

  PetscFunctionBegin;
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  cerr = cudaMemsetAsync(pscal->device, 0, sizeof(PetscScalar), cstream);CHKERRCUDA(cerr);
  *pscal->host = (PetscScalar)0.0;
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
  pscal->omask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarSetValue_CUDA(PetscStreamScalar pscal, const PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  cudaStream_t   cstream;

  PetscFunctionBegin;
  ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  if (val) {
    if (PetscMemTypeHost(mtype)) {
      cerr = cudaMemcpyAsync(pscal->host, val, sizeof(PetscScalar), cudaMemcpyHostToHost, cstream);CHKERRCUDA(cerr);
      cerr = cudaMemcpyAsync(pscal->device, val, sizeof(PetscScalar), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
    } else {
      cerr = cudaMemcpyAsync(pscal->device, val, sizeof(PetscScalar), cudaMemcpyDeviceToDevice, cstream);CHKERRCUDA(cerr);
      cerr = cudaMemcpyAsync(pscal->host, val, sizeof(PetscScalar), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
    }
  } else {
    cerr = cudaMemsetAsync(pscal->device, 0, sizeof(PetscScalar), cstream);CHKERRQ(ierr);
    cerr = cudaMemsetAsync(pscal->host, 0, sizeof(PetscScalar), cstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
  pscal->omask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamScalarGetHost_CUDA(PetscStreamScalar pscal, PetscScalar **val, PetscBool update, PetscStream pstream)
{
  PetscFunctionBegin;
  /* Sometimes we just want the host pointer, such as during writes */
  if (update && pscal->omask == PETSC_OFFLOAD_GPU) {
    PetscErrorCode ierr;
    cudaStream_t   cstream;
    cudaError_t    cerr;

    ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(pscal->host, pscal->device, sizeof(PetscScalar), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
    ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
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

static PetscErrorCode PetscStreamScalarAccumOpDispatch_CUDA(PetscStreamScalar pscalret, PetscInt n, PetscStreamScalar pscal[], PetscStreamComputeOp epiop, PetscStreamComputeOp accop, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarAccumOpDispatch_Internal(pscalret, n, pscal, epiop, accop, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const struct _ScalOps scalcuops = {
  PetscStreamScalarCreate_CUDA,
  PetscStreamScalarDestroy_CUDA,
  PetscStreamScalarSetup_CUDA,
  PetscStreamScalarSetValue_CUDA,
  PetscStreamScalarGetHost_CUDA,
  PetscStreamScalarRestoreHost_CUDA,
  PetscStreamScalarGetDevice_CUDA,
  NULL,
  PetscStreamScalarAccumOpDispatch_CUDA
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
