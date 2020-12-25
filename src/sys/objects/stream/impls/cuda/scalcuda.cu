#include "streamcuda.h" /*I "petscdevice.h" I*/

/* How to change this? None of these routine accept a communicator on which they may call
   petscoptions routines... */
static const PetscInt poolSize = 100;
static PetscScalar    *devicePool, *hostPool;
static PetscInt       *poolIDs;
static PetscBool      poolSetup = PETSC_FALSE;

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarDestroy_CUDA(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!poolIDs[pscal->poolID])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pool ID %D has apparently already been released!",pscal->poolID);
  poolIDs[pscal->poolID] = 0;
  pscal->poolID = PETSC_DEFAULT;
  pscal->host = NULL;
  pscal->device = NULL;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSetup_CUDA(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < poolSize; ++i) {
    /* Find open slot */
    if (!poolIDs[i]) {
      poolIDs[i]    = 1;
      pscal->poolID = i;
      break;
    }
  }
  /* If pscal->poolID was unchanged it means no free pool alloc is found, and since CUDA has no cudaRealloc() function,
   we have no choice but to error out here */
  if (PetscUnlikely(pscal->poolID == PETSC_DEFAULT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Maximum pool allocation %D reached, no free pool ID remains. Rerun with larger pool size",poolSize);
  pscal->device = devicePool+pscal->poolID;
  pscal->host   = hostPool+pscal->poolID;
  pscal->omask  = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSetValue_CUDA(PetscStreamScalar pscal, const PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  cudaStream_t   cstream;

  PetscFunctionBegin;
  ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  if (val) {
    if (PetscMemTypeHost(mtype)) {
      /* If host value we copy to both */
      cerr = cudaMemcpyAsync(pscal->device, val, sizeof(PetscScalar), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
      cerr = cudaMemcpyAsync(pscal->host, val, sizeof(PetscScalar), cudaMemcpyHostToHost, cstream);CHKERRCUDA(cerr);
      pscal->omask = PETSC_OFFLOAD_BOTH;
    } else {
      /* If device-only we leave on device, no need to tie up PCI lanes unnecessarily */
      cerr = cudaMemcpyAsync(pscal->device, val, sizeof(PetscScalar), cudaMemcpyDeviceToDevice, cstream);CHKERRCUDA(cerr);
      pscal->omask = PETSC_OFFLOAD_GPU;
    }
  } else {
    /* use NULL as 0 since it can be used for both host and device memtype */
    cerr = cudaMemsetAsync(pscal->device, 0, sizeof(PetscScalar), cstream);CHKERRQ(ierr);
    cerr = cudaMemsetAsync(pscal->host, 0, sizeof(PetscScalar), cstream);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  }
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarAwait_CUDA(PetscStreamScalar pscal, PetscScalar *val, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
  if (pscal->omask == PETSC_OFFLOAD_GPU) {
    cudaStream_t cstream;
    cudaError_t  cerr;

    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(pscal->host, pscal->device, sizeof(PetscScalar), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  }
  ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
  *val = *pscal->host;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarUpdateDevice_CUDA_Internal(PetscStreamScalar pscal, PetscBool update, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Wait event here in case this stream scalar had work queued on another stream */
  ierr = PetscStreamWaitEvent(pstream,pscal->event);CHKERRQ(ierr);
  if (update && pscal->omask == PETSC_OFFLOAD_CPU) {
    cudaStream_t cstream;
    cudaError_t  cerr;

    ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(pscal->device,pscal->host,sizeof(PetscScalar),cudaMemcpyHostToDevice,cstream);CHKERRCUDA(cerr);
    ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    /* Record event so other streams can wait on it */
    ierr = PetscStreamRecordEvent(pstream,pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  } else {pscal->omask = PETSC_OFFLOAD_GPU;}
  /* Note no wait require if the value was never moved */
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarGetDevice_CUDA(PetscStreamScalar pscal, PetscScalar **val, PetscBool update, PetscStream pstream)
{
  const PetscScalar dummy = 1.0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarUpdateDevice_CUDA_Internal(pscal,update,pstream);CHKERRQ(ierr);
  *val = pscal->device;
  /* MEMTYPE_DEVICE == invalidate the cache */
  ierr = PetscStreamScalarUpdateCache_Internal(pscal,&dummy,PETSC_MEMTYPE_DEVICE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarRestoreDevice_CUDA(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  const PetscScalar dummy = 1.0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  pscal->omask = PETSC_OFFLOAD_GPU;
  *val = NULL;
  /* MEMTYPE_DEVICE == invalidate the cache, perhaps could launch kernel to check? */
  ierr = PetscStreamScalarUpdateCache_Internal(pscal,&dummy,PETSC_MEMTYPE_DEVICE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarAXTY_CUDA(PetscScalar a, PetscStreamScalar pscalx, PetscStreamScalar pscaly, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarUpdateDevice_CUDA_Internal(pscalx,PETSC_TRUE,pstream);CHKERRQ(ierr);
  /* pscaly == NULL means pscaly = 1.0 */
  if (pscaly) {
    ierr = PetscStreamScalarUpdateDevice_CUDA_Internal(pscaly,PETSC_TRUE,pstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamScalarAXTY_CUDA_Kernel(a,pscalx,pscaly,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarAYDX_CUDA(PetscScalar a, PetscStreamScalar pscaly, PetscStreamScalar pscalx, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarUpdateDevice_CUDA_Internal(pscalx,PETSC_TRUE,pstream);CHKERRQ(ierr);
  /* pscaly == NULL means pscaly = 1.0 */
  if (pscaly) {
    ierr = PetscStreamScalarUpdateDevice_CUDA_Internal(pscaly,PETSC_TRUE,pstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamScalarAYDX_CUDA_Kernel(a,pscaly,pscalx,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const struct _ScalOps scalcuops = {
  PetscStreamScalarCreate_CUDA,
  PetscStreamScalarDestroy_CUDA,
  PetscStreamScalarSetup_CUDA,
  PetscStreamScalarSetValue_CUDA,
  PetscStreamScalarAwait_CUDA,
  PetscStreamScalarGetDevice_CUDA,
  PetscStreamScalarRestoreDevice_CUDA,
  PetscStreamScalarAXTY_CUDA,
  PetscStreamScalarAYDX_CUDA
};

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarFinalize_CUDA(void)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;

  PetscFunctionBegin;
  /* Heres hoping this no-op loop is optimized away by compiler when not in debug... */
  for (int i = 0; i < poolSize; ++i) {
    /* Probably wrong error type, I can't tell what the canonical "you haven't cleaned up" error code is */
    if (PetscUnlikelyDebug(poolIDs[i])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEMC,"CUDA PetscStreamScalar pool ID %D is still checked out in PetscFinalize()",poolIDs[i]);
  }
  cerr = cudaFreeHost(hostPool);CHKERRCUDA(cerr);
  cerr = cudaFree(devicePool);CHKERRCUDA(cerr);
  ierr = PetscFree(poolIDs);CHKERRQ(ierr);
  poolSetup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarCreate_CUDA(PetscStreamScalar pscal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!poolSetup) {
    cudaError_t cerr;

    poolSetup = PETSC_TRUE;
    /* Use cudaHostAlloc to allow the flag to be changed, could maybe be useful */
    cerr = cudaHostAlloc((void **) &hostPool, poolSize*sizeof(PetscScalar), cudaHostAllocDefault);CHKERRCUDA(cerr);
    cerr = cudaMemset(hostPool, 0, poolSize*sizeof(PetscScalar));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void **) &devicePool, poolSize*sizeof(PetscScalar));CHKERRCUDA(cerr);
    cerr = cudaMemset(devicePool, 0, poolSize*sizeof(PetscScalar));CHKERRCUDA(cerr);
    ierr = PetscCalloc1(poolSize, &poolIDs);CHKERRQ(ierr);
    ierr = PetscRegisterFinalize(PetscStreamScalarFinalize_CUDA);CHKERRQ(ierr);
  }
  ierr = PetscMemcpy(pscal->ops, &scalcuops, sizeof(scalcuops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
