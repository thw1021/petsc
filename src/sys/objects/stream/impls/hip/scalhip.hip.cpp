#include "streamhip.h"

/* If this every becomes changeable, also change error message in
   PetscStreamScalarSetup_HP */
static const PetscInt poolSize = 100;
static PetscScalar    *devicePool, *hostPool;
static PetscInt       *poolIDs;
static PetscBool      poolSetup = PETSC_FALSE;

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarDestroy_HIP(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!poolIDs[pscal->poolID])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pool ID%D has apparently already been released!",pscal->poolID);
  --poolIDs[pscal->poolID];
  pscal->poolID = PETSC_DEFAULT;
  pscal->host   = NULL;
  pscal->device = NULL;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSetup_HIP(PetscStreamScalar pscal)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(pscal->poolID != PETSC_DEFAULT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscStreamScalar already has an assigned pool ID: %D",pscal->poolID);
  for (PetscInt i = 0; i < poolSize; ++i) {
    /* Find open slot */
    if (!poolIDs[i]) {
      ++poolIDs[i];
      pscal->poolID = i;
      break;
    }
  }
  /* If pscal->poolID was unchanged it means no free pool alloc is found, and since HIP has no hipRealloc() function,
     we have no choice but to error out here */
  if (PetscUnlikelyDebug(pscal->poolID == PETSC_DEFAULT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Maximum pool allocation %D reached, no free pool ID remains. Recompile with larger pool size",poolSize);
  pscal->device = devicePool+pscal->poolID;
  pscal->host   = hostPool+pscal->poolID;
  pscal->omask  = PETSC_OFFLOAD_BOTH;
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
      /* If host value we copy to both */
      herr = hipMemcpyAsync(pscal->device,val,sizeof(PetscScalar),hipMemcpyHostToDevice,hstream);CHKERRHIP(herr);
      herr = hipMemcpyAsync(pscal->host,val,sizeof(PetscScalar),hipMemcpyHostToHost,hstream);CHKERRHIP(herr);
      pscal->omask = PETSC_OFFLOAD_BOTH;
    } else {
      /* If device-only we leave on device, no need to tie up PCI lanes unnecessarily */
      herr = hipMemcpyAsync(pscal->device,val,sizeof(PetscScalar),hipMemcpyDeviceToDevice,hstream);CHKERRHIP(herr);
      pscal->omask = PETSC_OFFLOAD_GPU;
    }
  } else {
    /* use NULL as 0 since it can be used for both host and device memtype */
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

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarUpdateDevice_HIP_Internal(PetscStreamScalar pscal, PetscBool update, PetscStream pstream)
{
  PetscFunctionBegin;
  if (update && pscal->omask == PETSC_OFFLOAD_CPU) {
    PetscErrorCode ierr;
    hipStream_t    hstream;
    hipError_t     herr;

    ierr = PetscStreamGetStream(pstream,&hstream);CHKERRQ(ierr);
    herr = hipMemcpyAsync(pscal->device,pscal->host,sizeof(PetscScalar),hipMemcpyHostToDevice,hstream);CHKERRHIP(herr);
    ierr = PetscStreamRestoreStream(pstream,&hstream);CHKERRQ(ierr);
    /* Record event so other streams can wait on it */
    ierr = PetscStreamRecordEvent(pstream,pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  } else {pscal->omask = PETSC_OFFLOAD_GPU;}
  /* Note no wait require if the value was never moved */
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarGetDevice_HIP(PetscStreamScalar pscal, PetscScalar **val, PetscBool update, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarUpdateDevice_HIP_Internal(pscal,update,pstream);CHKERRQ(ierr);
  *val = pscal->device;
  if (update) {
    /* MEMTYPE_DEVICE == invalidate the cache */
    const PetscScalar dummy = 1;
    ierr = PetscStreamScalarUpdateCache_Internal(pscal,&dummy,PETSC_MEMTYPE_DEVICE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarRestoreDevice_HIP(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  const PetscScalar dummy = 1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  pscal->omask = PETSC_OFFLOAD_GPU;
  *val = NULL;
  /* MEMTYPE_DEVICE == invalidate the cache, perhaps could launch kernel to check? */
  ierr = PetscStreamScalarUpdateCache_Internal(pscal,&dummy,PETSC_MEMTYPE_DEVICE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const struct _ScalOps scalhipops = {
  PetscStreamScalarCreate_HIP,
  PetscStreamScalarDestroy_HIP,
  PetscStreamScalarSetup_HIP,
  PetscStreamScalarSetValue_HIP,
  PetscStreamScalarAwait_HIP,
  PetscStreamScalarGetDevice_HIP,
  PetscStreamScalarRestoreDevice_HIP,
  NULL,
  NULL
};

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarFinalize_HIP(void)
{
  PetscErrorCode ierr;
  hipError_t    herr;

  PetscFunctionBegin;
  /* Heres hoping this no-op loop is optimized away by compiler when not in debug... */
  for (int i = 0; i < poolSize; ++i) {
    /* Probably wrong error type, I can't tell what the canonical "you haven't cleaned up" error code is */
    if (PetscUnlikelyDebug(poolIDs[i])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEMC,"HIP PetscStreamScalar pool ID %d is still checked out in PetscFinalize()",i);
  }
  herr = hipHostFree(hostPool);CHKERRHIP(herr);
  herr = hipFree(devicePool);CHKERRHIP(herr);
  ierr = PetscFree(poolIDs);CHKERRQ(ierr);
  poolSetup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarCreate_HIP(PetscStreamScalar pscal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!poolSetup) {
    hipError_t herr;

    poolSetup = PETSC_TRUE;
    /* Use hipHostMalloc to allow the flag to be changed, could maybe be useful */
    herr = hipHostMalloc((void **)&hostPool,poolSize*sizeof(PetscScalar),0);CHKERRHIP(herr);
    herr = hipMemset(hostPool,0,poolSize*sizeof(PetscScalar));CHKERRHIP(herr);
    herr = hipMalloc((void **)&devicePool,poolSize*sizeof(PetscScalar));CHKERRHIP(herr);
    herr = hipMemset(devicePool,0,poolSize*sizeof(PetscScalar));CHKERRHIP(herr);
    ierr = PetscCalloc1(poolSize,&poolIDs);CHKERRQ(ierr);
    ierr = PetscRegisterFinalize(PetscStreamScalarFinalize_HIP);CHKERRQ(ierr);
  }
  ierr = PetscMemcpy(pscal->ops,&scalhipops,sizeof(scalhipops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
