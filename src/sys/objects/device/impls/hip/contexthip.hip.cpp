#include "contexthip.hip.hpp" /*I "petscdevice.h" I*/

static rocblas_handle    hipblashandle   = NULL;
static hipSolverHandle_t hipsolverhandle = NULL;

/* hipblas */
static PetscErrorCode PetscHIPBLASDestroyHandle_Internal(void)
{
  hipblasStatus_t hberr;

  PetscFunctionBegin;
  if (hipblasv2handle) {
    hberr         = hipblasDestroy(hipblashandle);CHKERRHIPBLAS(hberr);
    hipblashandle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

/* hipsolver */
static PetscErrorCode PetscHIPSOLVERDestroyHandle_Internal(void)
{
  hipsolverStatus_t hserr;

  PetscFunctionBegin;
  if (hipsolverhandle) {
    hserr           = hipsolverDestroy(hipsolverhandle);CHKERRHIPSOLVER(hserr);
    hipsolverhandle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscHIPBLASGetHandle_Internal(PetscDeviceContext_HIP *dch)
{
  hipStream_t    hipblasStream;
  rocblas_status rberr;

  PetscFunctionBegin;
  if (!hipblashandle) {
    PetscErrorCode ierr;

    rberr = hipblasCreate(&hipblashandle);CHKERRHIPBLAS(hberr);
    ierr = PetscRegisterFinalize(PetscHIPBLASDestroyHandleDestroyHandle_Internal);CHKERRQ(ierr);
  }
  rberr = hipblasGetStream(hipblashandle,&hipblasStream);CHKERRHIPBLAS(rberr);
  if (hipblasStream != dch->stream) {
    rberr = hipblasSetStream(hipblashandle,dch->stream);CHKERRHIPBLAS(rberr);
  }
  dch->blas = hipblashandle;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscHIPSOLVERGetHandle_Internal(PetscDeviceContext_HIP *dch)
{
  hipStream_t       hipsolverStream;
  hipsolverStatus_t hserr;

  PetscFunctionBegin;
  if (!hipsolverhandle) {
    PetscErrorCode ierr;

    hserr = hipsolverCreate(&hipsolverhandle);CHKERRHIPSOLVER(hserr);
    ierr = PetscRegisterFinalize(PetscHIPSOLVERDestroyHandle_Internal);CHKERRQ(ierr);
  }
  hserr = hipsolverGetStream(hipsolverhandle,&hipsolverStream);CHERRHIPSOLVER(hserr);
  if (hipsolverStream != dch->stream) {
    hserr = hipsolverSetStream(hipsolverhandle,dch->stream);CHKERRHIPSOLVER(hserr);
  }
  dch->solver = hipsolverhandle;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextDestroy_HIP(PetscDeviceContext dctx)
{
  PetscDeviceContext_HIP *dch = (PetscDeviceContext_HIP *)dctx->data;
  PetscErrorCode         ierr;
  hipError_t             herr;

  PetscFunctionBegin;
  if (dch->stream) {herr = hipStreamDestroy(dch->stream);CHKERRHIP(herr);}
  if (dch->event)  {herr = hipEventDestroy(dch->event);CHKERRHIP(herr);}
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
