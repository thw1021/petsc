/*
 Management of HIPBLAS and HIPSOLVER handles

 Unlike CUDA, hipSOLVER is just for dense matrices so there is
 no distinguishing being dense and sparse.  Also, hipSOLVER is
 very immature so we often have to do the mapping between roc and
 cuda manually.
 */

#include <petscsys.h>
#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

static hipblasHandle_t     hipblasv2handle   = NULL;
static hipsolverHandle_t   hipsolverhandle = NULL;

/*
   Destroys the HIPBLAS handle.
   This function is intended and registered for PetscFinalize - do not call manually!
 */
static PetscErrorCode PetscHIPBLASDestroyHandle()
{
  hipblasStatus_t cberr;

  PetscFunctionBegin;
  if (hipblasv2handle) {
    cberr          = hipblasDestroy(hipblasv2handle);CHKERRHIPBLAS(cberr);
    hipblasv2handle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

/*
    Initializing the hipBLAS handle can take 1/2 a second therefore
    initialize in PetscInitialize() before being timing so it does
    not distort the -log_view information
*/
PetscErrorCode PetscHIPBLASInitializeHandle(void)
{
  PetscErrorCode ierr;
  hipblasStatus_t cberr;

  PetscFunctionBegin;
  if (!hipblasv2handle) {
    cberr = hipblasCreate(&hipblasv2handle);CHKERRHIPBLAS(cberr);
    /* Make sure that the handle will be destroyed properly */
    ierr = PetscRegisterFinalize(PetscHIPBLASDestroyHandle);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t *handle)
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  ierr = PetscDeviceContextGetCurrentContextAssertKind_Internal(&dctx,PETSC_DEVICE_HIP);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* hipsolver */
static PetscErrorCode PetscHIPSOLVERDestroyHandle()
{
  hipsolverStatus_t  cerr;

  PetscFunctionBegin;
  if (hipsolverhandle) {
    cerr             = hipsolverDestroy(hipsolverhandle);CHKERRHIPSOLVER(cerr);
    hipsolverhandle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHIPSOLVERInitializeHandle(void)
{
  PetscErrorCode    ierr;
  hipsolverStatus_t  cerr;

  PetscFunctionBegin;
  if (!hipsolverhandle) {
    cerr = hipsolverCreate(&hipsolverhandle);CHKERRHIPSOLVER(cerr);
    ierr = PetscRegisterFinalize(PetscHIPSOLVERDestroyHandle);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t *handle)
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  ierr = PetscDeviceContextGetCurrentContextAssertKind_Internal(&dctx,PETSC_DEVICE_HIP);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetSOLVERHandle_Internal(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
