/*
 Management of CUBLAS and CUSOLVER handles
 */

#include <petscsys.h>
#include <petsc/private/petscimpl.h>
#include <petsccublas.h>
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
#include <omp.h>
#endif
static PetscBool          thread_2h_initialized = PETSC_FALSE;
static PetscBool          thread_solverh_initialized = PETSC_FALSE;
static cublasHandle_t     cublasv2handle[PETSC_MAX_THREADS];
static cusolverDnHandle_t cusolverdnhandle[PETSC_MAX_THREADS];

/*
   Destroys the CUBLAS handle.
   This function is intended and registered for PetscFinalize - do not call manually!
 */
static PetscErrorCode PetscCUBLASDestroyHandle()
{
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (thread_2h_initialized) {
    PetscInt i=0;
    do {
      if (cublasv2handle[i]) {
        cberr             = cublasDestroy(cublasv2handle[i]);CHKERRCUBLAS(cberr);
        cublasv2handle[i] = NULL;  /* Ensures proper reinitialization */
      }
    } while (++i < PETSC_MAX_THREADS);
  }
  PetscFunctionReturn(0);
}

/*
    Initializing the cuBLAS handle can take 1/2 a second therefore
    initialize in PetscInitialize() before being timing so it does
    not distort the -log_view information
*/
PetscErrorCode PetscCUBLASInitializeHandle(PetscInt tid)
{
  PetscErrorCode ierr;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (!thread_2h_initialized) {
    for (int i=0; i<PETSC_MAX_THREADS; i++) cublasv2handle[i] = NULL;
    thread_2h_initialized = PETSC_TRUE;
    /* Make sure that the handle will be destroyed properly */
    ierr = PetscRegisterFinalize(PetscCUBLASDestroyHandle);CHKERRQ(ierr);
  }
  if (!cublasv2handle[tid]) {
    for (int i=0; i<3; i++) {
      cberr = cublasCreate(&cublasv2handle[tid]);
      if (cberr == CUBLAS_STATUS_SUCCESS) break;
      if (cberr != CUBLAS_STATUS_ALLOC_FAILED && cberr != CUBLAS_STATUS_NOT_INITIALIZED) CHKERRCUBLAS(cberr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (cberr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS for thread %D", tid);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle)
{
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  PetscInt tid = omp_get_thread_num();
#else
  PetscInt tid = 0;
#endif

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  if (!thread_2h_initialized || !cublasv2handle[tid]) {ierr = PetscCUBLASInitializeHandle(tid);CHKERRQ(ierr);}
  *handle = cublasv2handle[tid];
  PetscFunctionReturn(0);
}

/* cusolver */
static PetscErrorCode PetscCUSOLVERDnDestroyHandle()
{
  cusolverStatus_t cerr;

  PetscFunctionBegin;
  if (thread_solverh_initialized) {
    PetscInt i=0;
    do {
      if (cusolverdnhandle[i]) {
        cerr               = cusolverDnDestroy(cusolverdnhandle[i]);CHKERRCUSOLVER(cerr);
        cusolverdnhandle[i] = NULL;  /* Ensures proper reinitialization */
      }
    } while (++i < PETSC_MAX_THREADS);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnInitializeHandle(PetscInt tid)
{
  PetscErrorCode   ierr;
  cusolverStatus_t cerr;

  PetscFunctionBegin;
  if (!thread_solverh_initialized) {
    for (int i=0; i<PETSC_MAX_THREADS; i++) cusolverdnhandle[i] = NULL;
    thread_solverh_initialized = PETSC_TRUE;
    /* Make sure that the handle will be destroyed properly */
    ierr = PetscRegisterFinalize(PetscCUSOLVERDnDestroyHandle);CHKERRQ(ierr);
  }
  if (!cusolverdnhandle[tid]) {
    for (int i=0; i<3; i++) {
      cerr = cusolverDnCreate(&cusolverdnhandle[tid]);
      if (cerr == CUSOLVER_STATUS_SUCCESS) break;
      if (cerr != CUSOLVER_STATUS_ALLOC_FAILED) CHKERRCUSOLVER(cerr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (cerr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize Solver for thread %D", tid);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  PetscErrorCode     ierr;
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  PetscInt tid = omp_get_thread_num();
#else
  PetscInt tid = 0;
#endif

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  if (!thread_solverh_initialized || !cusolverdnhandle[tid]) {ierr = PetscCUSOLVERDnInitializeHandle(tid);CHKERRQ(ierr);}
  *handle = cusolverdnhandle[tid];
  PetscFunctionReturn(0);
}
