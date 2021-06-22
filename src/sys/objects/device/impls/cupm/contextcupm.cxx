#include "contextcupm.hpp" /*I "petscdevice.h" I*/

enum class PetscDeviceBackends {CUDA, HIP};

template <PetscDeviceBackends T> struct cupmTypeTraits;

template <>
struct cupmTypeTraits<PetscDeviceBackends::CUDA>
{
  static constexpr char name[] = "CUDA";

  typedef cudaError_t        cupmError_t;
  typedef cudaEvent_t        cupmEvent_t;
  typedef cudaStream_t       cupmStream_t;
  typedef cublasHandle_t     cupmBlasHandle_t;
  typedef cublasStatus_t     cupmBlasError_t;
  typedef cusolverDnHandle_t cupmSolverHandle_t;
  typedef cusolverStatus_t   cupmSolverError_t;

  static constexpr auto cupmGetErrorName   = cudaGetErrorName;
  static constexpr auto cupmGetErrorString = cudaGetErrorString;
  static constexpr auto cupmEventCreate    = cudaEventCreate;
  static constexpr auto cupmEventDestroy   = cudaEventDestroy;
  static constexpr auto cupmStreamCreate   = cudaStreamCreate;
  static constexpr auto cupmStreamDestroy  = cudaStreamDestroy;
};

template <>
struct cupmTypeTraits<PetscDeviceBackends::HIP>
{
  static constexpr char name[] = "HIP";

  typedef hipError_t        cupmError_t;
  typedef hipEvent_t        cupmEvent_t;
  typedef hipStream_t       cupmStream_t;
  typedef rocblas_handle    cupmBlasHandle_t;
  typedef roblas_status     cupmBlasError_t;
  typedef hipsolverHandle_t cupmSolverHandle_t;
  typedef hipsolverStatus_t cupmSolverError_t;

  static constexpr auto cupmGetErrorName   = hipGetErrorName;
  static constexpr auto cupmGetErrorString = hipGetErrorString;
  static constexpr auto cupmEventCreate    = hipEventCreate;
  static constexpr auto cupmEventDestroy   = hipEventDestroy;
  static constexpr auto cupmStreamCreate   = hipStreamCreate;
  static constexpr auto cupmStreamDestroy  = hipStreamDestroy;
};

#define CHKERRCUPM(cerr)                                                \
  do {                                                                  \
    if (PetscUnlikely(cerr)) {                                          \
      const char *name  = cupmGetErrorName(cerr);                       \
      const char *descr = cupmGetErrorString(cerr);                     \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",cupmName,(int)cerr,name,descr); \
    }                                                                   \
  } while (0)

template <PetscDeviceBackends T>
class cupmContext : cupmTypeTraits<T>
{
protected:
  using cupmType_t = cupmTypeTraits<T>;
  using cupmName   = cupmType_t::name;

  using cupmType_t::cupmError_t;
  using cupmType_t::cupmEvent_t;
  using cupmType_t::cupmStream_t;
  using cupmType_t::cupmBlasError_t;
  using cupmType_t::cupmSolverError_t;
  using cupmType_t::cupmBlasHandle_t;
  using cupmType_t::cupmSolverHandle_t;

  using cupmType_t::cupmGetErrorName;
  using cupmType_t::cupmGetErrorString;

  static cupmBlasHandle_t   _blas;
  static cupmSolverHandle_t _solver;

public:
  struct PetscDeviceContext_IMPLS
  {
    cupmStream_t       stream;
    cupmEvent_t        event;
    cupmBlasHandle_t   blas;
    cupmSolverHandle_t solver;
  };

  static PETSC_NODISCARD PetscErrorCode DestroyContext(PetscDeviceContext dctx) PETSC_NOEXCEPT
  {
    PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;
    cupmError_t              cperr;
    PetscErrorCode           ierr;

    PetscFunctionBegin;
    if (dci->stream) {cperr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cperr);}
    if (dci->event)  {cperr = cupmEventDestroy(dci->event);CHKERRCUPM(cperr);}
    ierr = PetscFree(dctx->data);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  explicit PETSC_CONSTEXPR cupmContext() PETSC_NOEXCEPT : _blas(NULL),_solver(NULL) {}
};

/* cublas */
static PetscErrorCode PetscCUBLASDestroyHandle_Internal(void)
{
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (cublasv2handle) {
    cberr          = cublasDestroy(cublasv2handle);CHKERRCUBLAS(cberr);
    cublasv2handle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

/* cusolver */
static PetscErrorCode PetscCUSOLVERDnDestroyHandle_Internal(void)
{
  cusolverStatus_t cerr;

  PetscFunctionBegin;
  if (cusolverdnhandle) {
    cerr             = cusolverDnDestroy(cusolverdnhandle);CHKERRCUSOLVER(cerr);
    cusolverdnhandle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCUBLASGetHandle_Internal(PetscDeviceContext_CUDA *dcu)
{
  cudaStream_t   cublasStream;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (!cublasv2handle) {
    PetscErrorCode ierr;

    for (int i=0; i<3; i++) {
      cberr = cublasCreate(&cublasv2handle);
      if (cberr == CUBLAS_STATUS_SUCCESS) break;
      if (cberr != CUBLAS_STATUS_ALLOC_FAILED && cberr != CUBLAS_STATUS_NOT_INITIALIZED) CHKERRCUBLAS(cberr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (cberr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS");
    /* Make sure that the handle will be destroyed properly */
    ierr = PetscRegisterFinalize(PetscCUBLASDestroyHandle_Internal);CHKERRQ(ierr);
  }
  cberr = cublasGetStream(cublasv2handle,&cublasStream);CHKERRCUBLAS(cberr);
  if (cublasStream != dcu->stream) {
    cberr = cublasSetStream(cublasv2handle,dcu->stream);CHKERRCUBLAS(cberr);
  }
  dcu->blas = cublasv2handle;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCUSOLVERDnGetHandle_Internal(PetscDeviceContext_CUDA *dcu)
{
  cudaStream_t     cusolverStream;
  cusolverStatus_t cerr;

  PetscFunctionBegin;
  if (!cusolverdnhandle) {
    PetscErrorCode ierr;

    for (int i=0; i<3; i++) {
      cerr = cusolverDnCreate(&cusolverdnhandle);
      if (cerr == CUSOLVER_STATUS_SUCCESS) break;
      if (cerr != CUSOLVER_STATUS_ALLOC_FAILED) CHKERRCUSOLVER(cerr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (cerr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuSolverDn");
    ierr = PetscRegisterFinalize(PetscCUSOLVERDnDestroyHandle_Internal);CHKERRQ(ierr);
  }
  cerr = cusolverDnGetStream(cusolverdnhandle,&cusolverStream);CHKERRCUSOLVER(cerr);
  if (cusolverStream != dcu->stream) {
    cerr = cusolverDnSetStream(cusolverdnhandle,dcu->stream);CHKERRCUSOLVER(cerr);
  }
  dcu->solver = cusolverdnhandle;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextDestroy_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  PetscErrorCode          ierr;
  cudaError_t             cerr;

  PetscFunctionBegin;
  if (dcu->stream) {cerr = cudaStreamDestroy(dcu->stream);CHKERRCUDA(cerr);}
  if (dcu->event)  {cerr = cudaEventDestroy(dcu->event);CHKERRCUDA(cerr);}
  /* don't need to do anything to the handles, they live on without us */
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextSetUp_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  PetscErrorCode          ierr;
  cudaError_t             cerr;

  PetscFunctionBegin;
  switch (dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* don't create a stream for global blocking */
    dcu->stream = NULL;
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
    cerr = cudaStreamCreate(&dcu->stream);CHKERRCUDA(cerr);
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    cerr = cudaStreamCreateWithFlags(&dcu->stream,cudaStreamNonBlocking);CHKERRCUDA(cerr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %D",(PetscInt)dctx->streamType);
    break;
  }
  cerr = cudaEventCreate(&dcu->event);CHKERRCUDA(cerr);
  ierr = PetscCUBLASGetHandle_Internal(dcu);CHKERRQ(ierr);
  ierr = PetscCUSOLVERDnGetHandle_Internal(dcu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextQuery_CUDA(PetscDeviceContext dctx, PetscBool *idle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  *idle = cudaStreamQuery(dcu->stream) == cudaErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextWaitForContext_CUDA(PetscDeviceContext dctxa, PetscDeviceContext dctxb)
{
  PetscDeviceContext_CUDA *dcua = (PetscDeviceContext_CUDA *)dctxa->data;
  PetscDeviceContext_CUDA *dcub = (PetscDeviceContext_CUDA *)dctxb->data;
  cudaError_t             cerr;

  PetscFunctionBegin;
  cerr = cudaEventRecord(dcub->event,dcub->stream);CHKERRCUDA(cerr);
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11011) /* 11.1.1 */
  cerr = cudaStreamWaitEvent(dcua->stream,dcub->event,cudaEventWaitDefault);CHKERRCUDA(cerr);
#else
  cerr = cudaStreamWaitEvent(dcua->stream,dcub->event,0);CHKERRCUDA(cerr);
#endif /* 11.1.1 */
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextSynchronize_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  cudaError_t             cerr;

  PetscFunctionBegin;
  /* in case anything was queued on the event */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11011) /* 11.1.1 */
  cerr = cudaStreamWaitEvent(dcu->stream,dcu->event,cudaEventWaitDefault);CHKERRCUDA(cerr);
#else
  cerr = cudaStreamWaitEvent(dcu->stream,dcu->event,0);CHKERRCUDA(cerr);
#endif /* 11.1.1 */
  cerr = cudaStreamSynchronize(dcu->stream);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static const struct _DeviceContextOps cuops = {
  PetscDeviceContextCreate_CUDA,
  PetscDeviceContextDestroy_CUDA,
  PetscDeviceContextSetUp_CUDA,
  PetscDeviceContextQuery_CUDA,
  PetscDeviceContextWaitForContext_CUDA,
  PetscDeviceContextSynchronize_CUDA,
};

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dcu);CHKERRQ(ierr);
  dctx->data = (void *)dcu;
  ierr = PetscMemcpy(dctx->ops,&cuops,sizeof(cuops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
