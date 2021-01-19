#include <petsc/private/deviceimpl.h>

#if PetscDefined(HAVE_CUDA) && PetscDefined(HAVE_CXX)
#define PETSC_STREAM_DEVICE_POOL 100
static PetscStream PetscDefaultCUDAStream = NULL;

template <typename Type>
struct PetscDevicePool {
private:
  static PetscInt  reservedPool[PETSC_STREAM_DEVICE_POOL];
  static PetscInt  refcount = 0;
  static PetscBool setup = PETSC_FALSE;

protected:
  static Type *devicePool;
  static Type *hostPool;

public:
  PetscInt poolID;

  PetscDevicePool() : poolID(-1)
  {
    if (!(this->setup)) {
      PetscErrorCode ierr;
      cudaError_t    cerr;

      ierr = PetscArrayzero(this->reservedPool, PETSC_STREAM_DEVICE_POOL);CHKERRQ(ierr);
      cerr = cudaMalloc((void **)&this->devicePool, PETSC_STREAM_DEVICE_POOL*sizeof(Type));CHKERRCUDA(cerr);
      cerr = cudaMallocHost((void **)&this->hostPool, PETSC_STREAM_DEVICE_POOL*sizeof(Type));CHKERRCUDA(cerr);
      cerr = cudaMemset((void *)this->devicePool, 0, PETSC_STREAM_DEVICE_POOL*sizeof(Type));CHKERRCUDA(cerr);
      cerr = cudaMemset((void *)this->hostPool, 0, PETSC_STREAM_DEVICE_POOL*sizeof(Type));CHKERRCUDA(cerr);
      this->setup = PETSC_TRUE;
    }
    for (PetscInt i = 0; i < PETSC_STREAM_DEVICE_POOL; ++i) {
      if (!(this->reservedPool[i])) {
        this->reservedPool[i] = 1; // Reserved
        this->poolID = i;
        ++(this->refcount);
        break;
      }
    }
    if ((this->poolID) == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Cannot allocate additional scalars, pool is full\n");
  }

  ~PetscDevicePool()
  {
    this->reservedPool[this->poolID] = 0;
    if (!(--refcount)) {
      cudaError_t cerr;

      cerr = cudaFree((void *)this->devicePool);CHKERRCUDA(cerr);
      cerr = cudaFree((void *)this->hostPool);CHKERRCUDA(cerr);
      this->setup = PETSC_FALSE;
    }
  }
};

template <typename Type>
struct PetscDeviceScalar : PetscDevicePool<Type> {
private:
  using pool = PetscDevicePool<Type>;
  PetscErrorCode PetscDeviceScalar_Private(Type val, PetscMemType mtype, PetscStream pstream=PetscDefaultCUDAStream)
  {
    PetscErrorCode ierr;
    cudaError_t    cerr;
    cudaStream_t   cstream;

    PetscFunctionBegin;
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    if (PetscMemTypeHost(mtype)) {
      *this->host = val ? *val : static_cast<Type>(0);
      cerr = cudaMemcpyAsync(this->device, this->host, sizeof(Type), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
    } else {
      if (val) {
        cerr = cudaMemcpyAsync(this->device, val, sizeof(Type), cudaMemcpyDeviceToDevice, cstream);CHKERRCUDA(cerr);
        cerr = cudaMemcpyAsync(this->host, val, sizeof(Type), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
        ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
      } else {
        *this->host = static_cast<Type>(0);
      }
    }
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream, this->event);CHKERRQ(ierr);
    this->isZero = static_cast<PetscBool>(*this->host == static_cast<Type>(0));
    this->isOne  = static_cast<PetscBool>(*this->host == static_cast<Type>(1));
    PetscFunctionReturn(0);
  }

  PetscErrorCode syncDevice(PetscStream pstream=PetscDefaultCUDAStream)
  {
    PetscFunctionBegin;
    if (this->omask == PETSC_OFFLOAD_CPU) {
      PetscErrorCode ierr;
      cudaStream_t   cstream;
      cudaError_t    cerr;

      ierr = PetscStreamWaitEvent(pstream, this->event);CHKERRQ(ierr);
      ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
      cerr = cudaMemcpyAsync(this->device, this->host, sizeof(Type), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
      ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
      ierr = PetscStreamRecordEvent(pstream, this->event);CHKERRQ(ierr);
      this->omask = PETSC_OFFLOAD_BOTH;
    }
    PetscFunctionReturn(0);
  }

  PetscErrorCode syncHost(PetscStream pstream=PetscDefaultCUDAStream)
  {
    PetscFunctionBegin;
    if (this->omask == PETSC_OFFLOAD_GPU) {
      PetscErrorCode ierr;
      cudaStream_t   cstream;
      cudaError_t    cerr;

      ierr = PetscStreamWaitEvent(pstream, this->event);CHKERRQ(ierr);
      ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
      cerr = cudaMemcpyAsync(this->host, this->device, sizeof(Type), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
      ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
      this->omask = PETSC_OFFLOAD_BOTH;
      ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

public:
  Type             *host, *device;
  PetscBool        isOne, isZero;
  PetscOffloadMask omask;
  PetscEvent       event;

  PetscDeviceScalar() :
    host(&pool::hostPool[pool::poolID]),
    device(&pool::devicePool[pool::poolID])
  {
    PetscErrorCode ierr;

    ierr = PetscEventCreate(&this->event);CHKERRQ(ierr);
    ierr = PetscEventSetType(this->event, PETSC_STREAM_CUDA);CHKERRQ(ierr);
    ierr = PetscEventSetUp(this->event);CHKERRQ(ierr);
    ierr = PetscDeviceScalar_Private(0, PETSC_MEMTYPE_HOST);CHKERRQ(ierr);
  }

  ~PetscDeviceScalar()
  {
    PetscErrorCode ierr;

    ierr = PetscEventDestroy(&this->event);CHKERRQ(ierr);
  }

  PetscDeviceScalar(Type val, PetscMemType mtype, PetscStream pstream) :
    host(&pool::hostPool[pool::poolID]),
    device(&pool::devicePool[pool::poolID])
  {
    PetscErrorCode ierr;

    ierr = PetscEventCreate(&this->event);CHKERRQ(ierr);
    ierr = PetscEventSetType(this->event, PETSC_STREAM_CUDA);CHKERRQ(ierr);
    ierr = PetscEventSetUp(this->event);CHKERRQ(ierr);
    ierr = PetscDeviceScalar_Private(val, mtype, pstream);CHKERRQ(ierr);
  }

  PetscErrorCode getHost(Type **val, PetscBool update, PetscStream pstream=PetscDefaultCUDAStream)
  {
    PetscFunctionBegin;
    /* Sometimes we just want the host pointer, such as during writes */
    if (update) {
      PetscErrorCode ierr;

      ierr = this->syncHost(pstream);CHKERRQ(ierr);
    }
    *val = this->host;
    PetscFunctionReturn(0);
  }

  PetscErrorCode restoreHost(Type **val, PetscStream pstream=PetscDefaultCUDAStream)
  {
    PetscErrorCode ierr;
    cudaError_t    cerr;
    cudaStream_t   cstream;

    PetscFunctionBegin;
    /* Assumption is that a host-side write is due to performing an operation not possible on device, but that device will
       soon use result. So we immediately pipe value to device */
    ierr = PetscStreamWaitEvent(pstream, this->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(this->device, this->host, sizeof(Type), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream, this->event);CHKERRQ(ierr);
    this->omask = PETSC_OFFLOAD_BOTH;
    PetscFunctionReturn(0);
  }

  PetscErrorCode getDevice(Type **val, PetscStream pstream=PetscDefaultCUDAStream)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = (this->syncDevice)(pstream);CHKERRQ(ierr);
    *val = this->device;
    PetscFunctionReturn(0);
  }

  PetscDeviceScalar& operator=(const PetscDeviceScalar& other)
  {
    PetscErrorCode ierr;
    cudaStream_t   cstream;
    Type           *otherdevice;

    /* prepare both our and others data */
    ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
    ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, this->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(PetscDefaultCUDAStream, &cstream);CHKERRQ(ierr);
    /* We defer to others state instead of our own, but always prefer to stay on gpu */
    if (other.omask == PETSC_OFFLOAD_GPU || other.omask == PETSC_OFFLOAD_BOTH) {
      cerr = cudaMemcpyAsync(this->device, other.device, sizeof(Type), cudaMemcpyDeviceToDevice, cstream);CHKERRCUDA(cerr);
      this->omask = PETSC_OFFLOAD_GPU;
    } else {
      /* OFFLOAD_CPU */
      cerr = cudaMemcpyAsync(this->host, other.host, sizeof(Type), cudaMemcpyHostToHost, cstream);CHKERRCUDA(cerr);
      this->omask = PETSC_OFFLOAD_CPU;
    }
    ierr = PetscStreamRestoreStream(PetscDefaultCUDAStream, &cstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(PetscDefaultCUDAStream, this->event);CHKERRQ(ierr);
    return *this;
  }

  PetscDeviceScalar& operator+=(const PetscDeviceScalar& other)
  {
    PetscErrorCode ierr;

    /* We defer to others state instead of our own, but always prefer to stay on gpu */
    if (other.omask == PETSC_OFFLOAD_GPU || other.omask == PETSC_OFFLOAD_BOTH) {
      cudaStream_t cstream;

      /* prepare both our and others data */
      ierr = this->syncDevice();CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, this->event);CHKERRQ(ierr);
      ierr = PetscStreamGetStream(PetscDefaultCUDAStream, &cstream);CHKERRQ(ierr);
      kernelFunctor<<<1,1,0,cstream>>>(this->device, this->device, other.device, addFunctor());
      ierr = PetscStreamRestoreStream(PetscDefaultCUDAStream, &cstream);CHKERRQ(ierr);
      ierr = PetscStreamRecordEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      ierr = PetscStreamRecordEvent(PetscDefaultCUDAStream, this->event);CHKERRQ(ierr);
      this->omask = PETSC_OFFLOAD_GPU;
    } else {
      /* OFFLOAD_CPU */
      ierr = this->syncHost();CHKERRQ(ierr);
      *(this->host) += *(other.host);
      this->omask = PETSC_OFFLOAD_CPU;
    }
    return *this;
  }
};
#endif /* HAVE_CUDA && HAVE_CXX */
