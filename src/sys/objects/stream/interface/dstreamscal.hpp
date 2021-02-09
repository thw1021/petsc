#include <petsc/private/deviceimpl.h>

#if PetscDefined(HAVE_CUDA) && PetscDefined(HAVE_CXX)
#define PETSC_STREAM_DEVICE_POOL 100
static PetscStream PetscDefaultCUDAStream = NULL;

struct addFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return (*arg1)+arg2;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return arg1+(*arg2);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return (*arg1)+(*arg2);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return arg1+arg2;}
};

struct subFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return left2right ? (*arg1)-arg2 : arg2-(*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return left2right ? arg1-(*arg2) : (*arg2)-arg1;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return left2right ? (*arg1)-(*arg2) : (*arg2)-(*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return left2right ? arg1-arg2 : arg2-arg1;}
};

struct divFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return left2right ? (*arg1)/arg2 : arg2/(*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return left2right ? arg1/(*arg2) : (*arg2)/arg1;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return left2right ? (*arg1)/(*arg2) : (*arg2)/(*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return left2right ? arg1/arg2 : arg2/arg1;}
};

struct multFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return (*arg1)*arg2;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return arg1*(*arg2);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return (*arg1)*(*arg2);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return arg1*arg2;}
};

template <typename Type, typename functorType> __global__ __launch_bounds__(1)
void kernelFunctor(Type *res, Type *left, Type *right, functorType functor)
{
  *res = functor(left, right);
  return;
}

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
struct PetscDeviceContainer : PetscDevicePool<Type> {
private:
  PetscErrorCode PetscDeviceContainer_Private(const Type *val, PetscMemType mtype, PetscStream pstream=PetscDefaultCUDAStream)
  {
    PetscErrorCode ierr;
    cudaError_t    cerr;
    cudaStream_t   cstream;

    PetscFunctionBegin;
    ierr = PetscEventCreate(&this->event);CHKERRQ(ierr);
    ierr = PetscEventSetType(this->event, PETSC_STREAM_CUDA);CHKERRQ(ierr);
    ierr = PetscEventSetUp(this->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    if (PetscMemTypeHost(mtype)) {
      *(this->host) = val ? *val : static_cast<Type>(0);
      cerr = cudaMemcpyAsync(this->device, this->host, sizeof(Type), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
    } else {
      if (val) {
        cerr = cudaMemcpyAsync(this->device, val, sizeof(Type), cudaMemcpyDeviceToDevice, cstream);CHKERRCUDA(cerr);
        cerr = cudaMemcpyAsync(this->host, val, sizeof(Type), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
        ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
      } else {
        *(this->host) = static_cast<Type>(0);
      }
    }
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream, this->event);CHKERRQ(ierr);
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

  template <typename functorType>
  PetscErrorCode dispatchDevice(Type *const res, Type *const left, Type *const right, functorType& functor, PetscStream pstream=PetscDefaultCUDAStream)
  {
    PetscErrorCode ierr;
    cudaStream_t   cstream;

    PetscFunctionBegin;
    ierr = PetscStreamWaitEvent(pstream, this->event);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    kernelFunctor<<<1,1,0,cstream>>>(res, left, right, functor);
    ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream, this->event);CHKERRQ(ierr);
    PetccFunctionReturn(0);
  }

public:
  Type             *host, *device;
  PetscOffloadMask omask;
  PetscEvent       event;
  using pool = PetscDevicePool<Type>;

  PetscDeviceContainer() :
    host(&pool::hostPool[pool::poolID]),
    device(&pool::devicePool[pool::poolID]),
  {
    PetscErrorCode ierr;

    ierr = PetscDeviceContainer_Private(NULL, PETSC_MEMTYPE_HOST);CHKERRQ(ierr);
  }

  friend PetscDeviceContainer(const PetscDeviceContainer &other) :
    host(&pool::hostPool[pool::poolID]),
    device(&pool::devicePool[pool::poolID])
  {
    PetscErrorCode ierr;

    if (other.omask == PETSC_OFFLOAD_GPU || other.omask == PETSC_OFFLOAD_BOTH) {
      ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      ierr = PetscDeviceContainer_Private(other.device, PETSC_MEMTYPE_DEVICE, other.stream);CHKERRQ(ierr);
    } else {
      ierr = PetscDeviceContainer_Private(other.host, PETSC_MEMTYPE_HOST, other.stream);CHKERRQ(ierr);
    }
  }

  PetscDeviceContainer(const Type *val, PetscMemType mtype, PetscStream pstream) :
    host(&pool::hostPool[pool::poolID]),
    device(&pool::devicePool[pool::poolID])
  {
    PetscErrorCode ierr;

    ierr = PetscDeviceContainer_Private(val, mtype, pstream);CHKERRQ(ierr);
  }

  ~PetscDeviceContainer()
  {
    PetscErrorCode ierr;

    ierr = PetscEventDestroy(&this->event);CHKERRQ(ierr);
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

  PetscErrorCode restoreDevice()
  {
    PetscFunctionBegin;
    this->omask = PETSC_OFFLOAD_GPU;
    PetscFunctionReturn(0);
  }

  PetscDeviceContainer& operator=(const Type& other)
  {
    PetscErrorCode ierr;

    ierr = this->syncHost();CHKERRQ(ierr);
    *(this->host) = other;
  }

  PetscDeviceContainer& operator=(const PetscDeviceContainer& other)
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

  PetscDeviceContainer& operator+=(const PetscDeviceContainer& other)
  {
    PetscErrorCode ierr;

    /* We defer to others state instead of our own, but always prefer to stay on gpu */
    if (other.omask == PETSC_OFFLOAD_GPU || other.omask == PETSC_OFFLOAD_BOTH) {
      /* prepare both our and others data */
      ierr = this->syncDevice();CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      ierr = this->dispatchDevice(this->device, this->device, other.device, addFunctor());CHKERRQ(ierr);
      ierr = PetscStreamRecordEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      this->omask = PETSC_OFFLOAD_GPU;
    } else {
      /* OFFLOAD_CPU */
      ierr = this->syncHost();CHKERRQ(ierr);
      *(this->host) += *(other.host);
      this->omask = PETSC_OFFLOAD_CPU;
    }
    return *this;
  }

  PetscDeviceContainer& operator-=(const PetscDeviceContainer& other)
  {
    PetscErrorCode ierr;

    /* We defer to others state instead of our own, but always prefer to stay on gpu */
    if (other.omask == PETSC_OFFLOAD_GPU || other.omask == PETSC_OFFLOAD_BOTH) {
      /* prepare both our and others data */
      ierr = this->syncDevice();CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      ierr = this->dispatchDevice(this->device, this->device, other.device, subFunctor());CHKERRQ(ierr);
      ierr = PetscStreamRecordEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      this->omask = PETSC_OFFLOAD_GPU;
    } else {
      /* OFFLOAD_CPU */
      ierr = this->syncHost();CHKERRQ(ierr);
      *(this->host) -= *(other.host);
      this->omask = PETSC_OFFLOAD_CPU;
    }
    return *this;
  }

  PetscDeviceContainer& operator*=(const PetscDeviceContainer& other)
  {
    PetscErrorCode ierr;

    /* We defer to others state instead of our own, but always prefer to stay on gpu */
    if (other.omask == PETSC_OFFLOAD_GPU || other.omask == PETSC_OFFLOAD_BOTH) {
      /* prepare both our and others data */
      ierr = this->syncDevice();CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      ierr = this->dispatchDevice(this->device, this->device, other.device, multFunctor());CHKERRQ(ierr);
      ierr = PetscStreamRecordEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      this->omask = PETSC_OFFLOAD_GPU;
    } else {
      /* OFFLOAD_CPU */
      ierr = this->syncHost();CHKERRQ(ierr);
      *(this->host) *= *(other.host);
      this->omask = PETSC_OFFLOAD_CPU;
    }
    return *this;
  }

  PetscDeviceContainer& operator/=(const PetscDeviceContainer& other)
  {
    PetscErrorCode ierr;

    /* We defer to others state instead of our own, but always prefer to stay on gpu */
    if (other.omask == PETSC_OFFLOAD_GPU || other.omask == PETSC_OFFLOAD_BOTH) {
      /* prepare both our and others data */
      ierr = this->syncDevice();CHKERRQ(ierr);
      ierr = PetscStreamWaitEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      ierr = this->dispatchDevice(this->device, this->device, other.device, divFunctor());CHKERRQ(ierr);
      ierr = PetscStreamRecordEvent(PetscDefaultCUDAStream, other.event);CHKERRQ(ierr);
      this->omask = PETSC_OFFLOAD_GPU;
    } else {
      /* OFFLOAD_CPU */
      ierr = this->syncHost();CHKERRQ(ierr);
      *(this->host) /= *(other.host);
      this->omask = PETSC_OFFLOAD_CPU;
    }
    return *this;
  }

  friend PetscDeviceContainer operator+(const PetscDeviceContainer& left, const PetscDeviceContainer& right)
  {
    PetscDeviceContainer res = left;
    res += right;
    return res;
  }

  friend PetscDeviceContainer operator-(const PetscDeviceContainer& left, const PetscDeviceContainer& right)
  {
    PetscDeviceContainer res = left;
    res -= right;
    return res;
  }

  friend PetscDeviceContainer operator*(const PetscDeviceContainer& left, const PetscDeviceContainer& right)
  {
    PetscDeviceContainer res = left;
    res += right;
    return res;
  }

  friend PetscDeviceContainer operator/(const PetscDeviceContainer& left, const PetscDeviceContainer& right)
  {
    PetscDeviceContainer res = left;
    res /= right;
    return res;
  }
};
#endif /* HAVE_CUDA && HAVE_CXX */
