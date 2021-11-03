#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // why

#include <petsc/private/vecimpl.h>         /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h> // for Vec_Seq
#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>

#if !defined(__cplusplus) || !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#  error "VecSeqCUPM requires C++11"
#endif

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// TODO
// - refactor the AXPY's for code reuse
// - remove designated initializers
// - figure out how to call the host versions (Vec_Copy_CUDA)
// - figure out how to template which thrust namespace to use so we can do
//   thrust::<backend>::par.on(stream)
// - get rid of these undefs, they are for error checking purposes only
#undef CHKERRCUDA
#undef CHKERRCUBLAS
// - fix cublasXdot alias wrapper since sometimes it wraps to cublasXdotc
// - maybe reintroduce PetscDeviceMalloc()?
// - There is also an overloaded version of cudaMallocAsync that takes the same arguments as
//   cudaMallocFromPoolAsync

namespace Petsc
{

namespace Impl
{

template <CUPMDeviceType T>
struct VecSeq_CUPM : CUPMBlasInterface<T>
{
public:
  PETSC_INHERIT_CUPMBLAS_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t,T);

  struct Vec_CUPM
  {
    PetscScalar   *device_array; // gpu data
    PetscCopyMode ptr_ownership; // does PETSc own the array ptr?
    PetscBool     nvshmem;       // is array allocated in nvshmem? It is used to allocate
                                 // Mvctx->lvec in nvshmem
  };

private:
  // casting
  PETSC_NODISCARD static constexpr Vec_Seq*  __vec_impls_cast(Vec)  noexcept;
  PETSC_NODISCARD static constexpr Vec_CUPM* __cupm_impls_cast(Vec) noexcept;

  // retrieving the various handles
  PETSC_NODISCARD static PetscErrorCode __get_handle_dispatch(PetscDeviceContext*,cupmBlasHandle_t*,cupmStream_t*) noexcept;
  PETSC_NODISCARD static PetscErrorCode __get_handles(PetscDeviceContext*,cupmBlasHandle_t* = nullptr,cupmStream_t* = nullptr) noexcept;
  PETSC_NODISCARD static PetscErrorCode __get_handles(cupmStream_t*) noexcept;

  // the dispatcher for min and max
  template <typename Tt, typename Tu>
  PETSC_NODISCARD static PetscErrorCode __minmax_async(Tt,Tu,PetscReal,Vec,PetscInt*,PetscReal*) noexcept;

  // data movement
  PETSC_NODISCARD static PetscErrorCode __host_allocate_check(PetscDeviceContext PETSC_UNUSED dctx, Vec v) noexcept
  {
    PetscErrorCode ierr;
    auto           vseq = __vec_impls_cast(v);

    PetscFunctionBegin;
    if (!vseq) {
      ierr = PetscNewLog(PetscObjectCast(v),&vseq);CHKERRQ(ierr);
      v->data = vseq;
    }
    if (!vseq->array) {
      PetscScalar *array;
      const auto   n      = v->map->n;
      const auto   nbytes = n*sizeof(*array);

      if (nbytes > v->minimum_bytes_pinned_memory) {
        ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
        v->pinned_memory = PETSC_TRUE;
      }
      ierr = PetscMalloc1(n,&array);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(PetscObjectCast(v),nbytes);CHKERRQ(ierr);
      vseq->array           = array;
      vseq->array_allocated = array;
      if (nbytes > v->minimum_bytes_pinned_memory) {
        ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
      }
      if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) v->offloadmask = PETSC_OFFLOAD_CPU;
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __device_allocate_check(PetscDeviceContext dctx, Vec v) noexcept
  {
    PetscBLASInt   bn;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (v->spptr) PetscFunctionReturn(0);
    ierr = PetscBLASIntCast(v->map->n,&bn);CHKERRQ(ierr);
    ierr = PetscCalloc(sizeof(Vec_CUPM),&v->spptr);CHKERRQ(ierr);
    {
      cupmStream_t stream;
      cupmError_t  cerr;
      auto         vcu   = __cupm_impls_cast(v);
      auto&        array = vcu->device_array;

      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMallocAsync(static_cast<void**>(&array),bn*sizeof(*array),stream);CHKERRCUPM(cerr);
      vcu->ptr_ownership = PETSC_OWN_POINTER;
    }
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      if (v->data && __vec_impls_cast(v)->array) {
        v->offloadmask = PETSC_OFFLOAD_CPU;
      } else {
        v->offloadmask = PETSC_OFFLOAD_GPU;
      }
    }

    // Need to parse command line for minimum size to use for pinned memory allocations on
    // host here. Note: This same code duplicated in VecCreate_SeqCUDA_Private() and
    // VecCreate_MPICUDA_Private(). Is there a good way to avoid this?
    {
      static constexpr std::array<const char*,2> optionNames = {
        "VECCUDA Options",
        "VECHIP Options"
      };
      static_assert(util::integral_value(CUPMDeviceType::CUDA) == 0,"");
      static_assert(util::integral_value(CUPMDeviceType::HIP)  == 1,"");
      PetscBool flg;
      PetscInt  mem = 0;

      ierr = PetscOptionsBegin(PetscObjectComm(PetscObjectCast(v)),PetscObjectCast(v)->prefix,std::get<util::integral_value(T)>(optionNames),"Vec");CHKERRQ(ierr);
      ierr = PetscOptionsRangeInt("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",mem,&mem,&flg,0,std::numeric_limits<decltype(mem)>::max());CHKERRQ(ierr);
      if (flg) v->minimum_bytes_pinned_memory = mem;
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __copy_to_device(PetscDeviceContext dctx, Vec v) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __device_allocate_check(dctx,v);CHKERRQ(ierr);
    if (v->offloadmask == PETSC_OFFLOAD_CPU) {
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(__cupm_impls_cast(v)->device_array,__vec_impls_cast(v)->array,v->map->n*sizeof(PetscScalar),cupmMemcpyHostToDevice,stream);CHKERRCUPM(cerr);
      ierr = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __copy_to_host(PetscDeviceContext dctx, Vec v) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __host_allocate_check(dctx,v);CHKERRQ(ierr);
    if (v->offloadmask == PETSC_OFFLOAD_GPU) {
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(__vec_impls_cast(v)->array,__cupm_impls_cast(v)->device_array,v->map->n*sizeof(PetscScalar),cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
      ierr = PetscLogEventEnd(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
    PetscFunctionReturn(0);
  }

  template <bool read>
  struct vector_array
  {
    using pointer_type = util::conditional_t<read,const PetscScalar*&,PetscScalar*&>;

    const pointer_type ptr;

    operator pointer_type() const noexcept { return this->ptr; }

  protected:
    constexpr vector_array(pointer_type pointer) noexcept : ptr(pointer)  { }
  };

  // RAII VecCUPMGetArrayRead()
  struct device_array_read  : vector_array<true>
  {
    using base_type = vector_array<true>;

    constexpr device_array_read(PetscDeviceContext dctx, Vec vector) noexcept
      : base_type(__cupm_impls_cast(vector)->device_array)
    {
      CHKERRABORT(PETSC_COMM_SELF,__copy_to_device(dctx,vector));
    }
  };

  // RAII VecCUPMGetArray()
  struct device_array_write : vector_array<false>
  {
  private:
    const Vec _v;

  public:
    using base_type = vector_array<false>;

    constexpr device_array_write(PetscDeviceContext dctx, Vec vector) noexcept
      : base_type(__cupm_impls_cast(vector)->device_array), _v(vector)
    {
      CHKERRABORT(PETSC_COMM_SELF,__device_allocate_check(dctx,vector));
    }

    ~device_array_write()
    {
      // 1. can't actually do anything about the error since we may already be seterrq-ing out
      // 2. no clue what happens if two different error codes are in flight simultaneously
      auto PETSC_UNUSED ierr = PetscObjectStateIncrease(PetscObjectCast(_v));
      _v->offloadmask = PETSC_OFFLOAD_GPU;
    }
  };

  struct host_array_read    : vector_array<true>
  {
    using base_type = vector_array<true>;

    constexpr host_array_read(PetscDeviceContext dctx, Vec vector) noexcept
    : base_type(__vec_impls_cast(vector)->array)
    {
      CHKERRABORT(PETSC_COMM_SELF,__copy_to_host(dctx,vector));
    }
  };

  struct host_array_write   : vector_array<false>
  {
  private:
    const Vec _v;

  public:
    using base_type = vector_array<false>;

    constexpr host_array_write(PetscDeviceContext dctx, Vec vector) noexcept
      : base_type(__vec_impls_cast(vector)->array), _v(vector)
    {
      CHKERRABORT(PETSC_COMM_SELF,__host_allocate_check(dctx,vector));
    }

    ~device_array_write()
    {
      // 1. can't actually do anything about the error since we may already be seterrq-ing out
      // 2. no clue what happens if two different error codes are in flight simultaneously
      auto PETSC_UNUSED ierr = PetscObjectStateIncrease(PetscObjectCast(_v));
      _v->offloadmask = PETSC_OFFLOAD_CPU;
    }
  };

public:
  PETSC_NODISCARD static PetscErrorCode aypx_async(Vec,PetscScalar,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode axpy_async(Vec,PetscScalar,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode pointwisedivide_async(Vec,Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode pointwisemult_async(Vec,Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode reciprocal_async(Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode waxpy_async(Vec,PetscScalar,Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode maxpy_async(Vec,PetscInt,const PetscScalar*,Vec*) noexcept;
  PETSC_NODISCARD static PetscErrorCode dot_async(Vec,Vec,PetscScalar*) noexcept;
  PETSC_NODISCARD static PetscErrorCode set_async(Vec,PetscScalar) noexcept;
  PETSC_NODISCARD static PetscErrorCode scale_async(Vec,PetscScalar) noexcept;
  PETSC_NODISCARD static PetscErrorCode tdot_async(Vec,Vec,PetscScalar*) noexcept;
  PETSC_NODISCARD static PetscErrorCode copy_async(Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode swap_async(Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode axpby_async(Vec,PetscScalar,PetscScalar,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode axpbypcz_async(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode norm_async(Vec,NormType,PetscReal*) noexcept;
  PETSC_NODISCARD static PetscErrorCode dotnorm2_async(Vec,Vec,PetscScalar*,PetscScalar*) noexcept;
  PETSC_NODISCARD static PetscErrorCode destroy_async(Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode conjugate_async(Vec) noexcept;
  template <bool read>
  PETSC_NODISCARD static PetscErrorCode getlocalvector_async(Vec,Vec) noexcept;
  template <bool read>
  PETSC_NODISCARD static PetscErrorCode restorelocalvector_async(Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode max_async(Vec,PetscInt*,PetscReal*) noexcept;
  PETSC_NODISCARD static PetscErrorCode min_async(Vec,PetscInt*,PetscReal*) noexcept;
  PETSC_NODISCARD static PetscErrorCode sum_async(Vec,PetscScalar*) noexcept;
  PETSC_NODISCARD static PetscErrorCode shift_async(Vec,PetscScalar) noexcept;
  PETSC_NODISCARD static PetscErrorCode setrandom_async(Vec,PetscRandom) noexcept;

  const struct _VecOps ops = {
    .axpy                   = axpy_async,
    .aypx                   = aypx_async,
    .pointwisedivide        = pointwisedivide_async,
    .pointwisemult          = pointwisemult_async,
    .reciprocal             = reciprocal_async,
    .waxpy                  = waxpy_async,
    .maxpy                  = maxpy_async,
    .dot                    = dot_async,
    .set                    = set_async,
    .scale                  = scale_async,
    .tdot                   = tdot_async,
    .copy                   = copy_async,
    .swap                   = swap_async,
    .axpby                  = axpby_async,
    .axpbypcz               = axpbypcz_async,
    .norm                   = norm_async,
    .dotnorm2               = dotnorm2_async,
    .destroy                = destroy_async,
    .conjugate              = conjugate_async,
    .getlocalvector         = getlocalvector_async<false>,
    .getlocalvectorread     = getlocalvector_async<true>,
    .restorelocalvector     = restorelocalvector_async<false>,
    .restorelocalvectorread = restorelocalvector_async<true>,
    .max                    = max_async,
    .min                    = min_async,
    .sum                    = sum_async,
    .shift                  = shift_async,
    .setrandom              = setrandom_async
  };
};

template <CUPMDeviceType T>
inline constexpr Vec_Seq* VecSeq_CUPM<T>::__vec_impls_cast(Vec v) noexcept
{
  return static_cast<Vec_Seq*>(v->data);
}

template <CUPMDeviceType T>
inline constexpr typename VecSeq_CUPM<T>::Vec_CUPM* VecSeq_CUPM<T>::__cupm_impls_cast(Vec v) noexcept
{
  return static_cast<Vec_CUPM*>(v->spptr);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::__get_handle_dispatch(PetscDeviceContext *ctx, cupmBlasHandle_t *handle, cupmStream_t *stream) noexcept
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,cupmDeviceTypeToPetsc());CHKERRQ(ierr);
  if (handle) {ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,handle);CHKERRQ(ierr);}
  if (stream) {ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,stream);CHKERRQ(ierr);}
  if (ctx) *ctx = dctx;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::__get_handles(PetscDeviceContext *dctx, cupmBlasHandle_t *handle, cupmStream_t *stream) noexcept
{
  return __get_handle_dispatch(dctx,handle,stream);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::__get_handles(cupmStream_t *handle) noexcept
{
  return __get_handle_dispatch(nullptr,nullptr,handle);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin) noexcept
{
  const PetscInt     n = yin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    cupmError_t  cerr;
    cupmStream_t stream;

    ierr = __get_handles(&dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(device_array_write(dctx,yin).ptr,device_array_read(dctx,xin).ptr,n*sizeof(PetscScalar),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  } else {
    const auto       alphaIsOne = alpha == PetscScalar(1.0);
    cupmBlasError_t  cberr;
    cupmBlasHandle_t cupmBlasHandle;
    PetscBLASInt     bn;

    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      auto yarray = device_array_write(dctx,yin);
      auto xarray = device_array_read(dctx,xin);

      if (alphaIsOne) {
        cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {
        constexpr PetscScalar sone = 1.0;

        cberr = cupmBlasXscal(cupmBlasHandle,bn,&alpha,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&sone,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops((alphaIsOne ? 1 : 2)*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscScalar alpha, Vec xin) noexcept
{
  PetscErrorCode ierr;
  PetscBool      xiscuda;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) PetscFunctionReturn(0);
  ierr = PetscObjectTypeCompareAny(PetscObjectCast(xin),&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (xiscuda) {
    const auto         n = yin->map->n;
    PetscBLASInt       bn;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin),1,device_array_write(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseDivide_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    const auto         n = xin->map->n;
    PetscDeviceContext dctx;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      auto xptr = thrust::device_pointer_cast(device_array_read(dctx,xin).ptr);
      auto yptr = thrust::device_pointer_cast(device_array_read(dctx,yin).ptr);
      auto wptr = thrust::device_pointer_cast(device_array_write(dctx,win).ptr);

      thrust::transform(xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
    } catch (const std::exception& ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::pointwisemult_async(Vec win, Vec xin, Vec yin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseMult_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    const auto         n = xin->map->n;
    PetscDeviceContext dctx;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      auto xptr = thrust::device_pointer_cast(device_array_read(dctx,xin).ptr);
      auto yptr = thrust::device_pointer_cast(device_array_read(dctx,yin).ptr);
      auto wptr = thrust::device_pointer_cast(device_array_write(dctx,win).ptr);

      thrust::transform(xptr,xptr+n,yptr,wptr,thrust::multiplies<PetscScalar>());
    } catch (const std::exception& ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::reciprocal_async(Vec xin) noexcept
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    struct reciprocal
    {
      __host__ __device__ __forceinline__
      constexpr PetscScalar operator()(const PetscScalar& s) const
      {
        return s ? PetscScalar(1.0)/s : 0;
      }
    };

    auto xptr = thrust::device_pointer_cast(device_array_write(dctx,xin).ptr);

    thrust::transform(xptr,xptr+n,xptr,reciprocal());
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscScalar alpha, Vec xin, Vec yin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = copy_async(yin,win);CHKERRQ(ierr);
  } else {
    const auto         n = win->map->n;
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmStream_t       stream;

    ierr = __get_handles(&dctx,&cupmBlasHandle,&stream);CHKERRQ(ierr);
    {
      auto            warray = device_array_write(dctx,win);
      cupmBlasError_t cberr;
      cupmError_t     cerr;
      PetscBLASInt    bn;

      ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(warray.ptr,device_array_read(dctx,yin).ptr,n*sizeof(*warray.ptr),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
      cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin),1,warray,1);CHKERRCUPMBLAS(cberr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    }
    ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::maxpy_async(Vec xin, PetscInt nv, const PetscScalar *alpha, Vec *y) noexcept
{
  const auto          n = xin->map->n;
  PetscBLASInt        bn;
  PetscDeviceContext  dctx;
  cupmBlasHandle_t    cupmBlasHandle;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    cupmBlasError_t cberr;
    auto            xarray = device_array_write(dctx,xin);

    for (PetscInt j = 0; j < nv; ++j) {
      cberr = cupmBlasXaxpy(cupmBlasHandle,bn,alpha+j,device_array_read(dctx,y[j]),1,xarray,1);CHKERRCUPMBLAS(cberr);
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(nv*2*n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(nv*sizeof(*alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscScalar *z) noexcept
{
  const auto         n = xin->map->n;
  PetscBLASInt       bn;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  // arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the
  // second
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdot(cupmBlasHandle,bn,device_array_read(dctx,yin),1,device_array_read(dctx,xin),1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (n > 0) {ierr = PetscLogGpuFlops(2*(n-1));CHKERRQ(ierr);}
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscScalar alpha) noexcept
{
  const auto         n = xin->map->n;
  PetscErrorCode     ierr;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto xptr = thrust::device_pointer_cast(device_array_write(dctx,xin).ptr);

    thrust::fill(xptr,xptr+n,alpha);
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::scale_async(Vec xin, PetscScalar alpha) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = set_async(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != PetscScalar(1.0)) {
    const auto         n = xin->map->n;
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscBLASInt       bn;

    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXscal(cupmBlasHandle,bn,&alpha,device_array_write(dctx,xin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::tdot_async(Vec xin, Vec yin, PetscScalar *z) noexcept
{
  const auto         n = xin->map->n;
  PetscBLASInt       bn;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdotu(cupmBlasHandle,bn,device_array_read(dctx,xin),1,device_array_read(dctx,yin),1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (n > 0) {ierr = PetscLogGpuFlops(2*n-1);CHKERRQ(ierr);}
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin) noexcept
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = xin->map->n;
    auto               yiscuda = PETSC_TRUE,xondevice = PETSC_TRUE; // assume we start on device
    cupmMemcpyKind_t   mode;
    PetscDeviceContext dctx;
    cupmStream_t       stream;
    cupmError_t        cerr;
    PetscErrorCode     ierr;

    switch (xin->offloadmask) {
    case PETSC_OFFLOAD_KOKKOS:      // technically an error
    case PETSC_OFFLOAD_UNALLOCATED: // technically an error
    case PETSC_OFFLOAD_CPU:
      xondevice = PETSC_FALSE; // we assumed partially wrong
    case PETSC_OFFLOAD_GPU:
    case PETSC_OFFLOAD_BOTH:
      break;
      // no default case so warnings are thrown for new offloadmasks
    }

    switch (yin->offloadmask) {
    case PETSC_OFFLOAD_KOKKOS:
    case PETSC_OFFLOAD_UNALLOCATED:
    case PETSC_OFFLOAD_CPU:
      ierr = PetscObjectTypeCompareAny(PetscObjectCast(yin),&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
    case PETSC_OFFLOAD_GPU:
    case PETSC_OFFLOAD_BOTH:
      if (yiscuda) { // PETSC_TRUE by default (unless on the host)
        // even though y may be on the host, its a cuda vector, so it ought to be on the device
        mode = xondevice ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice;
      } else {
        // we assumed really wrong
        mode = xondevice ? cupmMemcpyDeviceToHost : cupmMemcpyHostToHost;
      }
      break;
    }

    ierr = __get_handles(&dctx,&stream);CHKERRQ(ierr);
    switch (mode) {
    case cupmMemcpyDeviceToDevice:
      // the best case
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(device_array_write(dctx,yin).ptr,device_array_read(dctx,xin).ptr,n*sizeof(PetscScalar),mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyHostToDevice:
      // not terrible
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(device_array_write(dctx,yin).ptr,host_array_read(dctx,xin).ptr,n*sizeof(PetscScalar),mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyDeviceToHost: {
      // not great
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(yarray,device_array_read(dctx,xin).ptr,n*sizeof(*yarray),mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    case cupmMemcpyHostToHost:   {
      // the worst case
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarrat);CHKERRQ(ierr);
      ierr = PetscArraycpy(ya,host_array_read(dctx,xin),n);CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"Unknown cupmMemcpyKind %d",mode);
    }
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::swap_async(Vec xin, Vec yin) noexcept
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = xin->map->n;
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscBLASInt       bn;
    PetscErrorCode     ierr;

    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXswap(cupmBlasHandle,bn,device_array_write(dctx,xin),1,device_array_write(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::axpby_async(Vec yin, PetscScalar alpha, PetscScalar beta, Vec xin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = scale_async(yin,beta);CHKERRQ(ierr);
  } else if (beta == PetscScalar(1.0)) {
    ierr = axpy_async(yin,alpha,xin);CHKERRQ(ierr);
  } else if (alpha == PetscScalar(1.0)) {
    ierr = aypx_async(yin,beta,xin);CHKERRQ(ierr);
  } else {
    const PetscInt     n          = yin->map->n;
    const bool         betaIsZero = beta == PetscScalar(0.0);
    PetscBLASInt       bn;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (betaIsZero) {
      cupmStream_t stream;
      auto         yarray = device_array_write(dctx,yin);

      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(yarray,device_array_read(dctx,xin).ptr,n*sizeof(*yarray.ptr),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
      cberr = cupmBlasXscal(cupmBlasHandle,bn,&alpha,yarray,1);CHKERRCUPMBLAS(cberr);
    } else {
      cberr = cupmBlasXscal(cupmBlasHandle,bn,&beta,yarray,1);CHKERRCUPMBLAS(cberr);
      cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin),1,yarray,1);CHKERRCUPMBLAS(cberr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops((betaIsZero ? 1 : 3)*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar((betaIsZero ? 1 : 2)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::axpbypcz_async(Vec zin, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec xin, Vec yin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (gamma != PetscScalar(1.0)) {
    // z <- a*x + b*y + c*z
    ierr = scale_async(zin,gamma);CHKERRQ(ierr);
  }
  ierr = axpy_async(zin,alpha,xin);CHKERRQ(ierr);
  ierr = axpy_async(zin,beta,yin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::norm_async(Vec xin, NormType type, PetscReal *z) noexcept
{
  const auto         n         = xin->map->n;
  PetscInt           flopCount = 0;
  PetscBLASInt       bn;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!n) {
    z[0] = 0.0;
    // yes this technically sets z[0] = 0 again half the time
    z[type == NORM_1_AND_2] = 0.0;
    PetscFunctionReturn(0);
  }
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    cupmBlasError_t cberr;
    auto            xarray = device_array_read(dctx,xin);

    switch (type) {
    case NORM_1_AND_2:
    case NORM_1:
      cberr = cupmBlasXasum(cupmBlasHandle,bn,xarray,1,z);CHKERRCUPMBLAS(cberr);
      flopCount = PetscMax(n-1,0);
      if (type == NORM_1_AND_2) ++z;
      else break;
    case NORM_2:
    case NORM_FROBENIUS:
      cberr = cupmBlasXnrm2(cupmBlasHandle,bn,xarray,1,z);CHKERRCUPMBLAS(cberr);
      flopCount += PetscMax(2*n-1,0); // +=  in case we've fallen through from NORM_1_AND_2
      break;
    case NORM_INFINITY: {
      cupmError_t  cerr;
      cupmStream_t stream;
      PetscScalar  zs;
      int          i;

      // this needs to be redone by hand
      cberr = cupmBlasIXamax(cupmBlasHandle,bn,xarray,1,&i);CHKERRCUPMBLAS(cberr);
      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(&zs,xarray.ptr+i-1,sizeof(zs),cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
      *z   = PetscAbsScalar(zs);
      // flopCount = ???
    } break;
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flopCount);CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::dotnorm2_async(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = dot_async(s,t,dp);CHKERRQ(ierr);
  ierr = dot_async(t,t,nm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v) noexcept
{
  PetscErrorCode ierr;
  auto           vcu  = __cupm_impls_cast(v);
  auto           vseq = __vec_impls_cast(v);

  PetscFunctionBegin;
  if (vcu) {
    switch (vcu->ptr_ownership) {
    case PETSC_COPY_VALUES:
    case PETSC_OWN_POINTER:
      if (PetscDefined(HAVE_NVSHMEM) && vcu->nvshmem) {
        ierr = PetscNvshmemFree(vcu->device_array);CHKERRQ(ierr);
      } else {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = __get_handles(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      }
    case PETSC_USE_POINTER:
      break;
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);CHKERRQ(ierr);
#endif
  if (vseq) {
    if (vseq->array_allocated) {
      if (v->pinned_memory) {ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);}
      ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
      if (v->pinned_memory) {
        ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
        v->pinned_memory = PETSC_FALSE;
      }
    }
    ierr = PetscFree(vs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::conjugate_async(Vec v) noexcept
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    const auto         n = xin->map->n;
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      struct conjugate
      {
        __host__ __device__ __forceinline__
        constexpr PetscScalar operator()(const PetscScalar& x) const { return PetscConj(x); }
      };
      auto xptr = thrust::device_pointer_cast(device_array_write(dctx,v).ptr);

      thrust::transform(xptr,xptr+n,xptr,conjugate());
    } catch (const std::exception& ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <bool read>
inline PetscErrorCode VecSeq_CUPM<T>::getlocalvector_async(Vec v, Vec w) noexcept
{
  PetscErrorCode ierr;
  PetscBool      wisseqcuda;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUDA,&wisseqcuda);CHKERRQ(ierr);
  if (w->data && wisseqcuda) {
    const auto& vseq = __vec_impls_cast(w);

    if (vseq->array_allocated) {
      if (w->pinned_memory) {
        ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
      }
      ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
      if (w->pinned_memory) {
        ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
        w->pinned_memory = PETSC_FALSE;
      }
    }
    vseq->array         = nullptr;
    vseq->unplacedarray = nullptr;
  }

  if (w->spptr && wisseqcuda) {
    const auto &vcu = __cupm_impls_cast(w);

    if (vcu->device_array) {
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = __get_handles(&stream);CHKERRQ(ierr);
      cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      vcu->device_array = nullptr;
    }
    ierr = PetscFree(w->spptr);CHKERRQ(ierr);
  }

  if (v->petscnative && wisseqcuda) {
    ierr = PetscFree(w->data);CHKERRQ(ierr);
    w->data          = v->data;
    w->offloadmask   = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr         = v->spptr;
    ierr = PetscObjectStateIncrease(PetscObjectCast(w));CHKERRQ(ierr);
  } else {
    if (read) {
      auto& array = __vec_impls_cast(w)->array;
      ierr = VecGetArrayRead(v,const_cast<const decltype(&array)>(&array));CHKERRQ(ierr);
    } else {
      ierr = VecGetArray(v,&__vec_impls_cast(w)->array);CHKERRQ(ierr);
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcuda) {ierr = VecCUDAAllocateCheck(w);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <bool read>
inline PetscErrorCode VecSeq_CUPM<T>::restorelocalvector_async(Vec v, Vec w) noexcept
{
  PetscErrorCode ierr;
  PetscBool      wisseqcuda;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUDA,&wisseqcuda);CHKERRQ(ierr);
  if (v->petscnative && wisseqcuda) {
    v->data          = w->data;
    v->offloadmask   = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr         = w->spptr;
    w->data          = nullptr;
    w->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
    w->spptr         = nullptr;
  } else {
    if (read) {
      auto& array = __vec_impls_cast(w)->array;
      ierr = VecRestoreArrayRead(v,const_cast<const decltype(&array)>(&array));CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(v,&__vec_impls_cast(w)->array);CHKERRQ(ierr);
    }

    if (w->spptr && wisseqcuda) {
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = __get_handles(&stream);CHKERRQ(ierr);
      cerr = cupmFreeAsync(__cupm_impls_cast(w)->device_array,stream);CHKERRCUPM(cerr);
      ierr = PetscFree(w->spptr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <typename Tt, typename Tu>
inline PetscErrorCode VecSeq_CUPM<T>::__minmax_async(Tt&& tuple_functor, Tu&& unary_functor, PetscReal initialValue, Vec v, PetscInt *p, PetscReal *m) noexcept
{
  PetscErrorCode     ierr;
  const auto         n = v->map->n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  if (!n) {
    *m = initialValue;
    if (p) *p = -1;
    PetscFunctionReturn(0);
  }
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto vptr = thrust::device_pointer_cast(device_array_read(dctx,v).ptr);

    if (p) {
      constexpr auto make_init = [=](){ return thrust::make_tuple(initialValue,PetscInt(-1)) };
      auto           zibit     = thrust::make_zip_iterator(
        thrust::make_tuple(vptr,thrust::make_counting_iterator(PetscInt(0)))
      );

      if (PetscDefined(USE_COMPLEX)) {
        struct real_part
        {
          __host__ __device__ __forceinline__
          thrust::tuple<PetscReal,PetscInt>
          operator()(const thrust::tuple<PetscScalar,PetscInt>& x) const
          {
            return thrust::make_tuple(PetscRealPart(x.get<0>()),x.get<1>());
          }
        };

        thrust::tie(*m,*p) = thrust::transform_reduce(
          zibit,zibit+n,real_part(),make_init(),std::forward<Tt>(tuple_functor)
        );
      } else {
        thrust::tie(*m,*p) = thrust::reduce(
          zibit,zibit+n,make_init(),std::forward<Tt>(tuple_functor)
        );
      }
    } else {
      if (PetscDefined(USE_COMPLEX)) {
        struct real_part
        {
          __host__ __device__ __forceinline__
          constexpr PetscReal operator()(const PetscScalar& x) const { return PetscRealPart(x); }
        };

        *m = thrust::transform_reduce(
          vptr,vptr+n,real_part(),initialValue,std::forward<Tu>(unary_functor)
        );
      } else {
        *m = thrust::reduce(vptr,vptr+n,initialValue,std::forward<Tu>(unary_functor));
      }
    }
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::max_async(Vec v, PetscInt *p, PetscReal *m) noexcept
{
  PetscErrorCode ierr;
  struct max_tuple
  {
    using tuple_type = thrust::tuple<PetscReal,PetscInt>;

    __host__ __device__
    tuple_type operator()(const tuple_type& x, const tuple_type& y) const
    {
      if (x.get<0>() < y.get<0>()) {
        return thrust::make_tuple(y.get<0>(),y.get<1>());
      } else if ((x.get<0>() != y.get<0>()) || (x.get<1>() <  y.get<1>())) {
        return thrust::make_tuple(x.get<0>(),x.get<1>());
      } else {
        return thrust::make_tuple(y.get<0>(),y.get<1>());
      }
    }
  };

  PetscFunctionBegin;
  ierr = __minmax_async(PETSC_MIN_REAL,max_tuple(),thrust::maximum<decltype(*m)>(),v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::min_async(Vec v, PetscInt *p, PetscReal *m) noexcept
{
  PetscErrorCode ierr;
  struct min_tuple
  {
    using tuple_type = thrust::tuple<PetscReal, PetscInt>;

    __host__ __device__ __forceinline__
    tuple_type operator()(const tuple_type& x, const tuple_type& y) const
    {
      if (x.get<0>() > y.get<0>()) {
        return thrust::make_tuple(y.get<0>(),y.get<1>());
      } else if ((x.get<0>() != y.get<0>()) || (x.get<1>() < y.get<1>())) {
        return thrust::make_tuple(x.get<0>(),x.get<1>());
      } else {
        return thrust::make_tuple(y.get<0>(),y.get<1>());
      }
    }
  };

  PetscFunctionBegin;
  ierr = __minmax_async(PETSC_MAX_REAL,min_tuple(),thrust::minimum<decltype(*m)>(),v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::sum_async(Vec v, PetscScalar *sum) noexcept
{
  PetscErrorCode     ierr;
  const auto         n = v->map->n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto dptr = thrust::device_pointer_cast(device_array_read(dctx,v).ptr);

    *sum = thrust::reduce(dptr,dptr+n,PetscScalar(0.0));
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::shift_async(Vec v, PetscScalar shift) noexcept
{
  PetscErrorCode     ierr;
  const auto         n = v->map->n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    struct shift
    {
      const PetscScalar _shift;

      constexpr shift(PetscScalar shift) noexcept : _shift(shift) { }

      __host__ __device__ __forceinline__
      constexpr PetscScalar operator()(PetscScalar x) { return x+_shift; }
    };

    auto dptr = thrust::device_pointer_cast(device_array_write(dctx,v).ptr);

    thrust::transform(dptr,dptr+n,dptr,shift(shift)); /* in-place transform */
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::setrandom_async(Vec v, PetscRandom rand) noexcept
{
  PetscErrorCode ierr;
  const auto     n = xin->map->n;
  PetscBool      iscurand;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)r,PETSCCURAND,&iscurand);CHKERRQ(ierr);
  if (iscurand) {
    ierr = PetscRandomGetValues(r,n,device_array_write(dctx,v));CHKERRQ(ierr);
  } else {
    ierr = PetscRandomGetValues(r,n,host_array_write(dctx,v));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
