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

namespace Petsc
{

namespace Impl
{

namespace detail
{

template <CUPMDeviceType T>
PETSC_STATIC_INLINE constexpr const char* CUPMVecSuperTypeName() noexcept;

template <CUPMDeviceType T>
PETSC_STATIC_INLINE constexpr const char* CUPMVecSubTypeName()   noexcept;

template <CUPMDeviceType T>
PETSC_STATIC_INLINE constexpr PetscDeviceType CUPMToDeviceType() noexcept;

#define CAT_(x,y) x ## y
#define CAT(x,y)  CAT_(x,y)

#define PETSC_VEC_CUPM_DEFINE_SUPERTYPE_GETTERS(PREFIX)                 \
  template <>                                                           \
  inline constexpr const char*                                          \
  CUPMVecSuperTypeName<CUPMDeviceType::PREFIX>() noexcept               \
  {                                                                     \
    return CAT_(VECSEQ,PREFIX);                                         \
  }

#define PETSC_VEC_CUPM_DEFINE_SUBTYPE_GETTERS(PREFIX)                   \
  template <>                                                           \
  inline constexpr const char*                                          \
  CUPMVecSubTypeName<CUPMDeviceType::PREFIX>() noexcept                 \
  {                                                                     \
    return CAT(VEC,PREFIX);                                             \
  }

#define PETSC_VEC_CUPM_DEFINE_UTILITY(PREFIX)           \
  PETSC_VEC_CUPM_DEFINE_SUPERTYPE_GETTERS(PREFIX);      \
  PETSC_VEC_CUPM_DEFINE_SUBTYPE_GETTERS(PREFIX)


PETSC_VEC_CUPM_DEFINE_UTILITY(CUDA);
PETSC_VEC_CUPM_DEFINE_UTILITY(HIP);

} // namespace detail

template <CUPMDeviceType T>
class VecSeq_CUPM : CUPMBlasInterface<T>
{
public:
  PETSC_INHERIT_CUPMBLAS_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t,T);

  struct Vec_CUPM
  {
    PetscScalar  *GPUarray;           /* this always holds the GPU data */
    PetscScalar  *GPUarray_allocated; /* if the array was allocated by PETSc this is its pointer */
    cupmStream_t stream;              /* A stream for doing asynchronous data transfers */
    PetscBool    nvshmem;             /* Is GPUarray_allocated allocated in nvshmem? It is used to allocate Mvctx->lvec in nvshmem */
  };

private:
  PETSC_NODISCARD static constexpr Vec_Seq* __vec_impls_cast(Vec v) noexcept
  {
    return static_cast<Vec_Seq*>(v->data);
  }

  PETSC_NODISCARD static constexpr Vec_CUPM* __cupm_impls_cast(Vec v) noexcept
  {
    return static_cast<Vec_CUPM*>(v->spptr);
  }

  PETSC_NODISCARD static PetscErrorCode __move_device_array(PetscDeviceContext dctx, Vec v) noexcept
  {
    PetscFunctionBegin;
    if (v->offloadmask == PETSC_OFFLOAD_CPU) {
      PetscErrorCode ierr;

      ierr = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
      ierr = PetscDeviceArraycpy(dctx,__cupm_impls_cast(v)->GPUArray,__vec_impls_cast(v)->array,v->map->n,PETSC_DEVICE_MEMCPY_HTOD);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
    PetscFunctionReturn(0);
  }

  template <bool read>
  struct device_array
  {
    using pointer_type = util::conditional_t<read,const PetscScalar *const,PetscScalar *const>;
    // const PetscScalar *const -> const PetscScalar *
    using cast_type    = util::remove_const_t<pointer_type>;

    pointer_type ptr;

    operator cast_type() const { return this->ptr; }

  protected:
    constexpr device_array(pointer_type pointer) noexcept : ptr(pointer) { }
  };

  struct device_array_read  : device_array<true>
  {
    constexpr device_array_read(PetscDeviceContext dctx, Vec vector) noexcept
      : device_array<true>(__cupm_impls_cast(vector)->GPUarray)
    { auto ierr = __move_device_array(dctx,vector);CHKERRABORT(PETSC_COMM_SELF,ierr); }
  };

  struct device_array_write : device_array<false>
  {
  private:
    const Vec _v;

  public:
    constexpr device_array_write(const PetscDeviceContext PETSC_UNUSED &dctx, Vec vector) noexcept
      : device_array<false>(__cupm_impls_cast(vector)->GPUarray), _v(vector)
    { }

    ~device_array_write()
    {
      // 1. can't actually do anything about the error since we may already be seterrq-ing out
      // 2. no clue what happens if two different error codes are in flight simultaneously
      (void)PetscObjectStateIncrease(reinterpret_cast<PetscObject>(_v));
      _v->offloadmask = PETSC_OFFLOAD_GPU;
    }
  };

  PETSC_NODISCARD static PetscErrorCode __get_handles(PetscDeviceContext *dctx) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(dctx,cupmDeviceTypeToPetsc());CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __get_handles(PetscDeviceContext *dctx, cupmBlasHandle_t *handle) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(dctx,cupmDeviceTypeToPetsc());CHKERRQ(ierr);
    ierr = PetscDeviceContextGetBLASHandle_Internal(*dctx,handle);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

public:
  const struct _VecOps ops = {
    .axpy            = axpy_async,
    .aypx            = aypx_async,
    .pointwisedivide = pointwisedivide_async,
    .pointwisemult   = pointwisemult_async,
    .reciprocal      = reciprocal_async,
    .waxpy           = waxpy_async,
    .maxpy           = maxpy_async,
    .dot             = dot_async,
    .set             = set_async,
    .scale           = scale_async,
    .tdot            = tdot_async,
    .copy            = copy_async,
    .swap            = swap_async,
    .axpby           = axpby_async,
    .axpbypcz        = axpbypcz_async,
  };

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
};

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin) noexcept
{
  const PetscInt     n = yin->map->n;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  if (alpha == PetscScalar(0.0)) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = PetscDeviceArraycpy(dctx,device_array_write(dctx,yin).ptr,device_array_read(dctx,xin).ptr,n,PETSC_DEVICE_MEMCPY_DTOD);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  } else {
    const bool      alphaIsOne = alpha == PetscScalar(1.0);
    cupmBlasError_t cberr;
    PetscBLASInt    bn;

    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscScalar alpha, Vec xin) noexcept
{
  PetscErrorCode ierr;
  PetscBool      xiscuda;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) PetscFunctionReturn(0);
  ierr = PetscObjectTypeCompareAny((PetscObject)xin,&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (xiscuda) {
    const PetscInt     n = yin->map->n;
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseDivide_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    const PetscInt     n = xin->map->n;
    PetscDeviceContext dctx;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      auto xptr = thrust::device_pointer_cast(device_array_read(dctx,xin).ptr);
      auto yptr = thrust::device_pointer_cast(device_array_read(dctx,yin).ptr);
      auto wptr = thrust::device_pointer_cast(device_array_write(dctx,win).ptr);

      thrust::transform(xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
    } catch (const std::exception &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::pointwisemult_async(Vec win, Vec xin, Vec yin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseMult_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    const PetscInt     n = xin->map->n;
    PetscDeviceContext dctx;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      auto xptr = thrust::device_pointer_cast(device_array_read(dctx,xin).ptr);
      auto yptr = thrust::device_pointer_cast(device_array_read(dctx,yin).ptr);
      auto wptr = thrust::device_pointer_cast(device_array_write(dctx,win).ptr);

      thrust::transform(xptr,xptr+n,yptr,wptr,thrust::multiplies<PetscScalar>());
    } catch (const std::exception &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

namespace detail
{

struct reciprocal
{
  __host__ __device__ __forceinline__ PetscScalar operator()(const PetscScalar& s)
  {
    return (s == PetscScalar(0.0)) ? 0 : PetscScalar(1.0)/s;
  }
};

} // namespace detail

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::reciprocal_async(Vec xin) noexcept
{
  const PetscInt     n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto xptr = thrust::device_pointer_cast(device_array_write(dctx,xin).ptr);

    thrust::transform(xptr,xptr+n,xptr,detail::reciprocal());
  } catch (const std::exception &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscScalar alpha, Vec xin, Vec yin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = copy_async(yin,win);CHKERRQ(ierr);
  } else {
    const PetscInt     n = win->map->n;
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;

    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    {
      auto            warray = device_array_write(dctx,win);
      cupmBlasError_t cberr;
      PetscBLASInt    bn;

      ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      ierr = PetscDeviceArraycpy(dctx,warray.ptr,device_array_read(dctx,yin).ptr,n,PETSC_DEVICE_MEMCPY_DTOD);CHKERRQ(ierr);
      cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin),1,warray,1);CHKERRCUPMBLAS(cberr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    }
    ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::maxpy_async(Vec xin, PetscInt nv, const PetscScalar *alpha, Vec *y) noexcept
{
  const PetscInt      n = xin->map->n;
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscScalar *z) noexcept
{
  const PetscInt     n = xin->map->n;
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscScalar alpha) noexcept
{
  const PetscInt     n = xin->map->n;
  PetscErrorCode     ierr;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto xptr = thrust::device_pointer_cast(device_array_write(dctx,xin).ptr);

    thrust::fill(xptr,xptr+n,alpha);
  } catch (const std::exception &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::scale_async(Vec xin, PetscScalar alpha) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = set_async(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != PetscScalar(1.0)) {
    const PetscInt     n = xin->map->n;
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::tdot_async(Vec xin, Vec yin, PetscScalar *z) noexcept
{
  const PetscInt     n = xin->map->n;
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin) noexcept
{
  PetscFunctionBegin;
  if (xin != yin) {
    const PetscInt     n = yin->map->n;
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    switch (xin->offloadmask) {
    case PETSC_OFFLOAD_GPU:
      PetscBool yiscuda;

      // why do we do this check here but not below?
      ierr = PetscObjectTypeCompareAny((PetscObject)yin,&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
      if (yiscuda) goto device_to_device_copy;
      else         goto device_to_host_copy;
    case PETSC_OFFLOAD_BOTH:
      // if xin is valid in both places, see where yin is and copy there (because it's probably
      // where we'll want to next use it)
      switch (yin->offloadmask) {
      case PETSC_OFFLOAD_GPU:
      case PETSC_OFFLOAD_BOTH:
        goto device_to_device_copy;
      case PETSC_OFFLOAD_UNALLOCATED:
      case PETSC_OFFLOAD_KOKKOS:
      case PETSC_OFFLOAD_CPU:
        // break and fall through to host_to_host_copy
        break;
        // no default case, so warnings are thrown when new offloadmask is implemented!
      }
    case PETSC_OFFLOAD_KOKKOS:
    case PETSC_OFFLOAD_UNALLOCATED:
    case PETSC_OFFLOAD_CPU:
      goto host_to_host_copy;
      // no default case, so warnings are thrown when new offloadmask is implemented!
    }
    device_to_device_copy:
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = PetscDeviceArraycpy(dctx,device_array_write(dctx,yin).ptr,device_array_read(dctx,xin).ptr,n,PETSC_DEVICE_MEMCPY_DTOD);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    goto end;

    device_to_host_copy:
    PetscScalar *yarray;

    ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = PetscDeviceArraycpy(dctx,yarray,device_array_read(dctx,xin),n,PETSC_DEVICE_MEMCPY_DTOH);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    goto end;

    host_to_host_copy:
    ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
  }
  end:
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::swap_async(Vec xin, Vec yin) noexcept
{
  PetscFunctionBegin;
  if (xin != yin) {
    const PetscInt     n = xin->map->n;
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::axpby_async(Vec yin, PetscScalar alpha, PetscScalar beta, Vec xin) noexcept
{
  PetscErrorCode     ierr;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  if (alpha == PetscScalar(0.0)) {
    ierr = scale_async(yin,beta);CHKERRQ(ierr);
  } else if (beta == PetscScalar(1.0)) {
    ierr = axpy_async(yin,alpha,xin);CHKERRQ(ierr);
  } else if (alpha == PetscScalar(1.0)) {
    ierr = aypx_async(yin,beta,xin);CHKERRQ(ierr);
  } else {
    const PetscInt  n = yin->map->n;
    const bool      betaIsZero = beta == PetscScalar(0.0);
    PetscBLASInt    bn;
    auto            yarray = device_array_write(dctx,yin);
    cupmBlasError_t cberr;

    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (betaIsZero) {
      ierr = PetscDeviceArraycpy(dctx,yarray,device_array_read(dctx,xin),n,PETSC_DEVICE_MEMCPY_DTOD);CHKERRQ(ierr);
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::axpbypcz_async(Vec zin, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec xin, Vec yin) noexcept
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

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::norm_async(Vec xin, NormType type, PetscReal *z) noexcept
{
  const PetscInt n = xin->map->n;

  PetscFunctionBegin;
  if (n) {
    PetscInt         flopCount = 0;
    PetscBLASInt     bn;
    cupmBlasHandle_t cupmBlasHandle;
    PetscErrorCode   ierr;

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
        int         i;
        PetscScalar zs;

        // this needs to be redone by hand
        cberr = cupmBlasIXamax(cupmBlasHandle,bn,xarray,1,&i);CHKERRCUPMBLAS(cberr);
        ierr = PetscDeviceArraycpy(dctx,&zs,xarray.ptr+i-1,1,PETSC_DEVICE_MEMCPY_DTOH);CHKERRQ(ierr);
        *z   = PetscAbsScalar(zs);
        // flopCount = ???
        break;
      }
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(flopCount);CHKERRQ(ierr);
    ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  } else {
    z[0] = 0.0;
    if (type == NORM_1_AND_2) z[1] = 0.0;
  }
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
