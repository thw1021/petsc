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
    PetscScalar   *device_array; // gpu data
    PetscCopyMode ptr_ownership; // does PETSc own the array ptr?
    PetscBool     nvshmem;       // is array allocated in nvshmem? It is used to allocate
                                 // Mvctx->lvec in nvshmem
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
      cupmStream_t   stream;
      cupmError_t    cerr;

      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(__cupm_impls_cast(v)->device_array,__vec_impls_cast(v)->array,v->map->n*sizeof(PetscScalar),cupmMemcpyHostToDevice,stream);CHKERRCUPM(cerr);
      ierr = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
    PetscFunctionReturn(0);
  }

  template <bool read>
  struct device_array
  {
    using pointer_type = util::conditional_t<read,const PetscScalar*&,PetscScalar*&>;

    const pointer_type ptr;

    operator pointer_type() const { return this->ptr; }

  protected:
    constexpr device_array(Vec vector) noexcept : ptr(__cupm_impls_cast(vector)->device_array) {  }
  };

  struct device_array_read  : device_array<true>
  {
    constexpr device_array_read(PetscDeviceContext dctx, Vec vector) noexcept
      : device_array<true>(vector)
    { CHKERRABORT(PETSC_COMM_SELF,__move_device_array(dctx,vector)); }
  };

  struct device_array_write : device_array<false>
  {
  private:
    const Vec _v;

  public:
    constexpr device_array_write(PetscDeviceContext dctx, Vec vector) noexcept
      : device_array<false>(vector), _v(vector)
    { }

    ~device_array_write()
    {
      // 1. can't actually do anything about the error since we may already be seterrq-ing out
      // 2. no clue what happens if two different error codes are in flight simultaneously
      (void)PetscObjectStateIncrease(reinterpret_cast<PetscObject>(_v));
      _v->offloadmask = PETSC_OFFLOAD_GPU;
    }
  };

  PETSC_NODISCARD static PetscErrorCode __get_handles(PetscDeviceContext *dctx, cupmBlasHandle_t *handle = nullptr) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(dctx,cupmDeviceTypeToPetsc());CHKERRQ(ierr);
    if (handle) {ierr = PetscDeviceContextGetBLASHandle_Internal(*dctx,handle);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __get_handles(PetscDeviceContext *dctx, cupmStream_t *handle) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __get_handles(dctx);CHKERRQ(ierr); // call the other overload
    if (handle) {ierr = PetscDeviceContextGetStreamHandle_Internal(*dctx,handle);CHKERRQ(ierr);}
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
    .norm            = norm_async,
    .dotnorm2        = dotnorm2_async,
    .destroy         = destroy_async,
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
  PETSC_NODISCARD static PetscErrorCode norm_async(Vec,NormType,PetscReal*) noexcept;
  PETSC_NODISCARD static PetscErrorCode dotnorm2_async(Vec,Vec,PetscScalar*,PetscScalar*) noexcept;
  PETSC_NODISCARD static PetscErrorCode destroy_async(Vec) noexcept;
};

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin) noexcept
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
    const bool       alphaIsOne = alpha == PetscScalar(1.0);
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
      cupmError_t     cerr;
      cupmStream_t    stream;
      PetscBLASInt    bn;

      ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
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
    const size_t       n = yin->map->n*sizeof(PetscScalar);
    PetscBool          yiscuda = PETSC_TRUE,xondevice = PETSC_TRUE; // assume we start on device
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
      ierr = PetscObjectTypeCompareAny((PetscObject)yin,&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
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
      cerr = cupmMemcpyAsync(device_array_write(dctx,yin).ptr,device_array_read(dctx,xin).ptr,n,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyHostToDevice: {
      // not terrible
      const PetscScalar *xarray;

      ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(device_array_write(dctx,yin).ptr,xarray,n,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    } break;
    case cupmMemcpyDeviceToHost: {
      // not great
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(yarray,device_array_read(dctx,xin).ptr,n,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    case cupmMemcpyHostToHost:
      // the worst case
      ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"Unknown cupmMemcpyKind %d",mode);
    }
  }
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
      cupmStream_t stream;

      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(yarray,device_array_read(dctx,xin),n*sizeof(*yarray),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
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
  } else {
    z[0] = 0.0;
    // yes this technically sets z[0] = 0 again half the time
    z[type == NORM_1_AND_2] = 0.0;
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::dotnorm2_async(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = dot_async(s,t,dp);CHKERRQ(ierr);
  ierr = dot_async(t,t,nm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T> inline
PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v) noexcept
{
  PetscErrorCode  ierr;
  auto           *vcu = __impls_cast(v->spptr);

  PetscFunctionBegin;
  if (vcu) {
    switch (vcu->ptr_ownership) {
    case PETSC_COPY_VALUES:
    case PETSC_OWN_POINTER:
      if (PetscDefined(HAVE_NVSHMEM) && vcu->nvshmem) {
        ierr = PetscNvshmemFree(vcu->device_array);CHKERRQ(ierr);
      } else {
        PetscDeviceContext dctx; // unused
        cupmStream_t       stream;
        cupmError_t        cerr;

        ierr = __get_handles(&dctx,&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      }
    case PETSC_USE_POINTER:
      break;
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = VecDestroy_SeqCUDA_Private(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
