#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // why

#include <petsc/private/vecimpl.h>          /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h> // for Vec_Seq
#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>

#if !defined(__cplusplus) || !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#  error "VecSeqCUPM requires C++11"
#endif

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace Petsc
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

  template <bool write>
  struct device_array
  {
  private:
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

  public:
    using pointer_type = std::conditional_t<write,PetscScalar *const,const PetscScalar *const>;

    const Vec    v;
    pointer_type ptr;

    constexpr device_array(PetscDeviceContext dctx, Vec vector) noexcept
      : v(write ? vector : nullptr)
    {
      if (write) {auto ierr = __move_device_array(dctx,vector);CHKERRABORT(PETSC_COMM_SELF,ierr);}
      ptr = __cupm_impls_cast(vector)->GPUarray;
    }

    ~device_array()
    {
      if (write) {
        auto ierr = PetscObjectStateIncrease(reinterpret_cast<PetscObject>(this->v));CHKERRABORT(PETSC_COMM_SELF,ierr);
        this->v->offloadmask = PETSC_OFFLOAD_GPU;
      }
    }
  };

  struct device_array_read  : device_array<false> { };
  struct device_array_write : device_array<true>  { };

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
  // TODO remove designated initializers
  const struct _VecOps ops = {
    .axpy            = axpy_async,
    .aypx            = aypx_async,
    .pointwisedivide = pointwisedivide_async,
    .waxpy           = waxpy_async,
    .maxpy           = maxpy_async,
    .dot             = dot_async,
  };

  PETSC_NODISCARD static PetscErrorCode aypx_async(Vec,PetscScalar,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode axpy_async(Vec,PetscScalar,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode pointwisedivide_async(Vec,Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode waxpy_async(Vec,PetscScalar,Vec,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode maxpy_async(Vec,PetscInt,const PetscScalar*,Vec*) noexcept;
  PETSC_NODISCARD static PetscErrorCode dot_async(Vec,Vec,PetscScalar*) noexcept;
};

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin) noexcept
{
  const PetscInt     n = yin->map->n;
  PetscBLASInt       bn;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  {
    cupmBlasError_t  cberr;
    auto             xarray = device_array_read(dctx,xin);
    auto             yarray = device_array_write(dctx,yin);

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (alpha == PetscScalar(0.0)) {
      ierr = PetscDeviceArraycpy(dctx,yarray.ptr,xarray.ptr,bn,PETSC_DEVICE_MEMCPY_DTOD);CHKERRQ(ierr);
    } else if (alpha == PetscScalar(1.0)) {
      cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,xarray.ptr,1,yarray.ptr,1);CHKERRCUPMBLAS(cberr);
      ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
    } else {
      constexpr PetscScalar sone = 1.0;

      cberr = cupmBlasXscal(cupmBlasHandle,bn,&alpha,yarray.ptr,1);CHKERRCUPMBLAS(cberr);
      cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&sone,xarray.ptr,1,yarray.ptr,1);CHKERRCUPMBLAS(cberr);
      ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  ierr = PetscLogCpuToGpuScalar(sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscScalar alpha, Vec xin) noexcept
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
    cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin).ptr,1,device_array_write(dctx,yin).ptr,1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin) noexcept
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseDivide_Seq(win,xin,yin);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  {
    const PetscInt n      = xin->map->n;
    auto           xarray = device_array_read(dctx,xin);
    auto           yarray = device_array_read(dctx,yin);
    auto           warray = device_array_write(dctx,win);

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      auto xptr = thrust::device_pointer_cast(xarray.ptr);
      auto yptr = thrust::device_pointer_cast(yarray.ptr);
      auto wptr = thrust::device_pointer_cast(warray.ptr);

      thrust::transform(xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
    } catch (const std::exception &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscScalar alpha, Vec xin, Vec yin) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = VecCopy_SeqCUDA(yin,win);CHKERRQ(ierr);
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
      cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin).ptr,1,warray.ptr,1);CHKERRCUPMBLAS(cberr);
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
  const PetscInt      n  = xin->map->n;
  PetscBLASInt        bn;
  PetscDeviceContext  dctx;
  cupmBlasHandle_t    cupmBlasHandle;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  {
    auto xarray = device_array_write(dctx,xin);

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    for (PetscInt j = 0; j < nv; ++j) {
      cupmBlasError_t cberr = cupmBlasXaxpy(cupmBlasHandle,bn,alpha+j,device_array_read(dctx,y[j]).ptr,1,xarray.ptr,1);CHKERRCUPMBLAS(cberr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  ierr = PetscLogGpuFlops(nv*2*n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(nv*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscScalar *z) noexcept
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
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the
   * second */
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdot(cupmBlasHandle,bn,device_array_read(dctx,yin).ptr,1,device_array_read(dctx,xin).ptr,1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (n > 0) {ierr = PetscLogGpuFlops(2*(n-1));CHKERRQ(ierr);}
  ierr = PetscLogGpuToCpuScalar(sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
