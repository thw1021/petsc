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
class PETSC_VISIBILITY_INTERNAL VecSeq_CUPM : CUPMBlasInterface<T>
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

  PETSC_NODISCARD static PetscErrorCode __get_device_array(PetscDeviceContext dctx, Vec v, PetscScalar **array) noexcept
  {
    PetscFunctionBegin;
    PetscValidPointer(array,3);
    *array = __cupm_impls_cast(v)->GPUarray;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __get_device_array_write(PetscDeviceContext dctx, Vec v, PetscScalar **array) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __get_device_array(dctx,v,array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __get_device_array_read(PetscDeviceContext dctx, Vec v, const PetscScalar **array) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __move_device_array(dctx,v);CHKERRQ(ierr);
    ierr = __get_device_array(dctx,v,const_cast<PetscScalar**>(array));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __restore_device_array_read(PetscDeviceContext dctx, Vec v, const PetscScalar **array) noexcept
  {
    PetscFunctionBegin;
    *array = nullptr;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __restore_device_array_write(PetscDeviceContext dctx, Vec v, PetscScalar **array) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    v->offloadmask = PETSC_OFFLOAD_GPU;
    ierr   = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
    *array = nullptr;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __get_device_context(PetscDeviceContext *dctx) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(dctx,cupmDeviceTypeToPetsc());CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

public:
  // TODO remove designated initializers
  const struct _VecOps ops = {
    .axpy = axpy_async,
    .aypx = aypx_async
  };

  PETSC_NODISCARD static PetscErrorCode aypx_async(Vec,PetscScalar,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode axpy_async(Vec,PetscScalar,Vec) noexcept;
  PETSC_NODISCARD static PetscErrorCode pointwisedivide_async(Vec,Vec,Vec) noexcept;
};

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin) noexcept
{
  const PetscScalar  *xarray;
  PetscScalar        *yarray;
  PetscBLASInt       one = 1,bn = 0;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = __get_device_context(&dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = __get_device_array_read(dctx,xin,&xarray);CHKERRQ(ierr);
  ierr = __get_device_array_write(dctx,yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (alpha == PetscScalar(0.0)) {
    ierr = PetscDeviceArraycpy(dctx,yarray,xarray,bn,PETSC_DEVICE_MEMCPY_DTOD);CHKERRQ(ierr);
  } else if (alpha == PetscScalar(1.0)) {
    cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,xarray,one,yarray,one);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuFlops(1.0*yin->map->n);CHKERRQ(ierr);
  } else {
    PetscScalar sone = 1.0;

    cberr = cupmBlasXscal(cupmBlasHandle,bn,&alpha,yarray,one);CHKERRCUPMBLAS(cberr);
    cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&sone,xarray,one,yarray,one);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = __restore_device_array_read(dctx,xin,&xarray);CHKERRQ(ierr);
  ierr = __restore_device_array_write(dctx,yin,&yarray);CHKERRQ(ierr);
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
    PetscBLASInt        one = 1,bn = 0;
    cupmBlasHandle_t    cupmBlasHandle;
    cupmBlasError_t     cberr;
    const PetscScalar  *xarray;
    PetscScalar        *yarray;
    PetscDeviceContext  dctx;

    ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
    ierr = __get_device_context(&dctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = __get_device_array_read(dctx,xin,&xarray);CHKERRQ(ierr);
    ierr = __get_device_array_write(dctx,yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXaxpy(cupmBlasHandle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = __restore_device_array_read(dctx,xin,&xarray);CHKERRQ(ierr);
    ierr = __restore_device_array_write(dctx,yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin) noexcept
{
  const PetscInt      n      = xin->map->n;
  const PetscScalar  *xarray = nullptr,*yarray = nullptr;
  PetscScalar        *warray = nullptr;
  PetscDeviceContext  dctx;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseDivide_Seq(win,xin,yin);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = __get_device_context(&dctx);CHKERRQ(ierr);
  ierr = __get_device_array_write(dctx,win,&warray);CHKERRQ(ierr);
  ierr = __get_device_array_read(dctx,xin,&xarray);CHKERRQ(ierr);
  ierr = __get_device_array_read(dctx,yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto xptr = thrust::device_pointer_cast(xarray);
    auto yptr = thrust::device_pointer_cast(yarray);
    auto wptr = thrust::device_pointer_cast(warray);

    thrust::transform(xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
  } catch (const std::exception &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
