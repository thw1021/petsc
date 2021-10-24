#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // why

#include <petsc/private/vecimpl.h>          /*I <petscvec.h> I*/
#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>

#if !defined(__cplusplus) || !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#  error "VecSeqCUPM requires C++11"
#endif

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

#define PETSC_VEC_CUPM_DEFINE_TO_DEVICE_TYPE(PREFIX)    \
  template <>                                           \
  inline constexpr PetscDeviceType                      \
  CUPMToDeviceType<CUPMDeviceType::PREFIX>() noexcept   \
  {                                                     \
    return CAT(PETSC_DEVICE_,PREFIX);                   \
  }

#define PETSC_VEC_CUPM_DEFINE_UTILITY(PREFIX)           \
  PETSC_VEC_CUPM_DEFINE_SUPERTYPE_GETTERS(PREFIX);      \
  PETSC_VEC_CUPM_DEFINE_SUBTYPE_GETTERS(PREFIX);        \
  PETSC_VEC_CUPM_DEFINE_TO_DEVICE_TYPE(PREFIX)


PETSC_VEC_CUPM_DEFINE_UTILITY(CUDA);
PETSC_VEC_CUPM_DEFINE_UTILITY(HIP);

} // namespace detail

template <CUPMDeviceType T>
class VecSeq_CUPM : CUPMInterface<T>
{
public:
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(cupmInterface_t,T);

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

  PETSC_NODISCARD static PetscErrorCode __moveDeviceArrayAsync(PetscDeviceContext dctx, Vec v) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
    ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
    if (v->offloadmask == PETSC_OFFLOAD_CPU) {
      const auto  xferSize = (v->map->n)*sizeof(*(__vec_impls_cast(v)->array))
      cupmError_t cerr;

      ierr = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
      cerr = cupmMemcpy(__cupm_impls_cast(v)->GPUArray,__vec_impls_cast(v)->array,xferSize,cupmMemcpyHostToDevice);CHKERRCUPM(cerr);
      ierr = PetscLogCpuToGpu(xferSize);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __getDeviceArrayAsync(PetscDeviceContext dctx, Vec v, PetscScalar **array) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PescValidPointer(array,3);
    ierr = __moveDeviceArrayAsync(dctx,v);CHKERRQ(ierr);
    *array = __cupm_impls_cast(v)->GPUarray;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __getDeviceArrayReadAsync(PetscDeviceContext dctx, Vec v, const PetscScalar **array) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __getDeviceArrayAsync(dctx,v,const_cast<PetscScalar**>(array));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode __getCurrentContext(PetscDeviceContext *dctx) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(dctx,detail::CUPMToDeviceType<T>());CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

public:
  PETSC_NODISCARD static PetscErrorCode aypxAsync(Vec,PetscScalar,Vec) noexcept;
};

template <CUPMDeviceType T>
inline PetscErrorCode VecSeq_CUPM<T>::aypxAsync(Vec yin, PetscScalar alpha, Vec xin) noexcept
{
  const PetscScalar  *xarray;
  PetscScalar        *yarray;
  PetscErrorCode     ierr;
  PetscBLASInt       one = 1,bn = 0;
  PetscScalar        sone = 1.0;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  cupmError_t        cerr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = __getCurrentContext(&dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,&handle);CHKERRQ(ierr);
  ierr = __getDeviceArrayReadAsync(dctx,xin,&xarray);CHKERRQ(ierr);
  ierr = __getDeviceArrayAsync(dctx,yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (alpha == PetscScalar(0.0)) {
    cerr = cupmMemcpy(yarray,xarray,bn*sizeof(*yarray),cupmMemcpyDeviceToDevice);CHKERRCUPM(cerr);
  } else if (alpha == PetscScalar(1.0)) {
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuFlops(1.0*yin->map->n);CHKERRQ(ierr);
  } else {
    cberr = cublasXscal(cublasv2handle,bn,&alpha,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,&sone,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
