#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // REVIEW ME: why

#include <petsc/private/vecimpl.h>         /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h> // for Vec_Seq
#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>
#include <petsc/private/randomimpl.h> // for _p_PetscRandom
#include <array>

#if !defined(__cplusplus) || !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#  error "VecSeq_CUPM requires C++11"
#endif

#if PetscDefined(HAVE_CUPM)
#  include <thrust/device_ptr.h>
#  include <thrust/transform.h>
#  include <thrust/transform_reduce.h>
#  include <thrust/reduce.h>
#  include <thrust/functional.h>
#  include <thrust/iterator/counting_iterator.h>
#endif

// TODO
// - refactor the AXPY's for code reuse
// - figure out how to template which thrust namespace to use so we can do
//   thrust::<backend>::par.on(stream)
// - get rid of these undefs, they are for error checking purposes only
#undef CHKERRCUDA
#undef CHKERRCUBLAS
// - finish the blas wrappers
// - maybe reintroduce PetscDeviceMalloc()?
// - There is also an overloaded version of cudaMallocAsync that takes the same arguments as
//   cudaMallocFromPoolAsync
// - touch up the docs for both implementations
// - pick one of the VecGetArray<modifier>() to explain data movement semantics in the docs and
//   have everyone else refer to it
// - remove the cuda and hip separate versions
// - remove bindtocpu?
// - do rocblas instead of hipblas
// - remove this define and use the right header (i.e. clean up the headers first)
#define PetscNvShmemFree(ptr) 0

namespace Petsc
{

namespace Impl
{

namespace detail
{

template <bool b> struct UseComplexTag { };

template <CUPMDeviceType T>
struct VecCUPMBase
{
  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 PetscLogEvent VEC_CUPMCopyToGPU())
  {
    switch (T) {
    case CUPMDeviceType::CUDA: return VEC_CUDACopyToGPU;
    case CUPMDeviceType::HIP:  return VEC_HIPCopyToGPU;
    }
  }

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 PetscLogEvent VEC_CUPMCopyFromGPU())
  {
    switch (T) {
    case CUPMDeviceType::CUDA: return VEC_CUDACopyFromGPU;
    case CUPMDeviceType::HIP:  return VEC_HIPCopyFromGPU;
    }
  }

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 const char* VECSEQCUPM())
  {
    switch (T) {
    case CUPMDeviceType::CUDA: return VECSEQCUDA;
    case CUPMDeviceType::HIP:  return VECSEQHIP;
    }
  };

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 const char* VECMPICUPM())
  {
    switch (T) {
    case CUPMDeviceType::CUDA: return VECMPICUDA;
    case CUPMDeviceType::HIP:  return VECMPIHIP;
    }
  };
};

enum class MemoryAccess {
  READ,
  WRITE,
  READ_WRITE
};

} // namespace detail

#define PETSC_VECCUPM_BASE_CLASS_HEADER(name,Tp)                        \
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t,Tp); \
  using MemoryAccess = Petsc::Impl::detail::MemoryAccess;               \
  using name = Petsc::Impl::detail::VecCUPMBase<Tp>;                    \
  using name::VEC_CUPMCopyToGPU;                                        \
  using name::VEC_CUPMCopyFromGPU;                                      \
  using name::VECSEQCUPM;                                               \
  using name::VECMPICUPM

#define CHKERRCXXCTOR(...) CHKERRABORT(PETSC_COMM_SELF,(__VA_ARGS__))
#define CHKERRCXXDTOR(...) CHKERRCONTINUE((__VA_ARGS__))

template <CUPMDeviceType T>
struct VecSeq_CUPM : detail::VecCUPMBase<T>,CUPMBlasInterface<T>
{
public:
  PETSC_VECCUPM_BASE_CLASS_HEADER(base_type,T);

  struct Vec_CUPM
  {
    PetscScalar   *device_array; // gpu data
    PetscCopyMode ptr_ownership; // does PETSc own the array ptr?
    PetscBool     nvshmem;       // is array allocated in nvshmem? It is used to allocate
                                 // Mvctx->lvec in nvshmem
    static_assert(std::is_same<decltype(device_array),decltype(Vec_Seq::array)>::value,"");
  };

private:
  // utility
  PETSC_CXX_COMPAT_DECL(PetscErrorCode CUPMBlasIntCast_(PetscInt x, cupmBlasInt_t *y))
  {
    using petsc_type = decltype(x);
    using blas_type  = util::remove_pointer_t<decltype(y)>;

    PetscFunctionBegin;
    if PETSC_CONSTEXPR_17 (!std::is_same<petsc_type,blas_type>::value) {
      if (PetscUnlikely(x > std::numeric_limits<blas_type>::max())) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"%" PetscInt_FMT " is too big for %s, which may be restricted to 32 bit integers",x,cupmBlasName());
    }
    if (PetscUnlikely(x < 0)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Passing negative integer to %s routine: %" PetscInt_FMT,cupmBlasName(),x);
    *y = static_cast<blas_type>(x);
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr Vec_Seq* VecSeqCast_(Vec v))
  {
    return static_cast<Vec_Seq*>(v->data);
  }

  PETSC_CXX_COMPAT_DECL(constexpr Vec_CUPM* CUPMCast_(Vec v))
  {
    return static_cast<Vec_CUPM*>(v->spptr);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetHandleDispatch_(PetscDeviceContext *dctx, cupmBlasHandle_t *handle, cupmStream_t *stream))
  {
    PetscDeviceContext dctx_;
    PetscErrorCode     ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx_,cupmDeviceTypeToPetscDeviceType());CHKERRQ(ierr);
    if (handle) {ierr = PetscDeviceContextGetBLASHandle_Internal(dctx_,handle);CHKERRQ(ierr);}
    if (stream) {ierr = PetscDeviceContextGetStreamHandle_Internal(dctx_,stream);CHKERRQ(ierr);}
    if (dctx) *dctx = dctx_;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetHandles_(PetscDeviceContext *dctx, cupmBlasHandle_t *handle = nullptr, cupmStream_t *stream = nullptr))
  {
    return GetHandleDispatch_(dctx,handle,stream);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetHandles_(PetscDeviceContext *dctx, cupmStream_t *handle))
  {
    return GetHandles_(dctx,nullptr,handle); // other overload
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetHandles_(cupmStream_t *handle))
  {
    return GetHandles_(nullptr,nullptr,handle); // other overload
  }

  // data movement
  PETSC_CXX_COMPAT_DECL(PetscErrorCode HostAllocateCheck_(PetscDeviceContext,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode DeviceAllocateCheck_(PetscDeviceContext,Vec,PetscScalar* = nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode CopyToDevice_(PetscDeviceContext,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode CopyToHost_(PetscDeviceContext,Vec));

  // a simple RAII helper for PetscMallocSet[CUDA|HIP]Host(). it exists because integrating the
  // regular versions would be an enormous pain to square with the templated types...
  struct PETSC_NODISCARD UseCUPMHostAlloc
  {
    // would have loved to just do
    //
    // const auto oldmalloc = PetscTrMalloc;
    //
    // but in order to use auto the member needs to be static; in order to be static it must
    // also be constexpr -- which in turn requires an initializer (also implicitly required by
    // auto). But constexpr needs a constant expression initializer, so we can't initialize it
    // with global (mutable) variables...
#define DECLTYPE_AUTO(left,right) decltype(right) left = right
    const DECLTYPE_AUTO(oldmalloc,PetscTrMalloc);
    const DECLTYPE_AUTO(oldfree,PetscTrFree);
    const DECLTYPE_AUTO(oldrealloc,PetscTrRealloc);
#undef DECLTYPE_AUTO
    const bool v;

    UseCUPMHostAlloc(bool useit) noexcept : v(useit)
    {
      if (useit) {
        // all unused arguments are un-named, this saves having to add PETSC_UNUSED to them all
        PetscTrMalloc  = [](size_t sz,PetscBool,int,const char*,const char*,void **ptr)
        {
          PetscFunctionBegin;
          CHKERRCUPM(cupmMallocHost(ptr,sz));
          PetscFunctionReturn(0);
        };
        PetscTrFree    = [](void *ptr,int,const char*,const char*)
        {
          PetscFunctionBegin;
          CHKERRCUPM(cupmFreeHost(ptr));
          PetscFunctionReturn(0);
        };
        PetscTrRealloc = [](size_t,int,const char*,const char*,void**)
        {
          // REVIEW ME: can be implemented by malloc->copy->free?
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEM,"%s has no realloc()",cupmName());
        };
      }
    }

    auto value() const noexcept -> decltype(this->v) { return this->v; }

    ~UseCUPMHostAlloc() noexcept
    {
      if (this->v) {
        PetscTrMalloc  = this->oldmalloc;
        PetscTrFree    = this->oldfree;
        PetscTrRealloc = this->oldrealloc;
      }
    }
  };

  // base class that determines constness of the pointer type, holds the pointer itself and
  // provides the implicit conversion operator
  template <PetscMemType MT, MemoryAccess MA>
  struct PETSC_NODISCARD vector_array
  {
    static const auto memory_type = MT;
    static const auto access_type = MA;

    // PetscScalar
    using value_type         = util::remove_pointer_t<decltype(Vec_Seq::array)>;
    // PetscScalar*
    using pointer_type       = util::add_pointer_t<value_type>;
    // const PetscScalar
    using const_value_type   = util::add_const_t<value_type>;
    // const PetscScalar*
    using const_pointer_type = util::add_pointer_t<const_value_type>;

    // PetscScalar *const
    const pointer_type ptr;

    constexpr operator const_pointer_type() const noexcept { return this->ptr; }
    constexpr operator       pointer_type() const noexcept { return this->ptr; }

  protected:
    vector_array(PetscDeviceContext dctx, Vec v) noexcept : ptr(initialize_(dctx,v)), v_(v) { }

    ~vector_array() noexcept { CHKERRCXXDTOR(restorearray_async<MT,MA>(v_,this->ptr)); }

  private:
    const Vec v_;

    PETSC_CXX_COMPAT_DECL(pointer_type initialize_(PetscDeviceContext dctx, Vec v))
    {
      pointer_type a;
      CHKERRCXXCTOR(getarray_async<MT,MA>(v,&a));
      return a;
    }
  };

  // RAII versions of the get/restore array routines
  struct PETSC_NODISCARD DeviceArrayRead  : vector_array<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ>
  {
    using base_type = vector_array<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ>;

    DeviceArrayRead(PetscDeviceContext dctx, Vec v) noexcept : base_type(dctx,v) { }
  };

  struct PETSC_NODISCARD DeviceArrayWrite : vector_array<PETSC_MEMTYPE_DEVICE,MemoryAccess::WRITE>
  {
    using base_type = vector_array<PETSC_MEMTYPE_DEVICE,MemoryAccess::WRITE>;

    DeviceArrayWrite(PetscDeviceContext dctx, Vec v) noexcept : base_type(dctx,v) { }
  };

  struct PETSC_NODISCARD HostArrayRead    : vector_array<PETSC_MEMTYPE_HOST,MemoryAccess::READ>
  {
    using base_type = vector_array<PETSC_MEMTYPE_HOST,MemoryAccess::READ>;

    HostArrayRead(PetscDeviceContext dctx, Vec v) noexcept : base_type(dctx,v) { }
  };

  struct PETSC_NODISCARD  HostArrayWrite   : vector_array<PETSC_MEMTYPE_HOST,MemoryAccess::WRITE>
  {
    using base_type = vector_array<PETSC_MEMTYPE_HOST,MemoryAccess::WRITE>;

    HostArrayWrite(PetscDeviceContext dctx, Vec v) noexcept : base_type(dctx,v) { }
  };

  // common core for min and max
  template <typename TupleFuncT, typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode minmax_async_(TupleFuncT&&,UnaryFuncT&&,PetscReal,Vec,PetscInt*,PetscReal*));
  // common core for pointwise unary operations
  template <typename BinaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisebinary_async_(BinaryFuncT&&,Vec,Vec,Vec));
  // mdot dispatchers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(detail::UseComplexTag<true>,Vec,PetscInt,const Vec[],PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(detail::UseComplexTag<false>,Vec,PetscInt,const Vec[],PetscScalar*));
  // dispatcher for the actual kernels for mdot when NOT configured for complex, called by
  // mdot_async_(use_complex_tag<false>,...)
  template <int N>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_kernel_dispatch_(PetscDeviceContext,cupmStream_t,const PetscScalar*,const Vec[],PetscInt,PetscScalar*,PetscInt*));
  // common core for the various create routines
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async_(Vec,PetscScalar* /*device_ptr*/= nullptr));

public:
  // callable directly via a bespoke function
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async(MPI_Comm,PetscInt,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createwithbotharrays_async(MPI_Comm,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],Vec*));
  template <PetscMemType mtype, MemoryAccess access>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarray_async(Vec,PetscScalar**));
  template <PetscMemType mtype, MemoryAccess access>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorearray_async(Vec,PetscScalar**));
  template <PetscMemType mtype>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec));
  template <PetscMemType mtype>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec,const PetscScalar*));
  template <PetscMemType mtype>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode replacearray_async(Vec,const PetscScalar*));

  // callable indirectly via function pointers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarrayandmemtype_async(Vec,PetscScalar**,PetscMemType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorearrayandmemtype_async(Vec,PetscScalar**));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode aypx_async(Vec,PetscScalar,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpy_async(Vec,PetscScalar,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisedivide_async(Vec,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisemult_async(Vec,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode reciprocal_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waxpy_async(Vec,PetscScalar,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode maxpy_async(Vec,PetscInt,const PetscScalar*,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec,Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async(Vec,PetscInt,const Vec[],PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_async(Vec,PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode scale_async(Vec,PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode tdot_async(Vec,Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy_async(Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode swap_async(Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpby_async(Vec,PetscScalar,PetscScalar,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpbypcz_async(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode norm_async(Vec,NormType,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dotnorm2_async(Vec,Vec,PetscScalar*,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode conjugate_async(Vec));
  template <bool read>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getlocalvector_async(Vec,Vec));
  template <bool read>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorelocalvector_async(Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec,PetscInt*,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec,PetscInt*,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode sum_async(Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode shift_async(Vec,PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setrandom_async(Vec,PetscRandom));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode bindtocpu_async(Vec,PetscBool));
};

template <CUPMDeviceType T> template <PetscMemType MT, detail::MemoryAccess MA>
const PetscMemType VecSeq_CUPM<T>::vector_array<MT,MA>::memory_type;

template <CUPMDeviceType T> template <PetscMemType MT, detail::MemoryAccess MA>
const detail::MemoryAccess VecSeq_CUPM<T>::vector_array<MT,MA>::access_type;

#undef PETSC_VECCUPM_BASE_CLASS_HEADER
#undef CHKERRCXXCTOR

// ================================================================================== //
//                                                                                    //
//                                  utility methods                                   //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                                  array accessors                                   //

#define STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype) \
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE),"Only comparisons between purely host and device memory are valid")

// v->ops->getarray[read|write] or VecCUPMGetArray[Read|Write]()
template <CUPMDeviceType T>
template <PetscMemType mtype, detail::MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getarray_async(Vec v, PetscScalar **a))
{
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  constexpr auto     hostmem = mtype == PETSC_MEMTYPE_HOST;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  if PETSC_CONSTEXPR_17 (access == MemoryAccess::WRITE) {
    ierr = (hostmem ? HostAllocateCheck_(dctx,v) : DeviceAllocateCheck_(dctx,v));CHKERRQ(ierr);
    // immediately assume modified
    v->offloadmask = hostmem ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  } else {
    // READ or READ_WRITE
    ierr = (hostmem ? CopyToHost_ : CopyToDevice_)(dctx,v);CHKERRQ(ierr);
    if PETSC_CONSTEXPR_17 (access == MemoryAccess::READ_WRITE) {
      v->offloadmask = hostmem ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
    }
  }
  *a = hostmem ? VecSeqCast_(v)->array : CUPMCast_(v)->device_array;
  PetscFunctionReturn(0);
}

// v->ops->restorearray[read|write] or VecCUPMRestoreArray[Read|Write]()
template <CUPMDeviceType T>
template <PetscMemType mtype, detail::MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorearray_async(Vec v, PetscScalar **a))
{
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if PETSC_CONSTEXPR_17 (access != MemoryAccess::READ) {
    constexpr auto hostmem = mtype == PETSC_MEMTYPE_HOST;
    // WRITE or READ_WRITE
    auto ierr = PetscObjectStateIncrease(PetscObjectCast(v));CHKERRQ(ierr);
    v->offloadmask = hostmem ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  *a = nullptr;
  PetscFunctionReturn(0);
}

#undef STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED

// v->ops->getarrayandmemtype
template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getarrayandmemtype_async(Vec v, PetscScalar **a, PetscMemType *mtype))
{
  PetscFunctionBegin;
  if (v->offloadmask & PETSC_OFFLOAD_GPU) {
    const auto vcu = CUPMCast_(v);
    // return device pointer when device has up-to-date data, such as when offloadmask is
    // PETSC_OFFLOAD_BOTH
    *a = vcu->device_array;
    // change the mask once GPU gets write access, don't wait until restore array
    v->offloadmask = PETSC_OFFLOAD_GPU;
    if (mtype) {
      // I could just as easily have done
      //
      // if PETSC_CONSTEXPR_17 (T == CUPMDeviceType::HIP) *mtype = PETSC_MEMTYPE_HIP;
      // else *mtype = cupm_impls_cast_(v)->nvshmem ? PETSC_MEMTYPE_NVSHMEM : PETSC_MEMTYPE_CUDA;
      //
      // but that would be very brittle to additions to CUPMDeviceType, as it would still
      // "work" silently
      if (vcu->nvshmem) *mtype = PETSC_MEMTYPE_NVSHMEM;
      else *mtype = cupmDeviceTypeToPetscMemType();
    }
  } else {
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = HostAllocateCheck_(dctx,v);CHKERRQ(ierr);
    *a   = *static_cast<decltype(a)>(v->data);
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(0);
}

// v->ops->restorearrayandmemtype
template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorearrayandmemtype_async(Vec v, PetscScalar **a))
{
  PetscFunctionBegin;
  *a             = nullptr;
  v->offloadmask = (v->offloadmask & PETSC_OFFLOAD_GPU) ? PETSC_OFFLOAD_GPU : PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::HostAllocateCheck_(PetscDeviceContext, Vec v))
{
  PetscErrorCode ierr;
  auto           vseq = VecSeqCast_(v);

  PetscFunctionBegin;
  if (!v->data) {
    ierr = PetscNewLog(PetscObjectCast(v),&vseq);CHKERRQ(ierr);
    v->data = vseq;
  }
  if (!vseq->array) {
    const auto n      = v->map->n;
    const auto nbytes = n*sizeof(*vseq->array_allocated);

    {
      const auto useit = UseCUPMHostAlloc(nbytes > v->minimum_bytes_pinned_memory);

      if (useit.value()) v->pinned_memory = PETSC_TRUE;
      ierr = PetscMalloc1(n,&vseq->array_allocated);CHKERRQ(ierr);
    }
    ierr = PetscLogObjectMemory(PetscObjectCast(v),nbytes);CHKERRQ(ierr);
    vseq->array = vseq->array_allocated;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) v->offloadmask = PETSC_OFFLOAD_CPU;
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::DeviceAllocateCheck_(PetscDeviceContext dctx, Vec v, PetscScalar *device_array))
{
  auto           vcu = CUPMCast_(v);
  cupmBlasInt_t  bn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (vcu) PetscFunctionReturn(0);
  else {
    PetscBool flg;
    auto      mem = static_cast<PetscInt>(v->minimum_bytes_pinned_memory);

    // Need to parse command line for minimum size to use for pinned memory allocations on
    // host here
    ierr = PetscObjectOptionsBegin(PetscObjectCast(v));CHKERRQ(ierr);
    ierr = PetscOptionsRangeInt("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",mem,&mem,&flg,0,std::numeric_limits<decltype(mem)>::max());CHKERRQ(ierr);
    if (flg) v->minimum_bytes_pinned_memory = mem;
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  ierr = PetscNewLog(PetscObjectCast(v),&vcu);CHKERRQ(ierr);
  v->spptr = vcu;
  // do a cast to blasint check because if blasint cant hold the size, then any subsequent
  // cupmblas calls can't use it either. Doing this now this means we don't have to check
  // during every function
  ierr = CUPMBlasIntCast_(v->map->n,&bn);CHKERRQ(ierr);
  if (device_array) {
    // array is being placed from the user
    vcu->device_array  = device_array;
    vcu->ptr_ownership = PETSC_USE_POINTER;
    v->offloadmask     = PETSC_OFFLOAD_GPU;
  } else {
    const auto   nbytes = bn*sizeof(*vcu->device_array);
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
    cerr = cupmMallocAsync(reinterpret_cast<void**>(&vcu->device_array),nbytes,stream);CHKERRCUPM(cerr);
    vcu->ptr_ownership = PETSC_OWN_POINTER;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      if (v->data && VecSeqCast_(v)->array) {
        v->offloadmask = PETSC_OFFLOAD_CPU;
      } else {
        v->offloadmask = PETSC_OFFLOAD_GPU;
      }
    }
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::CopyToDevice_(PetscDeviceContext dctx, Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DeviceAllocateCheck_(dctx,v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    const auto   xfersize = v->map->n*sizeof(*VecSeqCast_(v)->array);
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(VEC_CUPMCopyToGPU(),v,0,0,0);CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(CUPMCast_(v)->device_array,VecSeqCast_(v)->array,xfersize,cupmMemcpyHostToDevice,stream);CHKERRCUPM(cerr);
    ierr = PetscLogEventEnd(VEC_CUPMCopyToGPU(),v,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(xfersize);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::CopyToHost_(PetscDeviceContext dctx, Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = HostAllocateCheck_(dctx,v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    const auto   xfersize = v->map->n*sizeof(*CUPMCast_(v)->device_array);
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(VEC_CUPMCopyFromGPU(),v,0,0,0);CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(VecSeqCast_(v)->array,CUPMCast_(v)->device_array,xfersize,cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
    ierr = PetscLogEventEnd(VEC_CUPMCopyFromGPU(),v,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogGpuToCpu(xfersize);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::create_async_(Vec v, PetscScalar *device_array))
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)),&size);CHKERRMPI(ierr);
  if (PetscUnlikely(size > 1)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must create VecSeq on communicator of size 1, have size %d",size);
  ierr = PetscDeviceInitialize(cupmDeviceTypeToPetscDeviceType());CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(v,nullptr);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName(PetscObjectCast(v),VECSEQCUPM());CHKERRQ(ierr);
  ierr = bindtocpu_async(v,PETSC_FALSE);CHKERRQ(ierr);

  // Later, functions check for the Vec_CUPM structure existence, so do not create it without an
  // array attached
  if (device_array) {
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = DeviceAllocateCheck_(dctx,v,device_array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <typename BinaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisebinary_async_(BinaryFuncT&& unary, Vec win, Vec xin, Vec yin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto xptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,xin).ptr);
    auto yptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,yin).ptr);
    auto wptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,win).ptr);

    thrust::transform(xptr,xptr+n,yptr,wptr,std::forward<BinaryFuncT>(unary));
  } catch (const thrust::system_error& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwiseunary_async_(UnaryFuncT&& unary, Vec xin, Vec yin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto xptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,xin).ptr);
    auto yptr = thrust::device_pointer_cast(xin == yin ? xptr : DeviceArrayWrite(dctx,yin).ptr);

    thrust::transform(xptr,xptr+n,yptr,std::forward<BinaryFuncT>(unary));
  } catch (const thrust::system_error& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                                                                    //
//                                  public methods                                    //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                                   constructors                                     //

// v->ops->create
template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::create_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = create_async_(v);CHKERRQ(ierr);
  ierr = set_async(v,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPM()
template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async(MPI_Comm comm, PetscInt n, Vec *v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQCUPM());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPMWithArrays()
template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createwithbotharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar host_array[], const PetscScalar device_array[], Vec *v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*v,bs);CHKERRQ(ierr);
  // REVIEW ME: why no PetscLayoutSetUp()????
  // set v's device array to be device_array, do not allocate memory on host yet.
  ierr = create_async_(*v,PetscRemoveConstCast(device_array));CHKERRQ(ierr);
  if (host_array) {
    VecSeqCast_(*v)->array = PetscRemoveConstCast(host_array);
    (*v)->offloadmask = device_array ? PETSC_OFFLOAD_BOTH : PETSC_OFFLOAD_CPU;
  } else if (device_array) {
    (*v)->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    (*v)->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  // REVIEW ME: should this check exist? It is to assert the following from createwitharrays
  // docstrings,  but doing so is potentially ridiculously expensive:
  // "If both cpuarray and gpuarray are provided, the provided arrays must have identical
  // values."
  if (PetscDefined(USE_DEBUG) && device_array && host_array) {
    constexpr auto      atol   = PetscReal(1e-08),rtol = PetscReal(1e-05);
    const auto          nscal  = n*bs; // REVIEW ME
    const auto          nbytes = nscal*sizeof(*device_array);
    PetscDeviceContext  dctx;
    cupmError_t         cerr;
    cupmStream_t        stream;
    PetscScalar        *debug_array;

    ierr = PetscMalloc1(n,&debug_array);CHKERRQ(ierr);
    ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(debug_array,device_array,nbytes,cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
    ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
    for (PetscInt i = 0; i < nscal; ++i) {
      const auto hreal = PetscRealPart(host_array[i]),  himag = PetscImaginaryPart(host_array[i]);
      const auto dreal = PetscRealPart(debug_array[i]), dimag = PetscImaginaryPart(debug_array[i]);
      const auto close = PetscIsCloseAtTol(hreal,dreal,rtol,atol) && PetscIsCloseAtTol(himag,dimag,rtol,atol);

      if (PetscUnlikely(!close)) SETERRQ6(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Host and device arrays do not match: host[%" PetscInt_FMT "] %g+i%g != device[%" PetscInt_FMT "] %g+i%g",i,hreal,himag,i,dreal,dimag);
    }
    ierr = PetscFree(debug_array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::duplicate_async(Vec v, Vec *y))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = createseqcupm_async(PetscObjectComm(PetscObjectCast(v)),v->map->n,y);CHKERRQ(ierr);
  ierr = PetscLayoutReference(v->map,&(*y)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(PetscObjectCast(v)->olist,&(PetscObjectCast(*y)->olist));CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(PetscObjectCast(v)->qlist,&(PetscObjectCast(*y)->qlist));CHKERRQ(ierr);
  (*y)->stash.ignorenegidx = v->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                    mutatators                                      //

// v->ops->resetarray or VecCUPMResetArray()
template <CUPMDeviceType T>
template <PetscMemType mtype>
// yes (probably Jed :)), ideal world these should be arguments not template parameters. But I
// need to assign this function to a C compatible function pointer, so something like default
// arguments don't work no? Stubs seem like overkill too...
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::resetarray_async(Vec v))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  // REVIEW ME:
  // this is wildly inefficient but must be done if we assume that the placed array must have
  // correct values
  if (PetscMemTypeHost(mtype)) {
    ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
    ierr = VecResetArray_Seq(v);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_CPU;
  } else {
    const auto vseq = VecSeqCast_(v);

    ierr = CopyToDevice_(dctx,v);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease(PetscObjectCast(v));CHKERRQ(ierr);
    CUPMCast_(v)->device_array = vseq->unplacedarray;
    vseq->unplacedarray               = nullptr;
    v->offloadmask                    = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <CUPMDeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::placearray_async(Vec v, const PetscScalar *a))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  if PETSC_CONSTEXPR_17 (PetscMemTypeHost(mtype)) {
    ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
    ierr = VecPlaceArray_Seq(v,a);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_CPU;
  } else {
    const auto vseq = VecSeqCast_(v);

    if (PetscUnlikely(vseq->unplacedarray)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecCUDAPlaceArray()/VecPlaceArray() was already called on this vector, without a call to VecCUDAResetArray()/VecResetArray()");
    ierr = getarray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ_WRITE>(v,&vseq->unplacedarray);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease(PetscObjectCast(v));CHKERRQ(ierr);
    CUPMCast_(v)->device_array = const_cast<PetscScalar*>(a);
    // offload mask set by getarray
  }
  PetscFunctionReturn(0);
}

// v->ops->replacearray or VecCUPMReplaceArray()
template <CUPMDeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::replacearray_async(Vec v, const PetscScalar *a))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if (PetscMemTypeHost(mtype)) {
    const auto vseq = VecSeqCast_(v);

    if (vseq->array != vseq->array_allocated) {
      PetscDeviceContext dctx;
      // make sure the users array has the latest values.
      // REVIEW ME: why? we're about to free it
      ierr = GetHandles_(&dctx);CHKERRQ(ierr);
      ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
    }
    if (vseq->array_allocated) {
      const auto useit = UseCUPMHostAlloc(v->pinned_memory);
      ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
    }
    vseq->array_allocated = vseq->array = PetscRemoveConstCast(a);
    v->pinned_memory      = PETSC_FALSE; // REVIEW ME: we can determine this
    v->offloadmask        = PETSC_OFFLOAD_CPU;
  } else {
    const auto vcu = CUPMCast_(v);

    switch (vcu->ptr_ownership) {
    case PETSC_COPY_VALUES:
    case PETSC_OWN_POINTER:
      if (PetscDefined(HAVE_NVSHMEM) && vcu->nvshmem) {
        ierr = PetscNvshmemFree(vcu->device_array);CHKERRQ(ierr);
      } else {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = GetHandles_(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      }
    case PETSC_USE_POINTER:
      vcu->device_array = PetscRemoveConstCast(a);
      break;
    }
    ierr = PetscObjectStateIncrease(PetscObjectCast(v));CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}

// v->ops->getlocalvector or v->ops->getlocalvectorread
template <CUPMDeviceType T>
template <bool read>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getlocalvector_async(Vec v, Vec w))
{
  PetscErrorCode ierr;
  PetscBool      wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUPM(),&wisseqcupm);CHKERRQ(ierr);
  if (wisseqcupm) {
    if (const auto vseq = VecSeqCast_(w)) {
      if (vseq->array_allocated) {
        const auto useit = UseCUPMHostAlloc(w->pinned_memory);

        ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
        if (useit.value()) w->pinned_memory = PETSC_FALSE;
      }
      vseq->array         = nullptr;
      vseq->unplacedarray = nullptr;
    }
    if (const auto vcu = CUPMCast_(w)) {
      if (vcu->device_array) {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = GetHandles_(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      }
      ierr = PetscFree(w->spptr /* vcu */);CHKERRQ(ierr);
    }
  }
  if (v->petscnative && wisseqcupm) {
    ierr = PetscFree(w->data);CHKERRQ(ierr);
    w->data          = v->data;
    w->offloadmask   = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr         = v->spptr;
    ierr = PetscObjectStateIncrease(PetscObjectCast(w));CHKERRQ(ierr);
  } else {
    const auto arrayptr = &VecSeqCast_(w)->array;
    if (read) {
      ierr = VecGetArrayRead(v,const_cast<const PetscScalar**>(arrayptr));CHKERRQ(ierr);
    } else {
      ierr = VecGetArray(v,arrayptr);CHKERRQ(ierr);
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcupm) {
      PetscDeviceContext dctx;

      ierr = GetHandles_(&dctx);CHKERRQ(ierr);
      ierr = DeviceAllocateCheck_(dctx,w);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->restorelocalvector or v->ops->restorelocalvectorread
template <CUPMDeviceType T>
template <bool read>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorelocalvector_async(Vec v, Vec w))
{
  PetscErrorCode ierr;
  PetscBool      wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUPM(),&wisseqcupm);CHKERRQ(ierr);
  if (v->petscnative && wisseqcupm) {
    v->data          = w->data;
    v->offloadmask   = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr         = w->spptr;
    w->data          = nullptr;
    w->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
    w->spptr         = nullptr;
  } else {
    const auto array = &VecSeqCast_(w)->array;
    if (read) {
      ierr = VecRestoreArrayRead(v,const_cast<const decltype(array)>(array));CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(v,array);CHKERRQ(ierr);
    }
    if (w->spptr && wisseqcupm) {
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = GetHandles_(&stream);CHKERRQ(ierr);
      cerr = cupmFreeAsync(CUPMCast_(w)->device_array,stream);CHKERRCUPM(cerr);
      ierr = PetscFree(w->spptr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin))
{
  const auto         n = static_cast<cupmBlasInt_t>(yin->map->n);
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    const auto   nbytes = n*sizeof(DeviceArrayRead::value_type);
    cupmError_t  cerr;
    cupmStream_t stream;

    ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).ptr,DeviceArrayRead(dctx,xin).ptr,nbytes,cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  } else {
    const auto       alphaIsOne = alpha == PetscScalar(1.0);
    cupmBlasHandle_t cupmBlasHandle;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      cupmBlasError_t cberr;
      auto            yarray = DeviceArrayWrite(dctx,yin);
      auto            xarray = DeviceArrayRead(dctx,xin);

      if (alphaIsOne) {
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&alpha,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {
        constexpr PetscScalar sone = 1.0;

        cberr = cupmBlasXscal(cupmBlasHandle,n,&alpha,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&sone,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops((alphaIsOne ? 1 : 2)*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscScalar alpha, Vec xin))
{
  PetscErrorCode ierr;
  PetscBool      xiscupm;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) PetscFunctionReturn(0);
  ierr = PetscObjectTypeCompareAny(PetscObjectCast(xin),&xiscupm,VECSEQCUPM(),VECMPICUPM(),"");CHKERRQ(ierr);
  if (xiscupm) {
    const auto         n = static_cast<cupmBlasInt_t>(yin->map->n);
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXaxpy(cupmBlasHandle,n,&alpha,DeviceArrayRead(dctx,xin),1,DeviceArrayWrite(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseDivide_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    ierr = pointwisebinary_async_(thrust::divides<PetscScalar>(),win,xin,yin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisemult_async(Vec win, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseMult_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    ierr = pointwisebinary_async_(thrust::multiplies<PetscScalar>(),win,xin,yin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::reciprocal_async(Vec xin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    struct reciprocal
    {
      PETSC_HOSTDEVICE_DECL constexpr PetscScalar operator()(const PetscScalar& s) const
      {
        return s ? PetscScalar(1.0)/s : 0;
      }
    };

    auto xptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,xin).ptr);

    thrust::transform(xptr,xptr+n,xptr,reciprocal());
  } catch (const thrust::system_error& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscScalar alpha, Vec xin, Vec yin))
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

    ierr = GetHandles_(&dctx,&cupmBlasHandle,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      auto warray = DeviceArrayWrite(dctx,win);
      auto cerr   = cupmMemcpyAsync(warray.ptr,DeviceArrayRead(dctx,yin).ptr,n*sizeof(decltype(warray)::value_type),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
      auto cberr  = cupmBlasXaxpy(cupmBlasHandle,n,&alpha,DeviceArrayRead(dctx,xin),1,warray,1);CHKERRCUPMBLAS(cberr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::maxpy_async(Vec xin, PetscInt nv, const PetscScalar *alpha, Vec *y))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    auto xarray = DeviceArrayWrite(dctx,xin);

    for (decltype(nv) j = 0; j < nv; ++j) {
      auto cberr = cupmBlasXaxpy(cupmBlasHandle,n,alpha+j,DeviceArrayRead(dctx,y[j]),1,xarray,1);CHKERRCUPMBLAS(cberr);
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(nv*2*n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(nv*sizeof(*alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscScalar *z))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  // arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the
  // second
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdot(cupmBlasHandle,n,DeviceArrayRead(dctx,yin),1,DeviceArrayRead(dctx,xin),1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*(n-1),0));CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define MDOT_WORKGROUP_NUM  128
#define MDOT_WORKGROUP_SIZE MDOT_WORKGROUP_NUM

namespace kernels
{

PETSC_DEVICE_DECL static PetscInt entries_per_group(PetscInt size)
{
  const auto group_entries = (size-1)/(MDOT_WORKGROUP_SIZE+1);
  // for very small vectors, a group should still do some work
  return group_entries ? group_entries : 1;
}

template <int N>
PETSC_KERNEL_DECL static void mdot_kernel(const PetscScalar *PETSC_RESTRICT x, const PetscScalar *PETSC_RESTRICT y[PETSC_RESTRICT N], const PetscInt size, PetscScalar *PETSC_RESTRICT results)
{
  static_assert(N > 0,"");
  using iter_type = decltype(N);
  PETSC_SHAREDMEM_DECL PetscScalar shmem[N*MDOT_WORKGROUP_SIZE];
  const auto tx       = threadIdx.x,bx = blockIdx.x;
  const auto bdx      = blockDim.x,gdx = gridDim.x;
  const auto worksize = entries_per_group(size);
  const auto begin    = tx+bx*worksize;
  const auto end      = PetscMin((bx+1)*worksize,size);
  PetscScalar group_sum[N];

#pragma unroll
  for (auto i = iter_type(0); i < N; ++i) group_sum[i] = 0;

#pragma unroll
  for (auto i = begin; i < end; i += bdx) {
    const auto xi = x[i]; // load only once from global memory!

#pragma unroll
    for (auto j = iter_type(0); j < N; ++j) group_sum[j] += xi*y[j];
  }
#pragma unroll
  for (auto i = iter_type(0); i < N; ++i) shmem[tx+i*MDOT_WORKGROUP_SIZE] = group_sum[i];

  // parallel reduction
#pragma unroll
  for (auto stride = bdx/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
#pragma unroll
      for (auto i = iter_type(tx); i < N; i += MDOT_WORKGROUP_SIZE) shmem[i] += shmem[i+stride];
    }
  }
  // bottom N threads per block write to global memory
  if (tx < N) results[bx+tx*gdx] = shmem[tx*MDOT_WORKGROUP_SIZE];
  return;
}

} // namespace kernels

template <CUPMDeviceType T>
template <int N>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, const PetscScalar *xarr, const Vec yin[], PetscInt size, PetscScalar *results, PetscInt *yidx))
{
  static_assert(N > 0,"");
  using iter_type  = decltype(N);
  const auto yidxt = *yidx;
  const auto yint  = yin+yidxt;
  PetscScalar *device_y[N];
  cupmError_t cerr;

  PetscFunctionBegin;
  for (iter_type i = 0; i < N; ++i) device_y[i] = DeviceArrayRead(dctx,yint[i]);
  cerr = cupmLaunchKernel(kernels::mdot_kernel<N>,dim3(MDOT_WORKGROUP_NUM),dim3(MDOT_WORKGROUP_SIZE),0,stream,xarr,device_y,size,results+yidxt*MDOT_WORKGROUP_NUM);CHKERRCUPM(cerr);
  *yidx += N;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(detail::UseComplexTag<false>, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  const auto          n      = xin->map->n;
  const auto          nv1    = ((nv % 4) == 1) ? nv-1 : nv;
  const auto          nbytes = nv1*MDOT_WORKGROUP_NUM*sizeof(*VecSeqCast_(xin)->array);
  PetscScalar         *d_results;
  PetscDeviceContext  dctx;
  cupmStream_t        stream;
  cupmError_t         cerr;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
  // allocate scratchpad memory for the results of individual work groups
  cerr = cupmMallocAsync(reinterpret_cast<void**>(&d_results),nbytes,stream);CHKERRCUPM(cerr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    auto yidx = 0;
    auto xptr = DeviceArrayRead(dctx,xin);

    while (yidx < nv)
    {
      switch (nv-yidx) {
      case 7:
      case 6:
      case 5:
      case 4:
        ierr = mdot_kernel_dispatch_<4>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 3:
        ierr = mdot_kernel_dispatch_<3>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 2:
        ierr = mdot_kernel_dispatch_<2>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 1:
        ierr = mdot_kernel_dispatch_<1>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
      case 0:
        break;
      default: // 8 or more
        ierr = mdot_kernel_dispatch_<8>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
        break;
      }
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // copy results to CPU
  {
    std::array<PetscScalar,PETSC_MAX_PATH_LEN> stackarray;
    const auto allocate   = nv1*MDOT_WORKGROUP_NUM > stackarray.size();
    auto       h_results  = stackarray.data();

    if (allocate) {ierr = PetscMalloc1(nv1*MDOT_WORKGROUP_NUM,&h_results);CHKERRQ(ierr);}
    cerr = cupmMemcpyAsync(h_results,d_results,nbytes,cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
    // REVIEW ME: double count of flops??
    // do these now while memcpy is in flight
    ierr = PetscLogFlops(nv1*MDOT_WORKGROUP_NUM);CHKERRQ(ierr);
    ierr = PetscLogGpuToCpuScalar(nbytes);CHKERRQ(ierr);
    // for systems without async free this will synchronize implicitly
    cerr = cupmFreeAsync(d_results,stream);CHKERRCUPM(cerr);
    // REVIEW ME: need to hard sync here...
    ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
    // REVIEW ME: it is likely faster to do this in a micro kernel rather than do it on the
    // host which that requires synchronization
    // sum group results into z
    for (auto j = decltype(nv1)(0); j < nv1; ++j) {
      for (auto i = j*MDOT_WORKGROUP_NUM; i < (j+1)*MDOT_WORKGROUP_NUM; ++i) z[j] += h_results[i];
    }
    if (allocate) {ierr = PetscFree(h_results);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_NUM
#undef MDOT_WORKGROUP_SIZE

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(detail::UseComplexTag<true>, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    auto xptr = DeviceArrayRead(dctx,xin);

    for (decltype(nv) i = 0; i < nv; ++i) {
      auto cberr = cupmBlasXdot(cupmBlasHandle,n,DeviceArrayRead(dctx,yin+i),1,xptr,1,z+i);CHKERRCUPMBLAS(cberr);
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // REVIEW ME: flops?????
  ierr = PetscLogGpuToCpuScalar(nv*sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  using complex_tag = detail::UseComplexTag<PetscDefined(USE_COMPLEX)>;
  const auto     n = xin->map->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikely(nv <= 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUDA %" PetscInt_FMT " not positive.",nv);
  else if (PetscUnlikely(nv == 1)) {
    ierr = dot_async(xin,PetscRemoveConstCast(yin[0]),z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  // z will always need to be zeroed first, either for a quick return or for summing later on
  ierr = PetscArrayzero(z,nv);CHKERRQ(ierr);
  // nothing to do if x has no entries
  if (!n) PetscFunctionReturn(0);
  ierr = mdot_async_(complex_tag(),xin,nv,yin,z);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscScalar alpha))
{
  const auto         n = xin->map->n;
  PetscErrorCode     ierr;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0)) {
    const auto   nbytes = n*sizeof(DeviceArrayWrite::value_type);
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cupmMemsetAsync(DeviceArrayWrite(dctx,xin).ptr,0,nbytes,stream);CHKERRCUPM(cerr);
  } else {
    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      auto xptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,xin).ptr);

      thrust::fill(xptr,xptr+n,alpha);
    } catch (const thrust::system_error& ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::scale_async(Vec xin, PetscScalar alpha))
{
  PetscFunctionBegin;
  if (alpha == PetscScalar(1.0)) PetscFunctionReturn(0);
  else if (alpha == PetscScalar(0.0)) {
    auto ierr = set_async(xin,alpha);CHKERRQ(ierr);
  } else {
    const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscErrorCode     ierr;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXscal(cupmBlasHandle,n,&alpha,DeviceArrayWrite(dctx,xin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::tdot_async(Vec xin, Vec yin, PetscScalar *z))
{
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdotu(cupmBlasHandle,n,DeviceArrayRead(dctx,xin),1,DeviceArrayRead(dctx,yin),1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*n-1,0));CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin))
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = xin->map->n;
    const auto         nbytes = n*sizeof(*VecSeqCast_(xin)->array);
    auto               yiscupm = PETSC_TRUE,xondevice = PETSC_TRUE; // assume we start on device
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
      ierr = PetscObjectTypeCompareAny(PetscObjectCast(yin),&yiscupm,VECSEQCUPM(),VECMPICUPM(),"");CHKERRQ(ierr);
    case PETSC_OFFLOAD_GPU:
    case PETSC_OFFLOAD_BOTH:
      if (yiscupm) { // PETSC_TRUE by default (unless on the host)
        // even though y may be on the host, its a cupm vector, so it ought to be on the device
        mode = xondevice ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice;
      } else {
        // we assumed really wrong
        mode = xondevice ? cupmMemcpyDeviceToHost : cupmMemcpyHostToHost;
      }
      break;
    }

    ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
    switch (mode) {
    case cupmMemcpyDeviceToDevice:
      // the best case
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).ptr,DeviceArrayRead(dctx,xin).ptr,nbytes,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyHostToDevice:
      // not terrible
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).ptr,HostArrayRead(dctx,xin).ptr,nbytes,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyDeviceToHost: {
      // not great
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(yarray,DeviceArrayRead(dctx,xin).ptr,nbytes,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    case cupmMemcpyHostToHost:   {
      // the worst case
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      ierr = PetscArraycpy(yarray,HostArrayRead(dctx,xin),n);CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"Unknown cupmMemcpyKind %d",mode);
    }
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::swap_async(Vec xin, Vec yin))
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscErrorCode     ierr;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXswap(cupmBlasHandle,n,DeviceArrayWrite(dctx,xin),1,DeviceArrayWrite(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpby_async(Vec yin, PetscScalar alpha, PetscScalar beta, Vec xin))
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
    const auto         betaIsZero = beta == PetscScalar(0.0);
    const auto         n = static_cast<cupmBlasInt_t>(yin->map->n);
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      auto yarray = DeviceArrayWrite(dctx,yin);

      if (betaIsZero) {
        const auto   nbytes = n*sizeof(decltype(yarray)::value_type);
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
        cerr = cupmMemcpyAsync(yarray,DeviceArrayRead(dctx,xin).ptr,nbytes,cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
        cberr = cupmBlasXscal(cupmBlasHandle,n,&alpha,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {
        cberr = cupmBlasXscal(cupmBlasHandle,n,&beta,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&alpha,DeviceArrayRead(dctx,xin),1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops((betaIsZero ? 1 : 3)*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar((betaIsZero ? 1 : 2)*sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpbypcz_async(Vec zin, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec xin, Vec yin))
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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::norm_async(Vec xin, NormType type, PetscReal *z))
{
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscInt           flopCount = 0;
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
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    cupmBlasError_t cberr;
    auto            xarray = DeviceArrayRead(dctx,xin);

    switch (type) {
    case NORM_1_AND_2:
    case NORM_1:
      cberr = cupmBlasXasum(cupmBlasHandle,n,xarray,1,z);CHKERRCUPMBLAS(cberr);
      flopCount = PetscMax(n-1,0);
      if (type != NORM_1_AND_2) break;
      ++z; // fall-through
    case NORM_2:
    case NORM_FROBENIUS:
      cberr = cupmBlasXnrm2(cupmBlasHandle,n,xarray,1,z);CHKERRCUPMBLAS(cberr);
      flopCount += PetscMax(2*n-1,0); // +=  in case we've fallen through from NORM_1_AND_2
      break;
    case NORM_INFINITY: {
      cupmError_t  cerr;
      cupmStream_t stream;
      PetscScalar  zs;
      int          i;

      // REVIEW ME: this needs to be redone by hand
      cberr = cupmBlasXamax(cupmBlasHandle,n,xarray,1,&i);CHKERRCUPMBLAS(cberr);
      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(&zs,xarray.ptr+i-1,sizeof(zs),cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
      *z   = PetscAbsScalar(zs);
      // REVIEW ME: flopCount = ???
    } break;
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flopCount);CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dotnorm2_async(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = dot_async(s,t,dp);CHKERRQ(ierr);
  ierr = dot_async(t,t,nm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (auto vcu  = CUPMCast_(v)) {
    switch (vcu->ptr_ownership) {
    case PETSC_COPY_VALUES:
    case PETSC_OWN_POINTER:
      if (PetscDefined(HAVE_NVSHMEM) && vcu->nvshmem) {
        ierr = PetscNvshmemFree(vcu->device_array);CHKERRQ(ierr);
      } else {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = GetHandles_(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      }
    case PETSC_USE_POINTER:
      break;
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if PetscDefined(USE_LOG)
  ierr = PetscLogObjectState(PetscObjectCast(v),"Length=%" PetscInt_FMT,v->map->n);CHKERRQ(ierr);
#endif
  if (auto vseq = VecSeqCast_(v)) {
    if (vseq->array_allocated) {
      const auto useit = UseCUPMHostAlloc(v->pinned_memory);

      ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
      if (useit.value()) v->pinned_memory = PETSC_FALSE;
    }
    ierr = PetscFree(v->data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

namespace detail
{

struct conjugate
{
  PETSC_DEVICE_DECL constexpr PetscScalar operator()(PetscScalar x) const { return PetscConj(x); }
};

} // namespace detail

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::conjugate_async(Vec v))
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    const auto         n = v->map->n;
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      auto xptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,v).ptr);

      thrust::transform(xptr,xptr+n,xptr,detail::conjugate());
    } catch (const thrust::system_error& ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    // REVIEW ME: also at least n?
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

namespace detail
{

struct real_part
{
  PETSC_DEVICE_DECL
  thrust::tuple<PetscReal,PetscInt> operator()(const thrust::tuple<PetscScalar,PetscInt>& x) const
  {
    return thrust::make_tuple(PetscRealPart(x.get<0>()),x.get<1>());
  }

  PETSC_DEVICE_DECL
  constexpr PetscReal operator()(const PetscScalar& x) const { return PetscRealPart(x); }
};

} // namespace detail

template <CUPMDeviceType T>
template <typename TupleFuncT, typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::minmax_async_(TupleFuncT&& tuple_functor, UnaryFuncT&& unary_functor, PetscReal initval, Vec v, PetscInt *p, PetscReal *m))
{
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if (!n) {
    *m = initval;
    if (p) *p = -1;
    PetscFunctionReturn(0);
  }
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto vptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,v).ptr);

    if (p) {
      const auto init  = thrust::make_tuple(initval,PetscInt(-1));
      auto       zibit = thrust::make_zip_iterator(
        thrust::make_tuple(vptr,thrust::make_counting_iterator(PetscInt(0)))
      );

      if (PetscDefined(USE_COMPLEX)) {
        thrust::tie(*m,*p) = thrust::transform_reduce(
          zibit,zibit+n,detail::real_part(),init,std::forward<TupleFuncT>(tuple_functor)
        );
      } else {
        thrust::tie(*m,*p) = thrust::reduce(
          zibit,zibit+n,init,std::forward<TupleFuncT>(tuple_functor)
        );
      }
    } else {
      if (PetscDefined(USE_COMPLEX)) {
        *m = thrust::transform_reduce(
          vptr,vptr+n,detail::real_part(),initval,std::forward<UnaryFuncT>(unary_functor)
        );
      } else {
        *m = thrust::reduce(vptr,vptr+n,initval,std::forward<UnaryFuncT>(unary_functor));
      }
    }
  } catch (const thrust::system_error& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // REVIEW ME: flops?
  PetscFunctionReturn(0);
}

namespace detail
{

struct max_tuple
{
  using tuple_type = thrust::tuple<PetscReal,PetscInt>;

  PETSC_DEVICE_DECL tuple_type operator()(const tuple_type& x, const tuple_type& y) const
  {
    if ((x.get<0>() > y.get<0>()) || (x.get<1>() <  y.get<1>())) {
      return thrust::make_tuple(x.get<0>(),x.get<1>());
    } else {
      return thrust::make_tuple(y.get<0>(),y.get<1>());
    }
  }
};

} // namespace detail

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::max_async(Vec v, PetscInt *p, PetscReal *m))
{
  using tuple_functor = detail::max_tuple;
  using unary_functor = thrust::maximum<util::remove_pointer_t<decltype(m)>>;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = minmax_async_(tuple_functor(),unary_functor(),PETSC_MIN_REAL,v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

namespace detail
{

struct min_tuple
{
  using tuple_type = thrust::tuple<PetscReal,PetscInt>;

  PETSC_DEVICE_DECL tuple_type operator()(const tuple_type& x, const tuple_type& y) const
  {
    if ((x.get<0>() < y.get<0>()) || (x.get<1>() < y.get<1>())) {
      return thrust::make_tuple(x.get<0>(),x.get<1>());
    } else {
      return thrust::make_tuple(y.get<0>(),y.get<1>());
    }
  }
};

} // namespace detail
template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::min_async(Vec v, PetscInt *p, PetscReal *m))
{
  using tuple_functor = detail::min_tuple;
  using unary_functor = thrust::minimum<util::remove_pointer_t<decltype(m)>>;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = minmax_async_(tuple_functor(),unary_functor(),PETSC_MAX_REAL,v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::sum_async(Vec v, PetscScalar *sum))
{
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto dptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,v).ptr);

    *sum = thrust::reduce(dptr,dptr+n,PetscScalar(0.0));
  } catch (const thrust::system_error& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // REVIEW ME: must be at least n additions
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

namespace detail
{

struct shifter
{
  const PetscScalar s;

  PETSC_HOSTDEVICE_DECL constexpr PetscScalar operator()(PetscScalar x) const { return x+s; }
};

} // namespace detail

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::shift_async(Vec v, PetscScalar shift))
{
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto dptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,v).ptr);

    thrust::transform(dptr,dptr+n,dptr,detail::shifter{shift}); /* in-place transform */
  } catch (const thrust::system_error& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::setrandom_async(Vec v, PetscRandom rand))
{
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  PetscBool          iscurand;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare(PetscObjectCast(rand),PETSCCURAND,&iscurand);CHKERRQ(ierr);
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  if (iscurand) {
    ierr = PetscRandomGetValues(rand,n,DeviceArrayWrite(dctx,v));CHKERRQ(ierr);
  } else {
    ierr = PetscRandomGetValues(rand,n,HostArrayWrite(dctx,v));CHKERRQ(ierr);
  }
  // REVIEW ME: flops????
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (v->boundtocpu) PetscFunctionReturn(0);
  v->boundtocpu = usehost;
  // default random number generator
  ierr = PetscFree(v->defaultrandtype);CHKERRQ(ierr);
  if (usehost) {
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
    ierr = PetscStrallocpy(PETSCRANDER48,&v->defaultrandtype);CHKERRQ(ierr);
  } else {
    // REVIEW ME: hip seemingly has no equivalent?
    ierr = PetscStrallocpy(PETSCCURAND,&v->defaultrandtype);CHKERRQ(ierr);
  }
  v->ops->dot                    = v->ops->dot_local   = usehost ? VecDot_Seq   : dot_async;
  v->ops->norm                   = v->ops->norm_local  = usehost ? VecNorm_Seq  : norm_async;
  v->ops->tdot                   = v->ops->tdot_local  = usehost ? VecTDot_Seq  : tdot_async;
  v->ops->mdot                   = v->ops->mdot_local  = usehost ? VecMDot_Seq  : mdot_async;
  v->ops->mtdot                  = v->ops->mtdot_local = usehost ? VecMTDot_Seq : nullptr;
  v->ops->scale                  = usehost ? VecScale_Seq : scale_async;
  v->ops->copy                   = usehost ? VecCopy_Seq : copy_async;
  v->ops->set                    = usehost ? VecSet_Seq : set_async;
  v->ops->swap                   = usehost ? VecSwap_Seq : swap_async;
  v->ops->axpy                   = usehost ? VecAXPY_Seq : axpy_async;
  v->ops->axpby                  = usehost ? VecAXPBY_Seq : axpby_async;
  v->ops->axpbypcz               = usehost ? VecAXPBYPCZ_Seq : axpbypcz_async;
  v->ops->pointwisemult          = usehost ? VecPointwiseMult_Seq : pointwisemult_async;
  v->ops->pointwisedivide        = usehost ? VecPointwiseDivide_Seq : pointwisedivide_async;
  v->ops->setrandom              = usehost ? VecSetRandom_Seq : setrandom_async;
  v->ops->maxpy                  = usehost ? VecMAXPY_Seq : maxpy_async;
  v->ops->aypx                   = usehost ? VecAYPX_Seq : aypx_async;
  v->ops->waxpy                  = usehost ? VecWAXPY_Seq : waxpy_async;
  v->ops->dotnorm2               = usehost ? nullptr : dotnorm2_async;
  v->ops->conjugate              = usehost ? VecConjugate_Seq : conjugate_async;
  v->ops->max                    = usehost ? VecMax_Seq : max_async;
  v->ops->min                    = usehost ? VecMin_Seq : min_async;
  v->ops->reciprocal             = usehost ? VecReciprocal_Default : reciprocal_async;
  v->ops->sum                    = usehost ? nullptr : sum_async;
  v->ops->shift                  = usehost ? nullptr : shift_async;

  v->ops->placearray             = usehost ? VecPlaceArray_Seq : placearray_async<PETSC_MEMTYPE_HOST>;
  v->ops->replacearray           = replacearray_async<PETSC_MEMTYPE_HOST>;
  v->ops->resetarray             = usehost ? VecResetArray_Seq : resetarray_async<PETSC_MEMTYPE_HOST>;

  v->ops->duplicate              = usehost ? VecDuplicate_Seq : duplicate_async;
  v->ops->getlocalvector         = usehost ? nullptr : getlocalvector_async</*read = */false>;
  v->ops->getlocalvectorread     = usehost ? nullptr : getlocalvector_async</*read = */true>;
  v->ops->restorelocalvector     = usehost ? nullptr : restorelocalvector_async</*read = */false>;
  v->ops->restorelocalvectorread = usehost ? nullptr : restorelocalvector_async</*read = */true>;

  v->ops->bindtocpu              = bindtocpu_async;
  v->ops->destroy                = destroy_async;
  v->ops->getarray               = getarray_async<PETSC_MEMTYPE_HOST,MemoryAccess::READ_WRITE>;
  v->ops->restorearray           = restorearray_async<PETSC_MEMTYPE_HOST,MemoryAccess::READ_WRITE>;
  v->ops->getarraywrite          = usehost ? nullptr : getarray_async<PETSC_MEMTYPE_HOST,MemoryAccess::WRITE>;
  v->ops->restorearraywrite      = usehost ? nullptr : restorearray_async<PETSC_MEMTYPE_HOST,MemoryAccess::WRITE>;
  v->ops->getarrayread           = usehost ? nullptr : getarray_async<PETSC_MEMTYPE_HOST,MemoryAccess::READ>;
  v->ops->restorearrayread       = usehost ? nullptr : restorearray_async<PETSC_MEMTYPE_HOST,MemoryAccess::READ>;
  v->ops->getarrayandmemtype     = getarrayandmemtype_async;
  v->ops->restorearrayandmemtype = restorearrayandmemtype_async;
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
