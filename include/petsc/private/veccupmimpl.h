#ifndef PETSCVECCUPMIMPL_H
#define PETSCVECCUPMIMPL_H

#include <petsc/private/vecimpl.h>
#include <petsc/private/cupmblasinterface.hpp>

#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode VecCreate_CUDA(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqCUDA(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPICUDA(Vec);
PETSC_INTERN PetscErrorCode VecCUDAGetArrays_Private(Vec,const PetscScalar**,const PetscScalar**, PetscOffloadMask*);
#endif

#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode VecCreate_HIP(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIHIP(Vec);
PETSC_INTERN PetscErrorCode VecHIPGetArrays_Private(Vec,const PetscScalar**,const PetscScalar**, PetscOffloadMask*);
#endif

#if PetscDefined(HAVE_NVSHMEM)
PETSC_INTERN PetscErrorCode PetscNvshmemInitializeCheck(void);
PETSC_INTERN PetscErrorCode PetscNvshmemMalloc(size_t,void**);
PETSC_INTERN PetscErrorCode PetscNvshmemCalloc(size_t,void**);
PETSC_INTERN PetscErrorCode PetscNvshmemFree_Private(void*);
#define PetscNvshmemFree(ptr) ((ptr) && (PetscNvshmemFree_Private(ptr),(ptr)=PETSC_NULLPTR,0))
PETSC_INTERN PetscErrorCode PetscNvshmemSum(PetscInt,PetscScalar*,const PetscScalar*);
PETSC_INTERN PetscErrorCode PetscNvshmemMax(PetscInt,PetscReal*,const PetscReal*);
PETSC_INTERN PetscErrorCode VecNormAsync_NVSHMEM(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecAllocateNVSHMEM_SeqCUDA(Vec);
#else
#  define PetscNvshmemFree(ptr) 0
#endif

#if defined(__cplusplus)

#if (__cplusplus < 201103L) || !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#  error "Vec_CUPM requires C++11"
#endif

#include <limits>  // std::numeric_limits
#include <cstring> // std::memset

namespace Petsc
{

// REVIEW ME: using just Vec leads to compiler confusing Vec the type and Vec the namespace,
// what do?
namespace Vector
{

namespace CUPM
{

namespace Impl
{

// a simple RAII helper for PetscMallocSet[CUDA|HIP]Host(). it exists because integrating the
// regular versions would be an enormous pain to square with the templated types...
template <Device::CUPM::DeviceType T>
struct UseCUPMHostAlloc_ : Device::CUPM::Impl::Interface<T>
{
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type,T);
private:
  // would have loved to just do
  //
  // const auto oldmalloc = PetscTrMalloc;
  //
  // but in order to use auto the member needs to be static; in order to be static it must
  // also be constexpr -- which in turn requires an initializer (also implicitly required by
  // auto). But constexpr needs a constant expression initializer, so we can't initialize it
  // with global (mutable) variables...
#define DECLTYPE_AUTO(left,right) decltype(right) left = right
  const DECLTYPE_AUTO(oldmalloc_,PetscTrMalloc);
  const DECLTYPE_AUTO(oldfree_,PetscTrFree);
  const DECLTYPE_AUTO(oldrealloc_,PetscTrRealloc);
#undef DECLTYPE_AUTO
  const bool v_;

public:
  UseCUPMHostAlloc_(bool useit) noexcept : v_(useit)
  {
    if (useit) {
      // all unused arguments are un-named, this saves having to add PETSC_UNUSED to them all
      PetscTrMalloc  = [](size_t sz,PetscBool clear,int,const char*,const char*,void **ptr)
      {
        PetscFunctionBegin;
        CHKERRCUPM(cupmMallocHost(ptr,sz));
        if (clear) std::memset(*ptr,0,sz);
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
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"%s has no realloc()",cupmName());
      };
    }
  }

  PETSC_NODISCARD auto value() const PETSC_DECLTYPE_NOEXCEPT_RETURNS(v_);

  ~UseCUPMHostAlloc_() noexcept
  {
    if (v_) {
      PetscTrMalloc  = oldmalloc_;
      PetscTrFree    = oldfree_;
      PetscTrRealloc = oldrealloc_;
    }
  }
};

enum class MemoryAccess : unsigned {
  READ       = 1 << 0,
  WRITE      = 1 << 1,
  READ_WRITE = READ | WRITE
};

namespace
{

struct no_op
{
  template <typename...T> constexpr PetscErrorCode operator()(T&&...) const noexcept { return 0; }
};

} // anonymous namespace

// Base class for the VecSeq and VecMPI CUPM implementations. On top of the usual DeviceType
// template parameter it also uses CRTP to be able to use values/calls specific to either
// VecSeq or VecMPI. This is in effect "inside-out" polymorphism.
template <Device::CUPM::DeviceType T, typename Derived>
struct Vec_CUPMBase : Device::CUPM::Impl::BlasInterface<T>
{
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t,T);

private:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMAllocateCheck_(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode VecIMPLAllocateCheck_(Vec));
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

public:
  struct Vec_CUPM
  {
    PetscScalar   *device_array; // gpu data
    PetscCopyMode ptr_ownership; // does PETSc own the array ptr?
    PetscBool     nvshmem;       // is array allocated in nvshmem? It is used to allocate
                                 // Mvctx->lvec in nvshmem
  };

  PETSC_CXX_COMPAT_DECL(PetscErrorCode VecView_Debug(Vec v, const char* message = ""))
  {
    static_assert(PETSC_OFFLOAD_UNALLOCATED == 0,"");
    static_assert(PETSC_OFFLOAD_CPU == 1,"");
    static_assert(PETSC_OFFLOAD_GPU == 2,"");
    static_assert(PETSC_OFFLOAD_BOTH == 3,"");
    const char* PetscOffloadMasks[] = {
      "OFFLOAD_UNALLOCATED",
      "OFFLOAD_CPU",
      "OFFLOAD_GPU",
      "OFFLOAD_BOTH",
    };
    const auto     pobj  = PetscObjectCast(v);
    const auto     vimpl = VecIMPLCast(v);
    const auto     vcu   = VecCUPMCast(v);
    PetscBool      device_mem;
    MPI_Comm       comm;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidPointer(vimpl,1);
    PetscValidPointer(vcu,1);
    ierr = PetscObjectGetComm(pobj,&comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"---------- %s ----------\n",message);CHKERRQ(ierr);
    ierr = PetscObjectPrintClassNamePrefixType(pobj,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Address:             %p\n",v);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Size:                %" PetscInt_FMT "\n",v->map->n);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Offload mask:        %s\n",PetscOffloadMasks[v->offloadmask]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Host ptr:            %p\n",vimpl->array);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Device ptr:          %p\n",vcu->device_array);CHKERRQ(ierr);
    ierr = cupmIsDeviceMemory(vcu->device_array,&device_mem);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"dptr is device mem?  %s\n",device_mem ? "yes" : "no");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Device ptr ownership %s\n",PetscCopyModes[VecCUPMCast(v)->ptr_ownership]);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr auto VecCUPMCast(Vec v)) PETSC_DECLTYPE_AUTO_RETURNS(static_cast<Vec_CUPM*>(v->spptr));
  // This is a trick to get around the fact that in CRTP the derived class is not yet fully
  // defined because Base<Derived> must necessarily be instantiated before Derived is
  // complete. By using a dummy template parameter we make the type "dependent" and so will
  // only be determined when the derived class is instantiated (and therefore fully defined)
  template <typename U = Derived>
  PETSC_CXX_COMPAT_DECL(constexpr auto VecIMPLCast(Vec v)) PETSC_DECLTYPE_AUTO_RETURNS(U::VecIMPLCast_(v));
  template <typename U = Derived>
  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 auto VECTYPE()) PETSC_DECLTYPE_AUTO_RETURNS(U::VECTYPE_());

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 PetscLogEvent VEC_CUPMCopyToGPU())
  {
    switch (T) {
    case Device::CUPM::DeviceType::CUDA: return VEC_CUDACopyToGPU;
    case Device::CUPM::DeviceType::HIP:  return VEC_HIPCopyToGPU;
    }
    PetscUnreachable();
    return PETSC_LARGEST_EVENT;
  }

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 PetscLogEvent VEC_CUPMCopyFromGPU())
  {
    switch (T) {
    case Device::CUPM::DeviceType::CUDA: return VEC_CUDACopyFromGPU;
    case Device::CUPM::DeviceType::HIP:  return VEC_HIPCopyFromGPU;
    }
    PetscUnreachable();
    return PETSC_LARGEST_EVENT;
  }

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 VecType VECSEQCUPM())
  {
    switch (T) {
    case Device::CUPM::DeviceType::CUDA: return VECSEQCUDA;
    case Device::CUPM::DeviceType::HIP:  return VECSEQHIP;
    }
    PetscUnreachable();
    return "invalid";
  }

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 VecType VECMPICUPM())
  {
    switch (T) {
    case Device::CUPM::DeviceType::CUDA: return VECMPICUDA;
    case Device::CUPM::DeviceType::HIP:  return VECMPIHIP;
    }
    PetscUnreachable();
    return "invalid";
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode CUPMBlasIntCast_(PetscInt x, cupmBlasInt_t *y))
  {
    using petsc_type = decltype(x);
    using blas_type  = util::remove_pointer_t<decltype(y)>;

    PetscFunctionBegin;
    if (!std::is_same<petsc_type,blas_type>::value) {
      PetscCheck(x <= std::numeric_limits<blas_type>::max(),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"%" PetscInt_FMT " is too big for %s, which may be restricted to 32 bit integers",x,cupmBlasName());
    }
    PetscCheck(x >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Passing negative integer to %s routine: %" PetscInt_FMT,cupmBlasName(),x);
    *y = static_cast<blas_type>(x);
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetHandles_(PetscDeviceContext *dctx, cupmBlasHandle_t *handle = nullptr, cupmStream_t *stream = nullptr))
  {
    return GetHandleDispatch_(dctx,handle,stream);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetHandles_(PetscDeviceContext *dctx, cupmStream_t *stream))
  {
    return GetHandles_(dctx,nullptr,stream); // other overload
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetHandles_(cupmStream_t *stream))
  {
    return GetHandles_(nullptr,nullptr,stream); // other overload
  }

  // RAII versions of the get/restore array routines. Determines constness of the pointer type,
  // holds the pointer itself provides the implicit conversion operator
  template <PetscMemType MT, MemoryAccess MA, typename ValueType = PetscScalar>
  struct vector_array
  {
    static const auto memory_type = MT;
    static const auto access_type = MA;

    using value_type        = ValueType;
    // PetscScalar*
    using pointer_type      = util::add_pointer_t<value_type>;
    // cupmScalar_t*
    using cupm_pointer_type = util::add_pointer_t<cupmScalar_t>;

    // PetscScalar *const
    const pointer_type ptr;

    operator pointer_type() const noexcept { return const_cast<pointer_type>(this->ptr); }

    // in case pointer_type == cupmscalar_pointer_type we don't want this overload to exist, so
    // we make a dummy template parameter to allow SFINAE to nix it for us
    template <
      typename U = pointer_type,
      typename = util::enable_if_t<!std::is_same<U,cupm_pointer_type>::value>
    >
    operator cupm_pointer_type() const noexcept
    {
      return cupmScalarCast(const_cast<pointer_type>(this->ptr));
    }

    vector_array(PetscDeviceContext, Vec v) noexcept : ptr(initialize_(v)), v_(v) { }

    ~vector_array() noexcept
    {
      // REVIEW ME: could just as well CHKERRABORT() here
      PetscFunctionBegin;
      CHKERRCONTINUE((restorearray_async<MT,MA>(PetscRemoveConstCast(v_),nullptr)));
      PetscFunctionReturnVoid();
    }

  private:
    const Vec v_;

    PETSC_CXX_COMPAT_DECL(pointer_type initialize_(Vec v))
    {
      pointer_type array;

      PetscFunctionBegin;
      CHKERRABORT(PETSC_COMM_SELF,(getarray_async<MT,MA>(v,&array)));
      PetscFunctionReturn(array);
    }
  };

  // data movement
  PETSC_CXX_COMPAT_DECL(PetscErrorCode HostAllocateCheck_(PetscDeviceContext,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode DeviceAllocateCheck_(PetscDeviceContext,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode CopyToDevice_(PetscDeviceContext,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode CopyToHost_(PetscDeviceContext,Vec));

  // accessors
  template <PetscMemType,MemoryAccess>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarray_async(Vec,PetscScalar**));
  template <PetscMemType,MemoryAccess>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorearray_async(Vec,PetscScalar**));
  template <MemoryAccess>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarrayandmemtype_async(Vec,PetscScalar**,PetscMemType*));
  template <MemoryAccess>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorearrayandmemtype_async(Vec,PetscScalar**));

  // common ops shared between Seq and MPI
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Create_CUPMBase_(MPI_Comm,PetscInt,PetscInt,PetscInt,Vec*,PetscBool,PetscLayout/*reference*/=nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Initialize_CUPMBase_(Vec,PetscBool,PetscScalar*,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Destroy_CUPMBase_(Vec));
  template <typename SetupFunctionT = no_op>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode Duplicate_CUPMBase_(Vec,Vec*,SetupFunctionT&& = SetupFunctionT()));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode BindToCPU_CUPMBase_(Vec,PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode GetArrays_CUPMBase_(Vec,const PetscScalar**,const PetscScalar**,PetscOffloadMask*));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode Create_CUPM_(Vec));

  PETSC_CXX_COMPAT_DECL(auto UseCUPMHostAlloc(bool b))
  PETSC_DECLTYPE_AUTO_RETURNS(UseCUPMHostAlloc_<T>(b));
  PETSC_CXX_COMPAT_DECL(auto UseCUPMHostAlloc(PetscBool b))
  PETSC_DECLTYPE_AUTO_RETURNS(UseCUPMHostAlloc(static_cast<bool>(b)));
  PETSC_CXX_COMPAT_DECL(auto UseCUPMHostAlloc(Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(UseCUPMHostAlloc(v->pinned_memory == PETSC_TRUE));

  // need functions to create the vector arrays, otherwise usng them as an unnamed temporary
  // leads to most vexing parse
  PETSC_CXX_COMPAT_DECL(auto DeviceArrayRead(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ>{dctx,v});
  PETSC_CXX_COMPAT_DECL(auto DeviceArrayWrite(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_DEVICE,MemoryAccess::WRITE>{dctx,v});
  PETSC_CXX_COMPAT_DECL(auto DeviceArrayReadWrite(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ_WRITE>{dctx,v});
  PETSC_CXX_COMPAT_DECL(auto HostArrayRead(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_HOST,MemoryAccess::READ>{dctx,v});
  PETSC_CXX_COMPAT_DECL(auto HostArrayWrite(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_HOST,MemoryAccess::WRITE>{dctx,v});
  PETSC_CXX_COMPAT_DECL(auto HostArrayReadWrite(PetscDeviceContext dctx, Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(vector_array<PETSC_MEMTYPE_HOST,MemoryAccess::READ_WRITE>{dctx,v});
};

template <Device::CUPM::DeviceType T, typename D>
template <PetscMemType MT, MemoryAccess MA, typename VT>
const PetscMemType Vec_CUPMBase<T,D>::vector_array<MT,MA,VT>::memory_type;

template <Device::CUPM::DeviceType T, typename D>
template <PetscMemType MT, MemoryAccess MA, typename VT>
const MemoryAccess Vec_CUPMBase<T,D>::vector_array<MT,MA,VT>::access_type;

PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMCheckMinimumPinnedMemory_Internal(Vec v))
{
  auto           mem = static_cast<PetscInt>(v->minimum_bytes_pinned_memory);
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin(PetscObjectCast(v));CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",mem,&mem,&flg,0,std::numeric_limits<decltype(mem)>::max());CHKERRQ(ierr);
  if (flg) v->minimum_bytes_pinned_memory = mem;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::VecIMPLAllocateCheck_(Vec v))
{
  auto           vimpl = VecIMPLCast(v);
  cupmBlasInt_t  bn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLikely(vimpl)) PetscFunctionReturn(0);
  ierr = PetscNewLog(PetscObjectCast(v),&vimpl);CHKERRQ(ierr);
  v->data = vimpl;
  // do a cast to blasint check because if blasint cant hold the size, then any subsequent
  // cupmblas calls can't use it either. Doing this now this means we don't have to check
  // during every function
  ierr = CUPMBlasIntCast_(v->map->n,&bn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::HostAllocateCheck_(PetscDeviceContext, Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecIMPLAllocateCheck_(v);CHKERRQ(ierr);
  const auto vimpl = VecIMPLCast(v);
  if (PetscLikely(vimpl->array_allocated)) PetscFunctionReturn(0);
  else {
    const auto n      = v->map->n;
    const auto nbytes = n*sizeof(*vimpl->array_allocated);

    ierr = VecCUPMCheckMinimumPinnedMemory_Internal(v);CHKERRQ(ierr);
    {
      const auto useit = UseCUPMHostAlloc(nbytes > v->minimum_bytes_pinned_memory);

      v->pinned_memory = static_cast<decltype(v->pinned_memory)>(useit.value());
      ierr = PetscMalloc1(n,&vimpl->array_allocated);CHKERRQ(ierr);
    }
    ierr = PetscLogObjectMemory(PetscObjectCast(v),nbytes);CHKERRQ(ierr);
    if (!vimpl->array) vimpl->array = vimpl->array_allocated;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) v->offloadmask = PETSC_OFFLOAD_CPU;
  }
  PetscFunctionReturn(0);
}

// allocate the Vec_CUPM struct. this is normally done through DeviceAllocateCheck_(), but in
// certain circumstances (such as when the user places the device array) we do not want to do
// the full DeviceAllocateCheck_() as it also allocates the array
template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::VecCUPMAllocateCheck_(Vec v))
{
  auto           vcu = VecCUPMCast(v);
  cupmBlasInt_t  bn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLikely(vcu)) PetscFunctionReturn(0);
  ierr = PetscNewLog(PetscObjectCast(v),&vcu);CHKERRQ(ierr);
  v->spptr = vcu;
  // do a cast to blasint check because if blasint cant hold the size, then any subsequent
  // cupmblas calls can't use it either. Doing this now this means we don't have to check
  // during every function
  ierr = CUPMBlasIntCast_(v->map->n,&bn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::DeviceAllocateCheck_(PetscDeviceContext dctx, Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUPMAllocateCheck_(v);CHKERRQ(ierr);
  const auto vcu = VecCUPMCast(v);
  if (PetscLikely(vcu->device_array)) PetscFunctionReturn(0);
  else {
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
    cerr = cupmMallocAsync(reinterpret_cast<void**>(&vcu->device_array),v->map->n*sizeof(*vcu->device_array),stream);CHKERRCUPM(cerr);
    vcu->ptr_ownership = PETSC_OWN_POINTER;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      const auto vimp = VecIMPLCast(v);
      v->offloadmask = (vimp && vimp->array) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
    }
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::CopyToDevice_(PetscDeviceContext dctx, Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DeviceAllocateCheck_(dctx,v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    const auto   xfersize = v->map->n*sizeof(*VecIMPLCast(v)->array);
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(VEC_CUPMCopyToGPU(),v,0,0,0);CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(VecCUPMCast(v)->device_array,VecIMPLCast(v)->array,xfersize,cupmMemcpyHostToDevice,stream);CHKERRCUPM(cerr);
    ierr = PetscLogEventEnd(VEC_CUPMCopyToGPU(),v,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(xfersize);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::CopyToHost_(PetscDeviceContext dctx, Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = HostAllocateCheck_(dctx,v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    const auto   xfersize = v->map->n*sizeof(*VecCUPMCast(v)->device_array);
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(VEC_CUPMCopyFromGPU(),v,0,0,0);CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(VecIMPLCast(v)->array,VecCUPMCast(v)->device_array,xfersize,cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
    ierr = PetscLogEventEnd(VEC_CUPMCopyFromGPU(),v,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogGpuToCpu(xfersize);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

#define STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype) \
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE),"")

// v->ops->getarray[read|write] or VecCUPMGetArray[Read|Write]()
template <Device::CUPM::DeviceType T, typename D>
template <PetscMemType mtype, MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::getarray_async(Vec v, PetscScalar **a))
{
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  constexpr auto     hostmem = PetscMemTypeHost(mtype);
  // silence buggy gcc warning: "dctx" may be used uninitialized in this function
  PetscDeviceContext dctx = nullptr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  if (access == MemoryAccess::WRITE) {
    ierr = (hostmem ? HostAllocateCheck_ : DeviceAllocateCheck_)(dctx,v);CHKERRQ(ierr);
  } else {
    // READ or READ_WRITE
    if (hostmem) {
      ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
      // REVIEW ME: if we have a clearly defined async api then this sync is not necessary!
      // Otherwise we have to assume every pointer returned to host memory is immediately
      // dereferenced by the host, and must therefore hard-sync every time...
      ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
    } else {ierr = CopyToDevice_(dctx,v);CHKERRQ(ierr);}
  }
  if (access != MemoryAccess::READ) {
    // not read-only so immediately assume modified
    // REVIEW ME: this should probably also call PetscObjectStateIncrease() since we assume it
    // is immediately modified
    v->offloadmask = hostmem ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  *a = hostmem ? VecIMPLCast(v)->array : VecCUPMCast(v)->device_array;
  PetscFunctionReturn(0);
}

// v->ops->restorearray[read|write] or VecCUPMRestoreArray[Read|Write]()
template <Device::CUPM::DeviceType T, typename D>
template <PetscMemType mtype, MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::restorearray_async(Vec v, PetscScalar **a))
{
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if (access != MemoryAccess::READ) {
    // WRITE or READ_WRITE
    auto ierr = PetscObjectStateIncrease(PetscObjectCast(v));CHKERRQ(ierr);
    v->offloadmask = PetscMemTypeHost(mtype) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  if (a) *a = nullptr;
  PetscFunctionReturn(0);
}

// v->ops->getarrayandmemtype
template <Device::CUPM::DeviceType T, typename D>
template <MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::getarrayandmemtype_async(Vec v, PetscScalar **a, PetscMemType *mtype))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = getarray_async<PETSC_MEMTYPE_DEVICE,access>(v,a);CHKERRQ(ierr);
  if (mtype) *mtype = (PetscDefined(HAVE_NVSHMEM) && VecCUPMCast(v)->nvshmem) ? PETSC_MEMTYPE_NVSHMEM : cupmDeviceTypeToPetscMemType();
  PetscFunctionReturn(0);
}

// v->ops->restorearrayandmemtype
template <Device::CUPM::DeviceType T, typename D>
template <MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::restorearrayandmemtype_async(Vec v, PetscScalar **a))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = restorearray_async<PETSC_MEMTYPE_DEVICE,access>(v,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                      Common core between Seq and MPI                               //

// VecCreateCUPM()
template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::Create_CUPMBase_(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, Vec *v, PetscBool call_set_type, PetscLayout reference))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  if (reference) {ierr = PetscLayoutReference(reference,&(*v)->map);CHKERRQ(ierr);}
  ierr = VecSetSizes(*v,n,N);CHKERRQ(ierr);
  if (bs) {ierr = VecSetBlockSize(*v,bs);CHKERRQ(ierr);}
  if (call_set_type) {ierr = VecSetType(*v,VECTYPE());CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

// VecCreateIMPL_CUPM()
template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::Initialize_CUPMBase_(Vec v, PetscBool allocate_missing, PetscScalar *host_array, PetscScalar *device_array))
{
  // silence buggy gcc warning: "dctx" may be used uninitialized in this function
  PetscDeviceContext dctx = nullptr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceInitialize(cupmDeviceTypeToPetscDeviceType());CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName(PetscObjectCast(v),VECTYPE());CHKERRQ(ierr);
  ierr = D::bindtocpu_async(v,PETSC_FALSE);CHKERRQ(ierr);
  if (device_array) {
    // array is being placed from the user
    ierr = VecCUPMAllocateCheck_(v);CHKERRQ(ierr);
    VecCUPMCast(v)->device_array  = device_array;
    VecCUPMCast(v)->ptr_ownership = PETSC_USE_POINTER;
  }
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  if (host_array) {
    ierr = HostAllocateCheck_(dctx,v);CHKERRQ(ierr);
    VecIMPLCast(v)->array = host_array;
  }
  if (allocate_missing) {
    ierr = DeviceAllocateCheck_(dctx,v);CHKERRQ(ierr);
    ierr = HostAllocateCheck_(dctx,v);CHKERRQ(ierr);
    // calls device-version
    ierr = VecSet(v,0);CHKERRQ(ierr);
    // zero the host while device is underway
    ierr = PetscArrayzero(VecIMPLCast(v)->array,v->map->n);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  } else {
    if (host_array) {
      v->offloadmask = device_array ? PETSC_OFFLOAD_BOTH : PETSC_OFFLOAD_CPU;
    } else {
      v->offloadmask = device_array ? PETSC_OFFLOAD_GPU : PETSC_OFFLOAD_UNALLOCATED;
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::Destroy_CUPMBase_(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (auto vcu = VecCUPMCast(v)) {
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
  if (const auto vimpl = VecIMPLCast(v)) {
    const auto useit = UseCUPMHostAlloc(v);

    // do this ourselves since we may want to use the cupm functions
    ierr = PetscFree(vimpl->array_allocated);CHKERRQ(ierr);
  }
  v->pinned_memory = PETSC_FALSE;
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <Device::CUPM::DeviceType T, typename D>
template <typename SetupFunctionT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::Duplicate_CUPMBase_(Vec v, Vec *y, SetupFunctionT&& DerivedCreateIMPLCUPM_Async))
{
  // if the derived setup is the default no_op then we should call VecSetType()
  constexpr auto call_set_type = static_cast<PetscBool>(std::is_same<SetupFunctionT,no_op>::value);
  const auto     vobj = PetscObjectCast(v);
  const auto     map  = v->map;
  PetscInt       bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetBlockSize(v,&bs);CHKERRQ(ierr);
  ierr = Create_CUPMBase_(PetscObjectComm(vobj),bs,map->n,map->N,y,call_set_type,map);CHKERRQ(ierr);
  // Derived class can set up the remainder of the data structures here
  ierr = DerivedCreateIMPLCUPM_Async(*y);CHKERRQ(ierr);
  // in case the user has done some VecSetOps() tomfoolery
  ierr = PetscMemcpy((*y)->ops,v->ops,sizeof(*v->ops));CHKERRQ(ierr);
  {
    const auto yobj = PetscObjectCast(*y);
    ierr = PetscObjectListDuplicate(vobj->olist,&yobj->olist);CHKERRQ(ierr);
    ierr = PetscFunctionListDuplicate(vobj->qlist,&yobj->qlist);CHKERRQ(ierr);
  }
  (*y)->stash.donotstash   = v->stash.donotstash;
  (*y)->stash.ignorenegidx = v->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

#define VecSetOp_CUPM(op_name,op_host,...) do { \
    if (usehost) {                              \
      v->ops->op_name = op_host;                \
    } else {                                    \
      v->ops->op_name = __VA_ARGS__;            \
    }                                           \
  } while (0)

// v->ops->duplicate
template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::BindToCPU_CUPMBase_(Vec v, PetscBool usehost))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  v->boundtocpu = usehost;
  /* default random number generator */
  ierr = PetscFree(v->defaultrandtype);CHKERRQ(ierr);
  if (usehost) {
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
    ierr = PetscStrallocpy(PETSCRANDER48,&v->defaultrandtype);CHKERRQ(ierr);
  } else {
    // REVIEW ME: HIP default rng?
    ierr = PetscStrallocpy(PETSCCURAND,&v->defaultrandtype);CHKERRQ(ierr);
  }
  // set the base functions that are guaranteed to be the same for both
  v->ops->duplicate = D::duplicate_async;
  v->ops->create    = D::create_async;
  v->ops->destroy   = D::destroy_async;
  v->ops->bindtocpu = D::bindtocpu_async;
  VecSetOp_CUPM(getarray,nullptr,&getarray_async<PETSC_MEMTYPE_HOST,MemoryAccess::READ_WRITE>);
  VecSetOp_CUPM(restorearray,nullptr,&restorearray_async<PETSC_MEMTYPE_HOST,MemoryAccess::READ_WRITE>);
  VecSetOp_CUPM(getarraywrite,nullptr,&getarray_async<PETSC_MEMTYPE_HOST,MemoryAccess::WRITE>);
  VecSetOp_CUPM(restorearraywrite,nullptr,&restorearray_async<PETSC_MEMTYPE_HOST,MemoryAccess::WRITE>);
  // can't do ternary operator here because:
  // 1. signature of getarray/restorearray have PetscScalar** not const PetscScalar** -> use a
  //    lambda to convert the types
  // 2. each lambda is a unique type, which isn't convertible to std::nullptr_t
  VecSetOp_CUPM(getarrayread,nullptr,[](Vec v, const PetscScalar **a)
  {
    return getarray_async<PETSC_MEMTYPE_HOST,MemoryAccess::READ>(v,const_cast<PetscScalar**>(a));
  });
  VecSetOp_CUPM(restorearrayread,nullptr,[](Vec v, const PetscScalar **a)
  {
    return restorearray_async<PETSC_MEMTYPE_HOST,MemoryAccess::READ>(v,const_cast<PetscScalar**>(a));
  });
  VecSetOp_CUPM(getarrayandmemtype,nullptr,&getarrayandmemtype_async<MemoryAccess::READ_WRITE>);
  VecSetOp_CUPM(restorearrayandmemtype,nullptr,&restorearrayandmemtype_async<MemoryAccess::READ_WRITE>);
  VecSetOp_CUPM(getarraywriteandmemtype,nullptr,&getarrayandmemtype_async<MemoryAccess::WRITE>);
  VecSetOp_CUPM(restorearraywriteandmemtype,nullptr,[](Vec v,PetscScalar **a,PetscMemType*)
  {
    return restorearrayandmemtype_async<MemoryAccess::WRITE>(v,a);
  });
  PetscFunctionReturn(0);
}

// Called from VecGetSubVector()
template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::GetArrays_CUPMBase_(Vec v, const PetscScalar **host_array, const PetscScalar **device_array, PetscOffloadMask *mask))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  if (host_array)   {
    ierr = HostAllocateCheck_(dctx,v);CHKERRQ(ierr);
    *host_array = VecIMPLCast(v)->array;
  }
  if (device_array) {
    ierr = DeviceAllocateCheck_(dctx,v);CHKERRQ(ierr);
    *device_array = VecCUPMCast(v)->device_array;
  }
  if (mask) *mask = v->offloadmask;
  PetscFunctionReturn(0);
}

// VecCreate_CUPM()
template <Device::CUPM::DeviceType T, typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode Vec_CUPMBase<T,D>::Create_CUPM_(Vec v))
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)),&size);CHKERRMPI(ierr);
  ierr = VecSetType(v,size > 1 ? VECMPICUPM() : VECSEQCUPM());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define PETSC_VEC_CUPM_BASE_CLASS_HEADER(name,Tp,Derived)                       \
  using name = Petsc::Vector::CUPM::Impl::Vec_CUPMBase<Tp,Derived>;             \
  friend name;                                                                  \
  /* introspection */                                                           \
  using name::VecCUPMCast;                                                      \
  using name::VecIMPLCast;                                                      \
  using name::VECTYPE;                                                          \
  using name::VECSEQCUPM;                                                       \
  using name::VECMPICUPM;                                                       \
  /* utility */                                                                 \
  using name::Vec_CUPM;                                                         \
  using name::VecView_Debug;                                                    \
  using name::vector_array;                                                     \
  using name::UseCUPMHostAlloc;                                                 \
  using name::GetHandles_;                                                      \
  using name::HostAllocateCheck_;                                               \
  using name::DeviceAllocateCheck_;                                             \
  using name::CopyToDevice_;                                                    \
  using name::CopyToHost_;                                                      \
  using name::getarray_async;                                                   \
  using name::restorearray_async;                                               \
  using name::getarrayandmemtype_async;                                         \
  using name::restorearrayandmemtype_async;                                     \
  /* base functions */                                                          \
  using name::Create_CUPMBase_;                                                 \
  using name::Initialize_CUPMBase_;                                             \
  using name::Destroy_CUPMBase_;                                                \
  using name::Duplicate_CUPMBase_;                                              \
  using name::BindToCPU_CUPMBase_;                                              \
  using name::Create_CUPM_;                                                     \
  using name::DeviceArrayRead;                                                  \
  using name::DeviceArrayWrite;                                                 \
  using name::DeviceArrayReadWrite;                                             \
  using name::HostArrayRead;                                                    \
  using name::HostArrayWrite;                                                   \
  using name::HostArrayReadWrite;                                               \
  /* blas interface */                                                          \
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t,Tp)

} // namespace Impl

} // namespace CUPM

} // namespace Vector

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCVECCUPMIMPL_H
