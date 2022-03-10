#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include "objpool.hpp"

#include <cstring> // std::memset
#include <type_traits> // std::is_trivially_copyable

template <typename T>
struct PetscManagedTypeAllocator : Petsc::AllocatorBase<T>
{
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(T *mscal))
  {
    PetscFunctionBegin;
    PetscCall(PetscNew(mscal));
    PetscCall(reset(*mscal,false));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(T mscal))
  {
    PetscFunctionBegin;
    PetscCall(PetscFree(mscal));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset(T mscal, bool zero = true))
  {
    using Petsc::util::integral_value;

    PetscFunctionBegin;
    if (zero) {
      using Petsc::util::remove_pointer_t;
      static_assert(std::is_trivially_copyable<remove_pointer_t<T>>::value,"");
      PetscCallCXX(std::memset(mscal,0,sizeof(*mscal)));
    }
    mscal->h_cmode = PETSC_OWN_POINTER;
    mscal->d_cmode = PETSC_OWN_POINTER;
    static_assert(integral_value(PETSC_OWN_POINTER)         != 0,"");
    static_assert(integral_value(PETSC_DEVICE_HOST)         == 0,"");
    static_assert(integral_value(PETSC_OFFLOAD_UNALLOCATED) == 0,"");
    static_assert(integral_value(PETSC_FALSE)               == 0,"");
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) { return 0; }
};

template <typename PT, typename MT>
class PetscManagedTypeImpl
{
  using PetscType   = PT;
  using ManagedType = MT;
  using PoolType    = Petsc::ObjectPool<ManagedType,PetscManagedTypeAllocator<ManagedType>>;
  static PoolType pool;

  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy_values(PetscDeviceContext,ManagedType,PetscOffloadMask,PetscOffloadMask,const PetscType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode check_lock(ManagedType,bool = false));
  PETSC_CXX_COMPAT_DECL(bool           managed_type_equal(const ManagedType&,const ManagedType&));

public:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscCopyMode,PetscOffloadMask,ManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext,ManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode get_values(PetscDeviceContext,ManagedType,PetscMemType,PetscMemoryAccessMode,PetscBool,PetscType**));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_values(PetscDeviceContext,ManagedType,PetscMemType,const PetscType*,PetscInt));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode get_pointer_and_mem_type(PetscDeviceContext,ManagedType,PetscMemoryAccessMode,PetscType**,PetscMemType* = nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy(PetscDeviceContext,ManagedType,ManagedType));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode apply_operator(PetscDeviceContext,ManagedType,PetscOperatorType,PetscMemType,const PetscType*,ManagedType=nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode get_sub_range(PetscDeviceContext,ManagedType,PetscInt,PetscInt,ManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restore_sub_range(PetscDeviceContext,ManagedType,ManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode query(ManagedType,PetscType,PetscBool*,PetscBool*));
};

template <typename T, typename MT>
typename PetscManagedTypeImpl<T,MT>::PoolType PetscManagedTypeImpl<T,MT>::pool;

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::copy_values(PetscDeviceContext dctx, ManagedType scal, PetscOffloadMask mask, PetscOffloadMask src_mask, const PetscType *src_ptr))
{
  PetscDeviceCopyMode  mode;
  PetscType           *ptr;

  // need to actually allocate the stuff
  PetscFunctionBegin;
  PetscCall(get_values(dctx,scal,PetscOffloadMaskToMemType(mask),PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&ptr));
  PetscCall(PetscOffloadMaskToDeviceCopyMode(mask,src_mask,&mode));
  PetscCall(PetscDeviceArrayCopy(dctx,ptr,src_ptr,scal->n,mode));
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::check_lock(ManagedType scal, bool v))
{
  const auto  lock      = scal->state.locked;
  const auto  val       = static_cast<decltype(lock)>(v);
  const char *strings[] = {"unlocked","locked"};

  PetscFunctionBegin;
  PetscAssert(lock == val,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Managed type object is %s expected it to be %s",strings[lock],strings[val]);
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(bool PetscManagedTypeImpl<T,MT>::managed_type_equal(const ManagedType& l, const ManagedType& r))
{
  return (l->host == r->host) && (l->device == r->device) && (l->dtype == r->dtype) &&
    (l->mask == r->mask) && (l->d_cmode == r->d_cmode) && (l->h_cmode == r->h_cmode) &&
    (l->n == r->n) && (l->state.locked == r->state.locked) && (l->state.tainted == r->state.tainted);
}

template <typename T = void>
static PetscErrorCode destroy_managed_type_fn(PetscDeviceContext,...) noexcept
{
  static_assert(!std::is_same<T,T>::value,"This template should never be called");
  return PETSC_ERR_PLIB;
}

template <typename T = void>
static PetscErrorCode get_managed_values_fn(PetscDeviceContext,...) noexcept
{
  static_assert(!std::is_same<T,T>::value,"This template should never be called");
  return PETSC_ERR_PLIB;
}

template <typename T = void>
static PetscErrorCode apply_operator_fn(PetscDeviceContext,...) noexcept
{
  static_assert(!std::is_same<T,T>::value,"This template should never be called");
  return PETSC_ERR_PLIB;
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::create(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask, ManagedType *scal))
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (host_ptr && n) PetscValidPointer(host_ptr,2);
  PetscValidPointer(scal,8);
  if (host_ptr && device_ptr) {
    PetscAssert(mask != PETSC_OFFLOAD_UNALLOCATED,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Set both host and device pointer but offloadmask was PETSC_OFFLOAD_UNALLOCATED");
    // this is the only instance in which we believe whatever the user has fed us
  } else if (host_ptr) {
    // clearly no device_ptr, so we own it
    mask         = PETSC_OFFLOAD_CPU;
    device_cmode = PETSC_OWN_POINTER;
  } else if (device_ptr) {
    // clearly no host_ptr, so we own it
    mask       = PETSC_OFFLOAD_GPU;
    host_cmode = PETSC_OWN_POINTER;
  } else {
    mask       = PETSC_OFFLOAD_UNALLOCATED;
    host_cmode = device_cmode = PETSC_OWN_POINTER;
  }

  // finally get our pointer
  PetscCall(pool.get(*scal));

  // populate known quantities
  (*scal)->n       = n;
  (*scal)->mask    = mask;
  (*scal)->h_cmode = host_cmode;
  (*scal)->d_cmode = device_cmode;

  if (host_cmode == PETSC_COPY_VALUES) {
    PetscCall(copy_values(dctx,*scal,mask,PETSC_OFFLOAD_CPU,host_ptr));
  } else {
    // own_pointer or use_pointer
    (*scal)->host = host_ptr;
  }

  if (device_cmode == PETSC_COPY_VALUES) {
    PetscCall(copy_values(dctx,*scal,mask,PETSC_OFFLOAD_GPU,device_ptr));
  } else {
    // own_pointer or use_pointer
    (*scal)->device = device_ptr;
  }
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::destroy(PetscDeviceContext dctx, ManagedType *scal))
{
  PetscFunctionBegin;
  PetscValidPointer(scal,2);
  if (!*scal) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(destroy_managed_type_fn(dctx,*scal));
  // if the host pointer still exists at this point it is because it didn't belong to its
  // respective memory pool. If copy mode is PETSC_OWN_POINTER its because we have
  // co-opted the users pointer, so we should free it now.
  if ((*scal)->host && ((*scal)->h_cmode == PETSC_OWN_POINTER)) PetscCall(PetscFree((*scal)->host));
  // cannot handle device pointers though
  PetscAssert(!(*scal)->device || ((*scal)->d_cmode != PETSC_OWN_POINTER),PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContext (id %" PetscInt_FMT ", device type %s) failed to free the owned device pointer",dctx->id,PetscDeviceTypes[dctx->device->type]);
  PetscCall(pool.reclaim(std::move(*scal)));
  *scal = nullptr;
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::get_values(PetscDeviceContext dctx, ManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, PetscType **ptr))
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(scal,2);
  PetscCall(check_lock(scal));
  PetscValidPointer(ptr,6);
  PetscCall(get_managed_values_fn(dctx,scal,mtype,mode,ptr));
  // if user intends to write to device in any capacity then we are tainted
  if (PetscMemTypeDevice(mtype) && (mode != PETSC_MEMORY_ACCESS_READ)) scal->state.tainted = 1;
  if (sync) {
    if (PetscMemTypeHost(mtype)) scal->state.tainted = 0; // only way to reset this state
    PetscCall(PetscDeviceContextSynchronize(dctx));
  }
  PetscAssert(*ptr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Returned null pointer for mtype %d",mtype);
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::set_values(PetscDeviceContext dctx, ManagedType scal, PetscMemType mtype, const PetscType *ptr, PetscInt n))
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscValidPointer(ptr,4);
  PetscValidPointer(scal,2);
  PetscCall(check_lock(scal));
  if (n) {
    PetscMemType  scalmtype;
    PetscType    *scalptr;

    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscCall(get_pointer_and_mem_type(dctx,scal,PETSC_MEMORY_ACCESS_WRITE,&scalptr,&scalmtype));
    PetscAssert(n <= scal->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to write %" PetscInt_FMT " values to " PetscStringize(PetscManagedType) " but it only holds %" PetscInt_FMT " entries",n,scal->n);
    PetscCall(PetscDeviceArrayCopy(dctx,scalptr,ptr,n,PetscMemTypeToDeviceCopyMode(scalmtype,mtype)));
  }
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::get_pointer_and_mem_type(PetscDeviceContext dctx, ManagedType scal, PetscMemoryAccessMode mode, PetscType **ptr, PetscMemType *mtype))
{
  const auto assign = [&](PetscType *ptr_arg, PetscMemType mtype_arg)
  {
    *ptr = ptr_arg;
    if (mtype) *mtype = mtype_arg;
  };

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(check_lock(scal));
  PetscValidPointer(ptr,4);
  if (mtype) PetscValidPointer(mtype,5);
  switch (scal->mask) {
  case PETSC_OFFLOAD_CPU:
  UNALLOCATED_PREFER_HOST:
    assign(scal->host,PETSC_MEMTYPE_HOST);
    break;
  case PETSC_OFFLOAD_BOTH:
  case PETSC_OFFLOAD_GPU:
  UNALLOCATED_PREFER_DEVICE:
    assign(scal->device,PETSC_MEMTYPE_DEVICE);
    break;
  case PETSC_OFFLOAD_UNALLOCATED: {
    const auto prefer_host = dctx->device->type == PETSC_DEVICE_HOST;
    PetscCall(get_values(dctx,scal,prefer_host ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE,mode,PETSC_FALSE,ptr));
    if (prefer_host) {
      goto UNALLOCATED_PREFER_HOST;
    } else {
      goto UNALLOCATED_PREFER_DEVICE;
    }
  } break;
  case PETSC_OFFLOAD_KOKKOS:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for PETSC_OFFLOAD_KOKKOS");
  }
  PetscAssert(*ptr,PETSC_COMM_SELF,PETSC_ERR_PLIB,PetscStringize(PetscManagedType) " returned a null pointer for memtype %s as values",mtype ? (PetscMemTypeHost(*mtype) ? "host" : "device") : "unknown");
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::copy(PetscDeviceContext dctx, ManagedType dest, ManagedType src))
{
  // initialize to silence overzealous gcc, at least we can use auto now
  auto       dest_mtype = PETSC_MEMTYPE_DEVICE,src_mtype = PETSC_MEMTYPE_DEVICE;
  PetscType *dest_ptr,*src_ptr;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(dest,2);
  PetscValidPointer(src,3);
  PetscCheckCompatibleDeviceTypes(dest->dtype,2,src->dtype,3);
  PetscAssert(dest->n >= src->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Destination size %" PetscInt_FMT " not large enough for source size %" PetscInt_FMT,dest->n,src->n);
  PetscCall(get_pointer_and_mem_type(dctx,dest,PETSC_MEMORY_ACCESS_WRITE,&dest_ptr,&dest_mtype));
  PetscCall(get_pointer_and_mem_type(dctx,src,PETSC_MEMORY_ACCESS_READ,&src_ptr,&src_mtype));
  PetscCall(PetscDeviceArrayCopy(dctx,dest_ptr,src_ptr,dest->n,PetscMemTypeToDeviceCopyMode(dest_mtype,src_mtype)));
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::apply_operator(PetscDeviceContext dctx, ManagedType scal, PetscOperatorType otype, PetscMemType mtype, const PetscType *rhs, ManagedType ret))
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(scal,2);
  PetscCall(check_lock(scal));
  if (PetscMemTypeHost(mtype)) PetscValidPointer(rhs,5);
  if (!scal->state.tainted && (PetscOffloadHost(scal->mask) || PetscOffloadUnallocated(scal->mask)) && PetscMemTypeHost(mtype)) {
    const auto  src_access = ret ? PETSC_MEMORY_ACCESS_READ : PETSC_MEMORY_ACCESS_READ_WRITE;
    const auto  rhsv       = *rhs;
    const auto  n          = scal->n;
    PetscType  *ptr,*retptr;

    PetscCall(get_values(dctx,scal,PETSC_MEMTYPE_HOST,src_access,PETSC_TRUE,&ptr));
    if (ret) {
      PetscCall(get_values(dctx,ret,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,PETSC_TRUE,&retptr));
    } else {
      // in place
      retptr = ptr;
    }

    for (PetscInt i = 0; i < n; ++i) {
      switch (otype) {
      case PETSC_OPERATOR_PLUS:     retptr[i] = ptr[i]+rhsv; break;
      case PETSC_OPERATOR_MINUS:    retptr[i] = ptr[i]-rhsv; break;
      case PETSC_OPERATOR_MULTIPLY: retptr[i] = ptr[i]*rhsv; break;
      case PETSC_OPERATOR_DIVIDE:   retptr[i] = ptr[i]/rhsv; break;
      case PETSC_OPERATOR_EQUAL:    retptr[i] = rhsv;        break;
      }
    }
  } else {
    PetscCall(apply_operator_fn(dctx,scal,otype,mtype,rhs,ret));
  }
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::get_sub_range(PetscDeviceContext dctx, ManagedType in, PetscInt begin, PetscInt len, ManagedType *out))
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(in,2);
  PetscValidPointer(out,5);
  PetscCall(check_lock(in));
  PetscAssert(len > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot extract a subrange of negative size %" PetscInt_FMT,len);
  PetscAssert(begin+len <= in->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Trying to extract a subrange of [%" PetscInt_FMT ",%" PetscInt_FMT ") from managed type of size %" PetscInt_FMT,begin,begin+len,in->n);
  if (len == in->n) {
    // curious case of trying to extract a subrange that is exactly the size of the current
    // object, in which case we simply return ourselves and don't need to lock
    *out = in;
  } else {
    auto host_begin   = in->host;
    auto device_begin = in->device;
    const auto UpdatePointer = [=](PetscMemType mtype, PetscType **ptr)
    {
      PetscFunctionBegin;
      PetscCall(get_values(dctx,in,mtype,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_FALSE,ptr));
      PetscFunctionReturn(0);
    };

    if (!host_begin) PetscCall(UpdatePointer(PETSC_MEMTYPE_HOST,&host_begin));
    host_begin += begin;
    if (!device_begin && in->dtype != PETSC_DEVICE_HOST) {
      PetscCall(UpdatePointer(PETSC_MEMTYPE_DEVICE,&device_begin));
      // can't do this unconditionally since device may be NULL, in which case it is UB to
      // perform arithmetic on NULL pointers
      device_begin += begin;
    }
    PetscCall(create(dctx,host_begin,device_begin,len,PETSC_USE_POINTER,PETSC_USE_POINTER,in->mask,out));
    // copy state over to the subrange
    (*out)->state = in->state;
    // but lock ourselves
    in->state.locked = 1;
  }
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::restore_sub_range(PetscDeviceContext dctx, ManagedType in, ManagedType *out))
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(in,2);
  PetscValidPointer(out,3);
  PetscValidPointer(*out,3);
  if (in == *out) {
    // case where the subrange len was the same as the original (and hence we returned
    // ourselves). do a sanity check -- which to be honest I don't see how it could fail -- to
    // double check
    PetscAssert(managed_type_equal(in,*out),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Subrange '" PetscStringize(*out) "' being restored is pointer equavalent to '" PetscStringize(in) "' but not value equivalent");
    *out = nullptr;
    PetscFunctionReturn(0);
  }
  PetscCall(check_lock(in,true)); // assert that we are locked
  PetscCall(check_lock(*out));    // the restored obj can't also have an outstanding subrange
  if (PetscDefined(USE_DEBUG)) {
    const auto check_ownership = [&](const PetscType *begin, const PetscType *needle)
    {
      const auto end        = std::next(begin,static_cast<std::size_t>(in->n));
      const auto needle_end = std::next(needle,static_cast<std::size_t>((*out)->n));

      PetscFunctionBegin;
      PetscCheck((begin <= needle) && (needle < end),PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sub-range does not belong to input managed type, pointer %p not in range [%p, %p)",needle,begin,end);
      PetscCheck(needle_end <= end,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sub-range does not belong to input managed type, sub [%p,%p) not in range [%p, %p)",needle,needle_end,begin,end);
      PetscFunctionReturn(0);
    };
    PetscCheck((*out)->n <= in->n,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sub-range does not appear to belong to input managed type, sub length %" PetscInt_FMT " > input length %" PetscInt_FMT,(*out)->n,in->n);
    PetscCall(check_ownership(in->host,(*out)->host));
    PetscCall(check_ownership(in->device,(*out)->device));
  }
  // update our state (which includes unlocking since check_lock(*out) passed)
  in->state = (*out)->state;
  PetscCall(destroy(dctx,out));
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::query(ManagedType scal, PetscType val, PetscBool *known, PetscBool *equal))
{
  PetscFunctionBegin;
  PetscValidPointer(scal,1);
  PetscValidBoolPointer(known,3);
  PetscValidBoolPointer(equal,4);
  *equal = PETSC_FALSE; // assume not equal
  {
    const auto& state = scal->state;

    if (state.locked) {
      // if we are locked then the subrange could be doing any number of things to the data
      // right now
      *known = PETSC_FALSE;
    } else if (scal->mask == PETSC_OFFLOAD_UNALLOCATED) {
      // unallocated? clearly not "equal" to anything
      *known = PETSC_TRUE;
    } else {
      *known = state.tainted ? PETSC_FALSE : PETSC_TRUE;
      if (const auto host = scal->host) {
        const auto n     = scal->n;
        auto       eqcnt = 0;

        for (auto i = 0; i < n; ++i) eqcnt += val == host[i];
        *equal = (PetscBool)(eqcnt == n);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Scalar
#define PetscTypeSuffix_L scalar
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Real
#define PetscTypeSuffix_L real
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Int
#define PetscTypeSuffix_L int
#include "managedtype.inl"
