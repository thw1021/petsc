#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include "objpool.hpp"

template <typename T>
struct PetscManagedTypeAllocator : Petsc::AllocatorBase<T>
{
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(T *mscal))
  {
    using Petsc::util::integral_value;

    PetscFunctionBegin;
    PetscCall(PetscNew(mscal));
    (*mscal)->h_cmode = PETSC_OWN_POINTER;
    (*mscal)->d_cmode = PETSC_OWN_POINTER;
    static_assert(integral_value(PETSC_OWN_POINTER)         != 0,"");
    static_assert(integral_value(PETSC_DEVICE_HOST)         == 0,"");
    static_assert(integral_value(PETSC_OFFLOAD_UNALLOCATED) == 0,"");
    static_assert(integral_value(PETSC_FALSE)               == 0,"");
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(T mscal))
  {
    PetscFunctionBegin;
    PetscCall(PetscFree(mscal));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset(T mscal))
  {
    PetscFunctionBegin;
    PetscCallCXX(std::memset(mscal,0,sizeof(*mscal)));
    mscal->h_cmode = PETSC_OWN_POINTER;
    mscal->d_cmode = PETSC_OWN_POINTER;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) { return 0; }
};

template <typename PT, typename MT>
struct PetscManagedTypeImpl
{
  using PetscType   = PT;
  using ManagedType = MT;
  using PoolType    = Petsc::ObjectPool<ManagedType,PetscManagedTypeAllocator<ManagedType>>;
  static PoolType pool;

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscCopyMode,PetscOffloadMask,ManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext,ManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode get_values(PetscDeviceContext,ManagedType,PetscMemType,PetscMemoryAccessMode,PetscType**,PetscInt* = nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_values(PetscDeviceContext,ManagedType,PetscMemType,const PetscType*,PetscInt));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode get_pointer_and_mem_type(PetscDeviceContext,ManagedType,PetscMemoryAccessMode,PetscType**,PetscMemType* = nullptr,PetscInt* = nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode ensure_offload(PetscDeviceContext,ManagedType,PetscOffloadMask));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy(PetscDeviceContext,ManagedType,ManagedType));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode apply_operator(PetscDeviceContext,ManagedType,PetscOperatorType,PetscMemType,const PetscType*,ManagedType=nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode get_sub_range(PetscDeviceContext,ManagedType,PetscInt,PetscInt,ManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restore_sub_range(PetscDeviceContext,ManagedType,ManagedType*));

private:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy_values(PetscDeviceContext,ManagedType,PetscOffloadMask,PetscOffloadMask,const PetscType*));
};

template <typename T, typename MT>
typename PetscManagedTypeImpl<T,MT>::PoolType PetscManagedTypeImpl<T,MT>::pool;

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::copy_values(PetscDeviceContext dctx, ManagedType scal, PetscOffloadMask mask, PetscOffloadMask src_mask, const PetscType *src_ptr))
{
  PetscDeviceCopyMode  mode;
  PetscType           *ptr;
  PetscInt             n;

  // need to actually allocate the stuff
  PetscFunctionBegin;
  PetscCall(get_values(dctx,scal,PetscOffloadMaskToMemType(mask),PETSC_MEMORY_ACCESS_WRITE,&ptr,&n));
  PetscCall(PetscOffloadMaskToDeviceCopyMode(mask,src_mask,&mode));
  PetscCall(PetscDeviceArrayCopy(dctx,ptr,src_ptr,n,mode));
  PetscFunctionReturn(0);
}

template <typename T>
auto destroy_managed_type_fn(PetscDeviceContext) noexcept -> PetscErrorCode(*)(PetscDeviceContext,T)
{
  PetscUnreachable();
  return nullptr;
}

template <typename MT, typename T>
auto get_managed_values_fn(PetscDeviceContext) noexcept -> PetscErrorCode(*)(PetscDeviceContext,MT,PetscMemType,PetscMemoryAccessMode,T**)
{
  PetscUnreachable();
  return nullptr;
}

template <typename MT, typename T>
auto apply_operator_fn(PetscDeviceContext) noexcept -> PetscErrorCode(*)(PetscDeviceContext,MT,PetscOperatorType,const T*,MT)
{
  PetscUnreachable();
  return nullptr;
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
  if (!*scal) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall((*destroy_managed_type_fn<ManagedType>(dctx))(dctx,*scal));
  PetscCall(pool.reclaim(std::move(*scal)));
  *scal = nullptr;
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::get_values(PetscDeviceContext dctx, ManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr, PetscInt *n))
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(scal,2);
  PetscValidPointer(ptr,5);
  if (n) {
    PetscValidIntPointer(n,6);
    *n = scal->n;
  }
  PetscAssert(!scal->locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Object is locked, perhaps a subrange was not yet restored?");
  PetscCall((*get_managed_values_fn<ManagedType,T>(dctx))(dctx,scal,mtype,mode,ptr));
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::set_values(PetscDeviceContext dctx, ManagedType scal, PetscMemType mtype, const PetscType *ptr, PetscInt n))
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscValidPointer(ptr,4);
  PetscAssert(!scal->locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Object is locked, perhaps a subrange was not yet restored?");
  if (n) {
    PetscMemType  scalmtype;
    PetscType    *scalptr;
    PetscInt      scaln;

    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscCall(get_pointer_and_mem_type(dctx,scal,PETSC_MEMORY_ACCESS_WRITE,&scalptr,&scalmtype,&scaln));
    PetscAssert(n <= scaln,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to write %" PetscInt_FMT " values to " PetscStringize(PetscManagedType) " but it only holds %" PetscInt_FMT " entries",n,scaln);
    PetscCall(PetscDeviceArrayCopy(dctx,scalptr,ptr,n,PetscMemTypeToDeviceCopyMode(scalmtype,mtype)));
  }
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::get_pointer_and_mem_type(PetscDeviceContext dctx, ManagedType scal, PetscMemoryAccessMode mode, PetscType **ptr, PetscMemType *mtype, PetscInt *n))
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssert(!scal->locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Object is locked, perhaps a subrange was not yet restored?");
  PetscValidPointer(ptr,4);
  if (mtype) PetscValidPointer(mtype,5);
  if (n) {
    PetscValidIntPointer(n,6);
    *n = scal->n;
  }
  switch (scal->mask) {
  case PETSC_OFFLOAD_CPU:
  UNALLOCATED_PREFER_HOST:
    *ptr = scal->host;
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
    break;
  case PETSC_OFFLOAD_BOTH:
  case PETSC_OFFLOAD_GPU:
  UNALLOCATED_PREFER_DEVICE:
    *ptr = scal->device;
    if (mtype) *mtype = PETSC_MEMTYPE_DEVICE;
    break;
  case PETSC_OFFLOAD_UNALLOCATED: {
    const auto prefer_host = dctx->device->type == PETSC_DEVICE_HOST;
    PetscCall(get_values(dctx,scal,prefer_host ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE,mode,ptr,nullptr));
    if (prefer_host) {
      goto UNALLOCATED_PREFER_HOST;
    } else {
      goto UNALLOCATED_PREFER_DEVICE;
    }
  } break;
  }
  PetscAssert(*ptr,PETSC_COMM_SELF,PETSC_ERR_PLIB,PetscStringize(PetscManagedType) " returned a null pointer for memtype %s as values",mtype ? (PetscMemTypeHost(*mtype) ? "host" : "device") : "unknown");
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::ensure_offload(PetscDeviceContext dctx, ManagedType scal, PetscOffloadMask omask))
{
  PetscFunctionBegin;
  PetscValidPointer(scal,2);
  if (scal->mask != omask) {
    PetscType *ptr;

    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    if (PetscOffloadDevice(omask) || omask == PETSC_OFFLOAD_BOTH) {
      PetscCall(get_values(dctx,scal,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,&ptr));
    }
    if (PetscOffloadHost(omask) || omask == PETSC_OFFLOAD_BOTH) {
      PetscCall(get_values(dctx,scal,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&ptr));
    }
  }
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::copy(PetscDeviceContext dctx, ManagedType dest, ManagedType src))
{
  PetscMemType  dmtype,smtype;
  PetscInt      dest_n,src_n;
  PetscType    *dest_ptr,*src_ptr;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(dest,2);
  PetscValidPointer(src,3);
  PetscCall(get_pointer_and_mem_type(dctx,dest,PETSC_MEMORY_ACCESS_WRITE,&dest_ptr,&dmtype,&dest_n));
  PetscCall(get_pointer_and_mem_type(dctx,src,PETSC_MEMORY_ACCESS_READ,&src_ptr,&smtype,&src_n));
  PetscAssert(dest_n >= src_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Destination size %" PetscInt_FMT " not large enough for source size %" PetscInt_FMT,dest_n,src_n);
  PetscCall(PetscDeviceArrayCopy(dctx,dest_ptr,src_ptr,dest_n,PetscMemTypeToDeviceCopyMode(dmtype,smtype)));
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::apply_operator(PetscDeviceContext dctx, ManagedType scal, PetscOperatorType otype, PetscMemType mtype, const PetscType *rhs, ManagedType ret))
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(scal,2);
  if (PetscMemTypeHost(mtype)) PetscValidPointer(rhs,5);
  if ((PetscOffloadHost(scal->mask) || PetscOffloadUnallocated(scal->mask)) && PetscMemTypeHost(mtype)){
    const auto  src_access = ret ? PETSC_MEMORY_ACCESS_READ : PETSC_MEMORY_ACCESS_READ_WRITE;
    const auto  rhsv       = *rhs;
    PetscType  *ptr,*retptr;
    PetscInt    n;

    PetscCall(get_values(dctx,scal,PETSC_MEMTYPE_HOST,src_access,&ptr,&n));
    if (ret) {
      PetscCall(get_values(dctx,ret,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,&retptr));
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
    PetscCall((*apply_operator_fn<ManagedType,T>(dctx))(dctx,scal,otype,rhs,ret));
  }
  PetscFunctionReturn(0);
}

template <typename T, typename MT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<T,MT>::get_sub_range(PetscDeviceContext dctx, ManagedType in, PetscInt begin, PetscInt len, ManagedType *out))
{
  PetscType *tmp;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(in,2);
  PetscValidPointer(out,5);
  PetscAssert(!in->locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input managed object already has a sub-range checked out");
  PetscAssert(len > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot extract a subrange of negative size %" PetscInt_FMT,len);
  PetscAssert(begin+len < in->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Trying to extract a subrange of [%" PetscInt_FMT ",%" PetscInt_FMT ") from managed type of size %" PetscInt_FMT,begin,begin+len,in->n);
  if (!in->host)   PetscCall(get_values(dctx,in,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,&tmp));
  if (!in->device) PetscCall(get_values(dctx,in,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ_WRITE,&tmp));
  in->locked = PETSC_TRUE;
  PetscCall(create(dctx,in->host+begin,in->device+begin,len,PETSC_USE_POINTER,PETSC_USE_POINTER,in->mask,out));
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
  PetscAssert(in->locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input managed object did not have a sub-range checked out");
  in->locked = PETSC_FALSE;
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
  in->mask = (*out)->mask;
  PetscCall(destroy(dctx,out));
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
