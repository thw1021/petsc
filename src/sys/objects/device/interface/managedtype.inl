#ifdef DEBUG_MANAGED_TYPE_IMPL
#  include <petsc/private/deviceimpl.h>
#  include <petsc/private/cpputil.hpp>
#  include "objpool.hpp"
#  define PetscTypeSuffix   Scalar
#  define PetscTypeSuffix_L scalar
#endif

#if !defined(PetscTypeSuffix)
#  error "Must define PetscTypeSuffix"
#endif

#if !defined(PetscTypeSuffix_L)
#  error "Must define PetscTypeSuffix_L"
#endif

#if !defined(PetscType)
#  define PetscType PetscConcat(Petsc,PetscTypeSuffix)
#endif

#if !defined(PetscManagedType)
#  define PetscManagedType PetscConcat(PetscManaged,PetscTypeSuffix)
#endif

#if defined(PetscConcat3)
#  error "PetscConcat3 defined"
#else
#  define PetscConcat3(a,b,c) PetscConcat(PetscConcat(a,b),c)
#endif

#define PetscManagedTypeAllocator            PetscConcat(PetscManagedType,Allocator)
#define PetscManagedTypePool                 PetscConcat(PetscManagedType,Pool)
#define PetscValidTypePointer                PetscConcat3(PetscValid,PetscTypeSuffix,Pointer)

#define releasemanagedtype                   PetscConcat(releasemanaged,PetscTypeSuffix_L)
#define getmanagedvaluestype                 PetscConcat(getmanagedvalues,PetscTypeSuffix_L)
#define applyoperatortype                    PetscConcat(applyoperator,PetscTypeSuffix_L)

struct PetscManagedTypeAllocator : Petsc::AllocatorBase<PetscManagedType>
{
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscManagedType *mscal))
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

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscManagedType mscal))
  {
    PetscFunctionBegin;
    PetscCall(PetscFree(mscal));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset(PetscManagedType mscal))
  {
    PetscFunctionBegin;
    PetscCallCXX(std::memset(mscal,0,sizeof(*mscal)));
    mscal->h_cmode = PETSC_OWN_POINTER;
    mscal->d_cmode = PETSC_OWN_POINTER;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) { return 0; }
};

static auto PetscManagedTypePool = Petsc::ObjectPool<PetscManagedType,PetscManagedTypeAllocator>{};

namespace
{

static PetscErrorCode CopyValues(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask mask, PetscOffloadMask src_mask, const PetscType *src_ptr)
{
  PetscDeviceCopyMode  mode;
  PetscType           *ptr;
  PetscInt             n;

  // need to actually allocate the stuff
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeGetValues(dctx,scal,PetscOffloadMaskToMemType(mask),PETSC_MEMORY_ACCESS_WRITE,&ptr,&n));
  PetscCall(PetscOffloadMaskToDeviceCopyMode(mask,src_mask,&mode));
  PetscCall(PetscDeviceArrayCopy(dctx,ptr,src_ptr,n,mode));
  PetscFunctionReturn(0);
}

} // anonymous namespace

#define PetscManagedTypeCreate               PetscConcat(PetscManagedType,Create)
#define PetscManageHostType                  PetscConcat(PetscManageHost,PetscTypeSuffix)
#define PetscManagedTypeCreateDefault        PetscConcat(PetscManagedTypeCreate,Default)
#define PetscManagedTypeDestroy              PetscConcat(PetscManagedType,Destroy)
#define PetscManagedTypeGetValues            PetscConcat(PetscManagedType,GetValues)
#define PetscManagedTypeSetValues            PetscConcat(PetscManagedType,SetValues)
#define PetscManagedTypeGetPointerAndMemType PetscConcat(PetscManagedType,GetPointerAndMemType)
#define PetscManagedTypeEnsureOffload        PetscConcat(PetscManagedType,EnsureOffload)
#define PetscManagedTypeCopy                 PetscConcat(PetscManagedType,Copy)
#define PetscManagedTypeApplyOperator        PetscConcat(PetscManagedType,ApplyOperator)
#define PetscManagedTypeApplyManagedOperator PetscConcat(PetscManagedType,ApplyManagedOperator)
#define PetscManagedTypeGetSubRange          PetscConcat(PetscManagedType,GetSubRange)
#define PetscManagedTypeRestoreSubRange      PetscConcat(PetscManagedType,RestoreSubRange)

PetscErrorCode PetscManagedTypeCreate(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (host_ptr && n) PetscValidTypePointer(host_ptr,2);
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
  PetscCall(PetscManagedTypePool.get(*scal));

  // populate known quantities
  (*scal)->n       = n;
  (*scal)->mask    = mask;
  (*scal)->h_cmode = host_cmode;
  (*scal)->d_cmode = device_cmode;

  if (host_cmode == PETSC_COPY_VALUES) {
    PetscCall(CopyValues(dctx,*scal,mask,PETSC_OFFLOAD_CPU,host_ptr));
  } else {
    // own_pointer or use_pointer
    (*scal)->host = host_ptr;
  }

  if (device_cmode == PETSC_COPY_VALUES) {
    PetscCall(CopyValues(dctx,*scal,mask,PETSC_OFFLOAD_GPU,device_ptr));
  } else {
    // own_pointer or use_pointer
    (*scal)->device = device_ptr;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  if (!*scal) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall((*dctx->ops->releasemanagedtype)(dctx,*scal));
  PetscCall(PetscManagedTypePool.reclaim(std::move(*scal)));
  *scal = nullptr;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr, PetscInt *n)
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
  PetscCall((*dctx->ops->getmanagedvaluestype)(dctx,scal,mtype,mode,ptr));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeSetValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, const PetscType *ptr, PetscInt n)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscValidTypePointer(ptr,4);
  PetscAssert(!scal->locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Object is locked, perhaps a subrange was not yet restored?");
  if (n) {
    PetscMemType  scalmtype;
    PetscType    *scalptr;
    PetscInt      scaln;

    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscCall(PetscManagedTypeGetPointerAndMemType(dctx,scal,PETSC_MEMORY_ACCESS_WRITE,&scalptr,&scalmtype,&scaln));
    PetscAssert(n <= scaln,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to write %" PetscInt_FMT " values to " PetscStringize(PetscManagedType) " but it only holds %" PetscInt_FMT " entries",n,scaln);
    PetscCall(PetscDeviceArrayCopy(dctx,scalptr,ptr,n,PetscMemTypeToDeviceCopyMode(scalmtype,mtype)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeGetPointerAndMemType(PetscDeviceContext dctx, PetscManagedType scal, PetscMemoryAccessMode mode, PetscType **ptr, PetscMemType *mtype, PetscInt *n)
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
    PetscCall(PetscManagedTypeGetValues(dctx,scal,prefer_host ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE,mode,ptr,nullptr));
    if (prefer_host) {
      goto UNALLOCATED_PREFER_HOST;
    } else {
      goto UNALLOCATED_PREFER_DEVICE;
    }
    PetscUnreachable();
  } break;
  }
  if (!*ptr) {
    __builtin_dump_struct(scal,&printf);
  }
  PetscAssert(*ptr,PETSC_COMM_SELF,PETSC_ERR_PLIB,PetscStringize(PetscManagedType) " returned a null pointer for memtype %s as values",mtype ? (PetscMemTypeHost(*mtype) ? "host" : "device") : "unknown");
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeEnsureOffload(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask omask)
{
  PetscFunctionBegin;
  PetscValidPointer(scal,2);
  if (scal->mask != omask) {
    PetscType *ptr;

    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    if (PetscOffloadDevice(omask) || omask == PETSC_OFFLOAD_BOTH) {
      PetscCall(PetscManagedTypeGetValues(dctx,scal,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,&ptr,nullptr));
    }
    if (PetscOffloadHost(omask) || omask == PETSC_OFFLOAD_BOTH) {
      PetscCall(PetscManagedTypeGetValues(dctx,scal,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&ptr,nullptr));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeCopy(PetscDeviceContext dctx, PetscManagedType dest, PetscManagedType src)
{
  PetscMemType  dmtype,smtype;
  PetscInt      dest_n,src_n;
  PetscType    *dest_ptr,*src_ptr;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(dest,2);
  PetscValidPointer(src,3);
  PetscCall(PetscManagedTypeGetPointerAndMemType(dctx,dest,PETSC_MEMORY_ACCESS_WRITE,&dest_ptr,&dmtype,&dest_n));
  PetscCall(PetscManagedTypeGetPointerAndMemType(dctx,src,PETSC_MEMORY_ACCESS_READ,&src_ptr,&smtype,&src_n));
  PetscAssert(dest_n >= src_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Destination size %" PetscInt_FMT " not large enough for source size %" PetscInt_FMT,dest_n,src_n);
  PetscCall(PetscDeviceArrayCopy(dctx,dest_ptr,src_ptr,dest_n,PetscMemTypeToDeviceCopyMode(dmtype,smtype)));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeApplyOperator(PetscDeviceContext dctx, PetscManagedType scal, PetscOperatorType otype, PetscMemType mtype, const PetscType *rhs, PetscManagedType ret)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(scal,2);
  if (PetscMemTypeHost(mtype)) PetscValidTypePointer(rhs,4);
  if ((PetscOffloadHost(scal->mask) || PetscOffloadUnallocated(scal->mask)) && PetscMemTypeHost(mtype)) {
    const auto  src_access = ret ? PETSC_MEMORY_ACCESS_READ : PETSC_MEMORY_ACCESS_READ_WRITE;
    const auto  rhsv       = *rhs;
    PetscType  *ptr,*retptr;
    PetscInt    n;

    PetscCall(PetscManagedTypeGetValues(dctx,scal,PETSC_MEMTYPE_HOST,src_access,&ptr,&n));
    if (ret) {
      PetscCall(PetscManagedTypeGetValues(dctx,ret,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,&retptr,nullptr));
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
    PetscCall((*dctx->ops->applyoperatortype)(dctx,scal,otype,rhs,ret));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeGetSubRange(PetscDeviceContext dctx, PetscManagedType in, PetscInt begin, PetscInt end, PetscManagedType *out)
{
  const auto  size = end-begin;
  PetscType  *tmp;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssert(!in->locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input managed object already has a sub-range checked out");
  PetscAssert(size > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot extract a subrange of negative size %" PetscInt_FMT,size);
  PetscAssert(begin+size < in->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Trying to extract a subrange of [%" PetscInt_FMT ",%" PetscInt_FMT ") from managed type of size %" PetscInt_FMT,begin,end,in->n);
  if (!in->host) PetscCall(PetscManagedTypeGetValues(dctx,in,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,&tmp,nullptr));
  if (!in->device) PetscCall(PetscManagedTypeGetValues(dctx,in,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ_WRITE,&tmp,nullptr));
  in->locked = PETSC_TRUE;
  PetscCall(PetscManagedTypeCreate(dctx,in->device+begin,in->device+begin,size,PETSC_USE_POINTER,PETSC_USE_POINTER,in->mask,out));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeRestoreSubRange(PetscDeviceContext dctx, PetscManagedType in, PetscManagedType *out)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssert(in->locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input managed object did not have a sub-range checked out");
  in->locked = PETSC_FALSE;
  if (PetscDefined(USE_DEBUG)) {
    const auto check_ownership = [&](const PetscType *begin, const PetscType *needle)
    {
      PetscFunctionBegin;
      PetscAssert((begin <= needle) && (needle < (begin+in->n)),PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sub-range does not belong to input managed type");
      PetscFunctionReturn(0);
    };
    PetscCall(check_ownership(in->host,(*out)->host));
    PetscCall(check_ownership(in->device,(*out)->device));
  }
  PetscCall(PetscManagedTypeDestroy(dctx,out));
  PetscFunctionReturn(0);
}

#undef PetscType
#undef PetscManagedType
#undef PetscTypeSuffix
#undef PetscTypeSuffix_L
#undef PetscConcat3

#undef PetscManagedTypeAllocator
#undef PetscManagedTypePool
#undef PetscValidTypePointer

#undef releasemanagedtype
#undef getmanagedvaluestype
#undef applyoperatortype

#undef PetscManagedTypeCreate
#undef PetscManageHostType
#undef PetscManagedTypeCreateDefault
#undef PetscManagedTypeDestroy
#undef PetscManagedTypeGetValeus
#undef PetscManagedTypeSetValeus
#undef PetscManagedTypeGetPointerAndMemType
#undef PetscManagedTypeEnsureOffload
#undef PetscManagedTypeCopy
#undef PetscManagedTypeApplyOperator
#undef PetscManagedTypeApplyManagedOperator
#undef PetscManagedTypeGetSubRange
#undef PetscManagedTypeRestoreSubRange
