#ifdef DEBUG_MANAGED_TYPE_IMPL
#  include <petscdevicetypes.h>
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

#define PetscManagedTypeAllocator                 PetscConcat(PetscManagedType,Allocator)
#define PetscManagedTypePool                      PetscConcat(PetscManagedType,Pool)
#define PetscValidTypePointer                     PetscConcat3(PetscValid,PetscTypeSuffix,Pointer)
#define PetscDeviceContextCreateManagedTypeArray  PetscConcat3(PetscDeviceContextCreateManaged,PetscTypeSuffix,Array)
#define releasemanagedtype                        PetscConcat(releasemanaged,PetscTypeSuffix_L)
#define PetscDeviceContextDestroyManagedTypeArray PetscConcat3(PetscDeviceContextDestroyManaged,PetscTypeSuffix,Array)
#define getmanagedvaluestype                      PetscConcat(getmanagedvalues,PetscTypeSuffix_L)
#define PetscDeviceContextGetManagedTypeValues    PetscConcat3(PetscDeviceContextGetManaged,PetscTypeSuffix,Values)
#define PetscDeviceContextCopyManagedType         PetscConcat(PetscDeviceContextCopyManaged,PetscTypeSuffix)

struct PetscManagedTypeAllocator : Petsc::AllocatorBase<PetscManagedType>
{
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscManagedType *mscal))
  {
    using Petsc::util::integral_value;

    PetscFunctionBegin;
    PetscCall(PetscNew(mscal));
    (*mscal)->h_cmode = PETSC_OWN_POINTER;
    (*mscal)->d_cmode = PETSC_OWN_POINTER;
    static_assert(integral_value(PETSC_OWN_POINTER) != 0,"");
    static_assert(integral_value(PETSC_OFFLOAD_UNALLOCATED) == 0,"");
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
    mscal->n       = 0;
    mscal->host    = nullptr;
    mscal->device  = nullptr;
    mscal->mask    = PETSC_OFFLOAD_UNALLOCATED;
    mscal->h_cmode = PETSC_OWN_POINTER;
    mscal->d_cmode = PETSC_OWN_POINTER;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) { return 0; }
};

static auto PetscManagedTypePool = Petsc::ObjectPool<PetscManagedType,PetscManagedTypeAllocator>{};

PetscErrorCode PetscDeviceContextCreateManagedTypeArray(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask, PetscManagedType *scal)
{
  PetscFunctionBegin;
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

  PetscCall(PetscManagedTypePool.get(*scal));
  (*scal)->n       = n;
  switch (host_cmode) {
  case PETSC_OWN_POINTER:
  case PETSC_USE_POINTER:
    (*scal)->host = host_ptr;
    break;
  case PETSC_COPY_VALUES:
    PetscCall(PetscMalloc1(n,&((*scal)->host)));
    PetscArraycpy((*scal)->host,host_ptr,n);
    break;
  }
  switch (device_cmode) {
  case PETSC_OWN_POINTER:
  case PETSC_USE_POINTER:
    (*scal)->device = device_ptr;
    break;
  case PETSC_COPY_VALUES:
    PetscCall(PetscMalloc1(n,&((*scal)->host)));
    PetscArraycpy((*scal)->host,host_ptr,n);
    break;
  }
  (*scal)->device  = device_ptr;
  (*scal)->mask    = mask;
  (*scal)->h_cmode = host_cmode;
  (*scal)->d_cmode = device_cmode;
  PetscFunctionReturn(0);
}


PetscErrorCode PetscDeviceContextDestroyManagedTypeArray(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  if (!dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscValidDeviceContext(dctx,1);
  if (*scal) PetscFunctionReturn(0);
  PetscCall((*dctx->ops->releasemanagedtype)(dctx,*scal));
  PetscCall(PetscManagedTypePool.reclaim(std::move(*scal)));
  *scal = nullptr;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextGetManagedTypeValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr, PetscInt *n)
{
  PetscFunctionBegin;
  if (!dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(scal,2);
  PetscValidPointer(ptr,5);
  if (n) PetscValidIntPointer(n,6);
  PetscCall((*dctx->ops->getmanagedvaluestype)(dctx,scal,mtype,mode,ptr));
  if (n) *n = scal->n;
  PetscFunctionReturn(0);
}


PetscErrorCode PetscDeviceContextCopyManagedType(PetscDeviceContext dctx, PetscManagedType dest, PetscManagedType src)
{
  // prefer GPU if both
  const auto           dmask = dest->mask == PETSC_OFFLOAD_BOTH ? PETSC_OFFLOAD_GPU : dest->mask;
  PetscInt             dest_n,src_n;
  PetscType           *dest_ptr,*src_ptr;
  PetscDeviceCopyMode  mode;

  PetscFunctionBegin;
  if (!dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(dest,2);
  PetscValidPointer(src,3);
  PetscCall(PetscDetermineCopyMode(dmask,src->mask,&mode));
  PetscCall(PetscDeviceContextGetManagedTypeValues(dctx,dest,dmask,&dest_ptr,&dest_n));
  PetscCall(PetscDeviceContextGetManagedTypeValeus(dctx,src,dmask,&src_ptr,&src_n));
  PetscAssert(dest_n >= src_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Destination size %" PetscInt_FMT " not large enough for source size %" PetscInt_FMT,dest_n,src_n);
  PetscCall(PetscDeviceArrayCopy(dctx,dest_ptr,src_ptr,dest_n,mode));
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
#undef PetscDeviceContextCreateManagedTypeArray
#undef releasemanagedtype
#undef PetscDeviceContextDestroyManagedTypeArray
#undef getmanagedvaluestype
#undef PetscDeviceContextGetManagedTypeValues
#undef PetscDeviceContextCopyManagedType
