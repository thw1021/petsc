#ifndef DEBUG_MANAGED_TYPE_IMPL
#  include <petscdevicetypes.h>
#endif

#if !defined(PetscTypeSuffix)
#  error "PetscTypeSuffix must be defined"
#endif

#if !defined(PetscManagedType)
#  define PetscManagedType PetscConcat(PetscManaged,PetscTypeSuffix)
#endif

#if !defined(PetscType)
#  define PetscType PetscConcat(Petsc,PetscTypeSuffix)
#endif

#if defined(PetscConcat3)
#  error "PetscConcat3 defined"
#else
#  define PetscConcat3(a,b,c) PetscConcat(PetscConcat(a,b),c)
#endif

#if PetscDefined(HAVE_DEVICE)
#define _n_PetscManagedType PetscConcat(_n_,PetscManagedType)
struct _n_PetscManagedType
{
  PetscInt          n;
  PetscType        *host;
  PetscType        *device;
  PetscMemType      mtype;
  PetscOffloadMask  mask;
  PetscCopyMode     d_cmode;
  PetscCopyMode     h_cmode;
};

typedef struct _n_PetscManagedType *PetscManagedType;
#undef _n_PetscManagedType
#else
typedef PetscType *PetscManagedType;
#endif

// real implementations found in src/sys/objects/device/interface/managedtype.cxx
#define PetscDeviceContextCreateManagedTypeArray  PetscConcat3(PetscDeviceContextCreateManaged,PetscTypeSuffix,Array)
#define PetscDeviceContextDestroyManagedTypeArray PetscConcat3(PetscDeviceContextDestroyManaged,PetscTypeSuffix,Array)
#define PetscDeviceContextGetManagedTypeValues    PetscConcat3(PetscDeviceContextGetManaged,PetscTypeSuffix,Values)
#define PetscDeviceContextCopyManagedType         PetscConcat(PetscDeviceContextCopyManaged,PetscTypeSuffix)
#define PetscDeviceContextApplyOperatorManagedType PetscConcat(PetscDeviceContextApplyOperatorManaged,PetscTypeSuffix)
#define PetscManagedTypeCreate                    PetscConcat(PetscManagedType,Create)
#define PetscManageHostType                       PetscConcat(PetscManageHost,PetscTypeSuffix)
#define PetscManagedTypeCreateDefault             PetscConcat(PetscManagedTypeCreate,Default)
#define PetscManagedTypeDestroy                   PetscConcat(PetscManagedType,Destroy)
#define PetscManagedTypeGetValues                 PetscConcat(PetscManagedType,GetValues)
#define PetscManagedTypeCopy                      PetscConcat(PetscManagedType,Copy)
#define PetscManagedTypeApplyOperator             PetscConcat(PetscManagedType,ApplyOperator)

PETSC_EXTERN PetscErrorCode PetscDeviceContextCreateManagedTypeArray(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscCopyMode,PetscOffloadMask,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDestroyManagedTypeArray(PetscDeviceContext,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetManagedTypeValues(PetscDeviceContext,PetscManagedType,PetscMemType,PetscMemoryAccessMode,PetscType**,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextCopyManagedType(PetscDeviceContext,PetscManagedType,PetscManagedType);
PETSC_EXTERN PetscErrorCode PetscDeviceContextApplyOperatorManagedType(PetscDeviceContext,PetscManagedType,PetscOperatorType,PetscType,PetscManagedType);

static inline PetscErrorCode PetscManagedTypeCreate(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask, PetscManagedType *scal)
{
  PetscFunctionBegin;
  if (PetscDefined(HAVE_DEVICE)) {
    PetscCall(PetscDeviceContextCreateManagedTypeArray(dctx,host_ptr,device_ptr,n,host_cmode,device_cmode,mask,scal));
  } else {
    // have to cast here since otherwise compilers (rightfully!) complain about assigning
    // pointer to struct to a pointer to scalar type when PETSC_HAVE_DEVICE is defined
    *scal = (PetscManagedType)host_ptr;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManageHostType(PetscDeviceContext dctx, PetscType *host_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,host_ptr,PETSC_NULLPTR,n,PETSC_USE_POINTER,PETSC_OWN_POINTER,PETSC_OFFLOAD_CPU,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeCreateDefault(PetscDeviceContext dctx, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,PETSC_NULLPTR,PETSC_NULLPTR,n,PETSC_OWN_POINTER,PETSC_OWN_POINTER,PETSC_OFFLOAD_UNALLOCATED,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEVICE)) PetscCall(PetscDeviceContextDestroyManagedTypeArray(dctx,scal));
  *scal = PETSC_NULLPTR;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr, PetscInt *n)
{
  PetscFunctionBegin;
  if (PetscDefined(HAVE_DEVICE)) {
    PetscCall(PetscDeviceContextGetManagedTypeValues(dctx,scal,mtype,mode,ptr,n));
  } else {
    // have to cast here since otherwise compilers (rightfully!) complain about assigning
    // pointer to struct to a pointer to scalar type when PETSC_HAVE_DEVICE is defined
    *ptr = (PetscType*)scal;
    if (n) *n = PETSC_DECIDE;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeCopy(PetscDeviceContext dctx, PetscManagedType dest, PetscManagedType src)
{
  PetscFunctionBegin;
  if (PetscDefined(HAVE_DEVICE)) {
    PetscCall(PetscDeviceContextCopyManagedType(dctx,dest,src));
  } else {
    *dest = *src;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeApplyOperator(PetscDeviceContext dctx, PetscManagedType scal, PetscOperatorType otype, PetscType rhs, PetscManagedType ret)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextApplyOperatorManagedType(dctx,scal,otype,rhs,ret));
  PetscFunctionReturn(0);
}

#undef PetscConcat3
#undef PetscDeviceContextCreateManagedTypeArray
#undef PetscDeviceContextDestroyManagedTypeArray
#undef PetscDeviceContextGetManagedTypeValues
#undef PetscDeviceContextCopyManagedType

#undef PetscManagedTypeCreate
#undef PetscManageHostType
#undef PetscManagedTypeCreateDefault
#undef PetscManagedTypeDestroy
#undef PetscManagedTypeGetValeus
#undef PetscManagedTypeCopy
#undef PetscManagedTypeApplyOperator

#undef PetscTypeSuffix
#undef PetscManagedType
#undef PetscType
