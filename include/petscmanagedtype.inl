#ifdef DEBUG_MANAGED_TYPE_IMPL
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
#define PetscManagedTypeCreate               PetscConcat(PetscManagedType,Create)
#define PetscManagedTypeCreateDefault        PetscConcat(PetscManagedTypeCreate,Default)
#define PetscManageHostType                  PetscConcat(PetscManageHost,PetscTypeSuffix)
#define PetscManageDeviceType                PetscConcat(PetscManageDevice,PetscTypeSuffix)
#define PetscManagedTypeDestroy              PetscConcat(PetscManagedType,Destroy)
#define PetscManagedTypeGetValues            PetscConcat(PetscManagedType,GetValues)
#define PetscManagedTypeSetValues            PetscConcat(PetscManagedType,SetValues)
#define PetscManagedTypeCopy                 PetscConcat(PetscManagedType,Copy)
#define PetscManagedTypeApplyOperator        PetscConcat(PetscManagedType,ApplyOperator)
#define PetscManagedTypeApplyManagedOperator PetscConcat(PetscManagedType,ApplyManagedOperator)

PETSC_EXTERN PetscErrorCode PetscManagedTypeCreate(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscCopyMode,PetscOffloadMask,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext,PetscManagedType,PetscMemType, PetscMemoryAccessMode,PetscType**,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscManagedTypeSetValues(PetscDeviceContext,PetscManagedType,PetscMemType,const PetscType*,PetscInt);
PETSC_EXTERN PetscErrorCode PetscManagedTypeCopy(PetscDeviceContext,PetscManagedType,PetscManagedType);
PETSC_EXTERN PetscErrorCode PetscManagedTypeApplyOperator(PetscDeviceContext,PetscManagedType,PetscOperatorType,PetscMemType,const PetscType*,PetscManagedType);
PETSC_EXTERN PetscErrorCode PetscManagedTypeApplyManagedOperator(PetscDeviceContext,PetscManagedType,PetscOperatorType,PetscManagedType,PetscManagedType);

static inline PetscErrorCode PetscManageHostType(PetscDeviceContext dctx, PetscType *host_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,host_ptr,PETSC_NULLPTR,n,PETSC_USE_POINTER,PETSC_OWN_POINTER,PETSC_OFFLOAD_CPU,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManageDeviceType(PetscDeviceContext dctx, PetscType *device_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,PETSC_NULLPTR,device_ptr,n,PETSC_OWN_POINTER,PETSC_USE_POINTER,PETSC_OFFLOAD_GPU,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeCreateDefault(PetscDeviceContext dctx, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,PETSC_NULLPTR,PETSC_NULLPTR,n,PETSC_OWN_POINTER,PETSC_OWN_POINTER,PETSC_OFFLOAD_UNALLOCATED,scal));
  PetscFunctionReturn(0);
}

#undef PetscManagedTypeCreate
#undef PetscManageHostType
#undef PetscManageDeviceType
#undef PetscManagedTypeCreateDefault
#undef PetscManagedTypeDestroy
#undef PetscManagedTypeGetValeus
#undef PetscManagedTypeSetValeus
#undef PetscManagedTypeCopy
#undef PetscManagedTypeApplyOperator
#undef PetscManagedTypeApplyManagedOperator

#undef PetscTypeSuffix
#undef PetscManagedType
#undef PetscType
#undef PetscConcat3
