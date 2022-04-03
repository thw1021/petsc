#include <petscdevicetypes.h>

#if !defined(PetscTypeSuffix)
#  error "PetscTypeSuffix must be defined"
#endif

#if !defined(PetscManagedType)
#  define PetscManagedType PetscConcat(PetscManaged,PetscTypeSuffix)
#endif

#if !defined(PetscType)
#  define PetscType PetscConcat(Petsc,PetscTypeSuffix)
#endif

#if !defined(PetscManagedType)
#  error "PetscManagedType undefined"
#endif

#if !defined(PetscType)
#  error "PetscType undefined"
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
  PetscCopyMode     h_cmode,d_cmode;
};

typedef struct _n_PetscManagedType *PetscManagedType;
#undef _n_PetscManagedType
#else
typedef PetscType *PetscManagedType;
#endif

#if defined(PetscConcat3)
#  error "PetscConcat3 defined"
#endif

#define PetscConcat3(a,b,c) PetscConcat(PetscConcat(a,b),c)

// real implementations found in src/sys/objects/device/interface/managedtype.cxx
#define PetscDeviceContextCreateManagedTypeArray PetscConcat3(PetscDeviceContextCreateManaged,PetscTypeSuffix,Array)
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreateManagedTypeArray(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscCopyMode,PetscOffloadMask,PetscManagedType*);
#define PetscDeviceContextDestroyManagedTypeArray PetscConcat3(PetscDeviceContextDestroyManaged,PetscTypeSuffix,Array)
PETSC_EXTERN PetscErrorCode PetscDeviceContextDestroyManagedTypeArray(PetscDeviceContext,PetscManagedType*);
#define PetscDeviceContextGetManagedTypeValues PetscConcat3(PetscDeviceContextGetManaged,PetscTypeSuffix,Values)
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetManagedTypeValues(PetscDeviceContext,PetscManagedType,PetscMemType,PetscMemoryAccessMode,PetscType**,PetscInt*);

#define PetscManagedTypeCreate PetscConcat3(PetscManaged,PetscTypeSuffix,Create)
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

#define PetscManageHostType PetscConcat(PetscManageHost,PetscTypeSuffix)
static inline PetscErrorCode PetscManageHostType(PetscDeviceContext dctx, PetscType *host_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,host_ptr,PETSC_NULLPTR,n,PETSC_USE_POINTER,PETSC_OWN_POINTER,PETSC_OFFLOAD_CPU,scal));
  PetscFunctionReturn(0);
}

#define PetscManagedTypeDestroy PetscConcat(PetscManagedType,Destroy)
static inline PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEVICE)) PetscCall(PetscDeviceContextDestroyManagedTypeArray(dctx,scal));
  *scal = PETSC_NULLPTR;
  PetscFunctionReturn(0);
}

#define PetscManagedTypeGetValues PetscConcat(PetscManagedType,GetValues)
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

#define PetscManagedTypeCopy PetscConcat(PetscManagedType,Copy)
static inline PetscErrorCode PetscManagedTypeCopy(PetscDeviceContext dctx, PetscManagedType dest, PetscManagedType src)
{
  PetscFunctionBegin;
  if (PetscDefined(HAVE_DEVICE)) {

  } else {
    *dest = *src;
  }
  PetscFunctionReturn(0);
}

#undef PetscConcat3
#undef PetscDeviceContextCreateManagedTypeArray
#undef PetscDeviceContextDestroyManagedTypeArray
#undef PetscDeviceContextGetManagedTypeValues

#undef PetscManageHostType
#undef PetscManagedTypeCreate
#undef PetscManagedTypeDestroy
#undef PetscManagedTypeGetValeus
#undef PetscManagedTypeCopy

#undef PetscTypeSuffix
#undef PetscManagedType
#undef PetscType
