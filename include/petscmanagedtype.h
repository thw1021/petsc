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
  PetscCopyMode     cmode;
};

typedef struct _n_PetscManagedType *PetscManagedType;
#undef _n_PetscManagedType
#else
typedef PetscType *PetscManagedType;
#endif

#define PetscDeviceContextCreateManagedTypeArray PetscConcat(PetscConcat(PetscDeviceContextCreateManaged,PetscTypeSuffix),Array)
#define PetscDeviceContextDestroyManagedTypeArray PetscConcat(PetscConcat(PetscDeviceContextDestroyManaged,PetscTypeSuffix),Array)
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreateManagedTypeArray(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscMemType,PetscOffloadMask,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDestroyManagedTypeArray(PetscDeviceContext,PetscManagedType*);

#define PetscManageHostType PetscConcat(PetscManageHost,PetscTypeSuffix)
static inline PetscErrorCode PetscManageHostType(PetscDeviceContext dctx, PetscType *host_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_DEVICE)
  PetscCall(PetscDeviceContextCreateManagedTypeArray(dctx,host_ptr,PETSC_NULLPTR,n,PETSC_USE_POINTER,PETSC_MEMTYPE_HOST,PETSC_OFFLOAD_CPU,scal));
#else
  *scal = host_ptr;
#endif
  PetscFunctionReturn(0);
}

#define PetscManagedTypeDestroy PetscConcat(PetscManagedType,Destroy)
static inline PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  if (*scal) PetscCall(PetscDeviceContextDestroyManagedTypeArray(dctx,scal));
  *scal = PETSC_NULLPTR;
  PetscFunctionReturn(0);
}

#define PetscManagedTypeGetValues PetscConcat(PetscManagedType,GetValues)
static inline PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask mask, PetscType *ptr, PetscInt *n)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_DEVICE)
  if (mask == scal->mask) {
    switch (mask) {
    case PETSC_OFFLOAD_CPU:
      *ptr = scal->host;
      break;
    case PETSC_OFFLOAD_GPU:
    case PETSC_OFFLOAD_BOTH:
      *ptr = scal->device;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unhandleable offload mask %d",(int)mask);
    }
  } else {
    // call
  }
  if (n) *n = scal->n;
#else
  *ptr = scal;
#endif
  PetscFunctionReturn(0);
}

#undef PetscDeviceContextCreateManagedTypeArray
#undef PetscDeviceContextDestroyManagedTypeArray

#undef PetscManageHostType
#undef PetscManagedTypeDestroy
#undef PetscDeviceContextCreateManagedTypeArray
#undef PetscDeviceContextDestroyManagedTypeArray

#undef PetscTypeSuffix
#undef PetscManagedType
#undef PetscType
