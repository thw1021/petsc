#ifdef DEBUG_MANAGED_TYPE_IMPL
#  include <petscdevicetypes.h>
#endif

#include <petscdevicetypes.h>
#include <petscsys.h>
#include <petsc/private/petscadvancedmacros.h>

#if !defined(PetscTypeSuffix)
#  error "PetscTypeSuffix must be defined"
#endif

#if !defined(PetscManagedType)
#  define PetscManagedType PetscConcat(PetscManaged,PetscTypeSuffix)
#endif

#if !defined(PetscType)
#  define PetscType PetscConcat(Petsc,PetscTypeSuffix)
#endif

// real implementations found in src/sys/objects/device/interface/managedtype.cxx
#define PetscManagedTypeCreate               PetscConcat(PetscManagedType,Create)
#define PetscManagedTypeCreateDefault        PetscConcat(PetscManagedTypeCreate,Default)
#define PetscManageHostType                  PetscConcat(PetscManageHost,PetscTypeSuffix)
#define PetscManageDeviceType                PetscConcat(PetscManageDevice,PetscTypeSuffix)
#define PetscManagedHostTypeDestroy          PetscConcat(PetscConcat(PetscManagedHost,PetscTypeSuffix),Destroy)
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
#define PetscManagedTypeEqual                PetscConcat(PetscManagedType,Equal)
#define PetscManagedTypeGetSize              PetscConcat(PetscManagedType,GetSize)

#define PETSC_NO_DEVICE_UNUSED PetscIfPetscDefined(HAVE_CXX,,PETSC_UNUSED)
#define PETSC_MANAGED_TYPE_DECL_OR_STUB(FunctionDecl,...) PetscIfPetscDefined(HAVE_CXX,PETSC_EXTERN FunctionDecl;,static inline FunctionDecl __VA_ARGS__)

#define _n_PetscManagedType PetscConcat(_n_,PetscManagedType)
struct _n_PetscManagedType
{
  PetscType             *host;
#if PetscDefined(HAVE_CXX)
  PetscType             *device;
  PetscDeviceType        dtype;
  PetscOffloadMask       mask;
  PetscCopyMode          d_cmode;
#endif
  PetscCopyMode          h_cmode;
  PetscInt               n;
  // REVIEW ME: probably inline this instead
  PetscManagedTypeState  state;
};
typedef struct _n_PetscManagedType *PetscManagedType;
#undef _n_PetscManagedType

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeCreate(PetscDeviceContext PETSC_NO_DEVICE_UNUSED dctx, PetscType *host_ptr, PetscType PETSC_NO_DEVICE_UNUSED *device_ptr, PetscInt n, PetscCopyMode h_cmode, PetscCopyMode PETSC_NO_DEVICE_UNUSED d_cmode, PetscOffloadMask PETSC_NO_DEVICE_UNUSED mask, PetscManagedType *scal),
{
  PetscFunctionBegin;
  PetscCall(PetscNew(scal));
  if (h_cmode == PETSC_COPY_VALUES) {
    PetscCall(PetscMalloc1(n,&(*scal)->host));
    PetscCall(PetscArraycpy((*scal)->host,host_ptr,n));
    h_cmode = PETSC_OWN_POINTER;
  } else {
    (*scal)->host = host_ptr;
  }
  (*scal)->h_cmode = h_cmode;
  (*scal)->n       = n;
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext PETSC_NO_DEVICE_UNUSED dctx, PetscManagedType *scal),
{
  PetscFunctionBegin;
  if (*scal) {
    if ((*scal)->h_cmode != PETSC_USE_POINTER) PetscCall(PetscFree((*scal)->host));
      PetscCall(PetscFree(scal));
  }
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext PETSC_NO_DEVICE_UNUSED dctx, PetscManagedType scal, PetscMemType PETSC_NO_DEVICE_UNUSED mtype, PetscMemoryAccessMode PETSC_NO_DEVICE_UNUSED amode, PetscBool PETSC_NO_DEVICE_UNUSED sync, PetscType **ptr),
{
  PetscFunctionBegin;
  *ptr = scal->host;
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeSetValues(PetscDeviceContext PETSC_NO_DEVICE_UNUSED dctx, PetscManagedType scal, PetscMemType PETSC_NO_DEVICE_UNUSED mtype, const PetscType *ptr, PetscInt n),
{
  PetscFunctionBegin;
  PetscCall(PetscArraycpy(scal->host,ptr,n));
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeGetPointerAndMemType(PetscDeviceContext dctx, PetscManagedType scal, PetscMemoryAccessMode mode, PetscType **ptr, PetscMemType *mtype),
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeGetValues(dctx,scal,PETSC_MEMTYPE_HOST,mode,ptr));
  if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeEnsureOffload(PetscDeviceContext PETSC_NO_DEVICE_UNUSED dctx, PetscManagedType PETSC_NO_DEVICE_UNUSED mtype, PetscOffloadMask PETSC_NO_DEVICE_UNUSED mask),
{
  return 0;
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeCopy(PetscDeviceContext PETSC_NO_DEVICE_UNUSED dctx, PetscManagedType to, PetscManagedType from),
{
  PetscFunctionBegin;
  PetscAssert(to->n >= from->n,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Trying to copy %" PetscInt_FMT " values to managed type of size %" PetscInt_FMT,from->n,to->n);
  PetscCall(PetscArraycpy(to->host,from->host,from->n));
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeApplyOperator(PetscDeviceContext PETSC_NO_DEVICE_UNUSED dctx, PetscManagedType scal, PetscOperatorType op, PetscMemType PETSC_NO_DEVICE_UNUSED mtype, const PetscType *rhs, PetscManagedType ret),
{
  const PetscType  rhsv   = *rhs;
  PetscType       *retptr = ret ? ret->host : scal->host,*src = scal->host;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < scal->n; ++i) {
    switch (op) {
    case PETSC_OPERATOR_PLUS:     retptr[i] = src[i]+rhsv; break;
    case PETSC_OPERATOR_MINUS:    retptr[i] = src[i]-rhsv; break;
    case PETSC_OPERATOR_MULTIPLY: retptr[i] = src[i]*rhsv; break;
    case PETSC_OPERATOR_DIVIDE:   retptr[i] = src[i]/rhsv; break;
    case PETSC_OPERATOR_EQUAL:    retptr[i] = rhsv;        break;
    }
  }
  PetscFunctionReturn(0);
}
);

PETSC_EXTERN PetscErrorCode PetscManagedTypeApplyManagedOperator(PetscDeviceContext,PetscManagedType,PetscOperatorType,PetscManagedType,PetscManagedType);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeGetSubRange(PetscDeviceContext dctx, PetscManagedType in, PetscInt begin, PetscInt len, PetscManagedType *out),
{
  PetscFunctionBegin;
  PetscAssert(!in->state.locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input managed object already has a sub-range checked out");
  PetscAssert(len > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot extract a subrange of negative size %" PetscInt_FMT,len);
  PetscAssert(begin+len < in->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Trying to extract a subrange of [%" PetscInt_FMT ",%" PetscInt_FMT ") from managed type of size %" PetscInt_FMT,begin,begin+len,in->n);
  in->state.locked = 1;
  PetscCall(PetscManagedTypeCreate(dctx,in->host+begin,PETSC_NULLPTR,len,PETSC_USE_POINTER,PETSC_USE_POINTER,PETSC_OFFLOAD_CPU,out));
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeRestoreSubRange(PetscDeviceContext dctx, PetscManagedType in, PetscManagedType *out),
{
  PetscFunctionBegin;
  PetscAssert(in->state.locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Input managed object did not have a sub-range checked out");
  PetscAssert(!(*out)->state.locked,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Sub-range has its own sub-range checked out");
  in->state = (*out)->state;
  PetscCall(PetscManagedTypeDestroy(dctx,out));
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedTypeEqual(PetscManagedType scal, PetscType val, PetscBool *known, PetscBool *equal),
{
  PetscInt eqcnt = 0;

  PetscFunctionBegin;
  *known = PETSC_TRUE;
  for (PetscInt i = 0; i < scal->n; ++i) eqcnt += scal->host[i] == val;
  *equal = (PetscBool)(eqcnt == scal->n);
  PetscFunctionReturn(0);
}
);

PETSC_MANAGED_TYPE_DECL_OR_STUB(
PetscErrorCode PetscManagedHostTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal),
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeDestroy(dctx,scal));
  PetscFunctionReturn(0);
}
);

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

static inline PetscErrorCode PetscManagedTypeGetSize(PetscManagedType scal, PetscInt *n)
{
  PetscFunctionBegin;
  *n = scal->n;
  PetscFunctionReturn(0);
}

#undef PETSC_NO_DEVICE_UNUSED
#undef PETSC_MANAGED_TYPE_DECL_OR_STUB

#undef PetscManagedTypeCreate
#undef PetscManageHostType
#undef PetscManageDeviceType
#undef PetscManagedTypeCreateDefault
#undef PetscManagedTypeDestroy
#undef PetscManagedHostTypeDestroy
#undef PetscManagedTypeGetValues
#undef PetscManagedTypeSetValues
#undef PetscManagedTypeGetPointerAndMemType
#undef PetscManagedTypeEnsureOffload
#undef PetscManagedTypeCopy
#undef PetscManagedTypeApplyOperator
#undef PetscManagedTypeApplyManagedOperator
#undef PetscManagedTypeGetSubRange
#undef PetscManagedTypeRestoreSubRange
#undef PetscManagedTypeEqual
#undef PetscManagedTypeGetSize

#undef PetscTypeSuffix
#undef PetscType
#undef PetscManagedType
