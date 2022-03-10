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

#define destroymanagedtype   PetscConcat(destroymanaged,PetscTypeSuffix_L)
#define getmanagedvaluestype PetscConcat(getmanagedvalues,PetscTypeSuffix_L)
#define applyoperatortype    PetscConcat(applyoperator,PetscTypeSuffix_L)

template <typename... T>
auto destroy_managed_type_fn(PetscDeviceContext dctx, T&&... rest) noexcept
PETSC_DECLTYPE_AUTO_RETURNS((*dctx->ops->destroymanagedtype)(dctx,std::forward<T>(rest)...));

template <typename... T>
auto get_managed_values_fn(PetscDeviceContext dctx, T&&... rest) noexcept
PETSC_DECLTYPE_AUTO_RETURNS((*dctx->ops->getmanagedvaluestype)(dctx,std::forward<T>(rest)...));

template <typename... T>
auto apply_operator_fn(PetscDeviceContext dctx, T&&... rest) noexcept
PETSC_DECLTYPE_AUTO_RETURNS((*dctx->ops->applyoperatortype)(dctx,std::forward<T>(rest)...));

#undef destroymanagedtype
#undef getmanagedvaluestype
#undef applyoperatortype

#define PetscManagedTypeCreate               PetscConcat(PetscManagedType,Create)
#define PetscManageHostType                  PetscConcat(PetscManageHost,PetscTypeSuffix)
#define PetscManagedTypeCreateDefault        PetscConcat(PetscManagedTypeCreate,Default)
#define PetscManagedTypeDestroy              PetscConcat(PetscManagedType,Destroy)
#define PetscManagedHostTypeDestroy          PetscConcat(PetscConcat(PetscManagedHost,PetscTypeSuffix),Destroy)
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
#define PetscManagedTypeImpl                 PetscManagedTypeImpl<PetscType,PetscManagedType>

PetscErrorCode PetscManagedTypeCreate(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask, PetscManagedType *scal)
{
  return PetscManagedTypeImpl::create(dctx,host_ptr,device_ptr,n,host_cmode,device_cmode,mask,scal);
}

PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  return PetscManagedTypeImpl::destroy(dctx,scal);
}

PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, PetscType **ptr)
{
  return PetscManagedTypeImpl::get_values(dctx,scal,mtype,mode,sync,ptr);
}

PetscErrorCode PetscManagedTypeSetValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, const PetscType *ptr, PetscInt n)
{
  return PetscManagedTypeImpl::set_values(dctx,scal,mtype,ptr,n);
}

PetscErrorCode PetscManagedTypeGetPointerAndMemType(PetscDeviceContext dctx, PetscManagedType scal, PetscMemoryAccessMode mode, PetscType **ptr, PetscMemType *mtype)
{
  return PetscManagedTypeImpl::get_pointer_and_mem_type(dctx,scal,mode,ptr,mtype);
}

PetscErrorCode PetscManagedTypeCopy(PetscDeviceContext dctx, PetscManagedType dest, PetscManagedType src)
{
  return PetscManagedTypeImpl::copy(dctx,dest,src);
}

PetscErrorCode PetscManagedTypeApplyOperator(PetscDeviceContext dctx, PetscManagedType scal, PetscOperatorType otype, PetscMemType mtype, const PetscType *rhs, PetscManagedType ret)
{
  return PetscManagedTypeImpl::apply_operator(dctx,scal,otype,mtype,rhs,ret);
}

PetscErrorCode PetscManagedTypeGetSubRange(PetscDeviceContext dctx, PetscManagedType in, PetscInt begin, PetscInt len, PetscManagedType *out)
{
  return PetscManagedTypeImpl::get_sub_range(dctx,in,begin,len,out);
}

PetscErrorCode PetscManagedTypeRestoreSubRange(PetscDeviceContext dctx, PetscManagedType in, PetscManagedType *out)
{
  return PetscManagedTypeImpl::restore_sub_range(dctx,in,out);
}

PetscErrorCode PetscManagedTypeEqual(PetscManagedType scal, PetscType val, PetscBool *known, PetscBool *equal)
{
  return PetscManagedTypeImpl::query(scal,val,known,equal);
}

PetscErrorCode PetscManagedHostTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(scal,2);
  PetscCall(PetscManagedTypeEnsureOffload(dctx,*scal,PETSC_OFFLOAD_CPU));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscManagedTypeDestroy(dctx,scal));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeEnsureOffload(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask omask)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(scal,2);
  PetscAssert(omask != PETSC_OFFLOAD_UNALLOCATED,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot request PETSC_OFFLOAD_UNALLOCATED as offload mask");
  {
    auto& mask = scal->mask;

    if (mask != omask && mask != PETSC_OFFLOAD_BOTH) {
      const auto OffloadToMemType = [&](PetscMemType mtype)
      {
        PetscType PETSC_UNUSED *ptr;

        PetscFunctionBegin;
        PetscCall(PetscManagedTypeGetValues(dctx,scal,mtype,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&ptr));
        PetscFunctionReturn(0);
      };

      static_assert(PetscOffloadHost(PETSC_OFFLOAD_BOTH),"");
      if (PetscOffloadHost(omask)) PetscCall(OffloadToMemType(PETSC_MEMTYPE_HOST));
      static_assert(PetscOffloadDevice(PETSC_OFFLOAD_BOTH),"");
      if (PetscOffloadDevice(omask)) PetscCall(OffloadToMemType(PETSC_MEMTYPE_DEVICE));
      PetscAssert(mask == omask || mask == PETSC_OFFLOAD_BOTH,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Managed type offload mask %d != requested mask %d",mask,omask);
    }
  }
  PetscFunctionReturn(0);
}

#undef PetscManagedTypeImpl
#undef PetscManagedTypeCreate
#undef PetscManageHostType
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

#undef PetscType
#undef PetscManagedType
#undef PetscTypeSuffix
#undef PetscTypeSuffix_L
