#ifdef DEBUG_MANAGED_TYPE_IMPL
#  include <petscdevicetypes.h>
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
#endif

#define PetscConcat3(a,b,c) PetscConcat(PetscConcat(a,b),c)

#define acquiremanagedtype PetscConcat(acquiremanaged,PetscTypeSuffix_L)
template <>
PETSC_CXX_COMPAT_DEFN(constexpr auto PetscManagedTypeImpl<PetscType,PetscManagedType>::acquire_func_ptr(PetscDeviceContext dctx)) PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->acquiremanagedtype);
#define releasemanagedtype PetscConcat(releasemanaged,PetscTypeSuffix_L)
template <>
PETSC_CXX_COMPAT_DEFN(constexpr auto PetscManagedTypeImpl<PetscType,PetscManagedType>::release_func_ptr(PetscDeviceContext dctx)) PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->releasemanagedtype);
#define getmanagedvaluestype PetscConcat(getmanagedvalues,PetscTypeSuffix_L)
template <>
PETSC_CXX_COMPAT_DEFN(constexpr auto PetscManagedTypeImpl<PetscType,PetscManagedType>::getvalues_func_ptr(PetscDeviceContext dctx)) PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->getmanagedvaluestype);

#define PetscDeviceContextCreateManagedTypeArray PetscConcat3(PetscDeviceContextCreateManaged,PetscTypeSuffix,Array)
PetscErrorCode PetscDeviceContextCreateManagedTypeArray(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeImpl<PetscType,PetscManagedType>::create(dctx,host_ptr,device_ptr,n,host_cmode,device_cmode,mask,scal));
  PetscFunctionReturn(0);
}

#define PetscDeviceContextDestroyManagedTypeArray PetscConcat3(PetscDeviceContextDestroyManaged,PetscTypeSuffix,Array)
PetscErrorCode PetscDeviceContextDestroyManagedTypeArray(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeImpl<PetscType,PetscManagedType>::destroy(dctx,scal));
  PetscFunctionReturn(0);
}

#define PetscDeviceContextGetManagedTypeValues PetscConcat3(PetscDeviceContextGetManaged,PetscTypeSuffix,Values)
PetscErrorCode PetscDeviceContextGetManagedTypeValues(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask mask, PetscType **ptr, PetscInt *n)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeImpl<PetscType,PetscManagedType>::getvalues(dctx,scal,mask,ptr,n));
  PetscFunctionReturn(0);
}

#define PetscDeviceContextCopyManagedType PetscConcat(PetscDeviceContextCopyManaged,PetscTypeSuffix)
PetscErrorCode PetscDeviceContextCopyManagedType(PetscDeviceContext dctx, PetscManagedType dest, PetscManagedType src)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeImpl<PetscType,PetscManagedType>::copy(dctx,dest,src));
  PetscFunctionReturn(0);
}

#undef PetscType
#undef PetscManagedType
#undef PetscTypeSuffix
#undef PetscTypeSuffix_L

#undef acquiremanagedtype
#undef releasemanagedtype
#undef getmanagedvaluestype

#undef PetscConcat3
#undef PetscManagedTypeImpl
#undef PetscDeviceContextCreateManagedTypeArray
#undef PetscDeviceContextDestroyManagedTypeArray
#undef PetscDeviceContextGetManagedTypeValues
#undef PetscDeviceContextCopyManagedType
