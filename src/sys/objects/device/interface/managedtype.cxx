#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include "objpool.hpp"

#include <array>

template <typename PetscType, typename PetscManagedType>
class PetscManagedTypeImpl
{
  struct PetscManagedTypeAllocator : Petsc::AllocatorBase<PetscManagedType>
  {
    PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscManagedType mscal))
    {
      using Petsc::util::integral_value;

      PetscFunctionBegin;
      PetscCall(PetscNew(mscal));
      mscal->cmode = PETSC_OWN_POINTER;
      static_assert(integral_value(PETSC_OWN_POINTER) != 0,"");
      static_assert(integral_value(PETSC_MEMTYPE_HOST) == 0,"");
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
      mscal->n      = 0;
      mscal->host   = nullptr;
      mscal->device = nullptr;
      mscal->mtype  = PETSC_MEMTYPE_HOST;
      mscal->mask   = PETSC_OFFLOAD_UNALLOCATED;
      mscal->cmode  = PETSC_OWN_POINTER;
      PetscFunctionReturn(0);
    }

    PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) { return 0; }
  };

  using pool_type               = Petsc::ObjectPool<PetscManagedType,PetscManagedTypeAllocator>;
  using acquiremanagedtype_fptr = PetscErrorCode(*)(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscMemType,PetscOffloadMask,PetscManagedType);
  using releasemanagedtype_fptr = PetscErrorCode(*)(PetscDeviceContext,PetscManagedType);
  using getmanagedvaluestype_fptr = PetscErrorCode(*)(PetscDeviceContext,PetscManagedType,PetscOffloadMask,PetscType**);

  PETSC_CXX_COMPAT_DECL(constexpr acquiremanagedtype_fptr acquire_func_ptr(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(constexpr releasemanagedtype_fptr release_func_ptr(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(constexpr getmanagedvaluestype_fptr getvalues_func_ptr(PetscDeviceContext));

  pool_type pool_;

public:
  PETSC_NODISCARD PetscErrorCode destroy(PetscDeviceContext,PetscManagedType*) noexcept;
  PETSC_NODISCARD PetscErrorCode create(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscMemType,PetscOffloadMask,PetscManagedType*) noexcept;
  PETSC_NODISCARD PetscErrorCode getvalues(PetscDeviceContext,PetscManagedType,PetscOffloadMask,PetscType**) noexcept;
};

template <typename PetscType, typename PetscManagedType>
PetscErrorCode PetscManagedTypeImpl<PetscType,PetscManagedType>::destroy(PetscDeviceContext dctx, PetscManagedType *scal) noexcept
{
  PetscFunctionBegin;
  if (!dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscValidDeviceContext(dctx,1);
  if (*scal) PetscFunctionReturn(0);
  PetscCall((*release_func_ptr(dctx))(dctx,*scal));
  PetscCall(pool_.reclaim(std::move(*scal)));
  *scal = nullptr;
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PetscErrorCode PetscManagedTypeImpl<PetscType,PetscManagedType>::create(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode cmode, PetscMemType mtype, PetscOffloadMask mask, PetscManagedType *scal) noexcept
{
  static auto firstTime = true;

  PetscFunctionBegin;
  if (!dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscValidDeviceContext(dctx,1);
  if (host_ptr && n) PetscValidScalarPointer(host_ptr,2);
  PetscValidPointer(scal,8);
  if (firstTime) {
    auto seed = std::array<PetscManagedType,3>{};

    // seed a few managed types
    for (auto& tmp : seed) PetscCall(pool_.get(tmp));
    for (auto  tmp : seed) PetscCall(destroy(dctx,&tmp));
    firstTime = false;
  }
  PetscCall(pool_.get(*scal));
  PetscCall((*acquire_func_ptr(dctx))(dctx,host_ptr,device_ptr,n,cmode,mtype,mask,*scal));
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PetscErrorCode PetscManagedTypeImpl<PetscType,PetscManagedType>::getvalues(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask mask, PetscType **ptr) noexcept
{
  PetscFunctionBegin;
  if (!dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(ptr,4);
  PetscCall((*getvalues_func_ptr(dctx))(dctx,scal,mask,ptr));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */

using PetscManagedScalarImplType = PetscManagedTypeImpl<PetscScalar,PetscManagedScalar>;
template <>
constexpr auto PetscManagedScalarImplType::acquire_func_ptr(PetscDeviceContext dctx) noexcept PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->acquiremanagedscalar);
template <>
constexpr auto PetscManagedScalarImplType::release_func_ptr(PetscDeviceContext dctx) noexcept PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->releasemanagedscalar);
template <>
constexpr auto PetscManagedScalarImplType::getvalues_func_ptr(PetscDeviceContext dctx) noexcept PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->getmanagedvaluesscalar);

static auto PetscManagedScalarImpl = PetscManagedScalarImplType{ };

PetscErrorCode PetscDeviceContextCreateManagedScalarArray(PetscDeviceContext dctx, PetscScalar *host_ptr, PetscScalar *device_ptr, PetscInt n, PetscCopyMode cmode, PetscMemType mtype, PetscOffloadMask mask, PetscManagedScalar *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedScalarImpl.create(dctx,host_ptr,device_ptr,n,cmode,mtype,mask,scal));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextDestroyManagedScalarArray(PetscDeviceContext dctx, PetscManagedScalar *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedScalarImpl.destroy(dctx,scal));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextGetManagedScalarValues(PetscDeviceContext dctx, PetscManagedScalar scal, PetscOffloadMask mask, PetscScalar **ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedScalarImpl.getvalues(dctx,scal,mask,ptr));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */

using PetscManagedRealImplType = PetscManagedTypeImpl<PetscReal,PetscManagedReal>;
template <>
constexpr auto PetscManagedRealImplType::acquire_func_ptr(PetscDeviceContext dctx) noexcept PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->acquiremanagedreal);
template <>
constexpr auto PetscManagedRealImplType::release_func_ptr(PetscDeviceContext dctx) noexcept PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->releasemanagedreal);
template <>
constexpr auto PetscManagedRealImplType::getvalues_func_ptr(PetscDeviceContext dctx) noexcept PETSC_DECLTYPE_AUTO_RETURNS(dctx->ops->getmanagedvaluesreal);

static auto PetscManagedRealImpl = PetscManagedTypeImpl<PetscReal,PetscManagedReal>{ };

PetscErrorCode PetscDeviceContextCreateManagedRealArray(PetscDeviceContext dctx, PetscReal *host_ptr, PetscReal *device_ptr, PetscInt n, PetscCopyMode cmode, PetscMemType mtype, PetscOffloadMask mask, PetscManagedReal *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedRealImpl.create(dctx,host_ptr,device_ptr,n,cmode,mtype,mask,scal));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextDestroyManagedRealArray(PetscDeviceContext dctx, PetscManagedReal *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedRealImpl.destroy(dctx,scal));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextGetManagedRealValues(PetscDeviceContext dctx, PetscManagedReal scal, PetscOffloadMask mask, PetscReal **ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedRealImpl.getvalues(dctx,scal,mask,ptr));
  PetscFunctionReturn(0);
}
