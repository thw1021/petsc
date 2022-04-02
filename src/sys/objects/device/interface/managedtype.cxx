#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include "objpool.hpp"

#include <array>

template <typename PetscType, typename PetscManagedType>
class PetscManagedTypeImpl
{
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

public:
  using PoolType = Petsc::ObjectPool<PetscManagedType,PetscManagedTypeAllocator>;

private:
  using acquiremanagedtype_fptr   = PetscErrorCode(*)(PetscDeviceContext,PetscManagedType);
  using releasemanagedtype_fptr   = PetscErrorCode(*)(PetscDeviceContext,PetscManagedType);
  using getmanagedvaluestype_fptr = PetscErrorCode(*)(PetscDeviceContext,PetscManagedType,PetscOffloadMask,PetscType**);

  PETSC_CXX_COMPAT_DECL(constexpr acquiremanagedtype_fptr acquire_func_ptr(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(constexpr releasemanagedtype_fptr release_func_ptr(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(constexpr getmanagedvaluestype_fptr getvalues_func_ptr(PetscDeviceContext));

  static PoolType pool_;

public:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext,PetscManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscCopyMode,PetscOffloadMask,PetscManagedType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getvalues(PetscDeviceContext,PetscManagedType,PetscOffloadMask,PetscType**,PetscInt*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy(PetscDeviceContext,PetscManagedType,PetscManagedType));
};

template <typename PT, typename PMT>
typename PetscManagedTypeImpl<PT,PMT>::PoolType PetscManagedTypeImpl<PT,PMT>::pool_;

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<PetscType,PetscManagedType>::destroy(PetscDeviceContext dctx, PetscManagedType *scal))
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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<PetscType,PetscManagedType>::create(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask, PetscManagedType *scal))
{
  static auto firstTime = true;

  PetscFunctionBegin;
  if (host_ptr && n) PetscValidScalarPointer(host_ptr,2);
  PetscValidPointer(scal,8);
  if (firstTime) {
    auto seed = std::array<PetscManagedType,3>{};

    if (!dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
    PetscValidDeviceContext(dctx,1);
    // seed a few managed types
    for (auto& tmp : seed) PetscCall(pool_.get(tmp));
    for (auto  tmp : seed) PetscCall(destroy(dctx,&tmp));
    firstTime = false;
  }
  PetscCall(pool_.get(*scal));
  (*scal)->n       = n;
  (*scal)->host    = host_ptr;
  (*scal)->device  = device_ptr;
  (*scal)->mask    = mask;
  (*scal)->h_cmode = host_ptr ? host_cmode : PETSC_OWN_POINTER;
  (*scal)->d_cmode = device_ptr ? device_cmode : PETSC_OWN_POINTER;
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<PetscType,PetscManagedType>::getvalues(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask mask, PetscType **ptr, PetscInt *n))
{
  const auto smask = scal->mask;

  PetscFunctionBegin;
  if (!dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(scal,2);
  PetscValidPointer(ptr,4);
  if (n) PetscValidIntPointer(n,5);
  PetscAssert(mask != PETSC_OFFLOAD_UNALLOCATED,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"De-allocating the values is not a valid destination");
  // scal->mask = PETSC_OFFLOAD_BOTH is strictly a superset of anything that mask can be.
  if (smask == PETSC_OFFLOAD_BOTH || (mask == smask)) {
    // ok data is supposedly already where user wants it, note offload both returns host
    // pointer as well
    *ptr = mask == PETSC_OFFLOAD_GPU ? scal->device : scal->host;
    if (smask != PETSC_OFFLOAD_BOTH) scal->mask = mask;
  } else {
    PetscCall((*getvalues_func_ptr(dctx))(dctx,scal,mask,ptr));
  }
  if (n) *n = scal->n;
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode PetscManagedTypeImpl<PetscType,PetscManagedType>::copy(PetscDeviceContext dctx, PetscManagedType dest, PetscManagedType src))
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
  PetscCall(getvalues(dctx,dest,dmask,&dest_ptr,&dest_n));
  PetscCall(getvalues(dctx,src,dmask,&src_ptr,&src_n));
  PetscAssert(dest_n >= src_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Destination size %" PetscInt_FMT " not large enough for source size %" PetscInt_FMT,dest_n,src_n);
  PetscCall(PetscDeviceArrayCopy(dctx,dest_ptr,src_ptr,dest_n,mode));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Scalar
#define PetscTypeSuffix_L scalar
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Real
#define PetscTypeSuffix_L real
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Int
#define PetscTypeSuffix_L int
#include "managedtype.inl"
