#ifndef HOSTCONTEXT_HPP
#define HOSTCONTEXT_HPP

#include "../../interface/hostdevice.hpp"
#include "../impldevicecontextbase.hpp"

namespace Petsc
{

namespace Device
{

namespace Host
{

namespace Impl
{

struct DeviceContext
{
private:
  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode malloc_wrapper(PetscType **ptr, std::size_t n))
  {
    PetscFunctionBegin;
    PetscCall(PetscMalloc1(n,ptr));
    PetscFunctionReturn(0);
  }

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode free_wrapper(PetscType *ptr))
  {
    PetscFunctionBegin;
    PetscCall(PetscFree(ptr));
    PetscFunctionReturn(0);
  }

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(auto managed_pool_()) -> decltype(Petsc::Device::Impl::make_segmented_memory_pool<PetscType>(malloc_wrapper<PetscType>,free_wrapper<PetscType>))&
  {
    static auto pool = Petsc::Device::Impl::make_segmented_memory_pool<PetscType>(
      malloc_wrapper<PetscType>,free_wrapper<PetscType>
    );
    return pool;
  }

public:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext))
  { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode changeStreamType(PetscDeviceContext,PetscStreamType))
  { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setUp(PetscDeviceContext))
  { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode query(PetscDeviceContext,PetscBool *idle))
  {
    PetscFunctionBegin;
    *idle = PETSC_TRUE; // the host is always idle
    PetscFunctionReturn(0);
  }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waitForContext(PetscDeviceContext,PetscDeviceContext))
  { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode synchronize(PetscDeviceContext))
  { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getBlasHandle(PetscDeviceContext,void*))
  { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getSolverHandle(PetscDeviceContext,void*))
  { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getStreamHandle(PetscDeviceContext,void*))
  { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode beginTimer(PetscDeviceContext))
  { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode endTimer(PetscDeviceContext,PetscLogDouble*))
  { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode arrayCopy(PetscDeviceContext,void*PETSC_RESTRICT,const void*PETSC_RESTRICT,std::size_t,PetscDeviceCopyMode));

  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroyManagedType(PetscDeviceContext,PetscManagedType));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getManagedTypeValues(PetscDeviceContext,PetscManagedType,PetscMemType,PetscMemoryAccessMode,PetscType**));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode applyOperatorType(PetscDeviceContext,PetscManagedType,PetscOperatorType,PetscMemType,const PetscType*,PetscManagedType));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode launchHostFunction(PetscDeviceContext,PetscHostFunction,void*));

  const struct _DeviceContextOps ops = {
    destroy,
    changeStreamType,
    setUp,
    query,
    waitForContext,
    synchronize,
    getBlasHandle,
    getSolverHandle,
    getStreamHandle,
    beginTimer,
    endTimer,
    arrayCopy,
    destroyManagedType<PetscScalar,PetscManagedScalar>,
    getManagedTypeValues<PetscScalar,PetscManagedScalar>,
    applyOperatorType<PetscScalar,PetscManagedScalar>,
    destroyManagedType<PetscReal,PetscManagedReal>,
    getManagedTypeValues<PetscReal,PetscManagedReal>,
    applyOperatorType<PetscReal,PetscManagedReal>,
    destroyManagedType<PetscInt,PetscManagedInt>,
    getManagedTypeValues<PetscInt,PetscManagedInt>,
    applyOperatorType<PetscInt,PetscManagedInt>,
    launchHostFunction
  };
};

PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::arrayCopy(PetscDeviceContext, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, PetscDeviceCopyMode mode))
{
  PetscFunctionBegin;
  PetscCheck(mode == PETSC_DEVICE_COPY_HTOH,PETSC_COMM_SELF,PETSC_ERR_SUP,"Host device context can only copy host-to-host");
  PetscCall(PetscMemcpy(dest,src,n));
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::destroyManagedType(PetscDeviceContext, PetscManagedType scal))
{
  PetscFunctionBegin;
  PetscCall(managed_pool_<PetscType>().release(&scal->host));
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::getManagedTypeValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr))
{
  auto& mask = scal->mask;
  auto& sptr = scal->host;

  PetscFunctionBegin;
  PetscAssert(mtype == PETSC_MEMTYPE_HOST,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Host device can only access host memory from a managed scalar");
  PetscAssert(PetscOffloadHost(mask) || PetscOffloadUnallocated(mask),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Host device can only manage scalars on the host");
  PetscAssert(scal->dtype == PETSC_DEVICE_HOST,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Host device incompatible with managed scalar with device type %s",PetscDeviceTypes[scal->dtype]);
  // the only values we can "get" is the host pointer
  if (!sptr) PetscCall(managed_pool_<PetscType>().get(scal->n,&sptr));
  *ptr = sptr;
  mask = PETSC_OFFLOAD_CPU; // a host managed scalar is always offloaded on host
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::applyOperatorType(PetscDeviceContext, PetscManagedType scal, PetscOperatorType, PetscMemType, const PetscType*, PetscManagedType))
{
  PetscFunctionBegin;
  // we should never get here
  PetscAssert(PetscOffloadHost(scal->mask),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Host device can only manage scalars on the host");
  PetscAssert(scal->dtype == PETSC_DEVICE_HOST,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Host device incompatible with managed scalar with device type %s",PetscDeviceTypes[scal->dtype]);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Host apply operator can only apply a rhs from host memory");
}

PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::launchHostFunction(PetscDeviceContext dctx, PetscHostFunction func, void *ctx))
{
  PetscFunctionBegin;
  // the host variant of "launchHostFunction" just executes the function
  PetscCall(func(dctx,ctx));
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace Host

} // namespace Device

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_HOST(PetscDeviceContext dctx)
{
  static constexpr auto hostctx = Petsc::Device::Host::Impl::DeviceContext{};

  PetscFunctionBegin;
  PetscCall(PetscArraycpy(dctx->ops,&hostctx.ops,1));
  PetscFunctionReturn(0);
}

#endif // HOSTCONTEXT_HPP
