#ifndef HOSTCONTEXT_HPP
#define HOSTCONTEXT_HPP

#include "../../interface/hostdevice.hpp"

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
  PETSC_CXX_COMPAT_DECL(PetscErrorCode arrayCopy(PetscDeviceContext,void*PETSC_RESTRICT,const void*PETSC_RESTRICT,std::size_t,PetscDeviceCopyMode))
  { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented"); }

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
    arrayCopy
  };
};

}

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
