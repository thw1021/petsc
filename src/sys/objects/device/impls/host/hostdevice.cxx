#include "../../interface/hostdevice.hpp"

namespace Petsc
{

namespace Device
{

namespace Host
{

PetscErrorCode Device::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscDeviceInitType *defaultInitType) noexcept
{
  auto initType = std::make_pair(*defaultInitType,PETSC_FALSE);
  auto initId   = std::make_pair(*defaultDeviceId,PETSC_FALSE);
  auto initView = std::make_pair(PETSC_FALSE,PETSC_FALSE);

  PetscFunctionBegin;
  PetscCall(base_type::PetscOptionDeviceAll(comm,initType,initId,initView));
  if (initId.first == PETSC_DECIDE) initId.first = 0;
  // host should probably always be "device" 0, but we humor the user for the options query
  PetscCheck(initId.first == 0,comm,PETSC_ERR_USER_INPUT,"The host is always device 0");
  if (initView.first && initView.second) {
    PetscViewer vwr;

    PetscCall(PetscLogInitialize());
    PetscCall(PetscViewerASCIIGetStdout(comm,&vwr));
    PetscCall(viewDevice(nullptr,vwr));
  }
  *defaultDeviceId = initId.first;
  *defaultInitType = initType.first;
  PetscFunctionReturn(0);
}

PetscErrorCode Device::getDevice(PetscDevice device, PetscInt) const noexcept
{
  PetscFunctionBegin;
  // device id does not matter here all host devices are device '0'
  device->deviceId           = 0;
  device->ops->createcontext = create_;
  device->ops->configure     = this->configureDevice;
  device->ops->view          = this->viewDevice;
  PetscFunctionReturn(0);
}

PetscErrorCode Device::configureDevice(PetscDevice) noexcept { return 0; }

PetscErrorCode Device::viewDevice(PetscDevice device, PetscViewer viewer) noexcept
{
  const auto vobj = PetscObjectCast(viewer);
  PetscBool  iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare(vobj,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    using buffer_type = char[256];
    buffer_type arch,hostname,username,date;
    char        pname[PETSC_MAX_PATH_LEN];
    PetscViewer sviewer;
    MPI_Comm    comm;
    PetscMPIInt rank,size;

#define BUFFER_COMMA_SIZE(buf) buf,sizeof(buf)
    PetscCall(PetscGetArchType(BUFFER_COMMA_SIZE(arch)));
    PetscCall(PetscGetHostName(BUFFER_COMMA_SIZE(hostname)));
    PetscCall(PetscGetUserName(BUFFER_COMMA_SIZE(username)));
    PetscCall(PetscGetProgramName(BUFFER_COMMA_SIZE(pname)));
    PetscCall(PetscGetDate(BUFFER_COMMA_SIZE(date)));
#undef BUFFER_COMMA_SIZE

    PetscCall(PetscObjectGetComm(vobj,&comm));
    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCallMPI(MPI_Comm_size(comm,&size));

    PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    if (device) {
      // it is secretely possible to call this without a device, otherwise the initialization
      // sequence can't view from options
      PetscCall(PetscViewerASCIIPrintf(sviewer,"[%d] device %" PetscInt_FMT "\n",rank,device->deviceId));
    }
    PetscCall(PetscViewerASCIIPrintf(sviewer,"[%d] %s on a %s named %s with %d processor(s), by %s %s\n",rank,pname,arch,hostname,size,username,date));
#if PetscDefined(HAVE_OPENMP)
    PetscCall(PetscViewerASCIIPrintf(sviewer,"Using %" PetscInt_FMT " OpenMP threads\n",PetscNumOMPThreads));
#endif
    PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));

    PetscCall(PetscViewerFlush(viewer));
  }
  PetscFunctionReturn(0);
}

} // namespace Host

} // namespace Device

} // namespace Petsc
