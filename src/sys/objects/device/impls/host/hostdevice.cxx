#include "../../interface/hostdevice.hpp"

namespace Petsc
{

namespace Device
{

namespace Host
{

PetscErrorCode Device::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscDeviceInitType *defaultInitType) noexcept
{
  // host always "initializes" eagerly
  auto           initType = std::make_pair(PETSC_DEVICE_INIT_EAGER,PETSC_FALSE);
  // host is also always "device" 0
  auto           initId   = std::make_pair(0,PETSC_FALSE);
  auto           initView = std::make_pair(PETSC_FALSE,PETSC_FALSE);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = base_type::PetscOptionDeviceAll(comm,initType,initId,initView);CHKERRQ(ierr);
  if (initView.first && initView.second) {
    PetscViewer vwr;

    ierr = PetscLogInitialize();CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetStdout(comm,&vwr);CHKERRQ(ierr);
    ierr = viewDevice(nullptr,vwr);CHKERRQ(ierr);
  }
  PetscCheck(initType.first == PETSC_DEVICE_INIT_EAGER,comm,PETSC_ERR_USER_INPUT,"The host can't lazily initialize");
  PetscCheck(initId.first == 0,comm,PETSC_ERR_USER_INPUT,"The host is always device 0");
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
  const auto     vobj = PetscObjectCast(viewer);
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare(vobj,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    using buffer_type = char[256];
    buffer_type arch,hostname,username,date;
    char        pname[PETSC_MAX_PATH_LEN];
    PetscViewer sviewer;
    MPI_Comm    comm;
    PetscMPIInt rank,size;

#define BUFFER_COMMA_SIZE(buf) buf,sizeof(buf)
    ierr = PetscGetArchType(BUFFER_COMMA_SIZE(arch));CHKERRQ(ierr);
    ierr = PetscGetHostName(BUFFER_COMMA_SIZE(hostname));CHKERRQ(ierr);
    ierr = PetscGetUserName(BUFFER_COMMA_SIZE(username));CHKERRQ(ierr);
    ierr = PetscGetProgramName(BUFFER_COMMA_SIZE(pname));CHKERRQ(ierr);
    ierr = PetscGetDate(BUFFER_COMMA_SIZE(date));CHKERRQ(ierr);
#undef BUFFER_COMMA_SIZE

    ierr = PetscObjectGetComm(vobj,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    if (device) {
      // it is secretely possible to call this without a device, otherwise the initialization
      // sequence can't view from options
      ierr = PetscViewerASCIIPrintf(sviewer,"[%d] device %d\n",rank,device->deviceId);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(sviewer,"[%d] %s on a %s named %s with %d processor(s), by %s %s\n",rank,pname,arch,hostname,size,username,date);CHKERRQ(ierr);
#if PetscDefined(HAVE_OPENMP)
    ierr = PetscViewerASCIIPrintf(sviewer,"Using %" PetscInt_FMT " OpenMP threads\n",PetscNumOMPThreads);CHKERRQ(ierr);
#endif
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);

    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

} // namespace Host

} // namespace Device

} // namespace Petsc
