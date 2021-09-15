#include <petscdevice.h>
#include <petsc/private/petscimpl.h>
#include <Kokkos_Core.hpp>

PetscBool PetscKokkosInitialized = PETSC_FALSE;

PetscErrorCode PetscKokkosFinalize_Private(void)
{
  PetscFunctionBegin;
  Kokkos::finalize();
  PetscFunctionReturn(0);
}

PetscErrorCode PetscKokkosIsInitialized_Private(PetscBool *isInitialized)
{
  PetscFunctionBegin;
  *isInitialized = Kokkos::is_initialized() ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* Initialize Kokkos if not yet */
PetscErrorCode PetscKokkosInitializeCheck(void)
{
  Kokkos::InitArguments args;
  int                   devId = -1;

  PetscFunctionBegin;
  if (!Kokkos::is_initialized()) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    /* Kokkos does not support CUDA and HIP at the same time (but we do :)) */
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = PetscDeviceContextGetGlobalContext(&dctx);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(dctx->device->deviceId,&devId);CHKERRQ(ierr);
#endif
    args.device_id   = devId;
    Kokkos::initialize(args);
    PetscBeganKokkos = PETSC_TRUE;
  }
  PetscKokkosInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}
