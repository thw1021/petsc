#include <petsc/private/deviceimpl.h>
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

#define PETSC_AND_KOKKOS_HAVE(CUPM) (defined(KOKKOS_ENABLE_##CUPM) && PetscDefined(HAVE_##CUPM))

/* Initialize Kokkos if not yet */
PetscErrorCode PetscKokkosInitializeCheck(void)
{
  Kokkos::InitArguments args;
  int                   devId = -1;

  PetscFunctionBegin;
  if (!Kokkos::is_initialized()) {
#if PETSC_AND_KOKKOS_HAVE(CUDA) || PETSC_AND_KOKKOS_HAVE(HIP)
    /* Kokkos does not support CUDA and HIP at the same time (but we do :)) */
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(dctx->device->deviceId,&devId);CHKERRQ(ierr);
#endif
    args.device_id   = devId;
    Kokkos::initialize(args);
    PetscBeganKokkos = PETSC_TRUE;
  }
  PetscKokkosInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}
