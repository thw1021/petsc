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

#define PETSC_AND_KOKKOS_HAVE(THING)                                    \
  (defined(KOKKOS_ENABLE_ ## THING) && PetscDefined(PETSCPP_CONCAT(HAVE_,THING)))

/* Initialize Kokkos if not yet */
PetscErrorCode PetscKokkosInitializeCheck(void)
{
  PetscFunctionBegin;
  if (!Kokkos::is_initialized()) {
    Kokkos::InitArguments args; /* use default constructor */

    args.disable_warnings = PetscDefined(HAVE_KOKKOS_INIT_WARNINGS);
#if PETSC_AND_KOKKOS_HAVE(CUDA) || PETSC_AND_KOKKOS_HAVE(HIP)
    /* Kokkos does not support CUDA and HIP at the same time (but we do :)) */
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(dctx->device->deviceId,&args.device_id);CHKERRQ(ierr);
#endif

    /* To use PetscNumOMPThreads, one has to configure petsc --with-openmp.
       Otherwise, let's keep the default value (-1) of args.num_threads.
    */
#if PETSC_AND_KOKKOS_HAVE(OPENMP)
    args.num_threads = PetscNumOMPThreads;
#endif

    Kokkos::initialize(args);
    PetscBeganKokkos = PETSC_TRUE;
  }
  PetscKokkosInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}
