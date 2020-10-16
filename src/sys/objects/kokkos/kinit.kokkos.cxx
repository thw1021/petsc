#include <petscsys.h>
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
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  PetscErrorCode        ierr;
#endif
  Kokkos::InitArguments args;
  int                   devId = -1;

  PetscFunctionBegin;
#if defined(KOKKOS_ENABLE_CUDA)
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
#elif defined(KOKKOS_ENABLE_HIP)
  ierr = PetscHIPInitializeCheck();CHKERRQ(ierr);
#endif
  if (!Kokkos::is_initialized()) {
    ierr = PetscKokkosInitialize_Private();CHKERRQ(ierr);
    PetscBeganKokkos = PETSC_TRUE;
  }
  PetscKokkosInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}
