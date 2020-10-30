#if !defined(__PETSCDMDA_KOKKOS_HPP)
#define __PETSCDMDA_KOKKOS_HPP

#include <petscdmda.h>
#include <petscvec_kokkos.hpp>

#if defined(PETSC_HAVE_KOKKOS)

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView(DM,Vec,PetscScalarKokkosOffsetView1DType<MemorySpace>*,PetscBool overwrite=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,PetscScalarKokkosOffsetView1DType<MemorySpace>*,PetscBool overwrite=PETSC_FALSE);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView(DM,Vec,ConstPetscScalarKokkosOffsetView1DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,ConstPetscScalarKokkosOffsetView1DType<MemorySpace>*);

#endif

#endif
