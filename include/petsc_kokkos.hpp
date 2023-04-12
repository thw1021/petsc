#ifndef PETSC_KOKKOS_HPP
#define PETSC_KOKKOS_HPP

#include <Kokkos_Core.hpp>

extern Kokkos::DefaultExecutionSpace *PetscKokkosExecutionSpacePtr;

inline Kokkos::DefaultExecutionSpace &PetscKokkosGetExecutionSpace(void)
{
  return *PetscKokkosExecutionSpacePtr;
}

#endif
