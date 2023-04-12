#ifndef PETSC_KOKKOS_HPP
#define PETSC_KOKKOS_HPP

#include <Kokkos_Core.hpp>

extern Kokkos::DefaultExecutionSpace *PetscKokkosExecutionSpacePtr;

inline Kokkos::DefaultExecutionSpace &PetscGetKokkosExecutionSpace(void)
{
  return *PetscKokkosExecutionSpacePtr;
}

#endif
