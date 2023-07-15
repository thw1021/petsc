#ifndef PETSC_KOKKOS_HPP
#define PETSC_KOKKOS_HPP

#if defined(__clang__)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wextra-semi-stmt"
#elif defined(__GNUC__) || defined(__GNUG__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wextra-semi"
#endif
#include <Kokkos_Core.hpp>

/* SUBMANSEC = Sys */

extern Kokkos::DefaultExecutionSpace *PetscKokkosExecutionSpacePtr;

/*MC
  PetscGetKokkosExecutionSpace - Return the Kokkos execution space that petsc is using

  Level: beginner

M*/
inline Kokkos::DefaultExecutionSpace &PetscGetKokkosExecutionSpace(void)
{
  return *PetscKokkosExecutionSpacePtr;
}

#endif
