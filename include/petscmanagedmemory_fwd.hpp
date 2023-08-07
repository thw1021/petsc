#ifndef PETSCMANAGEDMEMORY_FWD_HPP
#define PETSCMANAGEDMEMORY_FWD_HPP

#include <petscsystypes.h> // PetscReal, PetscScalar
#include <petscmacros.h>   // PETSC_NODISCARD, PETSC_CXX_EXTERN

namespace Petsc
{

template <typename>
class ManagedMemory;

using ManagedReal   = ManagedMemory<PetscReal>;
using ManagedScalar = ManagedMemory<PetscScalar>;

PETSC_NODISCARD PETSC_CXX_EXTERN const ManagedScalar &MANAGED_SCAL_ONE() noexcept;
PETSC_NODISCARD PETSC_CXX_EXTERN const ManagedReal   &MANAGED_REAL_ONE() noexcept;
PETSC_NODISCARD PETSC_CXX_EXTERN const ManagedScalar &MANAGED_SCAL_MINUS_ONE() noexcept;
PETSC_NODISCARD PETSC_CXX_EXTERN const ManagedReal   &MANAGED_REAL_MINUS_ONE() noexcept;
PETSC_NODISCARD PETSC_CXX_EXTERN const ManagedScalar &MANAGED_SCAL_ZERO() noexcept;
PETSC_NODISCARD PETSC_CXX_EXTERN const ManagedReal   &MANAGED_REAL_ZERO() noexcept;

} // namespace Petsc

#endif // PETSCMANAGEDMEMORY_FWD_HPP
