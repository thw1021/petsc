#ifndef PETSCVECMPICUPM_HPP
#define PETSCVECMPICUPM_HPP

#include <petscmacros.h> // PetscDefined()

#include "vecmpicupm_fwd.hpp"

#if PetscDefined(USE_SINGLE_LIBRARY)
  #if defined(PETSC_INSTANTIATE_VECMPI_CUPM_NAME)
    #include "vecmpicupm_impl.hpp"
template class ::Petsc::vec::cupm::impl::VecMPI_CUPM<::Petsc::device::cupm::DeviceType::PETSC_INSTANTIATE_VECMPI_CUPM_NAME>;
  #endif
#else
  #include "vecmpicupm_impl.hpp"
#endif

#endif // PETSCVECMPICUPM_HPP
