#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#include <petscmacros.h> // PetscDefined()

#include "vecseqcupm_fwd.hpp"

#if PetscDefined(USE_SINGLE_LIBRARY)
  #if defined(PETSC_INSTANTIATE_VECSEQ_CUPM_NAME)
    #include "vecseqcupm_impl.hpp"
template class ::Petsc::vec::cupm::impl::VecSeq_CUPM<::Petsc::device::cupm::DeviceType::PETSC_INSTANTIATE_VECSEQ_CUPM_NAME>;
  #endif
#else
  #include "vecseqcupm_impl.hpp"
#endif

#endif // PETSCVECSEQCUPM_HPP
