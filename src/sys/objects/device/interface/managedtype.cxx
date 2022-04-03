#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include "objpool.hpp"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Scalar
#define PetscTypeSuffix_L scalar
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Real
#define PetscTypeSuffix_L real
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Int
#define PetscTypeSuffix_L int
#include "managedtype.inl"
