#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define vecsetvalue_      VECSETVALUE
  #define vecsetvaluelocal_ VECSETVALUELOCAL

#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define vecsetvalue_      vecsetvalue
  #define vecsetvaluelocal_ vecsetvaluelocal
#endif

PETSC_EXTERN void vecsetvalue_(Vec *v, PetscInt *i, PetscScalar *va, InsertMode *mode, PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses PetscCall() which has a return in it */
  *ierr = VecSetValues(*v, 1, i, va, *mode);
}
PETSC_EXTERN void vecsetvaluelocal_(Vec *v, PetscInt *i, PetscScalar *va, InsertMode *mode, PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses PetscCall() which has a return in it */
  *ierr = VecSetValuesLocal(*v, 1, i, va, *mode);
}
