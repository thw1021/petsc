#include <petsc/private/ftnimpl.h>
#include <petsc/private/taoimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define taogetconvergencehistory_ TAOGETCONVERGENCEHISTORY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define taogetconvergencehistory_ taogetconvergencehistory
#endif

PETSC_EXTERN void taogetconvergencehistory_(Tao *tao, PetscInt *nhist, PetscErrorCode *ierr)
{
  *ierr = TaoGetConvergenceHistory(*tao, NULL, NULL, NULL, NULL, nhist);
}
