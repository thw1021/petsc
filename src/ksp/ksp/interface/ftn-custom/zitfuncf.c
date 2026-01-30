#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspconvergeddefaultcreate_ KSPCONVERGEDDEFAULTCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspconvergeddefaultcreate_ kspconvergeddefaultcreate
#endif

PETSC_EXTERN void kspconvergeddefaultcreate_(PetscFortranAddr *ctx, PetscErrorCode *ierr)
{
  *ierr = KSPConvergedDefaultCreate((void **)ctx);
}
