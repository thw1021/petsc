#include <petsc/private/fortranimpl.h>
#include <petscdevice.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscdevicecontextviewfromoptions_ PETSCDEVICECONTEXTVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscdevicecontextviewfromoptions_ petscdevicecontextviewfromoptions
#endif

PETSC_EXTERN void petscdevicecontextviewfromoptions_(PetscDeviceContext *dctx, PetscObject obj, char *str, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(str, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscDeviceContextViewFromOptions(*dctx, obj, t);
  if (*ierr) return;
  FREECHAR(str, t);
}
