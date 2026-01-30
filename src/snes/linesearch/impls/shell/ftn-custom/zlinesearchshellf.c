#include <petsc/private/ftnimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define sneslinesearchshellsetapply_ SNESLINESEARCHSHELLSETAPPLY
  #define sneslinesearchshellgetapply_ SNESLINESEARCHSHELLGETAPPLY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define sneslinesearchshellsetapply_ sneslinesearchshellsetapply
  #define sneslinesearchshellgetapply_ sneslinesearchshellgetapply
#endif

PETSC_EXTERN void sneslinesearchshellgetapply_(SNESLineSearch *linesearch, void *func, void **ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  *ierr = SNESLineSearchShellGetApply(*linesearch, NULL, ctx);
}
