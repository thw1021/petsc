#include <petsc/private/ftnimpl.h>
#include <petscdmshell.h> /*I    "petscdmshell.h"  I*/

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmshellsetcreatefielddecomposition_ DMSHELLSETCREATEFIELDDECOMPOSITION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmshellsetcreatefielddecomposition_ dmshellsetcreatefielddecomposition
#endif

/*
 * C routines are required for matrix and global vector creation.  We define C routines here that call the corresponding
 * Fortran routine (indexed by _cb) that was set by the user.
 */

static struct {
  PetscFortranCallbackId createfielddecomposition;
} _cb;

static PetscErrorCode ourcreatefielddecomposition(DM dm, PetscInt *nfields, char ***names, IS **is, DM **subdms)
{
  PetscObjectUseFortranCallbackSubType(dm, _cb.createfielddecomposition, (DM *, PetscInt *, char ***, IS **, DM **, PetscErrorCode *), (&dm, nfields, names, is, subdms, &ierr));
}

PETSC_EXTERN void dmshellsetcreatefielddecomposition_(DM *dm, void (*func)(DM *, PetscInt *, char ***, IS **, DM **, PetscErrorCode *), PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm, PETSC_FORTRAN_CALLBACK_SUBTYPE, &_cb.createfielddecomposition, (PetscVoidFn *)func, NULL);
  if (*ierr) return;
  *ierr = DMShellSetCreateFieldDecomposition(*dm, ourcreatefielddecomposition);
}
