#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerexodusiigetzonalvariablename_ PETSCVIEWEREXODUSIIGETZONALVARIABLENAME
  #define petscviewerexodusiigetnodalvariablename_ PETSCVIEWEREXODUSIIGETNODALVARIABLENAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define petscviewerexodusiigetzonalvariablename_ petscviewerexodusiigetzonalvariablename
  #define petscviewerexodusiigetnodalvariablename_ petscviewerexodusiigetnodalvariablename
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void petscviewerexodusiigetzonalvariablename_(PetscViewer *viewer, int *idx, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char       *tmp = NULL;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscViewerExodusIIGetZonalVariableName(v, *idx, &tmp);
  *ierr = PetscStrncpy(name, tmp, len);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}

PETSC_EXTERN void petscviewerexodusiigetnodalvariablename_(PetscViewer *viewer, int *idx, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char       *tmp = NULL;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscViewerExodusIIGetNodalVariableName(v, *idx, &tmp);
  *ierr = PetscStrncpy(name, tmp, len);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}
