#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerexodusiisetzonalvariablename_ PETSCVIEWEREXODUSIISETZONALVARIABLENAME
  #define petscviewerexodusiisetnodalvariablename_ PETSCVIEWEREXODUSIISETNODALVARIABLENAME
  #define petscviewerexodusiigetzonalvariablename_ PETSCVIEWEREXODUSIIGETZONALVARIABLENAME
  #define petscviewerexodusiigetnodalvariablename_ PETSCVIEWEREXODUSIIGETNODALVARIABLENAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define petscviewerexodusiisetzonalvariablename_ petscviewerexodusiisetzonalvariablename
  #define petscviewerexodusiisetnodalvariablename_ petscviewerexodusiisetnodalvariablename
  #define petscviewerexodusiigetzonalvariablename_ petscviewerexodusiigetzonalvariablename
  #define petscviewerexodusiigetnodalvariablename_ petscviewerexodusiigetnodalvariablename
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void petscviewerexodusiisetzonalvariablename_(PetscViewer *viewer, int *idx, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char       *t1;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  FIXCHAR(name, len, t1);
  *ierr = PetscViewerExodusIISetZonalVariableName(v, *idx, t1);
  if (*ierr) return;
  FREECHAR(name, t1);
}

PETSC_EXTERN void petscviewerexodusiisetnodalvariablename_(PetscViewer *viewer, int *idx, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char       *t1;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  FIXCHAR(name, len, t1);
  *ierr = PetscViewerExodusIISetNodalVariableName(v, *idx, t1);
  if (*ierr) return;
  FREECHAR(name, t1);
}

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
