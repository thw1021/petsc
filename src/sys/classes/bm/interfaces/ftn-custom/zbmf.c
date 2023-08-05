#include <petsc/private/fortranimpl.h>
#include <petscbm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscbmsettype_          PETSCBMSETTYPE
  #define petscbmgettype_          PETSCBMGETTYPE
  #define petscbmsetoptionsprefix_ PETSCBMSETOPTIONSPREFIX
  #define petscbmviewfromoptions_  PETSCBMVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscbmsettype_          petscbmsettype
  #define petscbmgettype_          petscbmgettype
  #define petscbmsetoptionsprefix_ petscbmsetoptionsprefix
  #define petscbmviewfromoptions_  petscbmviewfromoptions
#endif

PETSC_EXTERN void petscbmsettype_(PetscBM *ctx, char *text, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(text, len, t);
  *ierr = PetscBMSetType(*ctx, t);
  if (*ierr) return;
  FREECHAR(text, t);
}

PETSC_EXTERN void petscbmgettype_(PetscBM *bm, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PetscBMGetType(*bm, &tname);
  if (*ierr) return;
  *ierr = PetscStrncpy(name, tname, len);
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}

PETSC_EXTERN void petscbmsetoptionsprefix_(PetscBM *ctx, char *text, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(text, len, t);
  *ierr = PetscBMSetOptionsPrefix(*ctx, t);
  if (*ierr) return;
  FREECHAR(text, t);
}

PETSC_EXTERN void petscbmviewfromoptions_(PetscBM *bm, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscBMViewFromOptions(*bm, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
