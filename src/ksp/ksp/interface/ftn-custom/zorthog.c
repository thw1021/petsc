#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define kspsetorthogonalization_                  KSPSETORTHOGONALIZATION
  #define kspmodifiedgramschmidtorthogonalization_  KSPMODIFIEDGRAMSCHMIDTORTHOGONALIZATION
  #define kspclassicalgramschmidtorthogonalization_ KSPCLASSICALGRAMSCHMIDTORTHOGONALIZATION
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define kspsetorthogonalization_                  kspsetorthogonalization
  #define kspmodifiedgramschmidtorthogonalization_  kspmodifiedgramschmidtorthogonalization
  #define kspclassicalgramschmidtorthogonalization_ kspclassicalgramschmidtorthogonalization
#endif

static struct {
  PetscFortranCallbackId orthog;
} _cb;

PETSC_EXTERN void kspmodifiedgramschmidtorthogonalization_(KSP *, Vec *, PetscInt *, Vec, PetscScalar *, PetscErrorCode *);
PETSC_EXTERN void kspclassicalgramschmidtorthogonalization_(KSP *, Vec *, PetscInt *, Vec, PetscScalar *, PetscErrorCode *);

static PetscErrorCode ourorthog(KSP ksp, Vec V[], PetscInt n, Vec x, PetscScalar h[])
{
  PetscObjectUseFortranCallback(ksp, _cb.orthog, (KSP *, Vec * V, PetscInt *n, Vec x, PetscScalar *h, PetscErrorCode *), (&ksp, V, &n, x, h, &ierr));
}

PETSC_EXTERN void kspsetorthogonalization_(KSP *ksp, void (*orthog)(KSP *, Vec *, PetscInt *, Vec, PetscScalar *, PetscErrorCode *), PetscErrorCode *ierr)
{
  if (orthog == kspmodifiedgramschmidtorthogonalization_) {
    *ierr = KSPSetOrthogonalization(*ksp, KSPModifiedGramSchmidtOrthogonalization);
  } else if (orthog == kspclassicalgramschmidtorthogonalization_) {
    *ierr = KSPSetOrthogonalization(*ksp, KSPClassicalGramSchmidtOrthogonalization);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*ksp, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.orthog, (PetscFortranCallbackFn *)orthog, NULL);
    if (*ierr) return;
    *ierr = KSPSetOrthogonalization(*ksp, ourorthog);
  }
}
