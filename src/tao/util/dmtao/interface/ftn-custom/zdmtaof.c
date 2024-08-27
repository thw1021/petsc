#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petsc/private/taoimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmtaosetobjective_            DMTAOSETOBJECTIVE
  #define dmtaosetgradient_             DMTAOSETGRADIENT
  #define dmtaosetobjectiveandgradient_ DMTAOSETOBJECTIVEANDGRADIENT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmtaosetobjective_            dmtaosetobjective
  #define dmtaosetgradient_             dmtaosetgradient
  #define dmtaosetobjectiveandgradient_ dmtaosetobjectiveandgradient
#endif

static struct {
  PetscFortranCallbackId obj;
  PetscFortranCallbackId grad;
  PetscFortranCallbackId objgrad;
  PetscFortranCallbackId prox;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  PetscFortranCallbackId function_pgiptr;
#endif
} _cb;

static PetscErrorCode ourdmtaoobjectiveroutine(DM dm, Vec x, PetscReal *f, void *ctx)
{
  PetscObjectUseFortranCallback(dm, _cb.obj, (DM *, Vec *, PetscReal *, void *, PetscErrorCode *), (&dm, &x, f, _ctx, &ierr));
}

static PetscErrorCode ourdmtaogradientroutine(DM dm, Vec x, Vec g, void *ctx)
{
  PetscObjectUseFortranCallback(dm, _cb.obj, (DM *, Vec *, Vec *, void *, PetscErrorCode *), (&dm, &x, &g, _ctx, &ierr));
}

static PetscErrorCode ourdmtaoobjectiveandgradientroutine(DM dm, Vec x, PetscReal *f, Vec g, void *ctx)
{
  PetscObjectUseFortranCallback(dm, _cb.objgrad, (DM *, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), (&dm, &x, f, &g, _ctx, &ierr));
}

PETSC_EXTERN void dmtaosetobjective_(DM *dm, void (*func)(DM *, Vec *, PetscReal *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.grad, (PetscVoidFn *)func, ctx);
  if (!*ierr) *ierr = DMTaoSetObjective(*dm, ourdmtaoobjectiveroutine, ctx);
}

PETSC_EXTERN void dmtaosetgradient_(DM *dm, void (*func)(DM *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.grad, (PetscVoidFn *)func, ctx);
  if (!*ierr) *ierr = DMTaoSetGradient(*dm, ourdmtaogradientroutine, ctx);
}

PETSC_EXTERN void dmtaosetobjectivegradient_(DM *dm, void (*func)(DM *, Vec *, PetscReal *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLFUNCTION(func);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*dm, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.objgrad, (PetscVoidFn *)func, ctx);
  if (!*ierr) *ierr = DMTaoSetObjectiveAndGradient(*dm, ourdmtaoobjectiveandgradientroutine, ctx);
}
