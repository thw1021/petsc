#include <petscsys.h>
#include <petscfix.h>
#include <petscfv.h>
#include <petscds.h>
#include <petscts.h>
#include <petsc/private/fortranimpl.h>
/* wrapper_petsc.c */
/* Fortran interface file */

#ifdef PETSC_USE_POINTER_CONVERSION

#if defined(__cplusplus)
extern "C" {
#endif
extern void *PetscToPointer(void*);
extern int PetscFromPointer(void *);
extern void PetscRmPointer(void*);
#if defined(__cplusplus)
}
#endif

#else

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))
#define PetscFromPointer(a) (PetscFortranAddr)(a)
#define PetscRmPointer(a)

#endif

/* Extra wrappers for PetscFV */
/* Routine PetscFVSetComponentName */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscfvsetcomponentname_ PETSCFVSETCOMPONENTNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscfvsetcomponentname_ petscfvsetcomponentname
#endif
/* Routine PetscFVView */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscfvview_ PETSCFVVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscfvview_ petscfvview
#endif
/* Routine PetscFVSetType */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscfvsettype_ PETSCFVSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscfvsettype_ petscfvsettype
#endif
/* Routine PetscDSView */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscdsview_ PETSCDSVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscdsview_ petscdsview
#endif
/* Extra wrappers for PetscDS */
/* Routine PetscDSSetRiemannSolver */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscdssetriemannsolver_ PETSCDSSETRIEMANNSOLVER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscdssetriemannsolver_ petscdssetriemannsolver
#endif
/* Routine PetscDSAddBoundary */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscdsaddboundary_ PETSCDSADDBOUNDARY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscdsaddboundary_ petscdsaddboundary
#endif
/* Extra wrappers for TS */
/* Routine DMTSSetBoundaryLocal */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmtssetboundarylocal_ DMTSSETBOUNDARYLOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmtssetboundarylocal_ dmtssetboundarylocal
#endif
/* Routine DMTSSetBoundaryLocal */
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmtssetrhsfunctionlocal_ DMTSSETRHSFUNCTIONLOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmtssetrhsfunctionlocal_ dmtssetrhsfunctionlocal
#endif

/* Definitions of extra wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
/* Routine PetscFVSetComponentName */
PETSC_EXTERN void petscfvsetcomponentname_(PetscFV fvm, PetscInt *comp, char* name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
    char *compname;
    FIXCHAR(name, len, compname);
    *ierr = PetscFVSetComponentName((PetscFV)PetscToPointer((fvm)), *comp, compname); if (*ierr) return;
    FREECHAR(name, compname);
}
/* Routine PetscFVView */
PETSC_EXTERN void petscfvview_(PetscFV *fvm, PetscViewer *vin, PetscErrorCode *ierr)
{
    PetscViewer v;
    PetscPatchDefaultViewers_Fortran(vin, v);
    *ierr = PetscFVView(*fvm, v);
}
/* Routine PetscFVSetType */
PETSC_EXTERN void petscfvsettype_(PetscFV *fvm, char* type_name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
    char *t;
    FIXCHAR(type_name, len, t);
    *ierr = PetscFVSetType(*fvm, t); if (*ierr) return;
    FREECHAR(type_name, t);
}
/* Routine PetscDSView */
PETSC_EXTERN void petscdsview_(PetscDS *prob, PetscViewer *vin, PetscErrorCode *ierr)
{
    PetscViewer v;
    PetscPatchDefaultViewers_Fortran(vin, v);
    *ierr = PetscDSView(*prob, v);
}
/* Routine PetscDSSetRiemannSolver */
PetscFortranCallbackId riemannsolver;
static PetscErrorCode ourriemannsolver(PetscInt dim, PetscInt Nf, PetscReal x[], PetscReal n[], PetscScalar uL[], PetscScalar uR[], PetscInt numConstants, PetscScalar constants[], PetscScalar flux[], void *ctx)
{
    PetscObjectUseFortranCallback((PetscDS)ctx, riemannsolver, (PetscInt*, PetscInt*, PetscReal*, PetscReal*, PetscScalar*, PetscScalar*, PetscInt*, PetscScalar*, PetscScalar*, void*, PetscErrorCode*),
                                  (&dim, &Nf, x, n, uL, uR, &numConstants, constants, flux, ctx, &ierr));
}
PETSC_EXTERN void petscdssetriemannsolver_(PetscDS *prob, PetscInt *f,
                                           void (*rs)(PetscInt *dim, PetscInt *Nf, PetscReal x[], PetscReal n[], PetscScalar uL[], PetscScalar uR[], PetscInt *numConstants, PetscScalar constants[], PetscScalar flux[], void *ctx, PetscErrorCode *jerr),
                                           PetscErrorCode *ierr)
{
    *ierr = PetscObjectSetFortranCallback((PetscObject)*prob, PETSC_FORTRAN_CALLBACK_CLASS, &riemannsolver, (PetscVoidFunction)rs, prob);
    *ierr = PetscDSSetRiemannSolver(*prob, *f, (void*)ourriemannsolver);
}
/* Routine PetscDSAddBoundary */
PetscFortranCallbackId bocofunc, bocofunc_time;
static PetscErrorCode ourbocofunc(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, const PetscScalar *a_xG, void *ctx)
{
    PetscObjectUseFortranCallback((PetscDS)ctx, bocofunc,
                                 (PetscReal*, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, void*, PetscErrorCode*),
                                 (&time, c, n, a_xI, a_xG, ctx, &ierr));
}
static PetscErrorCode ourbocofunc_time(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, const PetscScalar *a_xG, void *ctx)
{
    PetscObjectUseFortranCallback((PetscDS)ctx, bocofunc_time,
                                 (PetscReal*, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, void*, PetscErrorCode*),
                                 (&time, c, n, a_xI, a_xG, ctx, &ierr));
}
PETSC_EXTERN void petscdsaddboundary_(PetscDS *prob, DMBoundaryConditionType *type, char *name, char *labelname, PetscInt *field, PetscInt *numcomps, PetscInt *comps,
                                      void (*bcFunc)(void),
                                      void (*bcFunc_t)(void),
                                      PetscInt *numids, const PetscInt *ids, void *ctx, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T namelen, PETSC_FORTRAN_CHARLEN_T labelnamelen)
{
    char *newname, *newlabelname;
    FIXCHAR(name, namelen, newname);
    FIXCHAR(labelname, labelnamelen, newlabelname);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*prob, PETSC_FORTRAN_CALLBACK_CLASS, &bocofunc, (PetscVoidFunction)bcFunc, prob);
    *ierr = PetscObjectSetFortranCallback((PetscObject)*prob, PETSC_FORTRAN_CALLBACK_CLASS, &bocofunc_time, (PetscVoidFunction)bcFunc_t, prob);
    *ierr = PetscDSAddBoundary(*prob, *type, newname, newlabelname, *field, *numcomps, comps, (void (*)(void))ourbocofunc, (void (*)(void))ourbocofunc_time, *numids, ids, ctx);
    FREECHAR(name, newname);
    FREECHAR(labelname, newlabelname);
}
/* Routine DMTSSetBoundaryLocal */
PetscFortranCallbackId dmtsbocofunc;
static PetscErrorCode ourdmtsbocofunc(DM dm, PetscReal time, Vec locX, Vec locX_t, void *ctx)
{
    PetscObjectUseFortranCallback((DM)ctx, dmtsbocofunc,
                                  (DM*, PetscReal*, Vec*, Vec*, void*, PetscErrorCode*),
                                  (&dm, &time, &locX, &locX_t, ctx, &ierr));
}
PETSC_EXTERN void dmtssetboundarylocal_(DM *dm,
                                        void (*func)(DM *dm, PetscReal *time, Vec *locX, Vec *locX_t, void *context, PetscErrorCode *error),
                                        void *ctx, PetscErrorCode *ierr)
{
    *ierr = PetscObjectSetFortranCallback((PetscObject)*dm, PETSC_FORTRAN_CALLBACK_CLASS, &dmtsbocofunc, (PetscVoidFunction)func, dm);
    *ierr = DMTSSetBoundaryLocal(*dm, ourdmtsbocofunc, ctx);
}
/* Routine DMTSSetRHSFunctionLocal */
PetscFortranCallbackId dmtsrhsfunc;
static PetscErrorCode ourdmtsrhsfunc(DM dm, PetscReal time, Vec locX, Vec F, void *ctx)
{
    PetscObjectUseFortranCallback((DM)ctx, dmtsrhsfunc,
                                  (DM*, PetscReal*, Vec*, Vec*, void*, PetscErrorCode*),
                                  (&dm, &time, &locX, &F, ctx, &ierr));
}
PETSC_EXTERN void dmtssetrhsfunctionlocal_(DM *dm,
                                           void (*func)(DM *dm, PetscReal *time, Vec *locX, Vec *F, void *context, PetscErrorCode *error),
                                           void *ctx, PetscErrorCode *ierr)
{
    *ierr = PetscObjectSetFortranCallback((PetscObject)*dm, PETSC_FORTRAN_CALLBACK_CLASS, &dmtsrhsfunc, (PetscVoidFunction)func, dm);
    *ierr = DMTSSetRHSFunctionLocal(*dm, ourdmtsrhsfunc, ctx);
}
#if defined(__cplusplus)
}
#endif