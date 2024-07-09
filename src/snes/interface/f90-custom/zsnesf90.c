#include <petscsnes.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define snesgetconvergencehistory_     SNESGETCONVERGENCEHISTORY
  #define snesrestoreconvergencehistory_ SNESRESTORECONVERGENCEHISTORY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define snesgetconvergencehistory_     snesgetconvergencehistory
  #define snesrestoreconvergencehistory_ snesrestoreconvergencehistory
#endif

PETSC_EXTERN void snesgetconvergencehistory_(SNES *snes, F90Array1d *r, F90Array1d *fits, PetscInt *n, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscReal *hist;
  PetscInt  *its;
  *ierr = SNESGetConvergenceHistory(*snes, &hist, &its, n);
  if (*ierr) return;
  *ierr = F90Array1dCreate(hist, MPIU_REAL, 1, *n, r PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dCreate(its, MPIU_INT, 1, *n, fits PETSC_F90_2PTR_PARAM(ptrd2));
}

PETSC_EXTERN void snesrestoreconvergencehistory_(SNES *snes, F90Array1d *r, F90Array1d *fits, PetscInt *n, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  *ierr = F90Array1dDestroy(r, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(fits, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd2));
}
