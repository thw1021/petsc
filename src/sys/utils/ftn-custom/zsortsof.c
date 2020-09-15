#include <petsc/private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsctimsort_          PETSCTIMSORT
#define petsctimsortwitharray_ PETSCTIMSORTWITHARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsctimsort_          petsctimsort
#define petsctimsortwitharray_ petsctimsortwitharray
#endif

int cmp_via_fortran(void *a, void *b, void *ctx)
{
  int result;
  struct {
    void (*f_)(void *a, void *b, void *c, int *res);
    void *fctx;
  } *fc = ctx;
  fc->f_(a, b, fc->fctx, &result);
  return result;
}

PETSC_EXTERN void petsctimsort_(PetscInt n, void *arr, size_t size, void (*cmp)(void *, void *), void *, PetscErrorCode *ierr)
{

  *ierr = PetscTimSort(n,arr,size,cmp);
}

PETSC_EXTERN void petsctimsortwitharray_(PetscInt n, void *arr, size_t asize, void *barr, size_t bsize, void (*cmp)(void *, void *), void *, PetscErrorCode *ierr)
{
  *ierr = PetscTimSortWithArray(n,arr,asize,barr,bszie,cmp);
}
