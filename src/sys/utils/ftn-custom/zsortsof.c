#include <petsc/private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsctimsort_          PETSCTIMSORT
#define petsctimsortwitharray_ PETSCTIMSORTWITHARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsctimsort_          petsctimsort
#define petsctimsortwitharray_ petsctimsortwitharray
#endif

PETSC_EXTERN void petsctimsort_(PetscInt n, void *arr, size_t size, int (*cmp)(void *, void *), PetscErrorCode *ierr)
{
  *ierr = PetscTimSort(n,arr,size,cmp);
}

PETSC_EXTERN void petsctimsortwitharray_(PetscInt n, void *arr, size_t asize, void *barr, size_t bsize, int (*cmp)(void *, void *), PetscErrorCode *ierr)
{
  *ierr = PetscTimSortWithArray(n,arr,asize,barr,bszie,cmp);
}
