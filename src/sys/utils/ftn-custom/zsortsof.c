#include <petsc/private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsctimsort_          PETSCTIMSORT
#define petsctimsortwitharray_ PETSCTIMSORTWITHARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsctimsort_          petsctimsort
#define petsctimsortwitharray_ petsctimsortwitharray
#endif

typedef void (*ConstCompare)(const void*,const void*,void*,int*);
struct fc_c {
  ConstCompare fcmp;
  //void (*fcmp)(const void *a, const void *b, void *c, int *res);
  void        *fctx;
};

PETSC_STATIC_INLINE int cmp_via_fortran(const void *a, const void *b, void *ctx)
{
  int result;
  struct fc_c *fc = ctx;
  fc->fcmp(a, b, fc->fctx, &result);
  return result;
}

PETSC_EXTERN void petsctimsort_(PetscInt n, PetscInt *arr, size_t size, void (*cmp)(void *, void *, void *, int *), void *ctx, PetscErrorCode *ierr)
{
  struct fc_c fc = {(ConstCompare)cmp, ctx};
  *ierr = PetscTimSort(n,(void *)arr,size,cmp_via_fortran,&fc);
}

PETSC_EXTERN void petsctimsortwitharray_(PetscInt n, void *arr, size_t asize, void *barr, size_t bsize, void (*cmp)(const void *, const void *, void *, int *), void *ctx, PetscErrorCode *ierr)
{
  struct fc_c fc = {cmp, ctx};
  *ierr = PetscTimSortWithArray(n,arr,asize,barr,bsize,cmp_via_fortran,&fc);
}
