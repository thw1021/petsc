
#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matsetpreallocationcoo32_      MATSETPREALLOCATIONCOO32
  #define matsetpreallocationcoolocal32_ MATSETPREALLOCATIONCOOLOCAL32
  #define matsetpreallocationcoo64_      MATSETPREALLOCATIONCOO64
  #define matsetpreallocationcoolocal64_ MATSETPREALLOCATIONCOOLOCAL64
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matsetpreallocationcoo32_      matsetpreallocationcoo32
  #define matsetpreallocationcoolocal32_ matsetpreallocationcoolocal32
  #define matsetpreallocationcoo64       matsetpreallocationcoo64
  #define matsetpreallocationcoolocal64_ matsetpreallocationcoolocal64
#endif

PETSC_EXTERN void matsetpreallocationcoo32_(Mat *A, int *ncoo, PetscInt coo_i[], PetscInt coo_j[], int *ierr)
{
  *ierr = MatSetPreallocationCOO(*A, *ncoo, coo_i, coo_j);
}

PETSC_EXTERN void matsetpreallocationcoo64_(Mat *A, PetscInt64 *ncoo, PetscInt coo_i[], PetscInt coo_j[], int *ierr)
{
  *ierr = MatSetPreallocationCOO(*A, *ncoo, coo_i, coo_j);
}

PETSC_EXTERN void matsetpreallocationcoolocal32_(Mat *A, int *ncoo, PetscInt coo_i[], PetscInt coo_j[], int *ierr)
{
  *ierr = MatSetPreallocationCOOLocal(*A, *ncoo, coo_i, coo_j);
}

PETSC_EXTERN void matsetpreallocationcoolocal64_(Mat *A, PetscInt64 *ncoo, PetscInt coo_i[], PetscInt coo_j[], int *ierr)
{
  *ierr = MatSetPreallocationCOOLocal(*A, *ncoo, coo_i, coo_j);
}
