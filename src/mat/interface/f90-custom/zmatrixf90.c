#include <petscmat.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matmpiaijgetseqaij_        MATMPIAIJGETSEQAIJ
  #define matmpiaijrestoreseqaij_    MATMPIAIJRESTORESEQAIJ
  #define matdensegetarray_          MATDENSEGETARRAY
  #define matdenserestorearray_      MATDENSERESTOREARRAY
  #define matdensegetarrayread_      MATDENSEGETARRAYREAD
  #define matdenserestorearrayread_  MATDENSERESTOREARRAYREAD
  #define matdensegetarraywrite_     MATDENSEGETARRAYWRITE
  #define matdenserestorearraywrite_ MATDENSERESTOREARRAYWRITE
  #define matdensegetcolumn_         MATDENSEGETCOLUMN
  #define matdenserestorecolumn_     MATDENSERESTORECOLUMN
  #define matseqaijgetarray_         MATSEQAIJGETARRAY
  #define matseqaijrestorearray_     MATSEQAIJRESTOREARRAY
  #define matgetghosts_              MATGETGHOSTS
  #define matgetrowij_               MATGETROWIJ
  #define matrestorerowij_           MATRESTOREROWIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matmpiaijgetseqaij_        matmpiaijgetseqaij
  #define matmpiaijrestoreseqaij_    matmpiaijrestoreseqaij
  #define matdensegetarray_          matdensegetarray
  #define matdenserestorearray_      matdenserestorearray
  #define matdensegetarrayread_      matdensegetarrayread
  #define matdenserestorearrayread_  matdenserestorearrayread
  #define matdensegetarraywrite_     matdensegetarraywrite
  #define matdenserestorearraywrite_ matdenserestorearraywrite
  #define matdensegetcolumn_         matdensegetcolumn
  #define matdenserestorecolumn_     matdenserestorecolumn
  #define matseqaijgetarray_         matseqaijgetarray
  #define matseqaijrestorearray_     matseqaijrestorearray
  #define matgetghosts_              matgetghosts
  #define matgetrowij_               matgetrowij
  #define matrestorerowij_           matrestorerowij
#endif

PETSC_EXTERN void matgetghosts_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *ghosts;
  PetscInt        N;

  *ierr = MatGetGhosts(*mat, &N, &ghosts);
  if (*ierr) return;
  *ierr = F90Array1dCreate((PetscInt *)ghosts, MPIU_INT, 1, N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdensegetarray_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m, N, lda;
  *ierr = MatDenseGetArray(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array2dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array2dCreate(fa, MPIU_SCALAR, 1, m, 1, N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearray_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArray(*mat, &fa);
}
PETSC_EXTERN void matdensegetarrayread_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  PetscInt           m, N, lda;
  *ierr = MatDenseGetArrayRead(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array2dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array2dCreate((void **)fa, MPIU_SCALAR, 1, m, 1, N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearrayread_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArrayRead(*mat, &fa);
}
PETSC_EXTERN void matdensegetarraywrite_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m, N, lda;
  *ierr = MatDenseGetArrayWrite(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, NULL, &N);
  if (*ierr) return;
  *ierr = MatDenseGetLDA(*mat, &lda);
  if (*ierr) return;
  if (m != lda) { // TODO: add F90Array2dLDACreate()
    *ierr = PetscError(((PetscObject)*mat)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "Array lda %" PetscInt_FMT " must match number of local rows %" PetscInt_FMT, lda, m);
    return;
  }
  *ierr = F90Array2dCreate(fa, MPIU_SCALAR, 1, m, 1, N, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorearraywrite_(Mat *mat, F90Array2d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreArrayWrite(*mat, &fa);
}
PETSC_EXTERN void matdensegetcolumn_(Mat *mat, PetscInt *col, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m;
  *ierr = MatDenseGetColumn(*mat, *col, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, NULL);
  if (*ierr) return;
  *ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, m, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matdenserestorecolumn_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDenseRestoreColumn(*mat, &fa);
}
PETSC_EXTERN void matseqaijgetarray_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m, n;
  *ierr = MatSeqAIJGetArray(*mat, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*mat, &m, &n);
  if (*ierr) return;
  *ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, m * n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matseqaijrestorearray_(Mat *mat, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatSeqAIJRestoreArray(*mat, &fa);
}
PETSC_EXTERN void matgetrowij_(Mat *B, PetscInt *shift, PetscBool *sym, PetscBool *blockcompressed, PetscInt *n, F90Array1d *ia, F90Array1d *ja, PetscBool *done, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad) PETSC_F90_2PTR_PROTO(jad))
{
  const PetscInt *IA, *JA;
  *ierr = MatGetRowIJ(*B, *shift, *sym, *blockcompressed, n, &IA, &JA, done);
  if (*ierr) return;
  if (!*done) return;
  *ierr = F90Array1dCreate((PetscInt *)IA, MPIU_INT, 1, *n + 1, ia PETSC_F90_2PTR_PARAM(iad));
  *ierr = F90Array1dCreate((PetscInt *)JA, MPIU_INT, 1, IA[*n], ja PETSC_F90_2PTR_PARAM(jad));
}
PETSC_EXTERN void matrestorerowij_(Mat *B, PetscInt *shift, PetscBool *sym, PetscBool *blockcompressed, PetscInt *n, F90Array1d *ia, F90Array1d *ja, PetscBool *done, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(iad) PETSC_F90_2PTR_PROTO(jad))
{
  const PetscInt *IA, *JA;
  *ierr = F90Array1dAccess(ia, MPIU_INT, (void **)&IA PETSC_F90_2PTR_PARAM(iad));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ia, MPIU_INT PETSC_F90_2PTR_PARAM(iad));
  if (*ierr) return;
  *ierr = F90Array1dAccess(ja, MPIU_INT, (void **)&JA PETSC_F90_2PTR_PARAM(jad));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ja, MPIU_INT PETSC_F90_2PTR_PARAM(jad));
  if (*ierr) return;
  *ierr = MatRestoreRowIJ(*B, *shift, *sym, *blockcompressed, n, &IA, &JA, done);
}
PETSC_EXTERN void matmpiaijgetseqaij_(Mat *mat, Mat *A, Mat *B, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt        n;
  *ierr = MatMPIAIJGetSeqAIJ(*mat, A, B, &fa);
  if (*ierr) return;
  *ierr = MatGetLocalSize(*B, NULL, &n);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matmpiaijrestoreseqaij_(Mat *mat, Mat *A, Mat *B, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
}
