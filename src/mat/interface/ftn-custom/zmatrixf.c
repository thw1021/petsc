#include <petsc/private/fortranimpl.h>
#include <petsc/private/f90impl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matgetvalues_                MATGETVALUES
  #define matgetvalues0_               MATGETVALUES0
  #define matgetvaluesnn1_             MATGETVALUESnn1
  #define matgetvaluesnnnn_            MATGETVALUESnnnn
  #define matgetvalues11_              MATGETVALUES11
  #define matgetvalues11a_             MATGETVALUES11A
  #define matgetvalues1n_              MATGETVALUES1N
  #define matgetvaluesn1_              MATGETVALUESN1
  #define matgetvalueslocal_           MATGETVALUESLOCAL
  #define matgetvalueslocal0_          MATGETVALUESLOCAL0
  #define matgetvalueslocalnn1_        MATGETVALUESLOCALNN1
  #define matgetvalueslocalnnnn_       MATGETVALUESLOCALNNNN
  #define matgetvalueslocal11_         MATGETVALUESLOCAL11
  #define matgetvalueslocal11a_        MATGETVALUESLOCAL11A
  #define matgetvalueslocal1n_         MATGETVALUESLOCAL1N
  #define matgetvalueslocaln1_         MATGETVALUESLOCALN1
  #define matsetvalues_                MATSETVALUES
  #define matsetvaluesnnnn_            MATSETVALUESNNNN
  #define matsetvalues0_               MATSETVALUES0
  #define matsetvaluesnn1_             MATSETVALUESNN1
  #define matsetvalues11_              MATSETVALUES11
  #define matsetvalues1n_              MATSETVALUES1N
  #define matsetvaluesn1_              MATSETVALUESN1
  #define matsetvaluesblocked0_        MATSETVALUESBLOCKED0
  #define matsetvaluesblocked2_        MATSETVALUESBLOCKED2
  #define matsetvaluesblocked11_       MATSETVALUESBLOCKED11
  #define matsetvaluesblocked111_      MATSETVALUESBLOCKED111
  #define matsetvaluesblocked1n_       MATSETVALUESBLOCKED1N
  #define matsetvaluesblockedn1_       MATSETVALUESBLOCKEDN1
  #define matsetvaluesblockedlocal_    MATSETVALUESBLOCKEDLOCAL
  #define matsetvaluesblockedlocal0_   MATSETVALUESBLOCKEDLOCAL0
  #define matsetvaluesblockedlocal11_  MATSETVALUESBLOCKEDLOCAL11
  #define matsetvaluesblockedlocal111_ MATSETVALUESBLOCKEDLOCAL111
  #define matsetvaluesblockedlocal1n_  MATSETVALUESBLOCKEDLOCAL1N
  #define matsetvaluesblockedlocaln1_  MATSETVALUESBLOCKEDLOCALN1
  #define matsetvalueslocal_           MATSETVALUESLOCAL
  #define matsetvalueslocal0_          MATSETVALUESLOCAL0
  #define matsetvalueslocal11_         MATSETVALUESLOCAL11
  #define matsetvalueslocal11nn_       MATSETVALUESLOCAL11NN
  #define matsetvalueslocal111_        MATSETVALUESLOCAL111
  #define matsetvalueslocal1n_         MATSETVALUESLOCAL1N
  #define matsetvalueslocaln1_         MATSETVALUESLOCALN1
  #define matdestroymatrices_          MATDESTROYMATRICES
  #define matdestroysubmatrices_       MATDESTROYSUBMATRICES
  #define matgetrowij_                 MATGETROWIJ
  #define matrestorerowij_             MATRESTOREROWIJ
  #define matgetrow_                   MATGETROW
  #define matrestorerow_               MATRESTOREROW
  #define matseqaijgetarray_           MATSEQAIJGETARRAY
  #define matseqaijrestorearray_       MATSEQAIJRESTOREARRAY
  #define matdensegetarray_            MATDENSEGETARRAY
  #define matdensegetarrayread_        MATDENSEGETARRAYREAD
  #define matdenserestorearray_        MATDENSERESTOREARRAY
  #define matdenserestorearrayread_    MATDENSERESTOREARRAYREAD
  #define matcreatesubmatrices_        MATCREATESUBMATRICES
  #define matcreatesubmatricesmpi_     MATCREATESUBMATRICESMPI
  #define matnullspacesetfunction_     MATNULLSPACESETFUNCTION
  #define matfindnonzerorows_          MATFINDNONZEROROWS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matsetvalues_                matsetvalues
  #define matsetvaluesnnnn_            matsetvaluesnnnn
  #define matsetvalues0_               matsetvalues0
  #define matsetvaluesnn1_             matsetvaluesnn1
  #define matsetvalues11_              matsetvalues11
  #define matsetvaluesn1_              matsetvaluesn1
  #define matsetvalues1n_              matsetvalues1n
  #define matsetvalueslocal_           matsetvalueslocal
  #define matsetvalueslocal0_          matsetvalueslocal0
  #define matsetvalueslocal11_         matsetvalueslocal11
  #define matsetvalueslocal11nn_       matsetvalueslocal11nn
  #define matsetvalueslocal111_        matsetvalueslocal111
  #define matsetvalueslocal1n_         matsetvalueslocal1n
  #define matsetvalueslocaln1_         matsetvalueslocaln1
  #define matsetvaluesblocked_         matsetvaluesblocked
  #define matsetvaluesblocked0_        matsetvaluesblocked0
  #define matsetvaluesblocked2_        matsetvaluesblocked2
  #define matsetvaluesblocked11_       matsetvaluesblocked11
  #define matsetvaluesblocked111_      matsetvaluesblocked111
  #define matsetvaluesblocked1n_       matsetvaluesblocked1n
  #define matsetvaluesblockedn1_       matsetvaluesblockedn1
  #define matsetvaluesblockedlocal_    matsetvaluesblockedlocal
  #define matsetvaluesblockedlocal0_   matsetvaluesblockedlocal0
  #define matsetvaluesblockedlocal11_  matsetvaluesblockedlocal11
  #define matsetvaluesblockedlocal111_ matsetvaluesblockedlocal111
  #define matsetvaluesblockedlocal1n_  matsetvaluesblockedlocal1n
  #define matsetvaluesblockedlocaln1_  matsetvaluesblockedlocaln1
  #define matdestroymatrices_          matdestroymatrices
  #define matdestroysubmatrices_       matdestroysubmatrices
  #define matgetrowij_                 matgetrowij
  #define matrestorerowij_             matrestorerowij
  #define matgetrow_                   matgetrow
  #define matrestorerow_               matrestorerow
  #define matseqaijgetarray_           matseqaijgetarray
  #define matseqaijrestorearray_       matseqaijrestorearray
  #define matdensegetarray_            matdensegetarray
  #define matdensegetarrayread_        matdensegetarrayread
  #define matdenserestorearray_        matdenserestorearray
  #define matdenserestorearrayread_    matdenserestorearrayread
  #define matcreatesubmatrices_        matcreatesubmatrices
  #define matcreatesubmatricesmpi_     matcreatesubmatricesmpi
  #define matnullspacesetfunction_     matnullspacesetfunction
  #define matfindnonzerorows_          matfindnonzerorows
  #define matgetvalues_                matgetvalues
  #define matgetvalues0_               matgetvalues0
  #define matgetvaluesnn1_             matgetvaluesnn1
  #define matgetvaluesnnnn_            matgetvaluesnnnn
  #define matgetvalues11_              matgetvalues11
  #define matgetvalues11a_             matgetvalues11a
  #define matgetvalues1n_              matgetvalues1n
  #define matgetvaluesn1_              matgetvaluesn1
  #define matgetvalueslocal_           matgetvalueslocal
  #define matgetvalueslocal0_          matgetvalueslocal0
  #define matgetvalueslocalnn1_        matgetvalueslocalnn1
  #define matgetvalueslocalnnnn_       matgetvalueslocalnnnn
  #define matgetvalueslocal11_         matgetvalueslocal11
  #define matgetvalueslocal1n_         matgetvalueslocal1n
  #define matgetvalueslocaln1_         matgetvalueslocaln1
#endif

PETSC_EXTERN void matgetvalues_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  *ierr = MatGetValues(*mat, *m, idxm, *n, idxn, v);
}

PETSC_EXTERN void matgetvalues0_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalues_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvaluesnn1_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalues_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvaluesnnnn_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalues_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalues11_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalues_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalues11a_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalues_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalues1n_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalues_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvaluesn1_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalues_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalueslocal_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  *ierr = MatGetValuesLocal(*mat, *m, idxm, *n, idxn, v);
}

PETSC_EXTERN void matgetvalueslocal0_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalueslocal_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalueslocalnn1_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalueslocal_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalueslocalnnnn_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalueslocal_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalueslocal11_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalueslocal_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalueslocal11a_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalueslocal_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalueslocal1n_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalueslocal_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matgetvalueslocaln1_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], int *ierr)
{
  matgetvalueslocal_(mat, m, idxm, n, idxn, v, ierr);
}

PETSC_EXTERN void matsetvaluesblocked_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  *ierr = MatSetValuesBlocked(*mat, *m, idxm, *n, idxn, v, *addv);
}

PETSC_EXTERN void matsetvaluesblocked2_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], F90Array2d *y, InsertMode *addv, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(y, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  matsetvaluesblocked_(mat, m, idxm, n, idxn, fa, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblocked0_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblocked_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblocked11_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblocked_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblocked111_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblocked_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblocked1n_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblocked_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblockedn1_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblocked_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblockedlocal_(Mat *mat, PetscInt *nrow, PetscInt irow[], PetscInt *ncol, PetscInt icol[], PetscScalar y[], InsertMode *addv, int *ierr)
{
  *ierr = MatSetValuesBlockedLocal(*mat, *nrow, irow, *ncol, icol, y, *addv);
}

PETSC_EXTERN void matsetvaluesblockedlocal0_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblockedlocal_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblockedlocal11_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblockedlocal_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblockedlocal111_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblockedlocal_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblockedlocal1n_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblockedlocal_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesblockedlocaln1_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvaluesblockedlocal_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvalues_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  *ierr = MatSetValues(*mat, *m, idxm, *n, idxn, v, *addv);
}

PETSC_EXTERN void matsetvaluesnnnn_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvalues_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvalues0_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvalues_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesnn1_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvalues_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvalues11_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvalues_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvaluesn1_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvalues_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvalues1n_(Mat *mat, PetscInt *m, PetscInt idxm[], PetscInt *n, PetscInt idxn[], PetscScalar v[], InsertMode *addv, int *ierr)
{
  matsetvalues_(mat, m, idxm, n, idxn, v, addv, ierr);
}

PETSC_EXTERN void matsetvalueslocal_(Mat *mat, PetscInt *nrow, PetscInt irow[], PetscInt *ncol, PetscInt icol[], PetscScalar y[], InsertMode *addv, int *ierr)
{
  *ierr = MatSetValuesLocal(*mat, *nrow, irow, *ncol, icol, y, *addv);
}

PETSC_EXTERN void matsetvalueslocal0_(Mat *mat, PetscInt *nrow, PetscInt irow[], PetscInt *ncol, PetscInt icol[], PetscScalar y[], InsertMode *addv, int *ierr)
{
  matsetvalueslocal_(mat, nrow, irow, ncol, icol, y, addv, ierr);
}

PETSC_EXTERN void matsetvalueslocal11_(Mat *mat, PetscInt *nrow, PetscInt irow[], PetscInt *ncol, PetscInt icol[], PetscScalar y[], InsertMode *addv, int *ierr)
{
  matsetvalueslocal_(mat, nrow, irow, ncol, icol, y, addv, ierr);
}

PETSC_EXTERN void matsetvalueslocal11nn_(Mat *mat, PetscInt *nrow, PetscInt irow[], PetscInt *ncol, PetscInt icol[], PetscScalar y[], InsertMode *addv, int *ierr)
{
  matsetvalueslocal_(mat, nrow, irow, ncol, icol, y, addv, ierr);
}

PETSC_EXTERN void matsetvalueslocal111_(Mat *mat, PetscInt *nrow, PetscInt irow[], PetscInt *ncol, PetscInt icol[], PetscScalar y[], InsertMode *addv, int *ierr)
{
  matsetvalueslocal_(mat, nrow, irow, ncol, icol, y, addv, ierr);
}

PETSC_EXTERN void matsetvalueslocal1n_(Mat *mat, PetscInt *nrow, PetscInt irow[], PetscInt *ncol, PetscInt icol[], PetscScalar y[], InsertMode *addv, int *ierr)
{
  matsetvalueslocal_(mat, nrow, irow, ncol, icol, y, addv, ierr);
}

PETSC_EXTERN void matsetvalueslocaln1_(Mat *mat, PetscInt *nrow, PetscInt irow[], PetscInt *ncol, PetscInt icol[], PetscScalar y[], InsertMode *addv, int *ierr)
{
  matsetvalueslocal_(mat, nrow, irow, ncol, icol, y, addv, ierr);
}

static PetscErrorCode ournullfunction(MatNullSpace sp, Vec x, void *ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(MatNullSpace *, Vec *, void *, PetscErrorCode *))(((PetscObject)sp)->fortran_func_pointers[0]))(&sp, &x, ctx, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void matnullspacesetfunction_(MatNullSpace *sp, PetscErrorCode (*rem)(MatNullSpace, Vec, void *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*sp, 1);
  ((PetscObject)*sp)->fortran_func_pointers[0] = (PetscVoidFn *)rem;

  *ierr = MatNullSpaceSetFunction(*sp, ournullfunction, ctx);
}

PETSC_EXTERN void matgetrowij_(Mat *B, PetscInt *shift, PetscBool *sym, PetscBool *blockcompressed, PetscInt *n, PetscInt *ia, size_t *iia, PetscInt *ja, size_t *jja, PetscBool *done, PetscErrorCode *ierr)
{
  const PetscInt *IA, *JA;
  *ierr = MatGetRowIJ(*B, *shift, *sym, *blockcompressed, n, &IA, &JA, done);
  if (*ierr) return;
  *iia = PetscIntAddressToFortran(ia, (PetscInt *)IA);
  *jja = PetscIntAddressToFortran(ja, (PetscInt *)JA);
}

PETSC_EXTERN void matrestorerowij_(Mat *B, PetscInt *shift, PetscBool *sym, PetscBool *blockcompressed, PetscInt *n, PetscInt *ia, size_t *iia, PetscInt *ja, size_t *jja, PetscBool *done, PetscErrorCode *ierr)
{
  const PetscInt *IA = PetscIntAddressFromFortran(ia, *iia), *JA = PetscIntAddressFromFortran(ja, *jja);
  *ierr = MatRestoreRowIJ(*B, *shift, *sym, *blockcompressed, n, &IA, &JA, done);
}

/*
   This is a poor way of storing the column and value pointers
  generated by MatGetRow() to be returned with MatRestoreRow()
  but there is not natural,good place else to store them. Hence
  Fortran programmers can only have one outstanding MatGetRows()
  at a time.
*/
static int                matgetrowactive = 0;
static const PetscInt    *my_ocols        = NULL;
static const PetscScalar *my_ovals        = NULL;

PETSC_EXTERN void matgetrow_(Mat *mat, PetscInt *row, PetscInt *ncols, PetscInt *cols, PetscScalar *vals, PetscErrorCode *ierr)
{
  const PetscInt    **oocols = &my_ocols;
  const PetscScalar **oovals = &my_ovals;

  if (matgetrowactive) {
    *ierr = PetscError(PETSC_COMM_SELF, __LINE__, "MatGetRow_Fortran", __FILE__, PETSC_ERR_ARG_WRONGSTATE, PETSC_ERROR_INITIAL, "Cannot have two MatGetRow() active simultaneously\n\
               call MatRestoreRow() before calling MatGetRow() a second time");
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }

  CHKFORTRANNULLINTEGER(cols);
  if (!cols) oocols = NULL;
  CHKFORTRANNULLSCALAR(vals);
  if (!vals) oovals = NULL;

  *ierr = MatGetRow(*mat, *row, ncols, oocols, oovals);
  if (*ierr) return;

  if (oocols) {
    *ierr = PetscArraycpy(cols, my_ocols, *ncols);
    if (*ierr) return;
  }
  if (oovals) {
    *ierr = PetscArraycpy(vals, my_ovals, *ncols);
    if (*ierr) return;
  }
  matgetrowactive = 1;
}

PETSC_EXTERN void matrestorerow_(Mat *mat, PetscInt *row, PetscInt *ncols, PetscInt *cols, PetscScalar *vals, PetscErrorCode *ierr)
{
  const PetscInt    **oocols = &my_ocols;
  const PetscScalar **oovals = &my_ovals;

  if (!matgetrowactive) {
    *ierr = PetscError(PETSC_COMM_SELF, __LINE__, "MatRestoreRow_Fortran", __FILE__, PETSC_ERR_ARG_WRONGSTATE, PETSC_ERROR_INITIAL, "Must call MatGetRow() first");
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  CHKFORTRANNULLINTEGER(cols);
  if (!cols) oocols = NULL;
  CHKFORTRANNULLSCALAR(vals);
  if (!vals) oovals = NULL;

  *ierr           = MatRestoreRow(*mat, *row, ncols, oocols, oovals);
  matgetrowactive = 0;
}

PETSC_EXTERN void matseqaijgetarray_(Mat *mat, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  PetscScalar *mm;
  PetscInt     m, n;

  *ierr = MatSeqAIJGetArray(*mat, &mm);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, &m, &n);
  if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*mat, 1, fa, mm, m * n, ia);
  if (*ierr) return;
}

PETSC_EXTERN void matseqaijrestorearray_(Mat *mat, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  PetscScalar *lx;
  PetscInt     m, n;

  *ierr = MatGetSize(*mat, &m, &n);
  if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*mat, fa, *ia, m * n, &lx);
  if (*ierr) return;
  *ierr = MatSeqAIJRestoreArray(*mat, &lx);
  if (*ierr) return;
}

PETSC_EXTERN void matdensegetarray_(Mat *mat, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  PetscScalar *mm;
  PetscInt     m, n;

  *ierr = MatDenseGetArray(*mat, &mm);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, &m, &n);
  if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*mat, 1, fa, mm, m * n, ia);
  if (*ierr) return;
}

PETSC_EXTERN void matdenserestorearray_(Mat *mat, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  PetscScalar *lx;
  PetscInt     m, n;

  *ierr = MatGetSize(*mat, &m, &n);
  if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*mat, fa, *ia, m * n, &lx);
  if (*ierr) return;
  *ierr = MatDenseRestoreArray(*mat, &lx);
  if (*ierr) return;
}

PETSC_EXTERN void matdensegetarrayread_(Mat *mat, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  const PetscScalar *mm;
  PetscInt           m, n;

  *ierr = MatDenseGetArrayRead(*mat, &mm);
  if (*ierr) return;
  *ierr = MatGetSize(*mat, &m, &n);
  if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*mat, 1, fa, (PetscScalar *)mm, m * n, ia);
  if (*ierr) return;
}

PETSC_EXTERN void matdenserestorearrayread_(Mat *mat, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  const PetscScalar *lx;
  PetscInt           m, n;

  *ierr = MatGetSize(*mat, &m, &n);
  if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*mat, fa, *ia, m * n, (PetscScalar **)&lx);
  if (*ierr) return;
  *ierr = MatDenseRestoreArrayRead(*mat, &lx);
  if (*ierr) return;
}

/*
    MatCreateSubmatrices() is slightly different from C since the
    Fortran provides the array to hold the submatrix objects,while in C that
    array is allocated by the MatCreateSubmatrices()
*/
PETSC_EXTERN void matcreatesubmatrices_(Mat *mat, PetscInt *n, IS *isrow, IS *iscol, MatReuse *scall, Mat *smat, PetscErrorCode *ierr)
{
  Mat     *lsmat;
  PetscInt i;

  if (*scall == MAT_INITIAL_MATRIX) {
    *ierr = MatCreateSubMatrices(*mat, *n, isrow, iscol, *scall, &lsmat);
    for (i = 0; i <= *n; i++) { /* lsmat[*n] might be a dummy matrix for saving data structure */
      smat[i] = lsmat[i];
    }
    *ierr = PetscFree(lsmat);
  } else {
    *ierr = MatCreateSubMatrices(*mat, *n, isrow, iscol, *scall, &smat);
  }
}

/*
    MatCreateSubmatrices() is slightly different from C since the
    Fortran provides the array to hold the submatrix objects,while in C that
    array is allocated by the MatCreateSubmatrices()
*/
PETSC_EXTERN void matcreatesubmatricesmpi_(Mat *mat, PetscInt *n, IS *isrow, IS *iscol, MatReuse *scall, Mat *smat, PetscErrorCode *ierr)
{
  Mat     *lsmat;
  PetscInt i;

  if (*scall == MAT_INITIAL_MATRIX) {
    *ierr = MatCreateSubMatricesMPI(*mat, *n, isrow, iscol, *scall, &lsmat);
    for (i = 0; i <= *n; i++) { /* lsmat[*n] might be a dummy matrix for saving data structure */
      smat[i] = lsmat[i];
    }
    *ierr = PetscFree(lsmat);
  } else {
    *ierr = MatCreateSubMatricesMPI(*mat, *n, isrow, iscol, *scall, &smat);
  }
}

/*
    MatDestroyMatrices() is slightly different from C since the
    Fortran does not free the array of matrix objects, while in C that
    the array is freed
*/
PETSC_EXTERN void matdestroymatrices_(PetscInt *n, Mat *smat, PetscErrorCode *ierr)
{
  PetscInt i;

  for (i = 0; i < *n; i++) {
    PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(&smat[i]);
    *ierr = MatDestroy(&smat[i]);
    if (*ierr) return;
    PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(&smat[i]);
  }
}

/*
    MatDestroySubMatrices() is slightly different from C since the
    Fortran provides the array to hold the submatrix objects, while in C that
    array is allocated by the MatCreateSubmatrices()

    An extra matrix may be stored at the end of the array, hence the check see
    MatDestroySubMatrices_Dummy()
*/
PETSC_EXTERN void matdestroysubmatrices_(PetscInt *n, Mat *smat, PetscErrorCode *ierr)
{
  Mat     *lsmat;
  PetscInt i;

  if (*n == 0) return;
  *ierr = PetscMalloc1(*n + 1, &lsmat);
  if (*ierr) return;
  for (i = 0; i <= *n; i++) { lsmat[i] = smat[i]; }
  *ierr = MatDestroySubMatrices(*n, &lsmat);
  if (*ierr) return;
  for (i = 0; i <= *n; i++) { PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(&smat[i]); }
}
