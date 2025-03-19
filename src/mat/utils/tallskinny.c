/*
    Tall skinny routines:

    - operate on dense matrices
    - are intended for matrices that are m x n with n << m
    - any factors produced by these routines that are n x n are reproduced on each process so that the values are
      available for branching logic (such as selecting a subset of columns based on the small factors)
*/

#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <petscblaslapack.h>

/* Proposed interface that is useful here.

   It would be nice to have a general interface for building the individual factors of a factorization, but I don't have
   the time to do that right now.

   - Toby, March 2025
 */

/*
  MatQRFactorConstructFactors - Turn a QR factorization into the individual factors

  Collective

  Input Parameter:
+ QR      - A QR factorization of an $m \times n$ matrix $A$, computed by `MatQRFactor()` or `MatQRFactorNumeric()`
- reuse_Q - If `MAT_INPLACE_MATRIX`, the output `Q` will be in `QR` (`QR` will stop being a factored matrix);
            if `MAT_INITIAL_MATRIX` the output `Q` will be a newly constructed matrix;
            if `MAT_REUSE_MATRIX` the output `Q` is an existing $m \times n$ matrix that isn't `QR`

  Output Parameter:
+ Q     - an $m \times n$ matrix: the first `ncols` columns of `Q` are orthonormal
. R     - (optional) if not `NULL`, an $n \times n$ upper triangular matrix (only the first `ncols` rows of $R$ contain
          nonzeros)
. ncols - only the first `ncols` columns of `Q` are meaningful: the remainder are undefined and should not be used
- perm  - (optional) if not `NULL`, will reference the permutation $P$ such that $QR = AP$; if the $QR$ factorization
          did not use column pivoting, then $QR = A$ and `*perm` will be set to `NULL`

  Level: developer

  Notes:
  `ncols` indicates how many columns of `Q` are valid but may be greater than the numerical rank of $A$: the QR
  factorization may not have computed the rank of `R`.

  `perm` is a reference and should not be changed or destroyed.

  Developers Notes:
  Currently only implemented for serial dense matrices.

.seealso: [](ch_matrices), `Mat`, `MATDENSE`, `MatQRFactor()`, `MatGetFactor()`, `MatQRFactorSymbolic()`, `MatQRFactorNumeric()`
*/
PETSC_INTERN PetscErrorCode MatQRFactorConstructFactors(Mat QR, MatReuse reuse_Q, Mat *Q, Mat R, PetscInt *ncols, IS *perm)
{
  MPI_Comm      comm;
  PetscInt      m, n;
  MatFactorType factor_type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(QR, MAT_CLASSID, 1);
  PetscAssertPointer(Q, 3);
  if (R) PetscValidHeaderSpecific(R, MAT_CLASSID, 4);
  PetscAssertPointer(ncols, 5);
  if (perm) PetscAssertPointer(perm, 6);

  PetscCall(PetscObjectGetComm((PetscObject)QR, &comm));
  PetscCall(MatGetFactorType(QR, &factor_type));
  PetscCheck(factor_type == MAT_FACTOR_QR, comm, PETSC_ERR_ARG_WRONGSTATE, "QR is not a MAT_FACTOR_QR factored matrix");
  PetscCall(MatGetSize(QR, &m, &n));
  if (reuse_Q == MAT_INPLACE_MATRIX) {
    PetscCheck(*Q == QR, comm, PETSC_ERR_ARG_WRONG, "Q must point to QR if reuse_Q is MAT_INPLACE_MATRIX");
  } else if (reuse_Q == MAT_REUSE_MATRIX) {
    PetscLayout qr_row_layout, q_row_layout;
    PetscBool   same;
    PetscInt    Q_n;

    PetscAssertPointer(Q, 3);
    PetscValidHeaderSpecific(*Q, MAT_CLASSID, 3);
    PetscCall(MatGetLayouts(QR, &qr_row_layout, NULL));
    PetscCall(MatGetLayouts(*Q, &q_row_layout, NULL));
    PetscCall(PetscLayoutCompare(qr_row_layout, q_row_layout, &same));
    PetscCheck(same, comm, PETSC_ERR_ARG_SIZ, "row layout of Q does not match row layout of QR");
    PetscCall(MatGetSize(*Q, NULL, &Q_n));
    PetscCheck(Q_n == n, comm, PETSC_ERR_ARG_SIZ, "Q should have %" PetscInt_FMT "columns, has %" PetscInt_FMT " columns", n, Q_n);
  }
  if (R) {
    PetscLayout q_col_layout, r_row_layout;
    PetscInt    R_n;
    PetscBool   same;

    PetscCall(MatGetLayouts(QR, NULL, &q_col_layout));
    PetscCall(MatGetLayouts(R, &r_row_layout, NULL));
    PetscCall(PetscLayoutCompare(q_col_layout, r_row_layout, &same));
    PetscCheck(same, comm, PETSC_ERR_ARG_SIZ, "column layout of Q does not match row layout of R");
    PetscCall(MatGetSize(R, NULL, &R_n));
    PetscCheck(R_n == n, comm, PETSC_ERR_ARG_SIZ, "R shoud have %" PetscInt_FMT " columns, not %" PetscInt_FMT, n, R_n);
  }
  PetscUseMethod(QR, "MatQRFactorConstructFactors_C", (Mat, MatReuse, Mat *, Mat, PetscInt *, IS *), (QR, reuse_Q, Q, R, ncols, perm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
