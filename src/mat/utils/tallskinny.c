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

static PetscBool  tsqr_cite       = PETSC_FALSE;
static const char tsqr_citation[] = "@article{Demmel2012,"
                                    "  title = {Communication-optimal Parallel and Sequential QR and LU Factorizations},"
                                    "  volume = {34},"
                                    "  ISSN = {1095-7197},"
                                    "  url = {http://dx.doi.org/10.1137/080731992},"
                                    "  DOI = {10.1137/080731992},"
                                    "  number = {1},"
                                    "  journal = {SIAM Journal on Scientific Computing},"
                                    "  publisher = {Society for Industrial & Applied Mathematics (SIAM)},"
                                    "  author = {Demmel, James and Grigori, Laura and Hoemmen, Mark and Langou, Julien},"
                                    "  year = {2012},"
                                    "  month = jan,"
                                    "  pages = {A206-A239}"
                                    "}\n";

static PetscErrorCode MatDenseTSLQ_Internal(MPI_Comm comm, PetscInt n, PetscScalar *AQ, PetscScalar *L)
{
  PetscBLASInt bn, two_bn;
  PetscScalar *B_lo, *B_hi;
  PetscMPIInt  size;
  PetscInt     nlevels, lastlevel;
  PetscMPIInt *_lo, *_hi;
  PetscMPIInt  rank, tag;
  PetscScalar *tau, *work = NULL, *B;
  PetscBLASInt lwork, info;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscCommGetNewTag(comm, &tag));
  // compute nlevels = ceil(log2(size))
  nlevels = 0;
  for (PetscInt sizeshift = size - 1; sizeshift > 0; sizeshift >>= 1) nlevels++;
  if (nlevels == 0) {
    PetscCall(PetscArraycpy(AQ, L, n * n));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscBLASIntCast(n, &bn));
  PetscCall(PetscMalloc4(2 * n * n, &B, n, &tau, nlevels, &_lo, nlevels, &_hi));
  two_bn    = 2 * bn;
  B_lo      = B;
  B_hi      = &B[n * n];
  _lo[0]    = 0;
  _hi[0]    = size;
  lastlevel = nlevels - 1;
  for (PetscInt i = 1; i < nlevels; i++) {
    PetscMPIInt mid = _lo[i - 1] + (_hi[i - 1] - _lo[i - 1]) / 2;

    _lo[i] = (rank < mid) ? _lo[i - 1] : mid;
    _hi[i] = (rank < mid) ? mid : _hi[i - 1];

    if (_hi[i] == _lo[i] + 1) {
      lastlevel = i - 1;
      break;
    }
  }
  for (PetscInt i = lastlevel; i >= 0; i--) {
    PetscMPIInt  lo      = _lo[i];
    PetscMPIInt  hi      = _hi[i];
    PetscMPIInt  mid     = lo + (hi - lo) / 2;
    PetscMPIInt  partner = (rank < mid) ? mid + (rank - lo) : lo + (rank - mid);
    MPI_Request  reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    PetscScalar *B_this, *B_that;
    PetscScalar  one = 1.0, zero = 0.0;

    PetscAssert((lo < mid) && (mid < hi) && (rank >= lo) && (rank < hi) && (((mid - lo) == (hi - mid)) || ((mid - lo) + 1 == (hi - mid))), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invariants of splits not respected in TSQR");
    B_this = rank < mid ? B_lo : B_hi;
    B_that = rank < mid ? B_hi : B_lo;

    PetscCall(PetscArraycpy(B_this, L, bn * bn));
    if (rank >= mid && partner >= mid) PetscCallMPI(MPI_Irecv(B_that, bn * bn, MPIU_SCALAR, mid - 1, tag, comm, &reqs[0]));
    else {
      PetscCallMPI(MPI_Isend(B_this, bn * bn, MPIU_SCALAR, partner, tag, comm, &reqs[0]));
      PetscCallMPI(MPI_Irecv(B_that, bn * bn, MPIU_SCALAR, partner, tag, comm, &reqs[1]));
    }
    if (((hi - mid) > (mid - lo)) && rank == mid - 1) PetscCallMPI(MPI_Isend(B_this, bn * bn, MPIU_SCALAR, hi - 1, tag, comm, &reqs[2]));
    PetscCallMPI(MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE));

    // LQ factorization of B = [L_lo L_hi]
    if (work == NULL) {
      PetscScalar work_dummy;

      lwork = -1;
      PetscCallBLAS("LAPACKgelqf", LAPACKgelqf_(&bn, &two_bn, B, &bn, tau, &work_dummy, &lwork, &info));
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK gelqf %d", (int)info);
      lwork = (PetscBLASInt)PetscRealPart(work_dummy);
      PetscCall(PetscMalloc1(lwork, &work));
    }

    PetscCallBLAS("LAPACKgelqf", LAPACKgelqf_(&bn, &two_bn, B, &bn, tau, work, &lwork, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK geqlf %d", (int)info);

    PetscCall(PetscArraycpy(L, B, bn * bn));

    // zero out upper triangle of L
    for (PetscInt j = 0; j < bn; j++)
      for (PetscInt i = 0; i < j; i++) L[i + j * bn] = 0.0;

    // form new Q
    PetscCallBLAS("LAPACKorglq", LAPACKorglq_(&bn, &two_bn, &bn, B, &bn, tau, work, &lwork, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK orglq %d", (int)info);

    // compute Q_this * AQ
    if (i < lastlevel) {
      PetscCallBLAS("BLASgemm", BLASgemm_("N", "N", &bn, &bn, &bn, &one, B_this, &bn, AQ, &bn, &zero, B_that, &bn));
      PetscCall(PetscArraycpy(AQ, B_that, bn * bn));
    } else {
      // AQ was I in the base case
      PetscCall(PetscArraycpy(AQ, B_this, bn * bn));
    }
  }
  PetscCall(PetscFree(work));
  PetscCall(PetscFree4(B, tau, _lo, _hi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseSkinnyPrepareQ(Mat A, MatReuse reuse_Q, Mat *Q, const char U_name[])
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscAssertPointer(Q, 3);
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  if (reuse_Q == MAT_INPLACE_MATRIX) {
    PetscCheck(*Q == A, comm, PETSC_ERR_ARG_WRONG, "%s should be A if using MAT_INPLACE_MATRIX", U_name);
  } else if (reuse_Q == MAT_REUSE_MATRIX) {
    PetscBool same;
    PetscInt  M, N, Q_N;
    Mat       A_local, Q_local;

    PetscCall(MatGetSize(A, &M, &N));
    PetscCall(MatDenseGetLocalMatrix(A, &A_local));
    PetscValidHeaderSpecific(*Q, MAT_CLASSID, 2);
    PetscCall(PetscLayoutCompare(A->rmap, (*Q)->rmap, &same));
    PetscCheck(same, comm, PETSC_ERR_ARG_SIZ, "%s does not have the same row layout as A", U_name);
    PetscCall(MatDenseGetLocalMatrix(*Q, &Q_local));
    PetscCall(MatGetSize(*Q, NULL, &Q_N));
    PetscCheck(Q_N == N, comm, PETSC_ERR_ARG_SIZ, "%s should have %" PetscInt_FMT " rows, not %" PetscInt_FMT, U_name, N, Q_N);
    PetscCall(MatCopy(A_local, Q_local, UNKNOWN_NONZERO_PATTERN));
  } else PetscCall(MatDuplicate(A, MAT_COPY_VALUES, Q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseSkinnyPrepareR(Mat A, Mat *R, const char R_name[], PetscBool create_if_missing)
{
  PetscInt M, N;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatGetSize(A, &M, &N));
  if (*R != NULL) {
    MPI_Comm    R_comm;
    PetscInt    R_M, R_N;
    PetscMPIInt R_size;

    PetscValidHeaderSpecific(*R, MAT_CLASSID, 2);
    PetscCall(PetscObjectGetComm((PetscObject)*R, &R_comm));
    PetscCallMPI(MPI_Comm_size(R_comm, &R_size));
    PetscCheck(R_size == 1, comm, PETSC_ERR_ARG_WRONG, "%s must be PETSC_COMM_SELF on each process", R_name);
    PetscCall(MatGetSize(*R, &R_M, &R_N));
    PetscCheck(R_M == N && R_N == N, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "%s must be %" PetscInt_FMT " x %" PetscInt_FMT ", not %" PetscInt_FMT " x %" PetscInt_FMT, R_name, N, N, R_M, R_N);
  } else if (create_if_missing) {
    MatType mat_type;
    Mat     A_local;

    PetscCall(MatDenseGetLocalMatrix(A, &A_local));
    PetscCall(MatGetType(A_local, &mat_type));
    PetscCall(MatCreate(PETSC_COMM_SELF, R));
    PetscCall(MatSetSizes(*R, N, N, N, N));
    PetscCall(MatSetType(*R, mat_type));
    PetscCall(MatSetUp(*R));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatDenseSkinnyQR - Compute the QR factorization of a tall skinny dense matrix

  Collective

  Input Parameters:
+ A       - an $m \times n$ dense matrix `A`
- reuse_Q - if `MAT_INPLACE_MATRIX`, `Q` will overwrite `A`;
            if `MAT_REUSE_MATRIX`, `Q` should be an existing matrix;
            if `MAT_INITIAL_MATRIX`, `Q` will be a new matrix

  Output Parameters:
+ Q     - an $m \times n$ matrix, the first `ncols` columns of `Q` are orthonormal
. R     - (optional) if not `NULL`, a dense upper triangular $n \times n$ matrix such that $A = QR$;
          only the first `ncols` rows of `R` contain nonzeros;
          `R` is on the `PETSC_COMM_SELF` communicator and is duplicated on every process.
- ncols - only the first `ncols` columns of `Q` are meaningful: the remainder are undefined and should not be used

  Level: intermediate

  Notes:
  `MatDenseSkinnyQR()` is not a rank-revealing decomposition: `ncols` indicates how many columns of `Q` are orthonormal
  but may be greater than the numerical rank of `A`.

  In serial `MatDenseSkinnyQR()` uses a LAPACK Householder-QR algorithm; in parallel, `MatDenseSkinnyQR()` uses the TSQR
  algorithm (Demmel, Grigori, Hoemmen & Langou, 2012, "Communication-optimal Parallel and Sequential QR and LU
  Factorizations").

  `MatDenseSkinnyQR()` is intended for matrices without too many columns.  If you want to use a QR factorization to solve a
  least squares problem, use `MatQRFactor()`.

.seealso: [](ch_matrices), `Mat`, `MATDENSE`, `MatQRFactor()`, `MatDenseSkinnyQB()`, `MatDenseSkinnySVD()`
@*/
PetscErrorCode MatDenseSkinnyQR(Mat A, MatReuse reuse_Q, Mat *Q, Mat R, PetscInt *ncols)
{
  PetscMPIInt size;
  PetscInt    M, N, K, ncols_local;
  MPI_Comm    comm;
  Mat         QR_local, Q_local;
  Mat         R_in = R;
  IS          perm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  // makes sure Q and R exist and A is copied into Q
  PetscCall(MatDenseSkinnyPrepareQ(A, reuse_Q, Q, "Q"));
  PetscCall(MatDenseGetLocalMatrix(*Q, &QR_local));
  PetscCall(MatQRFactor(QR_local, NULL, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    PetscCall(MatQRFactorConstructFactors(QR_local, MAT_INPLACE_MATRIX, &QR_local, R, ncols, &perm));
    PetscCheck(perm == NULL, PETSC_COMM_SELF, PETSC_ERR_SUP, "MatDenseSkinnyQR() was given a pivoted QR factorization that it cannot use");
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatGetSize(A, &M, &N));
  K = PetscMin(M, N);
  if (ncols) *ncols = K;

  PetscCall(MatDenseSkinnyPrepareR(A, &R, "R", PETSC_TRUE));
  PetscCall(MatQRFactorConstructFactors(QR_local, MAT_INITIAL_MATRIX, &Q_local, R, &ncols_local, &perm));
  PetscCheck(perm == NULL, PETSC_COMM_SELF, PETSC_ERR_SUP, "MatDenseSkinnyQR() was given a pivoted QR factorization that it cannot use");

  // combine the local QR factorizations into a global QR factorization using the TSQR algorithm (turned on its side as TSLQ)
  {
    const PetscScalar *_R;
    PetscScalar       *_AQ, *L;
    PetscInt           ldR;
    Mat                AQ;
    Mat                Q_sub;

    PetscCall(PetscCitationsRegister(tsqr_citation, &tsqr_cite));

    PetscCall(PetscMalloc1(N * N, &_AQ));
    PetscCall(PetscCalloc1(N * N, &L));

    // copy R^T into a buffer L
    PetscCall(MatDenseGetLDA(R, &ldR));
    PetscCall(MatDenseGetArrayRead(R, &_R));
    for (PetscInt j = 0; j < ncols_local; j++)
      for (PetscInt i = 0; i < N; i++) L[i + j * N] = _R[j + i * ldR];
    PetscCall(MatDenseRestoreArrayRead(R, &_R));

    PetscCall(MatDenseTSLQ_Internal(comm, N, _AQ, L));

    // write L^T into R
    if (R_in) {
      PetscScalar *_R_in;

      PetscCall(MatDenseGetArray(R_in, &_R_in));
      for (PetscInt j = 0; j < N; j++)
        for (PetscInt i = 0; i < K; i++) _R_in[i + j * ldR] = L[j + i * N];
      PetscCall(MatDenseRestoreArray(R, &_R_in));
    }

    // Multipy QR_local = Q_local * AQ^T
    PetscCall(MatCreateDenseFromVecType(PETSC_COMM_SELF, VECSEQ, N, ncols_local, N, ncols_local, N, _AQ, &AQ));
    Q_sub = Q_local;
    if (ncols_local < N) PetscCall(MatDenseGetSubMatrix(Q_local, PETSC_DECIDE, PETSC_DECIDE, 0, ncols_local, &Q_sub));
    PetscCall(MatSetUnfactored(QR_local));
    PetscCall(MatMatTransposeMult(Q_sub, AQ, MAT_REUSE_MATRIX, PETSC_DECIDE, &QR_local));
    if (ncols_local < N) PetscCall(MatDenseRestoreSubMatrix(Q_local, &Q_sub));

    PetscCall(PetscFree(L));
    PetscCall(PetscFree(_AQ));
    PetscCall(MatDestroy(&AQ));
  }
  PetscCall(MatDestroy(&Q_local));
  if (R != R_in) PetscCall(MatDestroy(&R));
  PetscFunctionReturn(PETSC_SUCCESS);
}
