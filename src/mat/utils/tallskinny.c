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
    PetscValidHeaderSpecific(*Q, MAT_CLASSID, 3);
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

static PetscBool  svqb_cite       = PETSC_FALSE;
static const char svqb_citation[] = "@article{Stathopoulos2002,\n"
                                    "  title = {A Block Orthogonalization Procedure with Constant Synchronization Requirements},\n"
                                    "  volume = {23},\n"
                                    "  ISSN = {1095-7197},\n"
                                    "  url = {http://dx.doi.org/10.1137/S1064827500370883},\n"
                                    "  DOI = {10.1137/s1064827500370883},\n"
                                    "  number = {6},\n"
                                    "  journal = {SIAM Journal on Scientific Computing},\n"
                                    "  publisher = {Society for Industrial & Applied Mathematics (SIAM)},\n"
                                    "  author = {Stathopoulos,  Andreas and Wu,  Kesheng},\n"
                                    "  year = {2002},\n"
                                    "  month = jan,\n"
                                    "  pages = {2165-2182}\n"
                                    "}\n";

// overwrites A
static PetscErrorCode MatDenseSVD_LAPACK_Arrays(PetscScalar _A[], PetscInt m, PetscInt n, PetscInt ldA, PetscScalar _U[], PetscInt ldU, PetscReal _S[], PetscScalar _VH[], PetscInt ldV, PetscBLASInt *lwork, PetscScalar **work, PetscReal **rwork)
{
  PetscInt      k;
  PetscLogEvent event;

  PetscFunctionBegin;
  k = PetscMin(m, n);
  if (k == 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventRegister("LAPACKgesvd", MAT_CLASSID, &event));
  {
    PetscScalar  dummy_u = 0.0, dummy_v = 0.0;
    PetscBLASInt bm, bn, bk, bldA, bldU, bldV;
    PetscBLASInt lierr;
    const char  *form_u = (_U == NULL) ? "N" : (_U == _A) ? "O" : "S";
    const char  *form_v = (_VH == NULL) ? "N" : (_VH == _A) ? "O" : "A";

    if (_U == NULL) {
      _U  = &dummy_u;
      ldU = 1;
    }
    if (_VH == NULL) {
      _VH = &dummy_v;
      ldV = 1;
    }

    PetscCall(PetscBLASIntCast(ldA, &bldA));
    PetscCall(PetscBLASIntCast(ldU, &bldU));
    PetscCall(PetscBLASIntCast(ldV, &bldV));
    PetscCall(PetscBLASIntCast(m, &bm));
    PetscCall(PetscBLASIntCast(n, &bn));
    PetscCall(PetscBLASIntCast(k, &bk));
    if (PetscDefined(USE_COMPLEX)) {
      if (*rwork == NULL) { PetscCall(PetscMalloc1(5 * PetscMax(m, n), rwork)); }
    }

    // compute work size
    if (*work == NULL) {
      PetscScalar work_size;

      *lwork = -1;
#if !defined(PETSC_USE_COMPLEX)
      PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(form_u, form_v, &bm, &bn, _A, &bldA, _S, _U, &bldU, _VH, &bldV, &work_size, lwork, &lierr));
#else
      PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(form_u, form_v, &bm, &bn, _A, &bldA, _S, _U, &bldU, _VH, &bldV, &work_size, lwork, *rwork, &lierr));
#endif
      PetscCheck(lierr == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK gesvd %d", (int)lierr);

      *lwork = (PetscBLASInt)PetscRealPart(work_size);
      PetscCall(PetscMalloc1(*lwork, work));
    }

    PetscCall(PetscLogEventBegin(event, NULL, NULL, NULL, NULL));
#if !defined(PETSC_USE_COMPLEX)
    PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(form_u, form_v, &bm, &bn, _A, &bldA, _S, _U, &bldU, _VH, &bldV, *work, lwork, &lierr));
#else
    PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(form_u, form_v, &bm, &bn, _A, &bldA, _S, _U, &bldU, _VH, &bldV, *work, lwork, *rwork, &lierr));
#endif
    PetscCall(PetscLogEventEnd(event, NULL, NULL, NULL, NULL));
    PetscCheck(lierr == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK gesvd %d", (int)lierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// do a local MatHermitianTransposeMatMult, then all reduce the values on the given communicator
static PetscErrorCode MatHermitianTransposeMatMultAllReduce(MPI_Comm comm, Mat A, Mat B, MatReuse reuse, PetscReal fill, Mat *C)
{
  PetscInt     m, n, ldC;
  PetscScalar *_C;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    MPI_Comm    A_comm;
    PetscMPIInt A_size;

    PetscCall(PetscObjectGetComm((PetscObject)A, &A_comm));
    PetscCallMPI(MPI_Comm_size(A_comm, &A_size));
    PetscCheck(A_size == 1, comm, PETSC_ERR_ARG_NOTSAMECOMM, "A and B must be local (PETSC_COMM_SELF) matrices");
  }
  if (PetscDefined(USE_COMPLEX)) {
    Mat conjA;

    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &conjA));
    PetscCall(MatConjugate(conjA));
    PetscCall(MatTransposeMatMult(conjA, B, reuse, fill, C));
    PetscCall(MatDestroy(&conjA));
  } else {
    PetscCall(MatTransposeMatMult(A, B, reuse, fill, C));
  }
  PetscCall(MatGetSize(*C, &m, &n));
  PetscCall(MatDenseGetLDA(*C, &ldC));
  PetscCheck(ldC == m, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Require lda to be %" PetscInt_FMT ", not %" PetscInt_FMT, m, ldC);
  // even if C is PETSC_OFFLOAD_GPU, we are going to do our analysis/orthogonalization work on the host, so we use the host
  // arrays to reduce here
  PetscCall(MatDenseGetArray(*C, &_C));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, _C, m * n, MPIU_SCALAR, MPI_SUM, comm));
  PetscCall(MatDenseRestoreArray(*C, &_C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSwap(Mat *A, Mat *B)
{
  Mat swap = *A;

  PetscFunctionBegin;
  *A = *B;
  *B = swap;
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct _n_MatDenseSVQBWork {
  PetscBLASInt n;
  PetscReal   *D;
  PetscScalar *BBH;
  PetscScalar *work;
  PetscReal   *rwork;
  PetscBLASInt lwork;
} MatDenseSVQBWork;

static PetscErrorCode MatDenseSVQBWorkInitialize(PetscInt N, MatDenseSVQBWork *w)
{
  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(N, &w->n));
  PetscCall(PetscMalloc3(N * N, &w->BBH, N, &w->D, 5 * N, &w->rwork));
  w->work  = NULL;
  w->lwork = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseSVQBWorkReset(MatDenseSVQBWork *w)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(w->work));
  PetscCall(PetscFree3(w->BBH, w->D, w->rwork));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// YHY = Y' * Y
// A   = Y * B
// NB: we are trusting LAPACK algorthms to be bitwise identical on all processes
static PetscErrorCode MatDenseComputeSVQBUpdate(Mat YHY, Mat B, Mat Y_update, PetscBool *stop, PetscInt *r, PetscInt iter, PetscViewer viewer, MatDenseSVQBWork *work)
{
  PetscInt     ldB, ldYup, _r, ldYHY;
  PetscBLASInt bldW, bn, br;
  PetscScalar *_B, *_BBH, *_Y_update;
  PetscReal   *_D;
  PetscScalar *_YHY;
  PetscReal    _D_max, _S_max, _D_min, stop_tol = 0.5;
  PetscInt     n;
  PetscScalar  one = 1.0, zero = 0.0;

  PetscFunctionBegin;
  *stop = PETSC_TRUE;
  PetscCall(MatGetSize(YHY, &n, NULL));
  *r = n;
  if (n == 0) PetscFunctionReturn(PETSC_SUCCESS);
  // iniialize Y_update == I
  PetscCall(MatDenseGetLDA(Y_update, &ldYup));
  PetscAssert(ldYup == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Y_update should have leading dimension %" PetscInt_FMT ", has %" PetscInt_FMT, n, ldYup);
  PetscCall(MatDenseGetArrayWrite(Y_update, &_Y_update));
  PetscCall(PetscArrayzero(_Y_update, n * n));
  for (PetscInt i = 0; i < n; i++) _Y_update[i * (n + 1)] = 1.0;

  _D = work->D;
  bn = work->n;
  PetscCall(MatDenseGetLDA(B, &ldB));
  PetscCall(PetscBLASIntCast(ldB, &bldW));
  PetscCall(MatDenseGetArray(B, &_B));

  PetscCall(MatDenseGetLDA(YHY, &ldYHY));
  PetscAssert(ldYHY == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A should have leading dimension %" PetscInt_FMT ", has %" PetscInt_FMT, n, ldYHY);
  PetscCall(MatDenseGetArray(YHY, &_YHY));
  _BBH = work->BBH;

  // compute the Hadamard product BB' \otimes Y'Y
  PetscCallBLAS("BLASgemm", BLASgemm_("N", "C", &bn, &bn, &bn, &one, _B, &bldW, _B, &bldW, &zero, _BBH, &bn));
  for (PetscInt i = 0; i < n * n; i++) _BBH[i] *= _YHY[i];

  // use _D[k] to compute \sum_{i,j \geq} (BB' \otimes Y'Y)_{ij} = || Y_{:,k:} B_{k:,:} ||_F^2.
  for (PetscInt k = n - 1; k >= 0; k--) {
    _D[k] = 0.0;
    if (k < n - 1) _D[k] = _D[k + 1];
    _D[k] += PetscAbsScalar(_BBH[k * (n + 1)]);
    for (PetscInt j = k + 1; j < n; j++) _D[k] += PetscRealPart(_BBH[k + j * n] + _BBH[j + k * n]);
    _D[k] = PetscAbsReal(_D[k]);
  }

  /* _D[0] = || Y B ||_F^2 = || A ||_F^2.  If _D[k] \leq \epsilon^2 || A ||_F^2, then

       || A - Y_{:,:k} B_{:k,:} ||_F

       = || Y_{k:,:} B_{k:,:} ||_F

       \leq \epsilon || A ||_F,

     which means we are making a negligible perturbation to A by dropping Y_{:,k:} B_{k:,:}
   */
  {
    PetscReal frob2_est = _D[0];
    PetscReal err_tol   = frob2_est * PETSC_MACHINE_EPSILON * PETSC_MACHINE_EPSILON;

    for (PetscInt i = n - 1; i >= 0; i--) {
      if (_D[i] <= err_tol) *r = i;
      else break;
    }
  }
  _r = *r;

  if (viewer) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "MatDenseSVQB, iter %" PetscInt_FMT ": || YW ||_F^2 components =\n", iter + 1));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      for (PetscInt i = 0; i < n; i++) {
        if (i == _r) PetscCall(PetscViewerASCIIPrintf(viewer, "--------\n"));
        PetscCall(PetscViewerASCIIPrintf(viewer, "%g\n", (double)_D[i]));
      }
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }

  if (_r == 0) {
    PetscCall(MatDenseRestoreArray(YHY, &_YHY));
    PetscCall(MatDenseRestoreArray(B, &_B));
    PetscCall(MatDenseRestoreArrayWrite(Y_update, &_Y_update));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // get the diagonal of Y'Y
  for (PetscInt i = 0; i < _r; i++) _D[i] = PetscAbsScalar(_YHY[i * (n + 1)]);

  // get the smallest nonzero value of D
  _D_max = 0.0;
  for (PetscInt i = 0; i < _r; i++) _D_max = PetscMax(_D_max, _D[i]);

  if (_D_max == 0.0) {
    PetscCall(MatDenseRestoreArray(YHY, &_YHY));
    PetscCall(MatDenseRestoreArray(B, &_B));
    PetscCall(MatDenseRestoreArrayWrite(Y_update, &_Y_update));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  _D_min = _D_max;
  for (PetscInt i = 0; i < _r; i++) {
    if (_D[i] > 0.0) _D_min = PetscMin(_D_min, _D[i]);
  }

  // set _D_min as the floor
  for (PetscInt i = 0; i < _r; i++) _D[i] = PetscMax(_D_min, _D[i]);

  // D[i] = sqrt(D[i])
  for (PetscInt i = 0; i < _r; i++) _D[i] = PetscSqrtReal(_D[i]);

  // scale B on the left by D
  for (PetscInt j = 0; j < n; j++) {
    for (PetscInt i = 0; i < _r; i++) _B[i + j * ldB] *= _D[i];
  }

  // D[i] = 1.0 / D[i];
  for (PetscInt i = 0; i < _r; i++) _D[i] = 1.0 / _D[i];

  for (PetscInt j = 0; j < _r; j++) {
    for (PetscInt i = 0; i < _r; i++) _YHY[i + j * n] *= _D[i] * _D[j];
  }

  // scale Y_update on the right by D
  for (PetscInt i = 0; i < _r; i++) _Y_update[i * (n + 1)] = _D[i];

  // svd of Y'Y, storing U in _YHY and S in _D;
  PetscCall(MatDenseSVD_LAPACK_Arrays(_YHY, _r, _r, n, _YHY, n, _D, NULL, 1, &work->lwork, &work->work, &work->rwork));

  if (viewer) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "MatDenseSVQB, iter %" PetscInt_FMT ": singular values =\n", iter + 1));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      for (PetscInt i = 0; i < _r; i++) { PetscCall(PetscViewerASCIIPrintf(viewer, "%g\n", (double)_D[i])); }
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }

  _S_max = _D[0];
  /* compute D[i] = sqrt(max(S_max*eps, S[i]))

     The theory from the SVQB paper is that if all singular values are \geq S_max * c for a constant c > 0,
     then stopping the iteration after this update would yield

       || Y'Y - I ||_2 \leq c_0(m,n) \min( \epsilon / c, 1 )

   */
  for (PetscInt i = 0; i < _r; i++) {
    PetscReal s = _D[i];

    if (s <= _S_max * stop_tol) *stop = PETSC_FALSE;
    _D[i] = PetscMax(s, _S_max * PETSC_MACHINE_EPSILON);
    _D[i] = PetscSqrtReal(_D[i]);
  }

  // multiply Y_update = Y_update * U (Y_update is currently diagonal, this is a scaling of U)
  for (PetscInt i = 0; i < _r; i++) {
    PetscScalar d = _Y_update[i * (n + 1)];
    for (PetscInt j = 0; j < _r; j++) _Y_update[i + j * n] = _YHY[i + j * n] * d;
  }

  // multiply B = U' * B
  PetscCall(PetscBLASIntCast(_r, &br));
  PetscCallBLAS("BLASgemm", BLASgemm_("C", "N", &br, &bn, &br, &one, _YHY, &bn, _B, &bldW, &zero, _BBH, &bn));
  if (_r == n) PetscCall(PetscArraycpy(_B, _BBH, n * n));
  else {
    for (PetscInt j = 0; j < n; j++) PetscCall(PetscArraycpy(&_B[j * ldB], &_BBH[j * n], _r));
  }

  // multiply B = D * B
  for (PetscInt j = 0; j < n; j++) {
    for (PetscInt i = 0; i < _r; i++) _B[i + j * ldB] *= _D[i];
  }

  // invert D
  for (PetscInt i = 0; i < _r; i++) _D[i] = 1.0 / _D[i];

  // muliply Y_update = Y_update * D
  for (PetscInt j = 0; j < _r; j++) {
    for (PetscInt i = 0; i < n; i++) _Y_update[i + j * n] *= _D[j];
  }

  PetscCall(MatDenseRestoreArray(YHY, &_YHY));
  PetscCall(MatDenseRestoreArray(B, &_B));
  PetscCall(MatDenseRestoreArrayWrite(Y_update, &_Y_update));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseSVQBMonitor(Mat X, Mat Y, Mat YtY, Mat W, PetscInt r, PetscInt iter, PetscViewer viewer)
{
  Mat       X_local;
  Mat       Yr;
  Mat       Wr;
  Mat       YWminusX;
  Mat       YtYminusI;
  PetscReal ortho_err, recon_err, recon_err_local;
  PetscInt  M, N;
  MPI_Comm  comm;

  PetscFunctionBegin;
  if (viewer == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCall(MatGetSize(X, &M, &N));
  PetscCall(PetscViewerASCIIPrintf(viewer, "MatDenseSVQB, iter %" PetscInt_FMT ": %" PetscInt_FMT " x %" PetscInt_FMT " matrix, estimated rank %" PetscInt_FMT "\n", iter, M, N, r));
  PetscCall(MatDenseGetSubMatrix(Y, PETSC_DECIDE, PETSC_DECIDE, 0, r, &Yr));
  PetscCall(MatHermitianTransposeMatMultAllReduce(comm, Yr, Yr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &YtYminusI));
  PetscCall(MatShift(YtYminusI, -1.0));
  PetscCall(MatNorm(YtYminusI, NORM_FROBENIUS, &ortho_err));
  PetscCall(PetscViewerASCIIPrintf(viewer, "MatDenseSVQB, iter %" PetscInt_FMT ": orthogonality error || Y'Y - I ||_F = %e\n", iter, (double)ortho_err));
  PetscCall(MatDestroy(&YtYminusI));
  PetscCall(MatDenseGetSubMatrix(W, 0, r, PETSC_DECIDE, PETSC_DECIDE, &Wr));
  PetscCall(MatMatMult(Yr, Wr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &YWminusX));
  PetscCall(MatDenseGetLocalMatrix(X, &X_local));
  PetscCall(MatAXPY(YWminusX, -1.0, X_local, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(YWminusX, NORM_FROBENIUS, &recon_err_local));
  recon_err_local = recon_err_local * recon_err_local;
  PetscCallMPI(MPIU_Allreduce(&recon_err_local, &recon_err, 1, MPIU_REAL, MPI_SUM, comm));
  recon_err = PetscSqrtReal(recon_err);
  PetscCall(PetscViewerASCIIPrintf(viewer, "MatDenseSVQB, iter %" PetscInt_FMT ": reconstruction error || YW - X ||_F = %e\n", iter, (double)recon_err));
  PetscCall(MatDestroy(&YWminusX));
  PetscCall(MatDenseRestoreSubMatrix(W, &Wr));
  PetscCall(MatDenseRestoreSubMatrix(Y, &Yr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatDenseSkinnyQB - Compute a QB factorization of a tall skinny dense matrix

  Collective

  Input Parameters:
+ A       - an $m \times n$ dense matrix `A`
- reuse_Q - if `MAT_INPLACE_MATRIX`, `Q` will overwrite `A`;
            if `MAT_REUSE_MATRIX`, `Q` should be an existing matrix;
            if `MAT_INITIAL_MATRIX`, `Q` will be a new matrix

  Output Parameters:
+ Q     - an $m \times n$ matrix, the first `ncols` columns of `Q` are orthonormal
. B     - (optional) if not `NULL`, a dense $n \times n$ matrix such that $A = QB$;
          only the first `ncols` rows of `B` contain nonzeros;
          `B` is on the `PETSC_COMM_SELF` communicator and is duplicated on every process.
- ncols - only the first `ncols` columns of `Q` are meaningful: the remainder are undefined and should not be used

  Options Database Key:
. -mat_dense_svqb_monitor [viewertype]:... - monitor the iterative SVQB algorithm used to compute `MatDenseSkinnyQB()` (for debugging, incurs additional computation and communication)

  Level: intermediate

  Notes:
  `MatDenseSkinnyQB()` is not a rank-revealing decomposition: `ncols` indicates how many columns of `Q` are orthonormal
  but may be greater than the numerical rank of `A`.

  `MatDenseSkinnyQB()` uses the SVQB algorithm (Stathopoulos & Wu, 2002, "A Block Orthogonalization Procedure with
  Constant Synchronization Requirements").

.seealso: [](ch_matrices), `Mat`, `MATDENSE`, `MatQRFactor()`, `MatDenseSkinnyQR()`
@*/
PetscErrorCode MatDenseSkinnyQB(Mat A, MatReuse reuse_Q, Mat *Q, Mat B, PetscInt *ncols)
{
  PetscInt          M, N, K, r, i;
  PetscInt          max_it = 10;
  Mat               B_in   = B;
  Mat               QHQ, Y, Y_orig, Y_copy, Y_update;
  Mat               A_copy = NULL;
  MPI_Comm          comm;
  PetscOptions      options;
  const char       *prefix;
  PetscViewer       viewer;
  PetscViewerFormat format;
  MatDenseSVQBWork  work;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(svqb_citation, &svqb_cite));
  PetscCall(MatDenseSkinnyPrepareQ(A, reuse_Q, Q, "Q"));
  PetscCall(MatDenseSkinnyPrepareR(A, &B, "B", PETSC_TRUE));
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatGetSize(A, &M, &N));
  K = PetscMin(M, N);
  PetscCall(MatDenseGetLocalMatrix(*Q, &Y));
  Y_orig = Y;
  PetscCall(MatDuplicate(Y, MAT_SHARE_NONZERO_PATTERN, &Y_copy));

  PetscCall(PetscObjectGetOptions((PetscObject)A, &options));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)A, &prefix));
  PetscCall(PetscOptionsCreateViewer(comm, options, prefix, "-mat_dense_svqb_monitor", &viewer, &format, NULL));
  if (viewer) {
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_copy));
  }
  PetscCall(MatHermitianTransposeMatMultAllReduce(comm, Y, Y, MAT_INITIAL_MATRIX, PETSC_DECIDE, &QHQ));
  PetscCall(MatDuplicate(QHQ, MAT_SHARE_NONZERO_PATTERN, &Y_update));
  PetscCall(MatZeroEntries(B));
  PetscCall(MatShift(B, 1.0));

  /* Invariant: A = Y * B
     Initially: Y = A, B = I

     we will transform Y into an orthonormal basis using the parallel SVQB agorithm
     the iteration max of 10 should almost never be needed: it should use at most 3
     iterations for most matrices */

  r = N; // r is the number of leading y_i w_i^T pairs that contain all of the Frobenius norm mass of A
  PetscCall(MatDenseSVQBWorkInitialize(N, &work));
  for (i = 0; i < max_it; i++) {
    PetscBool stop;

    PetscCall(MatDenseSVQBMonitor(A_copy, Y, QHQ, B, r, i, viewer));
    PetscCall(MatDenseComputeSVQBUpdate(QHQ, B, Y_update, &stop, &r, i, viewer, &work));
    if (r > 0) {
      PetscCall(MatMatMult(Y, Y_update, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y_copy));
      PetscCall(MatSwap(&Y, &Y_copy));
    }
    if (stop || i + 1 == max_it) {
      if (viewer) { // monitor final matrices
        PetscCall(MatHermitianTransposeMatMultAllReduce(comm, Y, Y, MAT_REUSE_MATRIX, PETSC_DECIDE, &QHQ));
        PetscCall(MatDenseSVQBMonitor(A_copy, Y, QHQ, B, r, i + 1, viewer));
      }
      break;
    }
    PetscCall(MatHermitianTransposeMatMultAllReduce(comm, Y, Y, MAT_REUSE_MATRIX, PETSC_DECIDE, &QHQ));
  }
  PetscCall(MatDenseSVQBWorkReset(&work));
  if (Y != Y_orig) PetscCall(MatCopy(Y, Y_orig, SAME_NONZERO_PATTERN));
  r      = PetscMin(r, K);
  *ncols = r;

  if (B_in && r < N) {
    PetscScalar *_B;
    PetscInt     ldB;

    // zero trailing rows of B_in
    PetscCall(MatDenseGetLDA(B_in, &ldB));
    PetscCall(MatDenseGetArray(B_in, &_B));
    for (PetscInt j = 0; j < N; j++) PetscCall(PetscArrayzero(&_B[r + j * ldB], (N - r)));
    PetscCall(MatDenseRestoreArray(B_in, &_B));
  }

  PetscCall(MatDestroy(&A_copy));
  if (viewer) PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  if (Y != Y_orig) PetscCall(MatDestroy(&Y));
  if (Y_copy != Y_orig) PetscCall(MatDestroy(&Y_copy));
  PetscCall(MatDestroy(&Y_update));
  PetscCall(MatDestroy(&QHQ));
  if (B_in == NULL) PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef enum {
  MATDENSESKINNYSVD_SVQB,
  MATDENSESKINNYSVD_TSQR,
} MatDenseSkinnySVDAlgorithm;

const char *const MatDenseSkinnySVDAlgorithms[] = {"SVQB", "TSQR", "MatDenseSkinnySVDAlgorithm", "MATDENSESKINNYSVD_", NULL};

/*@
  MatDenseSkinnySVD - Compute the SVD of a tall skinny dense matrix

  Collective

  Input Parameters:
+ A       - An $m \times n$ dense matrix
- reuse_U - if `MAT_INPLACE_MATRIX`, `U` will overwrite `A`;
            if `MAT_REUSE_MATRIX`, `U` should be an existing matrix;
            if `MAT_INITIAL_MATRIX`, `U` will be a new matrix

  Output Parameters:
+ U     - an $m \times n$ matrix, the first `ncols` columns of `U` are left singular vectors of `A`
. S     - an $n$ vector holding singular values of `A`;
          `S` is on the `PETSC_COMM_SELF` communicator and is duplicated on every process.
. VH    - (optional) if not `NULL`, a dense $n \times n$ matrix, whose rows are the (conjugate transpose) right singular
          vectors of `A`;
          `VH` is on the `PETSC_COMM_SELF` communicator and is duplicated on every process.
- ncols - only the first `ncols` columns of `U` are meaningful: the remainder are undefined and should not be used

  Options Database Key:
. -mat_dense_svd_algorithm <svqb,tsqr> - the algorithm used to compute a QB decomposition of `A` as a first step to computing the SVD

  Level: intermediate

  Notes:
  `ncols` indicates how many columns of `U` are orthonormal but may be greater than the numerical rank of `A`\:
  inspect `S` to determine the numerical rank of `A`.

  In serial `MatDenseSkinnySVD()` uses LAPACK's SVD algorithm; in parallel, a QB decomposition `A = QB` is first computed using
  either the SVQB (`-mat_dense_svd_algorithm tsqr`) or TSQR algorithm (`-mat_dense_svd_algorithm svqb`) and the SVD
  of `A` is then computed from the SVD of `B`.

.seealso: [](ch_matrices), `Mat`, `MATDENSE`, `MatDenseSkinnyQR()`, `MatDenseSkinnyQB()`
@*/
PetscErrorCode MatDenseSkinnySVD(Mat A, MatReuse reuse_U, Mat *U, Vec S, Mat VH, PetscInt *ncols)
{
  PetscInt                   M, N;
  MPI_Comm                   comm;
  PetscMPIInt                size;
  MatDenseSkinnySVDAlgorithm algorithm = MATDENSESKINNYSVD_SVQB;
  PetscOptions               options;
  const char                *prefix;
  Mat                        B, B_sub;
  PetscInt                   dummy_ncols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(S, VEC_CLASSID, 4);
  if (VH) PetscValidHeaderSpecific(VH, MAT_CLASSID, 5);
  PetscAssertPointer(ncols, 6);
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatGetSize(A, &M, &N));
  {
    MPI_Comm    S_comm;
    PetscMPIInt S_size;
    PetscInt    S_N;

    PetscCall(PetscObjectGetComm((PetscObject)S, &S_comm));
    PetscCallMPI(MPI_Comm_size(S_comm, &S_size));
    PetscCheck(S_size == 1, comm, PETSC_ERR_ARG_WRONG, "S should be on PETSC_COMM_SELF, not a parallel vector");
    PetscCall(VecGetSize(S, &S_N));
    PetscCheck(S_N == N, S_comm, PETSC_ERR_ARG_SIZ, "S should have length %" PetscInt_FMT ", not %" PetscInt_FMT, N, S_N);
  }
  PetscCall(MatDenseSkinnyPrepareQ(A, reuse_U, U, "U"));
  PetscCall(MatDenseSkinnyPrepareR(A, &VH, "VH", PETSC_FALSE));
  if (size == 1) {
    PetscScalar *_AU;
    PetscScalar *_VH = NULL;
    PetscScalar *_S;
    PetscReal   *_Sreal;
    PetscInt     ldAU, ldV = 1;
    PetscBLASInt lwork = -1;
    PetscScalar *work  = NULL;
    PetscReal   *rwork = NULL;

    PetscCall(VecZeroEntries(S));
    PetscCall(MatDenseGetLDA(*U, &ldAU));
    PetscCall(MatDenseGetArray(*U, &_AU));
    if (VH) {
      PetscCall(MatDenseGetLDA(VH, &ldV));
      PetscCall(MatDenseGetArrayWrite(VH, &_VH));
    }

    PetscCall(VecGetArray(S, &_S));
#if !PetscDefined(USE_COMPLEX)
    _Sreal = _S;
#else
    PetscCall(PetscMalloc1(N, &_Sreal));
#endif

    PetscCall(MatDenseSVD_LAPACK_Arrays(_AU, M, N, ldAU, _AU, ldAU, _Sreal, _VH, ldV, &lwork, &work, &rwork));
    PetscCall(PetscFree(work));
    PetscCall(PetscFree(rwork));

#if PetscDefined(USE_COMPLEX)
    for (PetscInt i = 0; i < N; i++) _S[i] = _Sreal[i];
    PetscCall(PetscFree(_Sreal));
#endif

    if (VH) PetscCall(MatDenseRestoreArrayWrite(VH, &_VH));
    PetscCall(MatDenseRestoreArray(*U, &_AU));

    *ncols = PetscMin(M, N);
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, N, N, NULL, &B));
  PetscCall(MatSetUp(B));
  PetscCall(PetscObjectGetOptions((PetscObject)A, &options));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)A, &prefix));
  PetscCall(PetscOptionsGetEnum(options, prefix, "-mat_dense_svd_algorithm", MatDenseSkinnySVDAlgorithms, (PetscEnum *)&algorithm, NULL));
  if (algorithm == MATDENSESKINNYSVD_TSQR) PetscCall(MatDenseSkinnyQR(*U, MAT_INPLACE_MATRIX, U, B, ncols));
  else PetscCall(MatDenseSkinnyQB(*U, MAT_INPLACE_MATRIX, U, B, ncols));
  B_sub = B;
  if (*ncols < N) PetscCall(MatDenseGetSubMatrix(B, 0, *ncols, 0, N, &B_sub));
  PetscCall(MatDenseSkinnySVD(B_sub, MAT_INPLACE_MATRIX, &B_sub, S, VH, &dummy_ncols));
  {
    Mat U_local, U_sub, B_subsub, UB;

    PetscCall(MatDenseGetLocalMatrix(*U, &U_local));
    U_sub    = U_local;
    B_subsub = B_sub;

    if (*ncols < N) {
      PetscCall(MatDenseGetSubMatrix(U_local, PETSC_DECIDE, PETSC_DECIDE, 0, *ncols, &U_sub));
      PetscCall(MatDenseGetSubMatrix(B_sub, 0, *ncols, 0, *ncols, &B_subsub));
    }

    PetscCall(MatMatMult(U_sub, B_subsub, MAT_INITIAL_MATRIX, PETSC_DECIDE, &UB));
    PetscCall(MatCopy(UB, U_sub, UNKNOWN_NONZERO_PATTERN));
    PetscCall(MatDestroy(&UB));

    if (*ncols < N) {
      PetscCall(MatDenseRestoreSubMatrix(B_sub, &B_subsub));
      PetscCall(MatDenseRestoreSubMatrix(U_local, &U_sub));
    }
  }
  if (*ncols < N) PetscCall(MatDenseRestoreSubMatrix(B, &B_sub));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
