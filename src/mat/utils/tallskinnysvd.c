/*
    Routines to project vectors out of null spaces.
*/

#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <petscblaslapack.h>

static PetscBool  cite_registered = PETSC_FALSE;
static const char svb_citation[]  = "@article{Stathopoulos2002,\n"
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
                                    "  pages = {2165–2182}\n"
                                    "}\n";

static PetscErrorCode MatDenseSVD_ProcessArguments(Mat A, MatReuse reuse, PetscInt k, PetscInt K, Mat *U, Vec *S, Mat *VH)
{
  PetscInt    M, N, m, n;
  MPI_Comm    comm;
  VecType     vec_type;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCheck(reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX, comm, PETSC_ERR_ARG_OUTOFRANGE, "reuse must be MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX");
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetVecType(A, &vec_type));
  PetscAssertPointer(S, 6);
  {
    MatReuse reuse_s = reuse;
    if (reuse_s == MAT_REUSE_MATRIX) {
      PetscInt s_N;

      PetscValidHeaderSpecific(*S, VEC_CLASSID, 6);
      PetscCheckSameComm(A, 1, *S, 6);

      PetscCall(VecGetSize(*S, &s_N));
      if (s_N != K) {
        PetscCall(VecDestroy(S));
        reuse_s = MAT_INITIAL_MATRIX;
      }
    }
    if (reuse_s == MAT_INITIAL_MATRIX) {
      PetscCall(VecCreate(comm, S));
      PetscCall(VecSetSizes(*S, k, K));
      PetscCall(VecSetType(*S, vec_type));
      PetscCall(VecSetUp(*S));
    }
  }
  if (U) {
    MatReuse reuse_u = reuse;

    if (reuse_u == MAT_REUSE_MATRIX) {
      PetscLayout a_rlayout, u_rlayout, u_clayout, s_layout;
      PetscBool   rcongruent, ccongruent;

      PetscValidHeaderSpecific(*U, MAT_CLASSID, 5);
      PetscCheckSameComm(A, 1, *U, 5);
      PetscCall(MatGetLayouts(A, &a_rlayout, NULL));
      PetscCall(MatGetLayouts(*U, &u_rlayout, &u_clayout));
      PetscCall(VecGetLayout(*S, &s_layout));
      PetscCall(PetscLayoutCompare(a_rlayout, u_rlayout, &rcongruent));
      PetscCall(PetscLayoutCompare(u_clayout, s_layout, &ccongruent));
      if (!(rcongruent && ccongruent)) {
        PetscCall(MatDestroy(U));
        reuse_u = MAT_INITIAL_MATRIX;
      }
    }
    if (reuse_u == MAT_INITIAL_MATRIX) { PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, k, M, K, -1, NULL, U)); }
  }
  if (VH) {
    MatReuse reuse_v = reuse;

    if (reuse_v == MAT_REUSE_MATRIX) {
      PetscLayout a_clayout, v_rlayout, v_clayout, s_layout;
      PetscBool   rcongruent, ccongruent;

      PetscValidHeaderSpecific(*VH, MAT_CLASSID, 5);
      PetscCheckSameComm(A, 1, *VH, 5);

      PetscCall(MatGetLayouts(A, NULL, &a_clayout));
      PetscCall(MatGetLayouts(*VH, &v_rlayout, &v_clayout));
      PetscCall(VecGetLayout(*S, &s_layout));
      PetscCall(PetscLayoutCompare(a_clayout, v_clayout, &ccongruent));
      PetscCall(PetscLayoutCompare(v_rlayout, s_layout, &rcongruent));
      if (!(rcongruent && ccongruent)) {
        PetscCall(MatDestroy(VH));
        reuse_v = MAT_INITIAL_MATRIX;
      }
    }
    if (reuse_v == MAT_INITIAL_MATRIX) { PetscCall(MatCreateDenseFromVecType(comm, vec_type, k, n, K, N, -1, NULL, VH)); }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseSVD_LAPACK(Mat A, MatReuse reuse, Mat *U, Vec *S, Mat *VH)
{
  MPI_Comm      comm;
  PetscInt      m, n, k;
  PetscLogEvent event;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCheck(size == 1, comm, PETSC_ERR_ARG_WRONG, "Cannot use LAPACK gesvd on distributed matrix (size %" PetscInt_FMT ")", size);
  }
  PetscCall(MatGetSize(A, &m, &n));
  k = PetscMin(m, n);
  PetscCall(MatDenseSVD_ProcessArguments(A, reuse, k, k, U, S, VH));
  if (k == 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventRegister("LAPACKgesvd", MAT_CLASSID, &event));
  {
    PetscScalar  dummy_u = 0.0, dummy_v = 0.0;
    PetscScalar *_A, *_U = &dummy_u, *_VH = &dummy_v, *sing, *uwork;
    PetscInt     ldA, ldU = 1, ldV = 1;
    PetscBLASInt bm, bn, bk, bldA, bldU, bldV, lwork;
    PetscScalar  work_size = 0.0;
    PetscReal   *rwork2    = NULL;
    PetscReal   *sing_real = NULL;
    PetscBLASInt lierr;
    PetscInt     ulw;
    const char  *form_u = (U != NULL) ? "S" : "N";
    const char  *form_v = (VH != NULL) ? "S" : "N";

    ulw = PetscMax(PetscMax(1, 5 * k), 3 * k + PetscMax(m, n));
    PetscCall(MatDenseGetLDA(A, &ldA));
    if (U) PetscCall(MatDenseGetLDA(*U, &ldU));
    if (VH) PetscCall(MatDenseGetLDA(*VH, &ldV));
    PetscCall(PetscBLASIntCast(ulw, &lwork));
    PetscCall(PetscBLASIntCast(ldA, &bldA));
    PetscCall(PetscBLASIntCast(ldU, &bldU));
    PetscCall(PetscBLASIntCast(ldV, &bldV));
    PetscCall(PetscBLASIntCast(m, &bm));
    PetscCall(PetscBLASIntCast(n, &bn));
    PetscCall(PetscBLASIntCast(k, &bk));
    PetscCall(MatDenseGetArray(A, &_A));
    if (U) PetscCall(MatDenseGetArray(*U, &_U));
    if (VH) PetscCall(MatDenseGetArray(*VH, &_VH));
    PetscCall(VecGetArrayWrite(*S, &sing));
    if (PetscDefined(USE_COMPLEX)) { PetscCall(PetscMalloc2(5 * PetscMax(m, n), &rwork2, PetscMin(m, n), &sing_real)); }

    // compute work size
    lwork = -1;
#if !defined(PETSC_USE_COMPLEX)
    PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(form_u, form_v, &bm, &bn, _A, &bldA, sing, _U, &bldU, _VH, &bldV, &work_size, &lwork, &lierr));
#else
    PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(form_u, form_v, &bm, &bn, _A, &bldA, sing_real, _U, &bldU, _VH, &bldV, &work_size, &lwork, rwork2, &lierr));
#endif
    PetscCheck(lierr == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK gesvd %d", (int)lierr);

    lwork = (PetscBLASInt)PetscRealPart(work_size);
    PetscCall(PetscMalloc1(lwork, &uwork));
    PetscCall(PetscLogEventBegin(event, NULL, NULL, NULL, NULL));
#if !defined(PETSC_USE_COMPLEX)
    PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(form_u, form_v, &bm, &bn, _A, &bldA, sing, _U, &bldU, _VH, &bldV, uwork, &lwork, &lierr));
#else
    PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_(form_u, form_v, &bm, &bn, _A, &bldA, sing_real, _U, &bldU, _VH, &bldV, uwork, &lwork, rwork2, &lierr));
#endif
    PetscCall(PetscLogEventEnd(event, NULL, NULL, NULL, NULL));
    PetscCheck(lierr == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK gesvd %d", (int)lierr);
    PetscCall(PetscFree(uwork));
    if (PetscDefined(USE_COMPLEX)) {
      for (PetscInt i = 0; i < PetscMin(m, n); i++) sing[i] = sing_real[i];
      PetscCall(PetscFree2(rwork2, sing_real));
    }
    PetscCall(VecRestoreArrayWrite(*S, &sing));
    if (VH) PetscCall(MatDenseRestoreArray(*VH, &_VH));
    if (U) PetscCall(MatDenseRestoreArray(*U, &_U));
    PetscCall(MatDenseRestoreArray(A, &_A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHermitianTransposeMatMult(Mat A, Mat B, MatReuse reuse, PetscReal fill, Mat *C)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    Mat conjA;

    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &conjA));
    PetscCall(MatConjugate(conjA));
    PetscCall(MatTransposeMatMult(conjA, B, reuse, fill, C));
    PetscCall(MatDestroy(&conjA));
  } else {
    PetscCall(MatTransposeMatMult(A, B, reuse, fill, C));
  }
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

// A = Y' * Y
// X = Y * W
static PetscErrorCode MatDenseComputeSVBUpdate(Mat A, Mat W, Mat Yupdate, PetscBool *stop, PetscInt *r, PetscInt iter, PetscViewer viewer)
{
  MPI_Comm    comm;
  PetscMPIInt size;
  PetscReal   D_max, stop_tol = 0.5;
  PetscInt    n;
  Vec         D, Dinv;
  Mat         U;

  PetscFunctionBegin;
  *stop = PETSC_TRUE;
  PetscCall(MatGetSize(A, &n, NULL));
  if (n == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscMPIInt rank;
    Mat         A_local, W_local, Yupdate_local;
    PetscInt    m, n;
    PetscInt    stop_r[2];
    PetscViewer subviewer = NULL;

    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(MatDenseGetLocalMatrix(A, &A_local));
    PetscCall(MatGetLocalSize(A_local, &m, &n));
    PetscCheck((rank > 0 && m == 0) || (rank == 0 && m == n), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "A matrix must be square with all entries on rank 0");
    PetscCall(MatDenseGetLocalMatrix(W, &W_local));
    PetscCall(MatGetLocalSize(W_local, &m, &n));
    PetscCheck((rank > 0 && m == 0) || (rank == 0 && m == n), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "W matrix must be square with all entries on rank 0");
    PetscCall(MatDenseGetLocalMatrix(Yupdate, &Yupdate_local));
    PetscCall(MatGetLocalSize(Yupdate_local, &m, &n));
    PetscCheck((rank > 0 && m == 0) || (rank == 0 && m == n), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Yupdate must be square with all entries on rank 0");
    if (viewer) PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &subviewer));
    if (rank == 0) {
      PetscCall(MatDenseComputeSVBUpdate(A_local, W_local, Yupdate_local, stop, r, iter, subviewer));
      stop_r[0] = (*stop) == PETSC_TRUE ? 1 : 0;
      stop_r[1] = *r;
    }
    if (viewer) PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &subviewer));
    PetscCallMPI(MPI_Bcast(stop_r, 2, MPIU_INT, 0, comm));
    *stop = stop_r[0] == 1 ? PETSC_TRUE : PETSC_FALSE;
    *r    = stop_r[1];
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatCreateVecs(A, &D, NULL));
  PetscCall(MatGetDiagonal(A, D));
  PetscCall(VecSqrtAbs(D));
  PetscCall(VecMax(D, NULL, &D_max));
  PetscCall(VecShift(D, D_max * PETSC_SQRT_MACHINE_EPSILON));
  PetscCall(VecDuplicate(D, &Dinv));
  PetscCall(VecCopy(D, Dinv));
  PetscCall(VecReciprocal(Dinv));
  PetscCall(MatDiagonalScale(A, Dinv, Dinv));
  PetscCall(MatDiagonalScale(Yupdate, NULL, Dinv));
  PetscCall(MatDiagonalScale(W, D, NULL));
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &U));
  PetscCall(MatDenseSVD_LAPACK(A, MAT_REUSE_MATRIX, &U, &D, NULL));
  if (viewer) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "MatTallSkinnySVD, iter %" PetscInt_FMT ": scaled singular values\n", iter));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT));
      PetscCall(VecView(D, viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  {
    PetscScalar *sing;
    PetscReal    s_max;

    PetscCall(VecGetArray(D, &sing));
    s_max = PetscRealPart(sing[0]);
    for (PetscInt i = 0; i < n; i++) {
      PetscReal s = PetscRealPart(sing[i]);

      if (s > 0 && s <= s_max * stop_tol) *stop = PETSC_FALSE;
      if (s == 0.0) *r = PetscMin(*r, i);
      sing[i] = PetscMax(s, s_max * PETSC_MACHINE_EPSILON);
    }
    PetscCall(VecRestoreArray(D, &sing));
  }
  PetscCall(VecSqrtAbs(D));
  PetscCall(VecCopy(D, Dinv));
  PetscCall(VecReciprocal(Dinv));
  PetscCall(MatMatMult(Yupdate, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A));
  PetscCall(MatCopy(A, Yupdate, SAME_NONZERO_PATTERN));
  PetscCall(MatHermitianTransposeMatMult(U, W, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A));
  PetscCall(MatCopy(A, W, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(Yupdate, NULL, Dinv));
  PetscCall(MatDiagonalScale(W, D, NULL));
  PetscCall(MatDestroy(&U));
  PetscCall(VecDestroy(&Dinv));
  PetscCall(VecDestroy(&D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseTallSkinnySVDMonitor(Mat X, Mat Y, Mat YtY, Mat W, PetscInt r, PetscInt iter, PetscViewer viewer)
{
  Mat       Yr;
  Mat       Wr;
  Mat       YWminusX;
  Mat       YtYminusI;
  PetscReal ortho_err, recon_err;
  PetscInt  M, N;

  PetscFunctionBegin;
  if (viewer == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatGetSize(X, &M, &N));
  PetscCall(PetscViewerASCIIPrintf(viewer, "MatTallSkinnySVD, iter %" PetscInt_FMT ": %" PetscInt_FMT " x %" PetscInt_FMT " matrix, rank upper bound %" PetscInt_FMT "\n", iter, M, N, r));
  PetscCall(MatDenseGetSubMatrix(Y, PETSC_DECIDE, PETSC_DECIDE, 0, r, &Yr));
  PetscCall(MatHermitianTransposeMatMult(Yr, Yr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &YtYminusI));
  PetscCall(MatShift(YtYminusI, -1.0));
  PetscCall(MatNorm(YtYminusI, NORM_FROBENIUS, &ortho_err));
  PetscCall(PetscViewerASCIIPrintf(viewer, "MatTallSkinnySVD, iter %" PetscInt_FMT ": orthogonality error || Y'Y - I ||_F = %e\n", iter, (double)ortho_err));
  PetscCall(MatDestroy(&YtYminusI));
  PetscCall(MatDenseGetSubMatrix(W, 0, r, PETSC_DECIDE, PETSC_DECIDE, &Wr));
  PetscCall(MatMatMult(Yr, Wr, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &YWminusX));
  PetscCall(MatAXPY(YWminusX, -1.0, X, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(YWminusX, NORM_FROBENIUS, &recon_err));
  PetscCall(PetscViewerASCIIPrintf(viewer, "MatTallSkinnySVD, iter %" PetscInt_FMT ": reconstruction error || YW - X ||_F = %e\n", iter, (double)recon_err));
  PetscCall(MatDestroy(&YWminusX));
  PetscCall(MatDenseRestoreSubMatrix(W, &Wr));
  PetscCall(MatDenseRestoreSubMatrix(Y, &Yr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatDenseTallSkinnySVD - Compute the SVD of a tall skinny dense matrix.

  Collective

  Input Parameter:
+ X     - A dense matrix (`MATDENSE`): in parallel, all columns should be assigned to the first process
- reuse - MAT_INITIAL_MATIRX if output arguments should be created, MAT_REUSE_MATRIX if output arguments have already been created

  Output Parameters:
+ U - Optional.  If not NULL, an orthonormal dense matrix whose columns are the left singular vectors of `X`: it will have the same row layout as `X`.  `U` may have fewer columns if `X` has zero singular values.
. S - A vector containing nonzero singular values of `X`, in descending order.  This is a parallel distributed vector.
- VH - Optional.  If not NULL, an orthonormal matrix whose rows are the right singular vectors of `X`: its column layout will be the same as the column layout of `X`.  `V` may have fewer rows if `X` has zero singular values.

  Options Database Key:
. -mat_dense_tall_skinny_svd_monitor [viewertype]:... - monitor the iterative SVQB algorithm that computes an orthogonal basis for the range of `X`.

  Level: developer

  Note:
  This tall skinny SVD is computed by iterating the SVQB algorithm of Stathopoulos and Wu to compute an orthonormal basis of the range of `X`, using as few parallel synchronizations as possible.

.seealso: [](ch_matrices), `Mat`, `MATDENSE`
@*/
PetscErrorCode MatDenseTallSkinnySVD(Mat X, MatReuse reuse, Mat *U, Vec *S, Mat *VH)
{
  PetscInt          M, N, m, n, r, R, i;
  PetscMPIInt       size, rank;
  MPI_Comm          comm;
  Mat               Y, W, Yupdate, Ycopy, A;
  Mat               WU = NULL, WVH = NULL;
  Vec               WS;
  VecType           vec_type;
  PetscInt          max_it = 10;
  PetscBool         stop;
  PetscViewer       viewer = NULL;
  PetscOptions      options;
  PetscViewerFormat format;
  const char       *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    Mat Xcopy;

    PetscCall(MatDuplicate(X, MAT_COPY_VALUES, &Xcopy));
    PetscCall(MatDenseSVD_LAPACK(Xcopy, reuse, U, S, VH));
    PetscCall(MatDestroy(&Xcopy));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatGetSize(X, &M, &N));
  PetscCall(MatGetLocalSize(X, &m, &n));
  R = PetscMin(M, N);
  {
    PetscLayout     column_layout;
    const PetscInt *ranges;

    PetscCall(MatGetLayouts(X, NULL, &column_layout));
    PetscCall(PetscLayoutGetRanges(column_layout, &ranges));
    PetscCheck(ranges[1] == N, comm, PETSC_ERR_ARG_SIZ, "MatDenseTallSkinnySVD() requires all columns of X be assigned to the first process");
  }
  PetscCall(PetscCitationsRegister(svb_citation, &cite_registered));
  /* Invariant: X = Y * W
     Initially: Y = X, W = I

     we will transform Y into an orthonormal basis using the parallel SVB agorithm
     the iteration max of 10 should almost never be needed: it should use at most 3
     iterations for most matrices */

  PetscCall(MatDuplicate(X, MAT_COPY_VALUES, &Y));
  PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Ycopy));
  PetscCall(MatGetVecType(X, &vec_type));
  PetscCall(MatCreateDenseFromVecType(comm, vec_type, n, n, N, N, -1, NULL, &W));
  PetscCall(MatShift(W, 1.0));
  PetscCall(MatDuplicate(W, MAT_DO_NOT_COPY_VALUES, &A));
  PetscCall(MatDuplicate(W, MAT_DO_NOT_COPY_VALUES, &Yupdate));
  PetscCall(PetscObjectGetOptions((PetscObject)X, &options));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)X, &prefix));
  PetscCall(PetscOptionsGetViewer(comm, options, prefix, "-mat_dense_tall_skinny_svd_monitor", &viewer, &format, NULL));
  if (viewer) PetscCall(PetscViewerPushFormat(viewer, format));
  for (i = 0; i < max_it; i++) {
    PetscCall(MatHermitianTransposeMatMult(Y, Y, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A));
    PetscCall(MatDenseTallSkinnySVDMonitor(X, Y, A, W, R, i, viewer));
    PetscCall(MatZeroEntries(Yupdate));
    PetscCall(MatShift(Yupdate, 1.0));
    R = PetscMin(M, N);
    PetscCall(MatDenseComputeSVBUpdate(A, W, Yupdate, &stop, &R, i, viewer));
    PetscCall(MatMatMult(Y, Yupdate, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Ycopy));
    PetscCall(MatSwap(&Y, &Ycopy));
    if (stop) {
      i++;
      break;
    }
  }
  if (viewer) {
    PetscCall(MatHermitianTransposeMatMult(Y, Y, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A));
    PetscCall(MatDenseTallSkinnySVDMonitor(X, Y, A, W, R, i, viewer));
  }
  r = rank == 0 ? R : 0;
  PetscCall(MatDenseSVD_ProcessArguments(X, reuse, r, R, U, S, VH));
  if (U) PetscCall(MatDuplicate(W, MAT_DO_NOT_COPY_VALUES, &WU));
  if (VH) PetscCall(MatDuplicate(W, MAT_DO_NOT_COPY_VALUES, &WVH));
  PetscCall(MatCreateVecs(Y, &WS, NULL));
  if (rank == 0) {
    Mat Wlocal;
    Mat WUlocal = NULL, WVHlocal = NULL;
    Vec WSlocal;

    PetscCall(MatDenseGetLocalMatrix(W, &Wlocal));
    if (WU) PetscCall(MatDenseGetLocalMatrix(WU, &WUlocal));
    if (WVH) PetscCall(MatDenseGetLocalMatrix(WVH, &WVHlocal));
    PetscCall(VecCreateLocalVector(WS, &WSlocal));
    PetscCall(VecGetLocalVector(WS, WSlocal));
    PetscCall(MatDenseSVD_LAPACK(Wlocal, MAT_REUSE_MATRIX, &WUlocal, &WSlocal, WVH ? &WVHlocal : NULL));
    PetscCall(VecRestoreLocalVector(WS, WSlocal));
    PetscCall(VecDestroy(&WSlocal));
  }
  if (U) {
    Mat Ycopy_r;

    PetscCall(MatMatMult(Y, WU, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Ycopy));
    PetscCall(MatDenseGetSubMatrix(Ycopy, PETSC_DECIDE, PETSC_DECIDE, 0, R, &Ycopy_r));
    PetscCall(MatCopy(Ycopy_r, *U, SAME_NONZERO_PATTERN));
    PetscCall(MatDenseRestoreSubMatrix(Ycopy, &Ycopy_r));
  }
  if (rank == 0) {
    const PetscScalar *_WS;
    PetscScalar       *_S;
    PetscCall(VecGetArrayRead(WS, &_WS));
    PetscCall(VecGetArrayWrite(*S, &_S));
    PetscCall(PetscArraycpy(_S, _WS, R));
    PetscCall(VecRestoreArrayWrite(*S, &_S));
    PetscCall(VecRestoreArrayRead(WS, &_WS));
  }
  if (VH) {
    Mat WVH_r;

    PetscCall(MatDenseGetSubMatrix(WVH, 0, R, PETSC_DECIDE, PETSC_DECIDE, &WVH_r));
    PetscCall(MatCopy(WVH_r, *VH, SAME_NONZERO_PATTERN));
    PetscCall(MatDenseRestoreSubMatrix(WVH, &WVH_r));
  }
  PetscCall(VecDestroy(&WS));
  if (VH) PetscCall(MatDestroy(&WVH));
  if (U) PetscCall(MatDestroy(&WU));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Y));
  PetscCall(MatDestroy(&Ycopy));
  PetscCall(MatDestroy(&Yupdate));
  if (viewer) PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscOptionsRestoreViewer(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
