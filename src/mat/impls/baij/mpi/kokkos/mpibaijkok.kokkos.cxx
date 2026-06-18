#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petscpkg_version.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <../src/mat/impls/baij/seq/kokkos/baijkokkosimpl.hpp>
#include <map>
#include <set>
#include <vector>

/* Forward declarations */
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqAIJ(Mat, MatType, MatReuse, Mat *);
static PetscErrorCode       MatSetValuesBlocked_MPIBAIJKokkos(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
static PetscErrorCode       MatConvert_MPIBAIJKokkos_MPIAIJ(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIBAIJKokkos(Mat, MatType, MatReuse, Mat *);
static PetscErrorCode       MatBuildMPIBAIJKokkosFromMPIAIJ_Private(Mat, Mat, PetscBool);
static PetscErrorCode       MatSetUpMultiply_MPIBAIJKokkos(Mat);
static PetscErrorCode       MatMPIBAIJKokkosUnCompressB(Mat);
static PetscErrorCode       MatProductSetFromOptions_MPIBAIJKokkos(Mat);
static PetscErrorCode       MatProductSymbolic_MPIBAIJKokkos(Mat);
static PetscErrorCode       MatAXPY_MPIBAIJKokkos(Mat, PetscScalar, Mat, MatStructure);
static PetscErrorCode       MatCreateGraph_MPIBAIJKokkos(Mat, PetscBool, PetscBool, PetscReal, PetscInt, PetscInt[], Mat *);
static PetscErrorCode       MatProductSetFromOptions_mpibaijkokkos_mpiaij_C(Mat);
static PetscErrorCode       MatProductSetFromOptions_mpiaij_mpibaijkokkos_C(Mat);

/*
  MatAssemblyEnd_MPIBAIJKokkos - assemble the parallel rectangular-block Kokkos matrix.

  We do NOT call MatAssemblyEnd_MPIBAIJ(): its MatSetUpMultiply_MPIBAIJ()/MatDisAssemble_MPIBAIJ()
  are hardwired to square blocks (off-diagonal column layout/scatter sized by row_bs) and would
  rebuild the SEQBAIJKOKKOS off-diagonal B as a square SEQBAIJ, destroying the rectangular data.
  Instead we drain the off-process stash through our own rectangular MatSetValuesBlocked() and
  assemble the seq Kokkos A/B directly. The off-diagonal column compression (garray), lvec, and
  VecScatter for MatMult() are built in F1.3; F1.2 leaves B uncompressed (global block-columns).
*/
static PetscErrorCode MatAssemblyEnd_MPIBAIJKokkos(Mat mat, MatAssemblyType mode)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)mat->data;
  PetscInt    *row, *col;
  MatScalar   *val;
  PetscInt     i, j, rstart, ncols, flg, bs2 = baij->bs2;
  PetscBool    r1;
  PetscMPIInt  n, size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat), &size));
  // Drop any cached MPIAIJ copy (MatGetRow/MatNorm/MatGetInfo) — values/structure are about to change
  PetscCall(PetscObjectCompose((PetscObject)mat, "MatMPIBAIJKokkos_cached_aij", NULL));

  /* Drain the off-process stash and replay locally through our rectangular set-values.
     For GAMG prolongators the block rows are local, so the stash is empty; for the square
     operator case bs2 = row_bs*row_bs matches the block stash sizing. */
  if (!baij->donotstash && !mat->nooffprocentries) {
    while (1) {
      PetscCall(MatStashScatterGetMesg_Private(&mat->stash, &n, &row, &col, &val, &flg));
      if (!flg) break;
      for (i = 0; i < n;) {
        for (j = i, rstart = row[j]; j < n; j++)
          if (row[j] != rstart) break;
        ncols = (j < n) ? j - i : n - i;
        PetscCall(MatSetValues(mat, 1, row + i, ncols, col + i, val + i, mat->insertmode));
        i = j;
      }
    }
    PetscCall(MatStashScatterEnd_Private(&mat->stash));
    /* block stash values are stored column-oriented */
    r1                = baij->roworiented;
    baij->roworiented = PETSC_FALSE;
    while (1) {
      PetscCall(MatStashScatterGetMesg_Private(&mat->bstash, &n, &row, &col, &val, &flg));
      if (!flg) break;
      for (i = 0; i < n;) {
        for (j = i, rstart = row[j]; j < n; j++)
          if (row[j] != rstart) break;
        ncols = (j < n) ? j - i : n - i;
        PetscCall(MatSetValuesBlocked(mat, 1, row + i, ncols, col + i, val + i * bs2, mat->insertmode));
        i = j;
      }
    }
    PetscCall(MatStashScatterEnd_Private(&mat->bstash));
    baij->roworiented = r1;
  }

  /* If baij->garray is still set here, the matrix was already fully assembled (B compressed by a
     previous MatSetUpMultiply()) and no values were re-inserted: the set-values routines and the
     stash drain above un-compress B (clearing garray) whenever a real re-fill occurs. This is a
     redundant re-assembly (e.g. ex56 assembles the operator twice), so skip re-processing — calling
     MatAssemblyEnd() on the constructor-built compressed B would compact it (ilen==0) and lose the
     off-diagonal. */
  if (baij->garray) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatAssemblyBegin(baij->A, mode));
  PetscCall(MatAssemblyEnd(baij->A, mode));
  PetscCall(MatAssemblyBegin(baij->B, mode));
  PetscCall(MatAssemblyEnd(baij->B, mode));
  // F1.3: Build the column-compression + garray + lvec + VecScatter for MatMult
  if (mode == MAT_FINAL_ASSEMBLY && size > 1) PetscCall(MatSetUpMultiply_MPIBAIJKokkos(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMPIBAIJKokkosUnCompressB - restore the off-diagonal B to global block-columns.

  After a first assembly, MatSetUpMultiply_MPIBAIJKokkos() compresses B to local block-columns
  [0,ec) (with baij->garray mapping back to global). A subsequent re-assembly inserts off-diagonal
  blocks at GLOBAL columns and must re-compress from scratch; if B is left compressed, those
  insertions (and the re-compression) corrupt the off-diagonal. This undoes the compression: it
  remaps B's block-columns from local back to global via garray, resizes B to rmap->n x cmap->N,
  and clears garray/lvec/Mvctx. No-op if B is not currently compressed (garray == NULL).
*/
static PetscErrorCode MatMPIBAIJKokkosUnCompressB(Mat mat)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ *)mat->data;
  Mat_SeqBAIJKokkos *bc;
  PetscInt           row_bs, col_bs, mbs_loc, Nbs, cnblk, i, j, k;

  PetscFunctionBegin;
  if (!baij->garray || !baij->B) PetscFunctionReturn(PETSC_SUCCESS);
  bc      = (Mat_SeqBAIJKokkos *)baij->B->spptr;
  row_bs  = mat->rmap->bs;
  col_bs  = mat->cmap->bs;
  mbs_loc = baij->mbs;
  Nbs     = baij->Nbs;
  if (bc && mbs_loc > 0) {
    auto ci = bc->i_dual.view_host();
    auto cj = bc->j_dual.view_host();
    auto ca = bc->a_dual.view_host();
    cnblk   = ci[mbs_loc];

    MatRowMapKokkosView gi("B_i_glob", mbs_loc + 1);
    MatColIdxKokkosView gj("B_j_glob", cnblk);
    MatScalarKokkosView ga("B_a_glob", cnblk * row_bs * col_bs);
    auto                gih = Kokkos::create_mirror_view(gi);
    auto                gjh = Kokkos::create_mirror_view(gj);
    auto                gah = Kokkos::create_mirror_view(ga);

    for (i = 0; i <= mbs_loc; i++) gih[i] = ci[i];
    for (k = 0; k < cnblk; k++) {
      gjh[k] = baij->garray[cj[k]]; // local compressed block-col -> global block-col
      for (j = 0; j < row_bs * col_bs; j++) gah[k * row_bs * col_bs + j] = ca[k * row_bs * col_bs + j];
    }
    PetscCallCXX(Kokkos::deep_copy(gi, gih));
    PetscCallCXX(Kokkos::deep_copy(gj, gjh));
    PetscCallCXX(Kokkos::deep_copy(ga, gah));

    PetscCall(MatDestroy(&baij->B));
    PetscCall(MatCreate(PETSC_COMM_SELF, &baij->B));
    PetscCall(MatSetSizes(baij->B, mat->rmap->n, Nbs * col_bs, mat->rmap->n, Nbs * col_bs));
    PetscCall(MatSetBlockSizes(baij->B, row_bs, col_bs));
    PetscCall(MatSetType(baij->B, MATSEQBAIJKOKKOS));
    PetscCallCXX(baij->B->spptr = (void *)new Mat_SeqBAIJKokkos(row_bs, col_bs, mbs_loc, Nbs, cnblk, gi, gj, ga));
    baij->B->assembled = PETSC_TRUE;
    PetscCall(PetscLayoutDestroy(&baij->B->cmap));
    PetscCall(PetscLayoutCreateFromSizes(PETSC_COMM_SELF, Nbs * col_bs, Nbs * col_bs, col_bs, &baij->B->cmap));
    // Constructor leaves a_dual host mirror uninitialized (modify_device); sync so MatSetUpMultiply()'s
    // host re-scan below and any host reader of the uncompressed B see the real off-diagonal values.
    PetscCall(KokkosDualViewSyncHost(((Mat_SeqBAIJKokkos *)baij->B->spptr)->a_dual, PetscGetKokkosExecutionSpace()));
  }
  PetscCall(PetscFree(baij->garray));
  PetscCall(VecDestroy(&baij->lvec));
  PetscCall(VecScatterDestroy(&baij->Mvctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSetUpMultiply_MPIBAIJKokkos - col_bs-aware column compression and scatter setup for MPIBAIJKOKKOS.

  For a distributed rectangular-block matrix with off-diagonal B (sized rmap->n x cmap->N),
  the MatMult requires a VecScatter from the global column-space vector (distributed over cmap)
  to a local halo vector (lvec). The off-diagonal's block columns are global coarse-block indices
  0..Nbs-1. This function:
  1. Scans B's j array to collect unique coarse-block columns used (garray).
  2. Remaps B's block-column indices from global to local [0,ec).
  3. Rebuilds B as a SEQBAIJKOKKOS with the compressed column count.
  4. Creates lvec of size ec*col_bs (the halo vector).
  5. Builds baij->Mvctx, a VecScatter mapping global scalar indices garray[i]*col_bs+k
     to local lvec positions.
*/
static PetscErrorCode MatSetUpMultiply_MPIBAIJKokkos(Mat mat)
{
  Mat_MPIBAIJ       *baij    = (Mat_MPIBAIJ *)mat->data;
  Mat_SeqBAIJKokkos *bseqkok = NULL;
  PetscInt           row_bs, col_bs, ec = 0, *garray;
  PetscInt           i, j, k, nblk, *indices;
  PetscInt           mbs_loc, Nbs;
  IS                 from, to;
  Vec                gvec;
  PetscMPIInt        size;

  PetscFunctionBegin;
  if (!baij->B) PetscFunctionReturn(PETSC_SUCCESS);

  row_bs  = mat->rmap->bs;
  col_bs  = mat->cmap->bs;
  Nbs     = baij->Nbs;
  mbs_loc = baij->mbs;

  // If B was compressed by a previous assembly, restore global columns first so re-compression
  // below is correct (idempotent re-assembly); this also clears old garray/lvec/Mvctx.
  PetscCall(MatMPIBAIJKokkosUnCompressB(mat));

  // NB: do not early-return for an empty local off-diagonal (mbs_loc == 0 or no off-diagonal blocks):
  // the VecCreateMPIWithArray()/VecScatterCreate() below are collective on the matrix communicator and
  // must run on every rank. The scans naturally yield ec == 0 (empty lvec/scatter) for such ranks.
  bseqkok = (Mat_SeqBAIJKokkos *)baij->B->spptr;

  // Access host views to scan block-column indices
  auto i_host = bseqkok->i_dual.view_host();
  auto j_host = bseqkok->j_dual.view_host();
  auto a_host = bseqkok->a_dual.view_host();

  nblk = i_host[mbs_loc]; // Number of blocks in B

  // Mark which global block-columns are used and count unique ones
  PetscCall(PetscCalloc1(Nbs, &indices));
  for (i = 0; i < mbs_loc; i++) {
    for (k = i_host[i]; k < i_host[i + 1]; k++) {
      PetscInt bc = j_host[k]; // GLOBAL block-column
      if (!indices[bc]) ec++;
      indices[bc] = 1;
    }
  }

  // Allocate garray and populate with sorted unique global block-columns
  PetscCall(PetscMalloc1(ec, &garray));
  ec = 0;
  for (i = 0; i < Nbs; i++) {
    if (indices[i]) garray[ec++] = i;
  }

  // Remap indices: indices[g] now contains the local column index for global column g
  for (i = 0; i < Nbs; i++) indices[i] = 0;
  for (i = 0; i < ec; i++) indices[garray[i]] = i;

  // Remap B's j array and copy a array (on host)
  auto j_new_h = Kokkos::create_mirror_view(j_host);
  auto a_new_h = Kokkos::create_mirror_view(a_host);

  for (i = 0; i < mbs_loc; i++) {
    for (k = i_host[i]; k < i_host[i + 1]; k++) {
      // Remap j from global to local compressed index
      j_new_h[k] = indices[j_host[k]];
      // Copy block values
      for (j = 0; j < row_bs * col_bs; j++) {
        a_new_h[k * row_bs * col_bs + j] = a_host[k * row_bs * col_bs + j];
      }
    }
  }

  // Create new device-allocated views by copying from host
  MatRowMapKokkosView i_new_d("B_i_new", mbs_loc + 1);
  MatColIdxKokkosView j_new_d("B_j_new", nblk);
  MatScalarKokkosView a_new_d("B_a_new", nblk * row_bs * col_bs);

  auto i_new_h = Kokkos::create_mirror_view(i_new_d);
  for (i = 0; i <= mbs_loc; i++) i_new_h[i] = i_host[i];
  Kokkos::deep_copy(i_new_d, i_new_h);
  Kokkos::deep_copy(j_new_d, j_new_h);
  Kokkos::deep_copy(a_new_d, a_new_h);

  // Destroy old B and replace with a new compressed SEQBAIJKOKKOS
  PetscCall(MatDestroy(&baij->B));

  PetscCall(MatCreate(PETSC_COMM_SELF, &baij->B));
  PetscCall(MatSetSizes(baij->B, mat->rmap->n, ec * col_bs, mat->rmap->n, ec * col_bs));
  PetscCall(MatSetBlockSizes(baij->B, row_bs, col_bs));
  PetscCall(MatSetType(baij->B, MATSEQBAIJKOKKOS));

  // Construct the Kokkos struct with compressed columns
  Mat_SeqBAIJKokkos *B_new = new Mat_SeqBAIJKokkos(row_bs, col_bs, mbs_loc, ec, nblk, i_new_d, j_new_d, a_new_d);
  baij->B->spptr           = (void *)B_new;
  baij->B->assembled       = PETSC_TRUE;

  // The constructor marks a_dual modify_device with an UNINITIALIZED host mirror. Host-side readers of
  // the compressed B (MatConvert_MPIBAIJKokkos_MPIAIJ(), MatGetRow_MPIBAIJKokkos(), the cached AIJ for
  // MatNorm()/MatGetInfo()) read a_dual.view_host() directly. On a real device that host mirror is
  // separate uninitialized memory, so they would read garbage (host == device memory hides this on the
  // Serial backend). Sync the off-diagonal values to host so every host reader sees the assembled B.
  PetscCall(KokkosDualViewSyncHost(B_new->a_dual, PetscGetKokkosExecutionSpace()));

  // Update B's column layout to reflect ec compressed block-columns
  PetscCall(PetscLayoutDestroy(&baij->B->cmap));
  PetscCall(PetscLayoutCreateFromSizes(PETSC_COMM_SELF, ec * col_bs, ec * col_bs, col_bs, &baij->B->cmap));

  // Create the local halo vector (length ec * col_bs)
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ec * col_bs, &baij->lvec));
  PetscCall(VecSetType(baij->lvec, VECSEQKOKKOS));

  // Build the VecScatter from global columns to lvec
  // Create IS for global column indices: garray[i]*col_bs + k for k in [0,col_bs)
  PetscInt *global_indices;
  PetscCall(PetscMalloc1(ec * col_bs, &global_indices));
  for (i = 0; i < ec; i++) {
    for (j = 0; j < col_bs; j++) {
      global_indices[i * col_bs + j] = garray[i] * col_bs + j;
    }
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ec * col_bs, global_indices, PETSC_OWN_POINTER, &from));

  // Create IS for local lvec indices [0, ec*col_bs)
  PetscInt *local_indices;
  PetscCall(PetscMalloc1(ec * col_bs, &local_indices));
  for (i = 0; i < ec * col_bs; i++) local_indices[i] = i;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ec * col_bs, local_indices, PETSC_OWN_POINTER, &to));

  // Create temporary global vector for scatter
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat), 1, mat->cmap->n, mat->cmap->N, NULL, &gvec));

  // Create the scatter
  PetscCall(VecScatterCreate(gvec, from, baij->lvec, to, &baij->Mvctx));

  baij->garray = garray;

  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(VecDestroy(&gvec));
  PetscCall(PetscFree(indices));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Rectangular-block-aware preallocation for MPIBAIJKOKKOS.

  The diagonal A and off-diagonal B may have col_bs (from cmap->bs) differing from row_bs
  (from rmap->bs) for rectangular operators like GAMG prolongators. We call the base MPIBAIJ
  preallocation to set up the envelope (layouts, stash, row ownership), then:
  - fix the column-side block bookkeeping (cstartbs/cendbs/nbs/Nbs/bs2) into col_bs units, and
  - replace the square SEQBAIJ A/B created by the base with rectangular SEQBAIJKOKKOS blocks
    (A: local-rows x local-diagonal-coarse-cols; B: local-rows x global-coarse-cols, uncompressed
    in F1.2 — the off-diagonal column compression/garray is built in F1.3).
  Column ownership (the A/B split in MatSetValuesBlocked()) is determined by col_bs, not row_bs.
*/
static PetscErrorCode MatMPIBAIJSetPreallocation_MPIBAIJKokkos(Mat mat, PetscInt bs, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  Mat_MPIBAIJ *mpibaij;
  PetscInt     row_bs, col_bs;
  PetscMPIInt  size;

  PetscFunctionBegin;
  row_bs = bs;

  /* Rectangular blocks are signaled by an explicit column block size set via
     MatSetBlockSizes(mat, row_bs, col_bs) before preallocation (e.g. the GAMG prolongator,
     col_bs = #near-null vectors). The default cmap->bs is 1, so only a value > 1 that differs
     from row_bs indicates a genuine rectangular block; otherwise the blocks are square. */
  col_bs = (mat->cmap->bs > 1 && mat->cmap->bs != row_bs) ? mat->cmap->bs : row_bs;

  // Reset the column block size to row_bs before delegating to the base preallocator. The base calls
  // MatSetBlockSize(mat, bs), which locks both row and column block sizes to bs (square). On a *re*-
  // preallocation of an already-built rectangular matrix (e.g. the GAMG block AB/PtAP numeric phase),
  // cmap->bs is still the previous col_bs and the square lock would reject "change col bs col_bs to row_bs".
  // Clearing it here lets the base relock to row_bs cleanly; the true col_bs is restored just below.
  mat->cmap->bs = row_bs;

  // Call base MPIBAIJ preallocation to set up layouts, stash, ownership fields.
  // The base creates square SEQBAIJ A/B which we destroy and recreate below, so pass empty
  // nnz here: the caller's d_nnz/o_nnz are in col_bs-block units and the base would misread
  // them in row_bs units (fatal when col_bs < row_bs). Real allocation happens below.
  PetscCall(MatMPIBAIJSetPreallocation_MPIBAIJ(mat, bs, 0, NULL, 0, NULL));
  mpibaij = static_cast<Mat_MPIBAIJ *>(mat->data);

  // Fix column block size for rectangular case
  mat->cmap->bs = col_bs;

  // Recompute column-side block boundaries in col_bs units
  mpibaij->cstartbs = mat->cmap->rstart / col_bs;
  mpibaij->cendbs   = mat->cmap->rend / col_bs;
  mpibaij->nbs      = mat->cmap->n / col_bs;
  mpibaij->Nbs      = mat->cmap->N / col_bs;

  // Update block-value size (row_bs * col_bs, not row_bs * row_bs)
  mpibaij->bs2 = row_bs * col_bs;

  // Destroy the square A/B created by base call and recreate as rectangular SEQBAIJKOKKOS
  PetscCall(MatDestroy(&mpibaij->A));
  PetscCall(MatDestroy(&mpibaij->B));

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat), &size));

  // Create diagonal block A (local rows x local diagonal columns)
  PetscCall(MatCreate(PETSC_COMM_SELF, &mpibaij->A));
  PetscCall(MatSetSizes(mpibaij->A, mat->rmap->n, mat->cmap->n, mat->rmap->n, mat->cmap->n));
  PetscCall(MatSetBlockSizes(mpibaij->A, row_bs, col_bs));
  PetscCall(MatSetType(mpibaij->A, MATSEQBAIJKOKKOS));
  PetscCall(MatSeqBAIJSetPreallocation(mpibaij->A, row_bs, d_nz, d_nnz));

  // Create off-diagonal block B (local rows x global columns; uncompressed in F1.2).
  // On a single rank B has no columns, so its preallocation must be empty: the caller's
  // o_nz/o_nnz (off-diagonal counts) would otherwise exceed B's zero block-columns.
  PetscCall(MatCreate(PETSC_COMM_SELF, &mpibaij->B));
  PetscCall(MatSetSizes(mpibaij->B, mat->rmap->n, size > 1 ? mat->cmap->N : 0, mat->rmap->n, size > 1 ? mat->cmap->N : 0));
  PetscCall(MatSetBlockSizes(mpibaij->B, row_bs, col_bs));
  PetscCall(MatSetType(mpibaij->B, MATSEQBAIJKOKKOS));
  PetscCall(MatSeqBAIJSetPreallocation(mpibaij->B, row_bs, size > 1 ? o_nz : 0, size > 1 ? o_nnz : NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPIBAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIBAIJSetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpibaijkokkos_mpiaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpiaij_mpibaijkokkos_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_mpibaijkokkos_mpiaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_mpiaij_mpibaijkokkos_C", NULL));
  PetscCall(MatDestroy_MPIBAIJ(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Rectangular-block-aware MatSetValuesBlocked for MPIBAIJKOKKOS.

  GLOBAL block indices on input (exactly like MatSetValuesBlocked_MPIBAIJ).
  For each incoming block at (global block-row im[i], global block-col in[j]):
  - If im[i] is local: split by coarse-column ownership (using col_bs)
    - If in[j] is in diagonal part (in[j] >= cstartbs && in[j] < cendbs): add to A
    - Otherwise: add to B (and manage column map / garray)
  - If im[i] is off-process: stash it; MatAssemblyEnd_MPIBAIJKokkos() drains and replays it locally
*/
/*
  MatSetValues_MPIBAIJKokkos - Insert/add scalar values into a parallel rectangular-block Kokkos matrix.

  Handles scalar MatSetValues calls by routing to the appropriate diagonal (A) or
  off-diagonal (B) sequential block, or stashing off-process entries for later drain
  and replay. Uses column ownership to split diagonal vs off-diagonal.
*/
static PetscErrorCode MatSetValues_MPIBAIJKokkos(Mat mat, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode addv)
{
  Mat_MPIBAIJ *baij        = (Mat_MPIBAIJ *)mat->data;
  PetscBool    roworiented = baij->roworiented;
  PetscInt     i, j, row, col;
  PetscInt     rstart = mat->rmap->rstart, rend = mat->rmap->rend; /* Scalar row ownership */
  PetscInt     cstart = mat->cmap->rstart, cend = mat->cmap->rend; /* Scalar column ownership (diagonal part) */
  PetscScalar  value;

  PetscFunctionBegin;
  // Re-assembly into a previously compressed B: restore global columns before inserting
  if (baij->garray) PetscCall(MatMPIBAIJKokkosUnCompressB(mat));
  for (i = 0; i < m; i++) {
    if (im[i] < 0) continue;
    PetscCheck(im[i] < mat->rmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", im[i], mat->rmap->N);

    /* Check if im[i] (scalar row index) is owned by this rank */
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart; /* Convert to local scalar row index */
      for (j = 0; j < n; j++) {
        if (in[j] < 0) continue;
        PetscCheck(in[j] < mat->cmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", in[j], mat->cmap->N);

        /* Extract scalar value respecting row/column orientation */
        if (roworiented) {
          value = v[i * n + j];
        } else {
          value = v[i + j * m];
        }

        /* Check if the scalar column is in the diagonal (A) or off-diagonal (B) part */
        if (in[j] >= cstart && in[j] < cend) {
          col = in[j] - cstart; /* Local scalar column in diagonal part */
          PetscCall(MatSetValues(baij->A, 1, &row, 1, &col, &value, addv));
        } else {
          /* Off-diagonal: use global scalar column index directly (B is uncompressed in F1.2) */
          col = in[j];
          PetscCall(MatSetValues(baij->B, 1, &row, 1, &col, &value, addv));
        }
      }
    } else {
      /* Off-process scalar row: stash it; MatAssemblyEnd_MPIBAIJKokkos() drains and replays locally */
      PetscCheck(!mat->nooffprocentries, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Setting off process row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set", im[i]);
      if (!baij->donotstash) {
        if (roworiented) {
          PetscCall(MatStashValuesRow_Private(&mat->stash, im[i], n, in, v + i * n, PETSC_FALSE));
        } else {
          PetscCall(MatStashValuesCol_Private(&mat->stash, im[i], n, in, v + i, m, PETSC_FALSE));
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesBlocked_MPIBAIJKokkos(Mat mat, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode addv)
{
  Mat_MPIBAIJ *baij        = (Mat_MPIBAIJ *)mat->data;
  MatScalar   *barray      = baij->barray;
  PetscBool    roworiented = baij->roworiented;
  PetscInt     i, j, ii, jj, row, col;
  PetscInt     row_bs = mat->rmap->bs, col_bs = mat->cmap->bs;
  PetscInt     block_size = row_bs * col_bs;
  PetscInt     row_in, col_in;
  PetscInt     rstart = baij->rstartbs, rend = baij->rendbs; /* Block-row ownership bounds */
  PetscInt     cstart = baij->cstartbs, cend = baij->cendbs; /* Block-column ownership bounds (in col_bs units) */

  PetscFunctionBegin;
  // Re-assembly into a previously compressed B: restore global columns before inserting
  if (baij->garray) PetscCall(MatMPIBAIJKokkosUnCompressB(mat));
  if (!barray) {
    PetscCall(PetscMalloc1(block_size, &barray));
    baij->barray = barray;
  }

  for (i = 0; i < m; i++) {
    if (im[i] < 0) continue;
    PetscCheck(im[i] < baij->Mbs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Block indexed row too large %" PetscInt_FMT " max %" PetscInt_FMT, im[i], baij->Mbs - 1);

    /* Check if im[i] (GLOBAL block-row index) is owned by this rank */
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart; /* Convert to local block-row index */
      for (j = 0; j < n; j++) {
        if (in[j] < 0) continue;

        /* Copy the (i,j) block into barray as row_bs x col_bs ROW-MAJOR (what the seq Kokkos
           A/B require). Handle both row-oriented (default) and column-oriented caller input;
           the input is the logical (m*row_bs) x (n*col_bs) dense block layout. */
        for (ii = 0; ii < row_bs; ii++) {
          for (jj = 0; jj < col_bs; jj++) {
            if (roworiented) barray[ii * col_bs + jj] = v[(i * row_bs + ii) * (n * col_bs) + (j * col_bs + jj)];
            else barray[ii * col_bs + jj] = v[(j * col_bs + jj) * (m * row_bs) + (i * row_bs + ii)];
          }
        }

        PetscCheck(in[j] < baij->Nbs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Block indexed column too large %" PetscInt_FMT " max %" PetscInt_FMT, in[j], baij->Nbs - 1);

        /* Check if the block column is in the local diagonal (A) or off-diagonal (B) part
           (cstart/cend are in col_bs units)
        */
        if (in[j] >= cstart && in[j] < cend) {
          col    = in[j] - cstart;
          row_in = row;
          col_in = col;
          PetscCall(MatSetValuesBlocked(baij->A, 1, &row_in, 1, &col_in, barray, addv));
        } else {
          /* Off-diagonal: store at global block-column index (F1.2 uncompressed)
             B is sized rmap->n x cmap->N, so we use global block-column in[j] directly.
          */
          row_in = row;
          col_in = in[j]; /* GLOBAL block-column index */
          PetscCall(MatSetValuesBlocked(baij->B, 1, &row_in, 1, &col_in, barray, addv));
        }
      }
    } else {
      /* Off-process block row: stash it; MatAssemblyEnd_MPIBAIJKokkos() drains the stash and
         replays the entries locally through this routine. */
      PetscCheck(!mat->nooffprocentries, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Setting off process block indexed row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set", im[i]);
      if (!baij->donotstash) {
        if (roworiented) {
          PetscCall(MatStashValuesRowBlocked_Private(&mat->bstash, im[i], n, in, v, m, n, i));
        } else {
          PetscCall(MatStashValuesColBlocked_Private(&mat->bstash, im[i], n, in, v, m, n, i));
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMult_MPIBAIJKokkos - Compute y = A*x for a parallel rectangular-block Kokkos matrix.

  Uses VecScatter to gather remote coarse-DOF halo, then calls local sequential
  block-sparse MatMult on the diagonal and off-diagonal blocks.
  For serial (nproc=1), skips the scatter since there are no remote blocks.
*/
static PetscErrorCode MatMult_MPIBAIJKokkos(Mat A, Vec xx, Vec yy)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)A->data;
  PetscInt     nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  PetscCheck(nt == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of A (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", A->cmap->n, nt);
  PetscUseTypeMethod(baij->A, mult, xx, yy);
  if (baij->Mvctx) {
    PetscCall(VecScatterBegin(baij->Mvctx, xx, baij->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(baij->Mvctx, xx, baij->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscUseTypeMethod(baij->B, multadd, baij->lvec, yy, yy);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultAdd_MPIBAIJKokkos - Compute z = y + A*x for a parallel rectangular-block Kokkos matrix.
*/
static PetscErrorCode MatMultAdd_MPIBAIJKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)A->data;
  PetscInt     nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  PetscCheck(nt == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of A (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", A->cmap->n, nt);
  PetscUseTypeMethod(baij->A, multadd, xx, yy, zz);
  if (baij->Mvctx) {
    PetscCall(VecScatterBegin(baij->Mvctx, xx, baij->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(baij->Mvctx, xx, baij->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscUseTypeMethod(baij->B, multadd, baij->lvec, zz, zz);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultTranspose_MPIBAIJKokkos - Compute y = A^T x for a parallel rectangular-block Kokkos matrix.

  Computes off-diagonal (B^T x -> lvec) and diagonal (A^T x -> yy) separately,
  then gathers off-diagonal contributions via reverse scatter into yy.
*/
static PetscErrorCode MatMultTranspose_MPIBAIJKokkos(Mat A, Vec xx, Vec yy)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)A->data;
  PetscInt     nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  PetscCheck(nt == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of A (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", A->rmap->n, nt);
  if (baij->Mvctx) PetscUseTypeMethod(baij->B, multtranspose, xx, baij->lvec);
  PetscUseTypeMethod(baij->A, multtranspose, xx, yy);
  if (baij->Mvctx) {
    PetscCall(VecScatterBegin(baij->Mvctx, baij->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(baij->Mvctx, baij->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultTransposeAdd_MPIBAIJKokkos - Compute z = y + A^T x for a parallel rectangular-block Kokkos matrix.

  Computes off-diagonal (B^T x -> lvec), local (A^T x + y -> z), then gathers
  off-diagonal contributions via reverse scatter into z.
*/
static PetscErrorCode MatMultTransposeAdd_MPIBAIJKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)A->data;
  PetscInt     nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  PetscCheck(nt == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of A (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", A->rmap->n, nt);
  if (baij->Mvctx) PetscUseTypeMethod(baij->B, multtranspose, xx, baij->lvec);
  PetscUseTypeMethod(baij->A, multtransposeadd, xx, yy, zz);
  if (baij->Mvctx) {
    PetscCall(VecScatterBegin(baij->Mvctx, baij->lvec, zz, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(baij->Mvctx, baij->lvec, zz, ADD_VALUES, SCATTER_REVERSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  F10.2 — fully-blocked parallel PtAP. The coarse Galerkin operator C = P^T A P is returned as
  MATMPIBAIJKOKKOS (not MPIAIJ), so the GAMG solve hot path (coarse SpMV + smoothers) runs the block
  kernel at every level. "Blocked at all times" needs the coarse *operator* to be block, not the PtAP
  *compute* to be block-native, so we repackage the mature MPIAIJ PtAP result into block layout:
    1. operands A, P -> AIJ temporaries (NOT in place: product->A/B stay alive — the MatProductSymbolic()
       interface tail dereferences them in its MatSetBlockSizes() step, so freeing them is use-after-free);
    2. Cresult = MatPtAP(Aaij, Paij) into a reusable temporary (NOT on C);
    3. build C in place as block from Cresult (C is born block, never handed to AIJ ⇒ no MatHeaderReplace).
  The AIJ matrices (Aaij, Paij, Cresult) are STRICTLY TRANSIENT — created and freed within each
  symbolic/numeric call, never held for the lifetime of the coarse operator. So the persistent storage is
  the block C alone (the memory win), with only a transient setup-time peak holding both representations.
  That transient peak, and the AIJ spgemm itself, are what the deferred F2.2 block-native compute removes
  (local part on the seq native block PtAP); here numeric recomputes from scratch, so a MAT_INITIAL_MATRIX
  GAMG build runs the AIJ PtAP twice — acceptable for correctness-first F10.
*/
static PetscErrorCode MatProductComputeBlock_MPIBAIJKokkos(Mat C, PetscBool valuesonly)
{
  Mat_Product *product = C->product;
  Mat          A = product->A, B = product->B, Aaij, Baij, Cresult;
  PetscInt     row_bs, col_bs;

  PetscFunctionBegin;
  /* Operands -> transient AIJ. Do NOT convert product->A/B in place: the interface tail of
     MatProductSymbolic() reads the original A/B (block sizes), so they must stay alive. */
  PetscCall(MatConvert(A, MATMPIAIJ, MAT_INITIAL_MATRIX, &Aaij));
  PetscCall(MatConvert(B, MATMPIAIJ, MAT_INITIAL_MATRIX, &Baij));
  switch (product->type) {
  case MATPRODUCT_AB:
    /* C = A*B; block sizes (A rows) x (B cols). Used for GAMG prolongator smoothing (A*P0). */
    PetscCall(MatMatMult(Aaij, Baij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Cresult));
    row_bs = A->rmap->bs > 0 ? A->rmap->bs : 1;
    col_bs = B->cmap->bs > 0 ? B->cmap->bs : 1;
    break;
  case MATPRODUCT_PtAP:
    /* C = P^T A P; square block, size = P column block (near-null-space dimension). */
    PetscCall(MatPtAP(Aaij, Baij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Cresult));
    col_bs = B->cmap->bs > 0 ? B->cmap->bs : 1;
    row_bs = col_bs;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "Block product path not implemented for %s", MatProductTypes[product->type]);
  }
  PetscCall(MatSetBlockSizes(Cresult, row_bs, col_bs));
  /* Build C in place as block: structure + values when !valuesonly (symbolic), values only otherwise. */
  PetscCall(MatBuildMPIBAIJKokkosFromMPIAIJ_Private(Cresult, C, valuesonly));
  PetscCall(MatDestroy(&Aaij));
  PetscCall(MatDestroy(&Baij));
  PetscCall(MatDestroy(&Cresult));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumericBlock_MPIBAIJKokkos(Mat C)
{
  PetscFunctionBegin;
  /* Full rebuild (re-size + re-preallocate + refill): the AIJ operands/result are transient and the block
     C is recomputed from scratch, so structure and values are refreshed together. A values-only refill is
     not possible because the seq Kokkos blocks keep their CSR in device DualViews (base a->i is NULL after
     assembly), so MatZeroEntries() on the assembled block has no host structure to clear. The re-preallocation
     of the existing rectangular C is made safe by MatMPIBAIJSetPreallocation_MPIBAIJKokkos(), which clears the
     locked column block size before delegating to the square-only base preallocator. F2.2 will make this a
     block-native values recompute that avoids the transient AIJ product entirely. */
  PetscCall(MatProductComputeBlock_MPIBAIJKokkos(C, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolicBlock_MPIBAIJKokkos(Mat C)
{
  PetscBool isbaijkok;

  PetscFunctionBegin;
  /* C must be MATMPIBAIJKOKKOS so the block build (MatMPIBAIJSetPreallocation, block assembly) is in
     effect; set it before wiring productnumeric since MatSetType resets C->ops. */
  PetscCall(PetscObjectTypeCompare((PetscObject)C, MATMPIBAIJKOKKOS, &isbaijkok));
  if (!isbaijkok) PetscCall(MatSetType(C, MATMPIBAIJKOKKOS));

  PetscCall(MatProductComputeBlock_MPIBAIJKokkos(C, PETSC_FALSE));
  C->ops->productnumeric = MatProductNumericBlock_MPIBAIJKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductSymbolic_MPIBAIJKokkos - Compute symbolic stage for parallel products (AB, AtB, PtAP, RARt, ABC).

  PtAP (F10.2) and AB (F10.4) return a block C: PtAP makes the coarse Galerkin operators block; AB makes
  the GAMG-smoothed prolongator (A*P0) block so interpolation/restriction also run the block kernel. The
  remaining product types (AtB, RARt, ABC) still convert all block operands to MPIAIJ in place and
  re-dispatch, leaving C as MPIAIJ — none feeds the default agg GAMG hot path. The convert-and-redispatch
  keeps C's header intact (only product->A/B/C *contents* change), so MatProductSymbolic() proceeds without
  use-after-free of the stale product pointer. Operand snapshots are taken here, so MAT_REUSE_MATRIX
  numeric-only reuse recomputes from snapshot (acceptable for GAMG/ex56 which use MAT_INITIAL_MATRIX).
*/
static PetscErrorCode MatProductSymbolic_MPIBAIJKokkos(Mat C)
{
  Mat_Product *product = C->product;
  Mat          A, B, Cc;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_PtAP || product->type == MATPRODUCT_AB) {
    PetscCall(MatProductSymbolicBlock_MPIBAIJKokkos(C));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  // Convert each block operand to MPIAIJ in place, releasing the product's reference
  A = product->A;
  PetscCall(MatConvert(A, MATMPIAIJ, MAT_INITIAL_MATRIX, &product->A));
  PetscCall(MatDestroy(&A));
  B = product->B;
  PetscCall(MatConvert(B, MATMPIAIJ, MAT_INITIAL_MATRIX, &product->B));
  PetscCall(MatDestroy(&B));
  if (product->C) {
    Cc = product->C;
    PetscCall(MatConvert(Cc, MATMPIAIJ, MAT_INITIAL_MATRIX, &product->C));
    PetscCall(MatDestroy(&Cc));
  }
  // Re-dispatch now that all operands are MPIAIJ
  PetscCall(MatProductSetFromOptions(C));
  PetscCall(MatProductSymbolic(C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductSetFromOptions_MPIBAIJKokkos - Install symbolic for supported products.

  Supported product types: AB, AtB, PtAP, RARt, ABC, all dispatched via MatProductSymbolic_MPIBAIJKokkos.
  AB and PtAP return a block C (F10.4/F10.2); the rest convert and redispatch, leaving C as MPIAIJ.

  Also composed on MPIAIJ operand positions (via composed functions) so mixed-type
  products (e.g. MPIBAIJKokkos x MPIAIJ) are handled.
*/
static PetscErrorCode MatProductSetFromOptions_MPIBAIJKokkos(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
  case MATPRODUCT_AtB:
  case MATPRODUCT_PtAP:
  case MATPRODUCT_RARt:
  case MATPRODUCT_ABC:
    C->ops->productsymbolic = MatProductSymbolic_MPIBAIJKokkos;
    break;
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Composed query functions for mixed-type products:
  When a block operand is paired with an MPIAIJ operand, PETSc queries for
  MatProductSetFromOptions_mpibaijkokkos_mpiaij[_C]_C or the reverse.
  These shims set the symbolic to our convert-and-redispatch implementation.
*/
static PetscErrorCode MatProductSetFromOptions_mpibaijkokkos_mpiaij_C(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_MPIBAIJKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_mpiaij_mpibaijkokkos_C(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_MPIBAIJKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatAXPY_MPIBAIJKokkos - Compute Y += alpha*X for parallel rectangular-block Kokkos matrices.

  For SAME_NONZERO_PATTERN and SUBSET_NONZERO_PATTERN, delegates to A and B.
  Otherwise delegates to the base implementation (which will convert and try MPIAIJ machinery).
*/
static PetscErrorCode MatAXPY_MPIBAIJKokkos(Mat Y, PetscScalar alpha, Mat X, MatStructure str)
{
  Mat_MPIBAIJ *ybaij = (Mat_MPIBAIJ *)Y->data;
  Mat_MPIBAIJ *xbaij = (Mat_MPIBAIJ *)X->data;

  PetscFunctionBegin;
  PetscCheckTypeName(Y, MATMPIBAIJKOKKOS);
  PetscCheckTypeName(X, MATMPIBAIJKOKKOS);

  if (str == SAME_NONZERO_PATTERN) {
    // Identical nonzero pattern => identical off-diagonal garray, so the compressed B blocks share a
    // column space and the per-block device AXPY is valid. Operate directly on the SEQBAIJKOKKOS blocks.
    PetscCall(MatAXPY(ybaij->A, alpha, xbaij->A, str));
    PetscCall(MatAXPY(ybaij->B, alpha, xbaij->B, str));
  } else {
    // SUBSET/DIFFERENT: X and Y have different off-diagonal compressions (distinct garray), so their
    // compressed B blocks are not in the same local column space and cannot be added block-for-block.
    // Reconcile in global column space via MPIAIJ (mirrors the base MatAXPY_MPIBAIJ SUBSET path, which
    // falls back to the global-index MatAXPY_Basic). This is a setup-time operation, not the solve hot path.
    Mat Yaij, Xaij, Ynew;
    PetscCall(MatConvert(Y, MATMPIAIJ, MAT_INITIAL_MATRIX, &Yaij));
    PetscCall(MatConvert(X, MATMPIAIJ, MAT_INITIAL_MATRIX, &Xaij));
    PetscCall(MatAXPY(Yaij, alpha, Xaij, str));
    /* Repackage the AIJ sum back into a block matrix (block sizes are preserved across the AIJ round-trip)
       and swap it into Y so the caller's handle holds the result. */
    PetscCall(MatConvert(Yaij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &Ynew));
    PetscCall(MatHeaderReplace(Y, &Ynew));
    PetscCall(MatDestroy(&Yaij));
    PetscCall(MatDestroy(&Xaij));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatCreateGraph_MPIBAIJKokkos - Build scalar aggregation graph from MPIBAIJKokkos matrix for GAMG.

  Not Collective

  Input Parameters:
+ A     - the rectangular-block matrix
. sym   - whether the graph should be symmetrized
. scale - whether graph edge weights should be symmetrically scaled with diagonal
. filter - filter value for entries (< 0: does nothing; == 0: removes only 0.0; > 0: removes abs entries <= value)
. num_idx - size of index array (0 means use full block for weight computation)
. index - (optional) array of block row/column indices for weight computation
- graph - output scalar AIJ graph with block collapse applied

  Level: advanced

  Notes:
  For rectangular-block matrices with row_bs > 1, the blocks are collapsed to scalar nodes
  using MatGetBlockSize semantics: each block-row maps to a scalar node. This enables GAMG
  to build aggregations. The function converts the matrix to scalar MPIAIJ, calls MatCreateGraph
  on the MPIAIJ (which collapses blocks), then returns the result.

.seealso: [](ch_matrices), `Mat`, `MatCreateGraph()`, `PCGAMG`, `MATMPIBAIJKOKKOS`
*/
static PetscErrorCode MatCreateGraph_MPIBAIJKokkos(Mat A, PetscBool sym, PetscBool scale, PetscReal filter, PetscInt num_idx, PetscInt index[], Mat *graph)
{
  Mat      Aaij;
  PetscInt rbs, cbs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, sym, 2);
  PetscValidLogicalCollectiveBool(A, scale, 3);
  PetscAssertPointer(graph, 7);

  // Convert to scalar MPIAIJ to leverage the existing block-collapse logic
  PetscCall(MatConvert(A, MATMPIAIJ, MAT_INITIAL_MATRIX, &Aaij));

  // Ensure block sizes are set on the AIJ matrix so the collapse groups dofs into nodes correctly
  PetscCall(MatGetBlockSizes(A, &rbs, &cbs));
  PetscCall(MatSetBlockSizes(Aaij, rbs, cbs));

  // Build graph from the AIJ (its MatCreateGraph_Simple_AIJ will collapse blocks)
  PetscCall(MatCreateGraph(Aaij, sym, scale, filter, num_idx, index, graph));

  PetscCall(MatDestroy(&Aaij));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMPIBAIJKokkosGetCachedAIJ - lazily build and cache a value-exact MPIAIJ copy of A.

  The collective scalar-view ops MatNorm() and MatGetInfo() reach into baij->A/B by casting their
  ->data to Mat_SeqBAIJ and reading the raw i/j/a arrays, assuming square blocks. Our sub-blocks are
  SEQBAIJKOKKOS (rectangular, data in spptr), so those base routines return zero (report 0 nnz, which
  skews GAMG's per-process load balancing and changes the coarse hierarchy). We delegate them to a
  cached MPIAIJ conversion (value-exact, validated in F1.4). Lazy build is safe here because both
  callers are collective. MatGetRow() is Not Collective and must not build this cache (an empty-rank
  deadlock) — it has its own local MatGetRow_MPIBAIJKokkos(). The cache is composed on A (destroyed
  with it) and invalidated on (re)assembly.
*/
static PetscErrorCode MatMPIBAIJKokkosGetCachedAIJ(Mat A, Mat *aij)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "MatMPIBAIJKokkos_cached_aij", (PetscObject *)aij));
  if (!*aij) {
    PetscCall(MatConvert(A, MATMPIAIJ, MAT_INITIAL_MATRIX, aij));
    PetscCall(PetscObjectCompose((PetscObject)A, "MatMPIBAIJKokkos_cached_aij", (PetscObject)*aij));
    PetscCall(MatDestroy(aij)); /* compose holds the reference */
    PetscCall(PetscObjectQuery((PetscObject)A, "MatMPIBAIJKokkos_cached_aij", (PetscObject *)aij));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatGetRow_MPIBAIJKokkos - Not Collective, purely local. Mirrors MatGetRow_MPIBAIJ()/MatGetRow_MPIAIJ():
  reads the local diagonal (A) and off-diagonal (B) block-rows and merges them into one scalar row sorted
  by global column, with B mapped to global columns through baij->garray. Unlike the base MPIBAIJ routine
  this is rectangular-aware (row_bs != col_bs) and reads the SEQBAIJKOKKOS host views (valid after
  assembly) directly rather than the base Mat_SeqBAIJ arrays. Being local, it does NOT trigger the
  collective MatConvert cache build, so a rank that owns no rows simply never calls it — no empty-rank
  collective imbalance (cf. MatAXPY_Basic() in GAMG prolongator smoothing).
*/
static PetscErrorCode MatGetRow_MPIBAIJKokkos(Mat mat, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  Mat_MPIBAIJ       *baij   = (Mat_MPIBAIJ *)mat->data;
  Mat_SeqBAIJKokkos *Ak     = baij->A ? (Mat_SeqBAIJKokkos *)baij->A->spptr : NULL;
  Mat_SeqBAIJKokkos *Bk     = baij->B ? (Mat_SeqBAIJKokkos *)baij->B->spptr : NULL;
  PetscInt           row_bs = mat->rmap->bs, col_bs = mat->cmap->bs, bs2 = row_bs * col_bs;
  PetscInt           rstart = mat->rmap->rstart, cstart = mat->cmap->rstart;
  PetscInt           lrow, brow, ir, nzA = 0, nzB = 0, k, jj, p, ksplit;
  PetscInt          *idx_p;
  PetscScalar       *v_p;

  PetscFunctionBegin;
  PetscCheck(row >= rstart && row < mat->rmap->rend, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Only local rows");
  PetscCheck(!baij->getrowactive, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Already active");
  baij->getrowactive = PETSC_TRUE;

  PetscCheck(Ak && Bk, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "MATMPIBAIJKOKKOS blocks not built");
  auto ai = Ak->i_dual.view_host();
  auto aj = Ak->j_dual.view_host();
  auto aa = Ak->a_dual.view_host();
  auto bi = Bk->i_dual.view_host();
  auto bj = Bk->j_dual.view_host();
  auto ba = Bk->a_dual.view_host();

  /* One-time scratch sized to the longest local scalar row, as in MatGetRow_MPIBAIJ(). */
  if (!baij->rowvalues && (idx || v)) {
    PetscInt max = 1, i, na, nb;
    for (i = 0; i < baij->mbs; i++) {
      na = Ak ? ai[i + 1] - ai[i] : 0;
      nb = Bk ? bi[i + 1] - bi[i] : 0;
      if (max < na + nb) max = na + nb;
    }
    PetscCall(PetscMalloc2(max * col_bs, &baij->rowvalues, max * col_bs, &baij->rowindices));
  }

  lrow = row - rstart;
  brow = lrow / row_bs;
  ir   = lrow % row_bs;
  if (Ak) nzA = (ai[brow + 1] - ai[brow]) * col_bs;
  if (Bk) nzB = (bi[brow + 1] - bi[brow]) * col_bs;
  *nz = nzA + nzB;

  if ((v || idx) && *nz) {
    v_p   = baij->rowvalues;
    idx_p = baij->rowindices;
    p     = 0;
    /* B blocks below the diagonal range (garray sorted ascending => global columns are monotone). */
    ksplit = Bk ? bi[brow + 1] : 0;
    if (Bk) {
      for (k = bi[brow]; k < bi[brow + 1]; k++) {
        PetscInt gcol0 = (baij->garray ? baij->garray[bj[k]] : bj[k]) * col_bs;
        if (gcol0 >= cstart) {
          ksplit = k;
          break;
        }
        for (jj = 0; jj < col_bs; jj++, p++) {
          if (idx) idx_p[p] = gcol0 + jj;
          if (v) v_p[p] = ba[k * bs2 + ir * col_bs + jj];
        }
      }
    }
    /* Diagonal block A. */
    if (Ak) {
      for (k = ai[brow]; k < ai[brow + 1]; k++) {
        PetscInt gcol0 = cstart + aj[k] * col_bs;
        for (jj = 0; jj < col_bs; jj++, p++) {
          if (idx) idx_p[p] = gcol0 + jj;
          if (v) v_p[p] = aa[k * bs2 + ir * col_bs + jj];
        }
      }
    }
    /* B blocks above the diagonal range. */
    if (Bk) {
      for (k = ksplit; k < bi[brow + 1]; k++) {
        PetscInt gcol0 = (baij->garray ? baij->garray[bj[k]] : bj[k]) * col_bs;
        for (jj = 0; jj < col_bs; jj++, p++) {
          if (idx) idx_p[p] = gcol0 + jj;
          if (v) v_p[p] = ba[k * bs2 + ir * col_bs + jj];
        }
      }
    }
    if (v) *v = v_p;
    if (idx) *idx = idx_p;
  } else {
    if (v) *v = NULL;
    if (idx) *idx = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRestoreRow_MPIBAIJKokkos(Mat mat, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCheck(baij->getrowactive, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "MatGetRow() must be called first");
  baij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNorm_MPIBAIJKokkos(Mat A, NormType type, PetscReal *nrm)
{
  Mat aij = NULL;

  PetscFunctionBegin;
  PetscCall(MatMPIBAIJKokkosGetCachedAIJ(A, &aij));
  PetscCall(MatNorm(aij, type, nrm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetInfo_MPIBAIJKokkos(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat aij = NULL;

  PetscFunctionBegin;
  PetscCall(MatMPIBAIJKokkosGetCachedAIJ(A, &aij));
  PetscCall(MatGetInfo(aij, flag, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatCreateSubMatrix_MPIBAIJKokkos - Extract a parallel submatrix of a rectangular-block Kokkos matrix.

  The base MPIBAIJ submatrix path assumes square blocks and host CSR storage, so it cannot handle the
  rectangular-block SEQBAIJKOKKOS blocks (device DualViews). Reconcile through MPIAIJ: the row/column index
  sets carry the block sizes, so the round-trip preserves the block layout. This runs during GAMG level
  setup (prolongator repartitioning), not in the solve hot path.
*/
static PetscErrorCode MatCreateSubMatrix_MPIBAIJKokkos(Mat mat, IS isrow, IS iscol, MatReuse call, Mat *newmat)
{
  Mat Aaij, Csub, Cblk;

  PetscFunctionBegin;
  PetscCall(MatConvert(mat, MATMPIAIJ, MAT_INITIAL_MATRIX, &Aaij));
  PetscCall(MatCreateSubMatrix(Aaij, isrow, iscol, MAT_INITIAL_MATRIX, &Csub));
  PetscCall(MatConvert(Csub, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &Cblk));
  if (call == MAT_REUSE_MATRIX) PetscCall(MatHeaderReplace(*newmat, &Cblk));
  else *newmat = Cblk;
  PetscCall(MatDestroy(&Aaij));
  PetscCall(MatDestroy(&Csub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetOps_MPIBAIJKokkos(Mat B)
{
  PetscFunctionBegin;
  B->boundtocpu                 = PetscDefined(HAVE_KOKKOS_WITHOUT_GPU) ? PETSC_TRUE : PETSC_FALSE;
  B->ops->assemblyend           = MatAssemblyEnd_MPIBAIJKokkos;
  B->ops->destroy               = MatDestroy_MPIBAIJKokkos;
  B->ops->setvalues             = MatSetValues_MPIBAIJKokkos;
  B->ops->setvaluesblocked      = MatSetValuesBlocked_MPIBAIJKokkos;
  B->ops->getrow                = MatGetRow_MPIBAIJKokkos;
  B->ops->restorerow            = MatRestoreRow_MPIBAIJKokkos;
  B->ops->norm                  = MatNorm_MPIBAIJKokkos;
  B->ops->getinfo               = MatGetInfo_MPIBAIJKokkos;
  B->ops->mult                  = MatMult_MPIBAIJKokkos;
  B->ops->multadd               = MatMultAdd_MPIBAIJKokkos;
  B->ops->multtranspose         = MatMultTranspose_MPIBAIJKokkos;
  B->ops->multtransposeadd      = MatMultTransposeAdd_MPIBAIJKokkos;
  B->ops->axpy                  = MatAXPY_MPIBAIJKokkos;
  B->ops->createsubmatrix       = MatCreateSubMatrix_MPIBAIJKokkos;
  B->ops->creategraph           = MatCreateGraph_MPIBAIJKokkos;
  B->ops->productsetfromoptions = MatProductSetFromOptions_MPIBAIJKokkos;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIBAIJSetPreallocation_C", MatMPIBAIJSetPreallocation_MPIBAIJKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpibaijkokkos_mpiaij_C", MatConvert_MPIBAIJKokkos_MPIAIJ));
  /* Reverse convert (MPIAIJ -> MPIBAIJKOKKOS) queried on the destination type via MatConvert step (2). */
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpibaijkokkos_C", MatConvert_MPIAIJ_MPIBAIJKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPIBAIJKokkos(Mat A, MatType, MatReuse reuse, Mat *newmat)
{
  Mat          B;
  Mat_MPIBAIJ *a;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(A, *newmat, SAME_NONZERO_PATTERN));
  }
  B = *newmat;

  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECKOKKOS, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIBAIJKOKKOS));

  a = static_cast<Mat_MPIBAIJ *>(A->data);
  if (a->A) PetscCall(MatSetType(a->A, MATSEQBAIJKOKKOS));
  if (a->B) PetscCall(MatSetType(a->B, MATSEQBAIJKOKKOS));
  if (a->lvec) PetscCall(VecSetType(a->lvec, VECSEQKOKKOS));
  PetscCall(MatSetOps_MPIBAIJKokkos(B));

  // Compose mixed-type product query functions for GAMG
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpibaijkokkos_mpiaij_C", MatProductSetFromOptions_mpibaijkokkos_mpiaij_C));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpiaij_mpibaijkokkos_C", MatProductSetFromOptions_mpiaij_mpibaijkokkos_C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Convert MATMPIBAIJKOKKOS to MPIAIJ via global MatSetValues.

  For F1.3, B is compressed: block-column indices are local [0,ec) and map to global
  via garray. We iterate through blocks in A and B, expanding each to scalar entries
  with proper global indices, and emit via MatSetValues.
*/
PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJKokkos_MPIAIJ(Mat A, MatType, MatReuse reuse, Mat *newmat)
{
  Mat_MPIBAIJ *a_baij = (Mat_MPIBAIJ *)A->data;
  Mat          B;
  PetscInt     row_bs, col_bs, rstartbs, cstartbs, mbs_loc;
  PetscInt     br, k, ii, jj, bc;

  PetscFunctionBegin;
  PetscCheck(A->assembled, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Matrix must be assembled");

  row_bs   = A->rmap->bs;
  col_bs   = A->cmap->bs;
  rstartbs = a_baij->rstartbs;
  cstartbs = a_baij->cstartbs;
  mbs_loc  = a_baij->mbs;

  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
    PetscCall(MatZeroEntries(B));
  } else {
    Mat_SeqBAIJKokkos *aseqkok = a_baij->A ? (Mat_SeqBAIJKokkos *)a_baij->A->spptr : NULL;
    Mat_SeqBAIJKokkos *bseqkok = a_baij->B ? (Mat_SeqBAIJKokkos *)a_baij->B->spptr : NULL;
    PetscInt          *d_nnz, *o_nnz;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetType(B, MATMPIAIJ));
    PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
    PetscCall(MatSetBlockSizes(B, row_bs, col_bs));

    /* Exact scalar preallocation from the block structure: each scalar row of block-row br has
       (A-blocks in br)*col_bs diagonal and (B-blocks in br)*col_bs off-diagonal nonzeros. */
    PetscCall(PetscCalloc2(A->rmap->n, &d_nnz, A->rmap->n, &o_nnz));
    for (br = 0; br < mbs_loc; br++) {
      PetscInt nd = 0, no = 0;
      if (aseqkok && aseqkok->mbs > 0) nd = (aseqkok->i_dual.view_host()[br + 1] - aseqkok->i_dual.view_host()[br]) * col_bs;
      if (bseqkok && bseqkok->mbs > 0) no = (bseqkok->i_dual.view_host()[br + 1] - bseqkok->i_dual.view_host()[br]) * col_bs;
      for (ii = 0; ii < row_bs; ii++) {
        d_nnz[br * row_bs + ii] = nd;
        o_nnz[br * row_bs + ii] = no;
      }
    }
    PetscCall(MatMPIAIJSetPreallocation(B, 0, d_nnz, 0, o_nnz));
    PetscCall(PetscFree2(d_nnz, o_nnz));
  }

  // Process diagonal block A via MatSetValues
  if (a_baij->A) {
    Mat_SeqBAIJKokkos *aseqkok = (Mat_SeqBAIJKokkos *)a_baij->A->spptr;
    if (aseqkok && aseqkok->mbs > 0 && aseqkok->nbs > 0) {
      // Sync values device->host: an in-place device op (e.g. MatDiagonalScale/MatScale in GAMG
      // prolongator smoothing) may have modified a_dual on device, leaving the host mirror stale.
      PetscCall(KokkosDualViewSyncHost(aseqkok->a_dual, PetscGetKokkosExecutionSpace()));
      auto i_hv = aseqkok->i_dual.view_host();
      auto j_hv = aseqkok->j_dual.view_host();
      auto a_hv = aseqkok->a_dual.view_host();

      for (br = 0; br < mbs_loc; br++) {
        for (k = i_hv[br]; k < i_hv[br + 1]; k++) {
          bc = j_hv[k];
          // Emit all row_bs * col_bs entries in this block one-by-one
          for (ii = 0; ii < row_bs; ii++) {
            for (jj = 0; jj < col_bs; jj++) {
              PetscInt    grow = (rstartbs + br) * row_bs + ii;
              PetscInt    gcol = (cstartbs + bc) * col_bs + jj;
              PetscScalar val  = a_hv[k * row_bs * col_bs + ii * col_bs + jj];
              PetscCall(MatSetValues(B, 1, &grow, 1, &gcol, &val, INSERT_VALUES));
            }
          }
        }
      }
    }
  }

  // Process off-diagonal block B (compressed via F1.3 MatSetUpMultiply, use garray to map back to global)
  if (a_baij->B) {
    Mat_SeqBAIJKokkos *bseqkok = (Mat_SeqBAIJKokkos *)a_baij->B->spptr;
    if (bseqkok && bseqkok->mbs > 0) {
      // Sync values device->host (see the diagonal-block note above): a prior in-place device op may
      // have left the host mirror stale.
      PetscCall(KokkosDualViewSyncHost(bseqkok->a_dual, PetscGetKokkosExecutionSpace()));
      auto i_hv = bseqkok->i_dual.view_host();
      auto j_hv = bseqkok->j_dual.view_host();
      auto a_hv = bseqkok->a_dual.view_host();

      for (br = 0; br < mbs_loc; br++) {
        for (k = i_hv[br]; k < i_hv[br + 1]; k++) {
          bc = j_hv[k]; /* LOCAL compressed block-column [0,ec) if F1.3 done, else GLOBAL */
          // If B is compressed (F1.3), map local -> global via garray
          PetscInt gcol_block = a_baij->garray ? a_baij->garray[bc] : bc;
          // Emit all row_bs * col_bs entries in this block one-by-one
          for (ii = 0; ii < row_bs; ii++) {
            for (jj = 0; jj < col_bs; jj++) {
              PetscInt    grow = (rstartbs + br) * row_bs + ii;
              PetscInt    gcol = gcol_block * col_bs + jj; /* Global column */
              PetscScalar val  = a_hv[k * row_bs * col_bs + ii * col_bs + jj];
              PetscCall(MatSetValues(B, 1, &grow, 1, &gcol, &val, INSERT_VALUES));
            }
          }
        }
      }
    }
  }

  PetscCall(MatSetOption(B, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(B, MAT_NO_OFF_PROC_ENTRIES, PETSC_FALSE));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatBuildMPIBAIJKokkosFromMPIAIJ_Private - core repackage worker shared by the MPIAIJ->MPIBAIJKOKKOS
  convert (F10.1) and the parallel block product symbolic/numeric (F10.2).

  A is a scalar MPIAIJ source. M is an already-created MATMPIBAIJKOKKOS target. Block sizes are taken
  from A's layouts (A->rmap->bs x A->cmap->bs); the caller must MatSetBlockSizes() on A first when they
  differ from 1. When valuesonly is PETSC_FALSE, M is sized + block-preallocated from A's structure
  (M must be unsized/unassembled). When valuesonly is PETSC_TRUE, M's block structure already exists
  (built by an earlier call) and only its values are refreshed via MatZeroEntries + insert.

  Uses the public MatGetRow()/MatSetValuesBlocked() path (global indices), so it is agnostic to A's
  internal storage and handles rectangular blocks. The MPIBAIJKOKKOS assembly path
  (MatAssemblyEnd_MPIBAIJKokkos + MatSetUpMultiply) builds the compressed off-diagonal B and garray.
*/
static PetscErrorCode MatBuildMPIBAIJKokkosFromMPIAIJ_Private(Mat A, Mat M, PetscBool valuesonly)
{
  PetscInt           row_bs, col_bs, m, mbs, rstart, rstartbs, cstart, cend;
  PetscInt           bi, ii, k, ncols, grow, gbrow;
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscScalar       *block;

  PetscFunctionBegin;
  PetscCheck(A->assembled, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Matrix must be assembled");

  row_bs   = A->rmap->bs > 0 ? A->rmap->bs : 1;
  col_bs   = A->cmap->bs > 0 ? A->cmap->bs : 1;
  m        = A->rmap->n;
  rstart   = A->rmap->rstart;
  cstart   = A->cmap->rstart;
  cend     = A->cmap->rend;
  rstartbs = rstart / row_bs;
  PetscCheck(m % row_bs == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local rows %" PetscInt_FMT " not divisible by row_bs %" PetscInt_FMT, m, row_bs);
  PetscCheck(A->cmap->N % col_bs == 0, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_SIZ, "Global columns %" PetscInt_FMT " not divisible by col_bs %" PetscInt_FMT, A->cmap->N, col_bs);
  PetscCheck(rstart % row_bs == 0 && cstart % col_bs == 0, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_SIZ, "Row/column ownership ranges not aligned to block sizes");
  mbs = m / row_bs;

  if (valuesonly) {
    PetscCall(MatZeroEntries(M));
  } else {
    PetscInt *d_nnz, *o_nnz;

    /* Block preallocation: per block-row, count unique diagonal/off-diagonal block-columns. */
    PetscCall(PetscCalloc2(mbs, &d_nnz, mbs, &o_nnz));
    for (bi = 0; bi < mbs; bi++) {
      std::set<PetscInt> dset, oset;
      for (ii = 0; ii < row_bs; ii++) {
        grow = rstart + bi * row_bs + ii;
        PetscCall(MatGetRow(A, grow, &ncols, &cols, NULL));
        for (k = 0; k < ncols; k++) {
          if (cols[k] >= cstart && cols[k] < cend) dset.insert(cols[k] / col_bs);
          else oset.insert(cols[k] / col_bs);
        }
        PetscCall(MatRestoreRow(A, grow, &ncols, &cols, NULL));
      }
      d_nnz[bi] = (PetscInt)dset.size();
      o_nnz[bi] = (PetscInt)oset.size();
    }

    PetscCall(MatSetSizes(M, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
    PetscCall(MatSetBlockSizes(M, row_bs, col_bs));
    PetscCall(MatMPIBAIJSetPreallocation(M, row_bs, 0, d_nnz, 0, o_nnz));
    PetscCall(PetscFree2(d_nnz, o_nnz));
  }

  /* Fill: gather each global block-row into dense row_bs x col_bs blocks and insert (global indices). */
  PetscCall(PetscMalloc1(row_bs * col_bs, &block));
  for (bi = 0; bi < mbs; bi++) {
    std::map<PetscInt, std::vector<PetscScalar>> blocks; /* global block-col -> row-major dense block */

    for (ii = 0; ii < row_bs; ii++) {
      grow = rstart + bi * row_bs + ii;
      PetscCall(MatGetRow(A, grow, &ncols, &cols, &vals));
      for (k = 0; k < ncols; k++) {
        PetscInt gbcol = cols[k] / col_bs, jj = cols[k] % col_bs;
        auto    &b = blocks[gbcol];
        if (b.empty()) b.assign(row_bs * col_bs, 0.0);
        b[ii * col_bs + jj] = vals[k];
      }
      PetscCall(MatRestoreRow(A, grow, &ncols, &cols, &vals));
    }

    gbrow = rstartbs + bi;
    for (auto &kv : blocks) {
      PetscInt gbcol = kv.first;
      PetscCall(PetscArraycpy(block, kv.second.data(), row_bs * col_bs));
      PetscCall(MatSetValuesBlocked(M, 1, &gbrow, 1, &gbcol, block, INSERT_VALUES));
    }
  }
  PetscCall(PetscFree(block));

  PetscCall(MatSetOption(M, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(M, MAT_NO_OFF_PROC_ENTRIES, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatConvert_MPIAIJ_MPIBAIJKokkos - repackage a scalar MPIAIJ matrix into rectangular-block MATMPIBAIJKOKKOS.

  Reverse of MatConvert_MPIBAIJKokkos_MPIAIJ(). Block sizes are taken from A's layouts
  (A->rmap->bs x A->cmap->bs), so the caller must MatSetBlockSizes() on A first when they differ from 1
  (e.g. the GAMG coarse operator, whose block size is the near-null-space dimension).
*/
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIBAIJKokkos(Mat A, MatType, MatReuse reuse, Mat *newmat)
{
  Mat M;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  if (reuse == MAT_REUSE_MATRIX) {
    M = *newmat;
    PetscCall(MatBuildMPIBAIJKokkosFromMPIAIJ_Private(A, M, PETSC_TRUE));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &M));
    PetscCall(MatSetType(M, MATMPIBAIJKOKKOS));
    PetscCall(MatBuildMPIBAIJKokkosFromMPIAIJ_Private(A, M, PETSC_FALSE));
  }

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &M));
  } else if (reuse == MAT_INITIAL_MATRIX) {
    *newmat = M;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATMPIBAIJKOKKOS - "mpibaijkokkos", a matrix type for distributed block matrices with Kokkos.

   A parallel matrix type using Kokkos-Kernels for portability across different device types.
   Block-sparse format with rectangular blocks (row_bs may differ from col_bs).

   Options Database Key:
.  -mat_type mpibaijkokkos - sets the matrix type to `MATMPIBAIJKOKKOS`

   Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATSEQBAIJKOKKOS`, `MATMPIBAIJ`, `MATAIJKOKKOS`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate_MPIBAIJ(A));
  PetscCall(MatConvert_MPIBAIJ_MPIBAIJKokkos(A, MATMPIBAIJKOKKOS, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}
