#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petscpkg_version.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <../src/mat/impls/baij/seq/kokkos/baijkokkosimpl.hpp>
#include <../src/mat/impls/aij/seq/aij.h>    /* Mat_SeqAIJ: read the helper graph's diag/offdiag block-CSR (i,j) */
#include <../src/mat/impls/aij/mpi/mpiaij.h> /* Mat_MPIAIJ, MatCOOStruct_MPIAIJ: helper graph for blocked COO */
#include <petscsf.h>
#include <petscsection.h> /* root/leaf PetscSection -> PetscSFCreateSectionSF for the native block P_oth gather */
#include <petscdevice.h>  /* PetscGetMemType for the blocked COO value path */
#include <algorithm>
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
static PetscErrorCode       MatMPIBAIJKokkosGetCachedAIJ(Mat, Mat *);
static PetscErrorCode       MatProductNumericBlock_MPIBAIJKokkos(Mat);
static PetscErrorCode       MatProductSetFromOptions_mpibaijkokkos_mpiaij_C(Mat);
static PetscErrorCode       MatProductSetFromOptions_mpiaij_mpibaijkokkos_C(Mat);
static PetscErrorCode       MatSetPreallocationCOO_MPIBAIJKokkos(Mat, PetscCount, PetscInt[], PetscInt[]);
static PetscErrorCode       MatSetValuesCOO_MPIBAIJKokkos(Mat, const PetscScalar[], InsertMode);

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

  /* Device-built matrices (blocked COO, native products) carry the assembled block-CSR directly in the
     sub-blocks' DualViews, and their column compression + scatter (MatSetUpMultiply) were already built at
     preallocation. A redundant MatAssemblyEnd must be a complete no-op on every rank: re-running the seq
     A/B assembly would compact the device CSR against a NULL ilen, and re-running the COLLECTIVE
     MatSetUpMultiply on only some ranks would deadlock. Keying on the device_assembled flag (set uniformly
     at build, even on empty mbs==0 ranks) avoids the ilen==NULL ambiguity, which also holds for an empty
     host-built block and so would make an empty MatSetValues rank wrongly skip the collective. */
  {
    Mat_SeqBAIJKokkos *ak = baij->A ? (Mat_SeqBAIJKokkos *)baij->A->spptr : NULL;
    if (ak && ak->device_assembled) PetscFunctionReturn(PETSC_SUCCESS);
  }

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
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
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
/*
  MatProductOperandAsAIJ_MPIBAIJKokkos - get an MPIAIJ representation of a block product operand.

  When the operand is MATMPIBAIJKOKKOS, returns the shared, value-invalidated cached AIJ
  (MatMPIBAIJKokkosGetCachedAIJ): a single convert per operand is reused across the level's AB and PtAP
  products and across symbolic+numeric, instead of re-converting on every call (the F2.2 MatConvert
  bottleneck - 57% of block GAMG setup). The returned matrix is BORROWED (*owned = PETSC_FALSE): the caller
  must NOT destroy it; it lives on the operand until its values change (cache dropped at MatAssemblyEnd) or
  the operand is destroyed. For a non-block operand (mixed-type products via the composed shims), converts
  to a fresh transient (*owned = PETSC_TRUE) that the caller destroys.
*/
static PetscErrorCode MatProductOperandAsAIJ_MPIBAIJKokkos(Mat M, Mat *aij, PetscBool *owned)
{
  PetscBool isbaijkok;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)M, MATMPIBAIJKOKKOS, &isbaijkok));
  if (isbaijkok) {
    PetscCall(MatMPIBAIJKokkosGetCachedAIJ(M, aij));
    *owned = PETSC_FALSE;
  } else {
    PetscCall(MatConvert(M, MATMPIAIJ, MAT_INITIAL_MATRIX, aij));
    *owned = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------------------------------------------------------------------------------------------
   F2.2 Option B: native parallel block PtAP/AB compute for MATMPIBAIJKOKKOS.

   The parallel triple product is assembled from native seq block kernels (no operand convert, no AIJ
   spgemm) working entirely in GLOBAL coarse block-column space; the comm-heavy off-process row gather is a
   dedicated native block BCAST PetscSF (roots = P block-rows, leaves = A's garray), so no operand is ever
   converted to MPIAIJ. The result is assembled into the block C via blocked COO (the COO SF routes
   off-process coarse rows and the on-device scatter-sum builds C's values).

   Stage 2 adds values-only numeric reuse: the first (symbolic) build caches every seq building block and
   the P_oth section SF in a MatProductCtx_MPIBAIJKokkos; each later MAT_REUSE numeric refreshes only the
   block values (zero+refill the device CSR a_dual, structure i_dual/j_dual kept) and re-broadcasts just
   the block values of the P_oth gather through the cached section SF. GAMG calls numeric many times per
   symbolic (repeated solves, -pc_gamg_reuse_interpolation), so this removes the per-numeric structural
   rebuild (host CSR malloc, device deep_copy, spgemm symbolic, transpose-perm build).
   --------------------------------------------------------------------------------------------------- */

/*
  MatProductCtx_MPIBAIJKokkos - cached structures + comm plan for the native parallel block product,
  carried across numeric calls on C->product->data for values-only MAT_REUSE recompute (Stage 2).

  Every Mat here keeps its block sparsity (i_dual/j_dual) across reuse; only a_dual is refreshed. The
  P_oth section SF (sf_blocks) + its block MPI unit (blkunit) let the reuse numeric re-broadcast only the
  block values of the off-process P row gather. Cseq is the per-process triple product reused with the same
  nonzero structure (zeroed + refilled each numeric).
*/
struct MatProductCtx_MPIBAIJKokkos {
  Mat                 A_local     = NULL; /* [A_diag | A_offdiag shifted], mbs x (mbs+ec_A) */
  Mat                 P_localrows = NULL; /* P's local block-rows, GLOBAL coarse cols, mbs x pNbs */
  Mat                 P_oth       = NULL; /* gathered off-process P rows, ec_A x pNbs (NULL if size 1) */
  Mat                 P_stack     = NULL; /* [P_localrows ; P_oth], (mbs+ec_A) x pNbs */
  Mat                 AP          = NULL; /* A_local * P_stack */
  Mat                 R           = NULL; /* P_localrows^T (PtAP only) */
  Mat                 Cseq        = NULL; /* per-process triple product in global coarse cols */
  MatColIdxKokkosView perm;               /* transpose block-perm for R (PtAP only) */
  /* Device gather maps recorded at symbolic build, replayed on MAT_REUSE to refresh A_local/P_localrows
     block values straight from the source baij->A/baij->B device a_dual (no host round-trip). Per dest
     block slot: *_src = source block index in the source sub-block, *_isoff = 0 (baij->A diag) / 1 (baij->B offdiag). */
  MatColIdxKokkosView a_local_src, a_local_isoff;    /* A_local: dest block -> source block in A's diag/offdiag */
  MatColIdxKokkosView p_loc_src, p_loc_isoff;        /* P_localrows: dest block -> source block in P's diag/offdiag */
  PetscSF             sf_blocks = NULL;              /* block-granular BCAST SF: roots = P_localrows blocks, leaves = P_oth blocks */
  MPI_Datatype        blkunit   = MPI_DATATYPE_NULL; /* MPI_Type_contiguous(fb*cb, MPIU_SCALAR): one P block per SF unit */
  PetscInt            ec_A      = 0;                 /* A off-diagonal block-cols == P_oth block-rows */
  PetscInt            rowbase   = 0;                 /* C_seq block-row -> global block-row offset (AB only) */
  /* P's object state at the last P-side (re)build. With -pc_gamg_reuse_interpolation the prolongator P is
     fixed across hot solves while only A changes, so a reuse numeric whose P state matches this skips the
     entire P-side refresh (P_localrows/P_oth Bcast/P_stack/R) - the dominant numeric-phase SF cost. */
  PetscObjectState P_state = -1;
};

/*
  MatProductCtxDestroy_MPIBAIJKokkos - free the cached native-product context (C->product->destroy hook).
*/
static PetscErrorCode MatProductCtxDestroy_MPIBAIJKokkos(PetscCtxRt ctx)
{
  MatProductCtx_MPIBAIJKokkos *p;

  PetscFunctionBegin;
  p = *reinterpret_cast<MatProductCtx_MPIBAIJKokkos **>(ctx);
  if (p) {
    PetscCall(MatDestroy(&p->A_local));
    PetscCall(MatDestroy(&p->P_localrows));
    PetscCall(MatDestroy(&p->P_oth));
    PetscCall(MatDestroy(&p->P_stack));
    PetscCall(MatDestroy(&p->AP));
    PetscCall(MatDestroy(&p->R));
    PetscCall(MatDestroy(&p->Cseq));
    PetscCall(PetscSFDestroy(&p->sf_blocks));
    if (p->blkunit != MPI_DATATYPE_NULL) PetscCallMPI(MPI_Type_free(&p->blkunit));
    PetscCallCXX(delete p);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatCreateSeqBAIJKokkosFromHostCSR - build a fully-assembled MATSEQBAIJKOKKOS from host block-CSR arrays.

  i_h (length mbs+1), j_h (length nblk, block-columns), a_h (length nblk*row_bs*col_bs, row-major per
  block). Copies the host arrays into device views, constructs the Mat_SeqBAIJKokkos, and syncs the host
  mirror so both host and device sides are valid (host readers like the transpose helper need it). The
  caller retains ownership of i_h/j_h/a_h.
*/
static PetscErrorCode MatCreateSeqBAIJKokkosFromHostCSR(PetscInt row_bs, PetscInt col_bs, PetscInt mbs, PetscInt nbs, PetscInt nblk, const PetscInt *i_h, const PetscInt *j_h, const PetscScalar *a_h, Mat *out)
{
  Mat_SeqBAIJKokkos *mk;
  PetscInt           bs2 = row_bs * col_bs, i, k;

  PetscFunctionBegin;
  MatRowMapKokkosView i_d("i_seqbaijkok", mbs + 1);
  MatColIdxKokkosView j_d("j_seqbaijkok", nblk);
  MatScalarKokkosView a_d("a_seqbaijkok", (size_t)nblk * bs2);
  {
    auto ih = Kokkos::create_mirror_view(i_d);
    auto jh = Kokkos::create_mirror_view(j_d);
    auto ah = Kokkos::create_mirror_view(a_d);
    for (i = 0; i <= mbs; i++) ih(i) = i_h[i];
    for (k = 0; k < nblk; k++) jh(k) = j_h[k];
    if (a_h)
      for (k = 0; k < nblk * bs2; k++) ah(k) = a_h[k];
    else
      for (k = 0; k < nblk * bs2; k++) ah(k) = 0.0; /* values filled later (e.g. by blocked COO) */
    PetscCallCXX(Kokkos::deep_copy(i_d, ih));
    PetscCallCXX(Kokkos::deep_copy(j_d, jh));
    PetscCallCXX(Kokkos::deep_copy(a_d, ah));
  }
  PetscCall(MatCreate(PETSC_COMM_SELF, out));
  PetscCall(MatSetSizes(*out, mbs * row_bs, nbs * col_bs, mbs * row_bs, nbs * col_bs));
  PetscCall(MatSetBlockSizes(*out, row_bs, col_bs));
  PetscCall(MatSetType(*out, MATSEQBAIJKOKKOS));
  PetscCallCXX(mk = new Mat_SeqBAIJKokkos(row_bs, col_bs, mbs, nbs, nblk, i_d, j_d, a_d));
  mk->device_assembled = PETSC_TRUE; /* device CSR is authoritative; redundant MatAssemblyEnd must no-op */
  (*out)->spptr        = (void *)mk;
  (*out)->assembled    = PETSC_TRUE;
  /* The block-column count nbs may differ from the square default; set the column layout explicitly. */
  PetscCall(PetscLayoutDestroy(&(*out)->cmap));
  PetscCall(PetscLayoutCreateFromSizes(PETSC_COMM_SELF, nbs * col_bs, nbs * col_bs, col_bs, &(*out)->cmap));
  /* Constructor leaves the a_dual host mirror uninitialized (modify_device); sync so host readers see it. */
  PetscCall(KokkosDualViewSyncHost(mk->a_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatBuildSeqBAIJKokkosFromHostCSR - build (reuse == PETSC_FALSE) or value-refresh (reuse == PETSC_TRUE) a
  MATSEQBAIJKOKKOS from host block-CSR arrays.

  On build it forwards to MatCreateSeqBAIJKokkosFromHostCSR(). On reuse it overwrites only the block values
  of the existing *out (whose block structure i_h/j_h must be unchanged from the build call) with a_h,
  marking the host mirror modified so the next device read syncs it. Used by the Stage-2 values-only
  numeric to refresh the cached seq building blocks without reallocating their CSR.
*/
static PetscErrorCode MatBuildSeqBAIJKokkosFromHostCSR(PetscBool reuse, PetscInt row_bs, PetscInt col_bs, PetscInt mbs, PetscInt nbs, PetscInt nblk, const PetscInt *i_h, const PetscInt *j_h, const PetscScalar *a_h, Mat *out)
{
  Mat_SeqBAIJKokkos *mk;
  PetscInt           bs2 = row_bs * col_bs;

  PetscFunctionBegin;
  if (!reuse) {
    PetscCall(MatCreateSeqBAIJKokkosFromHostCSR(row_bs, col_bs, mbs, nbs, nblk, i_h, j_h, a_h, out));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  mk = static_cast<Mat_SeqBAIJKokkos *>((*out)->spptr);
  PetscCheck(mk, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing Mat_SeqBAIJKokkos spptr on reuse");
  PetscCheck(mk->nblks() == nblk, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reuse block count %" PetscInt_FMT " != cached %" PetscInt_FMT, nblk, (PetscInt)mk->nblks());
  PetscCall(PetscArraycpy(mk->a_host_data(), a_h, (size_t)nblk * bs2));
  PetscCallCXX(mk->a_dual.modify_host());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSeqBAIJKokkosGetHostCSR_Private - sync a SEQBAIJKOKKOS block to host and return its host block-CSR.

  Returns the block-CSR graph (i,j index blocks) and row-major block values, plus block dimensions.
  Values are synced device->host first so the host arrays are current.
*/
static PetscErrorCode MatSeqBAIJKokkosGetHostCSR_Private(Mat M, PetscInt *row_bs, PetscInt *col_bs, PetscInt *mbs, PetscInt *nblk, PetscInt **i_h, PetscInt **j_h, PetscScalar **a_h)
{
  Mat_SeqBAIJKokkos *mk = static_cast<Mat_SeqBAIJKokkos *>(M->spptr);

  PetscFunctionBegin;
  PetscCheck(mk, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing Mat_SeqBAIJKokkos spptr");
  PetscCall(KokkosDualViewSyncHost(mk->a_dual, PetscGetKokkosExecutionSpace()));
  if (row_bs) *row_bs = mk->row_bs;
  if (col_bs) *col_bs = mk->col_bs;
  if (mbs) *mbs = mk->mbs;
  if (nblk) *nblk = mk->nblks();
  if (i_h) *i_h = mk->i_host_data();
  if (j_h) *j_h = mk->j_host_data();
  if (a_h) *a_h = mk->a_host_data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatRefreshMergedDevice_MPIBAIJKokkos - MAT_REUSE device value refresh for a merged seq block matrix
  (A_local or P_localrows) assembled from an MPIBAIJ's diagonal baij->A and off-diagonal baij->B sub-blocks.

  gather_src[pos] / gather_isoff[pos] (recorded once at symbolic build, one entry per dest block slot pos):
  the source block index in baij->A (isoff == 0) or baij->B (isoff == 1). A single device parallel_for copies
  each dest block's row_bs*col_bs scalars straight from the source sub-block's device a_dual - no GetHostCSR,
  no host round-trip. The dest device mirror is marked modified so the downstream device kernels (seq AB,
  transpose refresh) read it without a host sync.
*/
static PetscErrorCode MatRefreshMergedDevice_MPIBAIJKokkos(Mat src, Mat dest, MatColIdxKokkosView gather_src, MatColIdxKokkosView gather_isoff)
{
  Mat_MPIBAIJ       *baij  = (Mat_MPIBAIJ *)src->data;
  Mat_SeqBAIJKokkos *destk = static_cast<Mat_SeqBAIJKokkos *>(dest->spptr);
  PetscInt           bs2 = destk->row_bs * destk->col_bs, nblk = destk->nblks();
  PetscBool          hasB = (PetscBool)(baij->B && baij->garray);

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosSyncDevice(baij->A));
  if (hasB) PetscCall(MatSeqBAIJKokkosSyncDevice(baij->B));
  {
    ConstMatScalarKokkosView a_d   = static_cast<Mat_SeqBAIJKokkos *>(baij->A->spptr)->a_dual.view_device();
    ConstMatScalarKokkosView b_d   = hasB ? static_cast<Mat_SeqBAIJKokkos *>(baij->B->spptr)->a_dual.view_device() : ConstMatScalarKokkosView();
    MatScalarKokkosView      dst   = destk->a_dual.view_device();
    MatColIdxKokkosView      src_v = gather_src, off_v = gather_isoff;

    Kokkos::parallel_for(
      "MatRefreshMergedDevice_MPIBAIJKokkos", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nblk), KOKKOS_LAMBDA(const PetscInt pos) {
        const MatScalarType *sblk = (off_v(pos) ? b_d.data() : a_d.data()) + (size_t)src_v(pos) * bs2;
        MatScalarType       *dblk = dst.data() + (size_t)pos * bs2;
        for (PetscInt b = 0; b < bs2; b++) dblk[b] = sblk[b];
      });
  }
  PetscCall(MatSeqBAIJKokkosModifyDevice(dest));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatBuildAlocal_MPIBAIJKokkos - build A_local = [A_diag | A_offdiag] as one MATSEQBAIJKOKKOS.

  A_local has mbs block-rows and (mbs + ec_A) block-columns: the diagonal block A_diag keeps its local
  block-columns [0, mbs); the off-diagonal A_offdiag's compressed columns [0, ec_A) are shifted to
  [mbs, mbs + ec_A). This stacked column space lines up with P_stack's row space (local P rows then
  P_oth rows) so a single seq AB computes A*P over the local fine rows. Square diagonal: A's col_bs == row_bs.
*/
static PetscErrorCode MatBuildAlocal_MPIBAIJKokkos(Mat A, PetscBool reuse, Mat *A_local, PetscInt *ec_A_out, MatColIdxKokkosView *gather_src, MatColIdxKokkosView *gather_isoff)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)A->data;
  PetscInt     fb = A->rmap->bs, mbs = baij->mbs, ec_A = 0;
  PetscInt     dmbs, dnblk, ombs, onblk = 0, drb, dcb, orb, ocb;
  PetscInt    *di, *dj, *oi = NULL, *oj = NULL, *li, *lj;
  PetscScalar *da, *oa = NULL, *la;
  PetscInt     bs2 = fb * fb, i, k, nnz, pos;

  PetscFunctionBegin;
  if (baij->B && baij->garray) ec_A = ((Mat_SeqBAIJKokkos *)baij->B->spptr)->nbs;
  if (reuse) { /* device value refresh through the cached gather map - no host round-trip */
    PetscCall(MatRefreshMergedDevice_MPIBAIJKokkos(A, *A_local, *gather_src, *gather_isoff));
    if (ec_A_out) *ec_A_out = ec_A;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatSeqBAIJKokkosGetHostCSR_Private(baij->A, &drb, &dcb, &dmbs, &dnblk, &di, &dj, &da));
  if (baij->B && baij->garray) PetscCall(MatSeqBAIJKokkosGetHostCSR_Private(baij->B, &orb, &ocb, &ombs, &onblk, &oi, &oj, &oa));
  nnz = dnblk + onblk;
  PetscCall(PetscMalloc3(mbs + 1, &li, nnz, &lj, (size_t)nnz * bs2, &la));
  MatColIdxKokkosView gsrc(Kokkos::view_alloc("a_local_src", Kokkos::WithoutInitializing), nnz);
  MatColIdxKokkosView goff(Kokkos::view_alloc("a_local_isoff", Kokkos::WithoutInitializing), nnz);
  auto                gsrc_h = Kokkos::create_mirror_view(gsrc);
  auto                goff_h = Kokkos::create_mirror_view(goff);
  li[0]                      = 0;
  pos                        = 0;
  for (i = 0; i < mbs; i++) {
    for (k = di[i]; k < di[i + 1]; k++) {
      lj[pos] = dj[k]; /* diagonal block-columns [0, mbs) unchanged */
      for (PetscInt b = 0; b < bs2; b++) la[(size_t)pos * bs2 + b] = da[(size_t)k * bs2 + b];
      gsrc_h(pos) = k;
      goff_h(pos) = 0;
      pos++;
    }
    if (oi) {
      for (k = oi[i]; k < oi[i + 1]; k++) {
        lj[pos] = mbs + oj[k]; /* off-diagonal compressed columns shifted to [mbs, mbs+ec_A) */
        for (PetscInt b = 0; b < bs2; b++) la[(size_t)pos * bs2 + b] = oa[(size_t)k * bs2 + b];
        gsrc_h(pos) = k;
        goff_h(pos) = 1;
        pos++;
      }
    }
    li[i + 1] = pos;
  }
  PetscCall(MatBuildSeqBAIJKokkosFromHostCSR(reuse, fb, fb, mbs, mbs + ec_A, nnz, li, lj, la, A_local));
  PetscCall(PetscFree3(li, lj, la));
  PetscCallCXX(Kokkos::deep_copy(gsrc, gsrc_h)); /* own the maps (deep_copy, not alias) for device reuse */
  PetscCallCXX(Kokkos::deep_copy(goff, goff_h));
  *gather_src   = gsrc;
  *gather_isoff = goff;
  if (ec_A_out) *ec_A_out = ec_A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatBuildPlocalGlobal_MPIBAIJKokkos - build P's local block-rows with GLOBAL coarse block-columns.

  Merges P_diag (local coarse columns, shifted by P's coarse block start) and P_offdiag (compressed
  columns mapped to global via P's garray) per block-row into one MATSEQBAIJKOKKOS of size mbs x pNbs
  (pNbs = global coarse block count), columns sorted ascending. Both inputs' per-row columns are already
  sorted in global numbering, so a two-way merge suffices.
*/
static PetscErrorCode MatBuildPlocalGlobal_MPIBAIJKokkos(Mat P, PetscBool reuse, Mat *P_localrows, MatColIdxKokkosView *gather_src, MatColIdxKokkosView *gather_isoff)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)P->data;
  PetscInt     fb = P->rmap->bs, cb = P->cmap->bs, mbs = baij->mbs, pNbs = baij->Nbs;
  PetscInt     cstartbs = baij->cstartbs;
  PetscInt    *garray   = baij->garray;
  PetscInt     dmbs, dnblk, ombs, onblk = 0, drb, dcb, orb, ocb;
  PetscInt    *di, *dj, *oi = NULL, *oj = NULL, *li, *lj;
  PetscScalar *da, *oa = NULL, *la;
  PetscInt     bs2 = fb * cb, i, kd, ko, nnz, pos;

  PetscFunctionBegin;
  if (reuse) { /* device value refresh through the cached gather map - no host round-trip */
    PetscCall(MatRefreshMergedDevice_MPIBAIJKokkos(P, *P_localrows, *gather_src, *gather_isoff));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatSeqBAIJKokkosGetHostCSR_Private(baij->A, &drb, &dcb, &dmbs, &dnblk, &di, &dj, &da));
  if (baij->B && garray) PetscCall(MatSeqBAIJKokkosGetHostCSR_Private(baij->B, &orb, &ocb, &ombs, &onblk, &oi, &oj, &oa));
  else onblk = 0;
  nnz = dnblk + onblk;
  PetscCall(PetscMalloc3(mbs + 1, &li, nnz, &lj, (size_t)nnz * bs2, &la));
  MatColIdxKokkosView gsrc(Kokkos::view_alloc("p_loc_src", Kokkos::WithoutInitializing), nnz);
  MatColIdxKokkosView goff(Kokkos::view_alloc("p_loc_isoff", Kokkos::WithoutInitializing), nnz);
  auto                gsrc_h = Kokkos::create_mirror_view(gsrc);
  auto                goff_h = Kokkos::create_mirror_view(goff);
  li[0]                      = 0;
  pos                        = 0;
  for (i = 0; i < mbs; i++) {
    kd              = di[i];
    ko              = oi ? oi[i] : 0;
    PetscInt kd_end = di[i + 1], ko_end = oi ? oi[i + 1] : 0;
    while (kd < kd_end || ko < ko_end) {
      PetscInt gd = (kd < kd_end) ? dj[kd] + cstartbs : PETSC_INT_MAX;
      PetscInt go = (ko < ko_end) ? garray[oj[ko]] : PETSC_INT_MAX;
      if (gd <= go) {
        lj[pos] = gd;
        for (PetscInt b = 0; b < bs2; b++) la[(size_t)pos * bs2 + b] = da[(size_t)kd * bs2 + b];
        gsrc_h(pos) = kd;
        goff_h(pos) = 0;
        kd++;
      } else {
        lj[pos] = go;
        for (PetscInt b = 0; b < bs2; b++) la[(size_t)pos * bs2 + b] = oa[(size_t)ko * bs2 + b];
        gsrc_h(pos) = ko;
        goff_h(pos) = 1;
        ko++;
      }
      pos++;
    }
    li[i + 1] = pos;
  }
  PetscCall(MatBuildSeqBAIJKokkosFromHostCSR(reuse, fb, cb, mbs, pNbs, nnz, li, lj, la, P_localrows));
  PetscCall(PetscFree3(li, lj, la));
  PetscCallCXX(Kokkos::deep_copy(gsrc, gsrc_h)); /* own the maps (deep_copy, not alias) for device reuse */
  PetscCallCXX(Kokkos::deep_copy(goff, goff_h));
  *gather_src   = gsrc;
  *gather_isoff = goff;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatBuildPoth_MPIBAIJKokkos - gather P's off-process block-rows (those referenced by A's off-diagonal
  block-columns) as one MATSEQBAIJKOKKOS with GLOBAL coarse block-columns.

  Native block gather (no operand convert): a dedicated BCAST PetscSF whose roots are P's local block-rows
  (already merged into global coarse columns as P_localrows) and whose leaves are A's off-diagonal
  compressed block-columns (A->garray, in garray order). The variable-length per-block-row data is moved by
  an expanded section SF (PetscSFCreateSectionSF) built from root/leaf PetscSections of per-row block
  counts. The sender broadcasts each row's global coarse block-columns (already sorted ascending) and its
  row-major fb x cb block values, so the receiver lays out P_oth's block-CSR directly with no remap. The
  result has ec_A block-rows (one per A off-diagonal compressed block-column, in A's garray order) and pNbs
  block-columns. Returns *P_oth_blk == NULL and ec_A == 0 on a single rank (no off-process rows).

  reuse == PETSC_FALSE (build): builds the BCAST + section SF (cached in *sf_blocks) and the block MPI unit
  (*blkunit), broadcasts counts/columns/values on host, and creates *P_oth_blk. reuse == PETSC_TRUE:
  re-broadcasts only the block values through the cached *sf_blocks, on DEVICE (PetscSFBcastWithMemType,
  PETSC_MEMTYPE_KOKKOS) - send source P_localrows device a_dual, receive into P_oth device a_dual - so the
  reuse value refresh never round-trips through host (structure unchanged: A and P structures are stable
  across GAMG reuse).
*/
static PetscErrorCode MatBuildPoth_MPIBAIJKokkos(Mat A, Mat P, Mat P_localrows, PetscBool reuse, PetscSF *sf_blocks, MPI_Datatype *blkunit, Mat *P_oth_blk, PetscInt *ec_A_out)
{
  Mat_MPIBAIJ             *abaij = (Mat_MPIBAIJ *)A->data, *pbaij = (Mat_MPIBAIJ *)P->data;
  PetscInt                 fb = P->rmap->bs, cb = P->cmap->bs, pNbs = pbaij->Nbs, bs2 = fb * cb;
  PetscInt                 P_mbs = pbaij->mbs, ec_A = 0, P_oth_nblk = 0;
  PetscInt                *pi, *pj;
  PetscScalar             *pa;
  std::vector<PetscInt>    ai, aj;
  std::vector<PetscScalar> aa;
  MPI_Comm                 comm;
  PetscMPIInt              size;

  PetscFunctionBegin;
  if (ec_A_out) *ec_A_out = 0;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) { /* single rank: no off-process rows */
    *P_oth_blk = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (abaij->B && abaij->garray) ec_A = ((Mat_SeqBAIJKokkos *)abaij->B->spptr)->nbs;

  if (!reuse) {
    PetscLayout  rowlayout;
    PetscSF      sf_rows;
    PetscSection rootSec, leafSec;
    PetscInt    *rootCount, *leafCount, *remoteOffsets, I;

    /* P_localrows is the send source: P's local block-rows in GLOBAL coarse columns, sorted ascending. */
    PetscCall(MatSeqBAIJKokkosGetHostCSR_Private(P_localrows, NULL, NULL, NULL, NULL, &pi, &pj, &pa));

    /* BCAST SF: roots = P block-rows (P->rmap block layout), leaves = A's off-diagonal block-cols (garray). */
    PetscCall(PetscLayoutCreate(comm, &rowlayout));
    PetscCall(PetscLayoutSetLocalSize(rowlayout, P_mbs));
    PetscCall(PetscLayoutSetBlockSize(rowlayout, 1));
    PetscCall(PetscLayoutSetUp(rowlayout));
    PetscCall(PetscSFCreate(comm, &sf_rows));
    PetscCall(PetscSFSetGraphLayout(sf_rows, rowlayout, ec_A, NULL, PETSC_COPY_VALUES, abaij->garray));
    PetscCall(PetscLayoutDestroy(&rowlayout));

    /* Broadcast per-block-row counts -> P_oth i-array; build root/leaf sections of those counts. */
    PetscCall(PetscMalloc2(P_mbs, &rootCount, ec_A, &leafCount));
    for (I = 0; I < P_mbs; I++) rootCount[I] = pi[I + 1] - pi[I];
    PetscCall(PetscSFBcastBegin(sf_rows, MPIU_INT, rootCount, leafCount, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf_rows, MPIU_INT, rootCount, leafCount, MPI_REPLACE));
    PetscCall(PetscSectionCreate(comm, &rootSec));
    PetscCall(PetscSectionSetChart(rootSec, 0, P_mbs));
    for (I = 0; I < P_mbs; I++) PetscCall(PetscSectionSetDof(rootSec, I, rootCount[I]));
    PetscCall(PetscSectionSetUp(rootSec));
    PetscCall(PetscSectionCreate(comm, &leafSec));
    PetscCall(PetscSectionSetChart(leafSec, 0, ec_A));
    for (I = 0; I < ec_A; I++) PetscCall(PetscSectionSetDof(leafSec, I, leafCount[I]));
    PetscCall(PetscSectionSetUp(leafSec));

    /* Expand to a block-granular section SF (one block per dof) for the variable-length data. */
    PetscCall(PetscSFCreateRemoteOffsets(sf_rows, rootSec, leafSec, &remoteOffsets));
    PetscCall(PetscSFCreateSectionSF(sf_rows, rootSec, remoteOffsets, leafSec, sf_blocks));
    PetscCall(PetscFree(remoteOffsets));

    /* P_oth i-array from the broadcast counts; allocate j/values buffers. */
    PetscCall(PetscSectionGetStorageSize(leafSec, &P_oth_nblk));
    PetscCallCXX(ai.resize(ec_A + 1));
    ai[0] = 0;
    for (I = 0; I < ec_A; I++) ai[I + 1] = ai[I] + leafCount[I];
    PetscCallCXX(aj.resize(P_oth_nblk));
    PetscCallCXX(aa.resize((size_t)P_oth_nblk * bs2));
    PetscCall(PetscFree2(rootCount, leafCount));
    PetscCall(PetscSectionDestroy(&rootSec));
    PetscCall(PetscSectionDestroy(&leafSec));
    PetscCall(PetscSFDestroy(&sf_rows));

    /* Broadcast global coarse block-columns (already sorted ascending per row) and the block values. */
    PetscCallMPI(MPI_Type_contiguous((PetscMPIInt)bs2, MPIU_SCALAR, blkunit));
    PetscCallMPI(MPI_Type_commit(blkunit));
    PetscCall(PetscSFBcastBegin(*sf_blocks, MPIU_INT, pj, aj.data(), MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(*sf_blocks, MPIU_INT, pj, aj.data(), MPI_REPLACE));
    PetscCall(PetscSFBcastBegin(*sf_blocks, *blkunit, pa, aa.data(), MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(*sf_blocks, *blkunit, pa, aa.data(), MPI_REPLACE));
    PetscCall(MatBuildSeqBAIJKokkosFromHostCSR(PETSC_FALSE, fb, cb, ec_A, pNbs, P_oth_nblk, ai.data(), aj.data(), aa.data(), P_oth_blk));
  } else {
    Mat_SeqBAIJKokkos *plk   = static_cast<Mat_SeqBAIJKokkos *>(P_localrows->spptr);
    Mat_SeqBAIJKokkos *pothk = static_cast<Mat_SeqBAIJKokkos *>((*P_oth_blk)->spptr);
    const PetscScalar *root;
    PetscScalar       *leaf;

    /* Reuse: structure is fixed; re-broadcast only the block values on device through the cached section SF.
       Send source = P_localrows device a_dual (refreshed in MatRefreshMergedDevice on this reuse); receive
       into P_oth device a_dual. GPU-aware MPI moves the blocks device-to-device; with it off PetscSF stages
       through host automatically (still correct). No host round-trip on the building blocks. */
    ec_A = pothk->mbs;
    PetscCall(MatSeqBAIJKokkosSyncDevice(P_localrows));
    root = plk->a_dual.view_device().data();
    leaf = pothk->a_dual.view_device().data();
    PetscCall(PetscSFBcastWithMemTypeBegin(*sf_blocks, *blkunit, PETSC_MEMTYPE_KOKKOS, root, PETSC_MEMTYPE_KOKKOS, leaf, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(*sf_blocks, *blkunit, root, leaf, MPI_REPLACE));
    PetscCall(MatSeqBAIJKokkosModifyDevice(*P_oth_blk));
  }
  if (ec_A_out) *ec_A_out = ec_A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatBuildPstack_MPIBAIJKokkos - stack P_localrows (mbs rows) atop P_oth (ec_A rows) into one
  MATSEQBAIJKOKKOS with the same global coarse block-columns. The combined row space [0, mbs+ec_A)
  matches A_local's stacked block-column space. P_oth may be NULL (single rank): then P_stack is just
  a copy of P_localrows.
*/
static PetscErrorCode MatBuildPstack_MPIBAIJKokkos(Mat P_localrows, Mat P_oth, PetscBool reuse, Mat *P_stack)
{
  PetscInt     fb, cb, lmbs, lnblk, pNbs, ombs = 0, onblk = 0, rb2, cb2;
  PetscInt    *li, *lj, *oi = NULL, *oj = NULL, *si, *sj;
  PetscScalar *la, *oa = NULL, *sa;
  PetscInt     i, k, bs2, nnz;

  PetscFunctionBegin;
  if (reuse) { /* device concat: P_stack a_dual is the contiguous [P_localrows blocks ; P_oth blocks] - no host round-trip */
    Mat_SeqBAIJKokkos *plk   = static_cast<Mat_SeqBAIJKokkos *>(P_localrows->spptr);
    Mat_SeqBAIJKokkos *pstk  = static_cast<Mat_SeqBAIJKokkos *>((*P_stack)->spptr);
    Mat_SeqBAIJKokkos *pothk = P_oth ? static_cast<Mat_SeqBAIJKokkos *>(P_oth->spptr) : NULL;
    PetscInt           lblk = plk->nblks(), oblk = pothk ? pothk->nblks() : 0, sbs2 = plk->row_bs * plk->col_bs;

    PetscCall(MatSeqBAIJKokkosSyncDevice(P_localrows));
    if (P_oth) PetscCall(MatSeqBAIJKokkosSyncDevice(P_oth));
    {
      auto exec = PetscGetKokkosExecutionSpace();
      auto ps_d = pstk->a_dual.view_device();
      PetscCallCXX(Kokkos::deep_copy(exec, Kokkos::subview(ps_d, Kokkos::make_pair((size_t)0, (size_t)lblk * sbs2)), plk->a_dual.view_device()));
      if (P_oth) PetscCallCXX(Kokkos::deep_copy(exec, Kokkos::subview(ps_d, Kokkos::make_pair((size_t)lblk * sbs2, (size_t)(lblk + oblk) * sbs2)), pothk->a_dual.view_device()));
    }
    PetscCall(MatSeqBAIJKokkosModifyDevice(*P_stack));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatSeqBAIJKokkosGetHostCSR_Private(P_localrows, &fb, &cb, &lmbs, &lnblk, &li, &lj, &la));
  pNbs = ((Mat_SeqBAIJKokkos *)P_localrows->spptr)->nbs;
  if (P_oth) PetscCall(MatSeqBAIJKokkosGetHostCSR_Private(P_oth, &rb2, &cb2, &ombs, &onblk, &oi, &oj, &oa));
  bs2 = fb * cb;
  nnz = lnblk + onblk;
  PetscCall(PetscMalloc3(lmbs + ombs + 1, &si, nnz, &sj, (size_t)nnz * bs2, &sa));
  for (i = 0; i <= lmbs; i++) si[i] = li[i];
  for (i = 1; i <= ombs; i++) si[lmbs + i] = lnblk + oi[i];
  for (k = 0; k < lnblk; k++) sj[k] = lj[k];
  for (k = 0; k < onblk; k++) sj[lnblk + k] = oj[k];
  for (k = 0; k < lnblk * bs2; k++) sa[k] = la[k];
  for (k = 0; k < onblk * bs2; k++) sa[lnblk * bs2 + k] = oa[k];
  PetscCall(MatBuildSeqBAIJKokkosFromHostCSR(reuse, fb, cb, lmbs + ombs, pNbs, nnz, si, sj, sa, P_stack));
  PetscCall(PetscFree3(si, sj, sa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductAssembleCNative_MPIBAIJKokkos - assemble the parallel block C from the per-process triple
  product C_seq via blocked COO assembly (Stage C). C_seq is a SEQBAIJKOKKOS in GLOBAL coarse block-cols
  holding this rank's owned + off-process block-rows, blocks stored row-major in a_dual in CSR order.

  SYMBOLIC (reuse == PETSC_FALSE, once): walk C_seq's host CSR and emit one global block (I,J) per stored
  block, in CSR order. MatCOOUseBlockIndices(C, TRUE) + MatSetPreallocationCOO(C) then build C's
  diagonal/off-diagonal block-CSR, the off-process SF, and the jmap/perm maps - subsuming Stage 3's
  hand-rolled block scatter and the preallocation union. Off-process block-rows become remote COO entries
  routed+summed by C's SF exactly like the owned/off-process duplicate (I,J) of scalar AIJ COO.

  NUMERIC (every call): the COO value array IS C_seq's device a_dual (row-major crb*ccb blocks in the same
  CSR order as the emitted (I,J), so no permutation). MatSetValuesCOO(C, ., INSERT_VALUES) does the
  on-device pack + SF reduce + diag/offdiag block scatter-sum, subsuming the Stage 2 values-only reuse and
  the Stage 3 leaf-pack/reduce/MatSetValuesBlocked. C_seq rows are GLOBAL coarse block-rows for PtAP
  (rowbase == 0) and local fine block-rows for AB (rowbase == this rank's fine block-row base).
*/
static PetscErrorCode MatProductAssembleCNative_MPIBAIJKokkos(Mat C, MatProductCtx_MPIBAIJKokkos *ctx, Mat A, Mat P, PetscBool isPtAP, PetscInt rowbase, PetscBool reuse)
{
  Mat_SeqBAIJKokkos *cseqk = static_cast<Mat_SeqBAIJKokkos *>(ctx->Cseq->spptr);
  const PetscScalar *cval;

  PetscFunctionBegin;
  if (!reuse) {
    PetscInt     crb, ccb, cmbs, cnblk, *ci, *cj, *coo_i, *coo_j, i, p;
    PetscScalar *ca;

    /* Emit one global block (I,J) per stored C_seq block, in CSR order (== a_dual block order). */
    PetscCall(MatSeqBAIJKokkosGetHostCSR_Private(ctx->Cseq, &crb, &ccb, &cmbs, &cnblk, &ci, &cj, &ca));
    PetscCall(PetscMalloc2(cnblk, &coo_i, cnblk, &coo_j));
    for (i = 0; i < cmbs; i++)
      for (p = ci[i]; p < ci[i + 1]; p++) {
        coo_i[p] = rowbase + i; /* PtAP: rowbase 0, C_seq rows are global coarse block-rows; AB: + fine base */
        coo_j[p] = cj[p];       /* already global coarse block-cols */
      }
    /* Size + block-size C, then build its block-CSR + SF + jmap/perm via blocked COO. */
    if (isPtAP) PetscCall(MatSetSizes(C, P->cmap->n, P->cmap->n, P->cmap->N, P->cmap->N));
    else PetscCall(MatSetSizes(C, A->rmap->n, P->cmap->n, A->rmap->N, P->cmap->N));
    PetscCall(MatSetBlockSizes(C, crb, ccb));
    PetscCall(MatCOOUseBlockIndices(C, PETSC_TRUE));
    PetscCall(MatSetPreallocationCOO(C, cnblk, coo_i, coo_j));
    PetscCall(PetscFree2(coo_i, coo_j));
  }
  /* Numeric: feed C_seq's device a_dual straight into the on-device blocked COO scatter-sum. */
  PetscCall(MatSeqBAIJKokkosSyncDevice(ctx->Cseq));
  cval = cseqk->a_dual.view_device().data();
  PetscCall(MatSetValuesCOO(C, cval, INSERT_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductComputeNative_MPIBAIJKokkos - native parallel block PtAP/AB compute (F2.2 Option B).

  Computes the per-process triple product C_seq with native seq block kernels (no AIJ spgemm, no operand
  convert) in GLOBAL coarse block-column space, then assembles the parallel block C from C_seq with blocked
  COO assembly (Stage C) - the COO SF routes off-process block-rows to their owners and the on-device
  scatter-sum builds C's block values, replacing the Stage 3 hand-rolled PetscSF scatter + host
  MatSetValuesBlocked tail.

    PtAP: AP = A_local * P_stack (A*P over local fine rows), R = P_localrows^T, C_seq = R * AP
          (global coarse rows; off-process rows routed to owners by the COO SF).
    AB:   C_seq = AP = A_local * P_stack (local fine rows; all owned, no off-process COO entries).

  valuesonly == PETSC_FALSE is the BUILD path (symbolic): every structure is created from scratch and
  cached in a MatProductCtx_MPIBAIJKokkos on C->product->data, and C's block-CSR + COO SF + jmap/perm are
  built once. valuesonly == PETSC_TRUE is the REUSE path (MAT_REUSE numeric): the cached structures are
  kept and only their block values are refreshed (Stage 2) - host CSR a-arrays recomputed and pushed to
  device, the seq AB numerics rerun over the cached symbolic graphs, the P_oth gather replays only its
  a-array, and C's values are refreshed by re-feeding C_seq's a_dual through the cached COO maps. The
  dispatch (operand types) is stable across symbolic->numeric, so a reuse call always finds its build-time ctx.
*/
static PetscErrorCode MatProductComputeNative_MPIBAIJKokkos(Mat C, PetscBool valuesonly)
{
  Mat_Product                 *product = C->product;
  Mat                          A = product->A, P = product->B;
  Mat_MPIBAIJ                 *abaij = (Mat_MPIBAIJ *)A->data;
  MatProductCtx_MPIBAIJKokkos *ctx;
  MatProductCtx_SeqBAIJKokkos  pdataAP, pdataC;
  PetscBool                    reuse  = valuesonly;
  PetscBool                    isPtAP = (PetscBool)(product->type == MATPRODUCT_PtAP);
  PetscBool                    refreshP;
  PetscInt                     ec_A = 0, ec_oth = 0, rowbase;
  PetscObjectState             Pstate;
  MPI_Comm                     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)C, &comm));
  if (reuse) {
    ctx = (MatProductCtx_MPIBAIJKokkos *)product->data;
    PetscCheck(ctx, comm, PETSC_ERR_PLIB, "Values-only native product numeric without a cached context");
  } else { /* fresh build: discard any stale ctx, cache a new one on product->data */
    if (product->data) PetscCall((*product->destroy)(&product->data));
    PetscCallCXX(ctx = new MatProductCtx_MPIBAIJKokkos());
    product->data    = ctx;
    product->destroy = MatProductCtxDestroy_MPIBAIJKokkos;
  }

  /* P is fixed across hot solves under -pc_gamg_reuse_interpolation (only A changes), so skip the entire
     P-side rebuild - P_localrows refresh, the P_oth Bcast, the P_stack concat, and the R transpose refresh -
     whenever P's object state is unchanged. That SF (re-broadcasting unchanged P_oth values every numeric)
     is the dominant numeric-phase cost; if P does change, refreshP falls back to the full rebuild. */
  PetscCall(PetscObjectStateGet((PetscObject)P, &Pstate));
  refreshP = (PetscBool)(!reuse || Pstate != ctx->P_state);

  /* Native local building blocks (all SEQBAIJKOKKOS, GLOBAL coarse columns). A_local always refreshes (A
     changes every numeric); the P-side only when refreshP. */
  PetscCall(MatBuildAlocal_MPIBAIJKokkos(A, reuse, &ctx->A_local, &ec_A, &ctx->a_local_src, &ctx->a_local_isoff));
  if (refreshP) {
    PetscCall(MatBuildPlocalGlobal_MPIBAIJKokkos(P, reuse, &ctx->P_localrows, &ctx->p_loc_src, &ctx->p_loc_isoff));
    PetscCall(MatBuildPoth_MPIBAIJKokkos(A, P, ctx->P_localrows, reuse, &ctx->sf_blocks, &ctx->blkunit, &ctx->P_oth, &ec_oth));
    PetscCheck(ec_A == ec_oth, comm, PETSC_ERR_PLIB, "A off-diagonal block-columns %" PetscInt_FMT " != gathered P_oth block-rows %" PetscInt_FMT, ec_A, ec_oth);
    PetscCall(MatBuildPstack_MPIBAIJKokkos(ctx->P_localrows, ctx->P_oth, reuse, &ctx->P_stack));
    ctx->ec_A    = ec_A;
    ctx->P_state = Pstate;
  } else PetscCheck(ec_A == ctx->ec_A, comm, PETSC_ERR_PLIB, "A off-diagonal block-columns %" PetscInt_FMT " changed under cached P_oth (%" PetscInt_FMT ") on reuse", ec_A, ctx->ec_A);

  if (isPtAP) {
    /* AP = A_local * P_stack; R = P_localrows^T; C_seq = R * AP (global coarse rows) */
    if (!reuse) {
      PetscCall(MatCreate(PETSC_COMM_SELF, &ctx->AP));
      PetscCall(MatProductSymbolicAB_SeqBAIJKokkos_Helper(ctx->AP, ctx->A_local, ctx->P_stack, &pdataAP));
      PetscCall(MatTransposeWithPerm_SeqBAIJKokkos_Private(ctx->P_localrows, &ctx->R, &ctx->perm));
    }
    PetscCall(MatProductNumericAB_SeqBAIJKokkos_Helper(ctx->AP, ctx->A_local, ctx->P_stack));
    if (refreshP) PetscCall(MatRefreshTransposeValues_SeqBAIJKokkos(ctx->P_localrows, ctx->R, ctx->perm));
    if (!reuse) {
      PetscCall(MatCreate(PETSC_COMM_SELF, &ctx->Cseq));
      PetscCall(MatProductSymbolicAB_SeqBAIJKokkos_Helper(ctx->Cseq, ctx->R, ctx->AP, &pdataC));
    }
    PetscCall(MatProductNumericAB_SeqBAIJKokkos_Helper(ctx->Cseq, ctx->R, ctx->AP));
    rowbase = 0; /* C_seq block-rows are already GLOBAL coarse block-rows [0, pNbs) */
  } else {       /* MATPRODUCT_AB: C_seq = A_local * P_stack; rows are this rank's local fine block-rows */
    if (!reuse) {
      PetscCall(MatCreate(PETSC_COMM_SELF, &ctx->Cseq));
      PetscCall(MatProductSymbolicAB_SeqBAIJKokkos_Helper(ctx->Cseq, ctx->A_local, ctx->P_stack, &pdataAP));
    }
    PetscCall(MatProductNumericAB_SeqBAIJKokkos_Helper(ctx->Cseq, ctx->A_local, ctx->P_stack));
    rowbase = abaij->rstartbs; /* local fine block-row i -> global fine block-row rstartbs + i */
  }

  /* Stage C: assemble block C from C_seq via blocked COO. Symbolic (!reuse) emits global block (I,J) and
     builds C's block-CSR + COO SF + jmap/perm once; numeric (every call) feeds C_seq's a_dual through the
     cached COO maps on device (SF routes off-process rows to owners, scatter-sum builds the values). */
  PetscCall(MatProductAssembleCNative_MPIBAIJKokkos(C, ctx, A, P, isPtAP, rowbase, reuse));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductComputeBlock_MPIBAIJKokkos(Mat C, PetscBool valuesonly)
{
  Mat_Product *product = C->product;
  Mat          A = product->A, B = product->B, Aaij, Baij, Cresult;
  PetscInt     row_bs, col_bs;
  PetscBool    ownA, ownB;

  PetscFunctionBegin;
  /* Operands -> AIJ via the shared cache (no per-call re-convert). Do NOT convert product->A/B in place:
     the interface tail of MatProductSymbolic() reads the original A/B (block sizes), so they must stay
     alive. The cached AIJ is borrowed (ownA/ownB == PETSC_FALSE) and must not be destroyed below. */
  PetscCall(MatProductOperandAsAIJ_MPIBAIJKokkos(A, &Aaij, &ownA));
  PetscCall(MatProductOperandAsAIJ_MPIBAIJKokkos(B, &Baij, &ownB));
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
  if (ownA) PetscCall(MatDestroy(&Aaij)); /* cached AIJ is borrowed; only free a private transient */
  if (ownB) PetscCall(MatDestroy(&Baij));
  PetscCall(MatDestroy(&Cresult));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductBlockIsNative_MPIBAIJKokkos - true when a block PtAP/AB product takes the native seq block
  compute (F2.2 Option B): both operands MATMPIBAIJKOKKOS and the product type PtAP or AB. Otherwise the
  Option-A AIJ path handles it (mixed/non-Kokkos operands). The decision depends only on operand types and
  product type, so it is stable from symbolic through every numeric.
*/
static PetscErrorCode MatProductBlockIsNative_MPIBAIJKokkos(Mat C, PetscBool *native)
{
  Mat_Product *product = C->product;
  PetscBool    isA, isB;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)product->A, MATMPIBAIJKOKKOS, &isA));
  PetscCall(PetscObjectTypeCompare((PetscObject)product->B, MATMPIBAIJKOKKOS, &isB));
  *native = (PetscBool)(isA && isB && (product->type == MATPRODUCT_PtAP || product->type == MATPRODUCT_AB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductComputeBlockDispatch_MPIBAIJKokkos - build dispatch (full structure + values, valuesonly ==
  PETSC_FALSE): native seq block compute when applicable, else the Option-A AIJ path. Used by the symbolic
  stage; the numeric stage routes through MatProductNumericBlock_MPIBAIJKokkos so the native path can take
  the values-only MAT_REUSE branch.
*/
static PetscErrorCode MatProductComputeBlockDispatch_MPIBAIJKokkos(Mat C, PetscBool valuesonly)
{
  PetscBool native;

  PetscFunctionBegin;
  PetscCall(MatProductBlockIsNative_MPIBAIJKokkos(C, &native));
  if (native) PetscCall(MatProductComputeNative_MPIBAIJKokkos(C, valuesonly));
  else PetscCall(MatProductComputeBlock_MPIBAIJKokkos(C, valuesonly));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumericBlock_MPIBAIJKokkos(Mat C)
{
  PetscBool native;

  PetscFunctionBegin;
  PetscCall(MatProductBlockIsNative_MPIBAIJKokkos(C, &native));
  /* Native path: values-only MAT_REUSE recompute over the cached structures (Stage 2) - refresh the device
     CSR values, keep i_dual/j_dual. AIJ fallback: full rebuild (a values-only refill is impossible because
     the seq Kokkos blocks keep their CSR only in device DualViews; base a->i is NULL after assembly). */
  if (native) PetscCall(MatProductComputeNative_MPIBAIJKokkos(C, PETSC_TRUE));
  else PetscCall(MatProductComputeBlock_MPIBAIJKokkos(C, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The block symbolic already does the full build (structure AND values), so the MAT_INITIAL_MATRIX
  MatProductNumeric() call that immediately follows it (e.g. GAMG/ex56) is redundant - it would re-run the
  whole PtAP/AB a second time (the F2.2 baseline showed MatPtAP firing 8x = 4 levels x 2). Skip that first
  numeric and self-rewire to the real numeric so any later MAT_REUSE recompute (changed operand values)
  still works.
*/
static PetscErrorCode MatProductNumericBlockSkipOnce_MPIBAIJKokkos(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric = MatProductNumericBlock_MPIBAIJKokkos;
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

  PetscCall(MatProductComputeBlockDispatch_MPIBAIJKokkos(C, PETSC_FALSE));
  C->ops->productnumeric = MatProductNumericBlockSkipOnce_MPIBAIJKokkos;
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

  /* Y's values are about to change. The SAME-pattern fast path below mutates the sub-blocks without
     raising Y's PetscObjectState, so the state-keyed cache in MatMPIBAIJKokkosGetCachedAIJ() would not
     see it; drop the cache explicitly here. (MatHeaderReplace in the SUBSET/DIFFERENT branch installs a
     fresh Y with no cache.) */
  PetscCall(PetscObjectCompose((PetscObject)Y, "MatMPIBAIJKokkos_cached_aij", NULL));

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

  Collective.

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
  Builds the collapsed scalar graph natively from the block CSR rather than converting A to a value-exact
  MPIAIJ first (which would pull the expensive operand->MPIAIJ MatConvert on the GAMG aggregation hot path).
  Each block (block-row, block-col) collapses to one scalar node with weight Sum |Re(block entry)| over the
  block (or over the (index x index) sub-block when num_idx > 0), exactly reproducing the bs > 1 collapse in
  MatCreateGraph_Simple_AIJ(). The collapsed weights are assembled into a scalar (bs == 1) MPIAIJ G0 of
  block dimensions, which is then handed to MatCreateGraph() (dispatching to MatCreateGraph_Simple_AIJ): G0
  has bs == 1 so the collapse is skipped and only the common symmetrize/scale/filter tail runs. The result
  bit-matches the previous convert-then-collapse path.

.seealso: [](ch_matrices), `Mat`, `MatCreateGraph()`, `PCGAMG`, `MATMPIBAIJKOKKOS`
*/
static PetscErrorCode MatCreateGraph_MPIBAIJKokkos(Mat A, PetscBool sym, PetscBool scale, PetscReal filter, PetscInt num_idx, PetscInt index[], Mat *graph)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ *)A->data;
  Mat_SeqBAIJKokkos *Ak   = baij->A ? (Mat_SeqBAIJKokkos *)baij->A->spptr : NULL;
  Mat_SeqBAIJKokkos *Bk   = baij->B ? (Mat_SeqBAIJKokkos *)baij->B->spptr : NULL;
  PetscInt           bs = A->rmap->bs, bs2 = A->rmap->bs * A->rmap->bs, mbs = baij->mbs;
  PetscInt           rstartbs = A->rmap->rstart / A->rmap->bs, cstartbs = A->cmap->rstart / A->rmap->bs;
  PetscInt           nmax = 0;
  PetscInt          *d_nnz, *o_nnz, *AJ;
  PetscScalar       *AA;
  Mat                G0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, sym, 2);
  PetscValidLogicalCollectiveBool(A, scale, 3);
  PetscAssertPointer(graph, 7);
  PetscCheck(Ak && Bk, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "MATMPIBAIJKOKKOS blocks not built");

  /* The device a_dual is authoritative after a device build/op; reading the stale host mirror would silently
     corrupt the collapsed weights on CUDA (and so the aggregation hierarchy). Sync both sub-blocks first. */
  PetscCall(KokkosDualViewSyncHost(Ak->a_dual, PetscGetKokkosExecutionSpace()));
  PetscCall(KokkosDualViewSyncHost(Bk->a_dual, PetscGetKokkosExecutionSpace()));
  auto ai = Ak->i_dual.view_host();
  auto aj = Ak->j_dual.view_host();
  auto aa = Ak->a_dual.view_host();
  auto bi = Bk->i_dual.view_host();
  auto bj = Bk->j_dual.view_host();
  auto ba = Bk->a_dual.view_host();

  /* Scalar (bs == 1) MPIAIJ graph of block dimensions: one node per block-row, one node per block-col. */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &G0));
  PetscCall(MatSetType(G0, MATMPIAIJ));
  PetscCall(MatSetSizes(G0, mbs, mbs, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetBlockSizes(G0, 1, 1));
  PetscCall(PetscMalloc2(mbs, &d_nnz, mbs, &o_nnz));
  for (PetscInt brow = 0; brow < mbs; brow++) {
    d_nnz[brow] = ai[brow + 1] - ai[brow];
    o_nnz[brow] = bi[brow + 1] - bi[brow];
    if (d_nnz[brow] + o_nnz[brow] > nmax) nmax = d_nnz[brow] + o_nnz[brow];
  }
  PetscCall(MatMPIAIJSetPreallocation(G0, 0, d_nnz, 0, o_nnz));
  PetscCall(PetscFree2(d_nnz, o_nnz));

  /* Collapse each block to Sum |Re(entry)| over the full block, or over the (index x index) sub-block when
     num_idx > 0, summing rows (outer) then columns (inner) to bit-match MatCreateGraph_Simple_AIJ(). The
     block is stored row-major: entry (ii,jj) is at a_dual[blk*bs2 + ii*bs + jj]. */
  PetscCall(PetscMalloc2(nmax, &AA, nmax, &AJ));
  for (PetscInt brow = 0; brow < mbs; brow++) {
    PetscInt grow = rstartbs + brow, n;

    n = ai[brow + 1] - ai[brow]; /* diagonal block A: local block-cols -> global cstartbs + aj[blk] */
    for (PetscInt kk = 0; kk < n; kk++) {
      PetscInt    blk = ai[brow] + kk;
      PetscScalar val = 0.0;

      if (num_idx == 0) {
        for (PetscInt ii = 0; ii < bs; ii++)
          for (PetscInt jj = 0; jj < bs; jj++) val += PetscAbs(PetscRealPart(aa[blk * bs2 + ii * bs + jj]));
      } else {
        for (PetscInt iii = 0; iii < num_idx; iii++)
          for (PetscInt jjj = 0; jjj < num_idx; jjj++) val += PetscAbs(PetscRealPart(aa[blk * bs2 + index[iii] * bs + index[jjj]]));
      }
      AJ[kk] = cstartbs + aj[blk];
      AA[kk] = val;
    }
    PetscCall(MatSetValues(G0, 1, &grow, n, AJ, AA, ADD_VALUES));

    n = bi[brow + 1] - bi[brow]; /* off-diagonal block B: garray[bj[blk]] is already the global block-col */
    for (PetscInt kk = 0; kk < n; kk++) {
      PetscInt    blk = bi[brow] + kk;
      PetscScalar val = 0.0;

      if (num_idx == 0) {
        for (PetscInt ii = 0; ii < bs; ii++)
          for (PetscInt jj = 0; jj < bs; jj++) val += PetscAbs(PetscRealPart(ba[blk * bs2 + ii * bs + jj]));
      } else {
        for (PetscInt iii = 0; iii < num_idx; iii++)
          for (PetscInt jjj = 0; jjj < num_idx; jjj++) val += PetscAbs(PetscRealPart(ba[blk * bs2 + index[iii] * bs + index[jjj]]));
      }
      AJ[kk] = baij->garray ? baij->garray[bj[blk]] : bj[blk];
      AA[kk] = val;
    }
    PetscCall(MatSetValues(G0, 1, &grow, n, AJ, AA, ADD_VALUES));
  }
  PetscCall(PetscFree2(AA, AJ));
  PetscCall(MatAssemblyBegin(G0, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G0, MAT_FINAL_ASSEMBLY));

  /* G0 has bs == 1, so MatCreateGraph_Simple_AIJ skips the collapse and applies only its common
     symmetrize/scale/filter tail (num_idx is already consumed above, so pass 0/NULL here). */
  PetscCall(MatCreateGraph(G0, sym, scale, filter, 0, NULL, graph));
  PetscCall(MatDestroy(&G0));
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
  with it).

  F2.2 also reuses this cache for the parallel block product operands (MatProductComputeBlock), which
  collapses the per-call MatConvert that dominated block GAMG setup. That makes correct invalidation
  essential: a stale cache fed to a product would silently produce wrong coarse operators. So freshness
  is keyed on A's PetscObjectState via a composed-data tag - any value mutation that increases the state
  (MatScale/MatDiagonalScale/MatShift/MatDiagonalSet/MatZeroEntries, and (re)assembly) forces a rebuild.
  MatAXPY's fast same-pattern path mutates the sub-blocks without raising A's state, so it drops the cache
  explicitly (see MatAXPY_MPIBAIJKokkos).
*/
static PetscErrorCode MatMPIBAIJKokkosGetCachedAIJ(Mat A, Mat *aij)
{
  static PetscInt state_id = -1; /* composed-data tag (registered once); -1 = not yet registered */
  PetscBool       fresh    = PETSC_FALSE;
  PetscInt        stamp    = 0;

  PetscFunctionBegin;
  if (state_id < 0) PetscCall(PetscObjectComposedDataRegister(&state_id));
  PetscCall(PetscObjectQuery((PetscObject)A, "MatMPIBAIJKokkos_cached_aij", (PetscObject *)aij));
  if (*aij) {
    /* GetInt sets fresh == TRUE only if the stamp was set at A's current state (values unchanged) */
    PetscCall(PetscObjectComposedDataGetInt((PetscObject)A, state_id, stamp, fresh));
    if (!fresh || stamp != 1) *aij = NULL; /* values changed (or no valid stamp) since build -> rebuild below */
  }
  if (!*aij) {
    PetscCall(MatConvert(A, MATMPIAIJ, MAT_INITIAL_MATRIX, aij));
    PetscCall(PetscObjectCompose((PetscObject)A, "MatMPIBAIJKokkos_cached_aij", (PetscObject)*aij));
    PetscCall(MatDestroy(aij)); /* compose holds the reference */
    PetscCall(PetscObjectQuery((PetscObject)A, "MatMPIBAIJKokkos_cached_aij", (PetscObject *)aij));
    PetscCall(PetscObjectComposedDataSetInt((PetscObject)A, state_id, 1)); /* stamp A's current state */
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

/*
  MatNorm_MPIBAIJKokkos - compute the matrix norm directly from the block-CSR values, with no convert to
  MPIAIJ. The base MatNorm_MPIBAIJ() reads the Mat_SeqBAIJ ->a arrays, which are empty for our
  SEQBAIJKOKKOS sub-blocks (values live in the Kokkos DualViews). NORM_FROBENIUS (the only norm on the GAMG
  setup path) is summed over the device a_dual of both sub-blocks; the rare NORM_1 / NORM_INFINITY are
  delegated to the cached MPIAIJ (off the aggregation hot path, so they do not reintroduce the per-level
  operator convert). a_dual is synced to host before reading (the device mirror is authoritative after a
  device build/op; reading the stale host mirror would silently corrupt the norm on CUDA).
*/
static PetscErrorCode MatNorm_MPIBAIJKokkos(Mat A, NormType type, PetscReal *nrm)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ *)A->data;
  Mat_SeqBAIJKokkos *Ak   = baij->A ? (Mat_SeqBAIJKokkos *)baij->A->spptr : NULL;
  Mat_SeqBAIJKokkos *Bk   = baij->B ? (Mat_SeqBAIJKokkos *)baij->B->spptr : NULL;
  PetscReal          sum  = 0.0;
  Mat                aij  = NULL;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    PetscCheck(Ak && Bk, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "MATMPIBAIJKOKKOS blocks not built");
    PetscCall(KokkosDualViewSyncHost(Ak->a_dual, PetscGetKokkosExecutionSpace()));
    PetscCall(KokkosDualViewSyncHost(Bk->a_dual, PetscGetKokkosExecutionSpace()));
    {
      auto     aa  = Ak->a_dual.view_host();
      auto     ba  = Bk->a_dual.view_host();
      PetscInt nzA = Ak->nblks() * Ak->row_bs * Ak->col_bs;
      PetscInt nzB = Bk->nblks() * Bk->row_bs * Bk->col_bs;
      for (PetscInt i = 0; i < nzA; i++) sum += PetscRealPart(PetscConj(aa[i]) * aa[i]);
      for (PetscInt i = 0; i < nzB; i++) sum += PetscRealPart(PetscConj(ba[i]) * ba[i]);
    }
    PetscCallMPI(MPIU_Allreduce(&sum, nrm, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)A)));
    *nrm = PetscSqrtReal(*nrm);
  } else {
    PetscCall(MatMPIBAIJKokkosGetCachedAIJ(A, &aij));
    PetscCall(MatNorm(aij, type, nrm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatGetInfo_MPIBAIJKokkos - report nonzero counts directly from the block-CSR structure, with no convert
  to MPIAIJ. The scalar nonzero count per sub-block is (stored blocks) * row_bs * col_bs, matching the
  value-exact MPIAIJ expansion that MatConvert would have produced. Removing the cached-AIJ dependency here
  also fixes the cosmetic 0-nnz / -nan operator complexity that -pc_gamg_view reported when the base
  MatGetInfo_MPIBAIJ read the empty Mat_SeqBAIJ ->a arrays. The block structure (i,j) is immutable after
  assembly and valid on host, so no device sync is needed for the counts.
*/
static PetscErrorCode MatGetInfo_MPIBAIJKokkos(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ *)A->data;
  Mat_SeqBAIJKokkos *Ak   = baij->A ? (Mat_SeqBAIJKokkos *)baij->A->spptr : NULL;
  Mat_SeqBAIJKokkos *Bk   = baij->B ? (Mat_SeqBAIJKokkos *)baij->B->spptr : NULL;
  PetscLogDouble     isend[5], irecv[5];
  PetscLogDouble     nz = 0.0;

  PetscFunctionBegin;
  if (Ak) nz += (PetscLogDouble)Ak->nblks() * Ak->row_bs * Ak->col_bs;
  if (Bk) nz += (PetscLogDouble)Bk->nblks() * Bk->row_bs * Bk->col_bs;

  info->block_size = (PetscReal)A->rmap->bs;
  isend[0]         = nz;  /* nz_used */
  isend[1]         = nz;  /* nz_allocated (device-resident: no over-allocation) */
  isend[2]         = 0.0; /* nz_unneeded */
  isend[3]         = 0.0; /* memory */
  isend[4]         = 0.0; /* mallocs */

  if (flag == MAT_LOCAL) {
    PetscCall(PetscArraycpy(irecv, isend, 5));
  } else if (flag == MAT_GLOBAL_MAX) {
    PetscCallMPI(MPIU_Allreduce(isend, irecv, 5, MPIU_PETSCLOGDOUBLE, MPI_MAX, PetscObjectComm((PetscObject)A)));
  } else if (flag == MAT_GLOBAL_SUM) {
    PetscCallMPI(MPIU_Allreduce(isend, irecv, 5, MPIU_PETSCLOGDOUBLE, MPI_SUM, PetscObjectComm((PetscObject)A)));
  } else SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Unknown MatInfoType argument %d", (int)flag);

  info->nz_used           = irecv[0];
  info->nz_allocated      = irecv[1];
  info->nz_unneeded       = irecv[2];
  info->memory            = irecv[3];
  info->mallocs           = irecv[4];
  info->assemblies        = (PetscReal)A->num_ass;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
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

/*
  Blocked COO assembly for MATMPIBAIJKOKKOS (opt in with MatCOOUseBlockIndices()).

  coo_i/coo_j are GLOBAL BLOCK indices; the values passed to MatSetValuesCOO() are one dense
  row_bs x col_bs block per entry, row-major (same layout as MatSetValuesBlocked() and a_dual). The
  block-CSR structure, the off-process PetscSF, and the diag/offdiag jmap/perm/imap maps are obtained
  by running the proven scalar MPIAIJ COO machinery on a helper mbs x nbs scalar "graph" matrix G whose
  scalar entries ARE the block indices; the numeric scatter then widens each scalar slot to a
  row_bs*col_bs block and the SF moves a contiguous row_bs*col_bs unit per leaf. Without the opt-in this
  falls back to scalar COO (MatSetPreallocationCOO_Basic / MatSetValuesCOO_Basic).
*/
struct MatCOOStruct_MPIBAIJKokkos {
  PetscCount           n;            /* number of COO block entries passed to MatSetPreallocationCOO() */
  PetscSF              sf;           /* off-process SF (block-granular: one leaf == one block) */
  PetscCount           Annz, Bnnz;   /* block-nonzeros in diagonal A and off-diagonal B */
  PetscCount           Annz2, Bnnz2; /* unique remote block-entries belonging to A and B */
  PetscInt             bs2;          /* row_bs*col_bs: scalars per block */
  MPI_Datatype         unit;         /* MPI_Type_contiguous(bs2, MPIU_SCALAR): one block per SF leaf/root */
  PetscCountKokkosView Ajmap1, Aperm1;
  PetscCountKokkosView Bjmap1, Bperm1;
  PetscCountKokkosView Aimap2, Ajmap2, Aperm2;
  PetscCountKokkosView Bimap2, Bjmap2, Bperm2;
  PetscCountKokkosView Cperm1;
  MatScalarKokkosView  sendbuf, recvbuf; /* block-wide buffers: sendlen*bs2 / recvlen*bs2 scalars */

  /* coo_h belongs to a temporary graph matrix destroyed right after construction, so DEEP-COPY into
     owned device views (create_mirror_view_and_copy aliases the host arrays when host == device, which
     becomes a use-after-free once the graph is destroyed). */
  static PetscCountKokkosView OwnView(const PetscCount *h, PetscCount len)
  {
    PetscCountKokkosView v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "coo_mpibaijkok"), len);
    Kokkos::deep_copy(v, PetscCountKokkosViewHost(const_cast<PetscCount *>(h), len));
    return v;
  }

  MatCOOStruct_MPIBAIJKokkos(const MatCOOStruct_MPIAIJ *coo_h, PetscInt bs2_in)
  {
    n       = coo_h->n;
    sf      = coo_h->sf;
    Annz    = coo_h->Annz;
    Bnnz    = coo_h->Bnnz;
    Annz2   = coo_h->Annz2;
    Bnnz2   = coo_h->Bnnz2;
    bs2     = bs2_in;
    Ajmap1  = OwnView(coo_h->Ajmap1, coo_h->Annz + 1);
    Aperm1  = OwnView(coo_h->Aperm1, coo_h->Atot1);
    Bjmap1  = OwnView(coo_h->Bjmap1, coo_h->Bnnz + 1);
    Bperm1  = OwnView(coo_h->Bperm1, coo_h->Btot1);
    Aimap2  = OwnView(coo_h->Aimap2, coo_h->Annz2);
    Ajmap2  = OwnView(coo_h->Ajmap2, coo_h->Annz2 + 1);
    Aperm2  = OwnView(coo_h->Aperm2, coo_h->Atot2);
    Bimap2  = OwnView(coo_h->Bimap2, coo_h->Bnnz2);
    Bjmap2  = OwnView(coo_h->Bjmap2, coo_h->Bnnz2 + 1);
    Bperm2  = OwnView(coo_h->Bperm2, coo_h->Btot2);
    Cperm1  = OwnView(coo_h->Cperm1, coo_h->sendlen);
    sendbuf = MatScalarKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "coo_sendbuf"), (size_t)coo_h->sendlen * bs2);
    recvbuf = MatScalarKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "coo_recvbuf"), (size_t)coo_h->recvlen * bs2);
    PetscCallVoid(PetscObjectReference((PetscObject)sf));
    PetscCallMPIAbort(PETSC_COMM_SELF, MPI_Type_contiguous((PetscMPIInt)bs2, MPIU_SCALAR, &unit));
    PetscCallMPIAbort(PETSC_COMM_SELF, MPI_Type_commit(&unit));
  }

  ~MatCOOStruct_MPIBAIJKokkos()
  {
    PetscCallVoid(PetscSFDestroy(&sf));
    PetscCallMPIAbort(PETSC_COMM_SELF, MPI_Type_free(&unit));
  }
};

static PetscErrorCode MatCOOStructDestroy_MPIBAIJKokkos(PetscCtxRt data)
{
  PetscFunctionBegin;
  PetscCallCXX(delete *static_cast<MatCOOStruct_MPIBAIJKokkos **>(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPreallocationCOO_MPIBAIJKokkos(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  PetscBool                   blocked;
  Mat_MPIBAIJ                *baij = (Mat_MPIBAIJ *)mat->data;
  Mat                         G    = NULL;
  Mat_MPIAIJ                 *g;
  Mat_SeqAIJ                 *ga, *gb;
  PetscInt                    row_bs, col_bs, m, n, M, N, mbs, nbs, Mbs, Nbs, bs2, ganz, gbnz, k;
  PetscInt                   *bj_glob = NULL;
  const PetscInt             *garray_g;
  MPI_Comm                    comm;
  PetscContainer              container_h;
  MatCOOStruct_MPIAIJ        *coo_h;
  MatCOOStruct_MPIBAIJKokkos *coo_d;

  PetscFunctionBegin;
  PetscCall(MatCOOGetUseBlockIndices(mat, &blocked));
  if (!blocked) {
    PetscCall(MatSetPreallocationCOO_Basic(mat, coo_n, coo_i, coo_j));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));

  row_bs = mat->rmap->bs;
  col_bs = mat->cmap->bs;
  m      = mat->rmap->n;
  n      = mat->cmap->n;
  M      = mat->rmap->N;
  N      = mat->cmap->N;
  PetscCheck(row_bs >= 1 && col_bs >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "Block sizes must be >= 1; got %" PetscInt_FMT " x %" PetscInt_FMT, row_bs, col_bs);
  PetscCheck(m % row_bs == 0 && n % col_bs == 0, comm, PETSC_ERR_ARG_SIZ, "Local sizes (%" PetscInt_FMT ",%" PetscInt_FMT ") not divisible by block sizes (%" PetscInt_FMT ",%" PetscInt_FMT ")", m, n, row_bs, col_bs);
  mbs = m / row_bs;
  nbs = n / col_bs;
  Mbs = M / row_bs;
  Nbs = N / col_bs;
  bs2 = row_bs * col_bs;

  /* Run the scalar MPIAIJ COO machinery on a helper graph G of BLOCK dims; its scalar entries are the
     block indices, so G's assembled diag/offdiag CSR are C's block-CSR graphs, G's SF is C's block SF,
     and G's jmap/perm/imap are the per-block-nonzero maps. */
  PetscCall(MatCreate(comm, &G));
  PetscCall(MatSetSizes(G, mbs, nbs, Mbs, Nbs));
  PetscCall(MatSetType(G, MATMPIAIJ));
  PetscCall(MatSetPreallocationCOO(G, coo_n, coo_i, coo_j));
  g    = (Mat_MPIAIJ *)G->data;
  ga   = (Mat_SeqAIJ *)g->A->data;
  gb   = (Mat_SeqAIJ *)g->B->data;
  ganz = ga->nz;
  gbnz = gb->nz;

  /* Re-make C's column-block bookkeeping (cstartbs/cendbs/nbs/Nbs/bs2) and an empty A/B envelope; we
     replace A/B below with the block-CSR carried by G. Pass empty nnz: the real structure comes from G. */
  PetscCall(MatMPIBAIJSetPreallocation(mat, row_bs, 0, NULL, 0, NULL));
  PetscCall(MatDestroy(&baij->A));
  PetscCall(MatDestroy(&baij->B));

  /* Diagonal A: G->A's (i,j) are already local block-columns [0, nbs); values start at zero. */
  PetscCall(MatCreateSeqBAIJKokkosFromHostCSR(row_bs, col_bs, mbs, nbs, ganz, ga->i, ga->j, NULL, &baij->A));

  /* Off-diagonal B: G->B's (i,j) are compressed local block-columns into G's garray. Expand to GLOBAL
     block-columns so MatSetUpMultiply_MPIBAIJKokkos() can re-derive an identically-ordered garray/scatter;
     the recompressed block-nonzero slot order matches G->B's (both sort garray ascending), keeping the
     jmap/perm 1:1. */
  garray_g = g->garray;
  PetscCall(PetscMalloc1(gbnz, &bj_glob));
  for (k = 0; k < gbnz; k++) bj_glob[k] = garray_g[gb->j[k]];
  PetscCall(MatCreateSeqBAIJKokkosFromHostCSR(row_bs, col_bs, mbs, Nbs, gbnz, gb->i, bj_glob, NULL, &baij->B));
  PetscCall(PetscFree(bj_glob));

  PetscCall(MatSetUpMultiply_MPIBAIJKokkos(mat));
  mat->assembled     = PETSC_TRUE;
  mat->was_assembled = PETSC_TRUE;

  /* Cache the device COO struct (block jmap/perm/imap + bs2 + contiguous-block SF unit). */
  PetscCall(PetscObjectQuery((PetscObject)G, "__PETSc_MatCOOStruct_Host", (PetscObject *)&container_h));
  PetscCall(PetscContainerGetPointer(container_h, (void **)&coo_h));
  PetscCallCXX(coo_d = new MatCOOStruct_MPIBAIJKokkos(coo_h, bs2));
  PetscCall(PetscObjectContainerCompose((PetscObject)mat, "__PETSc_MatCOOStruct_Device", coo_d, MatCOOStructDestroy_MPIBAIJKokkos));

  PetscCall(MatDestroy(&G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_MPIBAIJKokkos(Mat mat, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIBAIJ                *baij = (Mat_MPIBAIJ *)mat->data;
  Mat                         A = baij->A, B = baij->B;
  Mat_SeqBAIJKokkos          *ak = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  Mat_SeqBAIJKokkos          *bk = static_cast<Mat_SeqBAIJKokkos *>(B->spptr);
  ConstMatScalarKokkosView    v1;
  PetscMemType                memtype;
  PetscContainer              container;
  MatCOOStruct_MPIBAIJKokkos *coo;
  PetscBool                   blocked;
  auto                        exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCall(MatCOOGetUseBlockIndices(mat, &blocked));
  if (!blocked) {
    PetscCall(MatSetValuesCOO_Basic(mat, v, imode));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectQuery((PetscObject)mat, "__PETSc_MatCOOStruct_Device", (PetscObject *)&container));
  PetscCall(PetscContainerGetPointer(container, (void **)&coo));

  const PetscCount n      = coo->n;
  const PetscInt   bs2    = coo->bs2;
  const PetscCount Annz   = coo->Annz;
  const PetscCount Bnnz   = coo->Bnnz;
  const PetscCount Annz2  = coo->Annz2;
  const PetscCount Bnnz2  = coo->Bnnz2;
  const auto      &vsend  = coo->sendbuf;
  const auto      &v2     = coo->recvbuf;
  const auto      &Ajmap1 = coo->Ajmap1, &Aperm1 = coo->Aperm1;
  const auto      &Bjmap1 = coo->Bjmap1, &Bperm1 = coo->Bperm1;
  const auto      &Aimap2 = coo->Aimap2, &Ajmap2 = coo->Ajmap2, &Aperm2 = coo->Aperm2;
  const auto      &Bimap2 = coo->Bimap2, &Bjmap2 = coo->Bjmap2, &Bperm2 = coo->Bperm2;
  const auto      &Cperm1 = coo->Cperm1;

  PetscCall(PetscGetMemType(v, &memtype));
  if (PetscMemTypeHost(memtype)) v1 = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), ConstMatScalarKokkosViewHost(v, (size_t)n * bs2));
  else v1 = ConstMatScalarKokkosView(v, (size_t)n * bs2);

  if (imode != INSERT_VALUES) { /* ADD needs current device values */
    PetscCall(MatSeqBAIJKokkosSyncDevice(A));
    PetscCall(MatSeqBAIJKokkosSyncDevice(B));
  }
  auto Aa = ak->a_dual.view_device();
  auto Ba = bk->a_dual.view_device();

  PetscCall(PetscLogGpuTimeBegin());
  /* Pack off-process blocks to send to their owners (one contiguous bs2 block per leaf). */
  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(exec, 0, vsend.extent(0)), KOKKOS_LAMBDA(const PetscCount idx) {
      PetscCount leaf = idx / bs2;
      PetscInt   e    = idx % bs2;
      vsend(idx)      = v1(Cperm1(leaf) * bs2 + e);
    });

  /* Reduce remote blocks to owners (block unit) overlapped with the local diag/offdiag scatter-sum. */
  PetscCall(PetscSFReduceWithMemTypeBegin(coo->sf, coo->unit, PETSC_MEMTYPE_KOKKOS, vsend.data(), PETSC_MEMTYPE_KOKKOS, v2.data(), MPI_REPLACE));
  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(exec, 0, (Annz + Bnnz) * bs2), KOKKOS_LAMBDA(PetscCount idx) {
      PetscCount  i   = idx / bs2; /* block-nonzero slot (diag then offdiag) */
      PetscInt    e   = idx % bs2; /* element within the block (row-major) */
      PetscScalar sum = 0.0;
      if (i < Annz) {
        for (PetscCount k = Ajmap1(i); k < Ajmap1(i + 1); k++) sum += v1(Aperm1(k) * bs2 + e);
        Aa(i * bs2 + e) = (imode == INSERT_VALUES ? 0.0 : Aa(i * bs2 + e)) + sum;
      } else {
        i -= Annz;
        for (PetscCount k = Bjmap1(i); k < Bjmap1(i + 1); k++) sum += v1(Bperm1(k) * bs2 + e);
        Ba(i * bs2 + e) = (imode == INSERT_VALUES ? 0.0 : Ba(i * bs2 + e)) + sum;
      }
    });
  PetscCall(PetscSFReduceEnd(coo->sf, coo->unit, vsend.data(), v2.data(), MPI_REPLACE));

  /* Add received remote blocks to A and B. */
  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(exec, 0, (Annz2 + Bnnz2) * bs2), KOKKOS_LAMBDA(PetscCount idx) {
      PetscCount i = idx / bs2;
      PetscInt   e = idx % bs2;
      if (i < Annz2) {
        for (PetscCount k = Ajmap2(i); k < Ajmap2(i + 1); k++) Aa(Aimap2(i) * bs2 + e) += v2(Aperm2(k) * bs2 + e);
      } else {
        i -= Annz2;
        for (PetscCount k = Bjmap2(i); k < Bjmap2(i + 1); k++) Ba(Bimap2(i) * bs2 + e) += v2(Bperm2(k) * bs2 + e);
      }
    });
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(MatSeqBAIJKokkosModifyDevice(A));
  PetscCall(MatSeqBAIJKokkosModifyDevice(B));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
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
  /* Blocked COO assembly (opt in per matrix with MatCOOUseBlockIndices(); scalar fallback otherwise). */
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_MPIBAIJKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetValuesCOO_C", MatSetValuesCOO_MPIBAIJKokkos));
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
