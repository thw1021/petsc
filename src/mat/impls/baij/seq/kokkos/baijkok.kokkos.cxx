#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petscpkg_version.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/sfimpl.h>
#include <petsc/private/kokkosimpl.hpp>
#include <petscsys.h>

#include <KokkosBlas.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <KokkosSparse_sptrsv.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_InverseLU_Decl.hpp>

#include <../src/mat/impls/baij/seq/kokkos/baijkokkosimpl.hpp>
#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

// Forward declarations for SeqBAIJ lifecycle
PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_SeqBAIJ(Mat);

static PetscErrorCode MatSeqBAIJKokkosApplyOptions(Mat); /* applies -mat_baijkokkos_* into A->spptr */

#if PETSC_PKG_KOKKOS_KERNELS_VERSION_GE(3, 7, 0)
  #include <KokkosSparse_Utils.hpp>
using KokkosSparse::sort_crs_matrix;
using KokkosSparse::Impl::transpose_matrix;
#else
  #include <KokkosKernels_Sorting.hpp>
using KokkosKernels::sort_crs_matrix;
using KokkosKernels::Impl::transpose_matrix;
#endif

#if PETSC_PKG_KOKKOS_KERNELS_VERSION_GE(4, 6, 0)
using KokkosSparse::spiluk_symbolic;
using KokkosSparse::spiluk_numeric;
using KokkosSparse::sptrsv_symbolic;
using KokkosSparse::sptrsv_solve;
using KokkosSparse::Experimental::SPTRSVAlgorithm;
using KokkosSparse::Experimental::SPILUKAlgorithm;
#else
using KokkosSparse::Experimental::spiluk_symbolic;
using KokkosSparse::Experimental::spiluk_numeric;
using KokkosSparse::Experimental::sptrsv_symbolic;
using KokkosSparse::Experimental::sptrsv_solve;
using KokkosSparse::Experimental::SPTRSVAlgorithm;
using KokkosSparse::Experimental::SPILUKAlgorithm;
#endif

/*
  MatSeqBAIJKokkosSyncDevice - Sync block-value data to device if needed.

  If the host has been modified after the last device sync, copy a_dual to device.
  Reset transpose_updated and hermitian_updated flags since values changed.
*/
PETSC_INTERN PetscErrorCode MatSeqBAIJKokkosSyncDevice(Mat A)
{
  Mat_SeqBAIJKokkos *baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Can't sync factorized matrix from host to device");
  PetscCheck(baijkok, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected NULL (Mat_SeqBAIJKokkos*)A->spptr");
  if (baijkok->a_dual.need_sync_device()) PetscCall(KokkosDualViewSyncDevice(baijkok->a_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSeqBAIJKokkosModifyDevice - Mark block-value data on device as modified.

  Clears host-device sync state and marks device as having the latest values.
  Resets transpose_updated and hermitian_updated flags and increments object state.
*/
PETSC_INTERN PetscErrorCode MatSeqBAIJKokkosModifyDevice(Mat A)
{
  Mat_SeqBAIJKokkos *baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Not supported for factorized matrices");
  baijkok->a_dual.clear_sync_state();
  baijkok->a_dual.modify_device();
  ((Mat_SeqBAIJ *)A->data)->idiagvalid = PETSC_FALSE; /* cached block-diagonal inverse is now stale */
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSeqBAIJKokkosSyncHost - Sync block-value data from device to host.

  Copies a_dual from device to host if the device has newer data.
*/
static PetscErrorCode MatSeqBAIJKokkosSyncHost(Mat A)
{
  Mat_SeqBAIJKokkos *baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  auto               exec    = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCheckTypeName(A, MATSEQBAIJKOKKOS);
  /* We do not expect one needs factors on host */
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Can't sync factorized matrix from device to host");
  PetscCheck(baijkok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing BAIJKOK");
  PetscCall(KokkosDualViewSyncHost(baijkok->a_dual, exec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatDestroy_SeqBAIJKokkos - Destroy a SeqBAIJKokkos matrix.

  If spptr is non-NULL, delete the Mat_SeqBAIJKokkos struct, NULL it, remove
  composed functions, then chain to MatDestroy_SeqBAIJ.
*/
static PetscErrorCode MatDestroy_SeqBAIJKokkos(Mat A)
{
  Mat_SeqBAIJKokkos *baijkok;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  if (baijkok) {
    PetscCall(PetscFree(baijkok->imax));
    PetscCall(PetscFree(baijkok->ilen));
    delete baijkok;
  }
  A->spptr = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqbaijkokkos_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqbaijkokkos_seqbaijkokkos_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
  PetscCall(MatDestroy_SeqBAIJ(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSeqBAIJKokkosSetPreallocation_SeqBAIJKokkos - Implementation of preallocation for rectangular-block Kokkos matrix.

  Not Collective

  Input Parameters:
+ A      - the matrix
. row_bs - block row size
. col_bs - block column size
. nz     - number of blocks per block-row (if nnz is NULL)
- nnz    - array of block counts per block-row (or NULL for uniform nz)

  Level: intermediate

  Notes:
  Allocates the rectangular-block CSR structure on host. Sets block sizes and
  computes the block-row map, allocating DualViews for the block graph and values.
  Values are initialized to zero on host; device allocation/sync is deferred.

.seealso: `MatSetBlockSizes()`, `MatSeqBAIJSetPreallocation()`
*/
static PetscErrorCode MatSeqBAIJKokkosSetPreallocation_SeqBAIJKokkos(Mat A, PetscInt row_bs, PetscInt col_bs, PetscInt nz, const PetscInt nnz[])
{
  Mat_SeqBAIJKokkos *baijkok;
  PetscInt           mbs, nbs, total_nz, i;
  PetscInt          *i_row_map;

  PetscFunctionBegin;
  PetscCheck(row_bs >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "row_bs must be >= 1; got %" PetscInt_FMT, row_bs);
  PetscCheck(col_bs >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "col_bs must be >= 1; got %" PetscInt_FMT, col_bs);

  PetscCall(MatSetBlockSizes(A, row_bs, col_bs));
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));

  mbs = A->rmap->n / row_bs;
  nbs = A->cmap->n / col_bs;

  PetscCheck(mbs * row_bs == A->rmap->n && nbs * col_bs == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix size (%" PetscInt_FMT " x %" PetscInt_FMT ") not divisible by block sizes (%" PetscInt_FMT " x %" PetscInt_FMT ")", A->rmap->n, A->cmap->n, row_bs, col_bs);

  /* Validate the nz/nnz preallocation parameters */
  if (nz == MAT_SKIP_ALLOCATION) nz = 0;
  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  PetscCheck(nz >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nz cannot be less than 0; got %" PetscInt_FMT, nz);
  if (nnz) {
    for (i = 0; i < mbs; i++) {
      PetscCheck(nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nnz[%" PetscInt_FMT "] = %" PetscInt_FMT " must be >= 0", i, nnz[i]);
      PetscCheck(nnz[i] <= nbs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nnz[%" PetscInt_FMT "] = %" PetscInt_FMT " exceeds number of block-columns %" PetscInt_FMT, i, nnz[i], nbs);
    }
  }

  /* Build the prefix-sum block-row map and total block count */
  PetscCall(PetscMalloc1(mbs + 1, &i_row_map));
  i_row_map[0] = 0;
  if (nnz) {
    for (i = 0; i < mbs; i++) i_row_map[i + 1] = i_row_map[i] + nnz[i];
  } else {
    nz = PetscMin(nz, nbs);
    for (i = 0; i < mbs; i++) i_row_map[i + 1] = i_row_map[i] + nz;
  }
  total_nz = i_row_map[mbs];

  // Allocate or update the Kokkos structure
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  if (baijkok) delete baijkok;

  // Create device views (uninitialized for column indices, zeroed for values)
  MatRowMapKokkosView i_d("i", mbs + 1);
  MatColIdxKokkosView j_d("j", total_nz);
  MatScalarKokkosView a_d("a", total_nz * row_bs * col_bs);

  // Create mirror views on host
  auto i_h = Kokkos::create_mirror_view(i_d);
  auto j_h = Kokkos::create_mirror_view(j_d);
  auto a_h = Kokkos::create_mirror_view(a_d);

  // Fill the host views
  for (i = 0; i <= mbs; i++) i_h(i) = i_row_map[i];
  for (i = 0; i < total_nz * row_bs * col_bs; i++) a_h(i) = 0.0;

  // Copy row map to device (column map stays uninitialized)
  Kokkos::deep_copy(i_d, i_h);

  // Create the Kokkos structure using the device-view constructor
  // (The constructor will create mirrors from device to host)
  baijkok = new Mat_SeqBAIJKokkos(row_bs, col_bs, mbs, nbs, total_nz, i_d, j_d, a_d);

  /* Allocate host-side assembly bookkeeping: imax[i] = allocated blocks per row, ilen[i] = used */
  PetscCall(PetscMalloc1(mbs, &baijkok->imax));
  PetscCall(PetscMalloc1(mbs, &baijkok->ilen));
  for (i = 0; i < mbs; i++) {
    baijkok->imax[i] = i_row_map[i + 1] - i_row_map[i]; /* allocated slots per row */
    baijkok->ilen[i] = 0;                               /* no slots used yet */
  }

  A->spptr        = baijkok;
  A->preallocated = PETSC_TRUE;

  /* The Kokkos struct is built lazily here, after MatSetFromOptions; apply the per-matrix device-kernel
     tuning options now that A->spptr exists (-mat_baijkokkos_team_size / -mat_baijkokkos_generic_kernel). */
  PetscCall(MatSeqBAIJKokkosApplyOptions(A));

  /* The rectangular block-CSR lives entirely in spptr; the base SeqBAIJ storage is
     left empty (it cannot represent rectangular column-blocking). i_row_map has been
     copied into the device/host views, so free the temporary. */
  PetscCall(PetscFree(i_row_map));

  A->was_assembled = PETSC_FALSE;
  A->assembled     = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Adapter so the generic BAIJ API MatSeqBAIJSetPreallocation(A, bs, nz, nnz) preallocates a
  MATSEQBAIJKOKKOS matrix. Composed as "MatSeqBAIJSetPreallocation_C" on this type (overriding the
  SeqBAIJ base), so callers use the standard BAIJ preallocation entry point. The row block size is
  bs; the column block size is taken from the matrix's already-set column block size when it differs
  (e.g. the rectangular GAMG prolongator, where MatSetBlockSizes(P, bs, col_bs) was called first),
  otherwise it is bs (square operator). This lets rectangular block matrices be preallocated through
  the standard API without a separate rectangular entry point.
*/
static PetscErrorCode MatSeqBAIJSetPreallocation_SeqBAIJKokkos(Mat A, PetscInt bs, PetscInt nz, const PetscInt nnz[])
{
  PetscInt col_bs = (A->rmap->bs > 0 && A->cmap->bs > 0 && A->rmap->bs != A->cmap->bs) ? A->cmap->bs : bs;

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosSetPreallocation_SeqBAIJKokkos(A, bs, col_bs, nz, nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Helper: find or insert a block at position (br, bc) in the block-CSR structure.

  Given a block-row br, binary-search for block-column bc in the existing column indices.
  If found, return the slot position and PETSC_TRUE.
  If not found and space is available, insert keeping columns sorted and return the new position.
  If not found and no space, error.
  Return false if found (to allow reuse with INSERT_VALUES vs ADD_VALUES).

  Used by both MatSetValuesBlocked_SeqBAIJKokkos and MatSetValues_SeqBAIJKokkos.
*/
static PetscErrorCode MatSeqBAIJKokkos_FindOrInsertBlock(Mat_SeqBAIJKokkos *baijkok, PetscInt br, PetscInt bc, PetscInt *block_pos, PetscBool *is_new)
{
  PetscInt      *rp, *imax, *ilen;
  MatColIdxType *j_h;
  MatScalarType *a_h, *bap;
  PetscInt      *i_h;
  PetscInt       row_bs, col_bs, rmax, nrow, low, high, t, i, j;

  PetscFunctionBegin;
  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  imax   = baijkok->imax;
  ilen   = baijkok->ilen;

  j_h = baijkok->j_dual.view_host().data();
  a_h = baijkok->a_dual.view_host().data();
  i_h = baijkok->i_dual.view_host().data();

  rp   = j_h + i_h[br];                   /* pointer to column indices for block-row br */
  bap  = a_h + i_h[br] * row_bs * col_bs; /* pointer to block values for block-row br */
  rmax = imax[br];                        /* allocated slots */
  nrow = ilen[br];                        /* used slots */
  low  = 0;
  high = nrow;

  /* Binary search for bc in the used range [low, high) */
  while (high - low > 7) {
    t = (low + high) / 2;
    if (rp[t] > bc) high = t;
    else low = t;
  }

  /* Linear search in the narrowed range */
  *is_new = PETSC_TRUE;
  for (i = low; i < high; i++) {
    if (rp[i] > bc) break;
    if (rp[i] == bc) {
      /* Found existing block */
      *block_pos = i;
      *is_new    = PETSC_FALSE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  /* Block not found; insert new block if allowed */
  PetscCheck(nrow < rmax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Block row %" PetscInt_FMT " would exceed allocated space (allocated %" PetscInt_FMT ", already have %" PetscInt_FMT " blocks)", br, rmax, nrow);

  /* Shift later columns and blocks right to make room for the new entry */
  for (j = nrow; j > i; j--) {
    rp[j] = rp[j - 1];
    PetscCall(PetscArraycpy(bap + j * row_bs * col_bs, bap + (j - 1) * row_bs * col_bs, row_bs * col_bs));
  }

  /* Zero the new block and insert column index */
  rp[i] = bc;
  PetscCall(PetscArrayzero(bap + i * row_bs * col_bs, row_bs * col_bs));

  ilen[br]++;
  *block_pos = i;
  *is_new    = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSetValues_SeqBAIJKokkos - Insert/add scalar values into a Kokkos BAIJ matrix.

  Handles scalar MatSetValues calls by finding the appropriate block and updating
  the scalar entry within it. Converts scalar row/column indices (r, c) to block
  indices (br, bc, lr, lc) and inserts/updates the entry.

  For each scalar entry (im[i], in[j]):
  - br = im[i] / row_bs, lr = im[i] % row_bs
  - bc = in[j] / col_bs, lc = in[j] % col_bs
  - Find or insert block (br, bc)
  - Update/add scalar entry at [lr * col_bs + lc] within the block

  Row-oriented input: v is m x n dense, stored row-major.
  Negative indices are skipped (continue).
  This is a HOST-SIDE assembly function; device sync happens at MatAssemblyEnd.
*/
static PetscErrorCode MatSetValues_SeqBAIJKokkos(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode is)
{
  Mat_SeqBAIJKokkos *baijkok;
  PetscInt           row_bs, col_bs, i, j, br, bc, lr, lc, block_pos;
  MatScalarType     *a_h, *block_a;
  PetscInt          *i_h;
  PetscBool          is_new;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");
  /* Only row-oriented input is supported (the default) */
  PetscCheck(static_cast<Mat_SeqBAIJ *>(A->data)->roworiented, PETSC_COMM_SELF, PETSC_ERR_SUP, "Column-oriented input (MAT_ROW_ORIENTED PETSC_FALSE) not supported");

  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;

  /* Ensure host views are ready */
  a_h = baijkok->a_dual.view_host().data();
  i_h = baijkok->i_dual.view_host().data();

  /* Process each scalar entry */
  for (i = 0; i < m; i++) {
    PetscInt r = im[i];
    if (r < 0) continue;
    PetscCheck(r < baijkok->mbs * row_bs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", r, baijkok->mbs * row_bs);

    br = r / row_bs;
    lr = r % row_bs;

    for (j = 0; j < n; j++) {
      PetscInt c = in[j];
      if (c < 0) continue;
      PetscCheck(c < baijkok->nbs * col_bs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", c, baijkok->nbs * col_bs);

      bc = c / col_bs;
      lc = c % col_bs;

      /* Find or insert block (br, bc) */
      PetscCall(MatSeqBAIJKokkos_FindOrInsertBlock(baijkok, br, bc, &block_pos, &is_new));

      /* Get pointer to the scalar entry within the block (row_bs x col_bs dense block, row-major) */
      block_a = a_h + i_h[br] * row_bs * col_bs + block_pos * row_bs * col_bs;

      /* Update the scalar entry */
      PetscScalar scalar_val = v[i * n + j];
      if (is == ADD_VALUES) {
        block_a[lr * col_bs + lc] += scalar_val;
      } else {
        block_a[lr * col_bs + lc] = scalar_val;
      }
    }
  }

  /* Host is the assembly authority; clear the device-modified flag the preallocation constructor set
     (fresh device zeros) so marking host does not trip Kokkos' both-modified DualView guard on device. */
  baijkok->a_dual.clear_sync_state();
  baijkok->a_dual.modify_host();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSetValuesBlocked_SeqBAIJKokkos - Insert/add rectangular-block values into a Kokkos BAIJ matrix.

  Inserts or adds values to a block matrix using block indices and rectangular blocks.
  For row-oriented input v (the standard), the dense (m*row_bs) x (n*col_bs) block of
  values is stored row-major in v[]:
    v[(k*row_bs + ii) * (n*col_bs) + l*col_bs + jj]
  is the (ii,jj) entry of block (k,l). This is copied into the block-CSR storage
  a[block_offset + ii*col_bs + jj] (row-major per block).

  Does NOT reallocation on shortage; PETSC_ERR_ARG_OUTOFRANGE if a row would exceed imax[row].
  Uses binary search + insertion with shift for column ordering within each row.
  This is a HOST-SIDE assembly function; device sync happens at MatAssemblyEnd.
*/
static PetscErrorCode MatSetValuesBlocked_SeqBAIJKokkos(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode is)
{
  Mat_SeqBAIJKokkos *baijkok;
  PetscInt           row, col, i, j, k, l, ii, jj, nrow, rmax, low, high, t, lastcol = -1;
  PetscInt          *rp, *imax, *ilen;
  MatColIdxType     *j_h;
  MatScalarType     *a_h, *bap;
  const PetscScalar *value;
  PetscInt           row_bs, col_bs, nbs, ldv;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");
  /* Only row-oriented input is supported (the default); column-oriented is not implemented */
  PetscCheck(static_cast<Mat_SeqBAIJ *>(A->data)->roworiented, PETSC_COMM_SELF, PETSC_ERR_SUP, "Column-oriented input (MAT_ROW_ORIENTED PETSC_FALSE) not supported");

  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  nbs    = baijkok->nbs;
  imax   = baijkok->imax;
  ilen   = baijkok->ilen;

  /* Ensure host views are ready */
  j_h      = baijkok->j_dual.view_host().data();
  a_h      = baijkok->a_dual.view_host().data();
  auto i_h = baijkok->i_dual.view_host().data();

  ldv = n * col_bs; /* leading dimension (row width) of the row-oriented dense values array v */

  for (k = 0; k < m; k++) { /* loop over block-rows being inserted */
    row = im[k];
    if (row < 0) continue;
    PetscCheck(row < baijkok->mbs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Block row index too large %" PetscInt_FMT " max %" PetscInt_FMT, row, baijkok->mbs - 1);

    rp   = j_h + i_h[row];                   /* pointer to column indices for this block-row */
    bap  = a_h + i_h[row] * row_bs * col_bs; /* pointer to block values for this block-row */
    rmax = imax[row];                        /* allocated slots */
    nrow = ilen[row];                        /* used slots */
    low  = 0;
    high = nrow;

    for (l = 0; l < n; l++) { /* loop over block-columns being inserted */
      if (in[l] < 0) continue;
      PetscCheck(in[l] < nbs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Block column index too large %" PetscInt_FMT " max %" PetscInt_FMT, in[l], nbs - 1);
      col   = in[l];
      value = v + k * row_bs * ldv + l * col_bs; /* block (k,l) base in row-oriented v; entry (ii,jj) at value[ii*ldv + jj] */

      /* Binary search for the column within the used range [low, high) */
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;

      while (high - low > 7) {
        t = (low + high) / 2;
        if (rp[t] > col) high = t;
        else low = t;
      }

      /* Linear search in the narrowed range */
      MatScalarType *block_a   = NULL;
      PetscBool      found_col = PETSC_FALSE;
      for (i = low; i < high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          /* Found block at position i; update values */
          block_a = bap + i * row_bs * col_bs;
          if (is == ADD_VALUES) {
            /* Add: copy row-major block values from v into a, adding to existing */
            for (ii = 0; ii < row_bs; ii++) {
              for (jj = 0; jj < col_bs; jj++) {
                block_a[ii * col_bs + jj] += value[ii * ldv + jj];
              }
            }
          } else {
            /* Insert/overwrite: copy row-major block values from v */
            for (ii = 0; ii < row_bs; ii++) {
              for (jj = 0; jj < col_bs; jj++) {
                block_a[ii * col_bs + jj] = value[ii * ldv + jj];
              }
            }
          }
          found_col = PETSC_TRUE;
          break;
        }
      }

      if (!found_col) {
        /* Column not found; insert new block if allowed */
        PetscCheck(nrow < rmax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Block row %" PetscInt_FMT " would exceed allocated space (allocated %" PetscInt_FMT ", already have %" PetscInt_FMT " blocks)", row, rmax, nrow);

        /* Shift later columns and blocks right to make room */
        for (j = nrow; j > i; j--) {
          rp[j] = rp[j - 1];
          PetscCall(PetscArraycpy(bap + j * row_bs * col_bs, bap + (j - 1) * row_bs * col_bs, row_bs * col_bs));
        }

        /* Insert the new column index and copy block values */
        rp[i]   = col;
        block_a = bap + i * row_bs * col_bs;
        if (is == ADD_VALUES) {
          /* For new blocks, ADD_VALUES means initialize to the input values (they're being added to zero) */
          for (ii = 0; ii < row_bs; ii++) {
            for (jj = 0; jj < col_bs; jj++) {
              block_a[ii * col_bs + jj] = value[ii * ldv + jj];
            }
          }
        } else {
          /* Insert: copy row-major block values */
          for (ii = 0; ii < row_bs; ii++) {
            for (jj = 0; jj < col_bs; jj++) {
              block_a[ii * col_bs + jj] = value[ii * ldv + jj];
            }
          }
        }

        nrow++;
        high++;
      }

      low = i;
    }

    ilen[row] = nrow;
  }

  /* Host is the assembly authority; clear the device-modified flag the preallocation constructor set
     (fresh device zeros) so marking host does not trip Kokkos' both-modified DualView guard on device. */
  baijkok->a_dual.clear_sync_state();
  baijkok->j_dual.modify_host();
  baijkok->a_dual.modify_host();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatAssemblyEnd_SeqBAIJKokkos - Finalize assembly of a rectangular-block Kokkos matrix.

  Compacts rows (removes gaps between allocated and used slots), rebuilds the final
  row map, and shrinks the column/value arrays to exact size. Ensures columns within
  each row are sorted. Marks device as needing sync and updates the scalar block-graph.
*/
static PetscErrorCode MatAssemblyEnd_SeqBAIJKokkos(Mat A, MatAssemblyType mode)
{
  Mat_SeqBAIJKokkos *baijkok;
  PetscInt           mbs, row_bs, col_bs, i, fshift = 0, nblk_final, final_start = 0;
  PetscInt          *i_h, *j_h, *imax, *ilen;
  MatScalarType     *a_h;
  PetscObjectState   nzstate;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY || (A->was_assembled && A->ass_nonzerostate == A->nonzerostate)) PetscFunctionReturn(PETSC_SUCCESS);

  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");

  mbs    = baijkok->mbs;
  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  imax   = baijkok->imax;
  ilen   = baijkok->ilen;

  i_h = baijkok->i_dual.view_host().data();
  j_h = baijkok->j_dual.view_host().data();
  a_h = baijkok->a_dual.view_host().data();

  /* Compact rows: shift used blocks left to remove gaps (imax[i] - ilen[i] unused slots per row) */
  nblk_final = 0;
  for (i = 0; i < mbs; i++) {
    if (i > 0) fshift += imax[i - 1] - ilen[i - 1];
    if (fshift > 0) {
      /* Move this row's blocks left by fshift positions */
      PetscInt old_start    = i_h[i];
      PetscInt new_start    = old_start - fshift;
      PetscInt nblks_in_row = ilen[i];

      PetscCall(PetscArraymove(j_h + new_start, j_h + old_start, nblks_in_row));
      PetscCall(PetscArraymove(a_h + new_start * row_bs * col_bs, a_h + old_start * row_bs * col_bs, nblks_in_row * row_bs * col_bs));
    }
    nblk_final += ilen[i];
  }

  /* Rebuild the row map to reflect final (compacted) layout */
  for (i = 0; i < mbs; i++) {
    i_h[i] = final_start;
    final_start += ilen[i];
  }
  i_h[mbs] = final_start;

  PetscCheck(final_start == nblk_final, PETSC_COMM_SELF, PETSC_ERR_LIB, "Block count mismatch: computed %" PetscInt_FMT " vs summed %" PetscInt_FMT, final_start, nblk_final);

  /* Allocate final-sized DualViews (exact size, no gaps) and copy compacted data */
  if (nblk_final > 0) {
    /* Allocate final views */
    MatRowMapKokkosView i_d_final("i_final", mbs + 1);
    MatColIdxKokkosView j_d_final("j_final", nblk_final);
    MatScalarKokkosView a_d_final("a_final", nblk_final * row_bs * col_bs);

    /* Create host mirrors and copy compacted data */
    auto i_h_final = Kokkos::create_mirror_view(i_d_final);
    auto j_h_final = Kokkos::create_mirror_view(j_d_final);
    auto a_h_final = Kokkos::create_mirror_view(a_d_final);

    PetscCall(PetscArraycpy(i_h_final.data(), i_h, mbs + 1));
    PetscCall(PetscArraycpy(j_h_final.data(), j_h, nblk_final));
    PetscCall(PetscArraycpy(a_h_final.data(), a_h, nblk_final * row_bs * col_bs));

    /* Copy to device and replace DualViews */
    Kokkos::deep_copy(i_d_final, i_h_final);
    Kokkos::deep_copy(j_d_final, j_h_final);
    Kokkos::deep_copy(a_d_final, a_h_final);

    baijkok->i_dual = MatRowMapKokkosDualView(i_d_final, i_h_final);
    baijkok->j_dual = MatColIdxKokkosDualView(j_d_final, j_h_final);
    baijkok->a_dual = MatScalarKokkosDualView(a_d_final, a_h_final);
  } else {
    /* Empty matrix: allocate minimal views */
    MatRowMapKokkosView i_d_final("i_final", mbs + 1);
    MatColIdxKokkosView j_d_final("j_final", 0);
    MatScalarKokkosView a_d_final("a_final", 0);

    auto i_h_final = Kokkos::create_mirror_view(i_d_final);
    auto j_h_final = Kokkos::create_mirror_view(j_d_final);
    auto a_h_final = Kokkos::create_mirror_view(a_d_final);

    for (i = 0; i <= mbs; i++) i_h_final(i) = 0;
    Kokkos::deep_copy(i_d_final, i_h_final);

    baijkok->i_dual = MatRowMapKokkosDualView(i_d_final, i_h_final);
    baijkok->j_dual = MatColIdxKokkosDualView(j_d_final, j_h_final);
    baijkok->a_dual = MatScalarKokkosDualView(a_d_final, a_h_final);
  }

  /* Reset imax and ilen to reflect compacted layout */
  for (i = 0; i < mbs; i++) {
    imax[i] = ilen[i]; /* After assembly, allocated == used */
  }

  /* Rebuild the scalar block-graph CSR for symbolic operations. Only the (i,j) graph is
     used (by spgemm_symbolic in Phase B); its values array has length nblk (one scalar per
     block), NOT the rectangular nblk*row_bs*col_bs, so use a dummy values view of the right size. */
  {
    auto                i_d_final = baijkok->i_dual.view_device();
    auto                j_d_final = baijkok->j_dual.view_device();
    MatScalarKokkosView graph_vals("csrmat_graph_vals", nblk_final);

    baijkok->csrmat_graph = KokkosCsrMatrix("csrmat_graph", baijkok->nbs, graph_vals, KokkosCsrGraph(j_d_final, i_d_final));
  }

  A->assembled                         = PETSC_TRUE;
  A->was_assembled                     = PETSC_TRUE;
  ((Mat_SeqBAIJ *)A->data)->idiagvalid = PETSC_FALSE; /* cached block-diagonal inverse is now stale */
  nzstate                              = A->nonzerostate;
  A->ass_nonzerostate                  = nzstate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatConvert_SeqBAIJKokkos_SeqAIJ - Convert a SeqBAIJKokkos rectangular-block matrix to scalar SeqAIJ.

  Expands each rectangular block (row_bsxcol_bs) into scalar entries at the corresponding
  scalar row/column positions. Keeps all structurally-present blocks (even if numerically zero).
  Handles MAT_INITIAL_MATRIX (build new), MAT_REUSE_MATRIX (copy into existing),
  MAT_INPLACE_MATRIX (replace A in-place).
*/
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJKokkos_SeqAIJ(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat                  B;
  Mat_SeqBAIJKokkos   *baijkok;
  PetscInt             row_bs, col_bs, mbs, nbs, i, ii, jj, bid, bid_start, bid_end, nblks_in_row;
  PetscInt            *aij_i, *aij_j;
  MatScalarType       *block_a;
  const MatScalarType *baij_a_h;
  PetscInt             scalar_m, scalar_n;

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosSyncHost(A)); /* Ensure host has current values */
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not a valid SeqBAIJKokkos");

  row_bs   = baijkok->row_bs;
  col_bs   = baijkok->col_bs;
  mbs      = baijkok->mbs;
  nbs      = baijkok->nbs;
  scalar_m = mbs * row_bs;
  scalar_n = nbs * col_bs;

  /* Get the host block-CSR structure */
  aij_i    = baijkok->i_dual.view_host().data();
  aij_j    = baijkok->j_dual.view_host().data();
  baij_a_h = baijkok->a_dual.view_host().data();

  if (reuse == MAT_INITIAL_MATRIX) {
    /* Allocate scalar rowlengths (nnz per scalar row) */
    PetscInt *rowlengths;
    PetscCall(PetscMalloc1(scalar_m, &rowlengths));
    for (i = 0; i < mbs; i++) {
      nblks_in_row = aij_i[i + 1] - aij_i[i];
      for (ii = 0; ii < row_bs; ii++) {
        rowlengths[i * row_bs + ii] = nblks_in_row * col_bs;
      }
    }

    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetType(B, MATSEQAIJ));
    PetscCall(MatSetSizes(B, scalar_m, scalar_n, scalar_m, scalar_n));
    PetscCall(MatSetBlockSizes(B, A->rmap->bs, A->cmap->bs));
    PetscCall(MatSeqAIJSetPreallocation(B, 0, rowlengths));
    PetscCall(PetscFree(rowlengths));
  } else {
    B = *newmat;
  }

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    /* Find the maximum number of blocks in any block-row */
    PetscInt max_blocks_per_row = 0;
    for (i = 0; i < mbs; i++) {
      PetscInt nblks     = aij_i[i + 1] - aij_i[i];
      max_blocks_per_row = PetscMax(max_blocks_per_row, nblks);
    }

    /* Allocate enough space for the maximum number of scalar entries per row */
    PetscInt     max_entries = max_blocks_per_row * col_bs;
    PetscInt    *cols;
    PetscScalar *vals;
    PetscCall(PetscMalloc2(max_entries, &cols, max_entries, &vals));

    for (i = 0; i < mbs; i++) {
      bid_start = aij_i[i];
      bid_end   = aij_i[i + 1];

      for (ii = 0; ii < row_bs; ii++) {
        PetscInt scalar_row = i * row_bs + ii;
        PetscInt col_count  = 0;

        for (bid = bid_start; bid < bid_end; bid++) {
          PetscInt j = aij_j[bid];
          block_a    = (MatScalarType *)baij_a_h + bid * row_bs * col_bs + ii * col_bs;

          for (jj = 0; jj < col_bs; jj++) {
            cols[col_count] = j * col_bs + jj;
            vals[col_count] = block_a[jj];
            col_count++;
          }
        }

        PetscCall(MatSetValues(B, 1, &scalar_row, col_count, cols, vals, INSERT_VALUES));
      }
    }

    PetscCall(PetscFree2(cols, vals));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }

  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A, &B));
  else *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatConvert_SeqAIJ_SeqBAIJKokkos - Convert a scalar SeqAIJ matrix to SeqBAIJKokkos rectangular-block.

  Packs scalar entries into rectangular blocks (row_bsxcol_bs) based on block sizes set on A.
  Block sizes must be set via MatSetBlockSizes() before calling this function.
  Requires A->rmap->bs and A->cmap->bs to be >= 1 and to divide the matrix dimensions.
*/
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat                  B;
  Mat_SeqAIJ          *aseq;
  PetscInt             row_bs, col_bs, m, n, mbs, nbs;
  PetscInt             i, bi, bj, ii, jj, k;
  const PetscInt      *ai, *aj;
  const MatScalarType *aa;
  PetscInt            *nnz_per_brow;
  PetscScalar         *block_dense;
  PetscBool            missing;
  PetscHSetIJ          ht_blocks;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  aseq   = static_cast<Mat_SeqAIJ *>(A->data);
  m      = A->rmap->n;
  n      = A->cmap->n;
  row_bs = A->rmap->bs;
  col_bs = A->cmap->bs;

  /* Validate block sizes */
  PetscCheck(row_bs >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "row_bs must be >= 1; got %" PetscInt_FMT, row_bs);
  PetscCheck(col_bs >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "col_bs must be >= 1; got %" PetscInt_FMT, col_bs);
  PetscCheck(m % row_bs == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of rows %" PetscInt_FMT " not divisible by row_bs %" PetscInt_FMT, m, row_bs);
  PetscCheck(n % col_bs == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of columns %" PetscInt_FMT " not divisible by col_bs %" PetscInt_FMT, n, col_bs);

  mbs = m / row_bs;
  nbs = n / col_bs;
  ai  = aseq->i;
  aj  = aseq->j;
  aa  = aseq->a;

  if (reuse == MAT_INITIAL_MATRIX) {
    /* Determine block-sparsity pattern: for each block-row, count blocks */
    PetscCall(PetscHSetIJCreate(&ht_blocks));
    PetscCall(PetscCalloc1(mbs, &nnz_per_brow));

    for (i = 0; i < m; i++) {
      bi = i / row_bs;
      for (k = ai[i]; k < ai[i + 1]; k++) {
        bj = aj[k] / col_bs;
        PetscHashIJKey key;
        key.i = bi;
        key.j = bj;
        PetscCall(PetscHSetIJQueryAdd(ht_blocks, key, &missing));
        if (missing) nnz_per_brow[bi]++;
      }
    }
    PetscCall(PetscHSetIJDestroy(&ht_blocks));

    /* Create and preallocate the block matrix */
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetSizes(B, m, n, m, n));
    PetscCall(MatSetBlockSizes(B, row_bs, col_bs));
    PetscCall(MatSetType(B, MATSEQBAIJKOKKOS));
    PetscCall(MatSeqBAIJKokkosSetPreallocation_SeqBAIJKokkos(B, row_bs, col_bs, 0, nnz_per_brow));
    PetscCall(PetscFree(nnz_per_brow));
  } else {
    B = *newmat;
  }

  /* Allocate temporary dense block storage */
  PetscCall(PetscMalloc1(row_bs * col_bs, &block_dense));

  /* Fill the block matrix using MatSetValuesBlocked - the proper way */
  for (bi = 0; bi < mbs; bi++) {
    /* Collect block-columns for this block-row */
    PetscCall(PetscHSetIJCreate(&ht_blocks));

    for (i = bi * row_bs; i < (bi + 1) * row_bs; i++) {
      for (k = ai[i]; k < ai[i + 1]; k++) {
        bj = aj[k] / col_bs;
        PetscHashIJKey key;
        key.i = 0; /* not used */
        key.j = bj;
        PetscCall(PetscHSetIJQueryAdd(ht_blocks, key, &missing));
      }
    }

    /* Get the size of the set */
    PetscInt num_blocks;
    PetscCall(PetscHSetIJGetSize(ht_blocks, &num_blocks));

    /* Convert set to sorted array of block-columns */
    PetscInt *block_cols;
    PetscCall(PetscMalloc1(num_blocks, &block_cols));
    PetscInt bidx = 0;
    for (bj = 0; bj < nbs; bj++) {
      PetscHashIJKey key;
      key.i = 0;
      key.j = bj;
      PetscBool found;
      PetscCall(PetscHSetIJHas(ht_blocks, key, &found));
      if (found) block_cols[bidx++] = bj;
    }
    PetscCall(PetscHSetIJDestroy(&ht_blocks));

    /* For each block in this block-row, fill the dense block and insert */
    for (bidx = 0; bidx < num_blocks; bidx++) {
      bj = block_cols[bidx];

      /* Zero out the temporary block */
      PetscCall(PetscArrayzero(block_dense, row_bs * col_bs));

      /* Fill block from AIJ values */
      for (i = bi * row_bs; i < (bi + 1) * row_bs; i++) {
        ii = i - bi * row_bs;
        for (k = ai[i]; k < ai[i + 1]; k++) {
          if (aj[k] / col_bs == bj) {
            jj                            = aj[k] % col_bs;
            block_dense[ii * col_bs + jj] = aa[k];
          }
        }
      }

      /* Insert block using proper API */
      PetscCall(MatSetValuesBlocked(B, 1, &bi, 1, &bj, block_dense, INSERT_VALUES));
    }

    PetscCall(PetscFree(block_cols));
  }

  PetscCall(PetscFree(block_dense));

  if (reuse == MAT_INITIAL_MATRIX) {
    /* Trigger assembly to compact and move to device */
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  } else if (reuse == MAT_REUSE_MATRIX) {
    /* Trigger assembly on existing matrix */
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }

  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A, &B));
  else *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSeqBAIJKokkosApplyOptions - read the per-matrix device-kernel tuning options into the matrix's
  Mat_SeqBAIJKokkos struct (A->spptr). No-op if the struct does not exist yet (the Kokkos storage is
  built lazily at preallocation, after MatSetFromOptions). Reads the options with the matrix's own prefix
  so the values are per object, not a process-wide cache. Called both from MatSetFromOptions_SeqBAIJKokkos
  (handles the convert / programmatic case where the struct already exists) and at the end of
  MatSeqBAIJSetPreallocation_SeqBAIJKokkos (the common path, where the struct was just created).
*/
static PetscErrorCode MatSeqBAIJKokkosApplyOptions(Mat A)
{
  Mat_SeqBAIJKokkos *baijkok   = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  constexpr bool     on_device = !Kokkos::SpaceAccessibility<DefaultExecutionSpace, Kokkos::HostSpace>::accessible;
  const char        *prefix    = ((PetscObject)A)->prefix;
  PetscInt           ts        = 0;
  PetscBool          flg = PETSC_FALSE, set = PETSC_FALSE;

  PetscFunctionBegin;
  if (!baijkok) PetscFunctionReturn(PETSC_SUCCESS);
  /* Team size override is meaningful only on a device backend (host TeamPolicy must use 1). */
  if (on_device) {
    PetscCall(PetscOptionsGetInt(NULL, prefix, "-mat_baijkokkos_team_size", &ts, &set));
    if (set && ts > 0) baijkok->team_size = ts;
  }
  PetscCall(PetscOptionsGetBool(NULL, prefix, "-mat_baijkokkos_generic_kernel", &flg, &set));
  if (set) baijkok->use_generic = flg;
  PetscCall(PetscOptionsGetBool(NULL, prefix, "-mat_baijkokkos_spmv_noatomic", &flg, &set));
  if (set) baijkok->use_noatomic_spmv = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatSetFromOptions_SeqBAIJKokkos - register the device-kernel tuning options (for -help and to mark them
  used) and apply them. The Mat_SeqBAIJKokkos struct is built lazily at preallocation, so at the usual
  call site (before MatXxxSetPreallocation) A->spptr is still NULL and the values are applied later by
  MatSeqBAIJKokkosApplyOptions() from the preallocation path; this function applies them immediately when
  the struct already exists (convert / programmatic MatSetFromOptions after assembly).

  -mat_baijkokkos_team_size <n>   : device TeamPolicy team size (default BAIJKokkosTeamSizeDefault()).
  -mat_baijkokkos_generic_kernel  : force the runtime-sized generic kernel over the shape specialization.
  -mat_baijkokkos_spmv_noatomic   : opt in to the experimental atomic-free reduction MatMult (slower on the A100).
*/
static PetscErrorCode MatSetFromOptions_SeqBAIJKokkos(Mat A, PetscOptionItems PetscOptionsObject)
{
  Mat_SeqBAIJKokkos *baijkok       = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscInt           team_size     = baijkok ? baijkok->team_size : BAIJKokkosTeamSizeDefault();
  PetscBool          generic       = baijkok ? baijkok->use_generic : PETSC_FALSE;
  PetscBool          spmv_noatomic = baijkok ? baijkok->use_noatomic_spmv : PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SeqBAIJKokkos options");
  PetscCall(PetscOptionsInt("-mat_baijkokkos_team_size", "Device kernel TeamPolicy team size (block GEMV/GEMM)", "MatSetFromOptions", team_size, &team_size, NULL));
  PetscCall(PetscOptionsBool("-mat_baijkokkos_generic_kernel", "Force the runtime-sized generic block kernel over the compile-time shape specialization", "MatSetFromOptions", generic, &generic, NULL));
  PetscCall(PetscOptionsBool("-mat_baijkokkos_spmv_noatomic", "Opt in to the experimental atomic-free team-reduction block MatMult (slower than the default atomic kernel on the A100)", "MatSetFromOptions", spmv_noatomic, &spmv_noatomic, NULL));
  PetscOptionsHeadEnd();
  PetscCall(MatSeqBAIJKokkosApplyOptions(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  BlockGemvAccum - compile-time-sized dense block GEMV accumulate (shared by the MatMult family).

  TRANS == false: y(RBS) += A(RBS x CBS) * x(CBS)    (MatMult / MatMultAdd)
  TRANS == true:  y(CBS) += A(RBS x CBS)^T * x(RBS)  (MatMultTranspose / MatMultTransposeAdd)

  Blocks are row-major: element (ii,jj) is at aval[ii*CBS+jj]. The source vector segment is staged
  into a thread-local register array so the contraction reads it stride-1, and the A-block is read
  stride-1 along its fast (column) axis in both orientations; this mirrors the B-block staging in
  BlockGemmAccum() (F8c). Output uses Kokkos::atomic_add because team threads handling different
  A-blocks of a block-row (and, for the transpose, different block-rows) target the same output
  segment (F8b). The unrolled compile-time loops keep the staged data in registers.
*/
template <int RBS, int CBS, bool TRANS>
KOKKOS_INLINE_FUNCTION static void BlockGemvAccum(const MatScalarType *aval, const MatScalarType *xval, MatScalarType *yval)
{
  if constexpr (!TRANS) {
    MatScalarType x_reg[CBS];

#pragma unroll
    for (int jj = 0; jj < CBS; jj++) x_reg[jj] = xval[jj];

#pragma unroll
    for (int ii = 0; ii < RBS; ii++) {
      MatScalarType sum = 0.0;
#pragma unroll
      for (int jj = 0; jj < CBS; jj++) sum += aval[ii * CBS + jj] * x_reg[jj];
      Kokkos::atomic_add(&yval[ii], sum);
    }
  } else {
    MatScalarType y_acc[CBS];

#pragma unroll
    for (int jj = 0; jj < CBS; jj++) y_acc[jj] = 0.0;

#pragma unroll
    for (int ii = 0; ii < RBS; ii++) {
      MatScalarType xi = xval[ii];
#pragma unroll
      for (int jj = 0; jj < CBS; jj++) y_acc[jj] += aval[ii * CBS + jj] * xi;
    }

#pragma unroll
    for (int jj = 0; jj < CBS; jj++) Kokkos::atomic_add(&yval[jj], y_acc[jj]);
  }
}

/*
  BlockGemvAccumGeneric - runtime-sized fallback of BlockGemvAccum() for shapes lacking a specialization.
*/
KOKKOS_INLINE_FUNCTION static void BlockGemvAccumGeneric(const MatScalarType *aval, const MatScalarType *xval, MatScalarType *yval, PetscInt rbs, PetscInt cbs, bool trans)
{
  if (!trans) {
    for (PetscInt ii = 0; ii < rbs; ii++) {
      MatScalarType sum = 0.0;
      for (PetscInt jj = 0; jj < cbs; jj++) sum += aval[ii * cbs + jj] * xval[jj];
      Kokkos::atomic_add(&yval[ii], sum);
    }
  } else {
    for (PetscInt ii = 0; ii < rbs; ii++) {
      MatScalarType xi = xval[ii];
      for (PetscInt jj = 0; jj < cbs; jj++) Kokkos::atomic_add(&yval[jj], aval[ii * cbs + jj] * xi);
    }
  }
}

/*
  BlockRowAccum - compile-time-sized array reducer for the atomic-free non-transpose SpMV. A team
  cooperatively reduces its partial block-row contributions into RBS output entries via Kokkos::Sum,
  so each y entry is written once (no global atomics). Mirrors landau_inner_red::TensorValueType
  (src/ts/utils/dmplexlandau/kokkos/landau.kokkos.cxx): default-init to 0, copy, and += (plus the
  volatile += Kokkos still expects for some reducers).
*/
namespace baijkok_mult_red
{
template <int RBS>
struct BlockRowAccum {
  MatScalarType v[RBS];

  KOKKOS_INLINE_FUNCTION BlockRowAccum()
  {
    for (int i = 0; i < RBS; i++) v[i] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION BlockRowAccum(const BlockRowAccum &rhs)
  {
    for (int i = 0; i < RBS; i++) v[i] = rhs.v[i];
  }
  KOKKOS_INLINE_FUNCTION BlockRowAccum &operator+=(const BlockRowAccum &src)
  {
    for (int i = 0; i < RBS; i++) v[i] += src.v[i];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION void operator+=(const volatile BlockRowAccum &src) volatile
  {
    for (int i = 0; i < RBS; i++) v[i] += src.v[i];
  }
};
} // namespace baijkok_mult_red

namespace Kokkos
{ /* reduction identity must live in the Kokkos namespace */
template <int RBS>
struct reduction_identity<baijkok_mult_red::BlockRowAccum<RBS>> {
  KOKKOS_FORCEINLINE_FUNCTION static baijkok_mult_red::BlockRowAccum<RBS> sum() { return baijkok_mult_red::BlockRowAccum<RBS>(); }
};
} // namespace Kokkos

/*
  RunNumericMultNoAtomic_SeqBAIJKokkos - atomic-free, coalesced non-transpose block SpMV y (+)= A x.

  Experimental, opt-in via -mat_baijkokkos_spmv_noatomic. Measured ~15-20% SLOWER than the default
  atomic kernel (RunNumericMult_SeqBAIJKokkos) on the A100 (F9.0): the team-reduction epilogue plus the
  per-element div/mod outweigh the cheap L2 double-atomics they remove, and coalescing is not the
  bottleneck (the per-block reads are already cache-friendly). Kept behind the flag for reproducibility.

  One team per block-row i. The team's threads stride over block-row i's contiguous flat value span
  [a_start*bs2, a_end*bs2) via TeamThreadRange, so consecutive lanes read consecutive a_d entries
  (coalesced, F9.0 fix #2). Each thread maps its flat value index v to (block, ii, jj) and accumulates
  a_d(v)*x[bj*CBS+jj] into output-row ii of a per-team BlockRowAccum<RBS>; Kokkos::Sum reduces across
  the team so the RBS outputs are written exactly once by Kokkos::single (no per-row atomics, F9.0 fix
  #1). bs2 = RBS*CBS is compile-time so the div/mod compile to multiply-shift. The output is accumulated
  (+=) so the caller's pre-zero (Mult) or pre-seed (MultAdd, y already copied into the output vec) both
  work. Non-transpose only: the transpose's outputs scatter across block-rows, so it keeps the atomic
  kernel (RunNumericMult_SeqBAIJKokkos<...,true>).
*/
template <int RBS, int CBS, typename RowMapV, typename ColIdxV, typename ScalarV, typename XView, typename YView>
static PetscErrorCode RunNumericMultNoAtomic_SeqBAIJKokkos(PetscInt mbs, PetscInt row_bs, PetscInt col_bs, PetscInt team_size, RowMapV a_i_d, ColIdxV a_j_d, ScalarV a_d, XView xv, YView yv)
{
  using TeamPolicy  = Kokkos::TeamPolicy<DefaultExecutionSpace>;
  using Accum       = baijkok_mult_red::BlockRowAccum<RBS>;
  constexpr int bs2 = RBS * CBS;

  PetscFunctionBegin;
  (void)row_bs;
  (void)col_bs;
  Kokkos::parallel_for(
    "MatMultNoAtomic_SeqBAIJKokkos", TeamPolicy(mbs, team_size), KOKKOS_LAMBDA(const KokkosTeamMemberType &team) {
      PetscInt i       = team.league_rank();
      PetscInt a_start = a_i_d(i);
      PetscInt a_end   = a_i_d(i + 1);
      PetscInt v0      = a_start * bs2;
      Accum    out;

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, v0, a_end * bs2),
        [&](const PetscInt v, Accum &acc) {
          PetscInt local = v - v0;
          PetscInt blk   = local / bs2;
          PetscInt e     = local % bs2;
          PetscInt ii    = e / CBS;
          PetscInt jj    = e % CBS;
          PetscInt bj    = a_j_d(a_start + blk);
          acc.v[ii] += a_d(v) * xv(bj * CBS + jj);
        },
        Kokkos::Sum<Accum>(out));

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        for (int ii = 0; ii < RBS; ii++) yv(i * RBS + ii) += out.v[ii];
      });
    });
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  RunNumericMult_SeqBAIJKokkos - launch the team-parallel block SpMV y (+)= A x or A^T x.

  One team per block-row i of A; threads split row i's blocks via TeamThreadRange (F8a), each doing
  one per-block GEMV with atomic accumulation into the output (F8b). TRANS swaps which vector segment
  is the source and which is the target: non-transpose reads x[bj] and writes y[i]; transpose reads
  x[i] and writes y[bj]. When the template dims (RBS,CBS) are nonzero the per-block GEMV is the
  unrolled, register-staged BlockGemvAccum() specialization; RBS==0 selects BlockGemvAccumGeneric().
  The output vector must be zeroed by the caller (Mult) or pre-seeded with y (MultAdd) before launch.
  View types are deduced so the caller passes device views directly.
*/
template <int RBS, int CBS, bool TRANS, typename RowMapV, typename ColIdxV, typename ScalarV, typename XView, typename YView>
static PetscErrorCode RunNumericMult_SeqBAIJKokkos(PetscInt mbs, PetscInt row_bs, PetscInt col_bs, PetscInt team_size, RowMapV a_i_d, ColIdxV a_j_d, ScalarV a_d, XView xv, YView yv)
{
  using TeamPolicy = Kokkos::TeamPolicy<DefaultExecutionSpace>;

  PetscFunctionBegin;
  /* team_size comes from the matrix (per object, -mat_baijkokkos_team_size); a warp cooperates on a
     block-row on a device backend, one thread per team on host. NO Kokkos::AUTO. */
  Kokkos::parallel_for(
    "MatMult_SeqBAIJKokkos", TeamPolicy(mbs, team_size), KOKKOS_LAMBDA(const KokkosTeamMemberType &team) {
      PetscInt i       = team.league_rank();
      PetscInt a_start = a_i_d(i);
      PetscInt a_end   = a_i_d(i + 1);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, a_start, a_end), [&](const PetscInt a_p) {
        PetscInt             bj   = a_j_d(a_p); /* block-column */
        const MatScalarType *aval = a_d.data() + a_p * row_bs * col_bs;
        /* non-transpose: source x[bj], target y[i]; transpose: source x[i], target y[bj] */
        const MatScalarType *xseg = xv.data() + (TRANS ? i * row_bs : bj * col_bs);
        MatScalarType       *yseg = const_cast<MatScalarType *>(yv.data()) + (TRANS ? bj * col_bs : i * row_bs);

        if constexpr (RBS > 0) BlockGemvAccum<RBS, CBS, TRANS>(aval, xseg, yseg);
        else BlockGemvAccumGeneric(aval, xseg, yseg, row_bs, col_bs, TRANS);
      });
    });
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultDispatch_SeqBAIJKokkos - host-side shape dispatch shared by all four MatMult-family ops.

  Picks the compile-time BlockGemvAccum() specialization once on the host from (row_bs,col_bs) for
  the elasticity shapes ({1,3,6} combinations), falling back to the generic runtime kernel otherwise.
  TRANS selects y += A x (false) or y += A^T x (true).
*/
template <bool TRANS, typename RowMapV, typename ColIdxV, typename ScalarV, typename XView, typename YView>
static PetscErrorCode MatMultDispatch_SeqBAIJKokkos(PetscInt mbs, PetscInt row_bs, PetscInt col_bs, PetscInt team_size, PetscBool use_generic, PetscBool use_noatomic_spmv, RowMapV a_i_d, ColIdxV a_j_d, ScalarV a_d, XView xv, YView yv)
{
  /* Default: the atomic per-block kernel, which on the A100 is faster than the atomic-free team
     reduction (F9.0: the reduction epilogue + per-element div/mod cost more than the cheap L2 atomics,
     and the per-block reads are already cache-friendly) and faster than cusparse CSR. The atomic-free
     reduction is opt-in via -mat_baijkokkos_spmv_noatomic (non-transpose, specialized shapes only). The
     transpose path is always atomic (its outputs scatter across block-rows, so a per-row team reduction
     does not apply); -mat_baijkokkos_generic_kernel forces the generic atomic kernel; an unspecialized
     shape uses the generic atomic kernel. (use_generic is handled by the first branch below.) */
  PetscFunctionBegin;
#define BAIJKOK_MULT_DISPATCH(R, Cc) \
  do { \
    if constexpr (TRANS) PetscCall((RunNumericMult_SeqBAIJKokkos<R, Cc, TRANS>(mbs, row_bs, col_bs, team_size, a_i_d, a_j_d, a_d, xv, yv))); \
    else if (use_noatomic_spmv) PetscCall((RunNumericMultNoAtomic_SeqBAIJKokkos<R, Cc>(mbs, row_bs, col_bs, team_size, a_i_d, a_j_d, a_d, xv, yv))); \
    else PetscCall((RunNumericMult_SeqBAIJKokkos<R, Cc, TRANS>(mbs, row_bs, col_bs, team_size, a_i_d, a_j_d, a_d, xv, yv))); \
  } while (0)
  if (use_generic) PetscCall((RunNumericMult_SeqBAIJKokkos<0, 0, TRANS>(mbs, row_bs, col_bs, team_size, a_i_d, a_j_d, a_d, xv, yv))); /* -mat_baijkokkos_generic_kernel */
  else if (row_bs == 1 && col_bs == 1) BAIJKOK_MULT_DISPATCH(1, 1);
  else if (row_bs == 3 && col_bs == 3) BAIJKOK_MULT_DISPATCH(3, 3);
  else if (row_bs == 6 && col_bs == 6) BAIJKOK_MULT_DISPATCH(6, 6);
  else if (row_bs == 3 && col_bs == 6) BAIJKOK_MULT_DISPATCH(3, 6);
  else if (row_bs == 6 && col_bs == 3) BAIJKOK_MULT_DISPATCH(6, 3);
  else PetscCall((RunNumericMult_SeqBAIJKokkos<0, 0, TRANS>(mbs, row_bs, col_bs, team_size, a_i_d, a_j_d, a_d, xv, yv))); /* unspecialized shape -> generic atomic */
#undef BAIJKOK_MULT_DISPATCH
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMult_SeqBAIJKokkos - Compute y = A*x for a rectangular-block sparse matrix.

  Device implementation: one team per block-row, threads split the row's blocks and accumulate a
  per-block GEMV into y with atomics. For each block-row bi of A:
    y[bi*row_bs + ii] += sum (b in [i[bi], i[bi+1])) sum (jj in [0, col_bs))
                         a[b*row_bs*col_bs + ii*col_bs + jj] * x[j[b]*col_bs + jj]

  In other words, a per-block dense GEMV (row_bsxcol_bs * col_bs) accumulates
  into row_bs output entries.
*/
static PetscErrorCode MatMult_SeqBAIJKokkos(Mat A, Vec x, Vec y)
{
  Mat_SeqBAIJKokkos         *baijkok;
  PetscInt                   row_bs, col_bs, mbs;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");

  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosViewWrite(y, &yv));

  /* Zero y, then accumulate y = A*x with the shared block-GEMV kernel */
  PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), yv, 0.0));
  PetscCall(MatMultDispatch_SeqBAIJKokkos<false>(mbs, row_bs, col_bs, baijkok->team_size, baijkok->use_generic, baijkok->use_noatomic_spmv, baijkok->i_dual.view_device(), baijkok->j_dual.view_device(), baijkok->a_dual.view_device(), xv, yv));

  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(VecRestoreKokkosViewWrite(y, &yv));
  PetscCall(PetscLogGpuFlops(2.0 * baijkok->nblks() * row_bs * col_bs));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultAdd_SeqBAIJKokkos - Compute z = y + A*x for a rectangular-block sparse matrix (device).
*/
static PetscErrorCode MatMultAdd_SeqBAIJKokkos(Mat A, Vec x, Vec y, Vec z)
{
  Mat_SeqBAIJKokkos         *baijkok;
  PetscInt                   row_bs, col_bs, mbs;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");
  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  if (y != z) PetscCall(VecCopy(y, z)); /* z = y; then accumulate A*x into z */
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosView(z, &zv)); /* read-write: seeded with y, accumulate into it */
  PetscCall(MatMultDispatch_SeqBAIJKokkos<false>(mbs, row_bs, col_bs, baijkok->team_size, baijkok->use_generic, baijkok->use_noatomic_spmv, baijkok->i_dual.view_device(), baijkok->j_dual.view_device(), baijkok->a_dual.view_device(), xv, zv));
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(VecRestoreKokkosView(z, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * baijkok->nblks() * row_bs * col_bs));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultTranspose_SeqBAIJKokkos - Compute y = A^T*x for a rectangular-block sparse matrix (device).
  x has length mbs*row_bs (rows of A); y has length nbs*col_bs (columns of A).
*/
static PetscErrorCode MatMultTranspose_SeqBAIJKokkos(Mat A, Vec x, Vec y)
{
  Mat_SeqBAIJKokkos         *baijkok;
  PetscInt                   row_bs, col_bs, mbs;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");
  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosViewWrite(y, &yv));
  /* y[bj*col_bs + jj] += sum_ii a[block, ii, jj] * x[bi*row_bs + ii] */
  PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), yv, 0.0));
  PetscCall(MatMultDispatch_SeqBAIJKokkos<true>(mbs, row_bs, col_bs, baijkok->team_size, baijkok->use_generic, baijkok->use_noatomic_spmv, baijkok->i_dual.view_device(), baijkok->j_dual.view_device(), baijkok->a_dual.view_device(), xv, yv));
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(VecRestoreKokkosViewWrite(y, &yv));
  PetscCall(PetscLogGpuFlops(2.0 * baijkok->nblks() * row_bs * col_bs));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultTransposeAdd_SeqBAIJKokkos - Compute z = y + A^T*x for a rectangular-block sparse matrix (device).
*/
static PetscErrorCode MatMultTransposeAdd_SeqBAIJKokkos(Mat A, Vec x, Vec y, Vec z)
{
  Mat_SeqBAIJKokkos         *baijkok;
  PetscInt                   row_bs, col_bs, mbs;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");
  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  if (y != z) PetscCall(VecCopy(y, z)); /* z = y; then accumulate A^T*x into z */
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosView(z, &zv)); /* read-write: seeded with y, accumulate into it */
  PetscCall(MatMultDispatch_SeqBAIJKokkos<true>(mbs, row_bs, col_bs, baijkok->team_size, baijkok->use_generic, baijkok->use_noatomic_spmv, baijkok->i_dual.view_device(), baijkok->j_dual.view_device(), baijkok->a_dual.view_device(), xv, zv));
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(VecRestoreKokkosView(z, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * baijkok->nblks() * row_bs * col_bs));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductCtxDestroy_SeqBAIJKokkos - Destroy the MatProductCtx_SeqBAIJKokkos context.

  Frees the transpose matrix (if any) and the KernelHandle, then deletes the context struct.
*/
static PetscErrorCode MatProductCtxDestroy_SeqBAIJKokkos(PetscCtxRt pdata)
{
  MatProductCtx_SeqBAIJKokkos *ctx;

  PetscFunctionBegin;
  ctx = *reinterpret_cast<MatProductCtx_SeqBAIJKokkos **>(pdata);
  PetscCall(MatDestroy(&ctx->At));
  PetscCall(MatDestroy(&ctx->W));
  delete ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Forward declarations of transpose routines used by the product routines below.
   MatTransposeWithPerm_SeqBAIJKokkos_Private() is declared PETSC_INTERN in baijkokkosimpl.hpp
   (reused by the parallel MPIBAIJKOKKOS products, F2.2 Option B). */
static PetscErrorCode MatTranspose_SeqBAIJKokkos_Private(Mat A, Mat *At);
/* Forward declaration of the native PtAP numeric dispatcher (defined after the AB helpers it reuses) */
static PetscErrorCode MatProductNumericPtAP_SeqBAIJKokkos(Mat C);

/*
  Binary search helper (KOKKOS_INLINE_FUNCTION) - find column j in C's column indices.

  Returns the slot index if found, or -1 if not found.
  Assumes c_j is sorted over [c_start, c_end).
*/
KOKKOS_INLINE_FUNCTION
static PetscInt BinarySearchColumnInCRow(const MatColIdxType *c_j, PetscInt c_start, PetscInt c_end, PetscInt col_target)
{
  if (c_start >= c_end) return -1;
  PetscInt low = c_start, high = c_end;
  while (low < high) {
    PetscInt mid = (low + high) / 2;
    if (c_j[mid] < col_target) low = mid + 1;
    else if (c_j[mid] > col_target) high = mid;
    else return mid;
  }
  return -1;
}

/*
  MatProductSymbolicAB_SeqBAIJKokkos_Helper - Symbolic phase for AB product given explicit A and B matrices.

  Computes C's block sparsity pattern via spgemm_symbolic on the scalar block-graph CSR.
  Returns the completed C matrix (fully assembled MATSEQBAIJKOKKOS) ready for numeric phase.
*/
PETSC_INTERN PetscErrorCode MatProductSymbolicAB_SeqBAIJKokkos_Helper(Mat C, Mat A, Mat B, MatProductCtx_SeqBAIJKokkos *pdata)
{
  Mat_SeqBAIJKokkos *akok, *bkok, *ckok;
  MPI_Comm           comm;
  KokkosCsrMatrix    csrmatA, csrmatB, csrmatC;
  PetscInt           row_bs_C, col_bs_C, mbs_C, nbs_C;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)C, &comm));

  /* Sync A and B to device */
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  PetscCall(MatSeqBAIJKokkosSyncDevice(B));

  akok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  bkok = static_cast<Mat_SeqBAIJKokkos *>(B->spptr);

  PetscCheck(akok, comm, PETSC_ERR_PLIB, "A missing Mat_SeqBAIJKokkos");
  PetscCheck(bkok, comm, PETSC_ERR_PLIB, "B missing Mat_SeqBAIJKokkos");

  /* CONTRACTION CHECK: A's column block-size must match B's row block-size */
  PetscCheck(akok->col_bs == bkok->row_bs, comm, PETSC_ERR_ARG_SIZ, "A->col_bs (%" PetscInt_FMT ") != B->row_bs (%" PetscInt_FMT "); block dimensions must match for AB product", akok->col_bs, bkok->row_bs);

  /* C's block dimensions */
  row_bs_C = akok->row_bs;
  col_bs_C = bkok->col_bs;
  mbs_C    = akok->mbs;
  nbs_C    = bkok->nbs;

  /* Set C's sizes and block structure */
  PetscCall(MatSetSizes(C, mbs_C * row_bs_C, nbs_C * col_bs_C, mbs_C * row_bs_C, nbs_C * col_bs_C));
  PetscCall(MatSetBlockSizes(C, row_bs_C, col_bs_C));

  /* Get scalar block-graph CSRs for spgemm_symbolic */
  csrmatA = akok->csrmat_graph;
  csrmatB = bkok->csrmat_graph;

  /* Select spgemm algorithm (mirror AIJ Kokkos logic) */
  auto spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_DEFAULT;
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
  #if PETSC_PKG_CUDA_VERSION_LT(11, 4, 0)
  spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_KK;
  #endif
#endif
  PetscCallCXX(pdata->kh.create_spgemm_handle(spgemm_alg));

  PetscCall(PetscLogGpuTimeBegin());

  /* Symbolic phase on block graph: compute C's block sparsity pattern */
  PetscCallCXX(KokkosSparse::spgemm_symbolic(pdata->kh, csrmatA, false, csrmatB, false, csrmatC));

  /* The "fake numeric" quirk: spgemm_symbolic only populates C's rowmap, not column indices.
     Call spgemm_numeric to populate csrmatC.j_d. */
  PetscCallCXX(KokkosSparse::spgemm_numeric(pdata->kh, csrmatA, false, csrmatB, false, csrmatC));

  /* For KK<4.0.0, sort if needed */
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
  auto spgemmHandle = pdata->kh.get_spgemm_handle();
  if (spgemmHandle->get_sort_option() != 1) PetscCallCXX(sort_crs_matrix(csrmatC));
#endif

  PetscCall(PetscLogGpuTimeEnd());

  /* Extract C's block graph from csrmatC and build DualViews */
  auto     csrmatC_graph = csrmatC.graph;
  PetscInt nnzC          = csrmatC.nnz();

  /* Create non-const versions of the row and column views by copying */
  auto i_const = csrmatC_graph.row_map;
  auto j_const = csrmatC_graph.entries;

  MatRowMapKokkosView i_d_C("i_C", mbs_C + 1);
  MatColIdxKokkosView j_d_C("j_C", nnzC);

  /* Copy the const views to non-const */
  Kokkos::deep_copy(i_d_C, i_const);
  Kokkos::deep_copy(j_d_C, j_const);

  /* Allocate C's block values: length nnzC * row_bs_C * col_bs_C, zeroed on device */
  MatScalarKokkosView a_d_C("a_C_product", nnzC * row_bs_C * col_bs_C);

  /* Build C as a fully-formed MATSEQBAIJKOKKOS using the device-view constructor */
  PetscCallCXX(ckok = new Mat_SeqBAIJKokkos(row_bs_C, col_bs_C, mbs_C, nbs_C, nnzC, i_d_C, j_d_C, a_d_C));

  /* Set C as the matrix type and mark as assembled */
  PetscCall(MatSetType(C, MATSEQBAIJKOKKOS));
  C->spptr     = ckok;
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  BlockGemmAccum - compile-time-sized dense block GEMM accumulate: C_ij += A_ik * B_kj.

  Template parameters RBS/K/CBS are the block dimensions (rows of A, contraction, cols of B).
  Blocks are row-major: element (ii,jj) is at offset ii*ncols+jj. The B-block is staged into a
  thread-local register array b_reg[K][CBS] first, so the contraction reads it stride-1 along jj;
  the generic kernel's bval[kk*CBS+jj] access is column-strided over the contraction kk (F8c). The
  unrolled compile-time loops keep b_reg in registers. Updates use Kokkos::atomic_add because team
  threads handling different A-blocks in a row may target the same C-block (F8b).
*/
template <int RBS, int K, int CBS>
KOKKOS_INLINE_FUNCTION static void BlockGemmAccum(const MatScalarType *aval, const MatScalarType *bval, MatScalarType *cval)
{
  MatScalarType b_reg[K][CBS];

#pragma unroll
  for (int kk = 0; kk < K; kk++)
#pragma unroll
    for (int jj = 0; jj < CBS; jj++) b_reg[kk][jj] = bval[kk * CBS + jj];

#pragma unroll
  for (int ii = 0; ii < RBS; ii++) {
#pragma unroll
    for (int jj = 0; jj < CBS; jj++) {
      MatScalarType sum = 0.0;
#pragma unroll
      for (int kk = 0; kk < K; kk++) sum += aval[ii * K + kk] * b_reg[kk][jj];
      Kokkos::atomic_add(&cval[ii * CBS + jj], sum);
    }
  }
}

/*
  BlockGemmAccumGeneric - runtime-sized fallback of BlockGemmAccum() for shapes lacking a specialization.
*/
KOKKOS_INLINE_FUNCTION static void BlockGemmAccumGeneric(const MatScalarType *aval, const MatScalarType *bval, MatScalarType *cval, PetscInt rbsA, PetscInt kdim, PetscInt cbsB)
{
  for (PetscInt ii = 0; ii < rbsA; ii++) {
    for (PetscInt jj = 0; jj < cbsB; jj++) {
      MatScalarType sum = 0.0;
      for (PetscInt kk = 0; kk < kdim; kk++) sum += aval[ii * kdim + kk] * bval[kk * cbsB + jj];
      Kokkos::atomic_add(&cval[ii * cbsB + jj], sum);
    }
  }
}

/*
  RunNumericAB_SeqBAIJKokkos - launch the team-parallel block-GEMM C = A*B.

  One team per block-row i of A; threads in the team split row i's A-blocks via TeamThreadRange
  (F8a), each handling one A_ik and looping over the matching row k of B, accumulating into C with
  atomic_add (F8b). When the template dims (RBS,K,CBS) are nonzero the per-block GEMM is the
  unrolled, register-staged BlockGemmAccum() specialization; RBS==0 selects the runtime-sized
  BlockGemmAccumGeneric(). View types are deduced so the caller passes device views directly.
*/
template <int RBS, int K, int CBS, typename RowMapV, typename ColIdxV, typename ScalarV>
static PetscErrorCode RunNumericAB_SeqBAIJKokkos(PetscInt mbs_A, PetscInt rbsA, PetscInt kdim, PetscInt cbsB, PetscInt team_size, RowMapV a_i_d, ColIdxV a_j_d, ScalarV a_d, RowMapV b_i_d, ColIdxV b_j_d, ScalarV b_d, RowMapV c_i_d, ColIdxV c_j_d, ScalarV c_d)
{
  using TeamPolicy = Kokkos::TeamPolicy<DefaultExecutionSpace>;

  PetscFunctionBegin;
  /* team_size comes from the matrix (per object, -mat_baijkokkos_team_size); a warp's threads
     cooperate on a block-row's A-blocks on a device backend, one thread per team on host. NO
     Kokkos::AUTO (it has chosen poorly here). */
  Kokkos::parallel_for(
    "MatProductNumeric_SeqBAIJKokkos_AB", TeamPolicy(mbs_A, team_size), KOKKOS_LAMBDA(const KokkosTeamMemberType &team) {
      PetscInt i       = team.league_rank();
      PetscInt a_start = a_i_d(i);
      PetscInt a_end   = a_i_d(i + 1);
      PetscInt c_start = c_i_d(i);
      PetscInt c_end   = c_i_d(i + 1);

      /* Threads split the A-blocks in row i; each handles one A_ik and its B-row k */
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, a_start, a_end), [&](const PetscInt a_p) {
        PetscInt             k       = a_j_d(a_p); /* block-row of B */
        const MatScalarType *aval    = a_d.data() + a_p * rbsA * kdim;
        PetscInt             b_start = b_i_d(k);
        PetscInt             b_end   = b_i_d(k + 1);

        for (PetscInt b_q = b_start; b_q < b_end; b_q++) {
          PetscInt             j    = b_j_d(b_q); /* block-column of B and C */
          const MatScalarType *bval = b_d.data() + b_q * kdim * cbsB;
          PetscInt             p_c  = BinarySearchColumnInCRow(c_j_d.data(), c_start, c_end, j);

          if (p_c < 0) Kokkos::abort("Block (i,j) not in C's sparsity pattern; symbolic phase failed");
          MatScalarType *cval = const_cast<MatScalarType *>(c_d.data()) + p_c * rbsA * cbsB;
          if constexpr (RBS > 0) BlockGemmAccum<RBS, K, CBS>(aval, bval, cval);
          else BlockGemmAccumGeneric(aval, bval, cval, rbsA, kdim, cbsB);
        }
      });
    });
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductNumeric_SeqBAIJKokkos_Helper - Numeric phase for AB product given explicit A and B matrices.

  Computes C = A*B using the block-CSR graph (already set by symbolic). Both A and B are synced to
  device; the result C is marked as modified on device. Dispatches a compile-time block-GEMM
  specialization for the elasticity block shapes ({1,3,6} combinations), falling back to the generic
  runtime-sized kernel for any other shape.
*/
PETSC_INTERN PetscErrorCode MatProductNumericAB_SeqBAIJKokkos_Helper(Mat C, Mat A, Mat B)
{
  Mat_SeqBAIJKokkos *akok, *bkok, *ckok;
  PetscInt           rbsA, kdim, cbsB, mbs_A;

  PetscFunctionBegin;
  /* Sync A and B to device */
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  PetscCall(MatSeqBAIJKokkosSyncDevice(B));

  akok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  bkok = static_cast<Mat_SeqBAIJKokkos *>(B->spptr);
  ckok = static_cast<Mat_SeqBAIJKokkos *>(C->spptr);

  PetscCheck(akok && bkok && ckok, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Missing BAIJKOK structure");

  /* Extract block dimensions */
  rbsA  = akok->row_bs; /* C's row block size */
  kdim  = akok->col_bs; /* contraction dimension (A's col_bs == B's row_bs, checked in symbolic) */
  cbsB  = bkok->col_bs; /* C's col block size */
  mbs_A = akok->mbs;

  PetscCheck(kdim == bkok->row_bs, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Dimension mismatch: A.col_bs %" PetscInt_FMT " != B.row_bs %" PetscInt_FMT, kdim, bkok->row_bs);

  /* Capture device views as locals (shallow copies; no dereference of structs inside kernel) */
  auto a_i_d = akok->i_dual.view_device();
  auto a_j_d = akok->j_dual.view_device();
  auto a_d   = akok->a_dual.view_device();

  auto b_i_d = bkok->i_dual.view_device();
  auto b_j_d = bkok->j_dual.view_device();
  auto b_d   = bkok->a_dual.view_device();

  auto c_i_d = ckok->i_dual.view_device();
  auto c_j_d = ckok->j_dual.view_device();
  auto c_d   = ckok->a_dual.view_device();

  /* Zero C's block values on device */
  Kokkos::deep_copy(c_d, 0.0);

  PetscCall(PetscLogGpuTimeBegin());

  /* Dispatch a compile-time specialization for the elasticity block shapes; generic otherwise. The
     tuning (team size, generic toggle) comes from C's matrix (-mat_baijkokkos_*). */
#define BAIJKOK_AB_DISPATCH(R, KK, Cc) PetscCall((RunNumericAB_SeqBAIJKokkos<R, KK, Cc>(mbs_A, rbsA, kdim, cbsB, ckok->team_size, a_i_d, a_j_d, a_d, b_i_d, b_j_d, b_d, c_i_d, c_j_d, c_d)))
  if (ckok->use_generic) BAIJKOK_AB_DISPATCH(0, 0, 0);
  else if (rbsA == 1 && kdim == 1 && cbsB == 1) BAIJKOK_AB_DISPATCH(1, 1, 1);
  else if (rbsA == 3 && kdim == 3 && cbsB == 3) BAIJKOK_AB_DISPATCH(3, 3, 3);
  else if (rbsA == 3 && kdim == 3 && cbsB == 6) BAIJKOK_AB_DISPATCH(3, 3, 6);
  else if (rbsA == 3 && kdim == 6 && cbsB == 6) BAIJKOK_AB_DISPATCH(3, 6, 6);
  else if (rbsA == 6 && kdim == 3 && cbsB == 3) BAIJKOK_AB_DISPATCH(6, 3, 3);
  else if (rbsA == 6 && kdim == 3 && cbsB == 6) BAIJKOK_AB_DISPATCH(6, 3, 6);
  else if (rbsA == 6 && kdim == 6 && cbsB == 6) BAIJKOK_AB_DISPATCH(6, 6, 6);
  else BAIJKOK_AB_DISPATCH(0, 0, 0);
#undef BAIJKOK_AB_DISPATCH

  PetscCall(PetscLogGpuTimeEnd());

  /* Mark C's device values as modified */
  PetscCall(MatSeqBAIJKokkosModifyDevice(C));

  /* Ensure C stays marked as assembled */
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductNumeric_SeqBAIJKokkos - Dispatch numeric phase for AB or AtB products.

  For MATPRODUCT_AB: calls MatProductNumericAB_SeqBAIJKokkos_Helper(C, product->A, product->B).
  For MATPRODUCT_AtB: refreshes the values of the cached transpose product->data->At on device from
    product->A's current values via the cached block permutation (no structural rebuild, F2.3), then
    calls the AB numeric helper with (At, product->B).

  Level: internal
*/
static PetscErrorCode MatProductNumeric_SeqBAIJKokkos(Mat C)
{
  Mat_Product                 *product = C->product;
  Mat                          A, B;
  MatProductCtx_SeqBAIJKokkos *pdata;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);

  A     = product->A;
  B     = product->B;
  pdata = (MatProductCtx_SeqBAIJKokkos *)product->data;

  if (product->type == MATPRODUCT_AB) {
    /* Numeric phase for C = A*B */
    PetscCall(MatProductNumericAB_SeqBAIJKokkos_Helper(C, A, B));
  } else if (product->type == MATPRODUCT_AtB) {
    /* Numeric phase for C = A^T*B: refresh the cached transpose's values on device from A's current
       values via the cached block permutation (structure built once in symbolic), then recompute. */
    PetscCall(MatRefreshTransposeValues_SeqBAIJKokkos(A, pdata->At, pdata->transpose_block_perm));
    PetscCall(MatProductNumericAB_SeqBAIJKokkos_Helper(C, pdata->At, B));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "Product type %s not supported", MatProductTypes[product->type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TripleProductAccum - compile-time-sized fused block triple product: Ac_block += R*(A*P).

  R is CR x KR (= P_{iI}^T), A is KR x KR, P is KR x CR; the result block is CR x CR. The intermediate
  T = A*P (KR x CR) is staged in a thread-local register array so the second GEMM reads it stride-1.
  Blocks are row-major. Updates use Kokkos::atomic_add because team threads on different fine block-rows
  i of the same coarse block-row I may target the same Ac block (I,J).
*/
template <int CR, int KR>
KOKKOS_INLINE_FUNCTION static void TripleProductAccum(const MatScalarType *rval, const MatScalarType *aval, const MatScalarType *pval, MatScalarType *cval)
{
  MatScalarType T[KR][CR];

#pragma unroll
  for (int a = 0; a < KR; a++)
#pragma unroll
    for (int b = 0; b < CR; b++) {
      MatScalarType s = 0.0;
#pragma unroll
      for (int c = 0; c < KR; c++) s += aval[a * KR + c] * pval[c * CR + b];
      T[a][b] = s;
    }

#pragma unroll
  for (int x = 0; x < CR; x++)
#pragma unroll
    for (int b = 0; b < CR; b++) {
      MatScalarType s = 0.0;
#pragma unroll
      for (int a = 0; a < KR; a++) s += rval[x * KR + a] * T[a][b];
      Kokkos::atomic_add(&cval[x * CR + b], s);
    }
}

/*
  TripleProductAccumGeneric - runtime-sized fallback of TripleProductAccum() for shapes lacking a
  specialization. Avoids a variable-length temporary by recomputing the A*P column inside the R loop.
*/
KOKKOS_INLINE_FUNCTION static void TripleProductAccumGeneric(const MatScalarType *rval, const MatScalarType *aval, const MatScalarType *pval, MatScalarType *cval, PetscInt cr, PetscInt kr)
{
  for (PetscInt x = 0; x < cr; x++) {
    for (PetscInt b = 0; b < cr; b++) {
      MatScalarType s = 0.0;
      for (PetscInt a = 0; a < kr; a++) {
        MatScalarType ap = 0.0;
        for (PetscInt c = 0; c < kr; c++) ap += aval[a * kr + c] * pval[c * cr + b];
        s += rval[x * kr + a] * ap;
      }
      Kokkos::atomic_add(&cval[x * cr + b], s);
    }
  }
}

/*
  RunNumericPtAP_Fused_SeqBAIJKokkos - launch the fused single-pass block PtAP (Ac = P^T A P, MEMORY M1).

  One team per coarse block-row I; threads split the R-row's blocks via TeamThreadRange (each handles one
  fine block-row i = one R block R_{Ii}). For each i the thread loops A_{ik} over A-row i and P_{kJ} over
  P-row k, forms the CR x CR contribution R_{Ii}*A_{ik}*P_{kJ} with TripleProductAccum(), locates J in
  Ac-row I via BinarySearchColumnInCRow(), and atomic-accumulates. No intermediate W is materialized; the
  A*P sub-block is recomputed per coarse point (redundancy ~ avg coarse points per fine node).
*/
template <int CR, int KR, typename RowMapV, typename ColIdxV, typename ScalarV>
static PetscErrorCode RunNumericPtAP_Fused_SeqBAIJKokkos(PetscInt mbs_C, PetscInt cr, PetscInt kr, PetscInt team_size, RowMapV rt_i, ColIdxV rt_j, ScalarV rt_a, RowMapV a_i, ColIdxV a_j, ScalarV a_a, RowMapV p_i, ColIdxV p_j, ScalarV p_a, RowMapV c_i, ColIdxV c_j, ScalarV c_a)
{
  using TeamPolicy = Kokkos::TeamPolicy<DefaultExecutionSpace>;

  PetscFunctionBegin;
  /* team_size from the matrix (per object, -mat_baijkokkos_team_size): a warp per coarse block-row on
     device, one thread on host. */
  Kokkos::parallel_for(
    "MatProductNumeric_SeqBAIJKokkos_PtAP_Fused", TeamPolicy(mbs_C, team_size), KOKKOS_LAMBDA(const KokkosTeamMemberType &team) {
      PetscInt I       = team.league_rank();
      PetscInt r_start = rt_i(I);
      PetscInt r_end   = rt_i(I + 1);
      PetscInt c_start = c_i(I);
      PetscInt c_end   = c_i(I + 1);

      /* Threads split the R-row's blocks; each handles one fine block-row i and its A/P fan-out */
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, r_start, r_end), [&](const PetscInt rp) {
        PetscInt             i    = rt_j(rp); /* fine block-row */
        const MatScalarType *rval = rt_a.data() + rp * cr * kr;

        for (PetscInt q = a_i(i); q < a_i(i + 1); q++) {
          PetscInt             k    = a_j(q); /* fine block-col of A == block-row of P */
          const MatScalarType *aval = a_a.data() + q * kr * kr;

          for (PetscInt s = p_i(k); s < p_i(k + 1); s++) {
            PetscInt             J    = p_j(s); /* coarse block-col */
            const MatScalarType *pval = p_a.data() + s * kr * cr;
            PetscInt             pc   = BinarySearchColumnInCRow(c_j.data(), c_start, c_end, J);

            if (pc < 0) Kokkos::abort("PtAP fused: block (I,J) not in Ac sparsity pattern; symbolic phase failed");
            MatScalarType *cval = const_cast<MatScalarType *>(c_a.data()) + pc * cr * cr;
            if constexpr (CR > 0) TripleProductAccum<CR, KR>(rval, aval, pval, cval);
            else TripleProductAccumGeneric(rval, aval, pval, cval, cr, kr);
          }
        }
      });
    });
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductNumericPtAP_SeqBAIJKokkos - Numeric phase for the native block PtAP (Ac = P^T A P).

  Refreshes the cached transpose R = P^T values on device from P (F2.3), then either (SPEED) runs two AB
  numeric launches W = A*P and Ac = R*W reusing the cached W, or (MEMORY M1) runs the fused single-pass
  kernel that never materializes W.
*/
static PetscErrorCode MatProductNumericPtAP_SeqBAIJKokkos(Mat C)
{
  Mat_Product                 *product = C->product;
  MatProductCtx_SeqBAIJKokkos *pdata;
  Mat                          A, P;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  A     = product->A;
  P     = product->B;
  pdata = (MatProductCtx_SeqBAIJKokkos *)product->data;

  /* Refresh R = P^T values on device from P's current values via the cached block permutation */
  PetscCall(MatRefreshTransposeValues_SeqBAIJKokkos(P, pdata->At, pdata->transpose_block_perm));

  if (pdata->ptap_alg == MAT_BAIJKOK_PTAP_SPEED) {
    /* Two-product: W = A*P (reused), then Ac = R*W */
    PetscCall(MatProductNumericAB_SeqBAIJKokkos_Helper(pdata->W, A, P));
    PetscCall(MatProductNumericAB_SeqBAIJKokkos_Helper(C, pdata->At, pdata->W));
  } else {
    /* Fused single-pass (MEMORY M1) */
    Mat_SeqBAIJKokkos *rkok, *akok, *pkok, *ckok;
    PetscInt           cr, kr, mbs_C;

    PetscCall(MatSeqBAIJKokkosSyncDevice(A));
    PetscCall(MatSeqBAIJKokkosSyncDevice(P));
    rkok  = static_cast<Mat_SeqBAIJKokkos *>(pdata->At->spptr);
    akok  = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
    pkok  = static_cast<Mat_SeqBAIJKokkos *>(P->spptr);
    ckok  = static_cast<Mat_SeqBAIJKokkos *>(C->spptr);
    cr    = ckok->row_bs; /* coarse block size (== P->col_bs) */
    kr    = akok->row_bs; /* fine block size (== A->row_bs == P->row_bs) */
    mbs_C = ckok->mbs;

    auto rt_i = rkok->i_dual.view_device();
    auto rt_j = rkok->j_dual.view_device();
    auto rt_a = rkok->a_dual.view_device();
    auto a_i  = akok->i_dual.view_device();
    auto a_j  = akok->j_dual.view_device();
    auto a_a  = akok->a_dual.view_device();
    auto p_i  = pkok->i_dual.view_device();
    auto p_j  = pkok->j_dual.view_device();
    auto p_a  = pkok->a_dual.view_device();
    auto c_i  = ckok->i_dual.view_device();
    auto c_j  = ckok->j_dual.view_device();
    auto c_a  = ckok->a_dual.view_device();

    /* Zero Ac's block values on device (atomic accumulation follows) */
    PetscCallCXX(Kokkos::deep_copy(c_a, 0.0));

    PetscCall(PetscLogGpuTimeBegin());
#define BAIJKOK_PTAP_DISPATCH(CRc, KRc) PetscCall((RunNumericPtAP_Fused_SeqBAIJKokkos<CRc, KRc>(mbs_C, cr, kr, ckok->team_size, rt_i, rt_j, rt_a, a_i, a_j, a_a, p_i, p_j, p_a, c_i, c_j, c_a)))
    if (cr == 6 && kr == 3) BAIJKOK_PTAP_DISPATCH(6, 3);
    else if (cr == 3 && kr == 3) BAIJKOK_PTAP_DISPATCH(3, 3);
    else if (cr == 6 && kr == 6) BAIJKOK_PTAP_DISPATCH(6, 6);
    else if (cr == 1 && kr == 1) BAIJKOK_PTAP_DISPATCH(1, 1);
    else BAIJKOK_PTAP_DISPATCH(0, 0);
#undef BAIJKOK_PTAP_DISPATCH
    PetscCall(PetscLogGpuTimeEnd());

    PetscCall(MatSeqBAIJKokkosModifyDevice(C));
    C->assembled = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductSymbolic_SeqBAIJKokkos_SeqBAIJKokkos - Symbolic phase for C = A*B or C = A^T*B where A, B are MATSEQBAIJKOKKOS.

  Input Parameters:
+ C - the product matrix (preallocated, type set to MATSEQBAIJKOKKOS)

  Notes:
  - Handles MATPRODUCT_AB and MATPRODUCT_AtB.
  - For AB: uses the block matrices directly.
  - For AtB: transposes A to create At, stores it in the product context, then
    proceeds with the AB symbolic using At and B as the effective (A, B) pair.
  - Uses KokkosSparse::spgemm_symbolic on the scalar block-graph CSR to determine C's
    block sparsity pattern.
  - Allocates C's block values (zeroed) and builds C as a fully-assembled MATSEQBAIJKOKKOS.
  - Sets C->product->destroy and C->ops->productnumeric for the numeric phase.

  Level: internal
*/
static PetscErrorCode MatProductSymbolic_SeqBAIJKokkos_SeqBAIJKokkos(Mat C)
{
  Mat_Product                 *product = C->product;
  Mat                          A, B, Aeff;
  MatProductCtx_SeqBAIJKokkos *pdata;
  MPI_Comm                     comm;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(PetscObjectGetComm((PetscObject)C, &comm));
  PetscCheck(!product->data, comm, PETSC_ERR_PLIB, "Product data not empty");

  A = product->A;
  B = product->B;

  /* Create product context first */
  PetscCallCXX(product->data = pdata = new MatProductCtx_SeqBAIJKokkos());
  pdata->reusesym = product->api_user;

  /* Handle both AB and AtB */
  if (product->type == MATPRODUCT_AB) {
    Aeff = A;
  } else if (product->type == MATPRODUCT_AtB) {
    /* Build the transpose structure of A once, plus the block permutation used to refresh its values
       on device in the numeric phase (F2.3) */
    PetscCall(MatTransposeWithPerm_SeqBAIJKokkos_Private(A, &pdata->At, &pdata->transpose_block_perm));
    Aeff = pdata->At;
  } else {
    SETERRQ(comm, PETSC_ERR_SUP, "Product type %s not supported (use AB or AtB)", MatProductTypes[product->type]);
  }

  /* Call the AB symbolic helper with effective (Aeff, B) */
  PetscCall(MatProductSymbolicAB_SeqBAIJKokkos_Helper(C, Aeff, B, pdata));

  /* Register numeric and destroy callbacks */
  C->product->destroy    = MatProductCtxDestroy_SeqBAIJKokkos;
  C->ops->productnumeric = MatProductNumeric_SeqBAIJKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatTranspose_SeqBAIJKokkos_Private - Transpose a MATSEQBAIJKOKKOS matrix.

  Builds a new transposed matrix A^T with:
  - Transposed block-CSR structure (block row indices become columns, vice versa)
  - Each block transposed (row_bsxcol_bs block becomes col_bsxrow_bs block)
  - Per-block values transposed: entry (i,j) of A block becomes entry (j,i) of A^T block

  Host construction; the result is assembled and ready for device sync on demand.
*/
static PetscErrorCode MatTranspose_SeqBAIJKokkos_Private(Mat A, Mat *At)
{
  Mat_SeqBAIJKokkos *akok;
  PetscInt           mbs, nbs, row_bs, col_bs, nblk;
  PetscInt          *ai_h, *aj_h;
  MatScalarType     *aa_h;
  PetscInt          *nnzT;
  PetscScalar       *block_T;
  PetscInt           i, j, p, bi, bj;

  PetscFunctionBegin;
  akok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(akok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing spptr");

  /* Sync A to host */
  PetscCall(MatSeqBAIJKokkosSyncHost(A));

  /* Get block parameters */
  row_bs = akok->row_bs;
  col_bs = akok->col_bs;
  mbs    = akok->mbs; /* block-rows of A */
  nbs    = akok->nbs; /* block-columns of A */
  nblk   = akok->nblks();

  /* Host views */
  ai_h = akok->i_host_data();
  aj_h = akok->j_host_data();
  aa_h = akok->a_host_data();

  /* Allocate temporary storage for per-block transposition */
  PetscCall(PetscMalloc1(col_bs * row_bs, &block_T));

  /* Determine block-sparsity for transposed matrix: count blocks per transposed block-row */
  PetscCall(PetscCalloc1(nbs, &nnzT));
  for (p = 0; p < nblk; p++) {
    bj = aj_h[p];
    nnzT[bj]++;
  }

  /* Create and preallocate the transposed matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), At));
  PetscCall(MatSetSizes(*At, nbs * col_bs, mbs * row_bs, nbs * col_bs, mbs * row_bs));
  PetscCall(MatSetBlockSizes(*At, col_bs, row_bs));
  PetscCall(MatSetType(*At, MATSEQBAIJKOKKOS));
  PetscCall(MatSeqBAIJKokkosSetPreallocation_SeqBAIJKokkos(*At, col_bs, row_bs, 0, nnzT));

  /* Fill the transposed matrix */
  for (bi = 0; bi < mbs; bi++) {
    for (p = ai_h[bi]; p < ai_h[bi + 1]; p++) {
      bj = aj_h[p];

      /* Transpose the block: A block (bi, bj) is row_bs x col_bs; A^T block (bj, bi) is col_bs x row_bs */
      for (i = 0; i < row_bs; i++) {
        for (j = 0; j < col_bs; j++) {
          block_T[j * row_bs + i] = aa_h[p * row_bs * col_bs + i * col_bs + j];
        }
      }

      /* Insert the transposed block into A^T at position (bj, bi) */
      PetscCall(MatSetValuesBlocked(*At, 1, &bj, 1, &bi, block_T, INSERT_VALUES));
    }
  }

  /* Assemble the transposed matrix */
  PetscCall(MatAssemblyBegin(*At, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*At, MAT_FINAL_ASSEMBLY));

  /* Cleanup */
  PetscCall(PetscFree(block_T));
  PetscCall(PetscFree(nnzT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatTransposeWithPerm_SeqBAIJKokkos_Private - Build A^T's structure once and a block-value permutation
  for cheap value refresh across product reuse (F2.3).

  Builds A^T as a fully-assembled MATSEQBAIJKOKKOS directly from a manual block-CSR transpose (mirroring
  the AIJ MatSeqAIJKokkosGenerateTransposeStructure() at block granularity), and returns block_perm: a
  device array of length nblk mapping each A^T block-slot p to its source A block-slot. A^T's block
  values are left zero here; MatRefreshTransposeValues_SeqBAIJKokkos() fills them on device via the perm.
  Columns within each transposed block-row are sorted ascending, matching the assembled order the AB
  symbolic phase and BinarySearchColumnInCRow() assume.
*/
PETSC_INTERN PetscErrorCode MatTransposeWithPerm_SeqBAIJKokkos_Private(Mat A, Mat *At, MatColIdxKokkosView *block_perm)
{
  Mat_SeqBAIJKokkos *akok, *atkok;
  PetscInt           mbs, nbs, row_bs, col_bs, nblk, r, p, bi, disp;
  PetscInt          *ai_h, *aj_h;
  PetscInt          *Ti, *Tj, *perm, *offset;

  PetscFunctionBegin;
  akok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(akok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing spptr");

  /* Only the block graph (i,j) is needed for structure + perm; values are not read here */
  PetscCall(MatSeqBAIJKokkosSyncHost(A));
  row_bs = akok->row_bs;
  col_bs = akok->col_bs;
  mbs    = akok->mbs;
  nbs    = akok->nbs;
  nblk   = akok->nblks();
  ai_h   = akok->i_host_data();
  aj_h   = akok->j_host_data();

  /* Build the transposed block-row map by counting blocks per transposed block-row, then prefix-sum */
  PetscCall(PetscCalloc1(nbs + 1, &Ti));
  for (p = 0; p < nblk; p++) Ti[aj_h[p] + 1]++;
  for (r = 0; r < nbs; r++) Ti[r + 1] += Ti[r];

  /* Scatter block-columns and the source-block permutation into the transposed rows */
  PetscCall(PetscMalloc1(nblk, &Tj));
  PetscCall(PetscMalloc1(nblk, &perm));
  PetscCall(PetscCalloc1(nbs, &offset));
  for (bi = 0; bi < mbs; bi++) {
    for (p = ai_h[bi]; p < ai_h[bi + 1]; p++) {
      r          = aj_h[p]; /* transposed block-row */
      disp       = Ti[r] + offset[r];
      Tj[disp]   = bi; /* transposed block-column */
      perm[disp] = p;  /* source A block-slot */
      offset[r]++;
    }
  }
  PetscCall(PetscFree(offset));

  /* Sort each transposed row's columns (and the perm with them) ascending */
  for (r = 0; r < nbs; r++) PetscCall(PetscSortIntWithArray(Ti[r + 1] - Ti[r], Tj + Ti[r], perm + Ti[r]));

  /* Copy structure + perm to device; allocate zeroed A^T block values (filled by the numeric refresh) */
  MatRowMapKokkosView Ti_d("i_At", nbs + 1);
  MatColIdxKokkosView Tj_d("j_At", nblk);
  MatColIdxKokkosView perm_d("transpose_block_perm", nblk);
  MatScalarKokkosView a_d("a_At", nblk * row_bs * col_bs);
  {
    auto Ti_hm   = Kokkos::create_mirror_view(Ti_d);
    auto Tj_hm   = Kokkos::create_mirror_view(Tj_d);
    auto perm_hm = Kokkos::create_mirror_view(perm_d);
    for (r = 0; r <= nbs; r++) Ti_hm(r) = Ti[r];
    for (p = 0; p < nblk; p++) Tj_hm(p) = Tj[p];
    for (p = 0; p < nblk; p++) perm_hm(p) = perm[p];
    Kokkos::deep_copy(Ti_d, Ti_hm);
    Kokkos::deep_copy(Tj_d, Tj_hm);
    Kokkos::deep_copy(perm_d, perm_hm);
  }
  PetscCall(PetscFree(Ti));
  PetscCall(PetscFree(Tj));
  PetscCall(PetscFree(perm));

  /* Build A^T as a fully-formed MATSEQBAIJKOKKOS (col_bs x row_bs blocks, nbs x mbs block grid) */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), At));
  PetscCall(MatSetSizes(*At, nbs * col_bs, mbs * row_bs, nbs * col_bs, mbs * row_bs));
  PetscCall(MatSetBlockSizes(*At, col_bs, row_bs));
  PetscCall(MatSetType(*At, MATSEQBAIJKOKKOS));
  PetscCallCXX(atkok = new Mat_SeqBAIJKokkos(col_bs, row_bs, nbs, mbs, nblk, Ti_d, Tj_d, a_d));
  (*At)->spptr     = atkok;
  (*At)->assembled = PETSC_TRUE;
  *block_perm      = perm_d;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatRefreshTransposeValues_SeqBAIJKokkos - Refresh A^T's block values from A on device (F2.3).

  Given the cached structure + block_perm from MatTransposeWithPerm_SeqBAIJKokkos_Private(), gathers and
  transposes each block of A's current device values into A^T (At block-slot p <- A block-slot
  block_perm(p), with a per-block element transpose). No host sync, no structural rebuild. Mirrors the
  AIJ value refresh Ta(i) = Aa(perm(i)) (MatSeqAIJKokkosGenerateTranspose_Private), generalized to a
  block with element transpose.
*/
PETSC_INTERN PetscErrorCode MatRefreshTransposeValues_SeqBAIJKokkos(Mat A, Mat At, MatColIdxKokkosView block_perm)
{
  Mat_SeqBAIJKokkos *akok, *atkok;
  PetscInt           row_bs, col_bs, nblk;

  PetscFunctionBegin;
  akok  = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  atkok = static_cast<Mat_SeqBAIJKokkos *>(At->spptr);
  PetscCheck(akok && atkok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing spptr");

  row_bs = akok->row_bs;
  col_bs = akok->col_bs;
  nblk   = akok->nblks();

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  {
    auto a_d  = akok->a_dual.view_device();
    auto at_d = atkok->a_dual.view_device();
    auto perm = block_perm;
    Kokkos::parallel_for(
      "MatRefreshTransposeValues_SeqBAIJKokkos", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nblk), KOKKOS_LAMBDA(const PetscInt p) {
        PetscInt             src   = perm(p);
        const MatScalarType *ablk  = a_d.data() + src * row_bs * col_bs; /* A block (row_bs x col_bs) */
        MatScalarType       *atblk = at_d.data() + p * col_bs * row_bs;  /* A^T block (col_bs x row_bs) */
        for (PetscInt ii = 0; ii < row_bs; ii++)
          for (PetscInt jj = 0; jj < col_bs; jj++) atblk[jj * row_bs + ii] = ablk[ii * col_bs + jj];
      });
  }
  PetscCall(MatSeqBAIJKokkosModifyDevice(At));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatTranspose_SeqBAIJKokkos - Public transpose operation for MATSEQBAIJKOKKOS.

  Handles MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX, and MAT_INPLACE_MATRIX reuse modes.
*/
static PetscErrorCode MatTranspose_SeqBAIJKokkos(Mat A, MatReuse reuse, Mat *B)
{
  Mat At;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) PetscCall(MatTransposeCheckNonzeroState_Private(A, *B));

  /* Build the transpose */
  PetscCall(MatTranspose_SeqBAIJKokkos_Private(A, &At));

  if (reuse == MAT_INITIAL_MATRIX) {
    *B = At;
  } else if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &At));
  } else { /* MAT_REUSE_MATRIX: B must be an assembled transpose with the same structure; copy values */
    Mat_SeqBAIJKokkos *b_kok  = static_cast<Mat_SeqBAIJKokkos *>((*B)->spptr);
    Mat_SeqBAIJKokkos *at_kok = static_cast<Mat_SeqBAIJKokkos *>(At->spptr);

    PetscCheck((*B)->assembled && b_kok, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "MAT_REUSE_MATRIX requires an assembled B");
    PetscCheck(b_kok->a_dual.view_host().extent(0) == at_kok->a_dual.view_host().extent(0), PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_INCOMP, "MAT_REUSE_MATRIX: B has a different nonzero structure than the transpose of A");
    PetscCall(MatSeqBAIJKokkosSyncHost(At));
    PetscCallCXX(Kokkos::deep_copy(b_kok->a_dual.view_host(), at_kok->a_dual.view_host()));
    b_kok->a_dual.modify_host();
    PetscCall(MatDestroy(&At));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SpgemmBlockGraph_SeqBAIJKokkos - Run KokkosSparse::spgemm on two scalar block-graph CSRs and return
  the product block-graph (row map + column indices + nnz).

  Used by the native PtAP symbolic to chain graph(A*P) then graph(P^T*(A*P)). A local KernelHandle is
  created and destroyed per call so the two chained spgemms do not share handle state. The "fake
  numeric" call is the KK quirk that populates the column indices (spgemm_symbolic only fills the row
  map). Result views are fresh non-const copies (length mbsA+1 and nnz).
*/
static PetscErrorCode SpgemmBlockGraph_SeqBAIJKokkos(const KokkosCsrMatrix &csrmatA, const KokkosCsrMatrix &csrmatB, MatRowMapKokkosView &i_d_out, MatColIdxKokkosView &j_d_out, PetscInt &nnz_out)
{
  KernelHandle    kh;
  KokkosCsrMatrix csrmatC;
  PetscInt        mbsA = csrmatA.numRows();

  PetscFunctionBegin;
  auto spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_DEFAULT;
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
  #if PETSC_PKG_CUDA_VERSION_LT(11, 4, 0)
  spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_KK;
  #endif
#endif
  PetscCallCXX(kh.create_spgemm_handle(spgemm_alg));

  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCXX(KokkosSparse::spgemm_symbolic(kh, csrmatA, false, csrmatB, false, csrmatC));
  PetscCallCXX(KokkosSparse::spgemm_numeric(kh, csrmatA, false, csrmatB, false, csrmatC));
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
  {
    auto spgemmHandle = kh.get_spgemm_handle();
    if (spgemmHandle->get_sort_option() != 1) PetscCallCXX(sort_crs_matrix(csrmatC));
  }
#endif
  PetscCall(PetscLogGpuTimeEnd());

  nnz_out = csrmatC.nnz();
  PetscCallCXX(i_d_out = MatRowMapKokkosView("i_spgemm", mbsA + 1));
  PetscCallCXX(j_d_out = MatColIdxKokkosView("j_spgemm", nnz_out));
  PetscCallCXX(Kokkos::deep_copy(i_d_out, csrmatC.graph.row_map));
  PetscCallCXX(Kokkos::deep_copy(j_d_out, csrmatC.graph.entries));
  PetscCallCXX(kh.destroy_spgemm_handle());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductSymbolic_PtAP_SeqBAIJKokkos - Native block PtAP (Ac = P^T A P) symbolic phase.

  A = product->A (square, row_bs==col_bs), P = product->B (rectangular row_bs x col_bs allowed). Builds
  Ac's block sparsity by chaining two scalar-block-graph spgemms: graph(W = A*P), then
  graph(Ac = R*W) with R = P^T. The transpose R is built once with a block permutation (F2.3) and
  cached for the numeric value refresh. For the SPEED algorithm, W is materialized as a fully-formed
  MATSEQBAIJKOKKOS (zeroed values) and cached; the MEMORY (fused) algorithm keeps only the graphs.

  The numeric algorithm is selected from product->alg (-mat_product_algorithm): "default"/"speed" ->
  SPEED (materialize W), "memory_m1"/"memory" -> fused single-pass.
*/
static PetscErrorCode MatProductSymbolic_PtAP_SeqBAIJKokkos(Mat C)
{
  Mat_Product                 *product = C->product;
  Mat                          A, P;
  Mat_SeqBAIJKokkos           *akok, *pkok;
  MatProductCtx_SeqBAIJKokkos *pdata;
  MPI_Comm                     comm;
  MatRowMapKokkosView          i_d_W, i_d_C;
  MatColIdxKokkosView          j_d_W, j_d_C;
  PetscInt                     nnzW, nnzC, row_bs_C, col_bs_C, mbs_C, nbs_C, kdim;
  PetscBool                    is_mem = PETSC_FALSE, is_speed = PETSC_FALSE;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(PetscObjectGetComm((PetscObject)C, &comm));
  PetscCheck(!product->data, comm, PETSC_ERR_PLIB, "Product data not empty");

  A = product->A;
  P = product->B;

  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  PetscCall(MatSeqBAIJKokkosSyncDevice(P));
  akok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  pkok = static_cast<Mat_SeqBAIJKokkos *>(P->spptr);
  PetscCheck(akok && pkok, comm, PETSC_ERR_PLIB, "Missing Mat_SeqBAIJKokkos");
  PetscCheck(akok->row_bs == akok->col_bs, comm, PETSC_ERR_ARG_WRONGSTATE, "PtAP requires A square at block level (row_bs %" PetscInt_FMT " == col_bs %" PetscInt_FMT ")", akok->row_bs, akok->col_bs);
  PetscCheck(akok->col_bs == pkok->row_bs, comm, PETSC_ERR_ARG_SIZ, "A->col_bs (%" PetscInt_FMT ") != P->row_bs (%" PetscInt_FMT ")", akok->col_bs, pkok->row_bs);

  PetscCallCXX(product->data = pdata = new MatProductCtx_SeqBAIJKokkos());
  pdata->reusesym = product->api_user;

  /* Pick the numeric algorithm from -mat_product_algorithm */
  PetscCall(PetscStrcmp(product->alg, "memory_m1", &is_mem));
  if (!is_mem) PetscCall(PetscStrcmp(product->alg, "memory", &is_mem));
  PetscCall(PetscStrcmp(product->alg, "speed", &is_speed));
  pdata->ptap_alg = is_mem ? MAT_BAIJKOK_PTAP_MEMORY_M1 : MAT_BAIJKOK_PTAP_SPEED;
  (void)is_speed; /* "default" and "speed" both map to SPEED */

  /* C = Ac block dims: R(col_bs_P x row_bs_P) * W(row_bs_P x col_bs_P) -> (col_bs_P x col_bs_P) */
  row_bs_C = pkok->col_bs;
  col_bs_C = pkok->col_bs;
  mbs_C    = pkok->nbs; /* coarse block-rows */
  nbs_C    = pkok->nbs; /* coarse block-cols */
  kdim     = akok->row_bs;

  /* Build R = P^T structure + block permutation once (F2.3), cached for the numeric value refresh */
  PetscCall(MatTransposeWithPerm_SeqBAIJKokkos_Private(P, &pdata->At, &pdata->transpose_block_perm));

  /* Symbolic chain on block graphs: graph(W = A*P), then graph(Ac = R*W) */
  PetscCall(SpgemmBlockGraph_SeqBAIJKokkos(akok->csrmat_graph, pkok->csrmat_graph, i_d_W, j_d_W, nnzW));
  {
    /* Wrap the W block graph as a scalar CSR (dummy values) to feed the second spgemm */
    MatScalarKokkosView w_graph_vals("w_graph_vals", nnzW);
    KokkosCsrMatrix     csrmatW("csrmatW_graph", pkok->nbs, w_graph_vals, KokkosCsrGraph(j_d_W, i_d_W));
    Mat_SeqBAIJKokkos  *rkok = static_cast<Mat_SeqBAIJKokkos *>(pdata->At->spptr);
    PetscCall(SpgemmBlockGraph_SeqBAIJKokkos(rkok->csrmat_graph, csrmatW, i_d_C, j_d_C, nnzC));
  }

  /* Build Ac = C as a fully-formed MATSEQBAIJKOKKOS (zeroed block values) */
  PetscCall(MatSetSizes(C, mbs_C * row_bs_C, nbs_C * col_bs_C, mbs_C * row_bs_C, nbs_C * col_bs_C));
  PetscCall(MatSetBlockSizes(C, row_bs_C, col_bs_C));
  {
    MatScalarKokkosView a_d_C("a_C_ptap", nnzC * row_bs_C * col_bs_C);
    Mat_SeqBAIJKokkos  *ckok;
    PetscCallCXX(ckok = new Mat_SeqBAIJKokkos(row_bs_C, col_bs_C, mbs_C, nbs_C, nnzC, i_d_C, j_d_C, a_d_C));
    PetscCall(MatSetType(C, MATSEQBAIJKOKKOS));
    C->spptr     = ckok;
    C->assembled = PETSC_TRUE;
  }

  /* SPEED: materialize W (row_bs_P x col_bs_P) as a cached MATSEQBAIJKOKKOS with zeroed values */
  if (pdata->ptap_alg == MAT_BAIJKOK_PTAP_SPEED) {
    MatScalarKokkosView a_d_W("a_W_ptap", nnzW * pkok->row_bs * pkok->col_bs);
    Mat_SeqBAIJKokkos  *wkok;
    PetscCall(MatCreate(comm, &pdata->W));
    PetscCall(MatSetSizes(pdata->W, akok->mbs * pkok->row_bs, pkok->nbs * pkok->col_bs, akok->mbs * pkok->row_bs, pkok->nbs * pkok->col_bs));
    PetscCall(MatSetBlockSizes(pdata->W, pkok->row_bs, pkok->col_bs));
    PetscCall(MatSetType(pdata->W, MATSEQBAIJKOKKOS));
    PetscCallCXX(wkok = new Mat_SeqBAIJKokkos(pkok->row_bs, pkok->col_bs, akok->mbs, pkok->nbs, nnzW, i_d_W, j_d_W, a_d_W));
    pdata->W->spptr     = wkok;
    pdata->W->assembled = PETSC_TRUE;
  }
  (void)kdim;

  C->product->destroy    = MatProductCtxDestroy_SeqBAIJKokkos;
  C->ops->productnumeric = MatProductNumericPtAP_SeqBAIJKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatProductSetFromOptions_SeqBAIJKokkos - Register product operations for MATSEQBAIJKOKKOS.

  Dispatches MATPRODUCT_AB and MATPRODUCT_AtB to the custom symbolic kernel; MATPRODUCT_PtAP to the
  native block-PtAP symbolic (SPEED two-product or MEMORY fused, per -mat_product_algorithm).
  MATPRODUCT_RARt and MATPRODUCT_ABC are routed to MatProductSymbolic_ABC_Basic, which decomposes them
  into two pairwise products (AB and AtB) that dispatch back here.
*/
static PetscErrorCode MatProductSetFromOptions_SeqBAIJKokkos(Mat mat)
{
  Mat_Product *product = mat->product;
  PetscBool    Biskok = PETSC_FALSE, Ciskok = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)product->B, MATSEQBAIJKOKKOS, &Biskok));
  if (product->type == MATPRODUCT_ABC) PetscCall(PetscObjectTypeCompare((PetscObject)product->C, MATSEQBAIJKOKKOS, &Ciskok));
  if (Biskok && Ciskok) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
      mat->ops->productsymbolic = MatProductSymbolic_SeqBAIJKokkos_SeqBAIJKokkos;
      break;
    case MATPRODUCT_PtAP:
      mat->ops->productsymbolic = MatProductSymbolic_PtAP_SeqBAIJKokkos;
      break;
    case MATPRODUCT_RARt:
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    case MATPRODUCT_ABt:
      /* Deferred to later phases */
      SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Product type %s not yet supported", MatProductTypes[product->type]);
    default:
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatInvertBlockDiagonal_SeqBAIJKokkos - Invert each square diagonal block (host).

  PCPBJACOBI's host setup (PCSetUp_PBJacobi_Host) calls this and consumes the returned
  pointer as column-major bs x bs inverse blocks. SEQBAIJKOKKOS stores blocks ROW-MAJOR in
  device DualViews and never wires the base Mat_SeqBAIJ a->i/a->j/a->a host arrays, so the
  inherited host MatInvertBlockDiagonal_SeqBAIJ would deref NULL. We sync values to host,
  locate each diagonal block from the host i/j mirrors, transpose-copy it into column-major
  order, and invert in place with the same PetscKernel_A_gets_inverse_A helpers. The inverse
  is cached in the base a->idiag/a->idiagvalid (the contract storage), and invalidated by
  MatSeqBAIJKokkosModifyDevice()/MatAssemblyEnd_SeqBAIJKokkos() when values change.

  Square blocks only (row_bs == col_bs): PBJacobi is applied only to square level operators,
  never the rectangular prolongator.
*/
static PetscErrorCode MatInvertBlockDiagonal_SeqBAIJKokkos(Mat A, const PetscScalar **values)
{
  Mat_SeqBAIJKokkos *baijkok;
  Mat_SeqBAIJ       *a  = (Mat_SeqBAIJ *)A->data;
  PetscInt           bs = A->rmap->bs, bs2 = bs * bs, mbs, i, ipvt[5], *v_pivots = NULL;
  MatScalar         *diag, work[25], *v_work = NULL;
  PetscReal          shift = 0.0;
  PetscBool          allowzeropivot, zeropivotdetected = PETSC_FALSE;
  const PetscInt    *i_h, *j_h;
  const MatScalar   *a_h;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected NULL baijkok");
  PetscCheck(baijkok->row_bs == baijkok->col_bs, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MatInvertBlockDiagonal requires square blocks (row_bs %" PetscInt_FMT " != col_bs %" PetscInt_FMT ")", baijkok->row_bs, baijkok->col_bs);
  allowzeropivot = PetscNot(A->erroriffailure);

  if (a->idiagvalid) {
    if (values) *values = a->idiag;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  mbs = baijkok->mbs;
  if (!a->idiag) PetscCall(PetscMalloc1(bs2 * mbs, &a->idiag));
  diag = a->idiag;
  if (values) *values = a->idiag;

  PetscCall(MatSeqBAIJKokkosSyncHost(A));
  i_h = baijkok->i_dual.view_host().data();
  j_h = baijkok->j_dual.view_host().data();
  a_h = baijkok->a_dual.view_host().data();

  if (bs > 7) PetscCall(PetscMalloc2(bs, &v_work, bs, &v_pivots));
  for (i = 0; i < mbs; i++) {
    const MatScalar *odiag = NULL;
    PetscInt         ib, jb, block_pos;

    /* Find the diagonal block (block-column == i) within block-row i's [i_h[i], i_h[i+1]) */
    for (block_pos = i_h[i]; block_pos < i_h[i + 1]; block_pos++) {
      if (j_h[block_pos] == i) {
        odiag = a_h + (size_t)block_pos * bs2;
        break;
      }
    }
    PetscCheck(odiag, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Missing diagonal block at block-row %" PetscInt_FMT, i);

    /* Transpose row-major Kokkos block -> column-major diag (the layout the inverse kernels
       and PCPBJacobi expect): diag[ib + jb*bs] = M(ib,jb) = odiag[ib*bs + jb]. */
    for (ib = 0; ib < bs; ib++)
      for (jb = 0; jb < bs; jb++) diag[ib + jb * bs] = odiag[ib * bs + jb];

    switch (bs) {
    case 1:
      if (PetscAbsScalar(diag[0] + shift) < PETSC_MACHINE_EPSILON) {
        PetscCheck(allowzeropivot, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Zero pivot, row %" PetscInt_FMT " pivot value %g tolerance %g", i, (double)PetscAbsScalar(diag[0]), (double)PETSC_MACHINE_EPSILON);
        A->factorerrortype             = MAT_FACTOR_NUMERIC_ZEROPIVOT;
        A->factorerror_zeropivot_value = PetscAbsScalar(diag[0]);
        A->factorerror_zeropivot_row   = i;
        PetscCall(PetscInfo(A, "Zero pivot, row %" PetscInt_FMT "\n", i));
      }
      diag[0] = (PetscScalar)1.0 / (diag[0] + shift);
      break;
    case 2:
      PetscCall(PetscKernel_A_gets_inverse_A_2(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      break;
    case 3:
      PetscCall(PetscKernel_A_gets_inverse_A_3(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      break;
    case 4:
      PetscCall(PetscKernel_A_gets_inverse_A_4(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      break;
    case 5:
      PetscCall(PetscKernel_A_gets_inverse_A_5(diag, ipvt, work, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      break;
    case 6:
      PetscCall(PetscKernel_A_gets_inverse_A_6(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      break;
    case 7:
      PetscCall(PetscKernel_A_gets_inverse_A_7(diag, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      break;
    default:
      PetscCall(PetscKernel_A_gets_inverse_A(bs, diag, v_pivots, v_work, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    }
    diag += bs2;
  }
  if (bs > 7) PetscCall(PetscFree2(v_work, v_pivots));
  a->idiagvalid = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatGetDiagonal_SeqBAIJKokkos - Extract diagonal from a block matrix (device).

  For a square block matrix (row_bs == col_bs && mbs == nbs), fills vector v
  with the diagonal entries. v must have length mbs*row_bs (one scalar per row).
  One device thread per block-row searches for its diagonal block; absent blocks
  leave a zero diagonal (Jacobi smoother).
*/
static PetscErrorCode MatGetDiagonal_SeqBAIJKokkos(Mat A, Vec v)
{
  Mat_SeqBAIJKokkos    *baijkok;
  PetscInt              row_bs, col_bs, mbs;
  PetscScalarKokkosView vv;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected NULL baijkok");
  PetscCheck(baijkok->row_bs == baijkok->col_bs && baijkok->mbs == baijkok->nbs, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "MatGetDiagonal requires square matrix at block level");

  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosViewWrite(v, &vv));
  {
    auto i_d = baijkok->i_dual.view_device();
    auto j_d = baijkok->j_dual.view_device();
    auto a_d = baijkok->a_dual.view_device();

    Kokkos::parallel_for(
      "MatGetDiagonal_SeqBAIJKokkos", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, mbs), KOKKOS_LAMBDA(const PetscInt bi) {
        for (PetscInt ii = 0; ii < row_bs; ii++) vv(bi * row_bs + ii) = 0.0;
        /* Search for the diagonal block (block-column == bi) in [i_d(bi), i_d(bi+1)) */
        for (PetscInt block_pos = i_d(bi); block_pos < i_d(bi + 1); block_pos++) {
          if (j_d(block_pos) == bi) {
            const MatScalarType *block_a = a_d.data() + block_pos * row_bs * col_bs;
            for (PetscInt ii = 0; ii < row_bs; ii++) vv(bi * row_bs + ii) = block_a[ii * col_bs + ii];
            break;
          }
        }
      });
  }
  PetscCall(VecRestoreKokkosViewWrite(v, &vv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatDiagonalScale_SeqBAIJKokkos - Left and/or right diagonal scaling.

  Scales A as: A_ij *= (l? lv[bi*row_bs+ii] : 1) * (r? rv[bj*col_bs+jj] : 1)
  where bi, bj are block row/column indices and ii, jj are within-block indices.
  Device kernel (one thread per block-row). Handles l and/or r being NULL (smoothing passes r=NULL).
*/
static PetscErrorCode MatDiagonalScale_SeqBAIJKokkos(Mat A, Vec l, Vec r)
{
  Mat_SeqBAIJKokkos         *baijkok;
  ConstPetscScalarKokkosView lv, rv; /* empty when l/r is NULL; access guarded by has_l/has_r */
  PetscInt                   row_bs, col_bs, mbs, nbs;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected NULL baijkok");

  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;
  nbs    = baijkok->nbs;

  if (l) PetscCheck(A->rmap->n == row_bs * mbs, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Left vector size mismatch");
  if (r) PetscCheck(A->cmap->n == col_bs * nbs, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Right vector size mismatch");

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  if (l) PetscCall(VecGetKokkosView(l, &lv));
  if (r) PetscCall(VecGetKokkosView(r, &rv));
  {
    const bool has_l = (l != NULL);
    const bool has_r = (r != NULL);
    auto       i_d   = baijkok->i_dual.view_device();
    auto       j_d   = baijkok->j_dual.view_device();
    auto       a_d   = baijkok->a_dual.view_device();

    Kokkos::parallel_for(
      "MatDiagonalScale_SeqBAIJKokkos", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, mbs), KOKKOS_LAMBDA(const PetscInt bi) {
        for (PetscInt block_pos = i_d(bi); block_pos < i_d(bi + 1); block_pos++) {
          PetscInt       bj      = j_d(block_pos);
          MatScalarType *block_a = a_d.data() + block_pos * row_bs * col_bs;
          for (PetscInt ii = 0; ii < row_bs; ii++) {
            MatScalarType l_scale = has_l ? lv(bi * row_bs + ii) : 1.0;
            for (PetscInt jj = 0; jj < col_bs; jj++) {
              MatScalarType r_scale = has_r ? rv(bj * col_bs + jj) : 1.0;
              block_a[ii * col_bs + jj] *= l_scale * r_scale;
            }
          }
        }
      });
  }
  if (l) PetscCall(VecRestoreKokkosView(l, &lv));
  if (r) PetscCall(VecRestoreKokkosView(r, &rv));
  PetscCall(MatSeqBAIJKokkosModifyDevice(A));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatScale_SeqBAIJKokkos - Multiply all block entries by a scalar (device).
*/
static PetscErrorCode MatScale_SeqBAIJKokkos(Mat A, PetscScalar a)
{
  Mat_SeqBAIJKokkos *baijkok;
  PetscInt           nblk_vals;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected NULL baijkok");

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(A));
  {
    auto a_d  = baijkok->a_dual.view_device();
    nblk_vals = static_cast<PetscInt>(a_d.extent(0));
    Kokkos::parallel_for("MatScale_SeqBAIJKokkos", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nblk_vals), KOKKOS_LAMBDA(const PetscInt k) { a_d(k) *= a; });
  }
  PetscCall(MatSeqBAIJKokkosModifyDevice(A));
  PetscCall(PetscLogGpuFlops(nblk_vals));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatAXPY_SeqBAIJKokkos - Y = Y + alpha*X.

  Implements SAME_NONZERO_PATTERN and SUBSET_NONZERO_PATTERN.
  For SUBSET: X's block sparsity is a subset of Y's; all X blocks must exist in Y.
  Device kernels (element-wise for SAME; one thread per block-row with a binary search for SUBSET).
*/
static PetscErrorCode MatAXPY_SeqBAIJKokkos(Mat Y, PetscScalar alpha, Mat X, MatStructure str)
{
  Mat_SeqBAIJKokkos *ykok, *xkok;
  PetscInt           row_bs, col_bs, mbs;

  PetscFunctionBegin;
  PetscCheckTypeName(Y, MATSEQBAIJKOKKOS);
  PetscCheckTypeName(X, MATSEQBAIJKOKKOS);

  ykok = static_cast<Mat_SeqBAIJKokkos *>(Y->spptr);
  xkok = static_cast<Mat_SeqBAIJKokkos *>(X->spptr);
  PetscCheck(ykok && xkok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected NULL baijkok");

  row_bs = ykok->row_bs;
  col_bs = ykok->col_bs;
  mbs    = ykok->mbs;
  PetscCheck(xkok->row_bs == row_bs && xkok->col_bs == col_bs && xkok->mbs == mbs, PetscObjectComm((PetscObject)Y), PETSC_ERR_ARG_INCOMP, "MatAXPY requires same block sizes and dimensions");

  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqBAIJKokkosSyncDevice(Y));
  PetscCall(MatSeqBAIJKokkosSyncDevice(X));

  auto Yi_d = ykok->i_dual.view_device();
  auto Yj_d = ykok->j_dual.view_device();
  auto Ya_d = ykok->a_dual.view_device();
  auto Xi_d = xkok->i_dual.view_device();
  auto Xj_d = xkok->j_dual.view_device();
  auto Xa_d = xkok->a_dual.view_device();

  if (str == SAME_NONZERO_PATTERN) {
    /* X and Y have the same block sparsity pattern; direct element-wise addition */
    PetscInt nblk_vals = static_cast<PetscInt>(Ya_d.extent(0));
    PetscCheck(static_cast<PetscInt>(Xa_d.extent(0)) == nblk_vals, PetscObjectComm((PetscObject)Y), PETSC_ERR_ARG_INCOMP, "SAME_NONZERO_PATTERN requires identical block count");
    Kokkos::parallel_for("MatAXPY_SeqBAIJKokkos_same", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nblk_vals), KOKKOS_LAMBDA(const PetscInt k) { Ya_d(k) += alpha * Xa_d(k); });
  } else if (str == SUBSET_NONZERO_PATTERN) {
    /* X's block pattern is a subset of Y's; for each block in X, binary-search Y's row and add */
    PetscInt sz = row_bs * col_bs;
    Kokkos::parallel_for(
      "MatAXPY_SeqBAIJKokkos_subset", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, mbs), KOKKOS_LAMBDA(const PetscInt bi) {
        for (PetscInt px = Xi_d(bi); px < Xi_d(bi + 1); px++) {
          PetscInt col_x = Xj_d(px);
          PetscInt py    = BinarySearchColumnInCRow(Yj_d.data(), Yi_d(bi), Yi_d(bi + 1), col_x);
          if (py < 0) Kokkos::abort("MatAXPY SUBSET_NONZERO_PATTERN: a block in X is not present in Y");
          for (PetscInt k = 0; k < sz; k++) Ya_d(py * sz + k) += alpha * Xa_d(px * sz + k);
        }
      });
  } else {
    SETERRQ(PetscObjectComm((PetscObject)Y), PETSC_ERR_SUP, "MatAXPY with DIFFERENT_NONZERO_PATTERN not yet supported for MATSEQBAIJKOKKOS");
  }

  PetscCall(MatSeqBAIJKokkosModifyDevice(Y));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatCreateGraph_SeqBAIJKokkos - Build the scalar aggregation graph (for GAMG) from a SeqBAIJKokkos matrix.

  Converts the rectangular-block matrix to scalar SeqAIJ (carrying the block sizes) and calls the
  AIJ MatCreateGraph(), which collapses the bs blocks into scalar nodes. ops->creategraph for the type.
*/
static PetscErrorCode MatCreateGraph_SeqBAIJKokkos(Mat A, PetscBool sym, PetscBool scale, PetscReal filter, PetscInt num_idx, PetscInt index[], Mat *graph)
{
  Mat      Aaij;
  PetscInt rbs, cbs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, sym, 2);
  PetscValidLogicalCollectiveBool(A, scale, 3);
  PetscAssertPointer(graph, 7);

  /* Convert to scalar AIJ to leverage the existing block-collapse logic */
  PetscCall(MatConvert(A, MATSEQAIJ, MAT_INITIAL_MATRIX, &Aaij));

  /* Ensure block sizes are set on the AIJ matrix so the collapse groups dofs into nodes correctly */
  PetscCall(MatGetBlockSizes(A, &rbs, &cbs));
  PetscCall(MatSetBlockSizes(Aaij, rbs, cbs));

  /* Build graph from the AIJ (its MatCreateGraph_Simple_AIJ will collapse blocks) */
  PetscCall(MatCreateGraph(Aaij, sym, scale, filter, num_idx, index, graph));

  PetscCall(MatDestroy(&Aaij));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Factorization of a SEQBAIJKOKKOS matrix via a converted SeqAIJ copy.

  SEQBAIJKOKKOS keeps its block-CSR only in device DualViews and never maintains the base SeqBAIJ host
  arrays (a->i/a->j/a->a), so the inherited host SeqBAIJ factorization (MatLUFactorSymbolic_SeqBAIJ et al.)
  dereferences a NULL a->i and crashes — this is reached by the default GAMG coarse bjacobi/LU solver once
  the coarse operator is block (F10). To support it without maintaining a parallel host representation, we
  factor a scalar SeqAIJ conversion: MatGetFactor() returns a genuine SeqAIJ factor whose symbolic/numeric
  ops are intercepted to substitute the converted copy for the block input matrix. The converted copy is
  kept on the factor (refreshed values-only across numeric reuse) and freed when the factor is destroyed.
*/
typedef struct {
  Mat Aaij; /* SeqAIJ conversion of the block matrix being factored */
  PetscErrorCode (*symbolic_lu)(Mat, Mat, IS, IS, const MatFactorInfo *);
  PetscErrorCode (*symbolic_ilu)(Mat, Mat, IS, IS, const MatFactorInfo *);
  PetscErrorCode (*symbolic_cholesky)(Mat, Mat, IS, const MatFactorInfo *);
  PetscErrorCode (*symbolic_icc)(Mat, Mat, IS, const MatFactorInfo *);
  PetscErrorCode (*numeric_lu)(Mat, Mat, const MatFactorInfo *);
  PetscErrorCode (*numeric_cholesky)(Mat, Mat, const MatFactorInfo *);
} Mat_SeqBAIJKokkosFactor;

static PetscErrorCode MatSeqBAIJKokkosFactorDestroy(PetscCtxRt data)
{
  Mat_SeqBAIJKokkosFactor *fac = *(Mat_SeqBAIJKokkosFactor **)data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&fac->Aaij));
  PetscCall(PetscFree(fac));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode MatSeqBAIJKokkosFactorGet(Mat B, Mat_SeqBAIJKokkosFactor **fac)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectContainerQuery((PetscObject)B, "MatSeqBAIJKokkosFactor", fac));
  PetscCheck(*fac, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Missing SeqBAIJKokkos factor context");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Numeric wrappers: refresh the converted copy's values from the (current) block A, then delegate. */
static PetscErrorCode MatLUFactorNumeric_SeqBAIJKokkosFactor(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat_SeqBAIJKokkosFactor *fac;

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosFactorGet(B, &fac));
  PetscCall(MatConvert(A, MATSEQAIJ, MAT_REUSE_MATRIX, &fac->Aaij));
  PetscCall(fac->numeric_lu(B, fac->Aaij, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJKokkosFactor(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat_SeqBAIJKokkosFactor *fac;

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosFactorGet(B, &fac));
  PetscCall(MatConvert(A, MATSEQAIJ, MAT_REUSE_MATRIX, &fac->Aaij));
  PetscCall(fac->numeric_cholesky(B, fac->Aaij, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Symbolic wrappers: delegate on the converted copy, then re-install the numeric wrapper (the SeqAIJ
   symbolic call installs the real SeqAIJ numeric op, which we capture and override). The incoming row/col
   orderings are ignored — they would be block-sized (or absent, since the factor reports canuseordering =
   PETSC_FALSE so PCSetUp_LU skips MatGetOrdering on the block matrix); a fresh scalar ordering is computed
   on the AIJ copy instead. */
static PetscErrorCode MatLUFactorSymbolic_SeqBAIJKokkosFactor(Mat B, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_SeqBAIJKokkosFactor *fac;
  IS                       ar, ac;

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosFactorGet(B, &fac));
  PetscCall(MatGetOrdering(fac->Aaij, MATORDERINGND, &ar, &ac));
  PetscCall(fac->symbolic_lu(B, fac->Aaij, ar, ac, info));
  PetscCall(ISDestroy(&ar));
  PetscCall(ISDestroy(&ac));
  fac->numeric_lu         = B->ops->lufactornumeric;
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJKokkosFactor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatILUFactorSymbolic_SeqBAIJKokkosFactor(Mat B, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_SeqBAIJKokkosFactor *fac;
  IS                       ar, ac;

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosFactorGet(B, &fac));
  PetscCall(MatGetOrdering(fac->Aaij, MATORDERINGND, &ar, &ac));
  PetscCall(fac->symbolic_ilu(B, fac->Aaij, ar, ac, info));
  PetscCall(ISDestroy(&ar));
  PetscCall(ISDestroy(&ac));
  fac->numeric_lu         = B->ops->lufactornumeric;
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJKokkosFactor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqBAIJKokkosFactor(Mat B, Mat A, IS r, const MatFactorInfo *info)
{
  Mat_SeqBAIJKokkosFactor *fac;
  IS                       ar, ac;

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosFactorGet(B, &fac));
  PetscCall(MatGetOrdering(fac->Aaij, MATORDERINGND, &ar, &ac));
  PetscCall(fac->symbolic_cholesky(B, fac->Aaij, ar, info));
  PetscCall(ISDestroy(&ar));
  PetscCall(ISDestroy(&ac));
  fac->numeric_cholesky         = B->ops->choleskyfactornumeric;
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqBAIJKokkosFactor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatICCFactorSymbolic_SeqBAIJKokkosFactor(Mat B, Mat A, IS r, const MatFactorInfo *info)
{
  Mat_SeqBAIJKokkosFactor *fac;
  IS                       ar, ac;

  PetscFunctionBegin;
  PetscCall(MatSeqBAIJKokkosFactorGet(B, &fac));
  PetscCall(MatGetOrdering(fac->Aaij, MATORDERINGND, &ar, &ac));
  PetscCall(fac->symbolic_icc(B, fac->Aaij, ar, info));
  PetscCall(ISDestroy(&ar));
  PetscCall(ISDestroy(&ac));
  fac->numeric_cholesky         = B->ops->choleskyfactornumeric;
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqBAIJKokkosFactor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatGetFactor_SeqBAIJKokkos_petsc - Produce a PETSc factor for a SEQBAIJKOKKOS matrix by factoring a
  scalar SeqAIJ conversion. Registered (in dlregismat.c) ahead of the base MATSEQBAIJ handler so the
  prefix match in MatSolverTypeGet() resolves "seqbaijkokkos" here rather than to the host SeqBAIJ factor.
*/
PETSC_INTERN PetscErrorCode MatGetFactor_SeqBAIJKokkos_petsc(Mat A, MatFactorType ftype, Mat *B)
{
  Mat                      Aaij;
  Mat_SeqBAIJKokkosFactor *fac;

  PetscFunctionBegin;
  PetscCall(MatConvert(A, MATSEQAIJ, MAT_INITIAL_MATRIX, &Aaij));
  PetscCall(MatGetFactor(Aaij, MATSOLVERPETSC, ftype, B));
  /* PCSetUp_LU/Cholesky would otherwise call MatGetOrdering() on the block matrix, whose host CSR is
     absent; report that the factor does not consume a caller-provided ordering so that step is skipped.
     The symbolic wrappers compute their own ordering on the scalar copy. */
  (*B)->canuseordering = PETSC_FALSE;
  /* Capture the real SeqAIJ symbolic ops, then intercept them to substitute the converted copy. */
  PetscCall(PetscNew(&fac));
  fac->Aaij              = Aaij;
  fac->symbolic_lu       = (*B)->ops->lufactorsymbolic;
  fac->symbolic_ilu      = (*B)->ops->ilufactorsymbolic;
  fac->symbolic_cholesky = (*B)->ops->choleskyfactorsymbolic;
  fac->symbolic_icc      = (*B)->ops->iccfactorsymbolic;
  if (fac->symbolic_lu) (*B)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqBAIJKokkosFactor;
  if (fac->symbolic_ilu) (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqBAIJKokkosFactor;
  if (fac->symbolic_cholesky) (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqBAIJKokkosFactor;
  if (fac->symbolic_icc) (*B)->ops->iccfactorsymbolic = MatICCFactorSymbolic_SeqBAIJKokkosFactor;
  PetscCall(PetscObjectContainerCompose((PetscObject)*B, "MatSeqBAIJKokkosFactor", fac, MatSeqBAIJKokkosFactorDestroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATSEQBAIJKOKKOS - MATBAIJKOKKOS = "(seq)baijkokkos" - A matrix type for rectangular-block sparse matrices with Kokkos

   A matrix type using Kokkos for portability across different device types, designed for generalized (rectangular) block
   matrices. Unlike the square-block-only PETSc BAIJ format, this type supports block row size `row_bs` and block column
   size `col_bs` that need not be equal, enabling use cases such as rectangular prolongators in algebraic multigrid.

   Block-CSR storage:
   - Block row and column indices (not scalar indices)
   - Dense block values stored row-major per block
   - Automatic device synchronization via DualViews

   Options Database Key:
.  -mat_type seqbaijkokkos - sets the matrix type to `MATSEQBAIJKOKKOS` during a call to `MatSetFromOptions()`

   Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATSEQBAIJ`, `MATSEQAIJKOKKOS`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate_SeqBAIJ(A));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATSEQBAIJKOKKOS));
  /* The device MatMult family takes Kokkos Views of x/y, so MatCreateVecs() must hand back
     VECSEQKOKKOS vectors (e.g. the GAMG eigen-estimate and MG transfer work vectors). */
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECKOKKOS, &A->defaultvectype));
  A->ops->destroy             = MatDestroy_SeqBAIJKokkos;
  A->ops->setfromoptions      = MatSetFromOptions_SeqBAIJKokkos;
  A->ops->mult                = MatMult_SeqBAIJKokkos;
  A->ops->multadd             = MatMultAdd_SeqBAIJKokkos;
  A->ops->multtranspose       = MatMultTranspose_SeqBAIJKokkos;
  A->ops->multtransposeadd    = MatMultTransposeAdd_SeqBAIJKokkos;
  A->ops->transpose           = MatTranspose_SeqBAIJKokkos;
  A->ops->setvalues           = MatSetValues_SeqBAIJKokkos;
  A->ops->setvaluesblocked    = MatSetValuesBlocked_SeqBAIJKokkos;
  A->ops->assemblyend         = MatAssemblyEnd_SeqBAIJKokkos;
  A->ops->creategraph         = MatCreateGraph_SeqBAIJKokkos;
  A->ops->getdiagonal         = MatGetDiagonal_SeqBAIJKokkos;
  A->ops->invertblockdiagonal = MatInvertBlockDiagonal_SeqBAIJKokkos;
  A->ops->diagonalscale       = MatDiagonalScale_SeqBAIJKokkos;
  A->ops->scale               = MatScale_SeqBAIJKokkos;
  A->ops->axpy                = MatAXPY_SeqBAIJKokkos;
  A->spptr                    = NULL;
  /* Override the SeqBAIJ base so the generic MatSeqBAIJSetPreallocation() preallocates our storage
     (rectangular column block size honored from the matrix's block sizes; see the adapter). */
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqBAIJSetPreallocation_C", MatSeqBAIJSetPreallocation_SeqBAIJKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqbaijkokkos_seqaij_C", MatConvert_SeqBAIJKokkos_SeqAIJ));
  /* The reverse conversion (seqaij->seqbaijkokkos) is composed on seqaij matrices in
     MatCreate_SeqAIJ() and removed in MatDestroy_SeqAIJ() (the canonical PETSc pattern,
     mirroring MATSEQAIJKOKKOS), so it is not composed here. */
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqbaijkokkos_seqbaijkokkos_C", MatProductSetFromOptions_SeqBAIJKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}
