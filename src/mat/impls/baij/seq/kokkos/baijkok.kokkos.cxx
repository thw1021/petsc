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

// Forward declarations for SeqBAIJ lifecycle
PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_SeqBAIJ(Mat);

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

  A->assembled        = PETSC_TRUE;
  A->was_assembled    = PETSC_TRUE;
  nzstate             = A->nonzerostate;
  A->ass_nonzerostate = nzstate;

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
  MatMult_SeqBAIJKokkos - Compute y = A*x for a rectangular-block sparse matrix.

  This is a HOST implementation (v1). For each block-row bi of A:
    y[bi*row_bs + ii] += sum (b in [i[bi], i[bi+1])) sum (jj in [0, col_bs))
                         a[b*row_bs*col_bs + ii*col_bs + jj] * x[j[b]*col_bs + jj]

  In other words, a per-block dense GEMV (row_bsxcol_bs * col_bs) accumulates
  into row_bs output entries. A device Kokkos kernel is deferred to Phase B.
*/
static PetscErrorCode MatMult_SeqBAIJKokkos(Mat A, Vec x, Vec y)
{
  Mat_SeqBAIJKokkos   *baijkok;
  PetscInt             row_bs, col_bs, mbs, bi, block_idx, jj, ii;
  const PetscScalar   *xv;
  PetscScalar         *yv;
  MatRowMapType       *i_h;
  MatColIdxType       *j_h;
  const MatScalarType *a_h;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");

  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  /* Ensure host has current block values */
  PetscCall(MatSeqBAIJKokkosSyncHost(A));

  /* Get host views of block-CSR structure */
  i_h = baijkok->i_dual.view_host().data();
  j_h = baijkok->j_dual.view_host().data();
  a_h = baijkok->a_dual.view_host().data();

  /* Get vector array pointers and zero y before accumulation */
  PetscCall(VecGetArrayRead(x, &xv));
  PetscCall(VecGetArray(y, &yv));
  PetscCall(PetscArrayzero(yv, A->rmap->n));

  /* Block SpMV: for each block-row bi, accumulate into y[bi*row_bs : (bi+1)*row_bs] */
  for (bi = 0; bi < mbs; bi++) {
    /* Loop over blocks in row bi: [i_h[bi], i_h[bi+1]) */
    for (block_idx = i_h[bi]; block_idx < i_h[bi + 1]; block_idx++) {
      PetscInt bj = j_h[block_idx]; /* block-column index */
      /* GEMV: (row_bs x col_bs) * (col_bs) -> (row_bs)
         y[bi*row_bs + ii] += sum_jj a[block_offset + ii*col_bs + jj] * x[bj*col_bs + jj] */
      const MatScalarType *block_a = a_h + block_idx * row_bs * col_bs;

      for (ii = 0; ii < row_bs; ii++) {
        PetscScalar sum = 0.0;
        for (jj = 0; jj < col_bs; jj++) {
          sum += block_a[ii * col_bs + jj] * xv[bj * col_bs + jj];
        }
        yv[bi * row_bs + ii] += sum;
      }
    }
  }

  /* Restore vector pointers */
  PetscCall(VecRestoreArrayRead(x, &xv));
  PetscCall(VecRestoreArray(y, &yv));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultAdd_SeqBAIJKokkos - Compute z = y + A*x for a rectangular-block sparse matrix (host).
*/
static PetscErrorCode MatMultAdd_SeqBAIJKokkos(Mat A, Vec x, Vec y, Vec z)
{
  Mat_SeqBAIJKokkos   *baijkok;
  PetscInt             row_bs, col_bs, mbs, bi, block_idx, jj, ii;
  const PetscScalar   *xv;
  PetscScalar         *zv;
  MatRowMapType       *i_h;
  MatColIdxType       *j_h;
  const MatScalarType *a_h;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");
  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(MatSeqBAIJKokkosSyncHost(A));
  i_h = baijkok->i_dual.view_host().data();
  j_h = baijkok->j_dual.view_host().data();
  a_h = baijkok->a_dual.view_host().data();

  if (y != z) PetscCall(VecCopy(y, z)); /* z = y; then accumulate A*x into z */
  PetscCall(VecGetArrayRead(x, &xv));
  PetscCall(VecGetArray(z, &zv));
  for (bi = 0; bi < mbs; bi++) {
    for (block_idx = i_h[bi]; block_idx < i_h[bi + 1]; block_idx++) {
      PetscInt             bj      = j_h[block_idx];
      const MatScalarType *block_a = a_h + block_idx * row_bs * col_bs;

      for (ii = 0; ii < row_bs; ii++) {
        PetscScalar sum = 0.0;
        for (jj = 0; jj < col_bs; jj++) sum += block_a[ii * col_bs + jj] * xv[bj * col_bs + jj];
        zv[bi * row_bs + ii] += sum;
      }
    }
  }
  PetscCall(VecRestoreArrayRead(x, &xv));
  PetscCall(VecRestoreArray(z, &zv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultTranspose_SeqBAIJKokkos - Compute y = A^T*x for a rectangular-block sparse matrix (host).
  x has length mbs*row_bs (rows of A); y has length nbs*col_bs (columns of A).
*/
static PetscErrorCode MatMultTranspose_SeqBAIJKokkos(Mat A, Vec x, Vec y)
{
  Mat_SeqBAIJKokkos   *baijkok;
  PetscInt             row_bs, col_bs, mbs, bi, block_idx, jj, ii;
  const PetscScalar   *xv;
  PetscScalar         *yv;
  MatRowMapType       *i_h;
  MatColIdxType       *j_h;
  const MatScalarType *a_h;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");
  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(MatSeqBAIJKokkosSyncHost(A));
  i_h = baijkok->i_dual.view_host().data();
  j_h = baijkok->j_dual.view_host().data();
  a_h = baijkok->a_dual.view_host().data();

  PetscCall(VecGetArrayRead(x, &xv));
  PetscCall(VecZeroEntries(y));
  PetscCall(VecGetArray(y, &yv));
  /* y[bj*col_bs + jj] += sum_ii a[block, ii, jj] * x[bi*row_bs + ii] */
  for (bi = 0; bi < mbs; bi++) {
    for (block_idx = i_h[bi]; block_idx < i_h[bi + 1]; block_idx++) {
      PetscInt             bj      = j_h[block_idx];
      const MatScalarType *block_a = a_h + block_idx * row_bs * col_bs;

      for (jj = 0; jj < col_bs; jj++) {
        PetscScalar sum = 0.0;
        for (ii = 0; ii < row_bs; ii++) sum += block_a[ii * col_bs + jj] * xv[bi * row_bs + ii];
        yv[bj * col_bs + jj] += sum;
      }
    }
  }
  PetscCall(VecRestoreArrayRead(x, &xv));
  PetscCall(VecRestoreArray(y, &yv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMultTransposeAdd_SeqBAIJKokkos - Compute z = y + A^T*x for a rectangular-block sparse matrix (host).
*/
static PetscErrorCode MatMultTransposeAdd_SeqBAIJKokkos(Mat A, Vec x, Vec y, Vec z)
{
  Mat_SeqBAIJKokkos   *baijkok;
  PetscInt             row_bs, col_bs, mbs, bi, block_idx, jj, ii;
  const PetscScalar   *xv;
  PetscScalar         *zv;
  MatRowMapType       *i_h;
  MatColIdxType       *j_h;
  const MatScalarType *a_h;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix not preallocated");
  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(MatSeqBAIJKokkosSyncHost(A));
  i_h = baijkok->i_dual.view_host().data();
  j_h = baijkok->j_dual.view_host().data();
  a_h = baijkok->a_dual.view_host().data();

  if (y != z) PetscCall(VecCopy(y, z)); /* z = y; then accumulate A^T*x into z */
  PetscCall(VecGetArrayRead(x, &xv));
  PetscCall(VecGetArray(z, &zv));
  for (bi = 0; bi < mbs; bi++) {
    for (block_idx = i_h[bi]; block_idx < i_h[bi + 1]; block_idx++) {
      PetscInt             bj      = j_h[block_idx];
      const MatScalarType *block_a = a_h + block_idx * row_bs * col_bs;

      for (jj = 0; jj < col_bs; jj++) {
        PetscScalar sum = 0.0;
        for (ii = 0; ii < row_bs; ii++) sum += block_a[ii * col_bs + jj] * xv[bi * row_bs + ii];
        zv[bj * col_bs + jj] += sum;
      }
    }
  }
  PetscCall(VecRestoreArrayRead(x, &xv));
  PetscCall(VecRestoreArray(z, &zv));
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
  delete ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Forward declaration of MatTranspose_SeqBAIJKokkos_Private for use in product routines */
static PetscErrorCode MatTranspose_SeqBAIJKokkos_Private(Mat A, Mat *At);

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
static PetscErrorCode MatProductSymbolicAB_SeqBAIJKokkos_Helper(Mat C, Mat A, Mat B, MatProductCtx_SeqBAIJKokkos *pdata)
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
  MatProductNumeric_SeqBAIJKokkos_Helper - Numeric phase for AB product given explicit A and B matrices.

  Computes C = A*B using the block-CSR graph (already set by symbolic).
  Both A and B are synced to device; the result C is marked as modified on device.
*/
static PetscErrorCode MatProductNumericAB_SeqBAIJKokkos_Helper(Mat C, Mat A, Mat B)
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

  /* TeamPolicy: one team per block-row i of A (mbs_A teams) */
  using TeamPolicy = Kokkos::TeamPolicy<DefaultExecutionSpace>;
  Kokkos::parallel_for(
    "MatProductNumeric_SeqBAIJKokkos_AB", TeamPolicy(mbs_A, Kokkos::AUTO), KOKKOS_LAMBDA(const KokkosTeamMemberType &team) {
      PetscInt i = team.league_rank();

      /* Extract row ranges for this block-row of A and C */
      PetscInt a_start = a_i_d(i);
      PetscInt a_end   = a_i_d(i + 1);
      PetscInt c_start = c_i_d(i);
      PetscInt c_end   = c_i_d(i + 1);

      /* Iterate over blocks in row i of A (team parallelism deferred; serial for now) */
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (PetscInt a_p = a_start; a_p < a_end; a_p++) {
          PetscInt k = a_j_d(a_p); /* block-row of B */

          /* Pointer to block A_ik */
          MatScalarType *aval = const_cast<MatScalarType *>(a_d.data()) + a_p * rbsA * kdim;

          /* Iterate over blocks in row k of B */
          PetscInt b_start = b_i_d(k);
          PetscInt b_end   = b_i_d(k + 1);

          for (PetscInt b_q = b_start; b_q < b_end; b_q++) {
            PetscInt j = b_j_d(b_q); /* block-column of B and C */

            /* Pointer to block B_kj */
            MatScalarType *bval = const_cast<MatScalarType *>(b_d.data()) + b_q * kdim * cbsB;

            /* Binary search for (i,j) in C's row */
            PetscInt p_c = BinarySearchColumnInCRow(c_j_d.data(), c_start, c_end, j);
            if (p_c < 0) {
              Kokkos::abort("Block (i,j) not in C's sparsity pattern; symbolic phase failed");
            }

            /* Pointer to block C_ij */
            MatScalarType *cval = const_cast<MatScalarType *>(c_d.data()) + p_c * rbsA * cbsB;

            /* Dense block GEMM: C_ij += A_ik * B_kj
               All blocks are row-major: element (ii,jj) is at offset ii*col_bs+jj */
            for (PetscInt ii = 0; ii < rbsA; ii++) {
              for (PetscInt jj = 0; jj < cbsB; jj++) {
                PetscScalar sum = 0.0;
                for (PetscInt kk = 0; kk < kdim; kk++) {
                  sum += aval[ii * kdim + kk] * bval[kk * cbsB + jj];
                }
                Kokkos::atomic_add(&cval[ii * cbsB + jj], sum);
              }
            }
          }
        }
      });
    });

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
  For MATPRODUCT_AtB: refreshes the transpose stored in product->data->At by syncing
    product->A to host, rebuilding At via MatTranspose_SeqBAIJKokkos_Private with MAT_REUSE_MATRIX,
    then calls the AB numeric helper with (At, product->B).

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
    /* Numeric phase for C = A^T*B: refresh the transpose (rebuild from current A values) and
       recompute. Destroy the symbolic-phase transpose first so we do not leak it. */
    PetscCall(MatDestroy(&pdata->At));
    PetscCall(MatTranspose_SeqBAIJKokkos_Private(A, &pdata->At));
    PetscCall(MatProductNumericAB_SeqBAIJKokkos_Helper(C, pdata->At, B));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "Product type %s not supported", MatProductTypes[product->type]);
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
    /* Build the transpose of A */
    PetscCall(MatTranspose_SeqBAIJKokkos_Private(A, &pdata->At));
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
  MatProductSetFromOptions_SeqBAIJKokkos - Register product operations for MATSEQBAIJKOKKOS.

  Dispatches MATPRODUCT_AB and MATPRODUCT_AtB to the custom symbolic kernel;
  MATPRODUCT_PtAP, MATPRODUCT_RARt, and MATPRODUCT_ABC are routed to MatProductSymbolic_ABC_Basic,
  which decomposes them into two pairwise products (AB and AtB) that dispatch back here.
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
  MatGetDiagonal_SeqBAIJKokkos - Extract diagonal from a block matrix.

  For a square block matrix (row_bs == col_bs && mbs == nbs), fills vector v
  with the diagonal entries. v must have length mbs*row_bs (one scalar per row).
  Implemented on host.
*/
static PetscErrorCode MatGetDiagonal_SeqBAIJKokkos(Mat A, Vec v)
{
  Mat_SeqBAIJKokkos   *baijkok;
  const MatRowMapType *i;
  const MatColIdxType *j;
  const MatScalarType *a;
  PetscScalar         *xv;
  PetscInt             bi, row_bs, col_bs, mbs;
  PetscInt             block_pos;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected NULL baijkok");
  PetscCheck(baijkok->row_bs == baijkok->col_bs && baijkok->mbs == baijkok->nbs, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "MatGetDiagonal requires square matrix at block level");

  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;

  PetscCall(MatSeqBAIJKokkosSyncHost(A));
  i = baijkok->i_host_data();
  j = baijkok->j_host_data();
  a = baijkok->a_host_data();

  PetscCall(VecGetArray(v, &xv));

  for (bi = 0; bi < mbs; bi++) {
    /* Search for diagonal block (column == bi) in [i[bi], i[bi+1]) */
    PetscBool found = PETSC_FALSE;
    for (block_pos = i[bi]; block_pos < i[bi + 1]; block_pos++) {
      if (j[block_pos] == bi) {
        /* Found diagonal block; extract diagonal entries from this block */
        const MatScalarType *block_a = &a[block_pos * row_bs * col_bs];
        for (PetscInt ii = 0; ii < row_bs; ii++) {
          xv[bi * row_bs + ii] = block_a[ii * col_bs + ii];
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) {
      /* Diagonal block absent; set diagonal entries to 0 */
      for (PetscInt ii = 0; ii < row_bs; ii++) {
        xv[bi * row_bs + ii] = 0.0;
      }
    }
  }

  PetscCall(VecRestoreArray(v, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatDiagonalScale_SeqBAIJKokkos - Left and/or right diagonal scaling.

  Scales A as: A_ij *= (l? lv[bi*row_bs+ii] : 1) * (r? rv[bj*col_bs+jj] : 1)
  where bi, bj are block row/column indices and ii, jj are within-block indices.
  Implemented on host. Handles l and/or r being NULL (smoothing passes r=NULL).
*/
static PetscErrorCode MatDiagonalScale_SeqBAIJKokkos(Mat A, Vec l, Vec r)
{
  Mat_SeqBAIJKokkos   *baijkok;
  const MatRowMapType *i;
  const MatColIdxType *j;
  MatScalarType       *a;
  const PetscScalar   *lv = NULL, *rv = NULL;
  PetscInt             bi, bj, ii, jj, row_bs, col_bs, mbs, nbs;
  PetscInt             block_pos;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected NULL baijkok");

  row_bs = baijkok->row_bs;
  col_bs = baijkok->col_bs;
  mbs    = baijkok->mbs;
  nbs    = baijkok->nbs;

  PetscCall(MatSeqBAIJKokkosSyncHost(A));
  i = baijkok->i_host_data();
  j = baijkok->j_host_data();
  a = baijkok->a_host_data();

  if (l) {
    PetscCall(VecGetArrayRead(l, &lv));
    PetscCheck(A->rmap->n == row_bs * mbs, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Left vector size mismatch");
  }
  if (r) {
    PetscCall(VecGetArrayRead(r, &rv));
    PetscCheck(A->cmap->n == col_bs * nbs, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Right vector size mismatch");
  }

  /* Scale each block entry */
  for (bi = 0; bi < mbs; bi++) {
    for (block_pos = i[bi]; block_pos < i[bi + 1]; block_pos++) {
      bj                     = j[block_pos];
      MatScalarType *block_a = &a[block_pos * row_bs * col_bs];
      for (ii = 0; ii < row_bs; ii++) {
        for (jj = 0; jj < col_bs; jj++) {
          PetscScalar l_scale = l ? lv[bi * row_bs + ii] : 1.0;
          PetscScalar r_scale = r ? rv[bj * col_bs + jj] : 1.0;
          block_a[ii * col_bs + jj] *= l_scale * r_scale;
        }
      }
    }
  }

  if (l) PetscCall(VecRestoreArrayRead(l, &lv));
  if (r) PetscCall(VecRestoreArrayRead(r, &rv));

  baijkok->a_dual.modify_host();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatScale_SeqBAIJKokkos - Multiply all block entries by a scalar.

  Implemented on host.
*/
static PetscErrorCode MatScale_SeqBAIJKokkos(Mat A, PetscScalar a)
{
  Mat_SeqBAIJKokkos *baijkok;
  MatScalarType     *a_h;
  PetscInt           nblk_vals;

  PetscFunctionBegin;
  baijkok = static_cast<Mat_SeqBAIJKokkos *>(A->spptr);
  PetscCheck(baijkok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected NULL baijkok");

  PetscCall(MatSeqBAIJKokkosSyncHost(A));
  a_h       = baijkok->a_host_data();
  nblk_vals = static_cast<PetscInt>(baijkok->a_dual.view_host().extent(0));

  for (PetscInt k = 0; k < nblk_vals; k++) {
    a_h[k] *= a;
  }

  baijkok->a_dual.modify_host();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatAXPY_SeqBAIJKokkos - Y = Y + alpha*X.

  Implements SAME_NONZERO_PATTERN and SUBSET_NONZERO_PATTERN.
  For SUBSET: X's block sparsity is a subset of Y's; all X blocks must exist in Y.
  Implemented on host.
*/
static PetscErrorCode MatAXPY_SeqBAIJKokkos(Mat Y, PetscScalar alpha, Mat X, MatStructure str)
{
  Mat_SeqBAIJKokkos   *ykok, *xkok;
  const MatRowMapType *Xi, *Yi;
  const MatColIdxType *Xj, *Yj;
  const MatScalarType *Xa;
  MatScalarType       *Ya;
  PetscInt             bi, block_pos_x, block_pos_y, row_bs, col_bs, mbs;

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

  PetscCall(MatSeqBAIJKokkosSyncHost(Y));
  PetscCall(MatSeqBAIJKokkosSyncHost(X));

  Yi = ykok->i_host_data();
  Yj = ykok->j_host_data();
  Ya = ykok->a_host_data();

  Xi = xkok->i_host_data();
  Xj = xkok->j_host_data();
  Xa = xkok->a_host_data();

  if (str == SAME_NONZERO_PATTERN) {
    /* X and Y have the same block sparsity pattern; direct element-wise addition */
    PetscInt nblk_vals = static_cast<PetscInt>(ykok->a_dual.view_host().extent(0));
    PetscCheck(static_cast<PetscInt>(xkok->a_dual.view_host().extent(0)) == nblk_vals, PetscObjectComm((PetscObject)Y), PETSC_ERR_ARG_INCOMP, "SAME_NONZERO_PATTERN requires identical block count");
    for (PetscInt k = 0; k < nblk_vals; k++) {
      Ya[k] += alpha * Xa[k];
    }
  } else if (str == SUBSET_NONZERO_PATTERN) {
    /* X's block pattern is a subset of Y's; for each block in X, find and update the matching block in Y */
    for (bi = 0; bi < mbs; bi++) {
      for (block_pos_x = Xi[bi]; block_pos_x < Xi[bi + 1]; block_pos_x++) {
        PetscInt col_x = Xj[block_pos_x];
        /* Search for this block in Y's row bi */
        PetscBool found = PETSC_FALSE;
        for (block_pos_y = Yi[bi]; block_pos_y < Yi[bi + 1]; block_pos_y++) {
          if (Yj[block_pos_y] == col_x) {
            /* Found matching block; do dense block AXPY */
            const MatScalarType *x_block = &Xa[block_pos_x * row_bs * col_bs];
            MatScalarType       *y_block = &Ya[block_pos_y * row_bs * col_bs];
            PetscInt             sz      = row_bs * col_bs;
            for (PetscInt k = 0; k < sz; k++) {
              y_block[k] += alpha * x_block[k];
            }
            found = PETSC_TRUE;
            break;
          }
        }
        PetscCheck(found, PetscObjectComm((PetscObject)Y), PETSC_ERR_ARG_INCOMP, "MatAXPY SUBSET_NONZERO_PATTERN: block (%" PetscInt_FMT ",%" PetscInt_FMT ") in X not found in Y", bi, col_x);
      }
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)Y), PETSC_ERR_SUP, "MatAXPY with DIFFERENT_NONZERO_PATTERN not yet supported for MATSEQBAIJKOKKOS");
  }

  ykok->a_dual.modify_host();
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
  A->ops->destroy          = MatDestroy_SeqBAIJKokkos;
  A->ops->mult             = MatMult_SeqBAIJKokkos;
  A->ops->multadd          = MatMultAdd_SeqBAIJKokkos;
  A->ops->multtranspose    = MatMultTranspose_SeqBAIJKokkos;
  A->ops->multtransposeadd = MatMultTransposeAdd_SeqBAIJKokkos;
  A->ops->transpose        = MatTranspose_SeqBAIJKokkos;
  A->ops->setvalues        = MatSetValues_SeqBAIJKokkos;
  A->ops->setvaluesblocked = MatSetValuesBlocked_SeqBAIJKokkos;
  A->ops->assemblyend      = MatAssemblyEnd_SeqBAIJKokkos;
  A->ops->creategraph      = MatCreateGraph_SeqBAIJKokkos;
  A->ops->getdiagonal      = MatGetDiagonal_SeqBAIJKokkos;
  A->ops->diagonalscale    = MatDiagonalScale_SeqBAIJKokkos;
  A->ops->scale            = MatScale_SeqBAIJKokkos;
  A->ops->axpy             = MatAXPY_SeqBAIJKokkos;
  A->spptr                 = NULL;
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
