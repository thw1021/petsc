#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/

typedef struct {
  Mat          A;                   /* the MATSEQAIJ matrix that lives on MPI rank 0 and holds all the matrix data */
  PetscBool    shmem;               /* the arrays of A are in shared memory so all ranks write their values directly into them */
  PetscInt     nz;                  /* total number of unique nonzeros in A, significant only on rank 0 */
  PetscCount   n;                   /* number of COO entries this rank passed to MatSetPreallocationCOO() */
  PetscCount   Atot;                /* number of this rank's COO entries with nonnegative indices */
  PetscInt     annz;                /* number of unique nonzeros this rank contributes to A */
  PetscInt     nzstart;             /* offset of this rank's unique nonzeros in the arrays of A */
  PetscCount  *jmap;                /* [annz+1] perm[jmap[q]..jmap[q+1]) give the COO entries forming this rank's q-th unique nonzero */
  PetscCount  *perm;                /* [Atot] permutation from sorting this rank's COO entries by row and then by column */
  PetscScalar *Aa;                  /* address of the numerical values of A usable on this rank when shmem is true */
  PetscScalar *va;                  /* [annz] send buffer of this rank's unique nonzero values when shmem is false */
  PetscScalar *w;                   /* [nz] receive buffer on rank 0 for all the unique nonzero values when shmem is false */
  PetscMPIInt *recvcounts, *displs; /* [size] number and offset of each rank's unique nonzeros, on rank 0 */
  PetscBool    preallocated;        /* MatSetPreallocationCOO_MPISeqAIJ() has been called and not reset, used to prevent changing shmem after preallocation */
} Mat_MPISeqAIJ;

static PetscErrorCode MatResetCOO_MPISeqAIJ(Mat mat)
{
  Mat_MPISeqAIJ *mseq = (Mat_MPISeqAIJ *)mat->data;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (mseq->shmem) PetscCall(PetscShmgetUnmapAddresses(1, (void **)&mseq->Aa));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat), &rank));
  if (rank == 0) PetscCall(PetscShmgetDeallocateArray((void **)&mseq->Aa));
  PetscCall(PetscFree(mseq->jmap));
  PetscCall(PetscFree(mseq->perm));
  PetscCall(PetscFree(mseq->va));
  PetscCall(PetscFree(mseq->w));
  PetscCall(PetscFree2(mseq->recvcounts, mseq->displs));
  PetscCall(MatDestroy(&mseq->A));
  mseq->preallocated = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPreallocationCOO_MPISeqAIJ(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_MPISeqAIJ  *mseq = (Mat_MPISeqAIJ *)mat->data;
  Mat_SeqAIJ     *seqaij;
  MPI_Comm        comm;
  PetscInt        M, N, m, rstart, rend, row, iprev, annz, nz = 0, nzstart = 0;
  PetscInt       *i = coo_i, *j = coo_j;
  PetscInt       *rowcnt, *aj;
  PetscInt       *Ai = NULL, *Aj = NULL;
  PetscScalar    *Aa = NULL;
  PetscCount      k, p, q, nneg;
  PetscCount     *perm, *jmap;
  PetscMPIInt     rank, size, mannz, mm, mdispl = 0;
  PetscMPIInt    *rowrecv = NULL, *rowdispl = NULL;
  PetscBool       isorted;
  const PetscInt *range;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatResetCOO_MPISeqAIJ(mat));
  mseq->preallocated = PETSC_TRUE;
  PetscCall(MatGetSize(mat, &M, &N));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  m = rend - rstart;

  PetscCall(PetscMalloc1(coo_n, &perm));

  /* Ignore entries with negative row or column indices; at the same time, check that all entries are in this rank's rows and if i[] is already sorted */
  isorted = PETSC_TRUE;
  iprev   = PETSC_INT_MIN;
  for (k = 0; k < coo_n; k++) {
    if (j[k] < 0) i[k] = -1;
    PetscCheck(i[k] < 0 || (i[k] >= rstart && i[k] < rend), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "COO row index %" PetscInt_FMT " is not owned by this MPI process; MATMPISEQAIJ does not support off-process entries", i[k]);
    PetscCheck(i[k] == -1 || j[k] < N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "COO column index %" PetscInt_FMT " is >= the matrix column size %" PetscInt_FMT, j[k], N);
    if (isorted) {
      if (i[k] < iprev) isorted = PETSC_FALSE;
      else iprev = i[k];
    }
    perm[k] = k;
  }

  /* Sort by row if not already */
  if (!isorted) PetscCall(PetscSortIntWithIntCountArrayPair(coo_n, i, j, perm));

  /* Advance k to the first entry with a nonnegative row index */
  for (k = 0; k < coo_n; k++)
    if (i[k] >= 0) break;
  nneg = k;

  PetscCall(PetscMalloc1(coo_n - nneg + 1, &jmap)); /* +1 to make a CSR-like data structure. jmap[q] originally is the number of repeats of the q-th nonzero */
  PetscCall(PetscMalloc1(coo_n - nneg, &aj));       /* This rank has at most coo_n-nneg unique nonzeros */
  PetscCall(PetscCalloc1(m, &rowcnt));              /* Number of unique nonzeros in each of this rank's rows */
  jmap++;                                           /* Inc jmap by 1 for convenience */

  /* In each row, sort by column, then unique the column indices to get the row length */
  q = 0; /* q-th unique nonzero of this rank, with q starting from 0 */
  while (k < coo_n) {
    PetscBool  strictly_sorted = PETSC_TRUE; // this row is strictly sorted?
    PetscInt   jprev           = PETSC_INT_MIN;
    PetscCount start           = k, end;

    row = i[k];
    while (k < coo_n && i[k] == row) {
      if (strictly_sorted) {
        if (j[k] <= jprev) strictly_sorted = PETSC_FALSE;
        else jprev = j[k];
      }
      k++;
    }
    end = k;

    // sort by columns in a row. perm[] indicates their original order
    if (!strictly_sorted) PetscCall(PetscSortIntWithCountArray(end - start, j + start, perm + start));

    if (strictly_sorted) { // fast path to set aj[], jmap[], rowcnt[] and q
      for (p = start; p < end; p++, q++) {
        aj[q]   = j[p];
        jmap[q] = 1;
      }
      PetscCall(PetscIntCast(end - start, &rowcnt[row - rstart]));
    } else {
      aj[q]                = j[start]; /* Log the first nonzero in this row */
      jmap[q]              = 1;        /* Number of repeats of this nonzero entry */
      rowcnt[row - rstart] = 1;

      for (p = start + 1; p < end; p++) { /* Scan remaining nonzeros in this row */
        if (j[p] != j[p - 1]) {           /* Meet a new nonzero */
          q++;
          jmap[q] = 1;
          aj[q]   = j[p];
          rowcnt[row - rstart]++;
        } else {
          jmap[q]++;
        }
      }
      q++; /* Move to next row and thus next unique nonzero */
    }
  }
  PetscCall(PetscIntCast(q, &annz));

  jmap--; // Back to the beginning of jmap[]
  jmap[0] = 0;
  for (k = 0; k < annz; k++) jmap[k + 1] += jmap[k];

  if (annz < coo_n - nneg) { /* Reallocate with the actual number of unique nonzeros */
    PetscCount *jmap_new;

    PetscCall(PetscMalloc1(annz + 1, &jmap_new));
    PetscCall(PetscArraycpy(jmap_new, jmap, annz + 1));
    PetscCall(PetscFree(jmap));
    jmap = jmap_new;
  }

  if (nneg) { /* Discard heading entries with negative indices in perm[], as we'll access it from index 0 in MatSetValuesCOO() */
    PetscCount *perm_new;

    PetscCall(PetscMalloc1(coo_n - nneg, &perm_new));
    PetscCall(PetscArraycpy(perm_new, perm + nneg, coo_n - nneg));
    PetscCall(PetscFree(perm));
    perm = perm_new;
  }

  /* Determine each rank's offset in the arrays of A and the counts needed to gather all contributions on rank 0 */
  if (rank == 0) PetscCall(PetscMalloc2(size, &mseq->recvcounts, size, &mseq->displs));
  PetscCall(PetscMPIIntCast(annz, &mannz));
  PetscCallMPI(MPI_Gather(&mannz, 1, MPI_INT, mseq->recvcounts, 1, MPI_INT, 0, comm));
  if (rank == 0) { /* accumulate in PetscInt so an overflow of the PetscMPIInt displacements is caught by PetscMPIIntCast() */
    for (PetscMPIInt r = 0; r < size; r++) {
      PetscCall(PetscMPIIntCast(nz, &mseq->displs[r]));
      nz += mseq->recvcounts[r];
    }
  }
  PetscCallMPI(MPI_Scatter(mseq->displs, 1, MPI_INT, &mdispl, 1, MPI_INT, 0, comm));
  nzstart = mdispl;

  if (rank == 0) {
    PetscCall(PetscMalloc1(M + 1, &Ai));
    PetscCall(PetscMalloc1(nz, &Aj));
    PetscCall(PetscShmgetAllocateArray(mseq->shmem, nz, sizeof(PetscScalar), (void **)&Aa));
    PetscCall(PetscArrayzero(Ai, M + 1));
    PetscCall(PetscArrayzero(Aa, nz));
    PetscCall(PetscMalloc2(size, &rowrecv, size, &rowdispl));
    PetscCall(MatGetOwnershipRanges(mat, &range));
    for (PetscMPIInt r = 0; r < size; r++) {
      PetscCall(PetscMPIIntCast(range[r + 1] - range[r], &rowrecv[r]));
      PetscCall(PetscMPIIntCast(range[r], &rowdispl[r]));
    }
  }

  /* Gather the row counts and column indices of each rank's contribution on rank 0 */
  PetscCall(PetscMPIIntCast(m, &mm));
  PetscCallMPI(MPI_Gatherv(rowcnt, mm, MPIU_INT, PetscSafePointerPlusOffset(Ai, 1), rowrecv, rowdispl, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Gatherv(aj, mannz, MPIU_INT, Aj, mseq->recvcounts, mseq->displs, MPIU_INT, 0, comm));
  PetscCall(PetscFree2(rowrecv, rowdispl));
  PetscCall(PetscFree(rowcnt));
  PetscCall(PetscFree(aj));

  if (rank == 0) {
    for (row = 0; row < M; row++) Ai[row + 1] += Ai[row];
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, M, N, Ai, Aj, Aa, &mseq->A));
    seqaij          = (Mat_SeqAIJ *)mseq->A->data;
    seqaij->free_ij = PETSC_TRUE; /* give mseq->A ownership of Ai, Aj */
  }

  if (mseq->shmem) {
    PetscCall(PetscShmgetMapAddresses(comm, 1, (const void **)&Aa, (void **)&mseq->Aa));
  } else {
    mseq->Aa = Aa;
    PetscCall(PetscMalloc1(annz, &mseq->va));
  }

  mseq->nz      = nz;
  mseq->n       = coo_n;
  mseq->Atot    = coo_n - nneg;
  mseq->annz    = annz;
  mseq->nzstart = nzstart;
  mseq->jmap    = jmap;
  mseq->perm    = perm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_MPISeqAIJ(Mat mat, const PetscScalar v[], InsertMode imode)
{
  Mat_MPISeqAIJ *mseq = (Mat_MPISeqAIJ *)mat->data;
  MPI_Comm       comm;
  PetscCount     q, p;
  PetscCount    *jmap = mseq->jmap, *perm = mseq->perm;
  PetscMPIInt    rank, mannz;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (mseq->shmem) { /* Each rank writes its values directly into the shared memory arrays of A */
    PetscScalar *Aa = NULL;

    PetscCallMPI(MPI_Barrier(comm)); /* Ensure other ranks are not setting new values while rank 0 is still using the old values */
    if (rank == 0) PetscCall(MatSeqAIJGetArray(mseq->A, &Aa));
    else Aa = mseq->Aa + mseq->nzstart;
    for (q = 0; q < mseq->annz; q++) {
      PetscScalar sum = 0.0;

      for (p = jmap[q]; p < jmap[q + 1]; p++) sum += v[perm[p]];
      Aa[q] = (imode == INSERT_VALUES ? 0.0 : Aa[q]) + sum;
    }
    if (rank == 0) PetscCall(MatSeqAIJRestoreArray(mseq->A, &Aa));
    PetscCallMPI(MPI_Barrier(comm)); /* Ensure rank 0 does not use A before all ranks have written their values */
  } else {                           /* Gather the values on rank 0 with a single MPI_Gatherv() */
    PetscScalar *Aa = NULL;

    if (rank == 0) PetscCall(MatSeqAIJGetArray(mseq->A, &Aa));
    for (q = 0; q < mseq->annz; q++) {
      PetscScalar sum = 0.0;

      for (p = jmap[q]; p < jmap[q + 1]; p++) sum += v[perm[p]];
      mseq->va[q] = sum;
    }
    PetscCall(PetscMPIIntCast(mseq->annz, &mannz));
    if (imode == INSERT_VALUES) PetscCallMPI(MPI_Gatherv(mseq->va, mannz, MPIU_SCALAR, Aa, mseq->recvcounts, mseq->displs, MPIU_SCALAR, 0, comm));
    else {
      if (rank == 0 && mseq->w == NULL) PetscCall(PetscMalloc1(mseq->nz, &mseq->w));
      PetscCallMPI(MPI_Gatherv(mseq->va, mannz, MPIU_SCALAR, mseq->w, mseq->recvcounts, mseq->displs, MPIU_SCALAR, 0, comm));
      if (rank == 0)
        for (q = 0; q < mseq->nz; q++) Aa[q] += mseq->w[q];
    }
    if (rank == 0) PetscCall(MatSeqAIJRestoreArray(mseq->A, &Aa));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMPISeqAIJGetMat_MPISeqAIJ(Mat mat, Mat *A)
{
  Mat_MPISeqAIJ *mseq = (Mat_MPISeqAIJ *)mat->data;

  PetscFunctionBegin;
  *A = mseq->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMPISeqAIJGetMat - returns the `MATSEQAIJ` matrix that lives on MPI rank 0 of a `MATMPISEQAIJ` matrix

  Not Collective

  Input Parameter:
. mat - the `MATMPISEQAIJ` matrix

  Output Parameter:
. A - the `MATSEQAIJ` matrix on MPI rank 0, `NULL` on all other MPI processes

  Level: intermediate

  Note:
  The returned matrix is valid until `mat` is destroyed or its preallocation is changed with another call
  to `MatSetPreallocationCOO()`; it should not be destroyed by the caller.

.seealso: [](ch_matrices), `Mat`, `MATMPISEQAIJ`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`, `MatMPISeqAIJSetUseShmGet()`
@*/
PetscErrorCode MatMPISeqAIJGetMat(Mat mat, Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscAssertPointer(A, 2);
  PetscUseMethod(mat, "MatMPISeqAIJGetMat_C", (Mat, Mat *), (mat, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMPISeqAIJSetUseShmGet_MPISeqAIJ(Mat mat, PetscBool use)
{
  Mat_MPISeqAIJ *mseq = (Mat_MPISeqAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCheck(PetscDefined(HAVE_SHMGET) || !use, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "This system does not support Unix shared memory");
  PetscCheck(mseq->shmem == use || mseq->preallocated == PETSC_FALSE, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Cannot change shared memory usage after MatSetPreallocationCOO() but before MatReset()");
  mseq->shmem = use;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMPISeqAIJSetUseShmGet - Tells the `MATMPISEQAIJ` to use Unix shared memory in its internal communication

  Logically Collective

  Input Parameters:
+ mat - the `MATMPISEQAIJ` matrix
- use - `PETSC_TRUE` indicates use shared memory, the default if available

  Level: intermediate

  Note:
  Using Unix shared memory is faster

.seealso: [](ch_matrices), `Mat`, `MATMPISEQAIJ`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`, `MatMPISeqAIJGetMat()`
@*/
PetscErrorCode MatMPISeqAIJSetUseShmGet(Mat mat, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscUseMethod(mat, "MatMPISeqAIJSetUseShmGet_C", (Mat, PetscBool), (mat, use));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPISeqAIJ(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(MatResetCOO_MPISeqAIJ(mat));
  PetscCall(PetscFree(mat->data));
  PetscCall(PetscObjectChangeTypeName((PetscObject)mat, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatSetValuesCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatMPISeqAIJGetMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatMPISeqAIJSetUseShmGet_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_MPISeqAIJ(Mat A, PetscOptionItems PetscOptionsObject)
{
  Mat_MPISeqAIJ *mseq = (Mat_MPISeqAIJ *)A->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPISeqAIJ options");
  PetscCall(PetscOptionsBool("-mat_mpiseqaij_use_shmget", "determines if Unix shared memory is used within this object", "MATMPISEQAIJ", mseq->shmem, &mseq->shmem, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATMPISEQAIJ - MATMPISEQAIJ = "mpiseqaij" - A matrix type that provides a very efficient mechanism for multiple
  MPI processes to build a sequential (`MATSEQAIJ`) matrix that lives on MPI rank 0, for example, a matrix that will
  eventually be solved on a single GPU.

  Options Database Key:
. -mat_mpiseqaij_use_shmget (true|false) - use Unix shared memory for communication within the `Mat`, on by default

  Level: intermediate

  Notes:
  The matrix is shared by all the MPI processes of the communicator it is created on, but all its data is stored on
  MPI rank 0 in a `MATSEQAIJ` matrix that may be obtained with `MatMPISeqAIJGetMat()`.

  This functionality can be useful when using a CPU connected to a single GPU. The application code can build the
  matrix in parallel with MPI but then hand it off to be solved on the single GPU. To support this model completely additional
  code would be needed to construct the sequential matrix as a GPU matrix.

  The only way to set values in this matrix type is with `MatSetPreallocationCOO()` and `MatSetValuesCOO()`,
  with the constraint that each MPI process may only contribute entries in its
  own rows, that is, there is no stashing and communication of off-process matrix entries.

  If Unix shared memory is used, the default, see `MatMPISeqAIJSetUseShmGet()`, the memory for the matrix data is allocated
  with `PetscShmgetAllocateArray()` and each MPI process inserts its values with `MatSetValuesCOO()` directly
  into the rank 0 matrix without any copies or MPI communication of the numerical values. Otherwise the values are
  gathered onto MPI rank 0 with a single `MPI_Gatherv()`.

  Almost none of the standard PETSc matrix operations are provided by this matrix type; use the matrix obtained with
  `MatMPISeqAIJGetMat()` on MPI rank 0 to compute with the matrix.

  This matrix must reside on an MPI communicator whose MPI rank 0 process is the same process as the MPI rank 0 process of `PETSC_COMM_WORLD`.

  Developer Note:
  The `MatSetPreallocationCOO()` implementation could be optimized by passing the matrix row column lengths and indices to the MPI rank 0 process
  via shared memory instead of using MPI gather operations. This optimization is less important if the same matrix non-structure
  is reused many times.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MATSEQAIJ`, `MatMPISeqAIJGetMat()`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`,
          `MatMPISeqAIJSetUseShmGet()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_MPISeqAIJ(Mat B)
{
  Mat_MPISeqAIJ *mseq;
  MPI_Group      world_group, comm_group;
  PetscMPIInt    world_rank_of_comm_zero = -1, rank, rank_in_comm_zero = 0;

  PetscFunctionBegin;
  /* verify the first MPI process in B's communicator is also the first in PETSC_COMM_WORLD */

  // Only rank 0 of the given communicator needs to perform the check
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)B), &rank));
  if (rank == 0) {
    PetscCallMPI(MPI_Comm_group(PETSC_COMM_WORLD, &world_group));
    PetscCallMPI(MPI_Comm_group(PetscObjectComm((PetscObject)B), &comm_group));
    PetscCallMPI(MPI_Group_translate_ranks(comm_group, 1, &rank_in_comm_zero, world_group, &world_rank_of_comm_zero));
    PetscCallMPI(MPI_Group_free(&world_group));
    PetscCallMPI(MPI_Group_free(&comm_group));
    PetscCheck(world_rank_of_comm_zero == 0, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "The zeroth rank of the MPI communicator in the matrix must be the zeroth rank in PETSC_COMM_WORLD");
  }

  PetscCall(PetscNew(&mseq));
  B->data     = (void *)mseq;
  mseq->shmem = PetscDefined(HAVE_SHMGET) ? PETSC_TRUE : PETSC_FALSE;

  B->ops->destroy        = MatDestroy_MPISeqAIJ;
  B->ops->setfromoptions = MatSetFromOptions_MPISeqAIJ;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_MPISeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetValuesCOO_C", MatSetValuesCOO_MPISeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPISeqAIJGetMat_C", MatMPISeqAIJGetMat_MPISeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPISeqAIJSetUseShmGet_C", MatMPISeqAIJSetUseShmGet_MPISeqAIJ));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPISEQAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}
