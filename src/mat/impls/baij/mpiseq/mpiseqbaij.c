#include <../src/mat/impls/baij/seq/baij.h> /*I   "petscmat.h"  I*/

typedef struct {
  Mat seqmat;
} Mat_MPISeqBAIJ;

static PetscBool MatMPISeqBAIJActive_Private = PETSC_FALSE;

/*@
  MatMPISeqBAIJActive - determines if the process is currently constructing a `MATMPISEQBAIJ` matrix so matrix arrays allocated in shared memory with `PetscShmgetAllocateArray()`

  Not Collective

  Level: developer

.seealso: [](sec_pcmpi), `PCMPI`, `PCMPIServerBegin()`, `PetscShmgetAllocateArray()`, `PetscShmgetDeallocateArray()`, `PCMPIServerActive`, `PCMPIServerInSolve`, `PCMPIServerUseShmget`
@*/
PetscBool MatMPISeqBAIJActive(void)
{
  return MatMPISeqBAIJActive_Private;
}

/* rank in the code below is known to correspond to PetscGlobalRank */

static PetscErrorCode MatMPIBAIJSetPreallocation_MPISeqBAIJ(Mat B, PetscInt bs, PetscInt d_nz, const PetscInt *d_nnz, PetscInt o_nz, const PetscInt *o_nnz)
{
  PetscMPIInt size;
  MPI_Comm    comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)B, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) PetscCall(MatSeqBAIJSetPreallocation(B, bs, d_nz, d_nnz));
  else {
    Mat_MPISeqBAIJ *b = (Mat_MPISeqBAIJ *)B->data;
    PetscMPIInt     rank;
    const void     *addr[5];
    Mat_SeqBAIJ    *seqbaij;

    PetscCheck(d_nz > 0 && o_nz > 0, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "You must provide non-default on and off process d_nz and o_nz");
    PetscCall(MatSetBlockSize(B, bs));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)B), &rank));
    if (rank == 0) {
      MatMPISeqBAIJActive_Private = PETSC_TRUE;
      PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF, bs, B->rmap->N, B->cmap->N, d_nz + o_nz, NULL, &b->seqmat));
      MatMPISeqBAIJActive_Private = PETSC_FALSE;
      seqbaij                     = (Mat_SeqBAIJ *)b->seqmat->data;
      addr[0]                     = (const void *)seqbaij->i;
      addr[1]                     = (const void *)seqbaij->j;
      addr[2]                     = (const void *)seqbaij->a;
      addr[3]                     = (const void *)seqbaij->ilen;
      addr[4]                     = (const void *)seqbaij->imax;
    } else {
      PetscCall(MatCreate(PETSC_COMM_SELF, &b->seqmat));
      PetscCall(MatSetSizes(b->seqmat, B->rmap->N, B->cmap->N, B->rmap->N, B->cmap->N));
      PetscCall(MatSetBlockSize(b->seqmat, bs));
      PetscCall(MatSetType(b->seqmat, MATSEQBAIJ));
      seqbaij = (Mat_SeqBAIJ *)b->seqmat->data;
    }
    /* all the MATSEQBAIJ share the same array space */
    {
      void *newaddr[5];

      PetscCall(PetscShmgetMapAddresses(comm, 5, (const void **)addr, (void **)newaddr));
      if (rank > 0) {
        seqbaij->i              = (PetscInt *)newaddr[0];
        seqbaij->j              = (PetscInt *)newaddr[1];
        seqbaij->a              = (PetscScalar *)newaddr[2];
        seqbaij->ilen           = (PetscInt *)newaddr[3];
        seqbaij->imax           = (PetscInt *)newaddr[4];
        seqbaij->free_imax_ilen = PETSC_FALSE;
        seqbaij->free_a         = PETSC_FALSE;
        seqbaij->free_ij        = PETSC_FALSE;
        b->seqmat->preallocated = PETSC_TRUE;
        seqbaij->bs2            = bs * bs;
        seqbaij->mbs            = B->rmap->N / bs;
        seqbaij->nbs            = B->cmap->N / bs;
      }
      seqbaij->nonew = -2;
    }
  }
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValues_MPISeqBAIJ(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode is)
{
  Mat_MPISeqBAIJ *baij = (Mat_MPISeqBAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSetValues(baij->seqmat, m, im, n, in, v, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesBlocked_MPISeqBAIJ(Mat A, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode is)
{
  Mat_MPISeqBAIJ *baij = (Mat_MPISeqBAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSetValuesBlocked(baij->seqmat, m, im, n, in, v, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyBegin_MPISeqBAIJ(Mat A, MatAssemblyType type)
{
  Mat_MPISeqBAIJ *baij = (Mat_MPISeqBAIJ *)A->data;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCall(PetscBarrier((PetscObject)A));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  if (rank == 0) PetscCall(MatAssemblyBegin(baij->seqmat, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_MPISeqBAIJ(Mat A, MatAssemblyType type)
{
  Mat_MPISeqBAIJ *baij = (Mat_MPISeqBAIJ *)A->data;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  if (rank == 0) PetscCall(MatAssemblyEnd(baij->seqmat, type));
  PetscCall(PetscBarrier((PetscObject)A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_MPISeqBAIJ(Mat A, PetscViewer viewer)
{
  Mat_MPISeqBAIJ *baij = (Mat_MPISeqBAIJ *)A->data;
  PetscMPIInt     rank;
  PetscViewer     subviewer;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &subviewer));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  if (rank == 0) PetscCall(MatView(baij->seqmat, subviewer));
  PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &subviewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_MPISeqBAIJ(Mat A)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MATMPISEQBAIJ must be used with MatMPIBAIJSetPreallocation() or MatCreateMPISeqBAIJ()");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrices_MPISeqBAIJ(Mat A, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  Mat_MPISeqBAIJ *baij = (Mat_MPISeqBAIJ *)A->data;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  if (scall == MAT_INITIAL_MATRIX) PetscCall(PetscMalloc1(1, submat));
  if (rank > 0) {
    PetscCheck(n == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "MATMPISEQBAIJ cannot extract submatrices except on MPI rank 0");
    (*submat)[0] = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(n == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "MATMPISEQBAIJ can only extract exactly one submatrix on MPI rank 0");
  /* add check that isrow and iscol requesting the entire matrix */
  (*submat)[0] = baij->seqmat;
  if (scall == MAT_INITIAL_MATRIX) PetscCall(PetscObjectReference((PetscObject)baij->seqmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPISeqBAIJ(Mat mat)
{
  Mat_MPISeqBAIJ *baij = (Mat_MPISeqBAIJ *)mat->data;
  PetscMPIInt     rank;
  void           *newaddr[5];

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat), &rank));
  if (rank > 0 && baij->seqmat != NULL) {
    Mat_SeqBAIJ *seqbaij = (Mat_SeqBAIJ *)baij->seqmat->data;

    newaddr[0] = seqbaij->i;
    newaddr[1] = seqbaij->j;
    newaddr[2] = seqbaij->a;
    newaddr[3] = seqbaij->ilen;
    newaddr[4] = seqbaij->imax;
    PetscCall(PetscShmgetUnmapAddresses(5, (void **)newaddr));
  }
  PetscCall(MatDestroy(&baij->seqmat));
  PetscCall(PetscFree(mat->data));

  PetscCall(PetscObjectChangeTypeName((PetscObject)mat, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatMPIBAIJSetPreallocation_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATMPISEQBAIJ - "mpiseqbaij" - A matrix type that can be assembled across multiple MPI processes but whose storage is
   a `MATSEQBAIJ` on MPI rank zero. Intended for linear solves that take place on a single GPU.

   Options Database Keys:
. -mat_type mpiseqbaij - sets the matrix type to `MATMPISEQBAIJ` during a call to `MatSetFromOptions()`

   Level: intermediate

   Notes:
   Requires adequate preallocation using either `MatMPIBAIJSetPreallocation()` or `MatCreateMPISeqBAIJ()` and each MPI process can add values only to its owned rows.

   If used on one MPI process this produces a `MATSEQBAIJ` matrix

.seealso: [](ch_matrices), `Mat`, `MatCreateMPIBAIJ()`, `MatCreateSeqBAIJ()`
M*/

PETSC_EXTERN PetscErrorCode MatCreate_MPISeqBAIJ(Mat B)
{
  PetscMPIInt size, rank;
  MPI_Comm    comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)B, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (size == 1) PetscCall(MatSetType(B, MATSEQBAIJ));
  else {
    Mat_MPISeqBAIJ *b;

    PetscCall(PetscNew(&b));
    B->data                   = (void *)b;
    B->ops->destroy           = MatDestroy_MPISeqBAIJ;
    B->ops->setvalues         = MatSetValues_MPISeqBAIJ;
    B->ops->setvaluesblocked  = MatSetValuesBlocked_MPISeqBAIJ;
    B->ops->assemblybegin     = MatAssemblyBegin_MPISeqBAIJ;
    B->ops->assemblyend       = MatAssemblyEnd_MPISeqBAIJ;
    B->ops->view              = MatView_MPISeqBAIJ;
    B->ops->createsubmatrices = MatCreateSubMatrices_MPISeqBAIJ;
    B->ops->setup             = MatSetUp_MPISeqBAIJ;
    PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPISEQBAIJ));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIBAIJSetPreallocation_C", MatMPIBAIJSetPreallocation_MPISeqBAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateMPISeqBAIJ - Creates a sparse parallel matrix in `MATMPISEQBAIJ` format

  Collective

  Input Parameters:
+ comm  - MPI communicator
. bs    - size of block, the blocks are ALWAYS square. One can use `MatSetBlockSizes()` to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with `MatCreateVecs()`
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
          This value should be the same as the local size used in creating the
          y vector for the matrix-vector product $y = Ax$.
. n     - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
          This value should be the same as the local size used in creating the
          x vector for the matrix-vector product $y = Ax$.
. M     - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N     - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. d_nz  - number of nonzero blocks per block row in diagonal portion of local
          submatrix  (same for all local rows). Must be greater than zero.
. d_nnz - array containing the number of nonzero blocks in the various block rows
          of the in diagonal portion of the local (possibly different for each block
          row) or `NULL`.  If you plan to factor the matrix you must leave room for the diagonal entry
          and set it even if it is zero. Currently unused in parallel.
. o_nz  - number of nonzero blocks per block row in the off-diagonal portion of local
          submatrix (same for all local rows). Must be greater than zero.
- o_nnz - array containing the number of nonzero blocks in the various block rows of the
          off-diagonal portion of the local submatrix (possibly different for
          each block row) or `NULL`. Currently unused.

  Output Parameter:
. A - the matrix

  Level: intermediate

  Note:
  See `MATMPISEQBAIJ` and `MatCreateBAIJ()`

.seealso: `MATMPISEQBAIJ`, `Mat`, `MatCreate()`, `MatCreateSeqBAIJ()`, `MatSetValues()`, `MatMPIBAIJSetPreallocation()`
@*/
PetscErrorCode MatCreateMPISeqBAIJ(MPI_Comm comm, PetscInt bs, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatSetType(*A, MATMPISEQBAIJ));
    PetscCall(MatSetSizes(*A, m, n, M, N));
    PetscCall(MatMPIBAIJSetPreallocation(*A, bs, d_nz, d_nnz, o_nz, o_nnz));
  } else {
    PetscCall(MatSetType(*A, MATSEQBAIJ));
    PetscCall(MatSetSizes(*A, m, n, m, n));
    PetscCall(MatSeqBAIJSetPreallocation(*A, bs, d_nz, d_nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
