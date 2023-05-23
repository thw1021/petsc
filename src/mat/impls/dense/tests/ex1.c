const char help[] = "Test MatDenseColumns{GEMV,GEMM}";

#include <petscmat.h>
#include <petsc/private/veccupmimpl.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/deviceimpl.h>
#include <petscdevice.h>
#include <petscdevice_cuda.h>
#include <petscdevice_hip.h>
#include <petsclog.h>

static PetscErrorCode CreateColumnsMat(MPI_Comm comm, PetscInt m, PetscInt n, const char prefix[], Mat *mat)
{
  PetscFunctionBegin;
  PetscCall(MatCreateDense(comm, PETSC_DETERMINE, PETSC_DETERMINE, m, n, NULL, mat));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*mat, prefix));
  PetscCall(MatSetFromOptions(*mat));
  PetscCall(MatSetRandom(*mat, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef enum {
  EX_MEMTYPE_HOST,
  EX_MEMTYPE_DEVICE,
  EX_MEMTYPE_CUDA,
  EX_MEMTYPE_HIP,
  EX_MEMTYPE_NVSHMEM,
  EX_MEMTYPE_SYCL,
  EX_MEMTYPE_KOKKOS,
} ExMemType;

const char *const ExMemTypes[] = {"host", "device", "cuda", "hip", "nvshmem", "sycl", "kokkos", "PETSC_MEMTYPE_", NULL};

static PetscErrorCode VecPlaceArrayMemType(Vec v, const PetscScalar *array, PetscMemType memtype)
{
  PetscFunctionBegin;
  switch (memtype) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(VecPlaceArray(v, array));
    break;
  case PETSC_MEMTYPE_CUDA:
#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAPlaceArray(v, array));
#endif
    break;
  case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
    PetscCall(VecHIPPlaceArray(v, array));
#endif
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_PLIB, "Memory type unsupported for array placement");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecResetArrayMemType(Vec v, PetscMemType memtype)
{
  PetscFunctionBegin;
  switch (memtype) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(VecResetArray(v));
    break;
  case PETSC_MEMTYPE_CUDA:
#if defined(PETSC_HAVE_CUDA)
    PetscCall(VecCUDAResetArray(v));
#endif
    break;
  case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
    PetscCall(VecHIPResetArray(v));
#endif
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_PLIB, "Memory type unsupported for array placement");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestLevel1(PetscInt k_rows, PetscInt k_cols, PetscInt lda, Mat A, PetscInt a_start, Mat B, PetscInt b_start, Mat C, PetscInt c_start, Mat D, PetscInt d_start, PetscScalar alpha, PetscScalar beta, Mat D_copy, PetscInt n_iter)
{
  PetscFunctionBegin;

  MatType   C_type, D_type;
  PetscBool B_matches_C, A_matches_D;

  PetscCall(MatGetType(C, &C_type));
  PetscCall(MatGetType(D, &D_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, D_type, &A_matches_D));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, C_type, &B_matches_C));
  if (A_matches_D) {
    PetscCall(PetscObjectReference((PetscObject)A));
  } else {
    PetscCall(MatConvert(A, D_type, MAT_INITIAL_MATRIX, &A));
  }

  if (B_matches_C) {
    PetscCall(PetscObjectReference((PetscObject)B));
  } else {
    PetscCall(MatConvert(B, C_type, MAT_INITIAL_MATRIX, &B));
  }

  const PetscScalar *A_array;
  PetscMemType       memtype_A;
  PetscInt           ld_A;
  PetscCall(MatDenseGetLDA(A, &ld_A));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &A_array, &memtype_A));

  const PetscScalar *B_array;
  PetscMemType       memtype_B;
  PetscInt           ld_B;
  PetscCall(MatDenseGetLDA(B, &ld_B));
  PetscCall(MatDenseGetArrayReadAndMemType(B, &B_array, &memtype_B));

  // Place A's and B's columns into separate vectors,
  // that way the performance measurements are
  // over the same data
  Vec *As, *Bs;
  PetscCall(PetscMalloc2(k_rows, &As, k_cols, &Bs));
  for (PetscInt j = 0; j < k_rows; j++) {
    PetscCall(MatCreateVecs(A, NULL, &As[j]));
    PetscCall(MatCreateVecs(B, NULL, &Bs[j]));
    PetscCall(VecPlaceArrayMemType(As[j], &A_array[ld_A * (a_start + j)], memtype_A));
    PetscCall(VecPlaceArrayMemType(Bs[j], &B_array[ld_B * (b_start + j)], memtype_B));
  }

  PetscScalar *M;
  PetscCall(PetscMalloc1(lda * k_cols, &M));
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  for (PetscInt i = 0; i < lda * k_cols; i++) M[i] = rank;
  PetscLogStage level_1;
  PetscCall(MatCopy(D, D_copy, SAME_NONZERO_PATTERN));
  PetscCall(PetscLogStageRegister("Level 1", &level_1));
  for (size_t trip = 0; trip < 2; trip++) { // repeat each stage twice so timings avoid initialization times
    if (trip) PetscCall(PetscLogStagePush(level_1));
    for (PetscInt i = 0; i < n_iter; i++) {
      for (PetscInt j = 0; j < k_cols; j++) {
        Vec cj;
        PetscCall(MatDenseGetColumnVecRead(C, c_start + j, &cj));
        PetscCall(VecMDot(cj, k_rows, Bs, &M[j * lda]));
        PetscCall(MatDenseRestoreColumnVecRead(C, c_start + j, &cj));
      }
      for (PetscInt j = 0; j < k_rows; j++)
        for (PetscInt l = 0; l < k_cols; l++) M[j + l * lda] *= alpha;
      for (PetscInt j = 0; j < k_cols; j++) {
        Vec dj;
        PetscCall(MatDenseGetColumnVec(D_copy, d_start + j, &dj));
        PetscCall(VecScale(dj, beta));
        PetscCall(VecMAXPY(dj, k_rows, &M[j * lda], As));
        PetscCall(MatDenseRestoreColumnVec(D_copy, d_start + j, &dj));
      }
    }
    if (trip) PetscCall(PetscLogStagePop());
  }
  for (PetscInt j = 0; j < k_cols; j++) {
    for (PetscInt i = k_rows; i < lda; i++) { PetscCheck(M[i + j * lda] == rank, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Buffer modified outside of boundaries"); }
  }
  PetscCall(PetscFree(M));

  for (PetscInt j = 0; j < k_rows; j++) {
    PetscCall(VecResetArrayMemType(As[j], memtype_A));
    PetscCall(VecDestroy(&As[j]));
    PetscCall(VecResetArrayMemType(Bs[j], memtype_B));
    PetscCall(VecDestroy(&Bs[j]));
  }
  PetscCall(PetscFree2(As, Bs));
  PetscCall(MatDenseRestoreArrayReadAndMemType(B, &B_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &A_array));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestLevel2(PetscDeviceContext dctx, PetscInt k_rows, PetscInt k_cols, PetscInt lda, Mat A, PetscInt a_start, Mat B, PetscInt b_start, Mat C, PetscInt c_start, Mat D, PetscInt d_start, PetscScalar alpha, PetscScalar beta, Mat D_copy, PetscInt n_iter, PetscMemType memtype_M, PetscBool report_host_memory)
{
  PetscFunctionBegin;
  PetscScalar *M, *M_host;
  PetscMPIInt  rank;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, lda * k_cols, &M_host));
  for (PetscInt i = 0; i < lda * k_cols; i++) M_host[i] = rank;
  PetscCall(PetscDeviceMalloc(dctx, memtype_M, lda * k_cols, &M));
  PetscCall(PetscDeviceArrayCopy(dctx, M, M_host, lda * k_cols));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscInt malloc_current;

  PetscInt      malloc_2 = 0;
  PetscLogStage level_2;
  PetscCall(MatCopy(D, D_copy, SAME_NONZERO_PATTERN));
  PetscCall(PetscLogStageRegister("Level 2", &level_2));
  for (size_t trip = 0; trip < 2; trip++) {
    if (trip) PetscCall(PetscLogStagePush(level_2));
    for (PetscInt i = 0; i < n_iter; i++) {
      PetscCall(PetscMallocDebugGetCount(&malloc_current));
      for (PetscInt j = 0; j < k_cols; j++) {
        Vec cj;
        PetscCall(MatDenseGetColumnVecRead(C, c_start + j, &cj));
        PetscCall(MatDenseColumnsGEMVHermitianTranspose_Private(dctx, 1.0, B, b_start, b_start + k_rows, cj, 0.0, &M[j * lda], 1, memtype_M));
        PetscCall(MatDenseRestoreColumnVecRead(C, c_start + j, &cj));
      }
      for (PetscInt j = 0; j < k_cols; j++) {
        Vec dj;
        PetscCall(MatDenseGetColumnVec(D_copy, d_start + j, &dj));
        PetscCall(MatDenseColumnsGEMV_Private(dctx, alpha, A, a_start, a_start + k_rows, &M[j * lda], 1, memtype_M, beta, dj));
        PetscCall(MatDenseRestoreColumnVec(D_copy, d_start + j, &dj));
      }
    }
    PetscCall(PetscMallocDebugGetCount(&malloc_2));
    malloc_2 -= malloc_current;
    if (trip) PetscCall(PetscLogStagePop());
  }
  PetscCall(PetscDeviceArrayCopy(dctx, M_host, M, lda * k_cols));
  for (PetscInt j = 0; j < k_cols; j++) {
    for (PetscInt i = k_rows; i < lda; i++) { PetscCheck(M[i + j * lda] == rank, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Buffer modified outside of boundaries"); }
  }
  PetscCall(PetscDeviceFree(dctx, M));
  PetscCall(PetscDeviceFree(dctx, M_host));
  if (report_host_memory) {
    if (malloc_2 > 0) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "Malloc level 2 %" PetscInt_FMT "\n", malloc_2));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestLevel3(PetscDeviceContext dctx, PetscInt k_rows, PetscInt k_cols, PetscInt lda, Mat A, PetscInt a_start, Mat B, PetscInt b_start, Mat C, PetscInt c_start, Mat D, PetscInt d_start, PetscScalar alpha, PetscScalar beta, Mat D_copy, PetscInt n_iter, PetscMemType memtype_M, PetscBool report_host_memory)
{
  PetscFunctionBegin;
  PetscScalar *M, *M_host;
  PetscMPIInt  rank;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, lda * k_cols, &M_host));
  for (PetscInt i = 0; i < lda * k_cols; i++) M_host[i] = rank;
  PetscCall(PetscDeviceRegisterMemory(M_host, PETSC_MEMTYPE_HOST, lda * k_cols * sizeof(PetscScalar)));
  PetscCall(PetscDeviceMalloc(dctx, memtype_M, lda * k_cols, &M));
  PetscCall(PetscDeviceArrayCopy(dctx, M, M_host, lda * k_cols));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscInt malloc_current;

  PetscInt      malloc_3 = 0;
  PetscLogStage level_3;
  PetscCall(MatCopy(D, D_copy, SAME_NONZERO_PATTERN));
  PetscCall(PetscLogStageRegister("Level 3", &level_3));
  for (size_t trip = 0; trip < 2; trip++) {
    if (trip) PetscCall(PetscLogStagePush(level_3));
    PetscCall(PetscMallocDebugGetCount(&malloc_current));
    for (PetscInt i = 0; i < n_iter; i++) {
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(dctx, 1.0, B, b_start, b_start + k_rows, C, c_start, c_start + k_cols, 0.0, M, lda, memtype_M));
      PetscCall(MatDenseColumnsGEMM_Private(dctx, alpha, A, a_start, a_start + k_rows, M, lda, memtype_M, beta, D_copy, d_start, d_start + k_cols));
    }
    PetscCall(PetscMallocDebugGetCount(&malloc_3));
    malloc_3 -= malloc_current;
    if (trip) PetscCall(PetscLogStagePop());
  }
  PetscCall(PetscDeviceArrayCopy(dctx, M_host, M, lda * k_cols));
  for (PetscInt j = 0; j < k_cols; j++) {
    for (PetscInt i = k_rows; i < lda; i++) { PetscCheck(M[i + j * lda] == rank, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Buffer modified outside of boundaries"); }
  }
  PetscCall(PetscDeviceFree(dctx, M));
  PetscCall(PetscDeviceFree(dctx, M_host));
  if (report_host_memory) {
    if (malloc_3 > 0) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "Malloc level 3 %" PetscInt_FMT "\n", malloc_3));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt     m = 1000, k_rows = 10, k_cols = 13, a_extra = 3, b_extra = 4, c_extra = 5, d_extra = 6;
  PetscInt     lda                = -1;
  PetscInt     a_start            = a_extra / 2;
  PetscInt     b_start            = b_extra / 2;
  PetscInt     c_start            = c_extra / 2;
  PetscInt     d_start            = d_extra / 2;
  PetscInt     n_iter             = 100;
  PetscScalar  alpha              = 1.0 / 3.0;
  PetscScalar  beta               = 1.0 / 5.0;
  PetscMemType memtype_M          = PETSC_MEMTYPE_HOST;
  ExMemType    exmt_M             = EX_MEMTYPE_HOST;
  PetscBool    report_host_memory = PETSC_FALSE;
  PetscBool    report_memcpy      = PETSC_FALSE;
  PetscBool    explicit_dctx      = PETSC_FALSE;
  Mat          A, B, C, D, D_copy1, D_copy2, D_copy3;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, NULL, help, NULL);
  PetscCall(PetscOptionsInt("-m", "Global number of matrix rows", NULL, m, &m, NULL));
  PetscCall(PetscOptionsInt("-k_cols", "Number of columns used in the update", NULL, k_cols, &k_cols, NULL));
  PetscCall(PetscOptionsInt("-k_rows", "Number of rows used in the update", NULL, k_rows, &k_rows, NULL));
  PetscCall(PetscOptionsInt("-lda", "Leading dimension of scalar arrays (<0 to match k_rows)", NULL, lda, &lda, NULL));
  PetscCall(PetscOptionsInt("-n_iter", "Number of iterations in each stage", NULL, n_iter, &n_iter, NULL));
  PetscCall(PetscOptionsEnum("-temp_memtype", "PetscMemType of intermediate results", NULL, ExMemTypes, (PetscEnum)exmt_M, (PetscEnum *)&exmt_M, NULL));
  PetscCall(PetscOptionsBool("-report_host_memory", "Report host memory allocations that happen in each approach", NULL, report_host_memory, &report_host_memory, NULL));
  PetscCall(PetscOptionsBool("-report_memcpy", "Report host <-> device memcpys in each approach", NULL, report_memcpy, &report_memcpy, NULL));
  PetscCall(PetscOptionsBool("-explicit_dctx", "Pass explicit PetscDeviceContext to tests", NULL, explicit_dctx, &explicit_dctx, NULL));
  PetscOptionsEnd();

  if (lda < 0) lda = k_rows;

#define MEMTYPECASE(SUFF) \
  case EX_MEMTYPE_##SUFF: \
    memtype_M = PETSC_MEMTYPE_##SUFF; \
    break
  switch (exmt_M) {
    MEMTYPECASE(HOST);
    MEMTYPECASE(DEVICE);
    MEMTYPECASE(CUDA);
    MEMTYPECASE(HIP);
    MEMTYPECASE(NVSHMEM);
    MEMTYPECASE(SYCL);
    MEMTYPECASE(KOKKOS);
  }

  PetscCall(CreateColumnsMat(comm, m, k_rows + a_extra, "A_", &A));
  PetscCall(CreateColumnsMat(comm, m, k_rows + b_extra, "B_", &B));
  PetscCall(CreateColumnsMat(comm, m, k_cols + c_extra, "C_", &C));
  PetscCall(CreateColumnsMat(comm, m, k_cols + d_extra, "D_", &D));

  PetscCall(MatDuplicate(D, MAT_DO_NOT_COPY_VALUES, &D_copy1));
  PetscCall(MatDuplicate(D, MAT_DO_NOT_COPY_VALUES, &D_copy2));
  PetscCall(MatDuplicate(D, MAT_DO_NOT_COPY_VALUES, &D_copy3));

  PetscDeviceContext dctx = NULL;
  if (explicit_dctx) PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  //
  // We are going to compute
  //
  //     D[d_start:d_end,:] = beta D[:,d_start:d_end] + alpha A[:,a_start:a_end] * B[:,b_start:b_end]' * C[:,c_start:c_end]
  //
  // Three ways
  //

  //
  // Level-1 approach: VecMDot and VecmAXPY
  //
  PetscCall(TestLevel1(k_rows, k_cols, lda, A, a_start, B, b_start, C, c_start, D, d_start, alpha, beta, D_copy1, n_iter));

  //
  // Level-2 approach: MatDenseColumnsGEMVHermitianTranspose() and MatDenseColumnsGEMV()
  //
  PetscCall(TestLevel2(dctx, k_rows, k_cols, lda, A, a_start, B, b_start, C, c_start, D, d_start, alpha, beta, D_copy2, n_iter, memtype_M, report_host_memory));

  //
  // Level-3 approach: MatDenseColumnsGEMMHermitianTranspose() and MatDenseColumnsGEMM()
  //
  PetscCall(TestLevel3(dctx, k_rows, k_cols, lda, A, a_start, B, b_start, C, c_start, D, d_start, alpha, beta, D_copy3, n_iter, memtype_M, report_host_memory));

  // compute differences
  PetscReal err_12, err_13;
  PetscCall(MatAXPY(D_copy2, -1.0, D_copy1, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(D_copy2, NORM_INFINITY, &err_12));
  PetscCall(MatAXPY(D_copy3, -1.0, D_copy1, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(D_copy3, NORM_INFINITY, &err_13));

  PetscCheck(err_12 <= PetscMax(k_cols, k_rows) * m * PETSC_SMALL, comm, PETSC_ERR_PLIB, "Level 2 Error %g", (double)err_12);
  PetscCheck(err_13 <= PetscMax(k_cols, k_rows) * m * PETSC_SMALL, comm, PETSC_ERR_PLIB, "Level 3 Error %g", (double)err_13);

  PetscCall(MatDestroy(&D_copy3));
  PetscCall(MatDestroy(&D_copy2));
  PetscCall(MatDestroy(&D_copy1));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Verify that host based implementations have no mallocs, should dispatch just to BLAS [+ MPI_Allreduce without allocating an additional buffer ]
  test:
    suffix: 0
    nsize: {{1 2}}
    args: -report_host_memory -malloc_debug -explicit_dctx {{0 1}}

  # TODO: how to verify that there are no device mallocs()?
  # test:
  #   suffix: cuda
  #   args: -report_host_memory -malloc_debug -A_mat_type densecuda -B_mat_type densecuda -C_mat_type densecuda -D_mat_type densecuda -temp_memtype cuda

  # Use logging to verify no host <-> device memory transfer during kernels
  test:
    suffix: cuda_log
    requires: cuda
    args: -report_host_memory -malloc_debug -A_mat_type densecuda -B_mat_type densecuda -C_mat_type densecuda -D_mat_type densecuda -temp_memtype cuda -log_view -explicit_dctx {{0 1}}
    filter: grep "MatDenseColsGEM" | awk "{print \$1, \$23, \$24, \$25, \$26, \$27;}"

  # Use logging to verify no host <-> device memory transfer during kernels (if gpu aware mpi is used)
  test:
    nsize: 2
    suffix: cuda_log_mpi
    output_file: output/ex1_cuda_log.out
    requires: cuda defined(PETSC_HAVE_MPI_GPU_AWARE)
    args: -report_host_memory -malloc_debug -A_mat_type densecuda -B_mat_type densecuda -C_mat_type densecuda -D_mat_type densecuda -temp_memtype cuda -log_view -explicit_dctx {{0 1}}
    filter: grep "MatDenseColsGEM" | awk "{print \$1, \$23, \$24, \$25, \$26, \$27;}"

  ## Tests that verify correctness, not performance

  # GEMMH (host, host, device), GEMM (host, device, host)
  test:
    nsize: 2
    suffix: HHHHD
    output_file: output/ex1_0.out
    requires: cuda
    args: -n_iter 2 -temp_memtype cuda -explicit_dctx {{0 1}}

  # GEMMH (host, device, host), GEMM (host, host, device)
  test:
    nsize: 2
    suffix: HHDDH
    output_file: output/ex1_0.out
    requires: cuda
    args: -n_iter 2 -C_mat_type densecuda -D_mat_type densecuda -explicit_dctx {{0 1}}

  # GEMMH (host, device, device), GEMM (host, device, device)
  test:
    nsize: 2
    suffix: HHDDD
    output_file: output/ex1_0.out
    requires: cuda
    args: -n_iter 2 -C_mat_type densecuda -D_mat_type densecuda -temp_memtype cuda -explicit_dctx {{0 1}}

  # GEMMH (device, host, host), GEMM (device, host, host)
  test:
    nsize: 2
    suffix: DDHHH
    output_file: output/ex1_0.out
    requires: cuda
    args: -n_iter 2 -A_mat_type densecuda -B_mat_type densecuda -explicit_dctx {{0 1}}

  # GEMMH (device, host, device), GEMM (device, device, host)
  test:
    nsize: 2
    suffix: DDHHD
    output_file: output/ex1_0.out
    requires: cuda
    args: -n_iter 2 -A_mat_type densecuda -B_mat_type densecuda -temp_memtype cuda -explicit_dctx {{0 1}}

  # GEMMH (device, device, host), GEMM (device, host, device)
  test:
    nsize: 2
    suffix: DDDDH
    output_file: output/ex1_0.out
    requires: cuda
    args: -n_iter 2 -A_mat_type densecuda -B_mat_type densecuda -C_mat_type densecuda -D_mat_type densecuda -temp_memtype cuda -explicit_dctx {{0 1}}

  # Verify that routines are respecting the boundaries of the array if lda is largers than k_rows
  test:
    suffix: lda_test
    nsize: 2
    output_file: output/ex1_0.out
    args: -lda 20 -explicit_dctx {{0 1}}

  # TODO: nvhsmem tests?
TEST*/
