const char help[] = "Test MatDenseColumns{GEMV,GEMM}";

#include <petscmat.h>
#include <petsc/private/veccupmimpl.h>
#include <petscdevice_cuda.h>
#include <petscdevice_hip.h>

static PetscErrorCode CreateColumnsMat(MPI_Comm comm, PetscInt m, PetscInt n, const char prefix[], Mat *mat) {
  PetscFunctionBegin;
  PetscCall(MatCreateDense(comm, PETSC_DETERMINE, PETSC_DETERMINE, m, n, NULL, mat));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *mat, prefix));
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

int main(int argc, char **argv)
{
  PetscInt m = 1000, k = 10, a_extra = 3, b_extra = 4, c_extra = 5, d_extra = 6;
  PetscInt a_start = a_extra / 2;
  PetscInt b_start = b_extra / 2;
  PetscInt c_start = c_extra / 2;
  PetscInt d_start = c_extra / 2;
  PetscInt n_iter = 100;
  PetscInt alpha = 1.0 / 3.0;
  PetscInt beta = 1.0 / 5.0;
  PetscMemType memtype_M = PETSC_MEMTYPE_HOST;
  ExMemType exmt_M = EX_MEMTYPE_HOST;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, NULL, help, NULL);
  PetscCall(PetscOptionsInt("-m", "Global number of matrix rows", NULL, m, &m, NULL));
  PetscCall(PetscOptionsInt("-k", "Number of columns used in the update", NULL, k, &k, NULL));
  PetscCall(PetscOptionsEnum("-temp_memtype", "PetscMemType of intermediate results", NULL, ExMemTypes, exmt_M, (PetscEnum *) &exmt_M, NULL));
  PetscOptionsEnd();
#define MEMTYPECASE(SUFF) case EX_MEMTYPE_ ## SUFF: memtype_M = PETSC_MEMTYPE_ ## SUFF;break
  switch (exmt_M) {
    MEMTYPECASE(HOST);
    MEMTYPECASE(DEVICE);
    MEMTYPECASE(CUDA);
    MEMTYPECASE(HIP);
    MEMTYPECASE(NVSHMEM);
    MEMTYPECASE(SYCL);
    MEMTYPECASE(KOKKOS);
  }

  Mat A, B, C, D, D_copy1, D_copy2, D_copy3;

  PetscCall(CreateColumnsMat(comm, m, k + a_extra, "A_", &A));
  PetscCall(CreateColumnsMat(comm, m, k + b_extra, "B_", &B));
  PetscCall(CreateColumnsMat(comm, m, k + c_extra, "C_", &C));
  PetscCall(CreateColumnsMat(comm, m, k + d_extra, "D_", &D));

  PetscCall(MatDuplicate(D, MAT_DO_NOT_COPY_VALUES, &D_copy1));
  PetscCall(MatDuplicate(D, MAT_DO_NOT_COPY_VALUES, &D_copy2));
  PetscCall(MatDuplicate(D, MAT_DO_NOT_COPY_VALUES, &D_copy3));

  PetscLogStage level_1, level_2, level_3;

  PetscCall(PetscLogStageRegister("Level 1", &level_1));
  PetscCall(PetscLogStageRegister("Level 2", &level_2));
  PetscCall(PetscLogStageRegister("Level 3", &level_3));

  const PetscScalar *A_array;
  PetscMemType memtype_A;
  PetscInt ld_A;
  PetscCall(MatDenseGetLDA(A, &ld_A));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &A_array, &memtype_A));

  const PetscScalar *B_array;
  PetscMemType memtype_B;
  PetscInt ld_B;
  PetscCall(MatDenseGetLDA(B, &ld_B));
  PetscCall(MatDenseGetArrayReadAndMemType(B, &B_array, &memtype_B));
  Vec *Bs;
  PetscCall(PetscMalloc1(k, &Bs));
  for (PetscInt j = 0; j < k; j++) {
    PetscCall(MatCreateVecs(B, NULL, &Bs[j]));
    switch (memtype_B) {
    case PETSC_MEMTYPE_HOST:
      PetscCall(VecPlaceArray(Bs[j], &B_array[ld_B * (b_start + j)]));
      break;
#if defined(PETSC_HAVE_CUDA)
    case PETSC_MEMTYPE_CUDA:
      PetscCall(VecCUDAPlaceArray(Bs[j], &B_array[ld_B * (b_start + j)]));
      break;
#endif
#if defined(PETSC_HAVE_HIP)
    case PETSC_MEMTYPE_HIP:
      PetscCall(VecHIPPlaceArray(Bs[j], &B_array[ld_B * (b_start + j)]));
      break;
#endif
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unsupported memory type");
    }
  }

  PetscScalar *M;
  PetscCall(PetscMalloc1(k*k, &M));

  //
  // We are going to compute
  //
  //     D[d_start:d_end,:] = beta D[:,d_start:d_end] + alpha A[:,a_start:a_end] * B[:,b_start:b_end]' * C[:,c_start:c_end]
  //
  // Three ways
  //


  PetscLogDouble malloc;

  //
  // Level-1 approach
  //
  PetscCall(PetscLogStagePush(level_1));
  for (PetscInt i = 0; i < n_iter; i++) {
    if (i == 1) PetscCall(PetscMallocGetMaximumUsage(&malloc));
    PetscCall(MatCopy(D, D_copy1, SAME_NONZERO_PATTERN));
    for (PetscInt j = 0; j < k; j++) {
      Vec cj;
      PetscCall(MatDenseGetColumnVecRead(C, c_start + j, &cj));
      PetscCall(VecMDot(cj, k, Bs, &M[j * k]));
      PetscCall(MatDenseRestoreColumnVecRead(C, c_start + j, &cj));
    }
    for (PetscInt j = 0; j < k; j++) {
      Vec dj;
      PetscCall(MatDenseGetColumnVec(D_copy1, d_start + j, &dj));
      PetscCall(VecScale(dj, beta));
      for (PetscInt l = 0; l < k; l++) {
        Vec al;
        PetscCall(MatDenseGetColumnVecRead(A, a_start + l, &al));
        PetscCall(VecAXPY(dj, alpha * M[j*k + l], al));
        PetscCall(MatDenseRestoreColumnVecRead(A, a_start + l, &al));
      }
      PetscCall(MatDenseRestoreColumnVec(D_copy1, d_start + j, &dj));
    }
  }
  PetscLogDouble malloc_1;
  PetscCall(PetscMallocGetMaximumUsage(&malloc_1));
  PetscCall(PetscLogStagePop());
  malloc_1 -= malloc;

  for (PetscInt j = 0; j < k; j++) {
    PetscCall(VecResetArray(Bs[j]));
    PetscCall(VecDestroy(&Bs[j]));
  }
  PetscCall(PetscFree(Bs));
  PetscCall(MatDenseRestoreArrayReadAndMemType(B, &B_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &A_array));

  PetscCall(PetscFree(M));

  switch(memtype_M) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(PetscMalloc1(k * k, &M));
    break;
#if defined(PETSC_HAVE_CUDA)
  case PETSC_MEMTYPE_CUDA:
    PetscCallCUDA(cudaMalloc((void **)&M, k * k * sizeof(*M)));
    break;
#endif
#if defined(PETSC_HAVE_HIP)
  case PETSC_MEMTYPE_HIP:
    PetscCallHIP(hipMalloc((void **)&M, k * k * sizeof(*M)));
    break;
#endif
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unsupported memory type");
  }

  //
  // Level-2 approach
  //
  PetscCall(PetscLogStagePush(level_2));
  for (PetscInt i = 0; i < n_iter; i++) {
    if (i == 1) PetscCall(PetscMallocGetMaximumUsage(&malloc));
    PetscCall(MatCopy(D, D_copy2, SAME_NONZERO_PATTERN));
    for (PetscInt j = 0; j < k; j++) {
      Vec cj;
      PetscCall(MatDenseGetColumnVecRead(C, c_start + j, &cj));
      PetscCall(MatDenseColumnsGEMVHermitianTranspose(1.0, B, b_start, b_start+k, cj, 0.0, &M[j*k], 1, PETSC_MEMTYPE_HOST));
      PetscCall(MatDenseRestoreColumnVecRead(C, c_start + j, &cj));
    }
    for (PetscInt j = 0; j < k; j++) {
      Vec dj;
      PetscCall(MatDenseGetColumnVec(D_copy2, d_start + j, &dj));
      PetscCall(MatDenseColumnsGEMV(alpha, A, a_start, a_start+k, &M[j*k], 1, PETSC_MEMTYPE_HOST, beta, dj));
      PetscCall(MatDenseRestoreColumnVec(D_copy2, d_start + j, &dj));
    }
  }
  PetscLogDouble malloc_2;
  PetscCall(PetscMallocGetMaximumUsage(&malloc_2));
  PetscCall(PetscLogStagePop());
  malloc_2 -= malloc;

  //
  // Level-3 approach
  //
  PetscCall(PetscLogStagePush(level_3));
  for (PetscInt i = 0; i < n_iter; i++) {
    if (i == 1) PetscCall(PetscMallocGetMaximumUsage(&malloc));
    PetscCall(MatCopy(D, D_copy3, SAME_NONZERO_PATTERN));
    PetscCall(MatDenseColumnsGEMMHermitianTranspose(1.0, B, b_start, b_start+k, C, c_start, c_start+k, 0.0, M, k, PETSC_MEMTYPE_HOST));
    PetscCall(MatDenseColumnsGEMM(alpha, A, a_start, a_start+k, M, k, PETSC_MEMTYPE_HOST, beta, D_copy3, d_start, d_start+k));
  }
  PetscLogDouble malloc_3;
  PetscCall(PetscMallocGetMaximumUsage(&malloc_3));
  PetscCall(PetscLogStagePop());
  malloc_3 -= malloc;

  switch(memtype_M) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(PetscFree(M));
    break;
#if defined(PETSC_HAVE_CUDA)
  case PETSC_MEMTYPE_CUDA:
    PetscCallCUDA(cudaFree(M));
    break;
#endif
#if defined(PETSC_HAVE_HIP)
  case PETSC_MEMTYPE_HIP:
    PetscCallHIP(hipFree(M));
    break;
#endif
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unsupported memory type");
  }

  // compute differences
  PetscReal err_12, err_13;
  PetscCall(MatAXPY(D_copy2, -1.0, D_copy1, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(D_copy2, NORM_INFINITY, &err_12));
  PetscCall(MatAXPY(D_copy3, -1.0, D_copy1, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(D_copy3, NORM_INFINITY, &err_13));

  PetscCheck(err_12 <= PETSC_SMALL, comm, PETSC_ERR_PLIB, "Level 2 Error %g", (double) err_12);
  PetscCheck(err_13 <= PETSC_SMALL, comm, PETSC_ERR_PLIB, "Level 3 Error %g", (double) err_13);

  if (malloc_1 > 0) PetscCall(PetscPrintf(comm, "Malloc level 1 %g\n", (double) malloc_1));
  if (malloc_2 > 0) PetscCall(PetscPrintf(comm, "Malloc level 2 %g\n", (double) malloc_2));
  if (malloc_3 > 0) PetscCall(PetscPrintf(comm, "Malloc level 3 %g\n", (double) malloc_3));

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

  test:
    suffix: 0

  test:
    nsize: 2
    suffix: 1

  test:
    suffix: cuda
    args: -A_mat_type densecuda -B_mat_type densecuda -C_mat_type densecuda -D_mat_type densecuda -temp_memtype cuda

TEST*/
