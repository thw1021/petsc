const char help[] = "Test interactions of VECREDUNDANT with MATDENSE";

#include <petscmat.h>

static PetscErrorCode MatCreateDenseForRedundant(MPI_Comm comm, PetscInt m, PetscInt M, PetscInt N, PetscScalar *data, Mat *A)
{
  PetscMPIInt size, rank;
  PetscLayout cmap;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, rank == size - 1 ? N : 0, M, N));
  PetscCall(MatSetType(*A, MATDENSE));
  if (size > 1) {
    PetscCall(MatGetLayouts(*A, NULL, &cmap));
    cmap->redundant = PETSC_TRUE;
  }
  PetscCall(MatSeqDenseSetPreallocation(*A, data));
  PetscCall(MatMPIDenseSetPreallocation(*A, data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt N = 10;
  PetscInt M = 20;
  MPI_Comm comm;
  Vec      vec_r, vec_g, A_r;
  Mat      A;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(MatCreateDenseForRedundant(comm, PETSC_DECIDE, M, N, NULL, &A));
  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatCreateVecs(A, &vec_r, &A_r));
  PetscCall(VecSetRandom(vec_r, NULL));
  PetscCall(VecView(vec_r, PETSC_VIEWER_STDOUT_(comm)));
  PetscCall(VecCreateMPI(comm, N, PETSC_DETERMINE, &vec_g));
  {
    const PetscScalar *a;
    PetscCall(VecGetArrayRead(vec_r, &a));
    PetscCall(VecPlaceArray(vec_g, a));
    PetscCall(VecView(vec_g, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(VecResetArray(vec_g));
    PetscCall(VecRestoreArrayRead(vec_r, &a));
  }
  PetscCall(MatMult(A, vec_r, A_r));
  PetscCall(MatMultTranspose(A, A_r, vec_r));
  PetscCall(VecDestroy(&vec_g));
  PetscCall(VecDestroy(&A_r));
  PetscCall(VecDestroy(&vec_r));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2}}

TEST*/
