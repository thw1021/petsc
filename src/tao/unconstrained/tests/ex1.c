const char help[] = "test least-squares problem created from a mapped taoterm quadratic";

#include <petsctao.h>

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  Mat         A;       // data matrix
  Mat         W;       // weight matrix
  Vec         w;       // observation vector
  Vec         b;       // observation vector
  PetscInt    m = 100; // data size
  PetscInt    n = 20;  // model size
  TaoTerm     data_term;
  PetscRandom rand;
  Tao         tao;
  PetscInt    i, j;
  PetscReal   val, density = 0.3;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsBoundedInt("-m", "data size", "", m, &m, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-n", "model size", "", n, &n, NULL, 0));
  PetscOptionsEnd();

  PetscCall(TaoCreate(comm, &tao));

  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rand));

  // create the model data, A, W and b
  //PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &A));

  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetFromOptions(A));
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      PetscCall(PetscRandomGetValue(rand, &val));
      // Optionally make it sparse: only insert some entries
      if (val < density) PetscCall(MatSetValue(A, i, j, val, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, m, &b));
  PetscCall(VecSetRandom(b, rand));
  PetscCall(VecDuplicate(b, &w));
  PetscCall(VecSetRandom(w, rand));
  PetscCall(VecAbs(w));
  PetscCall(VecShift(w, 1.0));
  PetscCall(MatCreateDiagonal(w, &W));
  PetscCall(VecDestroy(&w));

  // the model term,  (1/2) || Ax - b ||_W^2
  PetscCall(TaoTermCreateQuadratic(W, &data_term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data_term, "data_"));
  PetscCall(TaoAddTerm(tao, "data_", 3.0, data_term, b, A));
  PetscCall(TaoTermDestroy(&data_term));

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: 0
    args: -tao_monitor_short -tao_view -tao_type nls

  test:
    suffix: 1
    args: -tao_view ::ascii_info_detail -tao_type nls

TEST*/
