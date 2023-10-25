const char help[] = "Profile the performance of MATLMVM MatSolve() in a loop";

#include <petscksp.h>

int main(int argc, char **argv)
{
  PetscInt      n        = 1000;
  PetscInt      n_epochs = 10;
  PetscInt      n_iters  = 10;
  Vec           x, f, dx, df, r, s;
  PetscRandom   rand;
  PetscLogStage matsolve_loop, main_stage;
  Mat           B;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, help, "KSP");
  PetscCall(PetscOptionsInt("-n", "Vector size", __FILE__, n, &n, NULL));
  PetscCall(PetscOptionsInt("-epochs", "Number of epochs", __FILE__, n_epochs, &n_epochs, NULL));
  PetscCall(PetscOptionsInt("-iters", "Number of iterations per epoch", __FILE__, n_iters, &n_iters, NULL));
  PetscOptionsEnd();
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(VecDuplicate(x, &dx));
  PetscCall(VecDuplicate(x, &df));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(VecDuplicate(x, &s));
  PetscCall(MatCreateLMVMBFGS(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &B));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatLMVMAllocate(B, x, f));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(VecSetRandom(r, rand));
  PetscCall(PetscLogStageRegister("LMVM MatSolve Loop", &matsolve_loop));
  PetscCall(PetscLogStageGetId("Main Stage", &main_stage));
  PetscCall(PetscLogStageSetVisible(main_stage, PETSC_FALSE));
  for (PetscInt epoch = 0; epoch < n_epochs; epoch++) {
    if (epoch == 1) PetscCall(PetscLogStagePush(matsolve_loop));
    PetscCall(VecZeroEntries(x));
    PetscCall(VecZeroEntries(f));
    PetscCall(MatLMVMUpdate(B, x, f));
    for (PetscInt iter = 0; iter < n_iters; iter++) {
      PetscScalar dot;
      PetscReal   absdot;

      PetscCall(VecSetRandom(dx, rand));
      PetscCall(VecSetRandom(df, rand));
      PetscCall(VecDot(dx, df, &dot));
      absdot = PetscAbsScalar(dot);
      PetscCall(VecAXPY(x, 1.0, dx));
      PetscCall(VecAXPY(f, absdot / dot, df));
      PetscCall(MatLMVMUpdate(B, x, f));
      PetscCall(MatSolve(B, r, s));
    }
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (epoch + 1 == n_epochs) PetscCall(PetscLogStagePop());
  }
  PetscCall(MatView(B, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD)));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&s));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&df));
  PetscCall(VecDestroy(&dx));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -mat_lmvm_scale_type none

TEST*/
