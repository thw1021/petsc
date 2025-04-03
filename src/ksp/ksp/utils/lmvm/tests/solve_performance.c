const char help[] = "Profile the performance of MATLMVM MatSolve() in a loop";

#include <petscksp.h>
#include <petscmath.h>

int main(int argc, char **argv)
{
  PetscInt      n        = 1000;
  PetscInt      n_epochs = 10;
  PetscInt      n_iters  = 10;
  PetscInt      x_n, x_N;
  Vec           x, g, dx, df, p;
  VecType       vec_type;
  PetscRandom   rand;
  PetscLogStage matsolve_loop, main_stage;
  PetscBool     is_cdiag, is_diag;
  Mat           B, J0;
  MatLMVMSymBroydenScaleType scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
  PetscBool     set;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, help, "KSP");
  PetscCall(PetscOptionsInt("-n", "Vector size", __FILE__, n, &n, NULL));
  PetscCall(PetscOptionsInt("-epochs", "Number of epochs", __FILE__, n_epochs, &n_epochs, NULL));
  PetscCall(PetscOptionsInt("-iters", "Number of iterations per epoch", __FILE__, n_iters, &n_iters, NULL));
  PetscOptionsEnd();
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(VecDuplicate(x, &dx));
  PetscCall(VecDuplicate(x, &df));
  PetscCall(VecDuplicate(x, &p));
  PetscCall(VecGetSize(x, &x_N));
  PetscCall(VecGetLocalSize(x, &x_n));
  PetscCall(VecGetType(x, &vec_type));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J0));
  PetscCall(MatSetSizes(J0, x_n, x_n, x_N, x_N));
  PetscCall(MatSetVecType(J0, vec_type));
  PetscCall(MatSetType(J0, MATCONSTANTDIAGONAL));
  PetscCall(MatSetOptionsPrefix(J0, "mat_lmvm_J0_"));
  PetscCall(MatSetFromOptions(J0));
  PetscCall(MatZeroEntries(J0));
  PetscCall(MatShift(J0, 1.0));
  PetscCall(MatCreateLMVMBFGS(PETSC_COMM_WORLD, PETSC_DETERMINE, n, &B));
  PetscCall(PetscObjectTypeCompare((PetscObject)J0, MATCONSTANTDIAGONAL, &is_cdiag));
  PetscCall(PetscObjectTypeCompare((PetscObject)J0, MATDIAGONAL, &is_diag));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatLMVMAllocate(B, x, g));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-mat_lmvm_scale_type", MatLMVMSymBroydenScaleTypes, (PetscEnum *)&scale_type, NULL));
  if (scale_type == MAT_LMVM_SYMBROYDEN_SCALE_USER) {
    if (is_cdiag) {
      PetscCall(MatLMVMSetJ0Scale(B, 1.0));
    } else if (is_diag) {
      Vec d;
      PetscCall(MatDiagonalGetDiagonal(J0, &d));
      PetscCall(MatLMVMSetJ0Diag(B, d));
      PetscCall(MatDiagonalRestoreDiagonal(J0, &d));
    }
  }
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscLogStageRegister("LMVM MatSolve Loop", &matsolve_loop));
  PetscCall(PetscLogStageGetId("Main Stage", &main_stage));
  PetscCall(PetscLogStageSetVisible(main_stage, PETSC_FALSE));
  for (PetscInt epoch = 0; epoch < n_epochs + 1; epoch++) {
    PetscScalar dot;
    PetscReal   xscale, fscale, absdot;
    PetscInt    history_size;

    PetscCall(VecSetRandom(dx, rand));
    PetscCall(VecSetRandom(df, rand));
    PetscCall(VecDot(dx, df, &dot));
    absdot = PetscAbsScalar(dot);
    PetscCall(VecSetRandom(x, rand));
    PetscCall(VecSetRandom(g, rand));
    xscale = 1.0;
    fscale = absdot / PetscRealPart(dot);
    PetscCall(MatLMVMGetHistorySize(B, &history_size));

    PetscCall(MatLMVMUpdate(B, x, g));
    for (PetscInt iter = 0; iter < history_size; iter++, xscale *= -1.0, fscale *= -1.0) {
      PetscCall(VecAXPY(x, xscale, dx));
      PetscCall(VecAXPY(g, fscale, df));
      PetscCall(MatLMVMUpdate(B, x, g));
      PetscCall(MatSolve(B, g, p));
    }
    if (epoch > 0) PetscCall(PetscLogStagePush(matsolve_loop));
    for (PetscInt iter = 0; iter < n_iters; iter++, xscale *= -1.0, fscale *= -1.0) {
      PetscCall(VecAXPY(x, xscale, dx));
      PetscCall(VecAXPY(g, fscale, df));
      PetscCall(MatLMVMUpdate(B, x, g));
      PetscCall(MatSolve(B, g, p));
    }
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (epoch > 0) PetscCall(PetscLogStagePop());
  }
  PetscCall(MatView(B, PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD)));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&J0));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&df));
  PetscCall(VecDestroy(&dx));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -mat_lmvm_scale_type none

TEST*/
