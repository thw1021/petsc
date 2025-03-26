const char help[] = "Test MATLMVMDIAGBROYDEN by comparing it to MATLMVMSYMBROYDEN for a scalar problem";

#include <petscksp.h>

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm comm = PETSC_COMM_SELF;

  Vec x, g, s, y, u, v_diag, v_sym;

  PetscCall(VecCreateSeq(comm, 1, &s));
  PetscCall(VecDuplicate(s, &y));
  PetscCall(VecDuplicate(s, &x));
  PetscCall(VecDuplicate(s, &g));
  PetscCall(VecDuplicate(s, &u));
  PetscCall(VecDuplicate(s, &v_diag));
  PetscCall(VecDuplicate(s, &v_sym));

  Mat sym, diag;

  PetscCall(MatCreateLMVMDiagBroyden(comm, 1, 1, &diag));
  PetscCall(MatCreateLMVMSymBroyden(comm, 1, 1, &sym));

  PetscInt m = 5;
  PetscCall(MatLMVMSetHistorySize(diag, m));
  PetscCall(MatLMVMSetHistorySize(sym, m));

  PetscReal phi = 0.618;
  PetscCall(MatLMVMSymBroydenSetPhi(sym, phi));
  PetscCall(MatLMVMSymBroydenSetScaleType(sym, MAT_LMVM_SYMBROYDEN_SCALE_NONE));

  PetscCall(MatSetOptionsPrefix(sym, "sym_"));
  PetscCall(MatSetFromOptions(sym));

  PetscCall(MatSetOptionsPrefix(diag, "diag_"));
  PetscCall(MatSetFromOptions(diag));

  PetscCall(MatSetUp(sym));
  PetscCall(MatSetUp(diag));

  PetscCall(MatViewFromOptions(sym, NULL, "-view"));
  PetscCall(MatViewFromOptions(diag, NULL, "-view"));

  PetscRandom rand;
  PetscCall(PetscRandomCreate(comm, &rand));
  if (PetscDefined(USE_COMPLEX)) {
    PetscScalar i = PetscSqrtScalar(-1.0);

    PetscCall(PetscRandomSetInterval(rand, -1.0 - i, 1.0 + i));
  } else {
    PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  }

  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecSetRandom(g, rand));

  PetscCall(MatLMVMUpdate(sym, x, g));
  PetscCall(MatLMVMUpdate(diag, x, g));

  for (PetscInt i = 0; i < m; i++) {
    PetscCall(VecSetRandom(s, rand));
    PetscCall(VecSetRandom(y, rand));

    PetscScalar dot;
    PetscCall(VecDot(s, y, &dot));
    PetscCall(VecScale(s, PetscAbsScalar(dot) / dot));

    PetscCall(VecAXPY(x, 1.0, s));
    PetscCall(VecAXPY(g, 1.0, y));

    PetscCall(MatLMVMUpdate(sym, x, g));
    PetscCall(MatLMVMUpdate(diag, x, g));

    PetscCall(VecSet(u, 1.0));

    PetscCall(MatMult(diag, u, v_diag));
    PetscCall(MatMult(sym, u, v_sym));

    PetscReal err, scale;
    PetscCall(VecAXPY(v_diag, -1.0, v_sym));
    PetscCall(VecNorm(v_sym, NORM_2, &scale));
    PetscCall(VecNorm(v_diag, NORM_2, &err));
    PetscCall(PetscInfo(diag, "Diagonal Broyden error %g, relative error %g\n", (double)err, (double)(err / scale)));
    PetscCheck(err <= PETSC_SMALL * scale, comm, PETSC_ERR_PLIB, "Diagonal Broyden error %g, relative error %g", (double)err, (double)(err / scale));
  }

  PetscCall(PetscRandomDestroy(&rand));

  PetscCall(MatDestroy(&sym));
  PetscCall(MatDestroy(&diag));

  PetscCall(VecDestroy(&v_sym));
  PetscCall(VecDestroy(&v_diag));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&s));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -diag_mat_lmvm_theta 0.618 -diag_mat_lmvm_sigma_hist 0 -diag_mat_lmvm_forward -sym_mat_lmvm_scale_type none

  test:
    suffix: 1
    output_file: output/ex2_0.out
    args: -diag_mat_lmvm_theta 0.618 -diag_mat_lmvm_sigma_hist 0 -diag_mat_lmvm_forward -sym_mat_lmvm_scale_type none
    args: -diag_mat_lmvm_scale_type scalar -diag_mat_lmvm_sigma_hist 1

  test:
    suffix: 2
    output_file: output/ex2_0.out
    args: -sym_mat_type lmvmbfgs -diag_mat_lmvm_theta 0.0 -diag_mat_lmvm_sigma_hist 0 -sym_mat_lmvm_scale_type none

  test:
    suffix: 3
    output_file: output/ex2_0.out
    args: -sym_mat_type lmvmbfgs -diag_mat_lmvm_theta 0.0 -diag_mat_lmvm_sigma_hist 0 -sym_mat_lmvm_scale_type none
    args: -diag_mat_lmvm_scale_type scalar -diag_mat_lmvm_sigma_hist 1

  test:
    suffix: 4
    output_file: output/ex2_0.out
    args: -sym_mat_type lmvmdfp -diag_mat_lmvm_theta 0.0 -diag_mat_lmvm_sigma_hist 0 -sym_mat_lmvm_scale_type none

  test:
    suffix: 5
    output_file: output/ex2_0.out
    args: -sym_mat_type lmvmdfp -diag_mat_lmvm_theta 0.0 -diag_mat_lmvm_sigma_hist 0 -sym_mat_lmvm_scale_type none
    args: -diag_mat_lmvm_scale_type scalar -diag_mat_lmvm_sigma_hist 1

TEST*/
