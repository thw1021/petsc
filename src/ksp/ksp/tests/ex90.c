static char help[] = "Tests KSPMatSolve() with KSPRICHARDSON and preconditioners that provide PCApplyRichardson().\n\
Use -set_initial_guess to fill both blocks of solutions with the same nonzero initial guess and -compare false to skip the comparison against KSPSolve().\n\
Use -nullspace to solve instead with a singular operator that has a constant null space.\n\
Use -shell to precondition with a PCSHELL that provides PCApplyRichardson() and -shell_block to make it provide PCMatApplyRichardson() as well.\n\
Use -n to set the size of the system and -nrhs to set the number of right-hand sides.\n\n";

#include <petscksp.h>

/*
   Single Jacobi sweep with a zero initial guess, x <- D^{-1} b, the context is the reciprocal of the diagonal of the operator
*/
static PetscErrorCode Apply_User(PC pc, Vec b, Vec x)
{
  Vec dinv;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &dinv));
  PetscCall(VecPointwiseMult(x, b, dinv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Jacobi sweeps, x <- x + D^{-1} (b - A x), the context is the reciprocal of the diagonal of the operator
*/
static PetscErrorCode ApplyRichardson_User(PC pc, Vec b, Vec x, Vec r, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits, PetscBool guesszero, PetscInt *its, PCRichardsonConvergedReason *reason)
{
  Mat      A;
  Vec      dinv;
  PetscInt i;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &dinv));
  PetscCall(PCGetOperators(pc, &A, NULL));
  for (i = 0; i < maxits; i++) {
    if (i == 0 && guesszero) PetscCall(VecCopy(b, r));
    else {
      PetscCall(MatMult(A, x, r));
      PetscCall(VecAYPX(r, -1.0, b));
    }
    PetscCall(VecPointwiseMult(r, r, dinv));
    if (i == 0 && guesszero) PetscCall(VecCopy(r, x));
    else PetscCall(VecAXPY(x, 1.0, r));
  }
  *its    = maxits;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Block analog of ApplyRichardson_User(), KSPRICHARDSON passes W = NULL so the work block of vectors is allocated here
*/
static PetscErrorCode MatApplyRichardson_User(PC pc, Mat B, Mat X, Mat W, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits, PetscBool guesszero, PetscInt *its, PCRichardsonConvergedReason *reason)
{
  Mat      A, R;
  Vec      dinv;
  PetscInt i;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &dinv));
  PetscCall(PCGetOperators(pc, &A, NULL));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &R));
  /* set the product A X up once so that only its numeric phase is run in the loop below */
  PetscCall(MatProductCreateWithMat(A, X, NULL, R));
  PetscCall(MatProductSetType(R, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(R));
  PetscCall(MatProductSymbolic(R));
  for (i = 0; i < maxits; i++) {
    if (i == 0 && guesszero) PetscCall(MatCopy(B, R, SAME_NONZERO_PATTERN));
    else {
      PetscCall(MatProductNumeric(R));
      PetscCall(MatAYPX(R, -1.0, B, SAME_NONZERO_PATTERN));
    }
    PetscCall(MatDiagonalScale(R, dinv, NULL));
    if (i == 0 && guesszero) PetscCall(MatCopy(R, X, SAME_NONZERO_PATTERN));
    else PetscCall(MatAXPY(X, 1.0, R, SAME_NONZERO_PATTERN));
  }
  PetscCall(MatProductClear(R));
  PetscCall(MatDestroy(&R));
  *its    = maxits;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat          A, B, X, Y;
  MatNullSpace nsp  = NULL;
  Vec          dinv = NULL, cb, cx;
  KSP          ksp;
  PC           pc;
  PetscInt     i, n = 20, nrhs = 5, m, Istart, Iend;
  PetscReal    nrm;
  PetscBool    shell = PETSC_FALSE, block = PETSC_FALSE, guess_nonzero = PETSC_FALSE, set_initial_guess = PETSC_FALSE, compare = PETSC_TRUE, nullspace = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-shell", &shell, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-shell_block", &block, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-set_initial_guess", &set_initial_guess, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-compare", &compare, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nullspace", &nullspace, NULL));

  /* tridiagonal operator, singular with a constant null space when -nullspace is used (1D Laplacian with Neumann ends) */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    if (nullspace) PetscCall(MatSetValue(A, i, i, i == 0 || i == n - 1 ? 1.0 : 2.0, INSERT_VALUES));
    else PetscCall(MatSetValue(A, i, i, 4.0, INSERT_VALUES));
    if (i > 0) PetscCall(MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(A, i, i + 1, -1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (nullspace) {
    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nsp));
    PetscCall(MatSetNullSpace(A, nsp));
  }

  /* block of right-hand sides and the two blocks of solutions to compare */
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, n, nrhs, NULL, &B));
  PetscCall(MatSetRandom(B, NULL));
  if (nsp) { /* project the null space out of each right-hand side so that the singular systems are consistent */
    for (i = 0; i < nrhs; i++) {
      PetscCall(MatDenseGetColumnVec(B, i, &cb));
      PetscCall(MatNullSpaceRemove(nsp, cb));
      PetscCall(MatDenseRestoreColumnVec(B, i, &cb));
    }
  }
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &Y));
  PetscCall(MatZeroEntries(X));
  PetscCall(MatZeroEntries(Y));
  if (set_initial_guess) { /* the same nonzero initial guess in both blocks of solutions */
    PetscCall(MatCopy(B, X, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(X, 100.0));
    PetscCall(MatCopy(X, Y, SAME_NONZERO_PATTERN));
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPRICHARDSON));
  if (shell) {
    PetscCall(MatCreateVecs(A, NULL, &dinv));
    PetscCall(MatGetDiagonal(A, dinv));
    PetscCall(VecReciprocal(dinv));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetName(pc, "Jacobi sweeps"));
    PetscCall(PCShellSetContext(pc, dinv));
    PetscCall(PCShellSetApply(pc, Apply_User));
    PetscCall(PCShellSetApplyRichardson(pc, ApplyRichardson_User));
    if (block) PetscCall(PCShellSetMatApplyRichardson(pc, MatApplyRichardson_User));
  }
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetInitialGuessNonzero(ksp, &guess_nonzero));

  /* solve with the whole block of right-hand sides at once, then one right-hand side at a time */
  PetscCall(KSPMatSolve(ksp, B, X));
  for (i = 0; i < nrhs; i++) {
    PetscCall(MatDenseGetColumnVecRead(B, i, &cb));
    /* KSPSolve() reads the column when it is used as the initial guess */
    if (guess_nonzero) PetscCall(MatDenseGetColumnVec(Y, i, &cx));
    else PetscCall(MatDenseGetColumnVecWrite(Y, i, &cx));
    PetscCall(KSPSolve(ksp, cb, cx));
    if (guess_nonzero) PetscCall(MatDenseRestoreColumnVec(Y, i, &cx));
    else PetscCall(MatDenseRestoreColumnVecWrite(Y, i, &cx));
    PetscCall(MatDenseRestoreColumnVecRead(B, i, &cb));
  }
  /* with tolerance-based stopping, block-aggregate and per-column convergence legitimately differ at the level of the relative tolerance, so the comparison is only meaningful for fixed-iteration runs */
  if (compare) {
    PetscCall(MatAXPY(Y, -1.0, X, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(Y, NORM_FROBENIUS, &nrm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "KSPMatSolve() and KSPSolve() %s\n", nrm < PETSC_SQRT_MACHINE_EPSILON ? "agree" : "disagree"));
  }

  PetscCall(MatNullSpaceDestroy(&nsp));
  PetscCall(VecDestroy(&dinv));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&Y));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/ex90.out
      nsize: {{1 2}}
      args: -ksp_type richardson -ksp_max_it 5 -ksp_norm_type none

      test:
         suffix: shell_matapplyrichardson
         args: -shell -shell_block

      test:
         suffix: shell_applyrichardson
         args: -shell

      test:
         suffix: shell_matapplyrichardson_guess
         args: -shell -shell_block -ksp_initial_guess_nonzero

      test:
         suffix: shell_applyrichardson_guess
         args: -shell -ksp_initial_guess_nonzero

      test:
         suffix: sor
         args: -pc_type sor

      test:
         suffix: jacobi
         args: -pc_type jacobi

      test:
         suffix: nullspace
         args: -nullspace -pc_type jacobi

   # KSPConvergedDefault() removes the null space of the operator from the left-preconditioned block of right-hand sides, as in the Vec path
   test:
      suffix: nullspace_guess
      nsize: {{1 2}}
      output_file: output/ex90.out
      args: -ksp_type richardson -ksp_max_it 5 -ksp_rtol 1e-50 -ksp_norm_type preconditioned -nullspace -pc_type jacobi -set_initial_guess -ksp_initial_guess_nonzero

   # the singular operator makes the Cholesky factorization fail, so the block iteration flags the whole block of solutions with MatSetInf() and stops with KSP_DIVERGED_PC_FAILED
   test:
      suffix: pc_failed
      nsize: 1
      args: -ksp_type richardson -nullspace -pc_type cholesky -pc_factor_shift_type none -compare false -ksp_converged_reason

   test:
      suffix: jacobi_guess
      args: -ksp_type richardson -pc_type jacobi -set_initial_guess -compare false -ksp_initial_guess_nonzero -ksp_norm_type {{unpreconditioned preconditioned}separate output} -ksp_rtol 1e-4 -ksp_max_it 100 -ksp_converged_reason

   test:
      suffix: jacobi_guess_batch
      args: -ksp_type richardson -pc_type jacobi -set_initial_guess -compare false -ksp_initial_guess_nonzero -ksp_norm_type unpreconditioned -ksp_rtol 1e-4 -ksp_max_it 100 -ksp_converged_reason -ksp_matsolve_batch_size 2

TEST*/
