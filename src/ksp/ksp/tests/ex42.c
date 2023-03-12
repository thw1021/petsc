
static char help[] = "Solves a linear system in parallel with MINRES."

#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b; /* approx solution, RHS */
  Mat         A;    /* linear system matrix */
  KSP         ksp;  /* linear solver context */
  PC          pc;   /* preconditioner */
  PetscInt    Ii, Istart, Iend, m = 11;
  PetscScalar v = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-vv", &v, NULL));

  /* Create parallel diagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A, 1, NULL, 1, NULL));
  PetscCall(MatSeqAIJSetPreallocation(A, 1, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

  for (Ii = Istart; Ii < Iend; Ii++) {
    PetscScalar vv = (PetscReal)Ii + 1;
    PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &vv, INSERT_VALUES));
  }
  /* Make A singular or indefinite */
  Ii = m - 1; /* last diagonal entry */
  PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &v, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(x, 1.0));
  PetscCall(MatMult(A, x, b));
  PetscCall(VecSet(x, 0.0));

  /* Create linear solver context */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPMINRES));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  /* test reuse */
  PetscCall(KSPSolve(ksp, b, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* Free work space. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_converged_reason

   test:
      suffix: 2
      nsize: 3
      args: -ksp_converged_reason

TEST*/
