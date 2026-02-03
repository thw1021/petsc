static char help[] = "MRE for SNESVIInactiveSet.\n\n";

#include "petscsnes.h"
#include "petscsys.h"
#include </opt/HPC/src/petsc-main/src/snes/impls/vi/rs/virsimpl.h>
PetscErrorCode computeFunction(SNES snes, Vec X, Vec F, void *ctx)
{
  PetscInt         n;
  PetscReal       *f;
  const PetscReal *x;

  PetscCall(VecGetSize(X, &n));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(F, &f));
  for (int i = 0; i < n; i++) { f[i] = x[i] - i; }
  PetscCall(VecRestoreArray(F, &f));
  PetscCall(VecRestoreArrayRead(X, &x));
  return 0;
}

PetscErrorCode computeJacobian(SNES snes, Vec X, Mat J, Mat P, void *ctx)
{
  PetscInt n;

  PetscCall(MatGetSize(J, &n, NULL));
  for (int i = 0; i < n; i++) { PetscCall(MatSetValue(J, i, i, 1., INSERT_VALUES)); }
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  return 0;
}

int main(int argc, char **argv)
{
  SNES      snes;
  Vec       X, F, Xl, Xu;
  Mat       A;
  PetscReal lb = -1, ub = 6;
  PetscInt  n = 5;
  IS        iA;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lb", &lb, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ub", &ub, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetFromOptions(A));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, PETSC_DECIDE, n));
  PetscCall(VecSetType(X, VECMPI));
  PetscCall(VecSet(X, 0.));
  PetscCall(PetscObjectSetName((PetscObject)X, "X"));
  PetscCall(VecDuplicate(X, &F));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, "vinewtonrsls"));
  PetscCall(SNESSetFunction(snes, F, computeFunction, NULL));
  PetscCall(SNESSetJacobian(snes, A, A, computeJacobian, NULL));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(VecDuplicate(X, &Xl));
  PetscCall(VecDuplicate(X, &Xu));
  PetscCall(VecSet(Xl, lb));
  PetscCall(VecSet(Xu, ub));
  PetscCall(SNESVISetVariableBounds(snes, Xl, Xu));

  PetscCall(SNESComputeFunction(snes, X, F));
  PetscCall(VecView(F, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(SNESComputeJacobian(snes, X, A, A));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(SNESSolve(snes, NULL, X));
  PetscCall(VecView(X, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(SNESVIGetActiveSetIS(snes, X, F, &iA));
  PetscCall(ISView(iA, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(ISDestroy(&iA));

  PetscCall(SNESVIGetInactiveSet(snes, &iA));
  PetscCall(ISView(iA, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(ISDestroy(&iA));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&Xl));
  PetscCall(VecDestroy(&Xu));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&F));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -lb 2.5

TEST*/
