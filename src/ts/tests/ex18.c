static char help[] = "Solves a DAE with a non-trivial mass matrix. \n\n";
/*
        U * dU/dt = U + V
        U - V = 0

   than can be rewritten in implicit form
                 x[0] * xdot[0] - x[0] - x[1]
   F(t,x,xdot) =
                 x[0] - x[1]
*/

#include <petscts.h>

PetscErrorCode IFunction(TS, PetscReal, Vec, Vec, Vec, void *);
PetscErrorCode IJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);

int main(int argc, char **argv)
{
  TS  ts;
  Vec x;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, NULL));
  PetscCall(TSSetIJacobian(ts, NULL, NULL, IJacobian, NULL));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, 2, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TSSetSolution(ts, x));
  PetscCall(VecDestroy(&x));

  PetscCall(TSSetTimeStep(ts, 1.0));
  PetscCall(TSSetMaxTime(ts, PETSC_MAX_REAL));
  PetscCall(TSSetMaxSteps(ts, 3));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode IFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  const PetscScalar *xdot, *x;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayWrite(F, &f));
  f[0] = x[0] * xdot[0] - x[0] - x[1];
  f[1] = x[0] - x[1];
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayWrite(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  const PetscScalar *xdot, *x;
  PetscFunctionBeginUser;

  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArrayRead(X, &x));
  if (shift != PETSC_MAX_REAL) {
    PetscCall(MatSetValue(B, 0, 0, shift * x[0] + xdot[0] - 1.0, INSERT_VALUES));
    PetscCall(MatSetValue(B, 0, 1, -1.0, INSERT_VALUES));
    PetscCall(MatSetValue(B, 1, 0, 1.0, INSERT_VALUES));
    PetscCall(MatSetValue(B, 1, 1, -1.0, INSERT_VALUES));
  } else {
    PetscCall(MatZeroEntries(B));
    PetscCall(MatSetValue(B, 0, 0, x[0], INSERT_VALUES));
    PetscCall(MatSetValue(B, 1, 1, PETSC_SQRT_MACHINE_EPSILON, INSERT_VALUES)); /* prevent from zeropivots */
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  testset:
    args: -ts_view_solution -ts_max_steps 10 -ts_dt 0.1 -ts_view_solution -ts_adapt_type none

    test:
      output_file: output/ex18_1.out
      suffix: bdf
      args: -ts_type bdf

    test:
      output_file: output/ex18_1.out
      suffix: dirk
      args: -ts_type dirk -ts_dirk_type {{212 es212 657a es648sa 658a s659a 7510sal es7510sa 759a s7511sal 8614a 8616sal es8516sal}}

TEST*/
