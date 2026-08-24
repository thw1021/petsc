static char help[] = "Tests MatDiagonalScale() for MATMFFD.\n\n";

#include <petscmat.h>

static PetscErrorCode MyF(PetscCtx ctx, Vec x, Vec y)
{
  const PetscScalar *xa;
  PetscScalar       *ya;

  PetscFunctionBegin;
  (void)ctx;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(y, &ya));
  ya[0] = 1.0e8 * (xa[0] - 2.0);
  ya[1] = xa[1] * xa[1] - 4.0;
  PetscCall(VecRestoreArray(y, &ya));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckScaledMult(Vec U, Vec a, Vec left, Vec right, PetscScalar h, Vec y)
{
  Vec       w, FU, Fw, expected;
  PetscReal norm, ynorm;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(U, &w));
  PetscCall(VecDuplicate(U, &FU));
  PetscCall(VecDuplicate(U, &Fw));
  PetscCall(VecDuplicate(U, &expected));

  if (right) PetscCall(VecPointwiseMult(w, right, a));
  else PetscCall(VecCopy(a, w));
  PetscCall(VecAYPX(w, h, U)); /* w <- U + h*(right .* a) */

  PetscCall(MyF(NULL, U, FU));
  PetscCall(MyF(NULL, w, Fw));
  PetscCall(VecAXPY(Fw, -1.0, FU));
  PetscCall(VecScale(Fw, 1.0 / h));
  if (left) PetscCall(VecPointwiseMult(expected, left, Fw));
  else PetscCall(VecCopy(Fw, expected));

  PetscCall(VecAXPY(expected, -1.0, y));
  PetscCall(VecNorm(expected, NORM_2, &norm));
  PetscCall(VecNorm(y, NORM_2, &ynorm));
  PetscCheck(norm <= 1.0e-8 * PetscMax(ynorm, 1.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMult_MFFD scaled result differs from the hand-computed L*(F(U+h*R*a)-F(U))/h value by %g", (double)norm);

  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&FU));
  PetscCall(VecDestroy(&Fw));
  PetscCall(VecDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMFFDScaling(PetscBool use_left, PetscBool use_right)
{
  Mat            A;
  Vec            U, a, y, y0, left = NULL, right = NULL, leftinv = NULL, rightinv = NULL;
  const PetscInt rows[] = {0, 1};
  PetscScalar    h;
  PetscReal      norm;

  PetscFunctionBeginUser;
  PetscCall(MatCreateMFFD(PETSC_COMM_WORLD, 2, 2, 2, 2, &A));
  PetscCall(MatCreateVecs(A, &U, NULL));
  PetscCall(VecDuplicate(U, &a));
  {
    const PetscScalar uvals[] = {1.5, 1.2};
    const PetscScalar avals[] = {2.0, 3.0};

    PetscCall(VecSetValues(U, 2, rows, uvals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(U));
    PetscCall(VecAssemblyEnd(U));
    PetscCall(VecSetValues(a, 2, rows, avals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(a));
    PetscCall(VecAssemblyEnd(a));
  }
  PetscCall(MatMFFDSetFunction(A, MyF, NULL));
  PetscCall(MatMFFDSetBase(A, U, NULL));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(VecDuplicate(U, &y0));
  PetscCall(MatMult(A, a, y0));

  if (use_left) {
    const PetscScalar lvals[] = {1.0e-8, 1.0};

    PetscCall(VecDuplicate(U, &left));
    PetscCall(VecSetValues(left, 2, rows, lvals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(left));
    PetscCall(VecAssemblyEnd(left));
  }
  if (use_right) {
    const PetscScalar rvals[] = {1.0e8, 1.0};

    PetscCall(VecDuplicate(U, &right));
    PetscCall(VecSetValues(right, 2, rows, rvals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(right));
    PetscCall(VecAssemblyEnd(right));
  }
  PetscCall(MatDiagonalScale(A, left, right));

  PetscCall(VecDuplicate(U, &y));
  PetscCall(MatMult(A, a, y));
  PetscCall(MatMFFDGetH(A, &h));
  PetscCall(CheckScaledMult(U, a, left, right, h, y));

  /* MatDiagonalScale() accumulates: scaling by the reciprocal vectors must exactly undo it. */
  if (left) {
    PetscCall(VecDuplicate(left, &leftinv));
    PetscCall(VecCopy(left, leftinv));
    PetscCall(VecReciprocal(leftinv));
  }
  if (right) {
    PetscCall(VecDuplicate(right, &rightinv));
    PetscCall(VecCopy(right, rightinv));
    PetscCall(VecReciprocal(rightinv));
  }
  PetscCall(MatDiagonalScale(A, leftinv, rightinv));
  PetscCall(MatMult(A, a, y));
  PetscCall(VecAXPY(y, -1.0, y0));
  PetscCall(VecNorm(y, NORM_2, &norm));
  PetscCheck(norm <= 1.0e-10, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Undoing MatDiagonalScale() with the reciprocal vectors did not restore the unscaled MatMult_MFFD() result, error %g", (double)norm);

  PetscCall(VecDestroy(&leftinv));
  PetscCall(VecDestroy(&rightinv));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&right));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&y0));
  PetscCall(VecDestroy(&a));
  PetscCall(VecDestroy(&U));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(TestMFFDScaling(PETSC_TRUE, PETSC_TRUE));
  PetscCall(TestMFFDScaling(PETSC_FALSE, PETSC_TRUE));
  PetscCall(TestMFFDScaling(PETSC_TRUE, PETSC_FALSE));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: !complex
    output_file: output/empty.out

TEST*/
