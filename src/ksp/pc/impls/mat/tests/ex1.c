const char help[] = "Test PCMatSetApplyOperation() and PCMatGetApplyOperation()";

#include <petscpc.h>

static PetscErrorCode TestVecEquality(Vec x, Vec y)
{
  PetscFunctionBegin;
  Vec       diff;
  PetscReal err;

  PetscCall(VecDuplicate(x, &diff));
  PetscCall(VecCopy(x, diff));
  PetscCall(VecAXPY(diff, -1.0, y));
  PetscCall(VecNorm(diff, NORM_2, &err));
  PetscCheck(err == 0.0, PetscObjectComm((PetscObject)x), PETSC_ERR_PLIB, "PC operation does not match Vec operation");
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMatEquality(Mat x, Mat y)
{
  PetscFunctionBegin;
  Mat       diff;
  PetscReal err;

  PetscCall(MatDuplicate(x, MAT_COPY_VALUES, &diff));
  PetscCall(MatAXPY(diff, -1.0, y, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(diff, NORM_FROBENIUS, &err));
  PetscCheck(err == 0.0, PetscObjectComm((PetscObject)x), PETSC_ERR_PLIB, "PC operation does not match Vec operation");
  PetscCall(MatDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm comm = PETSC_COMM_SELF;

  PetscInt n = 10;
  PetscInt k = 5;
  Vec      diag, diag_conj, diag_inv, x, x2, b;
  Mat      B, X, X2;
  PetscCall(VecCreateSeq(comm, n, &diag));
  PetscCall(VecSetRandom(diag, NULL));

  PetscCall(VecDuplicate(diag, &diag_conj));
  PetscCall(VecCopy(diag, diag_conj));
  PetscCall(VecConjugate(diag_conj));

  PetscCall(VecDuplicate(diag, &diag_inv));
  PetscCall(VecCopy(diag, diag_inv));
  PetscCall(VecReciprocal(diag_inv));

  PetscCall(VecDuplicate(diag, &b));
  PetscCall(VecSetRandom(b, NULL));

  PetscCall(VecDuplicate(diag, &x));
  PetscCall(VecDuplicate(diag, &x2));

  PetscCall(MatCreateSeqDense(comm, n, k, NULL, &B));
  PetscCall(MatSetRandom(B, NULL));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X2));

  Mat D;
  PetscCall(MatCreateVectorDiagonal(diag, &D));

  PC pc;
  PetscCall(PCCreate(comm, &pc));
  PetscCall(PCSetType(pc, PCMAT));
  PetscCall(PCSetOperators(pc, D, D));
  PetscCall(PCSetUp(pc));

  MatOperation default_op;
  PetscCall(PCMatGetApplyOperation(pc, &default_op));
  PetscCheck(default_op == MATOP_MULT, comm, PETSC_ERR_PLIB, "Default operation has changed");

  // Test setting an invalid operation
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  PetscErrorCode ierr = PCMatSetApplyOperation(pc, MATOP_SET_VALUES);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_INCOMP, comm, PETSC_ERR_PLIB, "Wrong error message for unsupported MatOperation");

  //
  // MATOP_MULT
  //
  PetscCall(PCMatSetApplyOperation(pc, MATOP_MULT));
  PetscCall(PCView(pc, NULL));

  PetscCall(PCApply(pc, b, x));
  PetscCall(VecPointwiseMult(x2, diag, b));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCApplyTranspose(pc, b, x));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCMatApply(pc, B, X));
  PetscCall(MatCopy(B, X2, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(X2, diag, NULL));
  PetscCall(TestMatEquality(X, X2));

  //
  // MATOP_MULT_TRANSPOSE
  //
  PetscCall(PCMatSetApplyOperation(pc, MATOP_MULT_TRANSPOSE));
  PetscCall(PCView(pc, NULL));

  PetscCall(PCApply(pc, b, x));
  PetscCall(VecPointwiseMult(x2, diag, b));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCApplyTranspose(pc, b, x));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCMatApply(pc, B, X));
  PetscCall(MatCopy(B, X2, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(X2, diag, NULL));
  PetscCall(TestMatEquality(X, X2));

  //
  // MATOP_MULT_HERMITIAN_TRANSPOSE
  //
  PetscCall(PCMatSetApplyOperation(pc, MATOP_MULT_HERMITIAN_TRANSPOSE));
  PetscCall(PCView(pc, NULL));

  PetscCall(PCApply(pc, b, x));
  PetscCall(VecPointwiseMult(x2, diag_conj, b));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCApplyTranspose(pc, b, x));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCMatApply(pc, B, X));
  PetscCall(MatCopy(B, X2, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(X2, diag_conj, NULL));
  PetscCall(TestMatEquality(X, X2));

  //
  // MATOP_SOLVE
  //
  PetscCall(PCMatSetApplyOperation(pc, MATOP_SOLVE));
  PetscCall(PCView(pc, NULL));

  PetscCall(PCApply(pc, b, x));
  PetscCall(VecPointwiseMult(x2, diag_inv, b));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCApplyTranspose(pc, b, x));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCMatApply(pc, B, X));
  PetscCall(MatCopy(B, X2, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(X2, diag_inv, NULL));
  PetscCall(TestMatEquality(X, X2));

  //
  // MATOP_SOLVE_TRANSPOSE
  //
  PetscCall(PCMatSetApplyOperation(pc, MATOP_SOLVE_TRANSPOSE));
  PetscCall(PCView(pc, NULL));

  PetscCall(PCApply(pc, b, x));
  PetscCall(VecPointwiseMult(x2, diag_inv, b));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCApplyTranspose(pc, b, x));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(PCMatApply(pc, B, X));
  PetscCall(MatCopy(B, X2, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(X2, diag_inv, NULL));
  PetscCall(TestMatEquality(X, X2));

  PetscCall(PCDestroy(&pc));

  PetscCall(MatDestroy(&X2));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&diag_inv));
  PetscCall(VecDestroy(&diag_conj));
  PetscCall(VecDestroy(&diag));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
