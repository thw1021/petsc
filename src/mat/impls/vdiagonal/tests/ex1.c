const char help[] = "Test MATVECTORDIAGONAL";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Vec      a, a2, b, b2, c, c2;
  Mat      A, B;
  PetscInt n = 10;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  MPI_Comm comm = PETSC_COMM_SELF;
  PetscCall(VecCreateSeq(comm, n, &a));
  PetscCall(VecDuplicate(a, &b));
  PetscCall(VecDuplicate(a, &c));
  PetscRandom rand;

  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(VecSetRandom(a, rand));
  PetscCall(VecSetRandom(b, rand));

  PetscCall(VecDuplicate(a, &a2));
  PetscCall(VecCopy(a, a2));
  PetscCall(VecDuplicate(b, &b2));
  PetscCall(VecCopy(b, b2));
  PetscCall(VecDuplicate(c, &c2));

  PetscCall(MatCreateVectorDiagonal(a2, &A));
  PetscCall(MatCreateVectorDiagonal(b2, &B));
  PetscCall(VecDestroy(&a2));
  PetscCall(VecDestroy(&b2));

  PetscCall(VecDuplicate(a, &a2));
  PetscCall(VecDuplicate(b, &b2));

  PetscCall(MatAXPY(A, 0.5, B, SAME_NONZERO_PATTERN));
  PetscCall(VecAXPY(a, 0.5, b));

  PetscReal mat_norm, vec_norm;
  PetscCall(VecNorm(a, NORM_2, &vec_norm));
  PetscCall(MatNorm(A, NORM_FROBENIUS, &mat_norm));
  PetscCheck(vec_norm == mat_norm, comm, PETSC_ERR_PLIB, "Norms don't match");

  // For diagonal matrix, all operator norms are the max norm of the vector
  PetscCall(VecNorm(a, NORM_INFINITY, &vec_norm));
  PetscCall(MatNorm(A, NORM_INFINITY, &mat_norm));
  PetscCheck(vec_norm == mat_norm, comm, PETSC_ERR_PLIB, "Norms don't match");
  PetscCall(MatNorm(A, NORM_1, &mat_norm));
  PetscCheck(vec_norm == mat_norm, comm, PETSC_ERR_PLIB, "Norms don't match");

  PetscCall(VecPointwiseMult(c, b, a));
  PetscCall(MatMult(A, b, c2));
  PetscCall(VecAXPY(c2, -1.0, c));
  PetscCall(VecNorm(c2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatMult is not like VecPointwiseMultiply");

  PetscCall(VecPointwiseMult(c, b, a));
  PetscCall(VecAXPY(c, 1.0, a));
  PetscCall(MatMultAdd(A, b, a, c2));
  PetscCall(VecAXPY(c2, -1.0, c));
  PetscCall(VecNorm(c2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatMultAdd gave unexpected value");

  PetscCall(VecPointwiseDivide(c, b, a));
  PetscCall(MatSolve(A, b, c2));
  PetscCall(VecAXPY(c2, -1.0, c));
  PetscCall(VecNorm(c2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatMult is not like VecPointwiseMultiply");

  Mat A_dup;
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &A_dup));
  PetscCall(MatDestroy(&A_dup));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_dup));
  PetscCall(MatGetDiagonal(A_dup, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatDuplicate with MAT_COPY_VALUES did not make a duplicate vector");
  PetscCall(MatDestroy(&A_dup));

  PetscCall(MatShift(A, 1.5));
  PetscCall(VecShift(a, 1.5));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatShift gave different result from VecShift");

  PetscCall(MatScale(A, 0.75));
  PetscCall(VecScale(a, 0.75));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatScale gave different result from VecScale");

  PetscCall(VecPointwiseMult(a, a, b));
  PetscCall(MatDiagonalScale(A, b, NULL));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatDiagonalScale gave unexpected result");

  PetscCall(VecPointwiseMult(a, a, b));
  PetscCall(MatDiagonalScale(A, NULL, b));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatDiagonalScale gave unexpected result");

  PetscCall(VecCopy(b, a));
  PetscCall(MatDiagonalSet(A, b, INSERT_VALUES));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatDiagonalSet gave unexpected result");

  PetscCall(VecSetRandom(a, rand));
  PetscCall(VecSetRandom(b, rand));
  PetscCall(MatDiagonalSet(A, a, INSERT_VALUES));
  PetscCall(VecAXPY(a, 1.0, b));
  PetscCall(MatDiagonalSet(A, b, ADD_VALUES));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatDiagonalSet gave unexpected result");

  PetscCall(VecSetRandom(a, rand));
  PetscCall(VecSetRandom(b, rand));
  PetscCall(MatDiagonalSet(A, a, INSERT_VALUES));
  PetscCall(VecPointwiseMax(a, a, b));
  PetscCall(MatDiagonalSet(A, b, MAX_VALUES));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatDiagonalSet gave unexpected result");

  PetscCall(VecSetRandom(a, rand));
  PetscCall(VecSetRandom(b, rand));
  PetscCall(MatDiagonalSet(A, a, INSERT_VALUES));
  PetscCall(VecPointwiseMin(a, a, b));
  PetscCall(MatDiagonalSet(A, b, MIN_VALUES));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatDiagonalSet gave unexpected result");

  PetscCall(VecSet(a, 0.5));
  PetscCall(VecSet(b, 2.0));
  PetscCall(MatVectorDiagonalSetInverse(A, b));
  PetscCall(MatGetDiagonal(A, a2));
  PetscCall(VecAXPY(a2, -1.0, a));
  PetscCall(VecNorm(a2, NORM_INFINITY, &vec_norm));
  PetscCheck(vec_norm < PETSC_SMALL, comm, PETSC_ERR_PLIB, "MatVectorDiagonalSetInverse gave unexpected result");

  PetscCall(MatZeroEntries(A));
  PetscCall(MatNorm(A, NORM_INFINITY, &mat_norm));
  PetscCheck(mat_norm == 0.0, comm, PETSC_ERR_PLIB, "MatZeroEntries gave unexpected result");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(VecDestroy(&c2));
  PetscCall(VecDestroy(&b2));
  PetscCall(VecDestroy(&a2));
  PetscCall(VecDestroy(&c));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&a));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
