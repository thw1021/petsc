const char help[] = "Test MATDENSEFROMVECTYPE";

#include <petscmat.h>

int main(int argc, char **argv)
{
  PetscInt  N = 20;
  Mat       A1, A2;
  Vec       v;
  VecType   vec_type;
  MatType   mat_type1, mat_type2;
  PetscBool same_type;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(VecSetSizes(v, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(v));
  PetscCall(VecGetType(v, &vec_type));
  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vec_type, PETSC_DECIDE, PETSC_DECIDE, N, N, PETSC_DECIDE, NULL, &A1));
  PetscCall(MatGetType(A1, &mat_type1));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A2));
  PetscCall(MatSetSizes(A2, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetVecType(A2, vec_type));
  PetscCall(MatSetType(A2, MATDENSEFROMVECTYPE));
  PetscCall(MatGetType(A2, &mat_type2));
  PetscCall(PetscStrcmp(mat_type1, mat_type2, &same_type));
  PetscCheck(same_type, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MATDENSEFROMVECTYPE does not match MatCreateDenseFromVecType()");
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&A1));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -vec_type standard
    output_file: output/ex1.out
    test:
      suffix: seq
      nsize: 1
    test:
      suffix: mpi
      nsize: 2

  testset:
    requires: cuda
    output_file: output/ex1.out
    args: -vec_type cuda
    test:
      suffix: seqcuda
      nsize: 1
    test:
      suffix: mpicuda
      nsize: 2

TEST*/
