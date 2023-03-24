#include <petscdevice_cuda.h>
#include <petscmat.h>
#include <petscconf.h>
#include <assert.h>

int main(int argc, char **args)
{
  Mat            A;
  Vec            X;
  PetscInt       N = 20, num_threads = 128;
  PetscLogEvent  event;
  PetscMPIInt    rank, size;
  PetscBool      iscuda = PETSC_FALSE, iship = PETSC_FALSE;
  PetscBool      optionflag, compareflag;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_threads", &num_threads, NULL));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-vectype", vectypename, sizeof(vectypename), &optionflag));
  PetscCall(PetscLogEventRegister("GPU operator", MAT_CLASSID, &event));

  if (optionflag) {
    PetscCall(PetscStrncmp(vectypename, "cuda", (size_t)4, &compareflag));
    if (compareflag) iscuda = PETSC_TRUE;
    PetscCall(PetscStrncmp(vectypename, "hip", (size_t)3, &compareflag));
    if (compareflag) iship = PETSC_TRUE;
  }

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X);
  PetscCall(VecSetSizes(X, PETSC_DECIDE, N));
  if (iscuda) {
    PetscCall(VecSetType(X, VECCUDA));
  } else if (iship) {
    PetscCall(VecSetType(X, VECHIP));
  } else {
    PetscCall(VecSetType(X, VECSTANDARD));
  }
  PetscCall(VecSetUp(X));
  PetscCall(PetscObjectSetName((PetscObject)X, "X_commworld"));

  PetscCall(MatCreateDenseFromVecType(X, PETSC_DECIDE, PETSC_DECIDE, N, N, NULL, &A));

  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* test */
  PestcCall(MatViewFromOptions(A, NULL, "-ex19_mat_view"));


  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&X));
  return 0;
}
