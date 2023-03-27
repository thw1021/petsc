#include <petscdevice_cuda.h>
#include <petscmat.h>
#include <petscconf.h>
#include <assert.h>

int main(int argc, char **args)
{
  Mat           A;
  Vec           X;
  PetscInt      N = 20, num_threads = 128;
  PetscLogEvent event;
  PetscMPIInt   rank, size;
  PetscBool     iscuda = PETSC_FALSE, iship = PETSC_FALSE;
  PetscBool     optionflag, compareflag;
  char          vectypename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_threads", &num_threads, NULL));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Creating Mat from Vec example", NULL);
  PetscCall(PetscOptionsGetString(NULL, NULL, "-vectype", vectypename, sizeof(vectypename), &optionflag));
  PetscCall(PetscLogEventRegister("GPU operator", MAT_CLASSID, &event));
  PetscOptionsEnd();

  if (optionflag) {
    PetscCall(PetscStrncmp(vectypename, "cuda", (size_t)4, &compareflag));
    if (compareflag) iscuda = PETSC_TRUE;
    PetscCall(PetscStrncmp(vectypename, "hip", (size_t)3, &compareflag));
    if (compareflag) iship = PETSC_TRUE;
    PetscCall(PetscStrncmp(vectypename, "standard", (size_t)8, &compareflag));
  }

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, PETSC_DECIDE, N));
  if (iscuda) {
    PetscCall(VecSetType(X, VECCUDA));
  } else if (iship) {
    PetscCall(VecSetType(X, VECHIP));
  } else {
    PetscCall(VecSetType(X, VECSTANDARD));
  }
  PetscCall(VecSetUp(X));

  PetscCall(MatCreateDenseFromVecType(X, PETSC_DECIDE, PETSC_DECIDE, N, N, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* test */
  PetscCall(MatViewFromOptions(A, NULL, "-ex19_mat_view"));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: cuda
      requires: cuda
      args: -vectype cuda -ex19_mat_view

   test:
      suffix: standard
      args: -vectype standard -ex19_mat_view
TEST*/
