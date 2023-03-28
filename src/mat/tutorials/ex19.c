#include <petscdevice_cuda.h>
#include <petscmat.h>
#include <petscconf.h>
#include <assert.h>

int main(int argc, char **args)
{
  Mat         A;
  Vec         X;
  PetscInt    N = 20;
  PetscBool   iscuda = PETSC_FALSE, iship = PETSC_FALSE;
  PetscBool   optionflag, compareflag;
  char        vectypename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Creating Mat from Vec example", NULL);
  PetscCall(PetscOptionsGetString(NULL, NULL, "-vectype", vectypename, sizeof(vectypename), &optionflag));
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

  PetscCall(MatCreateDenseMatchingVec(X, PETSC_DECIDE, PETSC_DECIDE, N, N, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  MPI_Comm X_comm = PetscObjectComm((PetscObject) X);
  MPI_Comm A_comm = PetscObjectComm((PetscObject) X);
  PetscMPIInt comp;
  PetscCall(MPI_Comm_compare(X_comm, A_comm, &comp));
  PetscAssert(comp == MPI_IDENT || comp == MPI_CONGRUENT, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Failed communicator guarantee in MatCreateDenseMatchingVec()");

  PetscMemType X_memtype, A_memtype;
  const PetscScalar *array;
  PetscCall(VecGetArrayReadAndMemType(X, &array, &X_memtype));
  PetscCall(VecRestoreArrayReadAndMemType(X, &array));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &array, &A_memtype));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &array));
  PetscAssert(A_memtype == X_memtype, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Failed memtype guarantee in MatCreateDenseMatchingVec()");

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
      suffix: hip
      requires: hip
      args: -vectype hip -ex19_mat_view

   test:
      suffix: standard
      args: -vectype standard -ex19_mat_view
TEST*/
