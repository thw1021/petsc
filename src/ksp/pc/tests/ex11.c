static const char help[] = "Tests PCMG rediscretized level operator setup with pc_use_amat true and false.\n\n";

#include <petscksp.h>
#include <petscdmda.h>

typedef struct {
  PetscBool expect_use_amat;
  PetscInt  ncreate;
  PetscInt  ncompute;
} AppCtx;

static PetscErrorCode AssembleDiagonal(Mat A, PetscScalar diag)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt row = rstart; row < rend; row++) PetscCall(MatSetValue(A, row, row, diag, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateOperators(KSP ksp, Mat *A, Mat *P, void *ctx)
{
  AppCtx   *user = (AppCtx *)ctx;
  DM        dm;
  PetscBool use_amat;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &dm));
  use_amat = user->expect_use_amat;

  PetscCall(DMCreateMatrix(dm, A));
  if (use_amat) PetscCall(DMCreateMatrix(dm, P));
  else *P = *A;
  user->ncreate++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperators(KSP ksp, Mat A, Mat P, void *ctx)
{
  AppCtx   *user = (AppCtx *)ctx;
  PetscBool use_amat;

  PetscFunctionBeginUser;
  use_amat = user->expect_use_amat;
  if (use_amat) {
    PetscCheck(A != P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected distinct Amat and Pmat on rediscretized PCMG level with pc_use_amat=true");
    PetscCall(AssembleDiagonal(A, 2.0));
    PetscCall(AssembleDiagonal(P, 3.0));
  } else {
    PetscCheck(A == P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected Pmat-only operator on rediscretized PCMG level with pc_use_amat=false");
    PetscCall(AssembleDiagonal(P, 3.0));
  }
  user->ncompute++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx    user;
  DM        dm;
  KSP       ksp, cksp;
  PC        pc;
  Mat       A, P, cA, cP;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 9, 1, 1, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  user.expect_use_amat = PETSC_TRUE;
  user.ncreate         = 0;
  user.ncompute        = 0;
  PetscCall(DMKSPSetCreateOperators(dm, CreateOperators, &user));
  PetscCall(DMKSPSetComputeOperators(dm, ComputeOperators, &user));

  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(DMCreateMatrix(dm, &P));
  PetscCall(AssembleDiagonal(A, 2.0));
  PetscCall(AssembleDiagonal(P, 3.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_OPERATOR, PETSC_FALSE));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 2, NULL));
  PetscCall(KSPSetOperators(ksp, A, P));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PCGetUseAmat(pc, &user.expect_use_amat));

  PetscCall(KSPSetUp(ksp));
  PetscCheck(user.ncreate > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected PCMG to create at least one rediscretized level operator");
  PetscCheck(user.ncompute > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected PCMG to compute at least one rediscretized level operator");

  PetscCall(PCMGGetCoarseSolve(pc, &cksp));
  PetscCall(KSPGetOperators(cksp, &cA, &cP));
  if (user.expect_use_amat) PetscCheck(cA != cP, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Coarse KSP operators were not distinct with pc_use_amat=true");
  else PetscCheck(cA == cP, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Coarse KSP operators were not Pmat-only with pc_use_amat=false");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&P));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   test:
      suffix: use_amat_true
      args: -pc_use_amat true -pc_mg_galerkin none
      output_file: output/empty.out

   test:
      suffix: use_amat_false
      args: -pc_use_amat false -pc_mg_galerkin none
      output_file: output/empty.out

TEST*/
