static char help[] = "Tests SNESFAS restriction of explicit left and right diagonal scaling to the coarse level.\n\n";

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

static PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscScalar *x, PetscScalar *f, void *ctx)
{
  PetscInt i;

  PetscFunctionBeginUser;
  (void)ctx;
  for (i = info->xs; i < info->xs + info->xm; i++) f[i] = x[i] * x[i] - 4.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar *x, Mat J, Mat P, void *ctx)
{
  PetscInt i;

  PetscFunctionBeginUser;
  (void)ctx;
  for (i = info->xs; i < info->xs + info->xm; i++) {
    MatStencil  row = {0};
    PetscScalar v   = 2.0 * x[i];

    row.i = i;
    PetscCall(MatSetValuesStencil(P, 1, &row, 1, &row, &v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Checks that every entry of scale equals value, within a loose tolerance, confirming that FAS
   restricted the fine-level scale vector down to the coarse level without distorting a constant.
*/
static PetscErrorCode CheckConstantScale(Vec scale, PetscScalar value, const char name[])
{
  Vec       diff;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(scale, &diff));
  PetscCall(VecSet(diff, value));
  PetscCall(VecAXPY(diff, -1.0, scale));
  PetscCall(VecNorm(diff, NORM_INFINITY, &norm));
  PetscCall(VecDestroy(&diff));
  PetscCheck(norm <= 1000.0 * PETSC_MACHINE_EPSILON * PetscAbsScalar(value), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse %s diagonal scale error %g exceeds tolerance", name, (double)norm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES        snes, coarse;
  DM          da;
  Vec         x, left, right, expected;
  PetscReal   norm, rtol, atol, tolerance;
  PetscMPIInt size;
  PetscBool   use_scale = PETSC_FALSE, isfas;
  PetscScalar leftval = 0.5, rightval = 2.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This test requires one MPI rank");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_scale", &use_scale, NULL));

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 5, 1, 1, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, da));
  PetscCall(DMDASNESSetFunctionLocal(da, INSERT_VALUES, (DMDASNESFunctionFn *)FormFunctionLocal, NULL));
  PetscCall(DMDASNESSetJacobianLocal(da, (DMDASNESJacobianFn *)FormJacobianLocal, NULL));

  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecDuplicate(x, &left));
  PetscCall(VecDuplicate(x, &right));
  PetscCall(VecSet(left, leftval));
  PetscCall(VecSet(right, rightval));
  if (use_scale) {
    PetscCall(SNESSetLeftDiagonalScale(snes, left));
    PetscCall(SNESSetRightDiagonalScale(snes, right));
  }

  PetscCall(SNESSetFromOptions(snes));
  PetscCall(VecSet(x, 1.0));
  PetscCall(SNESSolve(snes, NULL, x));

  PetscCall(VecDuplicate(x, &expected));
  PetscCall(VecSet(expected, 2.0));
  PetscCall(VecAXPY(expected, -1.0, x));
  PetscCall(VecNorm(expected, NORM_2, &norm));
  PetscCall(SNESGetTolerances(snes, &atol, &rtol, NULL, NULL, NULL));
  tolerance = 1000.0 * PetscMax(atol, rtol);
  PetscCheck(norm <= tolerance, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNES solution error %g exceeds tolerance %g", (double)norm, (double)tolerance);
  PetscCall(VecDestroy(&expected));

  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESFAS, &isfas));
  if (use_scale && isfas) {
    Vec       coarseleft, coarseright, expectedleft, diff;
    Mat       restrct;
    PetscInt  levels;
    PetscReal lnorm;

    PetscCall(SNESFASGetCoarseSolve(snes, &coarse));
    PetscCall(SNESGetLeftDiagonalScale(coarse, &coarseleft));
    PetscCall(SNESGetRightDiagonalScale(coarse, &coarseright));
    PetscCheck(coarseleft, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse SNES has no left diagonal scale");
    PetscCheck(coarseright, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse SNES has no right diagonal scale");

    /* rightscale (state space) is restricted with the constant-preserving SNESFASRestrict(), so a constant fine value stays constant */
    PetscCall(CheckConstantScale(coarseright, rightval, "right"));

    /* leftscale (dual/residual space) uses the same plain MatRestrict() FAS applies to defects, so only check it matches that operator directly */
    PetscCall(SNESFASGetLevels(snes, &levels));
    PetscCall(SNESFASGetRestriction(snes, levels - 1, &restrct));
    PetscCall(VecDuplicate(coarseleft, &expectedleft));
    PetscCall(MatRestrict(restrct, left, expectedleft));
    PetscCall(VecDuplicate(coarseleft, &diff));
    PetscCall(VecCopy(expectedleft, diff));
    PetscCall(VecAXPY(diff, -1.0, coarseleft));
    PetscCall(VecNorm(diff, NORM_INFINITY, &lnorm));
    PetscCheck(lnorm <= 1000.0 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse left diagonal scale does not match MatRestrict() of the fine left diagonal scale, error %g", (double)lnorm);
    PetscCall(VecDestroy(&diff));
    PetscCall(VecDestroy(&expectedleft));
  }

  PetscCall(VecDestroy(&right));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&x));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: fas
    output_file: output/empty.out
    args: -da_refine 1 -snes_type fas -fas_coarse_snes_max_it 1 -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly -snes_rtol 0 -snes_atol 1e-11 -snes_stol 0 -snes_max_it 20

  test:
    suffix: fas_scaled
    output_file: output/empty.out
    args: -use_scale -da_refine 1 -snes_type fas -fas_coarse_snes_max_it 1 -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly -snes_rtol 0 -snes_atol 1e-11 -snes_stol 0 -snes_max_it 20

TEST*/
