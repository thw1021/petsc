static char help[] = "Tests that PCComputeOperator() applied to PCApply() and PCApplyTranspose() are genuinely transposes\n\
of each other, for every PCMG cycle and a non-symmetric smoother (SOR forward), parameterized by -pc_mg_type and\n\
-pc_mg_symmetric on the command line. Only the multiplicative cycle with -pc_mg_symmetric true is expected to also\n\
give a symmetric operator: the additive cycle ignores -pc_mg_symmetric, and the Kaskade and full (F) cycles smooth\n\
just once per level while ascending, with no down-smoothing pass to sandwich it with.\n\n";

#include <petscksp.h>
#include <petscdmda.h>

typedef struct {
  PCMGType  cycle;
  PetscBool symmetric;
} UserCtx;

static PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat A, void *ctx)
{
  PetscInt rstart, rend, N;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(A, &N, NULL));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt row = rstart; row < rend; row++) {
    PetscCall(MatSetValue(A, row, row, 2.0, INSERT_VALUES));
    if (row > 0) PetscCall(MatSetValue(A, row, row - 1, -1.0, INSERT_VALUES));
    if (row < N - 1) PetscCall(MatSetValue(A, row, row + 1, -1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_PC(Mat A, Vec X, Vec Y)
{
  PC pc;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &pc));
  PetscCall(PCApplyTranspose(pc, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PCComputeOperator() forms explicit(PCApply()); there is no transpose counterpart in the public API,
   so mirror its implementation here with PCApplyTranspose() in place of PCApply(). */
static PetscErrorCode PCComputeOperatorTranspose(PC pc, Mat *mat)
{
  Mat      A, Apc;
  PetscInt M, N, m, n;

  PetscFunctionBegin;
  PetscCall(PCGetOperators(pc, &A, NULL));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)pc), m, n, M, N, pc, &Apc));
  PetscCall(MatShellSetOperation(Apc, MATOP_MULT, (PetscErrorCodeFn *)MatMultTranspose_PC));
  PetscCall(MatComputeOperator(Apc, NULL, mat));
  PetscCall(MatDestroy(&Apc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckMatEqual(Mat X, Mat Y, const char msg[])
{
  Mat       diff;
  PetscReal err, scale;
  PetscInt  m, n;

  PetscFunctionBegin;
  PetscCall(MatGetSize(X, &m, &n));
  PetscCall(MatDuplicate(X, MAT_COPY_VALUES, &diff));
  PetscCall(MatAXPY(diff, -1.0, Y, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(diff, NORM_FROBENIUS, &err));
  PetscCall(MatNorm(X, NORM_FROBENIUS, &scale));
  PetscCheck(err < PETSC_SMALL * m * n * PetscMax(scale, 1.0), PetscObjectComm((PetscObject)X), PETSC_ERR_PLIB, "%s: mismatch of norm %g (scale %g)", msg, (double)err, (double)scale);
  PetscCall(MatDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm comm;
  DM       da;
  KSP      ksp;
  PC       pc;
  Mat      Op, OpT, OpTexpected;
  UserCtx  user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_SELF;

  PetscCall(DMDACreate1d(comm, DM_BOUNDARY_NONE, 9, 1, 1, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix, NULL));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCMGGetType(pc, &user.cycle));
  PetscCall(PCMGGetSymmetric(pc, &user.symmetric));

  PetscCall(PCComputeOperator(pc, NULL, &Op));
  PetscCall(PCComputeOperatorTranspose(pc, &OpT));
  PetscCall(MatTranspose(Op, MAT_INITIAL_MATRIX, &OpTexpected));
  PetscCall(CheckMatEqual(OpT, OpTexpected, "explicit(PCApplyTranspose) is not the transpose of explicit(PCApply)"));

  if (user.cycle == PC_MG_MULTIPLICATIVE && user.symmetric) PetscCall(CheckMatEqual(Op, OpTexpected, "-pc_mg_symmetric true: explicit(PCApply) is not symmetric"));

  PetscCall(MatDestroy(&Op));
  PetscCall(MatDestroy(&OpT));
  PetscCall(MatDestroy(&OpTexpected));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     suffix: 0
     requires: !single
     output_file: output/empty.out
     args: -da_grid_x 9 -da_refine 1 -pc_type mg -pc_mg_levels 2 -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 2 \
           -mg_levels_ksp_convergence_test skip -mg_levels_pc_type sor -mg_levels_pc_sor_forward \
           -pc_mg_type {{multiplicative additive full kaskade}} -pc_mg_symmetric {{false true}}

TEST*/
