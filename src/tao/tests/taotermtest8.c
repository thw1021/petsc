#include <petsctao.h>

static char help[] = "Tests cached mapped TaoTerm Hessian products through the supported Tao API.\n";

typedef struct {
  Mat       raw_H;
  Mat       raw_Hpre;
  PetscInt  update;
  PetscBool initialized;
  PetscBool Hpre_is_H;
} AppCtx;

static PetscErrorCode FormObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  PetscFunctionBeginUser;
  *f = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  AppCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  ctx->raw_H    = H;
  ctx->raw_Hpre = Hpre;
  if (!ctx->initialized) {
    for (PetscInt i = 0; i < 3; i++) PetscCall(MatSetValue(H, i, i, i + 1.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    if (Hpre != H) {
      for (PetscInt i = 0; i < 3; i++) PetscCall(MatSetValue(Hpre, i, i, 2.0 * (i + 1.0), INSERT_VALUES));
      PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    }
    ctx->initialized = PETSC_TRUE;
  } else if (ctx->update == 1) {
    PetscCall(MatSetValue(H, 0, 0, 4.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  } else if (ctx->update == 2) {
    PetscCall(MatSetOption(H, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatSetValue(H, 0, 1, 0.5, INSERT_VALUES));
    PetscCall(MatSetValue(H, 1, 0, 0.5, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  ctx->update = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetPtAPCounts(PetscLogEvent numeric_event, PetscLogEvent symbolic_event, int *numeric, int *symbolic)
{
  PetscEventPerfInfo info;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, numeric_event, &info));
  *numeric = info.count;
  PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, symbolic_event, &info));
  *symbolic = info.count;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckMappedMatrix(Mat raw, Mat map, PetscReal scale, Mat actual)
{
  Mat       expected;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(MatPtAP(raw, map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &expected));
  PetscCall(MatScale(expected, scale));
  PetscCall(MatAXPY(expected, -1.0, actual, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(expected, NORM_FROBENIUS, &norm));
  PetscCheck(norm <= 1.e-10, PetscObjectComm((PetscObject)actual), PETSC_ERR_PLIB, "Cached mapped Hessian is incorrect (norm of error %g)", (double)norm);
  PetscCall(MatDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMappedCache(MPI_Comm comm, PetscBool Hpre_is_H, PetscLogEvent numeric_event, PetscLogEvent symbolic_event)
{
  const PetscReal scale = 2.0;
  AppCtx         ctx    = {.Hpre_is_H = Hpre_is_H};
  Tao            tao;
  TaoTerm        term;
  Mat            H, Hpre, map;
  Vec            x;
  int            numeric[2], symbolic[2];

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(comm, 3, 3, 3, NULL, &map));
  for (PetscInt i = 0; i < 3; i++) PetscCall(MatSetValue(map, i, i, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(map, MAT_FINAL_ASSEMBLY));

  PetscCall(TaoTermCreateShell(comm, &ctx, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, 3, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term, Hpre_is_H, MATAIJ, Hpre_is_H ? NULL : MATAIJ));
  PetscCall(TaoTermShellSetHessian(term, FormHessian));

  PetscCall(VecCreateSeq(comm, 3, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoAddTerm(tao, NULL, scale, term, NULL, map));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoSetUp(tao));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(numeric[1] == numeric[0] && symbolic[1] == symbolic[0], comm, PETSC_ERR_PLIB, "TaoSetUp() performed PtAP before the first genuine Hessian evaluation");
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));

  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(numeric[1] - numeric[0] == (Hpre_is_H ? 1 : 2), comm, PETSC_ERR_PLIB, "Initial mapped Hessian evaluation performed %d numeric PtAP operations instead of %d", numeric[1] - numeric[0], Hpre_is_H ? 1 : 2);
  PetscCheck(symbolic[1] - symbolic[0] == (Hpre_is_H ? 1 : 2), comm, PETSC_ERR_PLIB, "Initial mapped Hessian evaluation created %d symbolic PtAP structures instead of %d", symbolic[1] - symbolic[0], Hpre_is_H ? 1 : 2);
  PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));
  if (!Hpre_is_H) PetscCall(CheckMappedMatrix(ctx.raw_Hpre, map, scale, Hpre));

  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(numeric[1] == numeric[0] && symbolic[1] == symbolic[0], comm, PETSC_ERR_PLIB, "Unchanged raw Hessian and map did not reuse the cached mapped product");
  PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));

  ctx.update = 1;
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(numeric[1] - numeric[0] == 1 && symbolic[1] == symbolic[0], comm, PETSC_ERR_PLIB, "A raw-H value change did not use exactly one numeric PtAP update");
  PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));

  PetscCall(MatScale(map, 2.0));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(numeric[1] - numeric[0] == (Hpre_is_H ? 1 : 2) && symbolic[1] == symbolic[0], comm, PETSC_ERR_PLIB, "A map value change did not use numeric PtAP updates for each independent product");
  PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));
  if (!Hpre_is_H) PetscCall(CheckMappedMatrix(ctx.raw_Hpre, map, scale, Hpre));

  ctx.update = 2;
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(symbolic[1] - symbolic[0] == 1, comm, PETSC_ERR_PLIB, "A raw-H sparsity change did not rebuild exactly one PtAP symbolic structure");
  PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));

  PetscCall(MatSetOption(map, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetValue(map, 0, 1, 0.25, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(map, MAT_FINAL_ASSEMBLY));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(symbolic[1] - symbolic[0] == (Hpre_is_H ? 1 : 2), comm, PETSC_ERR_PLIB, "A map sparsity change did not rebuild each independent PtAP symbolic structure");
  PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));
  if (!Hpre_is_H) PetscCall(CheckMappedMatrix(ctx.raw_Hpre, map, scale, Hpre));

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscLogEvent numeric_event, symbolic_event;
  Mat           mat;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(MatCreate(PETSC_COMM_WORLD, &mat));
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscLogEventGetId("MatPtAPNumeric", &numeric_event));
  PetscCall(PetscLogEventGetId("MatPtAPSymbolic", &symbolic_event));
  PetscCheck(numeric_event >= 0 && symbolic_event >= 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PtAP logging events are not registered");
  PetscCall(TestMappedCache(PETSC_COMM_WORLD, PETSC_TRUE, numeric_event, symbolic_event));
  PetscCall(TestMappedCache(PETSC_COMM_WORLD, PETSC_FALSE, numeric_event, symbolic_event));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Mapped Hessian cache skips unchanged products and distinguishes numeric from symbolic updates\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: !complex defined(PETSC_USE_LOG)

TEST*/
