#include <petsctao.h>

static char help[] = "Tests cached mapped TaoTerm Hessian products through the supported Tao API.\n";

typedef struct {
  Mat       raw_H;
  Mat       raw_Hpre;
  PetscInt  update;
  PetscBool initialized;
  PetscBool Hpre_is_H;
} AppCtx;

enum {
  UPDATE_NONE,
  UPDATE_H_VALUES,
  UPDATE_H_STRUCTURE,
  UPDATE_HPRE_VALUES,
  UPDATE_HPRE_STRUCTURE
};

static PetscErrorCode SetDiagonal(Mat A, PetscReal scale)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(MatSetValue(A, i, i, scale * (i + 1.0), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetSymmetricOffDiagonal(Mat A, PetscScalar value)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  if (rstart <= 0 && 0 < rend) PetscCall(MatSetValue(A, 0, 1, value, INSERT_VALUES));
  if (rstart <= 1 && 1 < rend) PetscCall(MatSetValue(A, 1, 0, value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
    PetscCall(SetDiagonal(H, 1.0));
    if (Hpre != H) PetscCall(SetDiagonal(Hpre, 2.0));
    ctx->initialized = PETSC_TRUE;
  } else if (ctx->update == UPDATE_H_VALUES) {
    PetscCall(MatSetValue(H, 0, 0, 4.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  } else if (ctx->update == UPDATE_H_STRUCTURE) PetscCall(SetSymmetricOffDiagonal(H, 0.5));
  else if (ctx->update == UPDATE_HPRE_VALUES) {
    PetscCall(MatSetValue(Hpre, 0, 0, 8.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  } else if (ctx->update == UPDATE_HPRE_STRUCTURE) PetscCall(SetSymmetricOffDiagonal(Hpre, 0.75));
  ctx->update = UPDATE_NONE;
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

static PetscErrorCode CheckMatricesEqual(Mat expected, Mat actual, const char description[])
{
  Mat       difference;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(MatDuplicate(expected, MAT_COPY_VALUES, &difference));
  PetscCall(MatAXPY(difference, -1.0, actual, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(difference, NORM_FROBENIUS, &norm));
  PetscCheck(norm <= 1.e-10, PetscObjectComm((PetscObject)actual), PETSC_ERR_PLIB, "%s (norm of error %g)", description, (double)norm);
  PetscCall(MatDestroy(&difference));
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
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, 3, 3, 3, NULL, 3, NULL, &map));
  PetscCall(SetDiagonal(map, 1.0));

  PetscCall(TaoTermCreateShell(comm, &ctx, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, 3, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term, Hpre_is_H, MATAIJ, Hpre_is_H ? NULL : MATAIJ));
  PetscCall(TaoTermShellSetHessian(term, FormHessian));

  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, 3, &x));
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

  ctx.update = UPDATE_H_VALUES;
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

  if (!Hpre_is_H) {
    ctx.update = UPDATE_HPRE_VALUES;
    PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
    PetscCall(TaoComputeHessian(tao, x, H, Hpre));
    PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
    PetscCheck(numeric[1] - numeric[0] == 1 && symbolic[1] == symbolic[0], comm, PETSC_ERR_PLIB, "A raw-Hpre value change did not update only the Hpre numeric PtAP");
    PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));
    PetscCall(CheckMappedMatrix(ctx.raw_Hpre, map, scale, Hpre));

    ctx.update = UPDATE_HPRE_STRUCTURE;
    PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
    PetscCall(TaoComputeHessian(tao, x, H, Hpre));
    PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
    PetscCheck(symbolic[1] - symbolic[0] == 1, comm, PETSC_ERR_PLIB, "A raw-Hpre sparsity change did not rebuild only the Hpre symbolic PtAP");
    PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));
    PetscCall(CheckMappedMatrix(ctx.raw_Hpre, map, scale, Hpre));
  }

  ctx.update = UPDATE_H_STRUCTURE;
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(symbolic[1] - symbolic[0] == 1, comm, PETSC_ERR_PLIB, "A raw-H sparsity change did not rebuild exactly one PtAP symbolic structure");
  PetscCall(CheckMappedMatrix(ctx.raw_H, map, scale, H));

  PetscCall(SetSymmetricOffDiagonal(map, 0.25));
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

static PetscErrorCode TestOuterReconstruction(MPI_Comm comm, PetscLogEvent numeric_event, PetscLogEvent symbolic_event)
{
  AppCtx ctx[2];
  Tao    tao;
  TaoTerm term[2];
  Mat     H, Hpre, expected, expected_pre, map;
  Vec     x;
  int     numeric[2], symbolic[2];

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(ctx)));
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, 3, 3, 3, NULL, 3, NULL, &map));
  PetscCall(SetDiagonal(map, 1.0));
  for (PetscInt i = 0; i < 2; i++) {
    ctx[i].Hpre_is_H = PETSC_FALSE;
    PetscCall(TaoTermCreateShell(comm, &ctx[i], NULL, &term[i]));
    PetscCall(TaoTermSetParametersMode(term[i], TAOTERM_PARAMETERS_NONE));
    PetscCall(TaoTermSetSolutionSizes(term[i], PETSC_DECIDE, 3, 1));
    PetscCall(TaoTermShellSetObjectiveAndGradient(term[i], FormObjectiveAndGradient));
    PetscCall(TaoTermShellSetCreateHessianMatrices(term[i], TaoTermCreateHessianMatricesDefault));
    PetscCall(TaoTermSetCreateHessianMode(term[i], PETSC_FALSE, MATAIJ, MATAIJ));
    PetscCall(TaoTermShellSetHessian(term[i], FormHessian));
  }
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, 3, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  for (PetscInt i = 0; i < 2; i++) PetscCall(TaoAddTerm(tao, NULL, i + 1.0, term[i], NULL, map));
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCheck(H != Hpre, comm, PETSC_ERR_PLIB, "Mapped sum lost its separate Hpre configuration");
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(numeric[1] - numeric[0] == 4 && symbolic[1] - symbolic[0] == 4, comm, PETSC_ERR_PLIB, "Two mapped summands with separate Hpre created %d numeric and %d symbolic PtAP products instead of four of each", numeric[1] - numeric[0], symbolic[1] - symbolic[0]);
  PetscCall(CheckMappedMatrix(ctx[0].raw_H, map, 3.0, H));
  PetscCall(CheckMappedMatrix(ctx[0].raw_Hpre, map, 3.0, Hpre));
  PetscCall(MatDuplicate(H, MAT_COPY_VALUES, &expected));
  PetscCall(MatDuplicate(Hpre, MAT_COPY_VALUES, &expected_pre));

  PetscCall(MatShift(H, 7.0));
  PetscCall(MatShift(Hpre, 11.0));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[0], &symbolic[0]));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(GetPtAPCounts(numeric_event, symbolic_event, &numeric[1], &symbolic[1]));
  PetscCheck(numeric[1] == numeric[0] && symbolic[1] == symbolic[0], comm, PETSC_ERR_PLIB, "Reconstructing a solver-modified outer Hessian recomputed unchanged mapped summand products");
  PetscCall(CheckMatricesEqual(expected, H, "Recomputing the Hessian did not remove a solver modification from the outer matrix"));
  PetscCall(CheckMatricesEqual(expected_pre, Hpre, "Recomputing the Hessian did not remove a solver modification from the outer preconditioner"));

  PetscCall(MatDestroy(&expected));
  PetscCall(MatDestroy(&expected_pre));
  PetscCall(TaoDestroy(&tao));
  for (PetscInt i = 0; i < 2; i++) PetscCall(TaoTermDestroy(&term[i]));
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
  PetscCall(TestOuterReconstruction(PETSC_COMM_WORLD, numeric_event, symbolic_event));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Mapped Hessian cache skips unchanged products and distinguishes numeric from symbolic updates\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/taotermtest8_0.out
    requires: !complex defined(PETSC_USE_LOG)

    test:
      suffix: 0
      nsize: 1

    test:
      suffix: mpi
      nsize: 2

TEST*/
