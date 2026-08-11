#include <petsctao.h>
#include "taotermtestclassic.h"

static char help[] = "Solve one or two linear least-squares data terms through TaoAddTerm().\n";

typedef struct {
  PetscBool separate_callbacks;
  PetscBool provide_hessian_mult;
  PetscBool split_hpre;
  PetscInt *hessian_evals;
} TermCtx;

static PetscErrorCode FormObjective(TaoTerm, Vec, Vec, PetscReal *);
static PetscErrorCode FormGradient(TaoTerm, Vec, Vec, Vec);
static PetscErrorCode FormObjectiveGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode FormHessianMult(TaoTerm, Vec, Vec, Vec, Vec);
static PetscErrorCode CreateMap(MPI_Comm, PetscInt, PetscInt, PetscReal, Mat *);
static PetscErrorCode CreateDataTerm(MPI_Comm, const char[], PetscInt, TermCtx *, TaoTerm *);
static PetscErrorCode AddExpectedAction(Mat, PetscReal, Vec, Vec, Vec);
static PetscErrorCode CheckOperator(Tao, Vec, Mat *, PetscReal *, TaoTermMask *, PetscBool *, PetscInt, PetscBool, PetscBool, PetscBool, PetscBool, PetscInt *);

int main(int argc, char **argv)
{
  const PetscInt    n = 10;
  Tao               tao;
  Tao               ctao;
  TaoTerm           terms[2] = {NULL, NULL};
  TaoTerm           objective;
  TermCtx           ctx[2];
  ExampleClassicCtx cctx    = {0};
  Mat               maps[2] = {NULL, NULL};
  Mat               cH, cHpre;
  Vec               targets[2] = {NULL, NULL}, x, cx;
  PetscReal         scales[2]  = {1.0, 0.25};
  PetscInt          nterms = 1, m = 10;
  PetscBool         use_map[2] = {PETSC_TRUE, PETSC_TRUE}, provide_hessian_mult[2] = {PETSC_TRUE, PETSC_TRUE};
  PetscBool         second_term = PETSC_FALSE, no_map = PETSC_FALSE, check_hessian_mult = PETSC_FALSE, separate_callbacks = PETSC_FALSE, split_hpre = PETSC_FALSE;
  PetscBool         parameters_none = PETSC_FALSE, parameters_required = PETSC_FALSE, use_fd[2] = {PETSC_FALSE, PETSC_FALSE};
  PetscBool         none_with_parameters = PETSC_FALSE, required_without_parameters = PETSC_FALSE;
  PetscBool         check_first_hessian_only = PETSC_FALSE;
  PetscBool         check_hessian_cache      = PETSC_FALSE;
  TaoTermMask       masks[2]                 = {TAOTERM_MASK_NONE, TAOTERM_MASK_NONE};
  PetscInt          hessian_evals            = 0;
  MPI_Comm          comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, "", help, "Tao");
  PetscCall(PetscOptionsBool("-second_term", "Add a second least-squares data term", NULL, second_term, &second_term, NULL));
  PetscCall(PetscOptionsBool("-no_map", "Add terms directly in the Tao solution space", NULL, no_map, &no_map, NULL));
  use_map[0] = use_map[1] = (PetscBool)!no_map;
  PetscCall(PetscOptionsBool("-data_use_map", "Map the first data term", NULL, use_map[0], &use_map[0], NULL));
  PetscCall(PetscOptionsBool("-extra_use_map", "Map the second data term", NULL, use_map[1], &use_map[1], NULL));
  PetscCall(PetscOptionsBool("-separate_callbacks", "Register separate objective and gradient callbacks", NULL, separate_callbacks, &separate_callbacks, NULL));
  PetscCall(PetscOptionsBool("-provide_hessian_mult", "Register the direct Hessian-vector callback on both terms by default", NULL, provide_hessian_mult[0], &provide_hessian_mult[0], NULL));
  provide_hessian_mult[1] = provide_hessian_mult[0];
  PetscCall(PetscOptionsBool("-data_provide_hessian_mult", "Register HessianMult on the first data term", NULL, provide_hessian_mult[0], &provide_hessian_mult[0], NULL));
  PetscCall(PetscOptionsBool("-extra_provide_hessian_mult", "Register HessianMult on the second data term", NULL, provide_hessian_mult[1], &provide_hessian_mult[1], NULL));
  PetscCall(PetscOptionsBool("-split_hpre", "Use a distinct raw preconditioning matrix equal to two times the Hessian", NULL, split_hpre, &split_hpre, NULL));
  PetscCall(PetscOptionsBool("-check_hessian_mult", "Compare TaoComputeHessianMult() with the exact least-squares action", NULL, check_hessian_mult, &check_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-parameters_none", "Configure every term with parameter mode NONE", NULL, parameters_none, &parameters_none, NULL));
  PetscCall(PetscOptionsBool("-parameters_required", "Configure every term with parameter mode REQUIRED", NULL, parameters_required, &parameters_required, NULL));
  PetscCall(PetscOptionsBool("-none_with_parameters", "Deliberately supply parameters to terms configured with parameter mode NONE", NULL, none_with_parameters, &none_with_parameters, NULL));
  PetscCall(PetscOptionsBool("-required_without_parameters", "Deliberately omit parameters from terms configured with parameter mode REQUIRED", NULL, required_without_parameters, &required_without_parameters, NULL));
  PetscCall(PetscOptionsBool("-check_first_hessian_only", "Expect only the first data term to contribute to the Hessian", NULL, check_first_hessian_only, &check_first_hessian_only, NULL));
  PetscCall(PetscOptionsBool("-check_hessian_cache", "Check that assembled Hessian and HessianMult evaluations share cached raw Hessians", NULL, check_hessian_cache, &check_hessian_cache, NULL));
  PetscCall(PetscOptionsInt("-m", "Number of observations in the first data set", NULL, m, &m, NULL));
  PetscOptionsEnd();
  PetscCheck(!parameters_none || !parameters_required, comm, PETSC_ERR_USER_INPUT, "Select at most one explicit parameter mode");
  nterms = second_term ? 2 : 1;
  for (PetscInt i = 0; i < nterms; i++) {
    ctx[i].separate_callbacks   = separate_callbacks;
    ctx[i].provide_hessian_mult = provide_hessian_mult[i];
    ctx[i].split_hpre           = split_hpre;
    ctx[i].hessian_evals        = &hessian_evals;
  }

  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, n, &x));
  PetscCall(VecSet(x, 0.0));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao, "shell_"));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));

  for (PetscInt i = 0; i < nterms; i++) {
    const PetscInt mi = use_map[i] ? (i ? n + 2 : m) : n;
    char           prefix[16];

    PetscCall(PetscStrncpy(prefix, i ? "extra_" : "data_", sizeof(prefix)));
    if (use_map[i]) PetscCall(CreateMap(comm, mi, n, i ? 0.75 : 1.0, &maps[i]));
    PetscCall(CreateDataTerm(comm, prefix, mi, &ctx[i], &terms[i]));
    if (parameters_none || none_with_parameters) PetscCall(TaoTermSetParametersMode(terms[i], TAOTERM_PARAMETERS_NONE));
    else if (parameters_required || required_without_parameters) PetscCall(TaoTermSetParametersMode(terms[i], TAOTERM_PARAMETERS_REQUIRED));
    if (!parameters_none && !required_without_parameters) {
      PetscCall(VecCreateMPI(comm, PETSC_DECIDE, mi, &targets[i]));
      PetscCall(VecSet(targets[i], 1.0 + i));
    }
    PetscCall(TaoAddTerm(tao, prefix, scales[i], terms[i], targets[i], maps[i]));
  }

  PetscCall(TaoSetFromOptions(tao));
  cctx.nleaves = nterms;
  if (nterms > 1) PetscCall(TaoGetTerm(tao, NULL, &objective, NULL, NULL));
  for (PetscInt i = 0; i < nterms; i++) {
    cctx.leaves[i].type       = EXAMPLE_CLASSIC_LEAST_SQUARES;
    cctx.leaves[i].map        = maps[i];
    cctx.leaves[i].parameters = targets[i];
    cctx.leaves[i].scale      = scales[i];
    cctx.leaves[i].mask       = TAOTERM_MASK_NONE;
    if (nterms > 1) PetscCall(TaoTermSumGetTermMask(objective, i, &cctx.leaves[i].mask));
    masks[i] = cctx.leaves[i].mask;
  }
  if (check_first_hessian_only) PetscCheck(nterms == 2 && !(masks[0] & TAOTERM_MASK_HESSIAN) && (masks[1] & TAOTERM_MASK_HESSIAN), comm, PETSC_ERR_ARG_WRONG, "-check_first_hessian_only requires only the second term's Hessian to be masked");
  PetscCall(VecDuplicate(x, &cx));
  PetscCall(VecCopy(x, cx));
  PetscCall(ExampleClassicCreateTao(comm, tao, cx, &cctx, &ctao, &cH, &cHpre));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoSolve(ctao));
  PetscCall(ExampleClassicCompareResults(tao, x, ctao, cx));
  PetscCall(PetscOptionsGetBool(NULL, "data_", "-tao_term_hessian_use_fd", &use_fd[0], NULL));
  if (nterms > 1) PetscCall(PetscOptionsGetBool(NULL, "extra_", "-tao_term_hessian_use_fd", &use_fd[1], NULL));
  PetscCall(CheckOperator(tao, x, maps, scales, masks, use_fd, nterms, split_hpre, check_hessian_mult, check_hessian_cache, provide_hessian_mult[0] || provide_hessian_mult[1], &hessian_evals));
  PetscCall(PetscPrintf(comm, "Least-squares TaoTerm operator check passed\n"));

  PetscCall(TaoDestroy(&ctao));
  if (cHpre != cH) PetscCall(MatDestroy(&cHpre));
  PetscCall(MatDestroy(&cH));
  PetscCall(VecDestroy(&cctx.hessian_x));
  PetscCall(VecDestroy(&cx));
  for (PetscInt i = 0; i < nterms; i++) {
    PetscCall(TaoTermDestroy(&terms[i]));
    PetscCall(VecDestroy(&targets[i]));
    PetscCall(MatDestroy(&maps[i]));
  }
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode FormObjective(TaoTerm term, Vec x, Vec params, PetscReal *f)
{
  Vec         work;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &work));
  if (params) PetscCall(VecWAXPY(work, -1.0, params, x));
  else PetscCall(VecCopy(x, work));
  PetscCall(VecDot(work, work, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormGradient(TaoTerm term, Vec x, Vec params, Vec g)
{
  PetscFunctionBeginUser;
  if (params) PetscCall(VecWAXPY(g, -1.0, params, x));
  else PetscCall(VecCopy(x, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormObjectiveGradient(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(FormGradient(term, x, params, g));
  PetscCall(VecDot(g, g, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  (*ctx->hessian_evals)++;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(H, 1.0));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(Hpre, ctx->split_hpre ? 2.0 : 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianMult(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMap(MPI_Comm comm, PetscInt m, PetscInt n, PetscReal shift, Mat *A)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, 2, NULL, 2, NULL, A));
  PetscCall(MatGetOwnershipRange(*A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    const PetscInt j = i % n;

    PetscCall(MatSetValue(*A, i, j, shift + 0.05 * (i + 1), INSERT_VALUES));
    if (n > 1) PetscCall(MatSetValue(*A, i, (j + 1) % n, 0.1, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDataTerm(MPI_Comm comm, const char prefix[], PetscInt m, TermCtx *ctx, TaoTerm *term)
{
  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, ctx, NULL, term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*term, prefix));
  PetscCall(TaoTermSetSolutionSizes(*term, PETSC_DECIDE, m, 1));
  PetscCall(TaoTermSetParametersSizes(*term, PETSC_DECIDE, m, 1));
  if (ctx->separate_callbacks) {
    PetscCall(TaoTermShellSetObjective(*term, FormObjective));
    PetscCall(TaoTermShellSetGradient(*term, FormGradient));
  } else PetscCall(TaoTermShellSetObjectiveAndGradient(*term, FormObjectiveGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(*term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(*term, ctx->split_hpre ? PETSC_FALSE : PETSC_TRUE, MATAIJ, ctx->split_hpre ? MATAIJ : NULL));
  PetscCall(TaoTermShellSetHessian(*term, FormHessian));
  if (ctx->provide_hessian_mult) PetscCall(TaoTermShellSetHessianMult(*term, FormHessianMult));
  PetscCall(TaoTermSetFromOptions(*term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddExpectedAction(Mat A, PetscReal scale, Vec v, Vec expected, Vec work)
{
  PetscFunctionBeginUser;
  if (A) {
    Vec Av;

    PetscCall(MatCreateVecs(A, NULL, &Av));
    PetscCall(MatMult(A, v, Av));
    PetscCall(MatMultTranspose(A, Av, work));
    PetscCall(VecDestroy(&Av));
  } else PetscCall(VecCopy(v, work));
  PetscCall(VecAXPY(expected, scale, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckOperator(Tao tao, Vec x, Mat maps[], PetscReal scales[], TaoTermMask masks[], PetscBool use_fd[], PetscInt nterms, PetscBool split_hpre, PetscBool check_hessian_mult, PetscBool check_hessian_cache, PetscBool any_hessian_mult, PetscInt *hessian_evals)
{
  Mat       H, Hpre, H2 = NULL;
  Vec       v, actual, expected, expected_pre, work;
  PetscReal error, tolerance = 2.e-5;
  PetscInt  active_hessian_terms = 0, evals_before;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &actual));
  PetscCall(VecDuplicate(x, &expected));
  PetscCall(VecDuplicate(x, &expected_pre));
  PetscCall(VecDuplicate(x, &work));
  PetscCall(VecSet(v, 1.0));
  PetscCall(VecZeroEntries(expected));
  PetscCall(VecZeroEntries(expected_pre));
  for (PetscInt i = 0; i < nterms; i++) {
    if (masks[i] & TAOTERM_MASK_HESSIAN) continue;
    PetscCall(AddExpectedAction(maps[i], scales[i], v, expected, work));
    PetscCall(AddExpectedAction(maps[i], split_hpre && !use_fd[i] ? 2.0 * scales[i] : scales[i], v, expected_pre, work));
    active_hessian_terms++;
  }
  PetscCheck(active_hessian_terms, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "At least one term must contribute to the Hessian");

  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(MatMult(H, v, actual));
  PetscCall(VecAXPY(actual, -1.0, expected));
  PetscCall(VecNorm(actual, NORM_2, &error));
  PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Tao Hessian action differs from the exact least-squares action by %g", (double)error);

  if (Hpre != H) {
    PetscCall(MatMult(Hpre, v, actual));
    PetscCall(VecAXPY(actual, -1.0, expected_pre));
    PetscCall(VecNorm(actual, NORM_2, &error));
    PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Tao preconditioning action differs from its exact mapped action by %g", (double)error);
  }
  if (check_hessian_mult) {
    PetscCall(TaoComputeHessianMult(tao, x, v, actual));
    PetscCall(VecAXPY(actual, -1.0, expected));
    PetscCall(VecNorm(actual, NORM_2, &error));
    PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "TaoComputeHessianMult() differs from the exact sum action by %g", (double)error);
  }
  if (check_hessian_cache) {
    PetscCheck(!any_hessian_mult, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "-check_hessian_cache requires all HessianMult callbacks to be disabled");
    PetscCall(VecShift(x, 0.125));
    evals_before = *hessian_evals;
    PetscCall(TaoComputeHessian(tao, x, H, Hpre));
    PetscCheck(*hessian_evals == evals_before + active_hessian_terms, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "First Hessian evaluation at a new point called %" PetscInt_FMT " callbacks, expected %" PetscInt_FMT, *hessian_evals - evals_before, active_hessian_terms);
    PetscCall(MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &H2));
    PetscCall(MatAssemblyBegin(H2, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H2, MAT_FINAL_ASSEMBLY));
    evals_before = *hessian_evals;
    PetscCall(TaoComputeHessian(tao, x, H2, H2));
    PetscCall(TaoComputeHessianMult(tao, x, v, actual));
    PetscCheck(*hessian_evals == evals_before, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Repeated Hessian and HessianMult evaluations called %" PetscInt_FMT " redundant Hessian callbacks", *hessian_evals - evals_before);
    PetscCall(MatMult(H2, v, work));
    PetscCall(VecAXPY(work, -1.0, expected));
    PetscCall(VecNorm(work, NORM_2, &error));
    PetscCheck(error <= tolerance, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Hessian assembled into a second destination differs from the exact action by %g", (double)error);
    PetscCall(VecShift(x, -0.125));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Shared Hessian cache check passed\n"));
  }
  PetscCall(MatDestroy(&H2));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&actual));
  PetscCall(VecDestroy(&expected));
  PetscCall(VecDestroy(&expected_pre));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  testset:
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|parameter vector space|Mask \(|rows=.*cols=|Solution converged|operator check passed" | grep -v "MatType.*undefined"

    test:
      suffix: assembled
      args: -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: rectangular_map
      args: -m 15 -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: separate_callbacks
      args: -separate_callbacks -shell_tao_type nls -shell_tao_view ::ascii_info_detail
      filter: grep -E "methods have been set|Hessian (preconditioning )?MatType|parameter vector space|rows=.*cols=|Solution converged|operator check passed" | grep -v "MatType.*undefined"

    test:
      suffix: mapped_separate_hpre
      args: -split_hpre -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: mapped_fd
      args: -data_tao_term_hessian_use_fd -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: mapped_fd_separate_hpre
      args: -split_hpre -data_tao_term_hessian_use_fd -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: two_mapped_assembled
      args: -second_term -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: fallback_unmapped
      args: -second_term -no_map -provide_hessian_mult false -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell -shell_tao_view ::ascii_info_detail

    test:
      suffix: fallback_mapped
      args: -second_term -provide_hessian_mult false -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell -shell_tao_view ::ascii_info_detail

    test:
      suffix: fallback_separate_hpre
      args: -second_term -no_map -provide_hessian_mult false -split_hpre -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell -shell_tao_term_hessian_pre_is_hessian false
      args: -shell_tao_term_hessian_pre_mat_type aij -shell_tao_view ::ascii_info_detail

    test:
      suffix: fallback_mapped_separate_hpre
      args: -second_term -provide_hessian_mult false -split_hpre -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell -shell_tao_term_hessian_pre_is_hessian false
      args: -shell_tao_term_hessian_pre_mat_type aij -shell_tao_view ::ascii_info_detail

    test:
      suffix: sum_hessian_mult
      args: -second_term -check_hessian_mult -shell_tao_type nls
      args: -shell_tao_term_hessian_mat_type shell -shell_tao_view ::ascii_info_detail

    test:
      suffix: shared_hessian_cache
      args: -second_term -provide_hessian_mult false -check_hessian_cache -shell_tao_type nls
      filter: grep -E "Shared Hessian cache check passed|operator check passed"

    test:
      suffix: parameters_none
      args: -second_term -parameters_none -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: parameters_required
      args: -second_term -parameters_required -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: hessian_only_model
      args: -second_term -shell_tao_term_sum_extra_mask objective,gradient
      args: -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: shell_masked_hessian
      args: -second_term -provide_hessian_mult false -check_first_hessian_only
      args: -shell_tao_term_sum_extra_mask hessian -shell_tao_term_hessian_mat_type shell
      args: -shell_tao_type nls -shell_tao_view ::ascii_info_detail

  test:
    suffix: none_with_parameters
    args: -none_with_parameters -shell_tao_type nls -petsc_ci_portable_error_output -error_output_stdout
    filter: grep -E "Parameters passed to a TaoTerm with TAOTERM_PARAMETERS_NONE"

  test:
    suffix: required_without_parameters
    args: -required_without_parameters -shell_tao_type nls -petsc_ci_portable_error_output -error_output_stdout
    filter: grep -E "Parameters required but not provided for a TaoTerm with TAOTERM_PARAMETERS_REQUIRED"

  test:
    suffix: asymmetric_map_assembled
    filter: grep -E "Classic callback comparison passed|operator check passed"
    output_file: output/taotermtest2_classic_operator_comparison.out
    args: -second_term -data_use_map true -extra_use_map false -shell_tao_type nls

  test:
    suffix: mixed_direct_fallback
    filter: grep -E "Classic callback comparison passed|operator check passed"
    output_file: output/taotermtest2_classic_operator_comparison.out
    args: -second_term -data_use_map false -extra_use_map true
    args: -data_provide_hessian_mult false -extra_provide_hessian_mult true
    args: -shell_tao_type nls -shell_tao_term_hessian_mat_type shell

  test:
    suffix: asymmetric_map_mffd
    filter: grep -E "Classic callback comparison passed|operator check passed"
    output_file: output/taotermtest2_classic_operator_comparison.out
    args: -second_term -data_use_map true -extra_use_map false
    args: -shell_tao_type nls -shell_tao_term_hessian_mat_type mffd

  testset:
    filter: grep -E "Classic callback comparison passed|operator check passed"
    output_file: output/taotermtest2_classic_operator_comparison.out
    args: -second_term -shell_tao_type nls

    test:
      suffix: r092_second_analytic
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r093_second_fd
      args: -no_map -shell_tao_term_sum_data_mask hessian -extra_tao_term_hessian_use_fd

    test:
      suffix: r094_mapped_second_analytic
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r095_mapped_second_fd
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r096_first_analytic
      args: -no_map -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r097_first_fd
      args: -no_map -shell_tao_term_sum_extra_mask hessian -data_tao_term_hessian_use_fd

    test:
      suffix: r098_mapped_first_analytic
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r099_mapped_first_fd
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r100_analytic_analytic
      args: -no_map

    test:
      suffix: r101_analytic_fd
      args: -no_map -extra_tao_term_hessian_use_fd

    test:
      suffix: r102_fd_fd
      args: -no_map -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r104_mapped_analytic_fd
      args: -data_use_map true -extra_use_map false -extra_tao_term_hessian_use_fd

    test:
      suffix: r105_mapped_fd_fd
      args: -data_use_map true -extra_use_map false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r107_both_mapped_analytic_fd
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r108_both_mapped_fd_fd
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

  testset:
    filter: grep -E "Classic callback comparison passed|operator check passed"
    output_file: output/taotermtest2_classic_operator_comparison.out
    args: -second_term -split_hpre -shell_tao_type nls
    args: -shell_tao_term_hessian_pre_is_hessian false -shell_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r109_second_analytic_separate_hpre
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r110_second_fd_separate_hpre
      args: -no_map -shell_tao_term_sum_data_mask hessian -extra_tao_term_hessian_use_fd

    test:
      suffix: r111_mapped_second_analytic_separate_hpre
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r112_mapped_second_fd_separate_hpre
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r113_first_analytic_separate_hpre
      args: -no_map -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r114_first_fd_separate_hpre
      args: -no_map -shell_tao_term_sum_extra_mask hessian -data_tao_term_hessian_use_fd

    test:
      suffix: r115_mapped_first_analytic_separate_hpre
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r116_mapped_first_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r117_analytic_analytic_separate_hpre
      args: -no_map

    test:
      suffix: r118_analytic_fd_separate_hpre
      args: -no_map -extra_tao_term_hessian_use_fd

    test:
      suffix: r119_fd_fd_separate_hpre
      args: -no_map -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r120_mapped_analytic_analytic_separate_hpre
      args: -data_use_map true -extra_use_map false

    test:
      suffix: r121_mapped_analytic_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -extra_tao_term_hessian_use_fd

    test:
      suffix: r122_mapped_fd_fd_separate_hpre
      args: -data_use_map true -extra_use_map false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r123_both_mapped_analytic_analytic_separate_hpre

    test:
      suffix: r124_both_mapped_analytic_fd_separate_hpre
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r125_both_mapped_fd_fd_separate_hpre
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

TEST*/
