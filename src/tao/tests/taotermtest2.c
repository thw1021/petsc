#include <petsctao.h>
#include "taotermtestcommon.h"

static char help[] = "Solve one or two linear least-squares data terms through TaoAddTerm().\n";

typedef struct {
  PetscBool separate_callbacks;
  PetscBool split_hpre;
  PetscInt *hessian_evals;
} TermCtx;

typedef enum {
  TEST_PARAMETERS_OPTIONAL,
  TEST_PARAMETERS_NONE,
  TEST_PARAMETERS_REQUIRED,
  TEST_PARAMETERS_NONE_WITH_PARAMETERS,
  TEST_PARAMETERS_REQUIRED_WITHOUT_PARAMETERS
} TestParameters;

static const char *const TestParametersTypes[] = {"optional", "none", "required", "none_with_parameters", "required_without_parameters", "TestParameters", "TEST_PARAMETERS_", NULL};

typedef struct {
  ExampleTerm           terms[2];
  PetscInt              nterms;
  PetscBool             separate_callbacks;
  PetscBool             split_hpre;
  PetscBool             check_hessian_mult;
  PetscBool             check_first_hessian_only;
  PetscBool             check_hessian_cache;
  TestParameters        parameters_test;
  TaoTermParametersMode parameters_mode;
} TestOptions;

static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode TestOptionsSetFromOptions(MPI_Comm, TestOptions *);
static PetscErrorCode CreateMap(MPI_Comm, PetscInt, PetscInt, PetscReal, Mat *);
static PetscErrorCode CreateDataTerm(MPI_Comm, const char[], TermCtx *, ExampleTerm *);
static PetscErrorCode AddExpectedAction(Mat, PetscReal, Vec, Vec, Vec);
static PetscErrorCode CheckOperator(Tao, Vec, ExampleTerm *, PetscInt, PetscBool, PetscBool, PetscBool, PetscInt *);

int main(int argc, char **argv)
{
  const PetscInt   n = 10;
  Tao              tao;
  TaoTerm          objective;
  TermCtx          ctx[2];
  ExampleCtx       reference_ctx = {0};
  ExampleReference reference;
  TestOptions      options;
  Vec              x;
  PetscInt         hessian_evals = 0;
  MPI_Comm         comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(TestOptionsSetFromOptions(comm, &options));
  for (PetscInt i = 0; i < options.nterms; i++) {
    ctx[i].separate_callbacks = options.separate_callbacks;
    ctx[i].split_hpre         = options.split_hpre;
    ctx[i].hessian_evals      = &hessian_evals;
  }

  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, n, &x));
  PetscCall(VecSet(x, 0.0));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao, "shell_"));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));

  for (PetscInt i = 0; i < options.nterms; i++) {
    ExampleTerm *term = &options.terms[i];
    char         prefix[16];

    PetscCall(PetscStrncpy(prefix, i ? "extra_" : "data_", sizeof(prefix)));
    if (term->use_map) PetscCall(CreateMap(comm, term->size, n, i ? 0.75 : 1.0, &term->map));
    PetscCall(CreateDataTerm(comm, prefix, &ctx[i], term));
    PetscCall(TaoTermSetParametersMode(term->term, options.parameters_mode));
    if (term->supply_parameters) {
      PetscCall(VecCreateMPI(comm, PETSC_DECIDE, term->size, &term->parameters));
      PetscCall(VecSet(term->parameters, 1.0 + i));
    }
    PetscCall(TaoAddTerm(tao, prefix, term->scale, term->term, term->parameters, term->map));
  }

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(ExampleCheckRequestedHessianType(tao, options.nterms > 1 ? "shell_" : "data_"));
  reference_ctx.nleaves = options.nterms;
  if (options.nterms > 1) PetscCall(TaoGetTerm(tao, NULL, &objective, NULL, NULL));
  for (PetscInt i = 0; i < options.nterms; i++) {
    if (options.nterms > 1) PetscCall(TaoTermSumGetTermMask(objective, i, &options.terms[i].mask));
    PetscCall(ExampleTermSetLeaf(&options.terms[i], &reference_ctx.leaves[i]));
  }
  if (options.check_first_hessian_only)
    PetscCheck(options.nterms == 2 && !(options.terms[0].mask & TAOTERM_MASK_HESSIAN) && (options.terms[1].mask & TAOTERM_MASK_HESSIAN), comm, PETSC_ERR_ARG_WRONG, "-check_first_hessian_only requires only the second term's Hessian to be masked");
  PetscCall(ExampleReferenceCreate(comm, tao, x, &reference_ctx, &reference));
  PetscCall(ExampleReferenceSolveAndCompare(tao, x, &reference));
  PetscCall(PetscOptionsGetBool(NULL, "data_", "-tao_term_hessian_use_fd", &options.terms[0].use_fd, NULL));
  if (options.nterms > 1) PetscCall(PetscOptionsGetBool(NULL, "extra_", "-tao_term_hessian_use_fd", &options.terms[1].use_fd, NULL));
  PetscCall(CheckOperator(tao, x, options.terms, options.nterms, options.split_hpre, options.check_hessian_mult, options.check_hessian_cache, &hessian_evals));
  PetscCall(PetscPrintf(comm, "Least-squares TaoTerm operator check passed\n"));

  PetscCall(ExampleReferenceDestroy(&reference));
  for (PetscInt i = 0; i < options.nterms; i++) PetscCall(ExampleTermDestroy(&options.terms[i]));
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
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

static PetscErrorCode TestOptionsSetFromOptions(MPI_Comm comm, TestOptions *options)
{
  ExampleTerm *data, *extra;
  PetscBool    second_term = PETSC_FALSE, no_map = PETSC_FALSE, provide_hessian_mult = PETSC_TRUE;
  PetscInt     m = 10;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(options, sizeof(*options)));
  data                        = &options->terms[0];
  extra                       = &options->terms[1];
  data->use_map               = PETSC_TRUE;
  extra->use_map              = PETSC_TRUE;
  data->provide_hessian_mult  = PETSC_TRUE;
  extra->provide_hessian_mult = PETSC_TRUE;
  data->supply_parameters     = PETSC_TRUE;
  extra->supply_parameters    = PETSC_TRUE;
  data->scale                 = 1.0;
  extra->scale                = 0.25;
  data->size                  = 10;
  extra->size                 = 12;
  options->parameters_test    = TEST_PARAMETERS_OPTIONAL;
  options->parameters_mode    = TAOTERM_PARAMETERS_OPTIONAL;
  PetscOptionsBegin(comm, "", help, "Tao");
  PetscCall(PetscOptionsBool("-second_term", "Add a second least-squares data term", NULL, second_term, &second_term, NULL));
  PetscCall(PetscOptionsBool("-no_map", "Add terms directly in the Tao solution space", NULL, no_map, &no_map, NULL));
  data->use_map  = !no_map;
  extra->use_map = !no_map;
  PetscCall(PetscOptionsBool("-data_use_map", "Map the first data term", NULL, data->use_map, &data->use_map, NULL));
  PetscCall(PetscOptionsBool("-extra_use_map", "Map the second data term", NULL, extra->use_map, &extra->use_map, NULL));
  PetscCall(PetscOptionsBool("-separate_callbacks", "Register separate objective and gradient callbacks", NULL, options->separate_callbacks, &options->separate_callbacks, NULL));
  PetscCall(PetscOptionsBool("-provide_hessian_mult", "Register the direct Hessian-vector callback on both terms by default", NULL, provide_hessian_mult, &provide_hessian_mult, NULL));
  data->provide_hessian_mult  = provide_hessian_mult;
  extra->provide_hessian_mult = provide_hessian_mult;
  PetscCall(PetscOptionsBool("-data_provide_hessian_mult", "Register HessianMult on the first data term", NULL, data->provide_hessian_mult, &data->provide_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-extra_provide_hessian_mult", "Register HessianMult on the second data term", NULL, extra->provide_hessian_mult, &extra->provide_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-split_hpre", "Use a distinct raw preconditioning matrix equal to two times the Hessian", NULL, options->split_hpre, &options->split_hpre, NULL));
  PetscCall(PetscOptionsBool("-check_hessian_mult", "Compare TaoComputeHessianMult() with the exact least-squares action", NULL, options->check_hessian_mult, &options->check_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-check_first_hessian_only", "Expect only the first data term to contribute to the Hessian", NULL, options->check_first_hessian_only, &options->check_first_hessian_only, NULL));
  PetscCall(PetscOptionsBool("-check_hessian_cache", "Check that assembled Hessian and HessianMult evaluations share cached raw Hessians", NULL, options->check_hessian_cache, &options->check_hessian_cache, NULL));
  PetscCall(PetscOptionsEnum("-parameters", "Parameter contract and whether the test obeys it", NULL, TestParametersTypes, (PetscEnum)options->parameters_test, (PetscEnum *)&options->parameters_test, NULL));
  PetscCall(PetscOptionsInt("-m", "Number of observations in the first data set", NULL, m, &m, NULL));
  PetscOptionsEnd();
  options->nterms = second_term ? 2 : 1;
  data->size      = data->use_map ? m : 10;
  extra->size     = extra->use_map ? 12 : 10;
  if (options->parameters_test == TEST_PARAMETERS_NONE || options->parameters_test == TEST_PARAMETERS_NONE_WITH_PARAMETERS) options->parameters_mode = TAOTERM_PARAMETERS_NONE;
  else if (options->parameters_test == TEST_PARAMETERS_REQUIRED || options->parameters_test == TEST_PARAMETERS_REQUIRED_WITHOUT_PARAMETERS) options->parameters_mode = TAOTERM_PARAMETERS_REQUIRED;
  data->supply_parameters  = options->parameters_test != TEST_PARAMETERS_NONE && options->parameters_test != TEST_PARAMETERS_REQUIRED_WITHOUT_PARAMETERS;
  extra->supply_parameters = data->supply_parameters;
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

static PetscErrorCode CreateDataTerm(MPI_Comm comm, const char prefix[], TermCtx *ctx, ExampleTerm *term)
{
  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, ctx, NULL, &term->term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term->term, prefix));
  PetscCall(TaoTermSetSolutionSizes(term->term, PETSC_DECIDE, term->size, 1));
  PetscCall(TaoTermSetParametersSizes(term->term, PETSC_DECIDE, term->size, 1));
  if (ctx->separate_callbacks) {
    PetscCall(TaoTermShellSetObjective(term->term, ExampleIdentityLeastSquaresObjective));
    PetscCall(TaoTermShellSetGradient(term->term, ExampleIdentityLeastSquaresGradient));
  } else PetscCall(TaoTermShellSetObjectiveAndGradient(term->term, ExampleIdentityLeastSquaresObjectiveGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term->term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term->term, ctx->split_hpre ? PETSC_FALSE : PETSC_TRUE, MATAIJ, ctx->split_hpre ? MATAIJ : NULL));
  PetscCall(TaoTermShellSetHessian(term->term, FormHessian));
  if (term->provide_hessian_mult) PetscCall(TaoTermShellSetHessianMult(term->term, ExampleIdentityLeastSquaresHessianMult));
  PetscCall(TaoTermSetFromOptions(term->term));
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

static PetscErrorCode CheckOperator(Tao tao, Vec x, ExampleTerm terms[], PetscInt nterms, PetscBool split_hpre, PetscBool check_hessian_mult, PetscBool check_hessian_cache, PetscInt *hessian_evals)
{
  Mat       H, Hpre, H2 = NULL;
  Vec       v, actual, expected, expected_pre, work;
  PetscReal error, tolerance = 2.e-5;
  PetscInt  active_hessian_terms = 0, evals_before;
  PetscBool any_hessian_mult     = PETSC_FALSE;

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
    any_hessian_mult = any_hessian_mult || terms[i].provide_hessian_mult;
    if (terms[i].mask & TAOTERM_MASK_HESSIAN) continue;
    PetscCall(AddExpectedAction(terms[i].map, terms[i].scale, v, expected, work));
    PetscCall(AddExpectedAction(terms[i].map, split_hpre && !terms[i].use_fd ? 2.0 * terms[i].scale : terms[i].scale, v, expected_pre, work));
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
      args: -second_term -parameters none -shell_tao_type nls -shell_tao_view ::ascii_info_detail

    test:
      suffix: parameters_required
      args: -second_term -parameters required -shell_tao_type nls -shell_tao_view ::ascii_info_detail

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
    args: -parameters none_with_parameters -shell_tao_type nls -petsc_ci_portable_error_output -error_output_stdout
    filter: grep -E "Parameters passed to a TaoTerm with TAOTERM_PARAMETERS_NONE"

  test:
    suffix: required_without_parameters
    args: -parameters required_without_parameters -shell_tao_type nls -petsc_ci_portable_error_output -error_output_stdout
    filter: grep -E "Parameters required but not provided for a TaoTerm with TAOTERM_PARAMETERS_REQUIRED"

  test:
    suffix: asymmetric_map_assembled
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
    args: -second_term -data_use_map true -extra_use_map false -shell_tao_type nls

  test:
    suffix: mixed_direct_fallback
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
    args: -second_term -data_use_map false -extra_use_map true
    args: -data_provide_hessian_mult false -extra_provide_hessian_mult true
    args: -shell_tao_type nls -shell_tao_term_hessian_mat_type shell

  test:
    suffix: asymmetric_map_mffd
    TODO: TAOTERMSUM does not support an outer MFFD Hessian
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
    args: -second_term -data_use_map true -extra_use_map false
    args: -shell_tao_type nls -shell_tao_term_hessian_mat_type mffd

  testset:
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
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
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
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

  testset:
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
    args: -second_term -shell_tao_type nls -shell_tao_term_hessian_mat_type shell

    test:
      suffix: r126_second_analytic
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r127_second_fd
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_data_mask hessian -extra_tao_term_hessian_use_fd

    test:
      suffix: r128_second_hessianmult
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r129_mapped_second_analytic
      args: -data_use_map false -extra_use_map true -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r130_mapped_second_fd
      args: -data_use_map false -extra_use_map true -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_data_mask hessian
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r131_mapped_second_hessianmult
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r132_first_analytic
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r133_first_fd
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian -data_tao_term_hessian_use_fd

    test:
      suffix: r134_first_hessianmult
      args: -no_map -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r136_mapped_first_fd
      args: -data_use_map true -extra_use_map false -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_extra_mask hessian
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r137_mapped_first_hessianmult
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r139_analytic_fd
      args: -no_map -provide_hessian_mult false -extra_tao_term_hessian_use_fd

    test:
      suffix: r140_analytic_hessianmult
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult true

    test:
      suffix: r141_fd_fd
      args: -no_map -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r142_fd_hessianmult
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult true
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r143_hessianmult_hessianmult
      args: -no_map

    test:
      suffix: r144_mapped_analytic_analytic
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false

    test:
      suffix: r145_mapped_analytic_fd
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r147_mapped_fd_fd
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r151_both_mapped_analytic_fd
      args: -provide_hessian_mult false -extra_tao_term_hessian_use_fd

    test:
      suffix: r152_both_mapped_analytic_hessianmult
      args: -data_provide_hessian_mult false -extra_provide_hessian_mult true

    test:
      suffix: r153_both_mapped_fd_fd
      args: -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r154_both_mapped_fd_hessianmult
      args: -data_provide_hessian_mult false -extra_provide_hessian_mult true
      args: -data_tao_term_hessian_use_fd

  testset:
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
    args: -second_term -split_hpre -shell_tao_type nls
    args: -shell_tao_term_hessian_mat_type shell -shell_tao_term_hessian_pre_is_hessian false
    args: -shell_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r156_second_analytic_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r157_second_fd_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_data_mask hessian -extra_tao_term_hessian_use_fd

    test:
      suffix: r158_second_hessianmult_separate_hpre
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r159_mapped_second_analytic_separate_hpre
      args: -data_use_map false -extra_use_map true -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r160_mapped_second_fd_separate_hpre
      args: -data_use_map false -extra_use_map true -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_data_mask hessian
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r161_mapped_second_hessianmult_separate_hpre
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r162_first_analytic_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r163_first_fd_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult false
      args: -shell_tao_term_sum_extra_mask hessian -data_tao_term_hessian_use_fd

    test:
      suffix: r164_first_hessianmult_separate_hpre
      args: -no_map -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r165_mapped_first_analytic_separate_hpre
      args: -data_use_map true -extra_use_map false -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r166_mapped_first_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -data_provide_hessian_mult false
      args: -extra_provide_hessian_mult false -shell_tao_term_sum_extra_mask hessian
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r167_mapped_first_hessianmult_separate_hpre
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r169_analytic_fd_separate_hpre
      args: -no_map -provide_hessian_mult false -extra_tao_term_hessian_use_fd

    test:
      suffix: r170_analytic_hessianmult_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult true

    test:
      suffix: r171_fd_fd_separate_hpre
      args: -no_map -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r172_fd_hessianmult_separate_hpre
      args: -no_map -data_provide_hessian_mult false -extra_provide_hessian_mult true
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r173_hessianmult_hessianmult_separate_hpre
      args: -no_map

    test:
      suffix: r174_mapped_analytic_analytic_separate_hpre
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false

    test:
      suffix: r175_mapped_analytic_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false
      args: -extra_tao_term_hessian_use_fd

    test:
      suffix: r177_mapped_fd_fd_separate_hpre
      args: -data_use_map true -extra_use_map false -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r179_mapped_hessianmult_hessianmult_separate_hpre
      args: -data_use_map true -extra_use_map false

    test:
      suffix: r181_both_mapped_analytic_fd_separate_hpre
      args: -provide_hessian_mult false -extra_tao_term_hessian_use_fd

    test:
      suffix: r182_both_mapped_analytic_hessianmult_separate_hpre
      args: -data_provide_hessian_mult false -extra_provide_hessian_mult true

    test:
      suffix: r183_both_mapped_fd_fd_separate_hpre
      args: -provide_hessian_mult false
      args: -data_tao_term_hessian_use_fd -extra_tao_term_hessian_use_fd

    test:
      suffix: r184_both_mapped_fd_hessianmult_separate_hpre
      args: -data_provide_hessian_mult false -extra_provide_hessian_mult true
      args: -data_tao_term_hessian_use_fd

    test:
      suffix: r185_both_mapped_hessianmult_hessianmult_separate_hpre

  testset:
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
    args: -second_term -shell_tao_type nls -shell_tao_term_hessian_mat_type mffd

    test:
      suffix: r186_mffd_second
      TODO: TAOTERMSUM does not support an outer MFFD Hessian
      args: -no_map -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r187_mffd_mapped_second
      TODO: TAOTERMSUM does not support an outer MFFD Hessian
      args: -data_use_map false -extra_use_map true -shell_tao_term_sum_data_mask hessian

    test:
      suffix: r188_mffd_first
      TODO: TAOTERMSUM does not support an outer MFFD Hessian
      args: -no_map -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r189_mffd_mapped_first
      TODO: TAOTERMSUM does not support an outer MFFD Hessian
      args: -data_use_map true -extra_use_map false -shell_tao_term_sum_extra_mask hessian

    test:
      suffix: r190_mffd_both
      TODO: TAOTERMSUM does not support an outer MFFD Hessian
      args: -no_map

    test:
      suffix: r192_mffd_both_mapped
      TODO: TAOTERMSUM does not support an outer MFFD Hessian

  test:
    suffix: r018_hessianmult_separate_hpre
    filter: grep -E "Reference callback comparison passed|operator check passed"
    output_file: output/taotermtest2_reference_operator_comparison.out
    args: -no_map -split_hpre -shell_tao_type nls
    args: -data_tao_term_hessian_mat_type shell -data_tao_term_hessian_pre_is_hessian false
    args: -data_tao_term_hessian_pre_mat_type aij

TEST*/
