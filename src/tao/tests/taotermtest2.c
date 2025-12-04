#include <petsctao.h>

static char help[] = "Using TaoTermShell with mapping matrices that are not diagonal.\n";

typedef struct {
  Vec pdiff_work; /* Work vector for x - params */
} HalfL2Ctx;

typedef struct {
  Mat A;      /* Mapping matrix A */
  Vec p;      /* Target vector p */
  Vec Ax;     /* Work vector for A*x */
  Vec Ax_p;   /* Work vector for A*x - p */
} CallbackCtx;

static PetscErrorCode FormFunctionGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode CtxDestroy(void **ctx);

/* Callback functions for traditional TAO interface */
static PetscErrorCode FormObjectiveGradient_Callback(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_Callback(Tao, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  TaoTerm      objective;
  Tao          tao, tao2;
  PetscMPIInt  size;
  HalfL2Ctx    *ctx;
  MPI_Comm     comm;
  PetscInt     n = 10, m = 10;
  Mat          A;
  Vec          target;
  PetscRandom  rand;
  CallbackCtx  cb_ctx;
  Vec          x_term, x_callback, x2, diff;
  Mat          H2;
  PetscReal    norm_diff, rel_diff, diag_val;
  PetscBool    opt, is_diag, is_cdiag, is_aij, is_dense;
  const char  *mtype = MATAIJ;
  char         typeName[256] = "";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsInt("-n", "Problem size", "", n, &n, NULL));
  PetscCall(PetscOptionsInt("-m", "Mapping matrix row size", "", m, &m, NULL));
  PetscCall(PetscOptionsFList("-mapping_mtype", "Mapping matrix type", "", MatList, mtype, typeName, 256, &opt));
  PetscOptionsEnd();

  PetscCall(PetscNew(&ctx));
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  /* Initialize typeName to default if option was not set */
  if (!opt) PetscCall(PetscStrcpy(typeName, mtype));

  PetscCall(PetscStrcmp(typeName, MATDIAGONAL, &is_diag));
  PetscCall(PetscStrcmp(typeName, MATCONSTANTDIAGONAL, &is_cdiag));
  PetscCall(PetscStrcmp(typeName, MATAIJ, &is_aij));
  PetscCall(PetscStrcmp(typeName, MATDENSE, &is_dense));
  /* Create mapping matrix A: m x n (maps from solution space to term space) */
  if (is_diag) {
    /* Create a diagonal matrix */
    Vec      diag_vec;
    PetscInt diag_size;

    PetscCheck(m == n, comm, PETSC_ERR_ARG_INCOMP, "For diagonal matrix, m and n must be equal (got m=%" PetscInt_FMT ", n=%" PetscInt_FMT ")", m, n);
    diag_size = m;
    PetscCall(VecCreate(comm, &diag_vec));
    PetscCall(VecSetSizes(diag_vec, PETSC_DECIDE, diag_size));
    PetscCall(VecSetFromOptions(diag_vec));
    PetscCall(VecSetRandom(diag_vec, rand));
    PetscCall(MatCreateDiagonal(diag_vec, &A));
    PetscCall(VecDestroy(&diag_vec));
  } else if (is_cdiag) {
    /* Create a constant diagonal matrix */
    PetscScalar diag_val;
    PetscCheck(m == n, comm, PETSC_ERR_ARG_INCOMP, "For constant diagonal matrix, m and n must be equal (got m=%" PetscInt_FMT ", n=%" PetscInt_FMT ")", m, n);
    PetscCall(PetscRandomGetValue(rand, &diag_val));
    PetscCall(MatCreateConstantDiagonal(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, diag_val, &A));
  } else if (is_dense) {
    /* Create a dense matrix */
    PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetRandom(A, rand));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  } else {
    /* Create an AIJ matrix (default) */
    PetscCall(MatCreateSeqAIJ(comm, m, n, PETSC_DEFAULT, NULL, &A));
    PetscCall(MatSetRandom(A, rand));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }

  /* Create shell term that computes f(x) = 0.5 ||x||_2^2 */
  PetscCall(TaoTermCreateShell(comm, (void *)ctx, CtxDestroy, &objective));

  /* Set solution and parameter sizes to match the mapped space (m) */
  PetscCall(TaoTermSetSolutionSizes(objective, PETSC_DECIDE, m, 1));
  PetscCall(TaoTermSetParametersSizes(objective, PETSC_DECIDE, m, 1));

  PetscCall(TaoTermShellSetObjectiveAndGradient(objective, FormFunctionGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(objective, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(objective, PETSC_TRUE /* H == Hpre */, MATAIJ, NULL));
  PetscCall(TaoTermShellSetHessian(objective, FormHessian));
  PetscCall(TaoTermSetFromOptions(objective));

  PetscCall(TaoTermSetUp(objective));

  /* Create target vector for least squares problem (parameters) */
  PetscCall(TaoTermCreateParametersVec(objective, &target));
  PetscCall(VecSetRandom(target, rand));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao, "shell_"));
  PetscCall(TaoSetType(tao, TAOLMVM));

  /* Add term with mapping matrix A: f(Ax; p) = 0.5 ||Ax - p||_2^2 */
  PetscCall(TaoAddTerm(tao, NULL, 1.0, objective, target, A));

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  /* Allocate callback context */
  PetscCall(PetscMemzero(&cb_ctx, sizeof(CallbackCtx)));
  PetscCall(PetscObjectReference((PetscObject)A));
  PetscCall(PetscObjectReference((PetscObject)target));
  cb_ctx.A = A;
  cb_ctx.p = target;

  /* Create work vectors */
  PetscCall(MatCreateVecs(A, &cb_ctx.Ax, NULL));
  PetscCall(VecDuplicate(target, &cb_ctx.Ax_p));

  /* Create solution vector for callback version (size n, not m) - initialize to zero like tao */
  PetscCall(MatCreateVecs(A, NULL, &x2));
  PetscCall(VecZeroEntries(x2));

  /* Create Hessian matrix A^T * A */
  if (is_diag) {
    Vec A_diag, H2_diag;

    PetscCall(MatCreateVecs(A, &A_diag, NULL));
    PetscCall(MatGetDiagonal(A, A_diag));
    PetscCall(VecDuplicate(A_diag, &H2_diag));
    PetscCall(VecPointwiseMult(H2_diag, A_diag, A_diag));
    PetscCall(MatCreateDiagonal(H2_diag, &H2));
    PetscCall(VecDestroy(&A_diag));
    PetscCall(VecDestroy(&H2_diag));
  } else if (is_cdiag) {
    PetscScalar diag_val_squared;
    PetscInt    M, N;

    PetscCall(MatConstantDiagonalGetConstant(A, &diag_val));
    diag_val_squared = diag_val * diag_val;
    PetscCall(MatGetSize(A, &M, &N));
    PetscCall(MatCreateConstantDiagonal(comm, PETSC_DECIDE, PETSC_DECIDE, N, N, diag_val_squared, &H2));
  } else {
    PetscCall(MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &H2));
    PetscCall(MatAssemblyBegin(H2, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H2, MAT_FINAL_ASSEMBLY));
  }
  /* Create second TAO solver */
  PetscCall(TaoCreate(comm, &tao2));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao2, "regular_"));
  PetscCall(TaoSetType(tao2, TAOLMVM));
  PetscCall(TaoSetSolution(tao2, x2));
  PetscCall(TaoSetObjectiveAndGradient(tao2, NULL, FormObjectiveGradient_Callback, &cb_ctx));
  PetscCall(TaoSetHessian(tao2, H2, H2, FormHessian_Callback, &cb_ctx));
  PetscCall(TaoSetFromOptions(tao2));
  PetscCall(TaoSolve(tao2));

  /* Compare solutions */
  PetscCall(TaoGetSolution(tao, &x_term));
  PetscCall(TaoGetSolution(tao2, &x_callback));
  PetscCall(VecDuplicate(x_term, &diff));
  PetscCall(VecCopy(x_term, diff));
  PetscCall(VecAXPY(diff, -1.0, x_callback));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));
  rel_diff = norm_diff / PetscMax(norm_diff, 1.0e-10);
  PetscCall(PetscPrintf(comm, "Solutions match (relative difference < 1e-12): %6.10e\n", (double)rel_diff));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&diff));
  PetscCall(MatDestroy(&H2));
  PetscCall(TaoDestroy(&tao2));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&objective));
  PetscCall(PetscRandomDestroy(&rand));
  return 0;
}


/*
  FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

  Input Parameters:
+ term      - the `TaoTerm` for the objective function
. x         - input vector
- params    - optional vector of parameters

  Output Parameters:
+ f - function value
- G - vector containing the newly evaluated gradient

  Note:
  Computes f = 0.5 * ||x - params||_2^2 and g = x - params, matching TAOTERMHALFL2SQUARED.
*/
static PetscErrorCode FormFunctionGradient(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec G)
{
  HalfL2Ctx   *ctx;
  PetscScalar  v;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  if (params) {
    PetscCall(VecWAXPY(G, -1.0, params, x));
    PetscCall(VecDot(G, G, &v));
  } else {
    PetscCall(VecCopy(x, G));
    PetscCall(VecDot(G, G, &v));
  }
  *f = 0.5 * PetscRealPart(v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormHessian - Evaluates Hessian matrix.

  Input Parameters:
+ term      - the `TaoTerm` for the objective function
. x         - input vector
. params    - optional vector of parameters
- Hpre      - optional preconditioner matrix

  Output Parameters:
+ H    - Hessian matrix
- Hpre - Preconditioning matrix

  Note:
  Computes H = I (identity matrix), matching TAOTERMHALFL2SQUARED.
*/
static PetscErrorCode FormHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
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
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode CtxDestroy(void **ctx_ptr)
{
  HalfL2Ctx *ctx = *(HalfL2Ctx **)ctx_ptr;

  PetscFunctionBegin;
  if (ctx) {
    PetscCall(VecDestroy(&ctx->pdiff_work));
    PetscCall(PetscFree(ctx));
    *ctx_ptr = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormObjectiveGradient_Callback - Evaluates the objective and gradient for traditional TAO callback interface.

  Input Parameters:
+ tao  - the Tao solver context
. x    - input vector (size n)
- ctx  - user context containing A and p

  Output Parameters:
+ f - function value: 0.5 * ||Ax - p||_2^2
- g - gradient vector: A^T (Ax - p)

  Note:
  Computes f = 0.5 * ||Ax - p||_2^2 and g = A^T (Ax - p)
*/
static PetscErrorCode FormObjectiveGradient_Callback(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  CallbackCtx *cb_ctx = (CallbackCtx *)ctx;
  PetscScalar  v;

  PetscFunctionBeginUser;
  /* Compute Ax */
  PetscCall(MatMult(cb_ctx->A, x, cb_ctx->Ax));
  /* Compute Ax - p */
  PetscCall(VecCopy(cb_ctx->Ax, cb_ctx->Ax_p));
  PetscCall(VecAXPY(cb_ctx->Ax_p, -1.0, cb_ctx->p));
  /* Compute objective: 0.5 * ||Ax - p||_2^2 */
  PetscCall(VecDot(cb_ctx->Ax_p, cb_ctx->Ax_p, &v));
  *f = 0.5 * PetscRealPart(v);
  /* Compute gradient: A^T (Ax - p) */
  PetscCall(MatMultTranspose(cb_ctx->A, cb_ctx->Ax_p, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormHessian_Callback - Evaluates the Hessian matrix for traditional TAO callback interface.

  Input Parameters:
+ tao  - the Tao solver context
. x    - input vector
. H    - Hessian matrix (should be pre-allocated as A^T * A)
. Hpre - preconditioner matrix
- ctx  - user context containing A and p

  Output Parameters:
+ H    - Hessian matrix (A^T * A)
- Hpre - Preconditioning matrix

  Note:
  The Hessian for 0.5 * ||Ax - p||_2^2 is constant: H = A^T * A
*/
static PetscErrorCode FormHessian_Callback(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscFunctionBeginUser;
  /* Hessian is constant: A^T * A, which should already be set in H */
  if (Hpre && Hpre != H) PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

   test:
     args: -shell_tao_type nls -shell_tao_view ::ascii_info_detail -regular_tao_type nls -regular_tao_view ::ascii_info_detail
     args: -tao_term_hessian_mat_type diagonal

TEST*/
