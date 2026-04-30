#include <petsctao.h>

static char help[] = "Single mapped TaoTerm solved with shell matrix for matrix-free HessianMUlt.\n\
The objective is 0.5||Ax - p||_2^2, with a TAOTERMSHELL f(z) = 0.5||z - p||_2^2 that provides\n\
only a Hessian-vector product (no assembled Hessian), added with a mapping matrix A via TaoAddTerm().\n\
With -tao_term_hessian_mat_type shell the outer Hessian must apply A^T (grad^2 f)(Ax) A matrix-free.\n\
The solution is compared against a traditional callback solve whose Hessian is the assembled A^T A.\n";

typedef struct {
  Mat A;    /* Mapping matrix A */
  Vec p;    /* Target vector p */
  Vec Ax;   /* Work vector for A*x */
  Vec Ax_p; /* Work vector for A*x - p */
} CallbackCtx;

/* TaoTerm shell callbacks: f(z) = 0.5||z - p||_2^2, so grad = z - p and grad^2 = I */
static PetscErrorCode FormFunctionGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessianMult(TaoTerm, Vec, Vec, Vec, Vec);

/* Traditional callback interface for the reference solve */
static PetscErrorCode FormObjectiveGradient_Callback(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_Callback(Tao, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  TaoTerm      objective;
  Tao          tao, tao2;
  PetscMPIInt  size;
  MPI_Comm     comm;
  PetscInt     n = 10, m = 10;
  Mat          A, H2;
  Vec          target;
  CallbackCtx *cb_ctx;
  Vec          x_term, x_callback, x2, diff;
  PetscReal    norm_diff, diag_val = 1.1;
  PetscBool    opt, is_diag, is_cdiag, is_dense;
  const char  *mtype         = MATAIJ;
  char         typeName[256] = "";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsInt("-n", "Problem size", "", n, &n, NULL));
  PetscCall(PetscOptionsInt("-m", "Mapping matrix row size", "", m, &m, NULL));
  PetscCall(PetscOptionsReal("-diag_val", "Value of constant diagonal matrix", NULL, diag_val, &diag_val, NULL));
  PetscCall(PetscOptionsFList("-mapping_mtype", "Mapping matrix type", "", MatList, mtype, typeName, 256, &opt));
  PetscOptionsEnd();

  if (!opt) PetscCall(PetscStrcpy(typeName, mtype));
  PetscCall(PetscStrcmp(typeName, MATDIAGONAL, &is_diag));
  PetscCall(PetscStrcmp(typeName, MATCONSTANTDIAGONAL, &is_cdiag));
  PetscCall(PetscStrcmp(typeName, MATDENSE, &is_dense));

  /* Create mapping matrix A: m x n (maps solution space to term space) */
  if (is_diag) {
    Vec diag_vec;

    PetscCheck(m == n, comm, PETSC_ERR_ARG_INCOMP, "For diagonal matrix, m and n must be equal (got m=%" PetscInt_FMT ", n=%" PetscInt_FMT ")", m, n);
    PetscCall(VecCreate(comm, &diag_vec));
    PetscCall(VecSetSizes(diag_vec, PETSC_DECIDE, m));
    PetscCall(VecSetFromOptions(diag_vec));
    PetscCall(VecSetRandom(diag_vec, NULL));
    PetscCall(MatCreateDiagonal(diag_vec, &A));
    PetscCall(VecDestroy(&diag_vec));
  } else if (is_cdiag) {
    PetscCheck(m == n, comm, PETSC_ERR_ARG_INCOMP, "For constant diagonal matrix, m and n must be equal (got m=%" PetscInt_FMT ", n=%" PetscInt_FMT ")", m, n);
    PetscCall(MatCreateConstantDiagonal(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, diag_val, &A));
  } else if (is_dense) {
    PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetRandom(A, NULL));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  } else {
    PetscCall(MatCreateSeqAIJ(comm, m, n, PETSC_DEFAULT, NULL, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetRandom(A, NULL));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }

  /* Shell term f(z) = 0.5||z - p||_2^2 with only a Hessian-vector product (no assembled Hessian) */
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &objective));
  PetscCall(TaoTermSetSolutionSizes(objective, PETSC_DECIDE, m, 1));
  PetscCall(TaoTermSetParametersSizes(objective, PETSC_DECIDE, m, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(objective, FormFunctionGradient));
  PetscCall(TaoTermShellSetHessianMult(objective, FormHessianMult));
  PetscCall(TaoTermShellSetCreateHessianMatrices(objective, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(objective, PETSC_TRUE /* H == Hpre */, MATSHELL, NULL));
  PetscCall(TaoTermSetFromOptions(objective));
  PetscCall(TaoTermSetUp(objective));

  /* Target vector p (parameters) */
  PetscCall(TaoTermCreateParametersVec(objective, &target));
  PetscCall(VecSetRandom(target, NULL));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao, "shell_"));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoAddTerm(tao, NULL, 1.0, objective, target, A));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  /* Reference solve: traditional callbacks, assembled Hessian A^T A */
  PetscCall(PetscNew(&cb_ctx));
  cb_ctx->A = A;
  cb_ctx->p = target;
  PetscCall(MatCreateVecs(A, NULL, &cb_ctx->Ax));
  PetscCall(VecDuplicate(target, &cb_ctx->Ax_p));
  PetscCall(MatCreateVecs(A, &x2, NULL));

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
    PetscCall(MatCreateConstantDiagonal(comm, PETSC_DECIDE, PETSC_DECIDE, n, n, diag_val * diag_val, &H2));
  } else {
    PetscCall(MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &H2));
    PetscCall(MatAssemblyBegin(H2, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H2, MAT_FINAL_ASSEMBLY));
  }

  PetscCall(TaoCreate(comm, &tao2));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao2, "regular_"));
  PetscCall(TaoSetType(tao2, TAONLS));
  PetscCall(TaoSetSolution(tao2, x2));
  PetscCall(TaoSetObjectiveAndGradient(tao2, NULL, FormObjectiveGradient_Callback, cb_ctx));
  PetscCall(TaoSetHessian(tao2, H2, H2, FormHessian_Callback, cb_ctx));
  PetscCall(TaoSetFromOptions(tao2));
  PetscCall(TaoSolve(tao2));

  PetscCall(TaoGetSolution(tao, &x_term));
  PetscCall(TaoGetSolution(tao2, &x_callback));
  PetscCall(VecDuplicate(x_term, &diff));
  PetscCall(VecCopy(x_term, diff));
  PetscCall(VecAXPY(diff, -1.0, x_callback));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));
  if (norm_diff <= 1.e-6) PetscCall(PetscPrintf(comm, "Relative difference < 1e-6\n"));
  else PetscCall(PetscPrintf(comm, "Relative difference > 1e-6: %6.10e\n", (double)norm_diff));

  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&cb_ctx->Ax));
  PetscCall(VecDestroy(&cb_ctx->Ax_p));
  PetscCall(PetscFree(cb_ctx));
  PetscCall(VecDestroy(&target));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&H2));
  PetscCall(TaoDestroy(&tao2));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&objective));
  PetscCall(PetscFinalize());
  return 0;
}

/* f(z) = 0.5||z - p||_2^2, grad = z - p (matching TAOTERMHALFL2SQUARED) */
static PetscErrorCode FormFunctionGradient(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec G)
{
  PetscScalar v;

  PetscFunctionBeginUser;
  if (params) PetscCall(VecWAXPY(G, -1.0, params, x));
  else PetscCall(VecCopy(x, G));
  PetscCall(VecDot(G, G, &v));
  *f = 0.5 * PetscRealPart(v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* grad^2 f = I, so the Hessian-vector product is Hv = v */
static PetscErrorCode FormHessianMult(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Reference: f = 0.5||Ax - p||_2^2, g = A^T (Ax - p) */
static PetscErrorCode FormObjectiveGradient_Callback(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  CallbackCtx *cb_ctx = (CallbackCtx *)ctx;
  PetscScalar  v;

  PetscFunctionBeginUser;
  PetscCall(MatMult(cb_ctx->A, x, cb_ctx->Ax));
  PetscCall(VecWAXPY(cb_ctx->Ax_p, -1.0, cb_ctx->p, cb_ctx->Ax));
  PetscCall(VecDot(cb_ctx->Ax_p, cb_ctx->Ax_p, &v));
  *f = 0.5 * PetscRealPart(v);
  PetscCall(MatMultTranspose(cb_ctx->A, cb_ctx->Ax_p, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Reference Hessian is constant A^T A, already assembled into H */
static PetscErrorCode FormHessian_Callback(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscFunctionBeginUser;
  if (Hpre && Hpre != H) PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

   testset:
     output_file: output/taotermtest-hm.out
     args: -shell_tao_type nls -regular_tao_type nls -tao_term_hessian_mat_type shell

     test:
       suffix: shell_dense
       args: -mapping_mtype dense

     test:
       suffix: shell_dense_nsq
       args: -mapping_mtype dense -m 15

     test:
       suffix: shell_aij
       args: -mapping_mtype aij

     test:
       suffix: shell_diag
       args: -mapping_mtype diagonal

TEST*/
