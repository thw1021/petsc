const char help[] = "Coverage tests for TAOTERMGAUSSNEWTON";

#include <petsctao.h>
#include <petsctaoterm.h>

/* Linear residual R(x) = A x - b, so 0.5 ||R||^2 has gradient A^T (Ax - b) and
   Gauss-Newton Hessian A^T A. */
typedef struct {
  Mat A;
  Vec b;
} ResidualCtx;

static PetscErrorCode ComputeResidual(Tao tao, Vec x, Vec r, void *ctx)
{
  ResidualCtx *rc = (ResidualCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(MatMult(rc->A, x, r));
  PetscCall(VecAXPY(r, -1.0, rc->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeResidualJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
  PetscFunctionBeginUser;
  /* J = A is constant; nothing to do. */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckGaussNewtonTerm(MPI_Comm comm, MatType H_type, PetscBool hessian_mult_only)
{
  Tao         tao;
  TaoTerm     gn;
  Mat         A, Jres, J_from_term, AtA = NULL;
  Vec         b, x, r, g, g_ref, Hv, Hv_ref, v;
  PetscReal   obj, obj_ref;
  PetscInt    m = 5, n = 3;
  ResidualCtx rc;

  PetscFunctionBeginUser;
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &A));
  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSetRandom(b, NULL));
  PetscCall(VecSetRandom(x, NULL));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(VecDuplicate(x, &g_ref));
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecSetRandom(v, NULL));
  PetscCall(VecDuplicate(x, &Hv));
  PetscCall(VecDuplicate(x, &Hv_ref));
  rc.A = A;
  rc.b = b;

  /* Build a Tao that owns the residual machinery; the term will pull from it. */
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetResidualRoutine(tao, r, ComputeResidual, &rc));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Jres));
  PetscCall(TaoSetJacobianResidualRoutine(tao, Jres, Jres, ComputeResidualJacobian, &rc));

  PetscCall(TaoTermCreateGaussNewton(tao, &gn));

  if (H_type) PetscCall(TaoTermSetCreateHessianMode(gn, PETSC_TRUE, H_type, H_type));

  /* Reference values. */
  PetscCall(MatMult(A, x, r));
  PetscCall(VecAXPY(r, -1.0, b));
  {
    PetscScalar dot;
    PetscCall(VecDot(r, r, &dot));
    obj_ref = 0.5 * PetscRealPart(dot);
  }
  PetscCall(MatMultTranspose(A, r, g_ref));

  /* Objective + gradient via the term. */
  PetscCall(TaoTermComputeObjectiveAndGradient(gn, x, NULL, &obj, g));
  PetscCheck(PetscAbsReal(obj - obj_ref) < 1e-10 * PetscMax(1.0, PetscAbsReal(obj_ref)), comm, PETSC_ERR_PLIB, "objective mismatch: %g vs %g", (double)obj, (double)obj_ref);
  PetscCall(VecAXPY(g, -1.0, g_ref));
  {
    PetscReal err;
    PetscCall(VecNorm(g, NORM_2, &err));
    PetscCheck(err < 1e-9, comm, PETSC_ERR_PLIB, "gradient mismatch, ||err|| = %g", (double)err);
  }

  /* Cached Jacobian getter. */
  PetscCall(TaoTermGaussNewtonGetJacobian(gn, x, &J_from_term));
  PetscCheck(J_from_term == Jres, comm, PETSC_ERR_PLIB, "Jacobian getter returned wrong Mat");

  /* Hessian mult against (A^T A) v. */
  PetscCall(MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &AtA));
  PetscCall(MatMult(AtA, v, Hv_ref));
  PetscCall(TaoTermComputeHessianMult(gn, x, NULL, v, Hv));
  PetscCall(VecAXPY(Hv, -1.0, Hv_ref));
  {
    PetscReal err;
    PetscCall(VecNorm(Hv, NORM_2, &err));
    PetscCheck(err < 1e-9, comm, PETSC_ERR_PLIB, "HessianMult mismatch, ||err|| = %g", (double)err);
  }

  /* Assembled Hessian path: skip for explicit-only test rows where requested. */
  if (!hessian_mult_only) {
    Mat       H, Hpre;
    Vec       Hv2;
    PetscReal err;

    PetscCall(TaoTermCreateHessianMatrices(gn, &H, &Hpre));
    PetscCall(TaoTermComputeHessian(gn, x, NULL, H, Hpre));
    PetscCall(VecDuplicate(Hv_ref, &Hv2));
    PetscCall(MatMult(H, v, Hv2));
    PetscCall(VecAXPY(Hv2, -1.0, Hv_ref));
    PetscCall(VecNorm(Hv2, NORM_2, &err));
    PetscCheck(err < 1e-9, comm, PETSC_ERR_PLIB, "Hessian mismatch, ||err|| = %g", (double)err);
    PetscCall(VecDestroy(&Hv2));
    PetscCall(MatDestroy(&H));
    PetscCall(MatDestroy(&Hpre));
  }

  PetscCall(MatDestroy(&AtA));
  PetscCall(TaoTermDestroy(&gn));
  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&Jres));
  PetscCall(VecDestroy(&Hv_ref));
  PetscCall(VecDestroy(&Hv));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&g_ref));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Default: shell H/Hpre. */
  PetscCall(CheckGaussNewtonTerm(PETSC_COMM_WORLD, NULL, PETSC_FALSE));
  /* Explicit assembled MATAIJ (or MATDENSE on dense Jacobian). */
  PetscCall(CheckGaussNewtonTerm(PETSC_COMM_WORLD, MATAIJ, PETSC_FALSE));
  /* HessianMult-only path. */
  PetscCall(CheckGaussNewtonTerm(PETSC_COMM_WORLD, MATSHELL, PETSC_TRUE));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/empty.out

TEST*/
