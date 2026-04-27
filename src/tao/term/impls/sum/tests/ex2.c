const char help[] = "Tests for TAOTERMSUM nesting and lazy Hessian via MATSHELL";

#include <petsc/private/taoimpl.h>

static PetscErrorCode QuadraticObjective(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  PetscFunctionBeginUser;
  PetscCall(VecDot(x, x, value));
  *value *= 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QuadraticGradient(TaoTerm term, Vec x, Vec params, Vec g)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QuadraticHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBeginUser;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatShift(H, 1.0));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode QuadraticHessianMult(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateQuadraticHessianMatrices(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscInt n;

  PetscFunctionBeginUser;
  PetscCall(TaoTermGetSolutionSizes(term, NULL, &n, NULL));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)term), PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, H));
  PetscCall(MatSetUp(*H));
  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectReference((PetscObject)*H));
  *Hpre = *H;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateQuadraticTerm(MPI_Comm comm, PetscInt n, TaoTerm *term)
{
  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, term));
  PetscCall(TaoTermSetSolutionSizes(*term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetParametersMode(*term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermShellSetObjective(*term, QuadraticObjective));
  PetscCall(TaoTermShellSetGradient(*term, QuadraticGradient));
  PetscCall(TaoTermShellSetHessian(*term, QuadraticHessian));
  PetscCall(TaoTermShellSetHessianMult(*term, QuadraticHessianMult));
  PetscCall(TaoTermShellSetCreateHessianMatrices(*term, CreateQuadraticHessianMatrices));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestNesting(MPI_Comm comm)
{
  TaoTerm   outer, inner, sub_a, sub_b;
  PetscInt  n = 4, n_terms;
  Vec       x, g;
  PetscReal obj;

  PetscFunctionBeginUser;
  // Create inner sum: 2*f(x) + 3*f(x) = 5*f(x)
  PetscCall(TaoTermCreate(comm, &inner));
  PetscCall(TaoTermSetType(inner, TAOTERMSUM));
  PetscCall(CreateQuadraticTerm(comm, n, &sub_a));
  PetscCall(TaoTermSumAddTerm(inner, NULL, 2.0, sub_a, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub_a));
  PetscCall(CreateQuadraticTerm(comm, n, &sub_b));
  PetscCall(TaoTermSumAddTerm(inner, NULL, 3.0, sub_b, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub_b));

  // Create outer sum and add inner as nested term with scale 4: 4*(5*f(x)) = 20*f(x)
  PetscCall(TaoTermCreate(comm, &outer));
  PetscCall(TaoTermSetType(outer, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(outer, NULL, 4.0, inner, NULL, NULL));
  PetscCall(TaoTermDestroy(&inner));

  PetscCall(TaoTermSumGetNumberTerms(outer, &n_terms));
  PetscCheck(n_terms == 1, comm, PETSC_ERR_PLIB, "Expected 1 nested term, got %" PetscInt_FMT, n_terms);

  PetscCall(TaoTermSetUp(outer));
  PetscCall(TaoTermCreateSolutionVec(outer, &x));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(VecSet(x, 1.0));

  // f(x) = 0.5 * ||x||^2 = 0.5 * 4 = 2.0, so 20 * f(x) = 40.0
  PetscCall(TaoTermComputeObjective(outer, x, NULL, &obj));
  PetscCheck(PetscAbsReal(obj - 40.0) < 1e-10, comm, PETSC_ERR_PLIB, "Nested objective: expected 40.0, got %g", (double)obj);

  PetscCall(TaoTermComputeGradient(outer, x, NULL, g));
  {
    PetscReal gnorm;

    PetscCall(VecNorm(g, NORM_2, &gnorm));
    // grad = 20 * x, ||grad|| = 20 * ||x|| = 20 * 2 = 40
    PetscCheck(PetscAbsReal(gnorm - 40.0) < 1e-10, comm, PETSC_ERR_PLIB, "Nested gradient norm: expected 40.0, got %g", (double)gnorm);
  }

  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&x));
  PetscCall(TaoTermDestroy(&outer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestLazyHessian(MPI_Comm comm)
{
  TaoTerm   sum, sub_a, sub_b;
  PetscInt  n = 4;
  Vec       x, v, Hv;
  Mat       H, Hpre;
  PetscBool is_shell;

  PetscFunctionBeginUser;
  // Create sum: 2*f(x) + 3*f(x) = 5*f(x), where f(x) = 0.5*||x||^2
  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(CreateQuadraticTerm(comm, n, &sub_a));
  PetscCall(TaoTermSumAddTerm(sum, NULL, 2.0, sub_a, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub_a));
  PetscCall(CreateQuadraticTerm(comm, n, &sub_b));
  PetscCall(TaoTermSumAddTerm(sum, NULL, 3.0, sub_b, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub_b));

  PetscCall(TaoTermSetUp(sum));
  PetscCall(TaoTermCreateSolutionVec(sum, &x));
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &Hv));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecSet(v, 1.0));

  // Default H should be MATSHELL
  PetscCall(TaoTermCreateHessianMatrices(sum, &H, &Hpre));
  PetscCall(PetscObjectTypeCompare((PetscObject)H, MATSHELL, &is_shell));
  PetscCheck(is_shell, comm, PETSC_ERR_PLIB, "Expected MATSHELL for default H");
  PetscCheck(H != Hpre, comm, PETSC_ERR_PLIB, "Expected H != Hpre (Hpre_is_H should be FALSE)");

  // TaoTermComputeHessian should handle the MATSHELL: update shell, assemble Hpre
  PetscCall(TaoTermComputeHessian(sum, x, NULL, H, Hpre));

  // MatMult on the MATSHELL should give 5*v (since Hessian = 5*I)
  PetscCall(MatMult(H, v, Hv));
  {
    PetscReal norm;

    PetscCall(VecNorm(Hv, NORM_2, &norm));
    // Hv = 5*v, ||Hv|| = 5*||v|| = 5*2 = 10
    PetscCheck(PetscAbsReal(norm - 10.0) < 1e-10, comm, PETSC_ERR_PLIB, "MATSHELL MatMult norm: expected 10.0, got %g", (double)norm);
  }

  // Hpre should also be correct (assembled MATAIJ)
  {
    PetscReal trace;

    PetscCall(MatGetTrace(Hpre, &trace));
    // Hpre = 5*I, trace = 5*4 = 20
    PetscCheck(PetscAbsReal(trace - 20.0) < 1e-10, comm, PETSC_ERR_PLIB, "Hpre trace: expected 20.0, got %g", (double)trace);
  }

  // Direct TaoTermComputeHessianMult should also work
  PetscCall(TaoTermComputeHessianMult(sum, x, NULL, v, Hv));
  {
    PetscReal norm;

    PetscCall(VecNorm(Hv, NORM_2, &norm));
    PetscCheck(PetscAbsReal(norm - 10.0) < 1e-10, comm, PETSC_ERR_PLIB, "Direct HessianMult norm: expected 10.0, got %g", (double)norm);
  }

  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Hpre));
  PetscCall(VecDestroy(&Hv));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&x));
  PetscCall(TaoTermDestroy(&sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(TestNesting(comm));
  PetscCall(TestLazyHessian(comm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: nest
    output_file: output/empty.out

TEST*/
