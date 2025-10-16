/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>
#include "rosenbrock1.h" // defines AppCtx, AppCtxFormFunctionGradient(), and AppCtxFormHessian()

static char help[] = "This example demonstrates use of the TaoTerm\n\
interface for defining problems in the Tao library.  This example\n\
should be compared to rosenbrock1.c, which uses the callback interface\n\
to define the Rosenbrock function.\n";

static PetscErrorCode FormFunctionGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode CreateSolutionVec(TaoTerm, Vec *);

int main(int argc, char **argv)
{
  TaoTerm     objective;
  Tao         tao;  /* Tao solver context */
  PetscMPIInt size; /* number of processes running */
  AppCtx      user; /* user-defined application context */
  MPI_Comm    comm;

  /* Initialize TAO and PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  PetscOptionsBegin(comm, "", help, "none");
  PetscOptionsEnd();
  /* Initialize problem parameters */
  PetscCall(AppCtxInitialize(comm, &user));

  /* Define the objective function */
  PetscCall(TaoTermCreateShell(comm, (void *)&user, NULL /* no destructor is needed for `user` */, &objective));
  PetscCall(TaoTermSetParametersMode(objective, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermShellSetCreateSolutionVec(objective, CreateSolutionVec));
  PetscCall(TaoTermShellSetCreateHessianMatrices(objective, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(objective, PETSC_TRUE /* H == Hpre */, MATBAIJ, NULL));
  PetscCall(TaoTermShellSetObjectiveAndGradient(objective, FormFunctionGradient));
  PetscCall(TaoTermShellSetHessian(objective, FormHessian));

  /* Create TAO solver with desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));

  /* Set routines for function, gradient, hessian evaluation */
  PetscCall(TaoAddTerm(tao, NULL, 1.0, objective, NULL, NULL));

  /* Check for TAO command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  /* Clean up */
  PetscCall(AppCtxFinalize(&user, tao));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&objective));

  PetscCall(PetscFinalize());
  return 0;
}

/*
  FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

  Input Parameters:
+ term              - the `TaoTerm` for the objective function
. parameters_unused - optional vector of parameters that this rosenbrock function does not use
- X                 - input vector

  Output Parameters:
+ f - function value
- G - vector containing the newly evaluated gradient

  Note:
  Some optimization methods ask for the function and the gradient evaluation
  at the same time.  Evaluating both at once may be more efficient that
  evaluating each separately.
*/
static PetscErrorCode FormFunctionGradient(TaoTerm term, Vec X, Vec parameters_unused, PetscReal *f, Vec G)
{
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCheck(parameters_unused == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "Rosenbrock function does not take a parameter vector");
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(AppCtxFormFunctionGradient(user, X, f, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormHessian - Evaluates Hessian matrix.

  Input Parameters:
+ tao    - the Tao context
. x      - input vector
. params - optional vector of parameters that this rosenbrock function does not use
- ptr    - optional user-defined context, as set by TaoSetHessian()

  Output Parameters:
+ H    - Hessian matrix
- Hpre - Preconditiong matrix

  Note:  Providing the Hessian may not be necessary.  Only some solvers
  require this matrix.
*/
static PetscErrorCode FormHessian(TaoTerm term, Vec X, Vec params, Mat H, Mat Hpre)
{
  AppCtx *user;

  PetscFunctionBegin;
  PetscCheck(params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "Rosenbrock function does not take a parameter vector");
  PetscCall(TaoTermShellGetContext(term, &user));
  if (H) PetscCall(AppCtxFormHessianSingle(user, X, H));
  if (Hpre && Hpre != H) {
    if (H) PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
    else PetscCall(AppCtxFormHessianSingle(user, X, Hpre));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSolutionVec(TaoTerm term, Vec *solution)
{
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(AppCtxCreateSolution(user, solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex

   test:
     requires: !single
     output_file: output/rosenbrock1_1.out
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4

   test:
     suffix: test_gradient
     requires: !single
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4 -tao_test_gradient -tao_fd_delta 1.e-6 -n 4 -chained -tao_term_hessian_mat_type aij -alpha 49.0

   test:
     suffix: separate_hessians
     requires: !single
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4 -tao_term_hessian_pre_is_hessian 0 -tao_term_hessian_mat_type aij -tao_term_hessian_pre_mat_type sbaij -tao_view

TEST*/
