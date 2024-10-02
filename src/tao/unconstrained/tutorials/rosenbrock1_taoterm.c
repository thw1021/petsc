/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>
#include "rosenbrock1.h" // defines AppCtx, AppCtxFormFunctionGradient(), and AppCtxFormHessian()

static char help[] = "This example demonstrates use of the TaoTerm\n\
interface for defining problems in the Tao library.  This example\n\
should be compared to rosenbrock1.c, which uses the callback interface\n\
to define the Rosenbrock function.\n";

static PetscErrorCode FormFunctionGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode CreateVecs(TaoTerm, Vec *, Vec *);

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

  /* Initialize problem parameters */
  PetscCall(AppCtxInitialize(comm, &user));

  /* Define the objective function */
  PetscCall(TaoTermCreateShell(comm, (void *)&user, NULL /* no destructor is needed for `user` */, &objective));
  PetscCall(TaoTermSetParametersMode(objective, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermShellSetCreateVecs(objective, CreateVecs));
  PetscCall(TaoTermShellSetCreateHessianMatrices(objective, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(objective, PETSC_TRUE /* H == Hpre */, MATBAIJ, NULL));
  PetscCall(TaoTermShellSetObjectiveAndGradient(objective, FormFunctionGradient));
  PetscCall(TaoTermShellSetHessian(objective, FormHessian));

  /* Create TAO solver with desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));

  /* Set routines for function, gradient, hessian evaluation */
  PetscCall(TaoSetObjectiveTerm(tao, 1.0, objective, NULL, NULL));

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
  FormHessianSingle - Evaluates Hessian matrix.

  Input Parameters:
+ tao               - the Tao context
. x                 - input vector
. parameters_unused - optional vector of parameters that this rosenbrock function does not use
- ptr               - optional user-defined context, as set by TaoSetHessian()

  Output Parameters:
+ H    - Hessian matrix
- Hpre - Preconditiong matrix

  Note:  Providing the Hessian may not be necessary.  Only some solvers
  require this matrix.
*/
static PetscErrorCode FormHessianSingle(TaoTerm term, Vec X, Vec parameters_unused, Mat H)
{
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCheck(parameters_unused == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "Rosenbrock function does not take a parameter vector");
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(AppCtxFormHessianSingle(user, X, H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(TaoTerm term, Vec X, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  PetscCall(TaoTermHessianSingle(term, X, params, H, Hpre, FormHessianSingle, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateVecs(TaoTerm term, Vec *solution, Vec *parameters_unused)
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

TEST*/
