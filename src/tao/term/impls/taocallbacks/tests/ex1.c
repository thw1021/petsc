const char help[] = "TAOTERMTAOCALLBACKS coverage tests";

#include <petsctao.h>

static PetscErrorCode objective(Tao tao, Vec x, PetscReal *value, void *ctx)
{
  PetscFunctionBegin;
  *value = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gradient(Tao tao, Vec x, Vec g, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode objective_and_gradient(Tao tao, Vec x, PetscReal *value, Vec g, void *ctx)
{
  PetscFunctionBegin;
  *value = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode hessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscFunctionBegin;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testCallbacks(PetscBool separate)
{
  MPI_Comm    comm = PETSC_COMM_WORLD;
  Tao         tao;
  TaoTerm     term;
  TaoTermType type;
  PetscBool   same;
  PetscErrorCode (*_hessian)(Tao, Vec, Mat, Mat, void *);

  PetscFunctionBegin;
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoGetObjectiveTerm(tao, NULL, &term, NULL, NULL));
  PetscCall(TaoTermGetType(term, &type));
  PetscCall(PetscStrcmp(type, TAOTERMTAOCALLBACKS, &same));
  PetscCheck(same, comm, PETSC_ERR_PLIB, "wrong TaoTermType");
  if (separate) {
    PetscCall(TaoSetObjective(tao, objective, NULL));
    PetscCall(TaoSetGradient(tao, NULL, gradient, NULL));
  } else {
    PetscCall(TaoSetObjectiveAndGradient(tao, NULL, objective_and_gradient, NULL));
  }
  PetscCall(TaoSetHessian(tao, NULL, NULL, hessian, NULL));

  if (separate) {
    PetscErrorCode (*_objective)(Tao, Vec, PetscReal *, void *);
    PetscErrorCode (*_gradient)(Tao, Vec, Vec, void *);

    PetscCall(TaoGetObjective(tao, &_objective, NULL));
    PetscCall(TaoGetGradient(tao, NULL, &_gradient, NULL));
    PetscCheck(_objective == objective, comm, PETSC_ERR_PLIB, "wrong objective callback");
    PetscCheck(_gradient == gradient, comm, PETSC_ERR_PLIB, "wrong gradient callback");
  } else {
    PetscErrorCode (*_objective_and_gradient)(Tao, Vec, PetscReal *, Vec, void *);

    PetscCall(TaoGetObjectiveAndGradient(tao, NULL, &_objective_and_gradient, NULL));
    PetscCheck(_objective_and_gradient == objective_and_gradient, comm, PETSC_ERR_PLIB, "wrong objective and gradient callback");
  }
  PetscCall(TaoGetHessian(tao, NULL, NULL, &_hessian, NULL));
  PetscCheck(_hessian == hessian, comm, PETSC_ERR_PLIB, "wrong hessian callback");

  PetscCall(TaoDestroy(&tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(testCallbacks(PETSC_FALSE));
  PetscCall(testCallbacks(PETSC_TRUE));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
