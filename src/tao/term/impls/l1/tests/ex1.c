const char help[] = "TAOTERML1 proximal-map tests";

#include <petsctaoterm.h>

static PetscErrorCode SetValues(Vec x, const PetscScalar values[])
{
  PetscInt low, high, i;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(x, &low, &high));
  for (i = low; i < high; ++i) PetscCall(VecSetValue(x, i, values[i], INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckValues(Vec x, Vec work, const PetscScalar values[], const char name[])
{
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(SetValues(work, values));
  PetscCall(VecAXPY(work, -1.0, x));
  PetscCall(VecNorm(work, NORM_INFINITY, &error));
  PetscCheck(error <= 10.0 * PETSC_MACHINE_EPSILON, PetscObjectComm((PetscObject)x), PETSC_ERR_PLIB, "%s error %g", name, (double)error);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  const PetscScalar qvalues[6]       = {-3.0, -1.0, 0.25, 2.0, 5.0, -0.5};
  const PetscScalar pvalues[6]       = {1.0, -2.0, 0.5, 2.0, -1.0, 0.25};
  const PetscScalar ordinary[6]      = {-2.5, -0.5, 0.0, 1.5, 4.5, 0.0};
  const PetscScalar translated[6]    = {-2.5, -1.5, 0.5, 2.0, 4.5, 0.0};
  const PetscScalar zerocenter[6]    = {0.5, -0.5, 0.5, 0.5, -0.5, 0.25};
  const PetscScalar zeroobjective[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  TaoTerm           l1, l2;
  Vec               p, q, x, work;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6, &q));
  PetscCall(VecDuplicate(q, &p));
  PetscCall(VecDuplicate(q, &x));
  PetscCall(VecDuplicate(q, &work));
  PetscCall(SetValues(q, qvalues));
  PetscCall(SetValues(p, pvalues));
  PetscCall(TaoTermCreateL1(PETSC_COMM_WORLD, PETSC_DECIDE, 6, 0.0, &l1));

  PetscCall(TaoTermProximalMap(l1, NULL, 2.0, NULL, q, 4.0, x));
  PetscCall(CheckValues(x, work, ordinary, "ordinary proximal map"));
  PetscCall(VecCopy(q, x));
  PetscCall(TaoTermProximalMap(l1, NULL, 2.0, NULL, x, 4.0, x));
  PetscCall(CheckValues(x, work, ordinary, "in-place ordinary proximal map"));

  PetscCall(TaoTermProximalMap(l1, p, 2.0, NULL, q, 4.0, x));
  PetscCall(CheckValues(x, work, translated, "translated proximal map"));
  PetscCall(SetValues(p, pvalues));
  PetscCall(TaoTermProximalMap(l1, p, 2.0, NULL, q, 4.0, p));
  PetscCall(CheckValues(p, work, translated, "parameter-aliasing translated proximal map"));

  PetscCall(SetValues(p, pvalues));
  PetscCall(TaoTermProximalMap(l1, p, 2.0, NULL, NULL, 4.0, x));
  PetscCall(CheckValues(x, work, zerocenter, "zero-center translated proximal map"));
  PetscCall(TaoTermProximalMap(l1, NULL, 2.0, NULL, NULL, 4.0, x));
  PetscCall(CheckValues(x, work, zeroobjective, "zero-center ordinary proximal map"));

  PetscCall(TaoTermProximalMap(l1, p, 0.0, NULL, q, 4.0, x));
  PetscCall(CheckValues(x, work, qvalues, "zero objective scale"));
  PetscCall(TaoTermProximalMap(l1, p, 0.0, NULL, NULL, 4.0, x));
  PetscCall(CheckValues(x, work, zeroobjective, "zero objective scale and center"));

  PetscCall(TaoTermCreateHalfL2Squared(PETSC_COMM_WORLD, PETSC_DECIDE, 6, &l2));
  PetscCall(TaoTermProximalMap(l1, p, 2.0, l2, q, 4.0, x));
  PetscCall(CheckValues(x, work, translated, "explicit half-L2 proximal map"));

  PetscCall(TaoTermDestroy(&l2));
  PetscCall(TaoTermDestroy(&l1));
  PetscCall(VecDestroy(&work));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&q));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}

TEST*/
