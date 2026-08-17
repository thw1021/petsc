const char help[] = "TAOTERMBOX projection and indicator tests";

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
  const PetscScalar input[6]       = {-3.0, -1.0, 0.25, 2.0, 5.0, -0.5};
  const PetscScalar scalar[6]      = {-1.0, -1.0, 0.25, 1.0, 1.0, -0.5};
  const PetscScalar lbvalues[6]    = {-2.0, 0.0, -1.0, 3.0, 4.0, -0.25};
  const PetscScalar ubvalues[6]    = {-1.0, 1.0, 0.5, 4.0, 6.0, 0.0};
  const PetscScalar vector[6]      = {-2.0, 0.0, 0.25, 3.0, 5.0, -0.25};
  const PetscScalar mixedlower[6]  = {-2.0, 0.0, 0.25, 3.0, 4.0, -0.25};
  const PetscScalar mixedupper[6]  = {-0.5, -0.5, 0.25, 1.0, 4.0, -0.5};
  const PetscScalar mixedubvals[6] = {-0.25, 0.0, 0.5, 1.0, 4.0, 0.25};
  TaoTerm           box;
  Vec               q, x, work, lb, ub;
  PetscReal         value;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6, &q));
  PetscCall(VecDuplicate(q, &x));
  PetscCall(VecDuplicate(q, &work));
  PetscCall(VecDuplicate(q, &lb));
  PetscCall(VecDuplicate(q, &ub));
  PetscCall(SetValues(q, input));
  PetscCall(SetValues(lb, lbvalues));
  PetscCall(SetValues(ub, ubvalues));

  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &box));
  PetscCall(TaoTermSetType(box, TAOTERMBOX));
  PetscCall(TaoTermSetSolutionTemplate(box, q));
  PetscCall(TaoTermBoxSetBounds(box, -1.0, 1.0, NULL, NULL));

  PetscCall(TaoTermProximalMap(box, NULL, 2.0, NULL, q, 3.0, x));
  PetscCall(CheckValues(x, work, scalar, "out-of-place scalar projection"));
  PetscCall(VecCopy(q, x));
  PetscCall(TaoTermProximalMap(box, NULL, 2.0, NULL, x, 3.0, x));
  PetscCall(CheckValues(x, work, scalar, "in-place scalar projection"));

  PetscCall(TaoTermBoxSetBounds(box, -10.0, 10.0, lb, ub));
  PetscCall(TaoTermProximalMap(box, NULL, 2.0, NULL, q, 3.0, x));
  PetscCall(CheckValues(x, work, vector, "vector projection"));
  PetscCall(TaoTermComputeObjective(box, x, NULL, &value));
  PetscCheck(value == 0.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Point feasible for vector bounds has objective %g", (double)value);
  PetscCall(VecSetValue(x, 3, 2.0, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(TaoTermComputeObjective(box, x, NULL, &value));
  PetscCheck(value == PETSC_INFINITY, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Vector lower-bound violation has objective %g", (double)value);
  PetscCall(VecSetValue(x, 3, 5.0, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(TaoTermComputeObjective(box, x, NULL, &value));
  PetscCheck(value == PETSC_INFINITY, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Vector upper-bound violation has objective %g", (double)value);

  PetscCall(TaoTermBoxSetBounds(box, -10.0, 4.0, lb, NULL));
  PetscCall(TaoTermProximalMap(box, NULL, 2.0, NULL, q, 3.0, x));
  PetscCall(CheckValues(x, work, mixedlower, "vector lower scalar upper projection"));

  PetscCall(SetValues(ub, mixedubvals));
  PetscCall(TaoTermBoxSetBounds(box, -0.5, 10.0, NULL, ub));
  PetscCall(TaoTermProximalMap(box, NULL, 2.0, NULL, q, 3.0, x));
  PetscCall(CheckValues(x, work, mixedupper, "scalar lower vector upper projection"));

  PetscCall(TaoTermBoxSetBounds(box, -1.0, 1.0, NULL, NULL));
  PetscCall(TaoTermProximalMap(box, NULL, 2.0, NULL, q, 3.0, x));
  PetscCall(CheckValues(x, work, scalar, "cleared vector bounds"));

  PetscCall(TaoTermComputeObjective(box, x, NULL, &value));
  PetscCheck(value == 0.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Feasible point has objective %g", (double)value);
  PetscCall(VecSetValue(x, 0, -2.0, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(TaoTermComputeObjective(box, x, NULL, &value));
  PetscCheck(value == PETSC_INFINITY, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Lower-bound violation has objective %g", (double)value);
  PetscCall(VecSet(x, 2.0));
  PetscCall(TaoTermComputeObjective(box, x, NULL, &value));
  PetscCheck(value == PETSC_INFINITY, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Upper-bound violation has objective %g", (double)value);

  PetscCall(TaoTermProximalMap(box, NULL, 0.0, NULL, q, 3.0, x));
  PetscCall(CheckValues(x, work, input, "zero-scale map"));

  PetscCall(TaoTermDestroy(&box));
  PetscCall(VecDestroy(&ub));
  PetscCall(VecDestroy(&lb));
  PetscCall(VecDestroy(&work));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&q));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}

TEST*/
