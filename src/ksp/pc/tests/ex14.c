static const char help[] = "Test CUDA variable-block Jacobi on nonblocking device contexts.\n";

#include <petscksp.h>
#include <petscdevice.h>

static PetscErrorCode TestApply(PC pc, PC reference, Vec x, Vec y, Vec hx, Vec hy, PetscDeviceContext dctx, PetscBool barrier)
{
  const PetscScalar *actual, *expected;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  for (PetscInt transpose = 0; transpose < 2; ++transpose) {
    PetscCall(VecSet(x, 1));
    PetscCall(VecSet(y, 0));
    if (transpose) PetscCall(PCApplyTranspose(pc, x, y));
    else PetscCall(PCApply(pc, x, y));
    PetscCall(PetscDeviceContextSynchronize(dctx));

    PetscCall(VecSet(hx, 2));
    if (transpose) PetscCall(PCApplyTranspose(reference, hx, hy));
    else PetscCall(PCApply(reference, hx, hy));
    PetscCall(VecScale(hy, 3));

    PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    PetscCall(VecScale(x, 2));
    // A separate delay checks completion even when VecScale() honors the barrier.
    if (barrier) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    if (transpose) PetscCall(PCApplyTranspose(pc, x, y));
    else PetscCall(PCApply(pc, x, y));
    if (barrier) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCApply() returned with work pending on a barrier context");
    }
    PetscCall(VecScale(y, 3));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(VecGetArrayRead(y, &actual));
    PetscCall(VecGetArrayRead(hy, &expected));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(PetscAbsScalar(actual[i] - expected[i]) < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect variable-block Jacobi result at local entry %" PetscInt_FMT, i);
    PetscCall(VecRestoreArrayRead(hy, &expected));
    PetscCall(VecRestoreArrayRead(y, &actual));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat                   A, host;
  PC                    pc, reference;
  Vec                   x, y, hx, hy;
  PetscDeviceContext    saved, dctx;
  const PetscStreamType streams[]     = {PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};
  const PetscInt        blocks[]      = {1, 2, 2};
  const PetscScalar     entries[5][5] = {
    {2, 0, 0, 0, 0 },
    {0, 3, 1, 0, 0 },
    {0, 0, 4, 0, 0 },
    {0, 0, 0, 5, -1},
    {0, 0, 0, 2, 6 }
  };
  PetscInt start;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 2, NULL, 0, NULL, &host));
  PetscCall(MatGetOwnershipRange(host, &start, NULL));
  for (PetscInt i = 0; i < 5; ++i)
    for (PetscInt j = 0; j < 5; ++j)
      if (entries[i][j] != 0) PetscCall(MatSetValue(host, start + i, start + j, entries[i][j], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(host, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(host, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetVariableBlockSizes(host, 3, blocks));
  PetscCall(MatConvert(host, MATAIJCUSPARSE, MAT_INITIAL_MATRIX, &A));
  PetscCall(MatSetVariableBlockSizes(A, 3, blocks));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(MatCreateVecs(host, &hx, &hy));
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCVPBJACOBI));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCCreate(PETSC_COMM_WORLD, &reference));
  PetscCall(PCSetType(reference, PCVPBJACOBI));
  PetscCall(PCSetOperators(reference, host, host));
  PetscCall(PetscDeviceContextSynchronize(saved));
  for (PetscInt k = 0; k < PETSC_STATIC_ARRAY_LENGTH(streams); ++k) {
    PetscCall(PetscDeviceContextDuplicate(saved, &dctx));
    PetscCall(PetscDeviceContextSetStreamType(dctx, streams[k]));
    PetscCall(PetscDeviceContextSetUp(dctx));
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
    for (PetscInt update = 0; update < 2; ++update) {
      PetscCall(MatScale(host, 2));
      PetscCall(MatScale(A, 2));
      PetscCall(PCSetUp(pc));
      PetscCall(PCSetUp(reference));
      PetscCall(TestApply(pc, reference, x, y, hx, hy, dctx, (PetscBool)(streams[k] == PETSC_STREAM_NONBLOCKING_WITH_BARRIER)));
    }
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&dctx));
  }
  PetscCall(PCDestroy(&reference));
  PetscCall(PCDestroy(&pc));
  PetscCall(VecDestroy(&hy));
  PetscCall(VecDestroy(&hx));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&host));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: cuda
    requires: cuda
    nsize: {{1 2}}
    output_file: output/empty.out

TEST*/
