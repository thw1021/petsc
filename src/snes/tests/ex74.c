static char help[] = "Tests SNESVIGetInactiveSet() during and after the solve of a one-dimensional obstacle problem with SNESVINEWTONRSLS.\n\n";

/*
   Solves the bound-constrained linear complementarity problem

       F(u) = A u - f,   A = (1/h^2) tridiag(-1, 2, -1),   0 <= u <= ub

   whose unconstrained solution is the parabola f x (1 - x) / 2 with maximum f / 8 at x = 1/2, so that for ub < f / 8 the
   solution touches the upper bound in the middle of the domain. The inactive set returned by SNESVIGetInactiveSet() from a
   KSP monitor, from a SNES monitor and after SNESSolve() is compared with the complement of SNESVIGetActiveSetIS().
*/

#include <petscsnes.h>

typedef struct {
  SNES     snes;
  Mat      A;
  Vec      f;
  PetscInt kspcalls, snescalls;
} AppCtx;

static PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, PetscCtx ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(MatMult(user->A, x, F));
  PetscCall(VecAXPY(F, -1.0, user->f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat B, PetscCtx ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(MatCopy(user->A, B, SAME_NONZERO_PATTERN));
  if (J != B) PetscCall(MatCopy(user->A, J, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Checks that the inactive set returned by SNESVIGetInactiveSet() is the complement of the active set of the current iterate of snes */
static PetscErrorCode CheckInactiveSet(SNES snes)
{
  Vec       x, F;
  IS        inact, inact2, act, ref;
  PetscInt  rstart, rend;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(SNESVIGetInactiveSet(snes, &inact));
  PetscCheck(inact, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "SNESVIGetInactiveSet() returned NULL");
  PetscCall(SNESVIGetInactiveSet(snes, &inact2));
  PetscCheck(inact == inact2, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "SNESVIGetInactiveSet() did not return the cached inactive set");
  PetscCall(SNESGetSolution(snes, &x));
  PetscCall(SNESGetFunction(snes, &F, NULL, NULL));
  PetscCall(SNESVIGetActiveSetIS(snes, x, F, &act));
  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  PetscCall(ISComplement(act, rstart, rend, &ref));
  PetscCall(ISEqual(inact, ref, &equal));
  PetscCheck(equal, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "SNESVIGetInactiveSet() does not match the complement of SNESVIGetActiveSetIS()");
  PetscCall(ISDestroy(&act));
  PetscCall(ISDestroy(&ref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPMonitorInactiveSet(KSP ksp, PetscInt it, PetscReal rnorm, PetscCtx ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(CheckInactiveSet(user->snes));
  user->kspcalls++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESMonitorInactiveSet(SNES snes, PetscInt it, PetscReal fnorm, PetscCtx ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(CheckInactiveSet(snes));
  user->snescalls++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ViewInactiveSet(SNES snes)
{
  IS          inact, all;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(CheckInactiveSet(snes));
  PetscCall(SNESVIGetInactiveSet(snes, &inact));
  PetscCall(ISAllGather(inact, &all));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)snes), &rank));
  if (rank == 0) PetscCall(ISView(all, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(ISDestroy(&all));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES        snes;
  KSP         ksp;
  Vec         x, r, xl, xu;
  Mat         J;
  AppCtx      user;
  PetscInt    n = 15, rstart, rend;
  PetscReal   h, ub = 0.08;
  PetscScalar f = 1.0, v[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ub", &ub, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-f", &f, NULL));
  h              = 1.0 / (n + 1);
  user.kspcalls  = 0;
  user.snescalls = 0;

  PetscCall(MatCreate(PETSC_COMM_WORLD, &user.A));
  PetscCall(MatSetSizes(user.A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(user.A));
  PetscCall(MatSeqAIJSetPreallocation(user.A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(user.A, 3, NULL, 1, NULL));
  PetscCall(MatGetOwnershipRange(user.A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscInt cols[3] = {i - 1, i, i + 1}, ncols = 0, c[3];

    v[0] = -1.0 / (h * h);
    v[1] = 2.0 / (h * h);
    v[2] = -1.0 / (h * h);
    for (PetscInt j = 0; j < 3; j++) {
      if (cols[j] >= 0 && cols[j] < n) {
        c[ncols]   = cols[j];
        v[ncols++] = v[j];
      }
    }
    PetscCall(MatSetValues(user.A, 1, &i, ncols, c, v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(user.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatDuplicate(user.A, MAT_DO_NOT_COPY_VALUES, &J));

  PetscCall(MatCreateVecs(user.A, &x, &user.f));
  PetscCall(VecSet(user.f, f));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecSet(xl, 0.0));
  PetscCall(VecSet(xu, ub));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  user.snes = snes;
  PetscCall(SNESSetType(snes, SNESVINEWTONRSLS));
  PetscCall(SNESSetFunction(snes, r, FormFunction, &user));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, &user));
  PetscCall(SNESVISetVariableBounds(snes, xl, xu));
  PetscCall(SNESMonitorSet(snes, SNESMonitorInactiveSet, &user, NULL));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPMonitorSet(ksp, KSPMonitorInactiveSet, &user, NULL));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(VecSet(x, 0.0));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCheck(user.kspcalls > 0 && user.snescalls > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Monitors were not called");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Inactive set after the first solve\n"));
  PetscCall(ViewInactiveSet(snes));

  /* the cached inactive set must be recomputed for the solution of a second solve with a different upper bound */
  PetscCall(VecSet(xu, 2.0 * ub));
  PetscCall(VecSet(x, 0.0));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Inactive set after the second solve\n"));
  PetscCall(ViewInactiveSet(snes));

  /* SNESReset() destroys the cached inactive set, the third solve with the original upper bound must reproduce the first inactive set */
  PetscCall(SNESReset(snes));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, &user));
  PetscCall(SNESVISetVariableBounds(snes, xl, xu));
  PetscCall(VecSet(xu, ub));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Inactive set after the third solve\n"));
  PetscCall(ViewInactiveSet(snes));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&user.f));
  PetscCall(MatDestroy(&user.A));
  PetscCall(MatDestroy(&J));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      args: -ksp_rtol 1e-12 -snes_rtol 1e-10
      output_file: output/ex74_1.out
      test:
         suffix: 1
      test:
         suffix: 2
         nsize: 2

TEST*/
