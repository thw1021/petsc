static char help[] = "Nonlinear, time-dependent PDE in 2d.\n";

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscts.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormFunction(TS, PetscReal, Vec, Vec, void *), FormInitialSolution(DM, Vec);
extern PetscErrorCode MyTSMonitor(TS, PetscInt, PetscReal, Vec, void *);

int main(int argc, char **argv)
{
  TS        ts;   /* time integrator */
  Vec       x, r; /* solution, residual vectors */
  DM        da;
  PetscBool isCVode;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecDuplicate(x, &r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts, NULL, FormFunction, da));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine

     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner)
     -snes_mf_operator : form matrix used to construct the preconditioner as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSMonitorSet(ts, MyTSMonitor, PETSC_VIEWER_STDOUT_WORLD, NULL));
  PetscCall(TSSetDM(ts, da));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetType(ts, TSBEULER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(da, x));
  PetscCall(TSSetTimeStep(ts, .0001));
  PetscCall(TSSetSolution(ts, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Impose non-negativity bounds on the solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscObjectTypeCompareAny((PetscObject)ts, &isCVode, TSCVODEBDF, TSCVODEADAMS, ""));
  if (!isCVode) {
    /*
      Solves the ODE nonlinear system as a variational inequality to ensure it never produces
      negative entries. Since we know the negative entries creep in very slowly due to round-off
      we assume the variational inequality solution is the correct solution for the ODE integrator.

      One could also just set all negative entries in the solution to zero after SNESSolve()
      but TS/SNES has no API to perform that operation.
    */
    Vec  xlow, xhigh;
    SNES snes;

    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(VecDuplicate(x, &xlow));
    PetscCall(VecDuplicate(x, &xhigh));
    PetscCall(VecSet(xhigh, PETSC_INFINITY));
    PetscCall(SNESVISetVariableBounds(snes, xlow, xhigh));
    PetscCall(VecDestroy(&xlow));
    PetscCall(VecDestroy(&xhigh));
    PetscCall(SNESSetType(snes, SNESVINEWTONRSLS));
  } else {
    /*
     CVode tries to decrease the time-step to ensure the nonlinear system solution has
     no negative entries. This does not work very well resulting in extremely small time steps.
      */
    Vec ones;

    PetscCall(VecDuplicate(x, &ones));
    PetscCall(VecSet(ones, 1.0));
    PetscCall(TSCVodeSetConstraints(ts, ones));
    PetscCall(VecDestroy(&ones));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  ts - the TS context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(TS ts, PetscReal ftime, Vec X, Vec F, void *ptr)
{
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   two = 2.0, hx, hy, sx, sy;
  PetscScalar u, uxx, uyy, **x, **f;
  Vec         localX;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 1.0 / (PetscReal)(Mx - 1);
  sx = 1.0 / (hx * hx);
  hy = 1.0 / (PetscReal)(My - 1);
  sy = 1.0 / (hy * hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da, localX, &x));
  PetscCall(DMDAVecGetArray(da, F, &f));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      if (i == 0 || j == 0 || i == Mx - 1 || j == My - 1) {
        f[j][i] = x[j][i];
        continue;
      }
      u       = x[j][i];
      uxx     = (two * u - x[j][i - 1] - x[j][i + 1]) * sx;
      uyy     = (two * u - x[j - 1][i] - x[j + 1][i]) * sy;
      f[j][i] = -u * (uxx + uyy) - (4.0 - 1.0) * ((x[j][i + 1] - x[j][i - 1]) * (x[j][i + 1] - x[j][i - 1]) * .25 * sx + (x[j + 1][i] - x[j - 1][i]) * (x[j + 1][i] - x[j - 1][i]) * .25 * sy);
    }
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArrayRead(da, localX, &x));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscCall(PetscLogFlops(11.0 * ym * xm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     Initial solution is:
     * 1 at the center of the square
     * exp(-2063. * r * r * r) within the radius of .25 from the square's center
     * 1.e-14 at the radius of .25 from the square's center
     * 0 outside the radius of .25 from the square's center
*/
PetscErrorCode FormInitialSolution(DM da, Vec U)
{
  PetscInt      i, j, xs, ys, xm, ym, Mx, My;
  PetscScalar **u;
  PetscReal     hx, hy, x, y, r;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 1.0 / (PetscReal)(Mx - 1);
  hy = 1.0 / (PetscReal)(My - 1);

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArray(da, U, &u));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /*
     Compute initial solution function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    y = j * hy;
    for (i = xs; i < xs + xm; i++) {
      x = i * hx;
      r = PetscSqrtReal((x - .5) * (x - .5) + (y - .5) * (y - .5));
      if (r <= .25) u[j][i] = PetscExpReal(-2063. * r * r * r);
      else u[j][i] = 0.0;
    }
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MyTSMonitor(TS ts, PetscInt step, PetscReal ptime, Vec v, PetscCtx ctx)
{
  DM        da;
  PetscInt  Mx, My;
  PetscReal hx, hy;
  PetscReal norm, min;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS); /* step of -1 indicates an interpolated solution */
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  hx = 1.0 / (PetscReal)(Mx - 1);
  hy = 1.0 / (PetscReal)(My - 1);
  PetscCall(VecNorm(v, NORM_2, &norm));
  norm = norm * PetscSqrtReal(hx * hy);
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCall(PetscPrintf(comm, "timestep %" PetscInt_FMT " time %g norm %g\n", step, (double)ptime, (double)norm));
  PetscCall(VecMin(v, NULL, &min));
  PetscCheck(min >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Solution minimum is negative %g", min);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      args: -ts_max_time .0005 -ts_exact_final_time matchstep

    test:
      suffix: 2
      args: -ts_max_time .0005 -snes_mf_operator -ts_exact_final_time matchstep

    test:
      suffix: 3
      args: -ts_max_time .0005 -snes_mf -pc_type none -ts_exact_final_time matchstep

    test:
      requires: sundials
      suffix: cvodebdf
      args: -da_refine 4 -ts_max_time .05 -snes_mf -pc_type none -ts_exact_final_time matchstep -ts_type cvodebdf

    test:
      requires: sundials
      suffix: cvodeadams
      args: -da_refine 4 -ts_max_time .05 -snes_mf -pc_type none -ts_exact_final_time matchstep -ts_type cvodeadams

TEST*/
