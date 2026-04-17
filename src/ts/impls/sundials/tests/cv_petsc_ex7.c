/*-----------------------------------------------------------------
 * Programmer(s): Cody J. Balos @ LLNL
 *-----------------------------------------------------------------
 * Acknowledgement: This example is based on the PETSc TS ex7.c
 *-----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2025-2026, Lawrence Livermore National Security,
 * University of Maryland Baltimore County, and the SUNDIALS contributors.
 * Copyright (c) 2013-2025, Lawrence Livermore National Security
 * and Southern Methodist University.
 * Copyright (c) 2002-2013, Lawrence Livermore National Security.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 *---------------------------------------------------------------*/

static char help[] = "CVODE example based on PETSc TS ex7.c: Nonlinear, "
                     "time-dependent PDE in 2d.\n";

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
*/
#include <petscdmda.h>
#include <petscts.h>

/*
   Include "cvode.h" for access to the CVODE BDF integrator. Include
   "sunnonlinsol_petscsnes.h" for access to the SUNNonlinearSolver
   wrapper for PETSc SNES.
*/
#include <cvode/cvode.h>
#include <nvector/nvector_petsc.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>

/*
   User-defined routines for PETSc TS and SNES
*/
extern PetscErrorCode TSRHS(TS, PetscReal, Vec, Vec, DM);
extern PetscErrorCode FormInitialSolution(DM, Vec);
extern PetscErrorCode MySNESMonitor(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);

/*
   User-defined routines for CVODE
*/
extern SUNErrCode CVODERHS(PetscReal, N_Vector, N_Vector, void *);
extern SUNErrCode MyCVodeMonitor(long int, PetscReal, Vec, void *);

int main(int argc, char **argv)
{
  SUNContext         sunctx;
  void              *cvode_mem; /* integrator memory */
  N_Vector           nvecx;
  SUNNonlinearSolver NLS;
  long int           nsteps = 0;

  SNES                  snes;
  Vec                   x, r; /* solution, residual vectors */
  Mat                   Jmf;
  DM                    da;
  PetscViewerAndFormat *vf;
  PetscReal             T0, t, tf;

  /* set start and stop time */
  T0 = 0.;
  t  = 0.;
  tf = 0.0005;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(SUNContext_Create(PETSC_COMM_WORLD, &sunctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecDuplicate(x, &r));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create N_Vector wrapper of PETSc vector
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCallSUNDIALSMem(nvecx, N_VMake_Petsc, x, sunctx);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create CVODE integrator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCallSUNDIALSMem(cvode_mem, CVodeCreate, CV_BDF, sunctx);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions and integrator options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(da, x));
  PetscCallSUNDIALS(CVodeInit, cvode_mem, CVODERHS, T0, nvecx);

  /* provide the DM context as user data so we can access it in the RHS */
  PetscCallSUNDIALS(CVodeSetUserData, cvode_mem, (void *)da);

  /* use the PETSc TS default tolerances */
  PetscCallSUNDIALS(CVodeSStolerances, cvode_mem, 1e-4, 1e-4);

  /* set the max order to 1 for Backward Euler */
  PetscCallSUNDIALS(CVodeSetMaxOrd, cvode_mem, 1);

  /* create SNES nonlinear to be used by time integrator */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  /* create SUNNonlinearSolver object which uses SNES */
  PetscCallSUNDIALSMem(NLS, SUNNonlinSol_PetscSNES, nvecx, snes, sunctx);
  PetscCall(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf));
  PetscCall(SNESMonitorSet(snes, (PetscErrorCode (*)(SNES, PetscInt, PetscReal, PetscCtx))MySNESMonitor, vf, (PetscCtxDestroyFn *)PetscViewerAndFormatDestroy));
  PetscCall(MatCreateSNESMF(snes, &Jmf));
  PetscCall(SNESSetJacobian(snes, Jmf, Jmf, MatMFFDComputeJacobian, 0));
  PetscCall(SNESSetFromOptions(snes));

  PetscCallSUNDIALS(CVodeSetNonlinearSolver, cvode_mem, NLS);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Integrate the ODE
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  while (t < tf) {
    /* CV_ONE_STEP mode causes CVODE to return after every time step.
        We use it here to demonstrate how to print monitoring information
        at every time step. */
    PetscCallSUNDIALS(MyCVodeMonitor, nsteps, t, x, NULL);
    PetscCallSUNDIALS(CVode, cvode_mem, tf, nvecx, &t, CV_ONE_STEP);
    PetscCallSUNDIALS(CVodeGetNumSteps, cvode_mem, &nsteps);
  }
  PetscCallSUNDIALS(MyCVodeMonitor, nsteps, t, x, NULL);

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&Jmf));
  PetscCall(DMDestroy(&da));
  PetscCall(SNESDestroy(&snes));
  PetscCallSUNDIALSVoid(CVodeFree, &cvode_mem);
  PetscCallSUNDIALSVoid(N_VDestroy, nvecx);
  PetscCallSUNDIALS(SUNNonlinSolFree, NLS);
  PetscCallSUNDIALS(SUNContext_Free, &sunctx);
  PetscCall(PetscFinalize());
  return PETSC_SUCCESS;
}

/*
   Evaluates the right hand side of U_t = G(t,u) for CVODE
*/
SUNErrCode CVODERHS(PetscReal t, N_Vector x, N_Vector xdot, void *ptr)
{
  PetscCall(TSRHS(NULL, t, N_VGetVector_Petsc(x), N_VGetVector_Petsc(xdot), (DM)ptr));
  return 0;
}

/*
   Evaluates the right hand side of U_t = G(t,u) for TS
 */
PetscErrorCode TSRHS(TS ts, PetscReal ftime, Vec X, Vec F, DM da)
{
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   two = 2.0, hx, hy, sx, sy;
  PetscScalar u, uxx, uyy, **x, **f;
  Vec         localX;

  PetscFunctionBeginUser;
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
      u   = x[j][i];
      uxx = (two * u - x[j][i - 1] - x[j][i + 1]) * sx;
      uyy = (two * u - x[j - 1][i] - x[j + 1][i]) * sy;
      /*      f[j][i] = -(uxx + uyy); */
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
     Compute function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    y = j * hy;
    for (i = xs; i < xs + xm; i++) {
      x = i * hx;
      r = PetscSqrtReal((x - .5) * (x - .5) + (y - .5) * (y - .5));
      if (r < .125) u[j][i] = PetscExpReal(-30.0 * r * r * r);
      else u[j][i] = 0.0;
    }
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

SUNErrCode MyCVodeMonitor(long int step, PetscReal ptime, Vec v, void *ctx)
{
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(VecNorm(v, NORM_2, &norm));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)v), "timestep %ld time %g norm %g\n", step, (double)ptime, (double)norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   MySNESMonitor - illustrate how to set user-defined monitoring routine for SNES.
   Input Parameters:
     snes - the SNES context
     its - iteration number
     fnorm - 2-norm function value (may be estimated)
     ctx - optional user-defined context for private data for the
         monitor routine, as set by SNESMonitorSet()
 */
PetscErrorCode MySNESMonitor(SNES snes, PetscInt its, PetscReal fnorm, PetscViewerAndFormat *vf)
{
  PetscFunctionBeginUser;
  PetscCall(SNESMonitorDefaultShort(snes, its, fnorm, vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    build:
      requires: sundials

    test:
      args: -snes_atol 1.e-11

TEST*/
