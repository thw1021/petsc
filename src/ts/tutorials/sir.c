static char help[] = "A simple SIR model.\n";

/*
  The governing equations
    S_t = \mu - \beta * S * I - \mu * S
    I_t = \beta * S * I - \gamma * I - \mu * I
    R_t = \gamma * I - \mu * R
*/
#include <petscts.h>
#include <petscviewer.h>

typedef struct
{
  PetscViewer viewer;
  PetscReal   mu;
  PetscReal   beta;
  PetscReal   gamma;
} Parameters;

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *s)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u;
  const Parameters  *p; 

  PetscFunctionBegin;
  p = (Parameters*)s;
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);

  f[0] = p->mu - p->beta * u[0] * u[1] - p->mu * u[0];
  f[1] = p->beta * u[0] * u[1] - p->gamma * u[1] - p->mu * u[1];
  f[2] = p->gamma * u[1] - p->mu * u[2];

  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode monitor(TS ts, PetscInt steps, PetscReal time, Vec U, void *mctx)
{
  PetscErrorCode ierr;
  Parameters     *p = (Parameters*)mctx;

  PetscFunctionBegin;
  ierr = VecView(U, p->viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS                ts;
  Vec               U;
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscInt          n = 3;
  PetscScalar       *u;
  PetscReal         t, final_time = 14.0, dt = 0.05;
  TSAdapt           adapt;
  Parameters        p;
  PetscReal         bounds[] = {1.0, 3.3};


  ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;
  ierr = PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD), 1, bounds);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Only for sequential runs");

  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &p.viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(p.viewer, PETSCVIEWERASCII);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "sir.csv", &(p.viewer));CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(p.viewer, PETSC_VIEWER_ASCII_CSV);
  p.mu = 0.6;
  p.beta = 3;
  p.gamma = 0.4;

  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSROSW);CHKERRQ(ierr);

  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, &p);CHKERRQ(ierr);

  ierr = TSMonitorSet(ts, monitor, &p, NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &U);CHKERRQ(ierr);
  ierr = VecSetSizes(U, n, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(U);CHKERRQ(ierr);
  ierr = VecGetArray(U, &u);CHKERRQ(ierr);
  u[0] = 0.99;
  u[1] = 0.01;
  u[2] = 0.0;
  ierr = VecRestoreArray(U, &u);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, U);CHKERRQ(ierr);

  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, final_time);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, dt);CHKERRQ(ierr);
  ierr = TSGetAdapt(ts, &adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt, TSADAPTNONE);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts, U);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);

  if (PetscAbsReal(t - final_time) > 100 * PETSC_MACHINE_EPSILON)
  {
    ierr = PetscPrintf(PETSC_COMM_WORLD, 
    "Note: There is a difference of %g between the prescribed final time %g and the actual final time.\n", (double)(final_time - t), (double)final_time);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&p.viewer);CHKERRQ(ierr);

  ierr = PetscFinalize();

  return ierr;
}