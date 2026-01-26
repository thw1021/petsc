static char help[] = "2D Shallow water equations benchmark.\n"
                     "Implements 2D shallow water equations with 3 DOF per grid point (h, hu, hv).\n\n"
                     "Example usage:\n"
                     "  ./ex4 -steps 100 -nx 40 -ny 40\n"
                     "  ./ex4 -steps 500 -output_file output.txt\n\n";

#include <petscdmda.h>
#include <petscts.h>
#include <petscvec.h>

/* Default parameter values */
#define DEFAULT_NX            40
#define DEFAULT_NY            40
#define DEFAULT_STEPS         100
#define DEFAULT_G             9.81
#define DEFAULT_DT            0.02
#define DEFAULT_LX            80.0
#define DEFAULT_LY            80.0
#define DEFAULT_H0            1.5
#define DEFAULT_AX            0.2
#define DEFAULT_AY            0.2
#define DEFAULT_PROGRESS_FREQ 10

/* Flux scheme types */
typedef enum {
  EX4_FLUX_RUSANOV,
  EX4_FLUX_MC
} Ex4FluxType;

static const char *const Ex4FluxTypes[] = {"rusanov", "mc", "Ex4FluxType", "EX4_FLUX_", NULL};

typedef struct {
  DM          da;        /* 2D periodic DMDA for state */
  PetscInt    nx, ny;    /* Grid dimensions */
  PetscReal   Lx, Ly;    /* Domain size */
  PetscReal   dx, dy;    /* Grid spacing */
  PetscReal   g;         /* Gravity */
  PetscReal   dt;        /* Time step */
  TS          ts;        /* Time stepper */
  PetscReal   h0;        /* Mean height */
  PetscReal   Ax, Ay;    /* Wave amplitudes */
  Ex4FluxType flux_type; /* Flux scheme */
} ShallowWater2DCtx;

/*
  Limit - MC (Monotonized Central) limiter
*/
static PetscReal Limit(PetscReal a, PetscReal b)
{
  PetscReal c = 0.5 * (a + b);
  if (a * b <= 0.0) return 0.0;
  if (c > 0) return PetscMin(2.0 * a, PetscMin(2.0 * b, c));
  else return PetscMax(2.0 * a, PetscMax(2.0 * b, c));
}

/*
  ComputeFluxX - Compute physical flux in x-direction for shallow water
*/
static void ComputeFluxX(PetscReal g, PetscReal h, PetscReal hu, PetscReal hv, PetscReal *F_h, PetscReal *F_hu, PetscReal *F_hv, PetscReal *u, PetscReal *c)
{
  if (h > 1e-10) {
    *u    = hu / h;
    *c    = PetscSqrtReal(g * h);
    *F_h  = hu;
    *F_hu = hu * *u + 0.5 * g * h * h;
    *F_hv = hu * (hv / h);
  } else {
    *u    = 0.0;
    *c    = 0.0;
    *F_h  = 0.0;
    *F_hu = 0.0;
    *F_hv = 0.0;
  }
}

/*
  ComputeFluxY - Compute physical flux in y-direction for shallow water
*/
static void ComputeFluxY(PetscReal g, PetscReal h, PetscReal hu, PetscReal hv, PetscReal *G_h, PetscReal *G_hu, PetscReal *G_hv, PetscReal *v, PetscReal *c)
{
  if (h > 1e-10) {
    *v    = hv / h;
    *c    = PetscSqrtReal(g * h);
    *G_h  = hv;
    *G_hu = hv * (hu / h);
    *G_hv = hv * *v + 0.5 * g * h * h;
  } else {
    *v    = 0.0;
    *c    = 0.0;
    *G_h  = 0.0;
    *G_hu = 0.0;
    *G_hv = 0.0;
  }
}

/*
  ShallowWaterRHS2D - Compute the right-hand side of the 2D shallow water equations
*/
static PetscErrorCode ShallowWaterRHS2D(TS ts, PetscReal t, Vec X, Vec F_vec, void *ctx)
{
  ShallowWater2DCtx  *sw = (ShallowWater2DCtx *)ctx;
  Vec                 X_local;
  const PetscScalar ***x;
  PetscScalar       ***f;
  PetscInt            xs, ys, xm, ym, i, j;

  PetscFunctionBeginUser;
  (void)ts;
  (void)t;

  PetscCall(DMDAGetCorners(sw->da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMGetLocalVector(sw->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayDOFRead(sw->da, X_local, (void *)&x));
  PetscCall(DMDAVecGetArrayDOF(sw->da, F_vec, &f));

  if (sw->flux_type == EX4_FLUX_RUSANOV) {
    /* First-order Rusanov (Local Lax-Friedrichs) scheme */
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscReal h   = PetscRealPart(x[j][i][0]);
        PetscReal hu  = PetscRealPart(x[j][i][1]);
        PetscReal hv  = PetscRealPart(x[j][i][2]);
        PetscReal h_im1  = PetscRealPart(x[j][i - 1][0]);
        PetscReal hu_im1 = PetscRealPart(x[j][i - 1][1]);
        PetscReal hv_im1 = PetscRealPart(x[j][i - 1][2]);
        PetscReal h_ip1  = PetscRealPart(x[j][i + 1][0]);
        PetscReal hu_ip1 = PetscRealPart(x[j][i + 1][1]);
        PetscReal hv_ip1 = PetscRealPart(x[j][i + 1][2]);
        PetscReal h_jm1  = PetscRealPart(x[j - 1][i][0]);
        PetscReal hu_jm1 = PetscRealPart(x[j - 1][i][1]);
        PetscReal hv_jm1 = PetscRealPart(x[j - 1][i][2]);
        PetscReal h_jp1  = PetscRealPart(x[j + 1][i][0]);
        PetscReal hu_jp1 = PetscRealPart(x[j + 1][i][1]);
        PetscReal hv_jp1 = PetscRealPart(x[j + 1][i][2]);

        /* X-direction fluxes */
        PetscReal F_h_i, F_hu_i, F_hv_i, u, c;
        PetscReal F_h_im1, F_hu_im1, F_hv_im1, u_im1, c_im1;
        PetscReal F_h_ip1, F_hu_ip1, F_hv_ip1, u_ip1, c_ip1;

        ComputeFluxX(sw->g, h, hu, hv, &F_h_i, &F_hu_i, &F_hv_i, &u, &c);
        ComputeFluxX(sw->g, h_im1, hu_im1, hv_im1, &F_h_im1, &F_hu_im1, &F_hv_im1, &u_im1, &c_im1);
        ComputeFluxX(sw->g, h_ip1, hu_ip1, hv_ip1, &F_h_ip1, &F_hu_ip1, &F_hv_ip1, &u_ip1, &c_ip1);

        PetscReal alpha_left  = PetscMax(PetscAbsReal(u_im1) + c_im1, PetscAbsReal(u) + c);
        PetscReal alpha_right = PetscMax(PetscAbsReal(u) + c, PetscAbsReal(u_ip1) + c_ip1);

        PetscReal flux_h_left  = 0.5 * (F_h_im1 + F_h_i - alpha_left * (h - h_im1));
        PetscReal flux_hu_left = 0.5 * (F_hu_im1 + F_hu_i - alpha_left * (hu - hu_im1));
        PetscReal flux_hv_left = 0.5 * (F_hv_im1 + F_hv_i - alpha_left * (hv - hv_im1));

        PetscReal flux_h_right  = 0.5 * (F_h_i + F_h_ip1 - alpha_right * (h_ip1 - h));
        PetscReal flux_hu_right = 0.5 * (F_hu_i + F_hu_ip1 - alpha_right * (hu_ip1 - hu));
        PetscReal flux_hv_right = 0.5 * (F_hv_i + F_hv_ip1 - alpha_right * (hv_ip1 - hv));

        /* Y-direction fluxes */
        PetscReal G_h_j, G_hu_j, G_hv_j, v, c_y;
        PetscReal G_h_jm1, G_hu_jm1, G_hv_jm1, v_jm1, c_jm1;
        PetscReal G_h_jp1, G_hu_jp1, G_hv_jp1, v_jp1, c_jp1;

        ComputeFluxY(sw->g, h, hu, hv, &G_h_j, &G_hu_j, &G_hv_j, &v, &c_y);
        ComputeFluxY(sw->g, h_jm1, hu_jm1, hv_jm1, &G_h_jm1, &G_hu_jm1, &G_hv_jm1, &v_jm1, &c_jm1);
        ComputeFluxY(sw->g, h_jp1, hu_jp1, hv_jp1, &G_h_jp1, &G_hu_jp1, &G_hv_jp1, &v_jp1, &c_jp1);

        PetscReal beta_bottom = PetscMax(PetscAbsReal(v_jm1) + c_jm1, PetscAbsReal(v) + c_y);
        PetscReal beta_top    = PetscMax(PetscAbsReal(v) + c_y, PetscAbsReal(v_jp1) + c_jp1);

        PetscReal flux_h_bottom  = 0.5 * (G_h_jm1 + G_h_j - beta_bottom * (h - h_jm1));
        PetscReal flux_hu_bottom = 0.5 * (G_hu_jm1 + G_hu_j - beta_bottom * (hu - hu_jm1));
        PetscReal flux_hv_bottom = 0.5 * (G_hv_jm1 + G_hv_j - beta_bottom * (hv - hv_jm1));

        PetscReal flux_h_top  = 0.5 * (G_h_j + G_h_jp1 - beta_top * (h_jp1 - h));
        PetscReal flux_hu_top = 0.5 * (G_hu_j + G_hu_jp1 - beta_top * (hu_jp1 - hu));
        PetscReal flux_hv_top = 0.5 * (G_hv_j + G_hv_jp1 - beta_top * (hv_jp1 - hv));

        /* Update RHS using finite volume method */
        f[j][i][0] = -(flux_h_right - flux_h_left) / sw->dx - (flux_h_top - flux_h_bottom) / sw->dy;
        f[j][i][1] = -(flux_hu_right - flux_hu_left) / sw->dx - (flux_hu_top - flux_hu_bottom) / sw->dy;
        f[j][i][2] = -(flux_hv_right - flux_hv_left) / sw->dx - (flux_hv_top - flux_hv_bottom) / sw->dy;
      }
    }
  } else {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MC limiter not yet implemented for 2D");
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(sw->da, X_local, (void *)&x));
  PetscCall(DMDAVecRestoreArrayDOF(sw->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(sw->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWater2DContextCreate - Create and initialize a 2D shallow water context
*/
static PetscErrorCode ShallowWater2DContextCreate(DM da, PetscInt nx, PetscInt ny, PetscReal Lx, PetscReal Ly, PetscReal g, PetscReal dt, PetscReal h0, PetscReal Ax, PetscReal Ay, Ex4FluxType flux_type, ShallowWater2DCtx **ctx)
{
  ShallowWater2DCtx *sw;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&sw));
  sw->da        = da;
  sw->nx        = nx;
  sw->ny        = ny;
  sw->Lx        = Lx;
  sw->Ly        = Ly;
  sw->g         = g;
  sw->dx        = Lx / nx;
  sw->dy        = Ly / ny;
  sw->dt        = dt;
  sw->h0        = h0;
  sw->Ax        = Ax;
  sw->Ay        = Ay;
  sw->flux_type = flux_type;

  PetscCall(TSCreate(PetscObjectComm((PetscObject)da), &sw->ts));
  PetscCall(TSSetProblemType(sw->ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(sw->ts, NULL, ShallowWaterRHS2D, sw));
  PetscCall(TSSetType(sw->ts, TSRK));
  PetscCall(TSRKSetType(sw->ts, TSRK4));
  PetscCall(TSSetTimeStep(sw->ts, dt));
  PetscCall(TSSetMaxSteps(sw->ts, 1));
  PetscCall(TSSetMaxTime(sw->ts, dt));
  PetscCall(TSSetExactFinalTime(sw->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(sw->ts));

  *ctx = sw;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWater2DContextDestroy - Destroy a 2D shallow water context
*/
static PetscErrorCode ShallowWater2DContextDestroy(ShallowWater2DCtx **ctx)
{
  PetscFunctionBeginUser;
  if (!ctx || !*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSDestroy(&(*ctx)->ts));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterStep2D - Advance state vector one time step
*/
static PetscErrorCode ShallowWaterStep2D(Vec x_in, Vec x_out, void *ctx)
{
  ShallowWater2DCtx *sw = (ShallowWater2DCtx *)ctx;

  PetscFunctionBeginUser;
  if (x_in != x_out) PetscCall(VecCopy(x_in, x_out));

  PetscCall(TSSetTime(sw->ts, 0.0));
  PetscCall(TSSetStepNumber(sw->ts, 0));
  PetscCall(TSSetMaxTime(sw->ts, sw->dt));
  PetscCall(TSSolve(sw->ts, x_out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterSolution_Wave2D - Analytic 2D traveling wave solution
*/
static PetscErrorCode ShallowWaterSolution_Wave2D(PetscReal Lx, PetscReal Ly, PetscReal x, PetscReal y, PetscReal t, PetscReal g, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscReal *h, PetscReal *hu, PetscReal *hv)
{
  PetscReal kx, ky, omega_x, omega_y, c;

  PetscFunctionBeginUser;
  /* Wave parameters */
  c       = PetscSqrtReal(g * h0);
  kx      = 2.0 * PETSC_PI / Lx;
  ky      = 2.0 * PETSC_PI / Ly;
  omega_x = c * kx;
  omega_y = c * ky;

  /* Height field: superposition of waves in x and y */
  PetscReal h_pert_x = Ax * PetscSinReal(kx * x - omega_x * t);
  PetscReal h_pert_y = Ay * PetscSinReal(ky * y - omega_y * t);
  *h                 = h0 + h_pert_x + h_pert_y;

  /* Velocity fields (linearized) */
  PetscReal u = (c / h0) * Ax * PetscCosReal(kx * x - omega_x * t);
  PetscReal v = (c / h0) * Ay * PetscCosReal(ky * y - omega_y * t);

  *hu = (*h) * u;
  *hv = (*h) * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeL2Error - Compute L2 error against analytic solution
*/
static PetscErrorCode ComputeL2Error(Vec numerical, ShallowWater2DCtx *sw, PetscReal time, PetscReal *error)
{
  const PetscScalar ***x_array;
  PetscInt              xs, ys, xm, ym, i, j;
  PetscReal             local_sum = 0.0, global_sum;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(sw->da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAVecGetArrayDOFRead(sw->da, numerical, (void *)&x_array));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal x  = ((PetscReal)i + 0.5) * sw->dx;
      PetscReal y  = ((PetscReal)j + 0.5) * sw->dy;

      PetscReal h_num  = PetscRealPart(x_array[j][i][0]);
      PetscReal hu_num = PetscRealPart(x_array[j][i][1]);
      PetscReal hv_num = PetscRealPart(x_array[j][i][2]);

      PetscReal h_exact, hu_exact, hv_exact;
      PetscCall(ShallowWaterSolution_Wave2D(sw->Lx, sw->Ly, x, y, time, sw->g, sw->h0, sw->Ax, sw->Ay, &h_exact, &hu_exact, &hv_exact));

      PetscReal diff_h  = h_num - h_exact;
      PetscReal diff_hu = hu_num - hu_exact;
      PetscReal diff_hv = hv_num - hv_exact;

      local_sum += diff_h * diff_h + diff_hu * diff_hu + diff_hv * diff_hv;
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(sw->da, numerical, (void *)&x_array));
  PetscCall(MPIU_Allreduce(&local_sum, &global_sum, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)sw->da)));
  *error = PetscSqrtReal(global_sum / (sw->nx * sw->ny * 3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  /* Configuration parameters */
  const PetscInt    ndof           = 3; /* h, hu, hv */
  PetscInt          nx             = DEFAULT_NX;
  PetscInt          ny             = DEFAULT_NY;
  PetscInt          steps          = DEFAULT_STEPS;
  PetscInt          progress_freq  = DEFAULT_PROGRESS_FREQ;
  PetscReal         g              = DEFAULT_G;
  PetscReal         dt             = DEFAULT_DT;
  PetscReal         Lx             = DEFAULT_LX;
  PetscReal         Ly             = DEFAULT_LY;
  PetscReal         h0             = DEFAULT_H0;
  PetscReal         Ax             = DEFAULT_AX;
  PetscReal         Ay             = DEFAULT_AY;
  Ex4FluxType       flux_type      = EX4_FLUX_RUSANOV;
  char              output_file[PETSC_MAX_PATH_LEN];
  PetscBool         output_enabled = PETSC_FALSE;
  FILE             *fp             = NULL;

  /* PETSc objects */
  ShallowWater2DCtx *sw_ctx = NULL;
  DM                 da_state;
  Vec                state;
  PetscInt           step;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Parse command-line options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "2D Shallow Water Equations", NULL);
  PetscCall(PetscOptionsInt("-nx", "Number of grid points in x", "", nx, &nx, NULL));
  PetscCall(PetscOptionsInt("-ny", "Number of grid points in y", "", ny, &ny, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsReal("-g", "Gravitational constant", "", g, &g, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", "", dt, &dt, NULL));
  PetscCall(PetscOptionsReal("-Lx", "Domain length in x", "", Lx, &Lx, NULL));
  PetscCall(PetscOptionsReal("-Ly", "Domain length in y", "", Ly, &Ly, NULL));
  PetscCall(PetscOptionsReal("-h0", "Mean water height", "", h0, &h0, NULL));
  PetscCall(PetscOptionsReal("-Ax", "Wave amplitude in x", "", Ax, &Ax, NULL));
  PetscCall(PetscOptionsReal("-Ay", "Wave amplitude in y", "", Ay, &Ay, NULL));
  PetscCall(PetscOptionsInt("-progress_freq", "Print progress every N steps (0 = only first/last)", "", progress_freq, &progress_freq, NULL));
  PetscCall(PetscOptionsString("-output_file", "Output file for visualization data", "", "", output_file, sizeof(output_file), &output_enabled));
  PetscCall(PetscOptionsEnum("-ex4_flux", "Flux scheme (rusanov/mc)", "", Ex4FluxTypes, (PetscEnum)flux_type, (PetscEnum *)&flux_type, NULL));
  PetscOptionsEnd();

  /* Validate parameters */
  PetscCheck(nx > 0 && ny > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid dimensions must be positive");
  PetscCheck(steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative");
  PetscCheck(dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step must be positive");

  /* Create 2D periodic DMDA with 3 DOF (h, hu, hv) */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE, ndof, 2, NULL, NULL, &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));

  /* Create shallow water context */
  PetscCall(ShallowWater2DContextCreate(da_state, nx, ny, Lx, Ly, g, dt, h0, Ax, Ay, flux_type, &sw_ctx));

  /* Initialize state vector */
  PetscCall(DMCreateGlobalVector(da_state, &state));

  /* Set initial condition from analytic solution */
  {
    PetscScalar ***x_array;
    PetscInt      xs, ys, xm, ym, i, j;
    PetscCall(DMDAGetCorners(da_state, &xs, &ys, NULL, &xm, &ym, NULL));
    PetscCall(DMDAVecGetArrayDOF(da_state, state, &x_array));
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscReal x   = ((PetscReal)i + 0.5) * sw_ctx->dx;
        PetscReal y   = ((PetscReal)j + 0.5) * sw_ctx->dy;
        PetscReal h, hu, hv;
        PetscCall(ShallowWaterSolution_Wave2D(Lx, Ly, x, y, 0.0, g, h0, Ax, Ay, &h, &hu, &hv));
        x_array[j][i][0] = h;
        x_array[j][i][1] = hu;
        x_array[j][i][2] = hv;
      }
    }
    PetscCall(DMDAVecRestoreArrayDOF(da_state, state, &x_array));
  }

  /* Print configuration summary */
  {
    const char *flux_name = (flux_type == EX4_FLUX_RUSANOV) ? "Rusanov (1st order)" : "MC (2nd order)";
    PetscReal   dx        = Lx / nx;
    PetscReal   dy        = Ly / ny;
    PetscReal   c         = PetscSqrtReal(g * h0);
    PetscReal   cfl       = dt * c * (1.0 / dx + 1.0 / dy);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D Shallow Water Equations\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==========================\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "  Flux scheme           : %s\n"
                          "  Grid dimensions       : %" PetscInt_FMT " x %" PetscInt_FMT "\n"
                          "  State dimension       : %" PetscInt_FMT " (%" PetscInt_FMT " grid points x %d DOF)\n"
                          "  Domain size           : %.2f x %.2f\n"
                          "  Grid spacing          : dx=%.4f, dy=%.4f\n"
                          "  Mean height (h0)      : %.4f\n"
                          "  Wave amplitudes       : Ax=%.4f, Ay=%.4f\n"
                          "  Gravitational const   : %.4f\n"
                          "  Wave speed (c)        : %.4f\n"
                          "  Time step (dt)        : %.4f\n"
                          "  CFL number            : %.4f\n"
                          "  Total steps           : %" PetscInt_FMT "\n"
                          "  Total time            : %.4f\n\n",
                          flux_name, nx, ny, nx * ny * ndof, nx * ny, (int)ndof, (double)Lx, (double)Ly, (double)dx, (double)dy, (double)h0, (double)Ax, (double)Ay, (double)g, (double)c, (double)dt, (double)cfl, steps, (double)(steps * dt)));
  }

  /* Open output file if requested */
  if (output_enabled) {
    PetscCall(PetscFOpen(PETSC_COMM_WORLD, output_file, "w", &fp));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# 2D Shallow Water Equations Output\n"));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# nx=%d, ny=%d, ndof=%d\n", (int)nx, (int)ny, (int)ndof));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# Lx=%.6f, Ly=%.6f, dt=%.6f, g=%.6f\n", (double)Lx, (double)Ly, (double)dt, (double)g));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# h0=%.6f, Ax=%.6f, Ay=%.6f\n", (double)h0, (double)Ax, (double)Ay));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# Format: step time [h hu hv]x(nx*ny) L2_error\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Writing output to: %s\n\n", output_file));
  }

  /* Print initial condition */
  {
    PetscReal l2_error;
    PetscCall(ComputeL2Error(state, sw_ctx, 0.0, &l2_error));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4d, time %6.3f  L2_error %.5e [initial]\n", 0, 0.0, (double)l2_error));

    /* Write initial condition to file */
    if (output_enabled && fp) {
      const PetscScalar *x_array;
      PetscInt           i;
      PetscCall(VecGetArrayRead(state, &x_array));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "0 0.000000"));
      for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(x_array[i])));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e\n", (double)l2_error));
      PetscCall(VecRestoreArrayRead(state, &x_array));
    }
  }

  /* Main time stepping loop */
  for (step = 1; step <= steps; step++) {
    PetscReal time = step * dt;

    /* Advance one time step */
    PetscCall(ShallowWaterStep2D(state, state, sw_ctx));

    /* Compute error */
    PetscReal l2_error;
    PetscCall(ComputeL2Error(state, sw_ctx, time, &l2_error));

    /* Write to output file */
    if (output_enabled && fp) {
      const PetscScalar *x_array;
      PetscInt           i;
      PetscCall(VecGetArrayRead(state, &x_array));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "%d %.6f", (int)step, (double)time));
      for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(x_array[i])));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e\n", (double)l2_error));
      PetscCall(VecRestoreArrayRead(state, &x_array));
    }

    /* Progress reporting */
    if (progress_freq == 0) {
      if (step == steps) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  L2_error %.5e\n", step, (double)time, (double)l2_error));
    } else {
      if ((step % progress_freq == 0) || (step == steps)) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  L2_error %.5e\n", step, (double)time, (double)l2_error));
    }
  }

  /* Close output file */
  if (output_enabled && fp) {
    PetscCall(PetscFClose(PETSC_COMM_WORLD, fp));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nOutput written to: %s\n", output_file));
  }

  /* Cleanup */
  PetscCall(VecDestroy(&state));
  PetscCall(DMDestroy(&da_state));
  PetscCall(ShallowWater2DContextDestroy(&sw_ctx));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: !complex
    diff_args: -j
    args: -steps 10 -progress_freq 0 -nx 20 -ny 20

    test:
      suffix: wave2d
      args: -Ax 0.1 -Ay 0.1

    test:
      nsize: 2
      suffix: parallel
      args: -Ax 0.15 -Ay 0.15

TEST*/
