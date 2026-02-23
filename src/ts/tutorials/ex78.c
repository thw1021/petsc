static char help[] = "Zel'dovich Heat Conduction Test using Finite Volume discretization and Runge-Kutta Supertimesteppers\n";
/*
We test the Runge-Kutta supertimestepper (rks) implementation against a standard explicit Euler method by solving the Zel'dovich problem of a 1D propagating conduction heat front,
\begin{align}
  &\frac{\partial T}{\partial t} = \frac{\partial}{\partial z} \left(-T^{5/2} \frac{\partial T}{\partial z}\right), \;\;\;\;\;\;0\leq z \leq 2\\
  &T'\left(0,t\right)=0,
\end{align}
with the initial condition,
\begin{equation}
  T(z,t=0) = \begin{cases}
    \left(1-\frac{5}{4}z^2\right)^{2/5}, & 0 \leq z \leq \frac{2}{\sqrt{5}}, \\
    0, & \frac{2}{\sqrt{5}} < z \leq 2.
    \end{cases}
\end{equation}

The exact self-similar solution for this boundary value problem model is given by,
\begin{equation}
  T(z,t) = \begin{cases}
    \frac{1}{(1 + \frac{9}{2} t)^{2/9}} \left(1 - \frac{z^2}{\frac{4}{5} (1 + \frac{9}{2} t)^{4/9}}\right)^{2/5}, & 0 \leq z \leq \frac{2}{\sqrt{5}} (1 + \frac{9}{2} t)^{2/9} \\
    0, & \frac{2}{\sqrt{5}} (1 + \frac{9}{2} t)^{2/9} < z \leq 2
    \end{cases}
\end{equation}
*/

#include <petscts.h>
#include <petscdraw.h>
#include "petscdmda.h"
#include "petscdmlabel.h"

typedef struct {
  PetscInt    ostep;
  PetscReal   xmin;
  PetscReal   xmax;
  PetscReal   ymin;
  PetscReal   ymax;
  PetscReal   zmin;
  PetscReal   zmax;
  PetscReal   Tmin;
  PetscReal   Tmax;
  PetscReal   L2;
  PetscReal   Linf;
  PetscReal   L2_max;
  PetscReal   t_l2_max;
  PetscReal   L2_sum;
  PetscReal   Linf_sum;
  PetscDrawLG drawlg;
  PetscBool   compute_error;
  PetscBool   monitor_temp;
  PetscBool   useFCTLimiter;
  PetscBool   enforce_zero_T;
  PetscBool   printData;
  PetscBool   use_ghost_cells;
  PetscInt    num_ghost_cells[3];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt nGC = 3;

  PetscFunctionBeginUser;
  options->ostep              = 1;
  options->xmin               = 0.0;
  options->xmax               = 2.0;
  options->ymin               = -1.0;
  options->ymax               = 1.0;
  options->zmin               = -1.0;
  options->zmax               = 1.0;
  options->Tmin               = 0.0;
  options->Tmax               = 1.0;
  options->drawlg             = NULL;
  options->monitor_temp       = PETSC_FALSE;
  options->compute_error      = PETSC_FALSE;
  options->useFCTLimiter      = PETSC_FALSE;
  options->printData          = PETSC_FALSE;
  options->use_ghost_cells    = PETSC_FALSE;
  options->L2                 = 0;
  options->Linf               = 0;
  options->t_l2_max           = 0;
  options->L2_sum             = 0;
  options->Linf_sum           = 0;
  options->num_ghost_cells[0] = 0;
  options->num_ghost_cells[1] = 0;
  options->num_ghost_cells[2] = 0;

  PetscOptionsBegin(comm, "", "Zel'dovich test options", "TS");
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "", options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsReal("-xmin", "X min", "", options->xmin, &options->xmin, NULL));
  PetscCall(PetscOptionsReal("-xmax", "X max", "", options->xmax, &options->xmax, NULL));
  PetscCall(PetscOptionsReal("-ymin", "Y min", "", options->ymin, &options->ymin, NULL));
  PetscCall(PetscOptionsReal("-ymax", "Y max", "", options->ymax, &options->ymax, NULL));
  PetscCall(PetscOptionsReal("-zmin", "Z min", "", options->zmin, &options->zmin, NULL));
  PetscCall(PetscOptionsReal("-zmax", "Z max", "", options->zmax, &options->zmax, NULL));
  PetscCall(PetscOptionsReal("-Tmin", "T min", "", options->Tmin, &options->Tmin, NULL));
  PetscCall(PetscOptionsReal("-Tmax", "T max", "", options->Tmax, &options->Tmax, NULL));
  PetscCall(PetscOptionsBool("-compute_error", "Flag to output error at each step", "", options->compute_error, &options->compute_error, NULL));
  PetscCall(PetscOptionsBool("-temp_monitor", "Flag to plot temperature", "", options->monitor_temp, &options->monitor_temp, NULL));
  PetscCall(PetscOptionsBool("-use_fct_limiter", "Flag to use ARMS limiter", "", options->useFCTLimiter, &options->useFCTLimiter, NULL));
  PetscCall(PetscOptionsBool("-print_data", "Flag to print temperature data in monitor", "", options->printData, &options->printData, NULL));
  PetscCall(PetscOptionsBool("-use_ghost_cells", "Flag to pad real cells with ghost cells in all dimensions", "", options->use_ghost_cells, &options->use_ghost_cells, NULL));
  PetscCall(PetscOptionsIntArray("-num_ghost_cells", "Number of ghost cells on either end of mesh", "", options->num_ghost_cells, &nGC, NULL));

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupContext(DM dm, AppCtx *user)
{
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  if (user->monitor_temp) {
    PetscDraw     draw;
    PetscDrawAxis axis;
    const char   *legend[2] = {"Exact", "Numerical"};
    const int     colors[2] = {2, 1};

    PetscCall(PetscDrawCreate(PETSC_COMM_SELF, NULL, "Zel'dovich Temperature Profile", 400, 300, 400, 300, &draw));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawSetSave(draw, "ex78_temp"));
    PetscCall(PetscDrawLGCreate(draw, 2, &user->drawlg));
    PetscCall(PetscDrawLGSetLegend(user->drawlg, legend));
    PetscCall(PetscDrawLGSetColors(user->drawlg, colors));
    PetscCall(PetscDrawLGSetLimits(user->drawlg, user->xmin, user->xmax, user->Tmin, user->Tmax));

    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawLGGetAxis(user->drawlg, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Zel'dovich Temperature Profile", "z", "T(z)"));
    PetscCall(PetscDrawLGSetLimits(user->drawlg, user->xmin, user->xmax, 0.0, 1.1));
    PetscCall(PetscDrawAxisSetLimits(axis, user->xmin, user->xmax, -0.1, 1.1));
    PetscCall(PetscDrawAxisSetHoldLimits(axis, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(PetscDrawLGDestroy(&user->drawlg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PlotTemperature(TS ts, PetscInt step, PetscReal t, Vec T, void *ctx)
{
  AppCtx            *user  = (AppCtx *)ctx;
  const PetscInt     ostep = user->ostep;
  DM                 dm;
  const PetscScalar *t_array;
  PetscInt           i, M, xs, xm, ys, ym, zs, zm;
  char               title[PETSC_MAX_PATH_LEN];
  PetscReal          dx, dy, dz;
  PetscReal          xvals[2], vals[2];
  PetscReal          z, Tnum, Texact, tscale, front, denom;
  PetscDraw          draw;
  PetscDrawAxis      axis;

  PetscFunctionBeginUser;
  if (user->printData && step == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "t\tz\tT_exact\tT_numerical\n"));
  if (step % ostep == 0) {
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMGetLocalVector(dm, &Tloc));
    PetscCall(DMGlobalToLocalBegin(dm, T, INSERT_VALUES, Tloc));
    PetscCall(DMGlobalToLocalEnd(dm, T, INSERT_VALUES, Tloc));
    DMGlobalToLocalBegin(dm, T, INSERT_VALUES, Tloc);
    DMGlobalToLocalEnd(dm, T, INSERT_VALUES, Tloc);
    PetscCall(VecGetArrayRead(Tloc, &t_array));
    PetscCall(PetscDrawLGReset(user->drawlg));
    PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm));
    dx = (user->xmax - user->xmin) / (xm - xs - 2 * user->num_ghost_cells[0]);
    dy = (user->ymax - user->ymin) / (ym - ys - 2 * user->num_ghost_cells[1]);
    dz = (user->zmax - user->zmin) / (zm - zs - 2 * user->num_ghost_cells[2]);

    PetscReal err2 = 0.0, errmax = 0.0;
    PetscInt  j_center = ys + ym / 2;
    PetscInt  k_center = zs + zm / 2;

    for (PetscInt i = xs + user->num_ghost_cells[0]; i < xs + xm - user->num_ghost_cells[0]; i++) {
      PetscReal x   = user->xmin + (i - user->num_ghost_cells[0]) * dx + 0.5 * dx;
      PetscInt  idx = (k_center - zs) * ym * xm + (j_center - ys) * xm + (i - xs);

      PetscReal Tnum = PetscRealPart(t_array[idx]);

      /* exact solution */
      PetscReal Texact;
      PetscReal tscale = PetscPowReal(1.0 + 4.5 * t, 2.0 / 9.0);
      PetscReal front  = (2.0 / PetscSqrtReal(5.0)) * tscale;

      if (x <= front) {
        PetscReal denom = 0.8 * PetscPowReal(1.0 + 4.5 * t, 4.0 / 9.0);
        Texact          = PetscPowReal(1.0 + 4.5 * t, -2.0 / 9.0) * PetscPowReal(1.0 - x * x / denom, 0.4);
      } else Texact = 0.0;

      xvals[0] = x;
      xvals[1] = x;
      vals[0]  = Texact;
      vals[1]  = Tnum;

      PetscCall(PetscDrawLGAddPoint(user->drawlg, xvals, vals));
    }

    PetscCall(PetscDrawLGGetDraw(user->drawlg, &draw));
    PetscCall(PetscSNPrintf(title, sizeof(title), "Zel'dovich Temperature Profile t = %.6e", (double)t));
    PetscCall(PetscDrawSetTitle(draw, title));
    PetscCall(PetscDrawLGGetAxis(user->drawlg, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, title, "x", "T(x)"));

    PetscCall(PetscDrawLGDraw(user->drawlg));
    PetscCall(PetscDrawSave(draw));
    PetscCall(VecRestoreArrayRead(Tloc, &t_array));
    PetscCall(DMRestoreLocalVector(dm, &Tloc));
    PetscCall(VecViewFromOptions(T, NULL, "-T_view"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeError(TS ts, PetscInt step, PetscReal t, Vec T, void *ctx)
{
  AppCtx              *user = (AppCtx *)ctx;
  DM                   dm;
  Vec                  Tloc;
  const PetscScalar ***Tarr;
  PetscInt             xs, ys, zs, xm, ym, zm;
  PetscInt             i;

  PetscReal dx, dy, dz;
  PetscReal err2 = 0.0, errmax = 0.0, err2_RMS = 0.0;
  PetscReal dt, maxT;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetMaxTime(ts, &maxT));

  /* --- get local vector --- */
  PetscCall(DMGetLocalVector(dm, &Tloc));
  PetscCall(DMGlobalToLocalBegin(dm, T, INSERT_VALUES, Tloc));
  PetscCall(DMGlobalToLocalEnd(dm, T, INSERT_VALUES, Tloc));
  PetscCall(DMDAVecGetArrayRead(dm, Tloc, &Tarr));

  PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm));

  dx = (user->xmax - user->xmin) / (xm - xs - 2 * user->num_ghost_cells[0]);
  dy = (user->ymax - user->ymin) / (ym - ys - 2 * user->num_ghost_cells[1]);
  dz = (user->zmax - user->zmin) / (zm - zs - 2 * user->num_ghost_cells[2]);

  PetscInt Nx = xm - xs - 2 * user->num_ghost_cells[0];

  /* center slice in y,z */
  PetscInt j_center = ys + ym / 2;
  PetscInt k_center = zs + zm / 2;

  /* exact solution parameters */
  PetscReal alpha = 1.0 + 4.5 * t;
  PetscReal front = (2.0 / PetscSqrtReal(5.0)) * PetscPowReal(alpha, 2.0 / 9.0);
  PetscReal denom = 0.8 * PetscPowReal(alpha, 4.0 / 9.0);

  /* loop over owned interior x-cells */
  for (i = xs + user->num_ghost_cells[0]; i < xs + xm - user->num_ghost_cells[0]; i++) {
    PetscReal x = user->xmin + (i - xs - user->num_ghost_cells[0]) * dx + 0.5 * dx;

    PetscReal Tnum = PetscRealPart(Tarr[k_center][j_center][i]);

    PetscReal Texact = 0.0;
    if (x <= front) {
      PetscReal arg = 1.0 - (x * x) / denom;
      if (arg > 0.0) Texact = PetscPowReal(alpha, -2.0 / 9.0) * PetscPowReal(arg, 0.4);
    }

    PetscReal diff = Tnum - Texact;
    err2 += diff * diff;
    errmax = PetscMax(errmax, PetscAbsReal(diff));
  }

  PetscCall(DMDAVecRestoreArrayRead(dm, Tloc, &Tarr));
  PetscCall(DMRestoreLocalVector(dm, &Tloc));

  /* reduce across MPI */
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &err2, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &errmax, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)dm)));

  err2 = PetscSqrtReal(err2);

  user->L2   = err2;
  user->Linf = errmax;
  user->L2_sum += err2;
  user->Linf_sum += errmax;
  if (err2 > user->L2_max) {
    user->L2_max   = err2;
    user->t_l2_max = t;
  }
  PetscReal L2_avg   = (step > 0) ? user->L2_sum / step : 0.0;
  PetscReal Linf_avg = (step > 0) ? user->Linf_sum / step : 0.0;
  PetscReal Linf_avg = user->Linf_sum / step;

  if (step % user->ostep == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "t = %.6e  L2 = %.6e  Linf = %.6e  L2_avg = %.6e  Linf_avg = %.6e  t_max = %.6e  L2_max = %.6e\n", (double)t, (double)user->L2, (double)user->Linf, (double)L2_avg, (double)Linf_avg, (double)user->t_l2_max,
                          (double)user->L2_max));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal SignValue(PetscReal a, PetscReal x)
{
  return (x >= 0.0) ? a : -a;
}

static PetscErrorCode RHSFunction(TS ts, PetscReal time, Vec T, Vec dTdt, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  DM                 dm;
  PetscInt           xs, xm, ys, ym, zs, zm;
  const PetscScalar *t_arr;
  PetscScalar       *dtdt_arr;
  PetscReal          dx, dy, dz, dt;
  PetscReal          rho = 1.0, cv = 1.0, T_right = 0.0;
  PetscReal         *flux_x, *flux_y, *flux_z;
  PetscReal         *physFlux_x, *physFlux_y, *physFlux_z;
  PetscReal         *limiter_out, *limiter_in;
  PetscInt           local_size, step;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &step));

  PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm));
  dx = (user->xmax - user->xmin) / (xm - xs - 2 * user->num_ghost_cells[0]);
  dy = (user->ymax - user->ymin) / (ym - ys - 2 * user->num_ghost_cells[1]);
  dz = (user->zmax - user->zmin) / (zm - zs - 2 * user->num_ghost_cells[2]);

  PetscCall(VecGetArrayRead(T, &t_arr));
  PetscCall(VecGetArray(dTdt, &dtdt_arr));

  local_size = xm * ym * zm;
  PetscCall(PetscMalloc1(local_size, &flux_x));
  PetscCall(PetscMalloc1(local_size, &flux_y));
  PetscCall(PetscMalloc1(local_size, &flux_z));
  PetscCall(PetscMalloc1(local_size, &physFlux_x));
  PetscCall(PetscMalloc1(local_size, &physFlux_y));
  PetscCall(PetscMalloc1(local_size, &physFlux_z));
  PetscCall(PetscCalloc(local_size * sizeof(PetscReal), &limiter_out));
  PetscCall(PetscCalloc(local_size * sizeof(PetscReal), &limiter_in));
  /*
    Calculate raw, unlimited flux densities at all interior faces
    Note: We calculate the flux at the LEFT, BOTTOM, BACK face of each cell (i,j,k)
  */
  for (PetscInt k = zs; k < zs + zm; k++) {
    for (PetscInt j = ys; j < ys + ym; j++) {
      for (PetscInt i = xs; i < xs + xm; i++) {
        PetscInt current_idx = (k - zs) * ym * xm + (j - ys) * xm + (i - xs);
        flux_x[current_idx]  = 0.0;
        flux_y[current_idx]  = 0.0;
        flux_z[current_idx]  = 0.0;

        if (i > xs) {
          PetscReal T_L    = t_arr[(k - zs) * ym * xm + (j - ys) * xm + ((i - 1) - xs)];
          PetscReal T_R    = t_arr[current_idx];
          PetscReal sqrtTL = PetscSqrtReal(T_L);
          PetscReal sqrtTR = PetscSqrtReal(T_R);

          PetscReal TL52 = T_L * T_L * sqrtTL;
          PetscReal TR52 = T_R * T_R * sqrtTR;

          PetscReal k_avg_x   = 0.5 * (TL52 + TR52);
          flux_x[current_idx] = dt * k_avg_x * (T_R - T_L) / dx;
        }
        if (j > ys) {
          PetscReal T_B = t_arr[(k - zs) * ym * xm + ((j - 1) - ys) * xm + (i - xs)];
          PetscReal T_T = t_arr[current_idx];

          PetscReal sqrtTB = PetscSqrtReal(T_B);
          PetscReal sqrtTT = PetscSqrtReal(T_T);

          PetscReal TB52 = T_B * T_B * sqrtTB;
          PetscReal TT52 = T_T * T_T * sqrtTT;

          PetscReal k_avg_x   = 0.5 * (TB52 + TT52);
          flux_y[current_idx] = dt * k_avg_x * (T_T - T_B) / dy;
        }
        if (k > zs) {
          PetscReal T_B = t_arr[((k - 1) - zs) * ym * xm + (j - ys) * xm + (i - xs)];
          PetscReal T_F = t_arr[current_idx];

          PetscReal sqrtTB = PetscSqrtReal(T_B);
          PetscReal sqrtTF = PetscSqrtReal(T_F);

          PetscReal TB52 = T_B * T_B * sqrtTB;
          PetscReal TF52 = T_F * T_F * sqrtTF;

          PetscReal k_avg_x   = 0.5 * (TB52 + TF52);
          flux_z[current_idx] = dt * k_avg_x * (T_F - T_B) / dz;
        }
      }
    }
  }

  if (user->useFCTLimiter) {
    /* Calculate limiter coefficients for each INTERIOR (non-ghost) CELL */
    for (PetscInt idx = 0; idx < local_size; ++idx) {
      physFlux_x[idx] = flux_x[idx] * (dy * dz);
      physFlux_y[idx] = flux_y[idx] * (dx * dz);
      physFlux_z[idx] = flux_z[idx] * (dx * dy);
    }

    for (PetscInt k = zs; k < zs + zm; k++) {
      for (PetscInt j = ys; j < ys + ym; j++) {
        for (PetscInt i = xs; i < xs + xm; i++) {
          PetscInt current_idx = (k - zs) * ym * xm + (j - ys) * xm + (i - xs);

          PetscReal T_center = t_arr[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)];
          PetscReal T_im1    = (i > xs) ? t_arr[(k - zs) * ym * xm + (j - ys) * xm + ((i - 1) - xs)] : T_center;
          PetscReal T_ip1    = (i < xs + xm - 1) ? t_arr[(k - zs) * ym * xm + (j - ys) * xm + ((i + 1) - xs)] : T_right;
          PetscReal T_jm1    = (j > ys) ? t_arr[(k - zs) * ym * xm + ((j - 1) - ys) * xm + (i - xs)] : T_center;
          PetscReal T_jp1    = (j < ys + ym - 1) ? t_arr[(k - zs) * ym * xm + ((j + 1) - ys) * xm + (i - xs)] : T_center;
          PetscReal T_km1    = (k > zs) ? t_arr[((k - 1) - zs) * ym * xm + (j - ys) * xm + (i - xs)] : T_center;
          PetscReal T_kp1    = (k < zs + zm - 1) ? t_arr[((k + 1) - zs) * ym * xm + (j - ys) * xm + (i - xs)] : T_center;

          PetscReal Tmin_neighbor = PetscMin(T_center, PetscMin(T_im1, PetscMin(T_ip1, PetscMin(T_jm1, PetscMin(T_jp1, PetscMin(T_km1, T_kp1))))));
          PetscReal Tmax_neighbor = PetscMax(T_center, PetscMax(T_im1, PetscMax(T_ip1, PetscMax(T_jm1, PetscMax(T_jp1, PetscMax(T_km1, T_kp1))))));

          PetscReal cell_volume = dx * dy * dz;
          PetscReal Q_i_minus   = cell_volume * rho * cv * (T_center - Tmin_neighbor);
          PetscReal Q_i_plus    = cell_volume * rho * cv * (Tmax_neighbor - T_center);
          PetscReal Q_i         = cell_volume * rho * cv * Tmax_neighbor;

          PetscReal Fleft  = physFlux_x[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)];
          PetscReal Fright = (i < xs + xm - 1) ? physFlux_x[(k - zs) * ym * xm + (j - ys) * xm + ((i + 1) - xs)] : 0.0;
          PetscReal Fbot   = physFlux_y[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)];
          PetscReal Ftop   = (j < ys + ym - 1) ? physFlux_y[(k - zs) * ym * xm + ((j + 1) - ys) * xm + (i - xs)] : 0.0;
          PetscReal Fback  = physFlux_z[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)];
          PetscReal Ffront = (k < zs + zm - 1) ? physFlux_z[((k + 1) - zs) * ym * xm + (j - ys) * xm + (i - xs)] : 0.0;

          PetscReal Ftot = PetscAbs(Fleft) + PetscAbs(Fright) + PetscAbs(Fbot) + PetscAbs(Ftop) + PetscAbs(Fback) + PetscAbs(Ffront);
          PetscReal Fnet = (Fright - Fleft) + (Ftop - Fbot) + (Ffront - Fback);

          PetscReal Fin  = 0.5 * (Ftot + Fnet);
          PetscReal Fout = 0.5 * (Ftot - Fnet);

          PetscReal denom   = PetscMax(1e-25, 1e-10 * Q_i);
          PetscReal fracout = Q_i_minus / PetscMax(Fout, denom);
          PetscReal fracin  = Q_i_plus / PetscMax(Fin, denom);

          limiter_out[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)] = PetscMin(1.0, PetscMax(0.0, fracout));
          limiter_in[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)]  = PetscMin(1.0, PetscMax(0.0, fracin));
        }
      }
    }

    /* Apply the corrected limiter logic to the flux densities */
    for (PetscInt k = zs; k < zs + zm; k++) {
      for (PetscInt j = ys; j < ys + ym; j++) {
        for (PetscInt i = xs; i < xs + xm; i++) {
          PetscReal limiter_factor;
          if (i > xs) {
            PetscInt  idx_face  = (k - zs) * ym * xm + (j - ys) * xm + (i - xs);
            PetscReal Flux_sign = SignValue(0.5, flux_x[idx_face]);
            if (flux_x[idx_face] >= 0.0) {
              limiter_factor = ((0.5 + Flux_sign) * (PetscMin(limiter_in[idx_face], limiter_out[idx_face])) + (0.5 - Flux_sign) * (PetscMin(limiter_out[idx_face], limiter_in[idx_face])));
            } else {
              limiter_factor = ((0.5 + Flux_sign) * (PetscMin(limiter_in[(k - zs) * ym * xm + (j - ys) * xm + ((i - 1) - xs)], limiter_out[idx_face])) + (0.5 - Flux_sign) * (PetscMin(limiter_out[(k - zs) * ym * xm + (j - ys) * xm + ((i - 1) - xs)], limiter_in[idx_face])));
            }
            flux_x[idx_face] *= limiter_factor;
          }
          if (j > ys) {
            PetscInt  idx_face  = (k - zs) * ym * xm + (j - ys) * xm + (i - xs);
            PetscReal Flux_sign = SignValue(0.5, flux_y[idx_face]);
            if (flux_y[idx_face] >= 0.0) {
              limiter_factor = ((0.5 + Flux_sign) * (PetscMin(limiter_in[idx_face], limiter_out[idx_face])) + (0.5 - Flux_sign) * (PetscMin(limiter_out[idx_face], limiter_in[idx_face])));
            } else {
              limiter_factor = ((0.5 + Flux_sign) * (PetscMin(limiter_in[(k - zs) * ym * xm + ((j - 1) - ys) * xm + ((i)-xs)], limiter_out[idx_face])) + (0.5 - Flux_sign) * (PetscMin(limiter_out[(k - zs) * ym * xm + ((j - 1) - ys) * xm + (i - xs)], limiter_in[idx_face])));
            }
            flux_y[idx_face] *= limiter_factor;
          }
          if (k > zs) {
            PetscInt  idx_face  = (k - zs) * ym * xm + (j - ys) * xm + (i - xs);
            PetscReal Flux_sign = SignValue(0.5, flux_z[idx_face]);
            if (flux_z[idx_face] >= 0.0) {
              limiter_factor = ((0.5 + Flux_sign) * (PetscMin(limiter_in[idx_face], limiter_out[idx_face])) + (0.5 - Flux_sign) * (PetscMin(limiter_out[idx_face], limiter_in[idx_face])));
            } else {
              limiter_factor = ((0.5 + Flux_sign) * (PetscMin(limiter_in[((k - 1) - zs) * ym * xm + (j - ys) * xm + ((i)-xs)], limiter_out[idx_face])) + (0.5 - Flux_sign) * (PetscMin(limiter_out[((k - 1) - zs) * ym * xm + (j - ys) * xm + (i - xs)], limiter_in[idx_face])));
            }
            flux_z[idx_face] *= limiter_factor;
          }
        }
      }
    }
  }
  /* Calculate the final time derivative (divergence of limited flux) */
  for (PetscInt k = zs; k < zs + zm; k++) {
    for (PetscInt j = ys; j < ys + ym; j++) {
      for (PetscInt i = xs; i < xs + xm; i++) {
        PetscReal right_flux = (i < xs + xm - 1) ? flux_x[(k - zs) * ym * xm + (j - ys) * xm + ((i + 1) - xs)] : flux_x[(k - zs) * ym * xm + (j - ys) * xm + ((i)-xs)];
        PetscReal left_flux  = flux_x[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)];
        PetscReal top_flux   = (j < ys + ym - 1) ? flux_y[(k - zs) * ym * xm + ((j + 1) - ys) * xm + (i - xs)] : flux_y[(k - zs) * ym * xm + ((j)-ys) * xm + (i - xs)];
        PetscReal bot_flux   = flux_y[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)];
        PetscReal front_flux = (k < zs + zm - 1) ? flux_z[((k + 1) - zs) * ym * xm + (j - ys) * xm + (i - xs)] : flux_z[((k)-zs) * ym * xm + (j - ys) * xm + (i - xs)];
        PetscReal back_flux  = flux_z[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)];

        PetscReal div_x = (right_flux - left_flux) / (dt * dx);
        PetscReal div_y = (top_flux - bot_flux) / (dt * dy);
        PetscReal div_z = (front_flux - back_flux) / (dt * dz);

        dtdt_arr[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)] = div_x + div_y + div_z;
      }
    }
  }

  /* --- Cleanup --- */
  PetscCall(PetscFree(flux_x));
  PetscCall(PetscFree(flux_y));
  PetscCall(PetscFree(flux_z));
  PetscCall(PetscFree(physFlux_x));
  PetscCall(PetscFree(physFlux_y));
  PetscCall(PetscFree(physFlux_z));
  PetscCall(PetscFree(limiter_out));
  PetscCall(PetscFree(limiter_in));

  PetscCall(VecRestoreArrayRead(T, &t_arr));
  PetscCall(VecRestoreArray(dTdt, &dtdt_arr));
  PetscCall(VecViewFromOptions(dTdt, NULL, "-rhs_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InitialConditions(TS ts, Vec U)
{
  AppCtx      *user;
  DM           dm;
  PetscScalar *u_arr;
  PetscInt     xs, xm, ys, ym, zs, zm;
  PetscReal    dx, dy, dz;
  PetscReal    x_coord;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm));
  dx = (user->xmax - user->xmin) / (xm - xs - 2 * user->num_ghost_cells[0]);
  dy = (user->ymax - user->ymin) / (ym - ys - 2 * user->num_ghost_cells[1]);
  dz = (user->zmax - user->zmin) / (zm - zs - 2 * user->num_ghost_cells[2]);
  PetscCall(VecGetArrayWrite(U, &u_arr));
  for (PetscInt k = zs; k < zs + zm; k++) {
    for (PetscInt j = ys; j < ys + ym; j++) {
      for (PetscInt i = xs; i < xs + xm; i++) {
        if (i < user->num_ghost_cells[0]) x_coord = user->xmin + (user->num_ghost_cells[0] - (i + 1)) * dx + 0.5 * dx;
        else x_coord = user->xmin + (i - user->num_ghost_cells[0]) * dx + 0.5 * dx;

        PetscReal x_sq = x_coord * x_coord;
        PetscReal val  = 0.0;

        if (x_coord <= 2.0 / PetscSqrtReal(5.0)) val = PetscPowReal(1.0 - (5.0 / 4.0) * x_sq, 2.0 / 5.0);

        if (val < 0 || PetscIsInfOrNanReal(val)) val = 0.0;
        u_arr[(k - zs) * ym * xm + (j - ys) * xm + (i - xs)] = val;
      }
    }
  }
  PetscCall(VecRestoreArrayWrite(U, &u_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;
  Vec    u;
  TS     ts;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  /* Structured DMDA Grid */
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 100, 1, 1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMDASetUniformCoordinates(dm, user.xmin, user.xmax, user.ymin, user.ymax, 0, 0));

  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMSetApplicationContext(dm, &user));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, dm));
  PetscCall(TSSetMaxTime(ts, 8.0));
  PetscCall(TSSetTimeStep(ts, 0.00001));
  PetscCall(TSSetMaxSteps(ts, 10000));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  if (user.monitor_temp) PetscCall(TSMonitorSet(ts, PlotTemperature, &user, NULL));
  if (user.compute_error) PetscCall(TSMonitorSet(ts, ComputeError, &user, NULL));
  PetscCall(TSSetFromOptions(ts));

  PetscInt Ncell_x, Ncell_y, Ncell_z, Ncells;
  PetscCall(DMDAGetNumCells(dm, &Ncell_x, &Ncell_y, &Ncell_z, &Ncells));

  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));
  SNES snes;
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SetupContext(dm, &user));
  PetscCall(TSSetComputeInitialCondition(ts, InitialConditions));
  PetscCall(SNESSetJacobian(snes, NULL, NULL, SNESComputeJacobianDefault, NULL));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetBlockSize(u, 1));
  PetscCall(VecSetSizes(u, Ncell_x * Ncell_y * Ncell_z, PETSC_DECIDE));
  PetscCall(VecSetUp(u));

  PetscCall(TSSetSolution(ts, u));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(VecViewFromOptions(u, NULL, "-init_vec_view"));
  PetscCall(TSSolve(ts, u));

  PetscCall(VecDestroy(&u));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&dm));
  PetscCall(DestroyContext(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -da_grid_x 56 -da_grid_y 1 -da_grid_z 1 \
          -use_ghost_cells -num_ghost_cells 3,0,0 \
          -xmin 0.0 -xmax 2.0 -ymin -1.0 -ymax 1.0 -zmin -1.0 -zmax 1.0 \
          -ts_time_step 1e-4 -ts_max_steps 1000 -output_step 100 \
          -temp_monitor -compute_error
    test:
      suffix: rk2
      args: -ts_type rk -ts_rk_type 2b
    test:
      suffix: beuler
      args: -ts_type beuler -pc_type none -snes_fd
    test:
      suffix: rkc1
      args: -ts_type rks -ts_rks_type rkc1 -ts_rks_stages 10
    test:
      suffix: rkc2
      args: -ts_type rks -ts_rks_type rkc2 -ts_rks_stages 10
    test:
      suffix: rkl1
      args: -ts_type rks -ts_rks_type rkl1 -ts_rks_stages 10
    test:
      suffix: rkl2
      args: -ts_type rks -ts_rks_type rkl2 -ts_rks_stages 10
    test:
      suffix: rkl1_fct
      args: -ts_type rks -ts_rks_type rkl1 -ts_rks_stages 10 -use_fct_limiter

TEST*/
