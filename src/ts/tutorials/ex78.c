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
#include <petscdmplex.h>
#include <petscdraw.h>
#include <petsc/private/dmpleximpl.h>
#include "petscdm.h"
#include "petscdmda.h"
#include "petscdmlabel.h"
#include "petscmath.h"
#include <petscviewerhdf5.h>
#include <petsc/private/tsimpl.h>
#include <petscdraw.h>

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
  PetscInt    arms_limiter_form;
  PetscDrawLG drawlg;
  PetscBool   monitor_temp;
  PetscBool   use_plex;
  PetscBool   useFCTLimiter;
  PetscBool   useFluxLimiter;
  PetscInt    fluxLimiterType;
  PetscBool   enforce_zero_T;
  PetscBool   printData;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->ostep             = 1.0;
  options->xmin              = 0.0;
  options->xmax              = 2.0;
  options->ymin              = -1.0;
  options->ymax              = 1.0;
  options->zmin              = -1.0;
  options->zmax              = 1.0;
  options->Tmin              = 0.0;
  options->Tmax              = 1.0;
  options->arms_limiter_form = 1;
  options->drawlg            = NULL;
  options->monitor_temp      = PETSC_FALSE;
  options->use_plex          = PETSC_FALSE;
  options->useFluxLimiter    = PETSC_FALSE;
  options->fluxLimiterType   = 0;
  options->useFCTLimiter     = PETSC_FALSE;
  options->enforce_zero_T    = PETSC_FALSE;
  options->printData         = PETSC_FALSE;

  PetscFunctionBeginUser;
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
  PetscCall(PetscOptionsBool("-temp_monitor", "Flag to plot temperature", "", options->monitor_temp, &options->monitor_temp, NULL));
  PetscCall(PetscOptionsBool("-use_plex", "Flag to use DMPLEX instead of DMDA", "", options->use_plex, &options->use_plex, NULL));
  PetscCall(PetscOptionsBool("-use_fct_limiter", "Flag to use ARMS limiter", "", options->useFCTLimiter, &options->useFCTLimiter, NULL));
  PetscCall(PetscOptionsBool("-use_flux_limiter", "Flag to use flux limiter", "", options->useFluxLimiter, &options->useFluxLimiter, NULL));
  PetscCall(PetscOptionsInt("-flux_limiter_type", "Integer controlling which limiter to use", "", options->fluxLimiterType, &options->fluxLimiterType, NULL));
  PetscCall(PetscOptionsBool("-enforce_zero_T", "Flag to stop temperatures from going negative in RHSFunction", "", options->enforce_zero_T, &options->enforce_zero_T, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "", options->ostep, &options->ostep, NULL));
   PetscCall(PetscOptionsBool("-print_data", "Flag to print temperature data in monitor", "", options->printData, &options->printData, NULL));

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupContext(DM dm, AppCtx *user)
{
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  if (user->monitor_temp) {
    PetscDraw draw;
    PetscDrawAxis axis;
    const char *legend[2] = {"Exact", "Numerical"};
    const int colors[2] = {2,1};

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
    PetscCall(PetscDrawAxisSetLimits(axis,user->xmin, user->xmax, -0.1, 1.1));
    PetscCall(PetscDrawAxisSetHoldLimits(axis,PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(PetscDrawLGDestroy(&user->drawlg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "space"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PlotTemperature(TS ts, PetscInt step, PetscReal t, Vec T, void *ctx)
{
  AppCtx            *user = (AppCtx*) ctx;
  const PetscInt     ostep = user->ostep;
  DM                 dm;
  const PetscScalar *t_array;
  PetscInt           i, M;
  char               title[PETSC_MAX_PATH_LEN];
  PetscReal          dz;
  PetscReal          xvals[2], vals[2];
  PetscReal          z, Tnum, Texact, tscale, front, denom;
  PetscDraw          draw;
  PetscDrawAxis      axis;

  PetscFunctionBeginUser;
  if (user->printData && step == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"t\tz\tT_exact\tT_numerical\n"));
  if (step % ostep == 0) {
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(VecGetSize(T, &M));
    PetscCall(VecGetArrayRead(T, &t_array));

    PetscCall(PetscDrawLGReset(user->drawlg));

    dz = (user->xmax - user->xmin) / (PetscReal)M;

    PetscReal err2 = 0.0, errmax = 0.0;
    for (i = 0; i < M; i++) {
      PetscReal z = user->xmin + (i + 0.5) * dz;
      PetscReal Tnum = PetscRealPart(t_array[i]);

      PetscReal Texact;
      PetscReal tscale = PetscPowReal(1.0 + 4.5 * t, 2.0/9.0);
      PetscReal front  = (2.0 / PetscSqrtReal(5.0)) * tscale;
      if (z <= front) {
        PetscReal denom = 0.8 * PetscPowReal(1.0 + 4.5 * t, 4.0/9.0);
        Texact = PetscPowReal(1.0 + 4.5 * t, -2.0/9.0) * PetscPowReal(1.0 - z*z / denom, 0.4);
      } else {
        Texact = 0.0;
      }

      xvals[0] = z;
      xvals[1] = z;

      vals[0] = Texact;
      vals[1] = Tnum;

      if (user->printData) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%f\t%f\t%f\t%f\n",t,z,Texact,Tnum));

      PetscCall(PetscDrawLGAddPoint(user->drawlg, xvals, vals));
      PetscReal diff = Tnum - Texact;
      err2 += diff * diff;
      errmax = PetscMax(errmax, PetscAbsReal(diff));
    }
    err2 = PetscSqrtReal(err2 / M);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "t = %.3e  L2 error = %.6e  Linf error = %.6e\n", (double)t, (double)err2, (double)errmax));

    PetscCall(PetscDrawLGDraw(user->drawlg));

    PetscCall(PetscDrawLGGetDraw(user->drawlg, &draw));
    PetscCall(PetscDrawSave(draw));

    PetscCall(PetscSNPrintf(title, sizeof(title), "Zel'Dovich Temperature Profile t = %e", (double)t));
    PetscCall(PetscDrawSetTitle(draw, title));
    PetscCall(PetscDrawLGGetAxis(user->drawlg, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, title, "z", "T(z)"));

    PetscCall(VecRestoreArrayRead(T, &t_array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal Limiter_Minmod(PetscReal r) {
    if (r <= 0.0) return 0.0;
    return PetscMax(0.0, PetscMin(1.0, r));
}
static PetscReal Limiter_Superbee(PetscReal r) {
  if (r <= 0.0) return 0.0;
  return PetscMax(0.0, PetscMax(PetscMin(1.0, 2.0*r), PetscMin(r, 2.0)));
}

static PetscErrorCode RHSFunction_1d(TS ts, PetscReal time, Vec T, Vec dTdt, void *ctx)
{
  AppCtx            *user = (AppCtx*) ctx;
  DM                 dm;
  PetscInt           i, M;
  const PetscScalar *t;
  PetscScalar       *dtdt;
  PetscReal          dz, k_left, k_right, flux_left, flux_right, *Fleft, *Fright;
  PetscReal          tim1, ti, tip1;
  PetscReal          *limiter_out, *limiter_in;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(VecGetSize(T, &M));
  PetscCall(VecGetArrayRead(T, &t));
  PetscCall(VecGetArray(dTdt, &dtdt));
  PetscCall(PetscMalloc1(M,  &Fleft));
  PetscCall(PetscMalloc1(M, &Fright));
  PetscCall(PetscMalloc1(M,  &limiter_out));
  PetscCall(PetscMalloc1(M, &limiter_in));

  dz = (user->xmax - user->xmin) / M;
  for (i = 0; i < M; ++i) {
    ti = t[i];
    tim1  = (i == 0)  ? t[i] : t[i-1];
    tip1  = (i == M-1) ? t[i] : t[i+1];
    if (user->enforce_zero_T) {
      tim1 = PetscMax(PETSC_MACHINE_EPSILON,tim1);
      ti = PetscMax(PETSC_MACHINE_EPSILON,ti);
      tip1 = PetscMax(PETSC_MACHINE_EPSILON,tip1);
    }

    if (user->useFluxLimiter) {
      PetscReal tim2, tip2;
      tim2  = (i > 1)  ? t[i-2] : t[i];
      if (user->enforce_zero_T) {
        tim2 = PetscMax(PETSC_MACHINE_EPSILON,tim2);
      }

      PetscReal phi_im1, phi_i, phi_ip1;
      PetscReal r_im1, r_i;
      PetscReal f_low_im12, f_high_im12, f_low_ip12, f_high_ip12;

      PetscReal eps = 1e-14;
      r_im1 = (PetscAbsReal(ti - tim1) > eps) ? (tim1 - tim2) / (ti - tim1) : 0.0;
      r_i   = (PetscAbsReal(tip1 - ti) > eps) ? (ti - tim1) / (tip1 - ti) : 0.0;

      switch (user->fluxLimiterType) {
        case 0:
          phi_im1 = 1.;
          phi_i = 1.;
          break;
        case 1:
          phi_im1 = Limiter_Minmod(r_im1);
          phi_i   = Limiter_Minmod(r_i);
          break;
        case 2:
          phi_im1 = Limiter_Superbee(r_im1);
          phi_i   = Limiter_Superbee(r_i);
          break;
        default:
          SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid limiter type %" PetscInt_FMT"", user->fluxLimiterType);
      }

      if (PetscIsInfOrNanReal(flux_left) || PetscIsInfOrNanReal(flux_right)) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "NaN detected at cell %d: T = [%g %g %g], r_i = %g, r_im1 = %g", (int)i, (double)tim1, (double)ti, (double)tip1, (double)r_i, (double)r_im1));
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP, "NaN in Superbee limiter flux");
      }

      f_low_im12 = - PetscPowReal(tim1,2.5) * ((ti-tim1)/dz);
      f_high_im12 = - PetscPowReal(0.5*tim1+0.5*ti,2.5) * ((ti-tim1)/dz);
      f_low_ip12 = - PetscPowReal(ti,2.5) * ((tip1-ti)/dz);
      f_high_ip12 = - PetscPowReal(0.5*ti+0.5*tip1,2.5) * ((tip1-ti)/dz);

      flux_left = f_low_im12 + phi_im1*(f_high_im12-f_low_im12);
      flux_right = f_low_ip12 + phi_i*(f_high_ip12-f_low_ip12);

    } else {
      k_left  = PetscPowReal(PetscMax(PETSC_MACHINE_EPSILON,0.5*tim1+0.5*ti), 2.5);
      k_right = PetscPowReal(PetscMax(PETSC_MACHINE_EPSILON,0.5*ti+0.5*tip1), 2.5);
      flux_left  = -k_left  * (ti - tim1) / dz;
      flux_right = -k_right * (tip1 - ti) / dz;
    }
    Fleft[i] = flux_left;
    Fright[i] = flux_right;
    dtdt[i] = (flux_left - flux_right) / dz;
  }
  PetscFree(Fleft);
  PetscFree(Fright);
  PetscFree(limiter_out);
  PetscFree(limiter_in);
  PetscCall(VecRestoreArrayRead(T, &t));
  PetscCall(VecRestoreArray(dTdt, &dtdt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal SignValue(PetscReal a, PetscReal x) {
  return (x >= 0.0) ? a : -a;
}

static PetscErrorCode RHSFunction_1d_FCT(TS ts, PetscReal time, Vec T, Vec dTdt, void *ctx)
{
  AppCtx            *user = (AppCtx*) ctx;
  DM                 dm;
  PetscInt           i, M, step;
  const PetscScalar *t;
  PetscScalar       *dtdt;
  PetscReal         *flux_im12, *k_im12, *physFlux_im12;
  PetscReal          tim1, ti, tip1, T_left = 1.0, T_right = 0.0;
  PetscReal         *limiter_out, *limiter_in;
  PetscReal          rho = 1.0, cv = 1.0;
  PetscReal          dx, dy, dz, dt;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(VecGetSize(T, &M));
  PetscCall(VecGetArrayRead(T, &t));
  PetscCall(VecGetArray(dTdt, &dtdt));
  PetscCall(PetscMalloc1(M, &k_im12));
  PetscCall(PetscMalloc1(M, &physFlux_im12));
  PetscCall(PetscMalloc1(M, &flux_im12));
  PetscCall(PetscCalloc1(M, &limiter_out));
  PetscCall(PetscCalloc1(M, &limiter_in));
  if (user->use_plex) {
    PetscInt  xs, xm, ys, ym ,zs, zm;
    PetscReal xmin[3],xmax[3];
    Vec coords;
    PetscCall(DMGetBoundingBox(dm, xmin, xmax));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &xs, &xm));
    dx = (xmax[0] - xmin[0]) / (xm-xs);
    dy = (xmax[1] - xmin[1]) / 1.0;
    dz = (xmax[2] - xmin[2]) / 1.0;
  } else {
    PetscInt  xs, xm, ys, ym ,zs, zm;
    PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm));
    dx = (user->xmax - user->xmin) / (xm-xs);
    dy = (user->ymax - user->ymin) / (ym-ys);
    dz = (user->zmax - user->zmin) / (zm-zs);
  }
  for (i = 0; i < M; ++i) {
    ti = t[i];
    tim1  = (i == 0)  ? t[i] : t[i-1];
    if (user->enforce_zero_T) {
      tim1 = PetscMax(PETSC_MACHINE_EPSILON,tim1);
      ti = PetscMax(PETSC_MACHINE_EPSILON,ti);
    }

    k_im12[i]  = PetscPowReal(PetscMax(PETSC_MACHINE_EPSILON,0.5*tim1+0.5*ti), 2.5);
    flux_im12[i]  = k_im12[i]  * (ti - tim1) / dx;
    physFlux_im12[i] = flux_im12[i]*dt*dy*dz;
  }
  for (i = 0; i < M; ++i) {
    PetscReal Tmin_neighbor, Tmax_neighbor, Q_i_minus, Q_i_plus, Q_i;
    PetscReal Ftot, Fnet, Fin, Fout, fracout, fracin, denom;

    ti = t[i];
    tim1  = (i == 0)  ? t[i] : t[i-1];
    tip1  = (i == M-1)   ? t[i]: t[i+1];
    if (user->enforce_zero_T) {
      tim1 = PetscMax(PETSC_MACHINE_EPSILON,tim1);
      ti = PetscMax(PETSC_MACHINE_EPSILON,ti);
      tip1 = PetscMax(PETSC_MACHINE_EPSILON,tip1);
    }

    Tmin_neighbor = PetscMin(PetscMin(tim1,ti),tip1);
    Tmax_neighbor = PetscMax(PetscMax(tim1,ti),tip1);

    Q_i_minus = dx*dy*dz*rho*cv*(ti - Tmin_neighbor);
    Q_i_plus  = dx*dy*dz*rho*cv*(Tmax_neighbor - ti);
    Q_i  = dx*dy*dz*rho*cv*(Tmax_neighbor);

    Ftot = (i == M-1) ? PetscAbs(physFlux_im12[i]) + PetscAbs(physFlux_im12[i]) : PetscAbs(physFlux_im12[i+1]) + PetscAbs(physFlux_im12[i]);
    Fnet = (i == M-1) ? physFlux_im12[i] - physFlux_im12[i] : physFlux_im12[i+1] - physFlux_im12[i];
    Fin = 0.5*(Ftot + Fnet);
    Fout = 0.5*(Ftot - Fnet);

    denom = PetscMax(1e-25,1e-10*Q_i);
    fracout = Q_i_minus/PetscMax(Fout,denom);
    fracin = Q_i_plus/PetscMax(Fin,denom);

    limiter_out[i] = PetscMin(PetscMax(fracout,0.),1.);
    limiter_in[i] = PetscMin(PetscMax(fracin,0.),1.);
  }
  for (i = 0; i < M; ++i) {
    PetscReal limiterIn_im1, limiterOut_im1;
    PetscReal Flux_sign = SignValue(0.5,flux_im12[i]);
    limiterIn_im1  = (i == 0)   ? limiter_in[i] : limiter_in[i-1];
    limiterOut_im1  = (i == 0)   ? limiter_out[i] : limiter_out[i-1];
    flux_im12[i] = ((0.5+Flux_sign)*(PetscMin(limiterIn_im1,limiter_out[i])) + (0.5-Flux_sign)*(PetscMin(limiterOut_im1,limiter_in[i])))*flux_im12[i];
  }
  for (i = 0; i < M; ++i) {
    if (i == M-1) {
      dtdt[i] = (flux_im12[i] - flux_im12[i]) / dx;
    } else {
      dtdt[i] = (flux_im12[i+1] - flux_im12[i]) / dx;
    }
  }
  PetscFree(flux_im12);
  PetscFree(k_im12);
  PetscFree(physFlux_im12);
  PetscFree(limiter_out);
  PetscFree(limiter_in);
  PetscCall(VecRestoreArrayRead(T, &t));
  PetscCall(VecRestoreArray(dTdt, &dtdt));
  PetscCall(VecViewFromOptions(dTdt, NULL, "-sol_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InitialConditions(TS ts, Vec u)
{
  AppCtx      *user;
  DM           dm;
  PetscScalar *u_localptr;
  PetscInt     M;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(VecGetArrayWrite(u, &u_localptr));
  PetscCall(VecGetSize(u, &M));
  u_localptr[0] = 1.0;
  PetscInt xs, xm, ys, ym ,zs, zm;
  if (user->use_plex) {
    PetscCall(DMPlexGetHeightStratum(dm, 0, &xs, &xm));
  } else {
    PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm));
  }

  PetscReal dx = (user->xmax - user->xmin)/M;

  for (PetscInt i = 1; i < M-1; ++i ){
    PetscReal x_cc = user->xmin + (i+0.5)*dx;
    if (x_cc <= (2./PetscSqrtReal(5))) {
      u_localptr[i] = user->Tmax*PetscPowReal((1 - 1.25*x_cc*x_cc),0.4);
    } else {
      u_localptr[i] = user->Tmin;
    }
  }
  PetscCall(VecRestoreArrayWrite(u, &u_localptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm;
  Vec       u;
  TS        ts;
  AppCtx    user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  if (user.use_plex) {
    /*Unstructured DMPLEX Grid*/
    PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  }
  else {
    /*Structured DMDA Grid*/
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 100, 1, 1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &dm));
    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMDASetUniformCoordinates(dm, user.xmin, user.xmax, user.ymin, user.ymax, 0, 0));
  }
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
  PetscCall(TSSetFromOptions(ts));

  PetscInt Ncell_x, Ncell_y, Ncell_z, Ncells;
  if (user.use_plex) {
    PetscInt NcStart, NcEnd;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &NcStart, &NcEnd));
    Ncell_x = NcEnd - NcStart;
  }
  else {
    PetscCall(DMDAGetNumCells(dm, &Ncell_x, &Ncell_y, &Ncell_z, &Ncells));
  }

  if (user.useFCTLimiter) PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction_1d_FCT,&user));
  else PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction_1d,&user));
  SNES snes;
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESSetJacobian(snes, NULL, NULL, SNESComputeJacobianDefault, NULL));

  PetscCall(TSGetSolution(ts, &u));
  PetscCall(SetupContext(dm, &user));

  PetscCall(TSSetComputeInitialCondition(ts, InitialConditions));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetBlockSize(u, 1));
  PetscCall(VecSetSizes(u, Ncell_x, PETSC_DECIDE));
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
    args: -da_grid_x 50 -da_grid_y 1 -da_grid_z 1 \
          -xmin 0.0 -xmax 2.0 -ymin -0.25 -ymax 0.25 -zmin -0.25 -zmax 0.25 \
          -ts_max_steps 80000 -ts_dt 1e-4 -output_step 10000 \
          -pc_type lu -snes_fd -temp_monitor
    test:
      suffix: euler
      args: -ts_type euler
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
      suffix: rkl1_minmod
      args: -ts_type rks -ts_rks_type rkl1 -ts_rks_stages 10 -use_flux_limiter -flux_limiter_type 1
    test:
      suffix: rkl1_superbee
      args: -ts_type rks -ts_rks_type rkl1 -ts_rks_stages 10 -use_flux_limiter -flux_limiter_type 2 -enforce_zero_T
    test:
      suffix: rkl1_fct
      args: -ts_type rks -ts_rks_type rkl1 -ts_rks_stages 10 -use_fct_limiter
    test:
      suffix: rkg1
      args: -ts_type rks -ts_rks_type rkg1 -ts_rks_stages 10 -ts_rks_rkg_parameter_i 0
    test:
      suffix: rkg2
      args: -ts_type rks -ts_rks_type rkg2 -ts_rks_stages 10 -ts_rks_rkg_parameter_i 0

  testset:
    args: -use_plex -dm_plex_dim 3 -dm_plex_box_faces 50,1,1 -dm_plex_simplex 0 \
          -dm_plex_box_lower 0.0,-0.25,-0.25 -dm_plex_box_upper 2.0,0.25,0.25 -dm_plex_box_bd none,none,none \
          -snes_fd -ts_max_steps 800000 -ts_dt 1e-5 -output_step 100000 \
          -temp_monitor -use_fct_limiter -pc_type lu
    test:
      suffix: plex_rkc1
      args: -ts_type rks -ts_rks_type rkc1 -ts_rks_stages 10
TEST*/