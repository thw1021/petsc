static char help[] = "Forced axisymmetric Navier-Stokes flow whose vortex core collapses self-similarly in finite time.\n\
Reproduces the scaling structure of the finite-time blowup for the forced three-dimensional incompressible\n\
Navier-Stokes equations constructed in 'Finite time blowup for Navier-Stokes' (OpenAI, 2026).\n\n";

/*
   Background

   The paper defines the external force f as the residual of a chosen smooth incompressible flow (u, p) in

       u_t + (u . grad) u - nu Lap u + grad p = f,   div u = 0,   u(., 0) = 0,      on R^3 x [0, 1),

   so that the Navier-Stokes equations hold by construction and the whole difficulty is to make the flow blow up
   while f stays smooth through the singular time t = 1. The leading-order flow is an axisymmetric vortex with
   swirl whose core contracts onto the origin. With tau = 1 - t, the core obeys (Sections 2.1 and 3.1 of the paper,
   with a fixed exponent 0 < h < 1/100)

       radial extent        l_r ~ tau^{1/2},                axial extent   l_z ~ tau^{1/2-h},
       azimuthal and axial  |u_theta|, |u_z| ~ tau^{-1/2-h}, radial         |u_r| ~ tau^{-1/2},
       angular Reynolds     Re_theta = |u_theta| l_r / nu ~ tau^{-h} -> infinity,  Re_r = |u_r| l_r / nu = O(1),
       core kinetic energy  ~ tau^{1/2-3h} -> 0,            sup |u| -> infinity,   total kinetic energy bounded.

   Fluid spirals inward toward the axis, spins up, and leaves axially on both sides of a dividing layer near z = 0,
   with a slight upward bias so that u_z does not vanish at z = 0. Outside the core the flow is a slowly varying
   azimuthal vortex whose velocity has a limit as t -> 1 at every fixed positive radius.

   The model flow used here

   This example builds an explicit smooth flow with exactly these scalings and this qualitative structure,

       rho = r / tau^{1/2},   zeta = z / tau^{1/2-h},   chi(t) = smooth ramp from 0 (t <= 0) to 1 (t >= t_ramp),

       psi      = a chi tau^{1/2-h} (rho^2/2) e^{-rho^2} (zeta + beta) e^{-zeta^2}     Stokes stream function,
       u_r      = -psi_z / r,    u_z = psi_r / r,                                        meridional flow,
       u_theta  = a chi [ tau^{-1/2-h} rho e^{-rho^2} e^{-zeta^2} + W(r, t) ],           swirl,
       W(r, t)  = Gamma / (2 pi r) (1 - exp(-r^2 / (4 nu (t + t0)))),                   Lamb-Oseen exterior vortex,

   and solves the forced axisymmetric Navier-Stokes equations with the force equal to the analytic residual of this
   flow, so that the collapsing vortex is the exact solution of the problem being integrated. The equations are
   written in the standard swirl / azimuthal vorticity / stream function form, with Lap = d_rr + (1/r) d_r + d_zz,

       u_theta_t + u_r u_theta_r + u_z u_theta_z + u_r u_theta / r = nu (Lap u_theta - u_theta / r^2) + f_theta,
       omega_t   + u_r omega_r   + u_z omega_z   - u_r omega / r   = (u_theta^2)_z / r + nu (Lap omega - omega / r^2) + (curl f)_theta,
       omega = -(1/r) psi_zz - (1/r) psi_rr + psi_r / r^2,

   on (r, z) in [0, R] x [-Z, Z], discretized with second-order centered differences on a DMDA and integrated in time
   as an index-1 DAE with the fully implicit TS methods (TSBDF by default). The initial state is fluid at rest, the
   axis carries u_theta = omega = psi = 0, and the exact flow supplies Dirichlet data on the outer boundary.

   The monitor prints, each time tau halves, the maximum speed of the computed and exact flows, the compensated
   quantity tau^{1/2+h} max|u| (which tends to a constant for a self-similar collapse), the kinetic energy of both
   flows, the relative error of the swirl, and the size of the applied force. The computed flow follows the exact
   collapse until the core radius l_r ~ tau^{1/2} reaches a few grid cells; refine with -da_refine to follow it longer.

   What this example does and does not show

   It shows the kinematics and the forced dynamics of the leading-order singular flow of the paper: unbounded speed,
   bounded energy, the anisotropic contraction with the paper's exponents, and diverging angular Reynolds number.
   The force used here is the residual of the leading-order background alone. As explained in Section 2.2 of the
   paper this force becomes unbounded as t -> 1 (see the sup|f_theta| column). The paper removes this by adding
   oscillatory pulses on successively finer scales whose averaged Reynolds stresses cancel the singular part of
   the residual, and then further corrections, leaving a force that is smooth and compactly supported through
   t = 1. That part of the construction, as well as the paper's exact self-similar profiles, which solve a nonlinear
   profile problem, are not reproduced here; explicit Gaussian profiles with the same exponents are used instead.

   Options

     -blowup_h h              anisotropy exponent h (paper requires 0 < h < 1/100)
     -viscosity nu            kinematic viscosity
     -amplitude a             amplitude of the core flow
     -axial_bias beta         upward bias of the axial flow at z = 0
     -exterior_circulation G  circulation of the Lamb-Oseen exterior vortex
     -exterior_t0 t0          age of the exterior vortex at t = 0
     -ramp_time t_ramp        time over which the force ramps the flow up from rest
     -domain_r R -domain_z Z  computational domain [0, R] x [-Z, Z]
     -model_scaling           print the scaling table of the exact flow down to tiny tau instead of solving
     -model_scaling_decades n number of decades of tau in the table
     -model_check             verify the analytic derivatives of the model flow against finite differences

   Example runs

     ./ex78 -ts_max_time 0.95 -da_refine 1
     ./ex78 -ts_max_time 0.99 -da_refine 2 -ts_monitor_draw_solution
     ./ex78 -model_scaling -model_scaling_decades 10
*/

#include <petscts.h>
#include <petscdmda.h>

#define POLY_MAX 12

/* P(x) exp(-x^2) with P a polynomial */
typedef struct {
  PetscInt  deg;
  PetscReal c[POLY_MAX];
} GaussPoly;

/* coef tau^p P(rho) exp(-rho^2) Q(zeta) exp(-zeta^2), with P and Q and their first two derivatives */
typedef struct {
  PetscReal coef, p;
  GaussPoly P[3], Q[3];
} SimTerm;

/* value and the derivatives of a field needed by the momentum residual */
typedef struct {
  PetscReal val, r, z, rr, zz, t;
} Derivs;

/* the exact model flow and the force that sustains it at one point */
typedef struct {
  PetscReal uth, om, psi, ur, uz, fth, gth;
} Model;

typedef struct {
  PetscReal h, A, D, nu, amp, beta, circ, t0, tramp, R, Z;
  SimTerm   psi, uth, om[2];
  Vec       forcing;
  PetscReal tforcing;
  PetscBool forcing_valid;
  PetscReal tau_print, tlast;
} AppCtx;

/* d/dx of P(x) exp(-x^2) is (P'(x) - 2 x P(x)) exp(-x^2) */
static PetscErrorCode GaussPolyDeriv(const GaussPoly *in, GaussPoly *out)
{
  PetscInt k;

  PetscFunctionBeginUser;
  PetscCheck(in->deg + 1 < POLY_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Polynomial degree %" PetscInt_FMT " too large", in->deg + 1);
  out->deg = in->deg + 1;
  for (k = 0; k <= out->deg; k++) {
    out->c[k] = 0.0;
    if (k + 1 <= in->deg) out->c[k] += (k + 1) * in->c[k + 1];
    if (k >= 1) out->c[k] -= 2.0 * in->c[k - 1];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal GaussPolyEval(const GaussPoly *P, PetscReal x)
{
  PetscReal s = 0.0;
  PetscInt  k;

  for (k = P->deg; k >= 0; k--) s = s * x + P->c[k];
  return s * PetscExpReal(-x * x);
}

static PetscErrorCode SimTermSetUp(SimTerm *T, PetscReal coef, PetscReal p, PetscInt degP, const PetscReal cP[], PetscInt degQ, const PetscReal cQ[])
{
  PetscInt k;

  PetscFunctionBeginUser;
  T->coef     = coef;
  T->p        = p;
  T->P[0].deg = degP;
  T->Q[0].deg = degQ;
  for (k = 0; k <= degP; k++) T->P[0].c[k] = cP[k];
  for (k = 0; k <= degQ; k++) T->Q[0].c[k] = cQ[k];
  for (k = 1; k < 3; k++) {
    PetscCall(GaussPolyDeriv(&T->P[k - 1], &T->P[k]));
    PetscCall(GaussPolyDeriv(&T->Q[k - 1], &T->Q[k]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* rho = r tau^{-1/2} and zeta = z tau^{-D} so that d rho/dt = rho/(2 tau) and d zeta/dt = D zeta/tau */
static void SimTermEval(const SimTerm *T, PetscReal rho, PetscReal zeta, PetscReal tau, PetscReal D, Derivs *f)
{
  const PetscReal base = T->coef * PetscPowReal(tau, T->p), lr = PetscSqrtReal(tau), lz = PetscPowReal(tau, D);
  const PetscReal P0 = GaussPolyEval(&T->P[0], rho), P1 = GaussPolyEval(&T->P[1], rho), P2 = GaussPolyEval(&T->P[2], rho);
  const PetscReal Q0 = GaussPolyEval(&T->Q[0], zeta), Q1 = GaussPolyEval(&T->Q[1], zeta), Q2 = GaussPolyEval(&T->Q[2], zeta);

  f->val = base * P0 * Q0;
  f->r   = base * P1 * Q0 / lr;
  f->z   = base * P0 * Q1 / lz;
  f->rr  = base * P2 * Q0 / (lr * lr);
  f->zz  = base * P0 * Q2 / (lz * lz);
  f->t   = base / tau * (-T->p * P0 * Q0 + 0.5 * rho * P1 * Q0 + D * zeta * P0 * Q1);
}

static void DerivsAdd(Derivs *a, const Derivs *b)
{
  a->val += b->val;
  a->r += b->r;
  a->z += b->z;
  a->rr += b->rr;
  a->zz += b->zz;
  a->t += b->t;
}

/* multiply a field by the ramp chi(t) */
static void DerivsRamp(Derivs *a, PetscReal chi, PetscReal chip)
{
  a->t = chip * a->val + chi * a->t;
  a->val *= chi;
  a->r *= chi;
  a->z *= chi;
  a->rr *= chi;
  a->zz *= chi;
}

/* C-infinity ramp from 0 at t <= 0 to 1 at t >= tramp */
static void Ramp(PetscReal t, PetscReal tramp, PetscReal *chi, PetscReal *chip)
{
  const PetscReal s = t / tramp;
  PetscReal       a, b;

  if (s <= 0.0) {
    *chi  = 0.0;
    *chip = 0.0;
    return;
  }
  if (s >= 1.0) {
    *chi  = 1.0;
    *chip = 0.0;
    return;
  }
  a     = PetscExpReal(-1.0 / s);
  b     = PetscExpReal(-1.0 / (1.0 - s));
  *chi  = a / (a + b);
  *chip = a * b * (1.0 / (s * s) + 1.0 / ((1.0 - s) * (1.0 - s))) / ((a + b) * (a + b)) / tramp;
}

/* Lamb-Oseen vortex W(r,t) = Gamma/(2 pi r) (1 - exp(-r^2/(4 nu (t + t0)))), an exact solution of the azimuthal heat equation */
static void Exterior(const AppCtx *ctx, PetscReal r, PetscReal t, Derivs *w)
{
  const PetscReal c = ctx->circ / (2.0 * PETSC_PI), T = t + ctx->t0, L2 = 4.0 * ctx->nu * T, s = r * r / L2;
  const PetscReal e = PetscExpReal(-s), m = -expm1(-s);

  w->val = c * m / r;
  w->r   = c * (2.0 * e / L2 - m / (r * r));
  w->z   = 0.0;
  w->rr  = c * (-4.0 * r * e / (L2 * L2) - 2.0 * e / (L2 * r) + 2.0 * m / (r * r * r));
  w->zz  = 0.0;
  w->t   = -c * s * e / (r * T);
}

/* the exact model flow and its momentum residual, which is the applied force, at (r, z, t); tau = 1 - t is passed separately for accuracy */
static PetscErrorCode ModelEval(const AppCtx *ctx, PetscReal r, PetscReal z, PetscReal t, PetscReal tau, Model *m)
{
  const PetscReal nu = ctx->nu, rho = r / PetscSqrtReal(tau), zeta = z / PetscPowReal(tau, ctx->D);
  PetscReal       chi, chip;
  Derivs          psi, uth, om, tmp;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(m, sizeof(*m)));
  if (r <= 0.0) PetscFunctionReturn(PETSC_SUCCESS);
  Ramp(t, ctx->tramp, &chi, &chip);
  SimTermEval(&ctx->psi, rho, zeta, tau, ctx->D, &psi);
  SimTermEval(&ctx->uth, rho, zeta, tau, ctx->D, &uth);
  Exterior(ctx, r, t, &tmp);
  DerivsAdd(&uth, &tmp);
  SimTermEval(&ctx->om[0], rho, zeta, tau, ctx->D, &om);
  SimTermEval(&ctx->om[1], rho, zeta, tau, ctx->D, &tmp);
  DerivsAdd(&om, &tmp);
  DerivsRamp(&psi, chi, chip);
  DerivsRamp(&uth, chi, chip);
  DerivsRamp(&om, chi, chip);
  m->psi = psi.val;
  m->uth = uth.val;
  m->om  = om.val;
  m->ur  = -psi.z / r;
  m->uz  = psi.r / r;
  m->fth = uth.t + m->ur * uth.r + m->uz * uth.z + m->ur * uth.val / r - nu * (uth.rr + uth.r / r + uth.zz - uth.val / (r * r));
  m->gth = om.t + m->ur * om.r + m->uz * om.z - m->ur * om.val / r - 2.0 * uth.val * uth.z / r - nu * (om.rr + om.r / r + om.zz - om.val / (r * r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Sets up the separable terms of the model flow. With S(rho) = rho^2/2 exp(-rho^2) and G(zeta) = (zeta + beta) exp(-zeta^2)
   the stream function is psi = a tau^D S G and the azimuthal vorticity is

       omega = -(1/r) psi_zz - (1/r) (psi_rr - psi_r/r) = -a tau^{-1+h} (S/rho) G'' - a tau^{-1-h} ((S'' - S'/rho)/rho) G
*/
static PetscErrorCode ModelSetUp(AppCtx *ctx)
{
  const PetscReal cS[3] = {0.0, 0.0, 0.5}, cG[2] = {ctx->beta, 1.0}, cE[2] = {0.0, 1.0}, cH[1] = {1.0}, cS1[2] = {0.0, 0.5}, cS2[4] = {0.0, -4.0, 0.0, 2.0};
  GaussPoly       G, G1, G2;

  PetscFunctionBeginUser;
  PetscCheck(ctx->h >= 0.0 && ctx->h < 0.25, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "The exponent h = %g must lie in [0, 1/4); the paper uses 0 < h < 1/100", (double)ctx->h);
  ctx->A = 0.5 + ctx->h;
  ctx->D = 0.5 - ctx->h;
  G.deg  = 1;
  G.c[0] = cG[0];
  G.c[1] = cG[1];
  PetscCall(GaussPolyDeriv(&G, &G1));
  PetscCall(GaussPolyDeriv(&G1, &G2));
  PetscCall(SimTermSetUp(&ctx->psi, ctx->amp, ctx->D, 2, cS, 1, cG));
  PetscCall(SimTermSetUp(&ctx->uth, ctx->amp, -ctx->A, 1, cE, 0, cH));
  PetscCall(SimTermSetUp(&ctx->om[0], -ctx->amp, -1.0 + ctx->h, 1, cS1, G2.deg, G2.c));
  PetscCall(SimTermSetUp(&ctx->om[1], -ctx->amp, -1.0 - ctx->h, 3, cS2, 1, cG));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* evaluates the force and the Dirichlet data at the current time once per stage and caches them */
static PetscErrorCode ComputeForcing(AppCtx *ctx, DM da, PetscReal t)
{
  PetscInt       i, j, xs, ys, xm, ym, Mx, My;
  PetscReal      dr, dz;
  PetscScalar ***g;
  Model          m;

  PetscFunctionBeginUser;
  if (ctx->forcing_valid && t == ctx->tforcing) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMDAGetInfo(da, NULL, &Mx, &My, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  dr = ctx->R / (Mx - 1);
  dz = 2.0 * ctx->Z / (My - 1);
  PetscCall(DMDAVecGetArrayDOF(da, ctx->forcing, &g));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscCall(ModelEval(ctx, i * dr, -ctx->Z + j * dz, t, 1.0 - t, &m));
      if (i == 0) g[j][i][0] = g[j][i][1] = g[j][i][2] = 0.0;
      else if (i == Mx - 1 || j == 0 || j == My - 1) {
        g[j][i][0] = m.uth;
        g[j][i][1] = m.om;
        g[j][i][2] = m.psi;
      } else {
        g[j][i][0] = m.fth;
        g[j][i][1] = m.gth;
        g[j][i][2] = 0.0;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da, ctx->forcing, &g));
  ctx->tforcing      = t;
  ctx->forcing_valid = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* F(t, U, U_t) = 0 for U = (u_theta, omega, psi); the psi rows and the boundary rows are algebraic */
static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, PetscCtx actx)
{
  AppCtx              *ctx = (AppCtx *)actx;
  DM                   da;
  Vec                  Uloc;
  PetscInt             i, j, k, xs, ys, xm, ym, Mx, My;
  PetscReal            dr, dz, r, ur, uz;
  PetscReal            c[3], e[3], w[3], n[3], s[3], dR[3], dZ[3], dRR[3], dZZ[3];
  const PetscReal      nu = ctx->nu;
  const PetscScalar ***u, ***udot, ***g;
  PetscScalar       ***f;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(ComputeForcing(ctx, da, t));
  PetscCall(DMDAGetInfo(da, NULL, &Mx, &My, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  dr = ctx->R / (Mx - 1);
  dz = 2.0 * ctx->Z / (My - 1);
  PetscCall(DMGetLocalVector(da, &Uloc));
  PetscCall(DMGlobalToLocal(da, U, INSERT_VALUES, Uloc));
  PetscCall(DMDAVecGetArrayDOFRead(da, Uloc, &u));
  PetscCall(DMDAVecGetArrayDOFRead(da, Udot, &udot));
  PetscCall(DMDAVecGetArrayDOFRead(da, ctx->forcing, &g));
  PetscCall(DMDAVecGetArrayDOF(da, F, &f));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      if (i == 0) {
        for (k = 0; k < 3; k++) f[j][i][k] = u[j][i][k];
      } else if (i == Mx - 1 || j == 0 || j == My - 1) {
        for (k = 0; k < 3; k++) f[j][i][k] = u[j][i][k] - g[j][i][k];
      } else {
        r = i * dr;
        for (k = 0; k < 3; k++) {
          c[k]   = PetscRealPart(u[j][i][k]);
          e[k]   = PetscRealPart(u[j][i + 1][k]);
          w[k]   = PetscRealPart(u[j][i - 1][k]);
          n[k]   = PetscRealPart(u[j + 1][i][k]);
          s[k]   = PetscRealPart(u[j - 1][i][k]);
          dR[k]  = (e[k] - w[k]) / (2.0 * dr);
          dZ[k]  = (n[k] - s[k]) / (2.0 * dz);
          dRR[k] = (e[k] - 2.0 * c[k] + w[k]) / (dr * dr);
          dZZ[k] = (n[k] - 2.0 * c[k] + s[k]) / (dz * dz);
        }
        ur         = -dZ[2] / r;
        uz         = dR[2] / r;
        f[j][i][0] = udot[j][i][0] + ur * dR[0] + uz * dZ[0] + ur * c[0] / r - nu * (dRR[0] + dR[0] / r + dZZ[0] - c[0] / (r * r)) - g[j][i][0];
        f[j][i][1] = udot[j][i][1] + ur * dR[1] + uz * dZ[1] - ur * c[1] / r - 2.0 * c[0] * dZ[0] / r - nu * (dRR[1] + dR[1] / r + dZZ[1] - c[1] / (r * r)) - g[j][i][1];
        f[j][i][2] = r * c[1] + dZZ[2] + dRR[2] - dR[2] / r;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da, F, &f));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, ctx->forcing, &g));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, Udot, &udot));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, Uloc, &u));
  PetscCall(DMRestoreLocalVector(da, &Uloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* compares the computed flow with the exact collapsing vortex and prints one line of the scaling table */
static PetscErrorCode Diagnostics(TS ts, PetscReal t, Vec U, AppCtx *ctx, PetscBool header)
{
  DM                   da;
  Vec                  Uloc;
  PetscInt             i, j, xs, ys, xm, ym, Mx, My;
  PetscReal            dr, dz, r, z, ur, uz, uth, u2, tau = 1.0 - t;
  PetscReal            lmax[5] = {0, 0, 0, 0, 0}, gmax[5], lsum[2] = {0, 0}, gsum[2];
  const PetscScalar ***u;
  Model                m;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMDAGetInfo(da, NULL, &Mx, &My, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  dr = ctx->R / (Mx - 1);
  dz = 2.0 * ctx->Z / (My - 1);
  PetscCall(DMGetLocalVector(da, &Uloc));
  PetscCall(DMGlobalToLocal(da, U, INSERT_VALUES, Uloc));
  PetscCall(DMDAVecGetArrayDOFRead(da, Uloc, &u));
  for (j = PetscMax(ys, 1); j < PetscMin(ys + ym, My - 1); j++) {
    for (i = PetscMax(xs, 1); i < PetscMin(xs + xm, Mx - 1); i++) {
      r   = i * dr;
      z   = -ctx->Z + j * dz;
      uth = PetscRealPart(u[j][i][0]);
      ur  = -PetscRealPart(u[j + 1][i][2] - u[j - 1][i][2]) / (2.0 * dz * r);
      uz  = PetscRealPart(u[j][i + 1][2] - u[j][i - 1][2]) / (2.0 * dr * r);
      u2  = ur * ur + uth * uth + uz * uz;
      PetscCall(ModelEval(ctx, r, z, t, tau, &m));
      lmax[0] = PetscMax(lmax[0], PetscSqrtReal(u2));
      lmax[1] = PetscMax(lmax[1], PetscSqrtReal(m.ur * m.ur + m.uth * m.uth + m.uz * m.uz));
      lmax[2] = PetscMax(lmax[2], PetscAbsReal(uth - m.uth));
      lmax[3] = PetscMax(lmax[3], PetscAbsReal(m.uth));
      lmax[4] = PetscMax(lmax[4], PetscAbsReal(m.fth));
      lsum[0] += PETSC_PI * u2 * r * dr * dz;
      lsum[1] += PETSC_PI * (m.ur * m.ur + m.uth * m.uth + m.uz * m.uz) * r * dr * dz;
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFRead(da, Uloc, &u));
  PetscCall(DMRestoreLocalVector(da, &Uloc));
  PetscCallMPI(MPIU_Allreduce(lmax, gmax, 5, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)ts)));
  PetscCallMPI(MPIU_Allreduce(lsum, gsum, 2, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)ts)));
  if (header) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "Forced axisymmetric Navier-Stokes: h = %g, nu = %g, grid %" PetscInt_FMT " x %" PetscInt_FMT ", dr = %.3e\n", (double)ctx->h, (double)ctx->nu, Mx, My, (double)dr));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "     t        tau      max|u|   exact max|u| tau^A max|u|    KE     exact KE   rel err u_theta  sup|f_theta|\n"));
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "%9.6f %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e\n", (double)t, (double)tau, (double)gmax[0], (double)gmax[1], (double)(PetscPowReal(tau, ctx->A) * gmax[0]), (double)gsum[0], (double)gsum[1], (double)(gmax[3] > 0 ? gmax[2] / gmax[3] : 0.0), (double)gmax[4]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* prints the diagnostics each time tau = 1 - t has halved */
static PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal t, Vec U, PetscCtx actx)
{
  AppCtx         *ctx = (AppCtx *)actx;
  const PetscReal tau = 1.0 - t;

  PetscFunctionBeginUser;
  if (tau > ctx->tau_print) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(Diagnostics(ts, t, U, ctx, step == 0 ? PETSC_TRUE : PETSC_FALSE));
  while (ctx->tau_print >= tau) ctx->tau_print *= 0.5;
  ctx->tlast = t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* midpoint rule on a graded (r, z) mesh that resolves both the core, of size l_r x l_z, and the exterior */
static PetscErrorCode ModelKineticEnergy(const AppCtx *ctx, PetscReal t, PetscReal tau, PetscReal *KE)
{
  const PetscInt  n  = 256;
  const PetscReal lr = PetscSqrtReal(tau), lz = PetscPowReal(tau, ctx->D), r1 = PetscMin(8.0 * lr, ctx->R), z1 = PetscMin(8.0 * lz, ctx->Z);
  PetscReal      *rq, *wr, *zq, *wz, sum = 0.0;
  PetscInt        nr = 0, nz = 0, i, j;
  Model           m;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc4(2 * n, &rq, 2 * n, &wr, 3 * n, &zq, 3 * n, &wz));
  for (i = 0; i < n; i++, nr++) {
    rq[nr] = (i + 0.5) * r1 / n;
    wr[nr] = r1 / n;
  }
  if (r1 < ctx->R) {
    const PetscReal dl = PetscLogReal(ctx->R / r1) / n;
    for (i = 0; i < n; i++, nr++) {
      rq[nr] = r1 * PetscExpReal((i + 0.5) * dl);
      wr[nr] = rq[nr] * dl;
    }
  }
  if (z1 < ctx->Z) {
    const PetscReal dl = PetscLogReal(ctx->Z / z1) / n;
    for (j = 0; j < n; j++, nz++) {
      zq[nz] = -z1 * PetscExpReal((j + 0.5) * dl);
      wz[nz] = -zq[nz] * dl;
    }
  }
  for (j = 0; j < n; j++, nz++) {
    zq[nz] = -z1 + (j + 0.5) * 2.0 * z1 / n;
    wz[nz] = 2.0 * z1 / n;
  }
  if (z1 < ctx->Z) {
    const PetscReal dl = PetscLogReal(ctx->Z / z1) / n;
    for (j = 0; j < n; j++, nz++) {
      zq[nz] = z1 * PetscExpReal((j + 0.5) * dl);
      wz[nz] = zq[nz] * dl;
    }
  }
  for (j = 0; j < nz; j++) {
    for (i = 0; i < nr; i++) {
      PetscCall(ModelEval(ctx, rq[i], zq[j], t, tau, &m));
      sum += PETSC_PI * (m.ur * m.ur + m.uth * m.uth + m.uz * m.uz) * rq[i] * wr[i] * wz[j];
    }
  }
  PetscCall(PetscFree4(rq, wr, zq, wz));
  *KE = sum;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* the scaling laws of the exact flow, evaluated down to values of tau far below what any grid can resolve */
static PetscErrorCode ModelScalingTable(const AppCtx *ctx, PetscInt decades)
{
  const PetscInt nr = 121, nz = 241;
  PetscInt       k, i, j;
  PetscReal      tau, t, lr, lz, r, z, umax, uthmax, fmax, gmax, KE;
  Model          m;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Exact model flow: h = %g, nu = %g, l_r = tau^{1/2}, l_z = tau^{1/2-h}, A = 1/2 + h = %g\n", (double)ctx->h, (double)ctx->nu, (double)ctx->A));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "    tau       l_r       l_z     l_z/l_r   max|u|   tau^A max|u|  Re_theta     KE     sup|f_theta| sup|curl f|\n"));
  for (k = 1; k <= decades; k++) {
    tau  = PetscPowReal(10.0, -(PetscReal)k);
    t    = 1.0 - tau;
    lr   = PetscSqrtReal(tau);
    lz   = PetscPowReal(tau, ctx->D);
    umax = uthmax = fmax = gmax = 0.0;
    /* sample the core in similarity coordinates and the whole domain in physical coordinates */
    for (j = 0; j < nz; j++) {
      for (i = 1; i < nr; i++) {
        r = 6.0 * lr * i / (nr - 1);
        z = 6.0 * lz * (2.0 * j / (nz - 1) - 1.0);
        PetscCall(ModelEval(ctx, r, z, t, tau, &m));
        umax   = PetscMax(umax, PetscSqrtReal(m.ur * m.ur + m.uth * m.uth + m.uz * m.uz));
        uthmax = PetscMax(uthmax, PetscAbsReal(m.uth));
        fmax   = PetscMax(fmax, PetscAbsReal(m.fth));
        gmax   = PetscMax(gmax, PetscAbsReal(m.gth));
        r      = ctx->R * i / (nr - 1);
        z      = ctx->Z * (2.0 * j / (nz - 1) - 1.0);
        PetscCall(ModelEval(ctx, r, z, t, tau, &m));
        umax   = PetscMax(umax, PetscSqrtReal(m.ur * m.ur + m.uth * m.uth + m.uz * m.uz));
        uthmax = PetscMax(uthmax, PetscAbsReal(m.uth));
        fmax   = PetscMax(fmax, PetscAbsReal(m.fth));
        gmax   = PetscMax(gmax, PetscAbsReal(m.gth));
      }
    }
    PetscCall(ModelKineticEnergy(ctx, t, tau, &KE));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e\n", (double)tau, (double)lr, (double)lz, (double)(lz / lr), (double)umax, (double)(PetscPowReal(tau, ctx->A) * umax), (double)(uthmax * lr / ctx->nu), (double)KE, (double)fmax, (double)gmax));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* fourth-order centered finite differences of a component of the exact flow, used only to verify the analytic derivatives */
static PetscErrorCode FieldAt(const AppCtx *ctx, PetscInt which, PetscReal r, PetscReal z, PetscReal t, PetscReal *v)
{
  Model m;

  PetscFunctionBeginUser;
  PetscCall(ModelEval(ctx, r, z, t, 1.0 - t, &m));
  *v = which == 0 ? m.uth : (which == 1 ? m.om : m.psi);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FieldDerivsFD(const AppCtx *ctx, PetscInt which, PetscReal r, PetscReal z, PetscReal t, Derivs *d)
{
  const PetscReal dr = 1e-3, dz = 1e-3, dt = 1e-4;
  PetscReal       vp, vm, vpp, vmm;

  PetscFunctionBeginUser;
  PetscCall(FieldAt(ctx, which, r, z, t, &d->val));
  PetscCall(FieldAt(ctx, which, r + dr, z, t, &vp));
  PetscCall(FieldAt(ctx, which, r - dr, z, t, &vm));
  PetscCall(FieldAt(ctx, which, r + 2 * dr, z, t, &vpp));
  PetscCall(FieldAt(ctx, which, r - 2 * dr, z, t, &vmm));
  d->r  = (-vpp + 8 * vp - 8 * vm + vmm) / (12 * dr);
  d->rr = (-vpp + 16 * vp - 30 * d->val + 16 * vm - vmm) / (12 * dr * dr);
  PetscCall(FieldAt(ctx, which, r, z + dz, t, &vp));
  PetscCall(FieldAt(ctx, which, r, z - dz, t, &vm));
  PetscCall(FieldAt(ctx, which, r, z + 2 * dz, t, &vpp));
  PetscCall(FieldAt(ctx, which, r, z - 2 * dz, t, &vmm));
  d->z  = (-vpp + 8 * vp - 8 * vm + vmm) / (12 * dz);
  d->zz = (-vpp + 16 * vp - 30 * d->val + 16 * vm - vmm) / (12 * dz * dz);
  PetscCall(FieldAt(ctx, which, r, z, t + dt, &vp));
  PetscCall(FieldAt(ctx, which, r, z, t - dt, &vm));
  PetscCall(FieldAt(ctx, which, r, z, t + 2 * dt, &vpp));
  PetscCall(FieldAt(ctx, which, r, z, t - 2 * dt, &vmm));
  d->t = (-vpp + 8 * vp - 8 * vm + vmm) / (12 * dt);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ModelCheck(const AppCtx *ctx)
{
  const PetscReal pts[3][3] = {
    {0.3, 0.2,  0.6},
    {0.7, -0.5, 0.4},
    {1.2, 0.9,  0.8}
  };
  const PetscReal nu     = ctx->nu;
  PetscReal       err[4] = {0, 0, 0, 0}, r, z, t, om, ur, uz, fth, gth;
  PetscInt        k;
  Derivs          psi, uth, omd;
  Model           m;

  PetscFunctionBeginUser;
  for (k = 0; k < 3; k++) {
    r = pts[k][0];
    z = pts[k][1];
    t = pts[k][2];
    PetscCall(ModelEval(ctx, r, z, t, 1.0 - t, &m));
    PetscCall(FieldDerivsFD(ctx, 2, r, z, t, &psi));
    PetscCall(FieldDerivsFD(ctx, 0, r, z, t, &uth));
    PetscCall(FieldDerivsFD(ctx, 1, r, z, t, &omd));
    om     = -psi.zz / r - psi.rr / r + psi.r / (r * r);
    ur     = -psi.z / r;
    uz     = psi.r / r;
    fth    = uth.t + ur * uth.r + uz * uth.z + ur * uth.val / r - nu * (uth.rr + uth.r / r + uth.zz - uth.val / (r * r));
    gth    = omd.t + ur * omd.r + uz * omd.z - ur * omd.val / r - 2.0 * uth.val * uth.z / r - nu * (omd.rr + omd.r / r + omd.zz - omd.val / (r * r));
    err[0] = PetscMax(err[0], PetscAbsReal(om - m.om) / PetscAbsReal(m.om));
    err[1] = PetscMax(err[1], (PetscAbsReal(ur - m.ur) + PetscAbsReal(uz - m.uz)) / (PetscAbsReal(m.ur) + PetscAbsReal(m.uz)));
    err[2] = PetscMax(err[2], PetscAbsReal(fth - m.fth) / PetscAbsReal(m.fth));
    err[3] = PetscMax(err[3], PetscAbsReal(gth - m.gth) / PetscAbsReal(m.gth));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Model check, relative discrepancy between analytic and finite-difference derivatives:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  omega from psi %.1e, (u_r, u_z) from psi %.1e, f_theta %.1e, (curl f)_theta %.1e\n", (double)err[0], (double)err[1], (double)err[2], (double)err[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx      ctx;
  DM          da;
  TS          ts;
  SNES        snes;
  KSP         ksp;
  PC          pc;
  Vec         U;
  PetscMPIInt size;
  PetscInt    decades = 8, Mx;
  PetscReal   t, dr, lr;
  PetscBool   scaling = PETSC_FALSE, check = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  ctx.h             = 0.005;
  ctx.nu            = 1.0;
  ctx.amp           = 1.0;
  ctx.beta          = 0.1;
  ctx.circ          = 1.0;
  ctx.t0            = 1.0;
  ctx.tramp         = 0.25;
  ctx.R             = 3.0;
  ctx.Z             = 3.0;
  ctx.forcing       = NULL;
  ctx.tforcing      = 0.0;
  ctx.forcing_valid = PETSC_FALSE;
  ctx.tau_print     = 1.0;
  ctx.tlast         = -1.0;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Navier-Stokes finite-time blowup demonstration", "TS");
  PetscCall(PetscOptionsReal("-blowup_h", "Anisotropy exponent h; the paper requires 0 < h < 1/100", NULL, ctx.h, &ctx.h, NULL));
  PetscCall(PetscOptionsReal("-viscosity", "Kinematic viscosity", NULL, ctx.nu, &ctx.nu, NULL));
  PetscCall(PetscOptionsReal("-amplitude", "Amplitude of the collapsing core flow", NULL, ctx.amp, &ctx.amp, NULL));
  PetscCall(PetscOptionsReal("-axial_bias", "Upward bias of the axial flow at z = 0", NULL, ctx.beta, &ctx.beta, NULL));
  PetscCall(PetscOptionsReal("-exterior_circulation", "Circulation of the Lamb-Oseen exterior vortex", NULL, ctx.circ, &ctx.circ, NULL));
  PetscCall(PetscOptionsReal("-exterior_t0", "Age of the exterior vortex at t = 0", NULL, ctx.t0, &ctx.t0, NULL));
  PetscCall(PetscOptionsReal("-ramp_time", "Time over which the flow is ramped up from rest", NULL, ctx.tramp, &ctx.tramp, NULL));
  PetscCall(PetscOptionsReal("-domain_r", "Radial extent R of the domain [0, R] x [-Z, Z]", NULL, ctx.R, &ctx.R, NULL));
  PetscCall(PetscOptionsReal("-domain_z", "Axial extent Z of the domain [0, R] x [-Z, Z]", NULL, ctx.Z, &ctx.Z, NULL));
  PetscCall(PetscOptionsBool("-model_scaling", "Print the scaling table of the exact flow instead of solving", NULL, scaling, &scaling, NULL));
  PetscCall(PetscOptionsInt("-model_scaling_decades", "Number of decades of tau in the scaling table", NULL, decades, &decades, NULL));
  PetscCall(PetscOptionsBool("-model_check", "Verify the analytic derivatives of the model flow", NULL, check, &check, NULL));
  PetscOptionsEnd();
  PetscCall(ModelSetUp(&ctx));

  if (check) PetscCall(ModelCheck(&ctx));
  if (scaling) PetscCall(ModelScalingTable(&ctx, decades));
  else {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 65, 129, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, &da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDASetFieldName(da, 0, "u_theta"));
    PetscCall(DMDASetFieldName(da, 1, "omega_theta"));
    PetscCall(DMDASetFieldName(da, 2, "psi"));
    PetscCall(DMDASetUniformCoordinates(da, 0.0, ctx.R, -ctx.Z, ctx.Z, 0.0, 0.0));
    PetscCall(DMCreateGlobalVector(da, &U));
    PetscCall(DMCreateGlobalVector(da, &ctx.forcing));
    PetscCall(VecZeroEntries(U)); /* the fluid starts from rest */

    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetDM(ts, da));
    PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
    PetscCall(TSSetEquationType(ts, TS_EQ_DAE_IMPLICIT_INDEX1));
    PetscCall(TSSetIFunction(ts, NULL, IFunction, &ctx));
    PetscCall(TSSetIJacobian(ts, NULL, NULL, TSComputeIJacobianDefaultColor, NULL));
    PetscCall(TSSetType(ts, TSBDF));
    PetscCall(TSSetMaxTime(ts, 0.9));
    PetscCall(TSSetTimeStep(ts, 0.01));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSMonitorSet(ts, Monitor, &ctx, NULL));
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    if (size == 1) PetscCall(PCSetType(pc, PCLU));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSSolve(ts, U));

    PetscCall(TSGetTime(ts, &t));
    if (t != ctx.tlast) PetscCall(Diagnostics(ts, t, U, &ctx, PETSC_FALSE));
    PetscCall(DMDAGetInfo(da, NULL, &Mx, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    dr = ctx.R / (Mx - 1);
    lr = PetscSqrtReal(1.0 - t);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final core radius l_r = %.3e covers %.1f radial grid cells; refine with -da_refine to follow the collapse further\n", (double)lr, (double)(lr / dr)));

    PetscCall(TSDestroy(&ts));
    PetscCall(VecDestroy(&U));
    PetscCall(VecDestroy(&ctx.forcing));
    PetscCall(DMDestroy(&da));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: !complex !single

   test:
     suffix: model
     args: -model_check -model_scaling -model_scaling_decades 6

   test:
     suffix: solve
     args: -da_grid_x 17 -da_grid_y 33 -ts_max_time 0.5 -ts_time_step 0.05 -ts_adapt_type none

   test:
     suffix: solve_parallel
     nsize: 2
     args: -da_grid_x 17 -da_grid_y 33 -ts_max_time 0.5 -ts_time_step 0.05 -ts_adapt_type none -pc_type redundant
     output_file: output/ex78_solve.out

TEST*/
