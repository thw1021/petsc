static char help[] = "Particle basis Landau example using nonlinear solve + Implicit Midpoint-like time stepping.";

/*
  References:
    [1] https://arxiv.org/abs/1910.03080v2
    [2] https://arxiv.org/pdf/2012.07187.pdf
    [3] https://www.sciencedirect.com/science/article/pii/S002199912100615X?via%3Dihub
*/

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>
#include <petscviewer.h>
#include <petscmath.h>
#include <petsclandau.h>
#include "ex27.h"

/* Some useful constants */
#define BOLTZMANN_K 1.380649e-23 /* J/K */
#define KEV_J 6.241506479963235e15 /*  */
#define LIGHT_C 299792458
#define EPSILON_NOUGHT 8.8542e-12
#define ELEMENTARY_CHARGE 1.602176e-19
#define ELECTRON_MASS 9.10938356e-31
#define PROTON_MASS 1.6726219e-27

typedef struct {
  PetscInt  steps;                          /* Number of time steps */
  PetscReal step_size;                      /* Size of the time step */
  PetscReal gaussian_w;                     /* Width of quadrature evaulation on gaussian mollifiers */
  PetscReal epsilon;                        /* Gaussian regularization parameter */
  PetscReal t_0;                            /* time nondimensionalization */
  PetscInt  mass_units[LANDAU_MAX_SPECIES]; /* 0 for electron mass units, 1 for proton mass units */
  PetscReal masses[LANDAU_MAX_SPECIES];     /* Electron, Sr+ Mass [kg] */
  PetscReal T[LANDAU_MAX_SPECIES];          /* Electron, Ion Temperature [K] */
  PetscReal v0[LANDAU_MAX_SPECIES];         /* Species mean velocity in 1D */
  PetscReal n0[LANDAU_MAX_SPECIES];
  PetscReal charges[LANDAU_MAX_SPECIES];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt   nn0=LANDAU_MAX_SPECIES, nT=LANDAU_MAX_SPECIES;
  PetscInt   nmu=LANDAU_MAX_SPECIES, nm=LANDAU_MAX_SPECIES;
  PetscInt   nc=LANDAU_MAX_SPECIES;
  PetscBool  nmuflg, Tflg, cflg;

  PetscFunctionBeginUser;
  options->gaussian_w    = -1.;
  options->step_size     = 0.1;
  options->steps         = 1;
  options->epsilon       = 1.9;
  options->T[0]          = 5*1.16045250061657e7; /* 5kev converted to kelvin */
  options->T[1]          = 5*1.16045250061657e7; /* 5kev converted to kelvin */
  options->masses[0]     = ELECTRON_MASS;
  options->masses[1]     = ELECTRON_MASS;
  options->mass_units[0] = 0;
  options->mass_units[1] = 0;
  options->charges[0]    = -ELEMENTARY_CHARGE;
  options->charges[1]    = -ELEMENTARY_CHARGE;
  options->n0[0]         = 1.0e20;
  options->n0[1]         = 1.0e20;

  PetscOptionsBegin(comm, "", "Collision Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-steps", "max number of time steps to take", "ex29.c", options->steps, &options->steps, NULL));
  PetscCall(PetscOptionsReal("-step_size", "size of the time step", "ex29.c", options->step_size, &options->step_size, NULL));
  PetscCall(PetscOptionsReal("-gaussian_width", "Width of entropy gradient quadrature evaluation", "ex29.c", options->gaussian_w, &options->gaussian_w, NULL));
  PetscCall(PetscOptionsReal("-epsilon", "Mollification parameter", "ex29.c", options->epsilon, &options->epsilon, NULL));
  PetscCall(PetscOptionsRealArray("-dm_swarm_number_density", "The non normalized number density of each species", "", options->n0, &nn0, NULL));
  PetscCall(PetscOptionsRealArray("-dm_swarm_temperature", "The temperature of each species in KeV", "", options->T, &nT, &Tflg));
  PetscCall(PetscOptionsIntArray("-dm_swarm_mass_units", "0 for electron 1 for proton mass", "", options->mass_units, &nmu, &nmuflg));
  PetscCall(PetscOptionsRealArray("-dm_swarm_masses", "The mass of each species in multiples of fundamental mass units", "", options->masses, &nm, NULL));
  PetscCall(PetscOptionsRealArray("-dm_swarm_charges", "The charge of each species in fundamental charge units (-1,2,3...)", "", options->charges, &nc, &cflg));
  PetscOptionsEnd();

  /* If mass units were specified, get the mass array and compute masses */
  if (nmuflg) {
    PetscInt idx;
    PetscCheck(nm == nmu, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Number of mass units and number of masses given are not equal.")
    for (idx = 0; idx < nmu; ++idx) options->masses[idx] = options->mass_units[idx] == 0 ? ELECTRON_MASS*options->masses[idx] : PROTON_MASS*options->masses[idx];
  }
  if (Tflg) {
    PetscInt idx;
    for (idx = 0; idx < nT; ++idx) options->T[idx] *= 1.1604525e7;
    for (idx = 0; idx < nT; ++idx) options->v0[idx] = PetscSqrtReal(BOLTZMANN_K * options->T[idx] / options->masses[idx]);
  }
  if (cflg) {
    PetscInt idx;
    for (idx = 0; idx < nc; ++idx) options->charges[idx] *= ELEMENTARY_CHARGE;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscInt       dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "gradS", dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
  PetscCall(DMSwarmInitializeCoordinates(*sw));
  PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, user->v0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(PetscObjectSetName((PetscObject) *sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-swarm_view"));
  PetscFunctionReturn(0);
}

/* Internal dmplex function, same as found in dmpleximpl.h */
static void DMPlex_WaxpyD_Internal(PetscInt dim, PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w)
{
  PetscInt d;

  for (d = 0; d < dim; ++d) w[d] = a*x[d] + y[d];
}

/* Internal dmplex function, same as found in dmpleximpl.h */
static PetscReal DMPlex_DotD_Internal(PetscInt dim, const PetscScalar *x, const PetscReal *y)
{
  PetscReal sum = 0.0;
  PetscInt d;

  for (d = 0; d < dim; ++d) sum += PetscRealPart(x[d])*y[d];
  return sum;
}

/* Internal dmplex function, same as found in dmpleximpl.h */
static void DMPlex_MultAdd2DReal_Internal(const PetscReal A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[2];
  z[0] = x[0]; z[1] = x[ldx];
  y[0]   += A[0]*z[0] + A[1]*z[1];
  y[ldx] += A[2]*z[0] + A[3]*z[1];
  (void)PetscLogFlops(6.0);
}

/* Internal dmplex function, same as found in dmpleximpl.h to avoid private includes. */
static void DMPlex_MultAdd3DReal_Internal(const PetscReal A[], PetscInt ldx, const PetscScalar x[], PetscScalar y[])
{
  PetscScalar z[3];
  z[0] = x[0]; z[1] = x[ldx]; z[2] = x[ldx*2];
  y[0]     += A[0]*z[0] + A[1]*z[1] + A[2]*z[2];
  y[ldx]   += A[3]*z[0] + A[4]*z[1] + A[5]*z[2];
  y[ldx*2] += A[6]*z[0] + A[7]*z[1] + A[8]*z[2];
  (void)PetscLogFlops(15.0);
}

/*
  Gaussian - The Gaussian function G(x)

  Input Parameters:
+  dim   - The number of dimensions, or size of x
.  mu    - The mean, or center
.  sigma - The standard deviation, or width
-  x     - The evaluation point of the function

  Output Parameter:
. ret - The value G(x)
*/
static PetscReal Gaussian(PetscInt dim, const PetscReal mu[], PetscReal sigma, const PetscReal x[])
{
  PetscReal arg = 0.0;
  PetscInt  d;

  for (d = 0; d < dim; ++d) arg += PetscSqr(x[d] - mu[d]);
  return PetscPowReal(2.0*PETSC_PI*sigma, -dim/2.0) * PetscExpReal(-arg/(2.0*sigma));
}

/*
  ComputeGradS - Compute grad_v dS_eps/df

  Input Parameters:
+ dim      - The dimension
. Np       - The number of particles
. vp       - The velocity v_p of the particle at which we evaluate
. velocity - The velocity field for all particles
. epsilon  - The regularization strength
. pidx     - The index of the particle being evaluated
. ctx      - The user context
  Output Parameter:
. integral - The output grad_v dS_eps/df (v_p)

  Note:
  This comes from (3.6) in [1], and we are computing
$   \nabla_v S_p = \grad \psi_\epsilon(v_p - v) log \sum_q \psi_\epsilon(v - v_q)
  which is discretized by using a one-point quadrature in each box l at its center v^c_l
$   \sum_l h^d \nabla\psi_\epsilon(v_p - v^c_l) \log\left( \sum_q w_q \psi_\epsilon(v^c_l - v_q) \right)
  where h^d is the volume of each box. Quadrature points are evaluated on the disc or ball using algoim
  and are tabulated in ex27.h for a fixed \epsilon.
*/

static PetscErrorCode ComputeGradS(PetscInt dim, PetscReal* weight, PetscInt Np, const PetscReal vp[], const PetscReal velocity[], PetscReal integral[], PetscInt pidx, AppCtx *ctx)
{
  PetscReal sum, epsilon=ctx->epsilon, vc_l[3], *points, *qw;
  PetscInt  p, d, ncp, i;

  PetscFunctionBeginHot;
  switch (dim) {
    case 2:
      ncp = 400;
      points = QuadraturePoints_2D;
      qw = QuadratureWeights_2D;
      break;
    case 3:
      ncp = 729;
      points = QuadraturePoints_3D;
      qw = QuadratureWeights_3D;
      break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension%" PetscInt_FMT, dim);
  }

  if (ctx->gaussian_w <= 0) {
    ctx->gaussian_w = PetscSqrtReal(-2*epsilon*PetscLogReal(1000.*PETSC_MACHINE_EPSILON/(2*epsilon)));
  }

  for (d = 0; d < dim; ++d) integral[d] = 0.0;
  for (i = 0; i < ncp; ++i){
    sum = 0.;
    for (d = 0; d < dim; ++d) vc_l[d] = points[i*dim+d];
    for (p = 0; p < Np; ++p){

      if (p == pidx) continue;
      sum += weight[p] * qw[i] * Gaussian(dim, &velocity[p*dim], epsilon, vc_l);
    }
    sum = PetscLogReal(sum);
    for (d = 0; d < dim; ++d) integral[d] += qw[i] * (1./(epsilon))*(vp[d] - vc_l[d])*(Gaussian(dim, vp, epsilon, vc_l)) * sum;
  }
  PetscFunctionReturn(0);
}

/* Q = 1/|xi| (I - xi xi^T / |xi|^2), xi = vp - vq */
static PetscErrorCode QCompute(PetscInt dim, const PetscReal vp[], const PetscReal vq[], PetscReal Q[])
{
  PetscReal xi[3], xi2, xi3, mag;
  PetscInt  d, e;

  PetscFunctionBeginHot;
  DMPlex_WaxpyD_Internal(dim, -1.0, vq, vp, xi);
  xi2 = DMPlex_DotD_Internal(dim, xi, xi);
  mag = PetscSqrtReal(xi2);
  xi3 = xi2 * mag;
  for (d = 0; d < dim; ++d) {
    for (e = 0; e < dim; ++e) {
      Q[d*dim+e] = -xi[d]*xi[e] / xi3;
    }
    Q[d*dim+d] += 1. / mag;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  PetscInt           dbg  = 0;
  DM                 sw;                  /* Particles */
  const PetscScalar *u;                   /* input solution vector */
  PetscScalar       *r;
  PetscReal         *gradS, *weight;
  PetscReal          nu_alpha[LANDAU_MAX_SPECIES], nu_beta[LANDAU_MAX_SPECIES];
  PetscReal          lnLam=10., t0, nu_nd, m0=user->masses[0];
  PetscInt           dim, d, Np, p, q, s, *species, Ns;

  PetscFunctionBeginUser;

  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  /* Non dimensionalization of \nu, todo: nondimensionalization to be moved out of the solver into a part of swarm in future updates. */
  t0 = 8*PETSC_PI*PetscSqr(EPSILON_NOUGHT*m0/PetscSqr(user->charges[0]*ELEMENTARY_CHARGE))/lnLam/user->n0[0]*PetscPowReal(user->v0[0],3);
  nu_nd = t0*user->n0[0]/PetscPowReal(user->v0[0],3.);
  for (s = 0; s < Ns; ++s){
    nu_alpha[s] = PetscSqr(user->charges[s]*ELEMENTARY_CHARGE/m0)*m0/user->masses[s];
    nu_beta[s] = PetscSqr(user->charges[s]*ELEMENTARY_CHARGE/EPSILON_NOUGHT)*lnLam / (8*PETSC_PI) * nu_nd;
  }
  PetscCall(VecZeroEntries(R));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(VecGetArray(R, &r));
  PetscCall(VecViewFromOptions(U, NULL, "-sol_view"));
  PetscCall(VecGetArrayRead(U, &u));
  Np  /= dim;
  /* The dmswarm stores dS/dv_p precomputed in pre step */
  PetscCall(DMSwarmGetField(sw, "gradS", NULL, NULL, (void **) &gradS));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **) &species));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **) &weight));

  if (dbg) {PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Part  ppr     x        y\n"));}
  for (p = 0; p < Np; ++p) {
    for (q = 0; q < Np; ++q) {
      PetscReal GammaS[3] = {0., 0., 0.}, Q[9];

      if (q == p) continue;
      DMPlex_WaxpyD_Internal(dim, -1.0, (const PetscReal*)&gradS[q*dim], (const PetscReal*)&gradS[p*dim], GammaS);

      for (d=0; d < dim; ++d) GammaS[d] *= nu_alpha[species[p]]*nu_beta[species[q]]*weight[q];
      PetscCall(QCompute(dim, &u[p*dim], &u[q*dim], Q));
      switch (dim) {
        case 2: DMPlex_MultAdd2DReal_Internal(Q, 1, GammaS, &r[p*dim]);break;
        case 3: DMPlex_MultAdd3DReal_Internal(Q, 1, GammaS, &r[p*dim]);break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension%" PetscInt_FMT, dim);
      }
    }
    if (dbg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final %4" PetscInt_FMT " %10.8lf %10.8lf\n", p, r[p*dim+0], r[p*dim+1]));
  }
  PetscCall(DMSwarmRestoreField(sw, "gradS", NULL, NULL, (void **) &gradS));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **) &species));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &weight));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(R, &r));
  PetscCall(VecViewFromOptions(R, NULL, "-residual_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeIntGradS(TS ts)
{
  PetscInt       p, Np, dim;
  PetscReal     *gradS, *velocity, *weights;
  DM             sw;
  Vec            sol;
  AppCtx        *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(TSGetSolution(ts, &sol));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(sol, &Np));
  Np /= dim;
  PetscCall(DMSwarmGetField(sw, "gradS", NULL, NULL, (void **) &gradS));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocity));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **) &weights));
  for (p = 0; p < Np; ++p) {
    PetscCall(ComputeGradS(dim, weights, Np, &velocity[p*dim], velocity, &gradS[p*dim], p, user));
   }
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &weights));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity));
  PetscCall(DMSwarmRestoreField(sw, "gradS", NULL, NULL, (void **) &gradS));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDistribution(DM sw, PetscReal confidenceLevel, AppCtx *user)
{
  Vec            locv, locsv;
  PetscProbFunc  cdf;
  PetscReal      alpha;
  PetscScalar   *a;
  PetscReal     *velocity;
  PetscInt      *sn, *species;
  PetscInt       dim, d, n, p, Ns, s, off;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) sw, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  switch (dim) {
    case 1: cdf = PetscCDFMaxwellBoltzmann1D;break;
    case 2: cdf = PetscCDFMaxwellBoltzmann2D;break;
    case 3: cdf = PetscCDFMaxwellBoltzmann3D;break;
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension%" PetscInt_FMT, dim);
  }
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMSwarmGetLocalSize(sw, &n));
  if (Ns <= 1) {
    PetscCall(DMSwarmCreateLocalVectorFromField(sw, "velocity", &locv));
    PetscCall(PetscProbComputeKSStatistic(locv, cdf, &alpha));
    PetscCall(DMSwarmDestroyLocalVectorFromField(sw, "velocity", &locv));
    if (alpha < confidenceLevel) PetscCall(PetscPrintf(comm, "The KS test accepts the null hypothesis at level %.2g\n", (double) confidenceLevel));
    else                         PetscCall(PetscPrintf(comm, "The KS test rejects the null hypothesis at level %.2g (%.2g)\n", (double) confidenceLevel, (double) alpha));
  } else {
    PetscCall(PetscCalloc1(Ns, &sn));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocity));
    PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **) &species));
    for (p = 0; p < n; ++p) ++sn[species[p]];
    for (s = 0; s < Ns; ++s) {
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, sn[s]*dim, &locsv));
      PetscCall(VecSetBlockSize(locsv, dim));
      PetscCall(VecGetArray(locsv, &a));
      for (p = 0, off = 0; p < n; ++p) {
        if (species[p] == s) for (d = 0; d < dim; ++d) a[off++] = (user->v0[0]/user->v0[s]) * velocity[p*dim+d];
      }
      PetscCall(VecRestoreArray(locsv, &a));
      PetscCall(PetscProbComputeKSStatistic(locsv, cdf, &alpha));
      PetscCall(VecDestroy(&locsv));
      if (alpha < confidenceLevel) PetscCall(PetscPrintf(comm, "The KS test accepts the null hypothesis for species %" PetscInt_FMT " at level %.2g\n", s, (double) confidenceLevel));
      else                         PetscCall(PetscPrintf(comm, "The KS test rejects the null hypothesis for species %" PetscInt_FMT " at level %.2g (%.2g)\n", s, (double) confidenceLevel, (double) alpha));
    }
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity));
    PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **) &species));
    PetscCall(PetscFree(sn));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CalculateMomentsAndTemperatures(DM sw, PetscReal* momentum, PetscReal *KE, PetscReal* T)
{
  AppCtx        *user;
  PetscInt       Np, p, dim, d, cStart, cEnd, s;
  PetscInt      *species, Ns;
  PetscReal     *velocities, *weights;
  DM             plex;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(sw, &plex));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMGetApplicationContext(sw, (void **) &user));
  PetscCall(DMGetDimension(plex, &dim));
  PetscCall(DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmSortGetNumberOfPointsPerCell(sw, cStart, &Np));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocities));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **) &weights));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **) &species));
  for (p = 0; p < Np; ++p){
    PetscReal v2 = 0.;

    for (d=0; d < dim; ++d) momentum[species[p]*dim+d] += velocities[p*dim+d] * weights[p];
    for (d=0; d < dim; ++d) v2 += PetscSqr(velocities[p*dim+d]);
    KE[species[p]] +=  weights[p] * v2;

  }

  for (s=0; s < Ns; ++s){
    /* the two gets cancelled out. */
    PetscReal    dimensionalization, udotu=0., dimratio;
    dimratio = 2./dim;
    dimensionalization  = (user->masses[s]/BOLTZMANN_K);
    dimensionalization *= PetscSqr(user->v0[0]);
    T[s] = KE[s];
    for (d = 0; d < dim; ++d) udotu += PetscSqr(momentum[s*dim+d]);
    T[s] -= udotu;
    T[s] *= dimratio/2. * dimensionalization/1.16045250061657e7;
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocities));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &weights));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **) &species));
  PetscFunctionReturn(0);
}

static PetscErrorCode Monitor(TS ts)
{
  AppCtx    *user;
  DM         sw;
  PetscReal *T, *KE, *mom;
  PetscInt   s, Ns, dim;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void **) &user));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(PetscCalloc3(Ns, &T, Ns, &KE, dim*Ns, &mom));
  PetscCall(CalculateMomentsAndTemperatures(sw, mom, KE, T));
  for (s = 0; s < Ns; ++s){
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "momentum[%i]: %g KE[%i]: %g T[%i]: %g\n", s, mom[s*dim], s, KE[s], s, T[s]));
  }
  PetscCall(TestDistribution(sw, 0.05, user));
  PetscCall(PetscFree3(T, KE, mom));
  PetscFunctionReturn(0);
}

/*
 TS Post Step Function. Copy the solution back into the swarm for migration. We may also need to reform
 the solution vector in cases of particle migration, but we forgo that here since there is no velocity space grid
 to migrate between.
*/
static PetscErrorCode UpdateSwarm(TS ts)
{
  PetscInt idx, n;
  const PetscScalar *u;
  PetscScalar *velocity;
  DM sw;
  Vec sol;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocity));
  PetscCall(TSGetSolution(ts, &sol));
  PetscCall(VecGetArrayRead(sol, &u));
  PetscCall(VecGetLocalSize(sol, &n));
  for (idx = 0; idx < n; ++idx) velocity[idx] = u[idx];
  PetscCall(VecRestoreArrayRead(sol, &u));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity));
  PetscCall(Monitor(ts));
  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM             sw, plex;
  Vec            v;
  AppCtx        *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void **) &user));
  PetscCall(DMSwarmGetCellDM(sw, &plex));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(VecCopy(v, u));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;     /* nonlinear solver */
  DM             dm, sw; /* Velocity space mesh and Particle Swarm */
  Vec            u, v;   /* problem vector */
  PetscInt       Np;
  MPI_Comm       comm;
  AppCtx         user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  /* Initialize objects and set initial conditions */
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateSwarm(dm, &user, &sw));
  PetscCall(DMSetApplicationContext(sw, &user));
  PetscCall(DMSwarmVectorDefineField(sw, "velocity"));
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 100.0));
  PetscCall(TSSetTimeStep(ts, user.step_size));
  PetscCall(TSSetMaxSteps(ts, user.steps));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunctionParticles, &user));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(VecDuplicate(v, &u));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(TSSetPreStep(ts, ComputeIntGradS));
  PetscCall(TSSetPostStep(ts, UpdateSwarm));
  /* Test the initial distribution. */
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(PetscPrintf(comm, "Np: %i\n", Np));
  PetscCall(TestDistribution(sw, 0.05, &user));
  PetscCall(TSSolve(ts, u));
  PetscCall(VecDestroy(&u));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 2d_one_species
    requires: ks triangle !single !complex
    args: -steps 1 -step_size 0.01\
    -ts_type theta -ts_theta_theta 0.5\
    -dm_plex_simplex 0 -dm_plex_dim 2\
    -dm_plex_box_lower -1,-1\
    -dm_plex_box_upper 1,1\
    -dm_plex_box_faces 1,1\
    -dm_swarm_num_particles 150\
    -dm_swarm_coordinate_density gaussian\
    -snes_monitor\
    -snes_mf\
    -dm_swarm_num_species 2\
    -dm_swarm_masses 1.,1.\
    -dm_swarm_mass_units 0,0\
    -dm_swarm_charges -1,-1\
    -dm_swarm_temperature 5,6
  test:
    suffix: 2d_two_species
    requires: ks triangle !single !complex
    args: -steps 1 -step_size 0.01\
    -ts_type theta -ts_theta_theta 0.5\
    -dm_plex_simplex 0 -dm_plex_dim 2\
    -dm_plex_box_lower -1,-1\
    -dm_plex_box_upper 1,1\
    -dm_plex_box_faces 1,1\
    -dm_swarm_num_particles 150\
    -dm_swarm_coordinate_density gaussian\
    -snes_monitor\
    -snes_mf\
    -dm_swarm_num_species 2\
    -dm_swarm_masses 1.,2.\
    -dm_swarm_mass_units 0,1\
    -dm_swarm_charges -1,1\
    -dm_swarm_temperature 5,6
  test:
    suffix: 3d_one_species
    requires: ks triangle !single !complex
    args: -steps 1 -step_size 0.01\
    -ts_type theta -ts_theta_theta 0.5\
    -dm_plex_simplex 0 -dm_plex_dim 3\
    -dm_plex_box_lower -1,-1,-1\
    -dm_plex_box_upper 1,1,1\
    -dm_plex_box_faces 1,1,1\
    -dm_swarm_num_particles 150\
    -dm_swarm_coordinate_density gaussian\
    -snes_monitor\
    -snes_mf\
    -dm_swarm_num_species 2\
    -dm_swarm_masses 1.,1.\
    -dm_swarm_mass_units 0,0\
    -dm_swarm_charges -1,-1\
    -dm_swarm_temperature 5,6
  test:
    suffix: 3d_two_species
    requires: ks triangle !single !complex
    args: -steps 1 -step_size 0.01\
    -ts_type theta -ts_theta_theta 0.5\
    -dm_plex_simplex 0 -dm_plex_dim 3\
    -dm_plex_box_lower -1,-1,-1\
    -dm_plex_box_upper 1,1,1\
    -dm_plex_box_faces 1,1,1\
    -dm_swarm_num_particles 150\
    -dm_swarm_coordinate_density gaussian\
    -snes_monitor\
    -snes_mf\
    -dm_swarm_num_species 2\
    -dm_swarm_masses 1.,2.\
    -dm_swarm_mass_units 0,1\
    -dm_swarm_charges -1,1\
    -dm_swarm_temperature 5,6
TEST*/
