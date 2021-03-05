static char help[] = "Particle Basis Landau Example using nonlinear solve + Implicit Midpoint-like time stepping.";

/* TODO

1) SNES is sensitive to epsilon. Should we do continuation in it?

2) Code up new timestepper

3) Visualize distributions

*/

/* References
  [1] https://arxiv.org/abs/1910.03080v2
*/

#include <petscdmplex.h>
#include <petsc/private/petscfeimpl.h> /* For CoordinatesRefToReal() */
#include <petscdmswarm.h>
#include <petscts.h>
#include <petscviewer.h>
#include <petscmath.h>

typedef struct {
  /* Velocity space grid and functions */
  PetscInt  N;         /* The number of partices per spatial cell */
  PetscReal L;         /* Velocity space is [-L, L]^d */
  PetscReal h;         /* Spacing for grid 2L / N^{1/d} */
  PetscReal epsilon;   /* gaussian regularization parameter */
  /* Monitoring */
  PetscBool monitor;   /* Conservation montior */
  PetscInt  ostep;     /* print the energy at each ostep time steps */
  PetscReal momentTol; /* Tolerance for checking moment conservation */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->monitor   = PETSC_FALSE;
  options->N         = 1;
  options->momentTol = 100.0*PETSC_MACHINE_EPSILON;
  options->ostep     = 100;
  options->L         = 1.0;
  options->h         = -1.0;
  options->epsilon   = -1.0;

  ierr = PetscOptionsBegin(comm, "", "Collision Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitor", "Flag to use the TS histogram monitor", "ex27.c", options->monitor, &options->monitor, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-N", "Number of particles per spatial cell", "ex27.c", options->N, &options->N, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-output_step", "Number of time steps between output", "ex27.c", options->ostep, &options->ostep, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-L", "Velocity-space extent", "ex27.c", options->L, &options->L, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-h", "Velocity-space resolution", "ex27.c", options->h, &options->h, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-epsilon", "Mollifier regularization parameter", "ex27.c", options->epsilon, &options->epsilon, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialCoordinates(DM sw)
{
  AppCtx        *user;
  PetscRandom    rnd, rndv;
  DM             dm;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscReal     *centroid, *coords, *velocity, *xi0, *v0, *J, *invJ, detJ, *vals;
  PetscInt       dim, d, cStart, cEnd, c, Np, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Randomization for coordinates */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) sw), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -1.0, 1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);

  /*
    Randomization for velocity if a specific initial distribution function is not chosen.
    Its symmetric for no real reason, an assymetric function should have similar conservation
    as there are no weights being shifted, simply velocities.
  */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) sw), &rndv);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rndv, -1., 1.);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rndv);CHKERRQ(ierr);


  ierr = DMGetApplicationContext(sw, (void **) &user);CHKERRQ(ierr);
  Np   = user->N;
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  ierr = DMSwarmGetCellDM(sw, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) xi0[d] = -1.0;
  ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    if (Np == 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d){
        coords[c*dim+d] = centroid[d];
      }
      vals[c] = 1.0;
    } else {
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr); /* affine */
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        for (d = 0; d < dim; ++d) {
          ierr = PetscRandomGetValueReal(rnd, &refcoords[d]);CHKERRQ(ierr);
          sum += refcoords[d];
        }
        if (simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
        vals[n] = 1.0;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
      }
    }
  }
  /* Randomized velicities to start */
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      for (d = 0; d < dim; ++d) {
        PetscReal v_val;

        ierr = PetscRandomGetValueReal(rndv, &v_val);
        velocity[p*dim+d] = v_val;
      }
    }
  }
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = PetscFree5(centroid, xi0, v0, J, invJ);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rndv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Get velocities from swarm and place in solution vector */
static PetscErrorCode SetInitialConditions(DM dmSw, Vec u)
{
  DM             dm;
  AppCtx        *user;
  PetscReal     *velocity;
  PetscScalar   *initialConditions;
  PetscInt       dim, d, cStart, cEnd, c, Np, p, n;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(u, &n);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dmSw, (void **) &user);CHKERRQ(ierr);
  Np   = user->N;
  ierr = DMSwarmGetCellDM(dmSw, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmSw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);
  ierr = VecGetArray(u, &initialConditions);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;
      for (d = 0; d < dim; d++) {
        initialConditions[n*dim+d] = velocity[n*dim+d];
      }
    }
  }
  ierr = VecRestoreArray(u, &initialConditions);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmSw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np = user->N, p;
  PetscBool      view = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);

  /* h = 2L/n and N = n^d */
  if (user->h < 0.) user->h = 2.*user->L / PetscPowReal(user->N, 1./dim);
  /* From Section 4 in [1], \epsilon = 0.64 h^.98 */
  if (user->epsilon < 0.) user->epsilon = 0.64*pow(user->h, 1.98);
  ierr = PetscOptionsGetBool(NULL, NULL, "-param_view", &view, NULL);CHKERRQ(ierr);
  if (view) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "N: %D L: %g h: %g eps: %g\n", user->N, user->L, user->h, user->epsilon);CHKERRQ(ierr);
  }

  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;
      cellid[n] = c;
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
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

  Output Parameter:
. integral - The output grad_v dS_eps/df (v_p)

  Note:
  This comes from (3.6) in [1], and we are computing
$   \nabla_v S_p = \grad \psi_\epsilon(v_p - v) log \sum_q \psi_\epsilon(v - v_q)
  which is discretized by using a one-point quadrature in each box l at its center v^c_l
$   \sum_l h^d \nabla\psi_\epsilon(v_p - v^c_l) \log\left( \sum_q w_q \psi_\epsilon(v^c_l - v_q) \right)
  where h^d is the volume of each box.
*/
static PetscErrorCode ComputeGradS(PetscInt dim, PetscInt Np, const PetscReal vp[], const PetscReal velocity[], PetscReal integral[], AppCtx *ctx) {
  PetscReal vc_l[3], L = ctx->L, h = ctx->h, epsilon = ctx->epsilon, init = 0.5*h - L;
  PetscInt  nx = roundf(2.*L / h);
  PetscInt  ny = dim > 1 ? nx : 1;
  PetscInt  nz = dim > 2 ? nx : 1;
  PetscInt  i, j, k, d, q, dbg = 0;

  PetscFunctionBeginHot;
  for (d = 0; d < dim; ++d) integral[d] = 0.0;
  for (k = 0, vc_l[2] = init; k < nz; ++k, vc_l[2] += h) {
    for (j = 0, vc_l[1] = init; j < ny; ++j, vc_l[1] += h) {
      for (i = 0, vc_l[0] = init; i < nx; ++i, vc_l[0] += h) {
        PetscReal sum = 0.0;

        if (dbg) {PetscPrintf(PETSC_COMM_SELF, "(%D %D) vc_l: %g %g\n", i, j, vc_l[0], vc_l[1]);}
        /* \log \sum_k \psi(v - v_k)  */
        for (q = 0; q < Np; ++q) sum += Gaussian(dim, &velocity[q*dim], epsilon, vc_l);
        sum = PetscLogReal(sum);

        for (d = 0; d < dim; ++d) integral[d] += (-1./(epsilon))*PetscAbsReal(vp[d] - vc_l[d])*(Gaussian(dim, vp, epsilon, vc_l)) * sum;
      }
    }
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
  AppCtx            *user = (AppCtx *) ctx;
  PetscInt           dbg  = 0;
  DM                 sw;                  /* point tracking, problem topology */
  const PetscScalar *u;                   /* input solution vector */
  PetscScalar       *r;
  PetscReal         *velocity;
  PetscQuadrature    quad;                /* for integral evaluation */
  PetscInt           dim, Np, p, q;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(R);CHKERRQ(ierr);

  /* Create quadrature for integral evaluation */
  ierr = TSGetDM(ts, &sw);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMGetDimension(sw, &dim);
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  ierr = VecGetArray(R, &r);
  ierr = VecGetArrayRead(U, &u);
  Np  /= dim;

  /* REMOVE This is a cheat to get v^n */
  ierr = DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);

  ierr = PetscDTGaussTensorQuadrature(dim, 2*dim*Np,dim*Np, -1, 1, &quad);CHKERRQ(ierr);
  if (dbg) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Part  ppr     x        y\n");CHKERRQ(ierr);}
  for (p = 0; p < Np; ++p) {
    PetscReal gradS_p[3] = {0., 0., 0.};

    ierr = ComputeGradS(dim, Np, &velocity[p*dim], velocity, gradS_p, user);CHKERRQ(ierr);
    for (q = 0; q < Np; ++q) {
      PetscReal gradS_q[3] = {0., 0., 0.}, GammaS[3] = {0., 0., 0.}, Q[9];

      if (q == p) continue;
      ierr = ComputeGradS(dim, Np, &velocity[q*dim], velocity, gradS_q, user);CHKERRQ(ierr);
      DMPlex_WaxpyD_Internal(dim, -1.0, gradS_q, gradS_p, GammaS);
      ierr = QCompute(dim, &u[p*dim], &u[q*dim], Q);CHKERRQ(ierr);
      switch (dim) {
        case 2: DMPlex_MultAdd2DReal_Internal(Q, 1, GammaS, &r[p*dim]);break;
        case 3: DMPlex_MultAdd3DReal_Internal(Q, 1, GammaS, &r[p*dim]);break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension %D", dim);
      }
    }
    if (dbg) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Final %4D %10.8lf %10.8lf\n", p, r[p*dim+0], r[p*dim+1]);CHKERRQ(ierr);}
  }
  ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);

  /* REMOVE */
  ierr = DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);
  ierr = VecViewFromOptions(R, NULL, "-residual_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 TS Post Step Function. Copy the solution back into the swarm for migration. We may also need to reform
 the solution vector in cases of particle migration, but we forgo that here since there is no velocity space grid
 to migrate between.
*/
static PetscErrorCode UpdateSwarm(TS ts){
  PetscInt idx, n;
  const PetscScalar *u;
  PetscScalar *velocity;
  DM sw;
  Vec sol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = TSGetDM(ts, &sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);

  ierr = TSGetSolution(ts, &sol);CHKERRQ(ierr);
  ierr = VecGetArrayRead(sol, &u);CHKERRQ(ierr);
  ierr = VecGetLocalSize(sol, &n);
  for(idx = 0; idx < n; ++idx) velocity[idx] = u[idx];
  ierr = VecRestoreArrayRead(sol, &u);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  DM             dm;
  AppCtx        *user;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, (void **) &user);CHKERRQ(ierr);
  ierr = SetInitialCoordinates(dm);CHKERRQ(ierr);
  ierr = SetInitialConditions(dm, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx            *user  = (AppCtx *) ctx;
  const PetscScalar *u;
  PetscReal          tot_E = 0., tot_Momx = 0., tot_Momy = 0.;
  MPI_Comm           comm;
  PetscReal          dt;
  PetscInt           Np, p;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  if (step%user->ostep == 0) {
    ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
    if (!step) {ierr = PetscPrintf(comm, "Time     Step Part     Energy     Momentum     v0     v1\n");CHKERRQ(ierr);}
    ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
    ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
    ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
    Np /= 2;
    for (p = 0; p < Np; ++p) {
      PetscReal v  = PetscRealPart(u[p*2+0]*u[p*2+0]);
      v  += PetscRealPart(u[p*2+1]*u[p*2+1]);
      tot_Momx += PetscRealPart(u[p*2]);
      tot_Momy += PetscRealPart(u[p*2+1]);
      const PetscReal E  = 0.5*(v);
      const PetscReal mom = PetscSqrtReal(v);
      tot_E += E;
      ierr = PetscPrintf(comm, "%.6lf %4D %4D %10.8lf %10.8lf %10.8lf %10.8lf\n", t, step, p, (double) E, (double) mom, (double)u[p*2],(double)u[p*2+1]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm, "Total Energy: %10.8lf    Total Momentum x: %10.8lf    Total Momentum y: %10.8lf\n", tot_E, tot_Momx, tot_Momy);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if 0
/* Integrate the gradient of the entropy functional for a particle. This should be optimized out */
static PetscErrorCode ComputeGradS_Quad(PetscInt dim, PetscInt Np, PetscQuadrature quad, PetscReal *particle, PetscReal *field, PetscReal *integral, void *ctx)
{
  PetscInt         d, q, Nq;
  PetscReal       *xi0, *v0, *J, *invJ, detJ;
  const PetscReal *weights, *points;
  AppCtx*          user = (AppCtx*) ctx;
  DM               plex;
  PetscErrorCode   ierr;

  PetscFunctionBeginHot;
  ierr = DMSwarmGetCellDM(sw, &plex);

  ierr = DMGetDimension(plex, &dim);
  ierr = PetscMalloc4(dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  ierr = DMPlexComputeCellGeometryFEM(plex, 0, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
  /* Assuming 2D here, easily extrapolated to 3D when it finally works */
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, &points, &weights);CHKERRQ(ierr);
  for (q = 0; q < Nq; ++q) {
    PetscReal integrand[2] = {0.,0.};
    PetscReal w;
    PetscReal x[2];

    CoordinatesRefToReal(dim, dim, xi0, v0, J, &points[q*dim], x);CHKERRQ(ierr);
    w = detJ*weights[q];

    //GradGaussian(particle, dim, q_point, user->epsilon, integrand);
    mollifiedGaussian(dim, particle, &points[q*dim], user->epsilon, Np, field, integrand);
    for (d = 0; d < dim; ++d) integral[d] += integrand[d]; //*w;
    break;
  }
  ierr = PetscFree4(xi0, v0, J, invJ);
  PetscFunctionReturn(0);
}
#endif

int main(int argc,char **argv)
{
  TS             ts;     /* nonlinear solver */
  DM             dm, sw; /* Velocity space mesh and Particle Swarm */
  Vec            u, v;   /* problem vector */
  MPI_Comm       comm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);

  /* Initialize objects and set initial conditions */
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(sw, &user);CHKERRQ(ierr);
  ierr = DMSwarmVectorDefineField(sw, "velocity");CHKERRQ(ierr);
  /*
    The following statement depends on a fully discrete in velocity, or a semi-discrete in velocity case:

    Compute the time step for v^n+1 = 1(p,p')Q(v_i^n+1/2 - v_k^n+1/2)\Gamma(S^n, p, p')

    A Tensor Quadrature rule is used to evaluate the integrals for S = \int_R^d \grad \psi_\epsilon (v_p)

    the log of the summation for the term ln \sum_k \psi_\epsilon(v_p') is computed in relation to the gaussian centered at v_p

    The midpoint velocities are found using SNES for a non linear forward solve to compute the backwards evaluation for the midpoints.

  */
  ierr = TSCreate(comm, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, sw);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, 10.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 0.1);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, 1);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts, NULL, RHSFunctionParticles, &user);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetComputeInitialCondition(ts, InitializeSolve);CHKERRQ(ierr);
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "velocity", &v);CHKERRQ(ierr);
  ierr = VecDuplicate(v, &u);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v);CHKERRQ(ierr);
  ierr = TSComputeInitialCondition(ts, u);CHKERRQ(ierr);
  if (user.monitor) TSMonitorSet(ts, Monitor, &user, NULL);
  ierr = TSSetPostStep(ts, UpdateSwarm);CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   build:
     requires: triangle !single !complex
   test:
     suffix: euler
     args: -N 3 -output_step 5 -ts_type euler -dm_plex_box_dim 2 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1 -dm_plex_box_upper 1,1 -dm_view -monitor -output_step 1
   test:
     suffix: 1
     args: -N 3 -output_step 5 -ts_type theta -ts_theta_theta 1.0 -dm_plex_box_dim 2 -dm_plex_box_faces 1,1 -dm_plex_box_lower -2,-2 -dm_plex_box_upper 2,2 -dm_view -monitor -output_step 1
   test:
     suffix: 2
     args: -N 3 -output_step 5 -ts_type theta -ts_theta_theta 0.5 -snes_fd -dm_plex_box_dim 2 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1 -dm_plex_box_upper 1,1 -dm_view -monitor -output_step 1
TEST*/
