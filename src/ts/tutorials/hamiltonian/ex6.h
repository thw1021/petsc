
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>
#include <petscviewer.h>
#include <petscmath.h>
#include <petsclandau.h>

/* Some useful constants */
#define BOLTZMANN_K 1.380649e-23 /* J/K */
#define KEV_J 6.241506479963235e15 /*  */
#define LIGHT_C 299792458
#define EPSILON_NOUGHT 8.8542e-12
#define ELEMENTARY_CHARGE 1.602176e-19
#define ELECTRON_MASS 9.10938356e-31
#define PROTON_MASS 1.6726219e-27

// This needs to be replaced with a collisional swarm eventually.
// The idea is that a cellswarm comes out up front and we can operate on it
// directly and copy back in. For now, just make a struct and brute force the
// implementation for proof of concept.
typedef struct {
    Vec weights;
    Vec preStepVelocities;
    Vec dSdvp;
    PetscReal epsilon;
    PetscReal mass;
    PetscReal charge;
    PetscReal n0;
    PetscReal T;
} collisionData;




PetscErrorCode ComputeGradS_1D(PetscReal* weight, PetscInt Np, PetscReal velocity[], PetscReal integral[], collisionData *ctx)
{
  PetscInt  nHermite=6;
  PetscInt  debug = 0;

  PetscReal kHermite[6] = {-2.3506049736745, -1.3358490740137, -0.43607741192762, 0.43607741192762, 1.3358490740137, 2.3506049736745};
  PetscReal wHermite[6] = {0.0045300099055088, 0.15706732032286, 0.72462959522439, 0.72462959522439, 0.15706732032286, 0.0045300099055088};

  PetscFunctionBeginHot;
  for (PetscInt p = 0; p < Np; ++p){
    PetscReal SQRT2EPSM1, PI2EPSM1, coeff;
    SQRT2EPSM1 = 1./PetscSqrtReal(2.*ctx->epsilon);
    PI2EPSM1 = 1./(2*PETSC_PI * ctx->epsilon);

    for (PetscInt d = 0; d < 1; ++d) integral[p] = 0.0;
    for (PetscInt i=0; i < nHermite; i++){
      PetscReal logsum = 0, kpx, dx;

      for (PetscInt q = 0; q < Np; ++q) {
        kpx = kHermite[i] + velocity[p] * SQRT2EPSM1;
        dx = kpx - velocity[q] * SQRT2EPSM1;
        logsum += weight[q] * PetscExpReal(-PetscSqr(dx)) * PI2EPSM1;
      }
      logsum = wHermite[i]*(1. + PetscLogReal(logsum));
      integral[p] += logsum * kHermite[i];
    }
    coeff = -(2. * ctx->epsilon) / (PETSC_PI * ctx->epsilon);//renormalize for 1D gaussians
    integral[p] *= coeff;
  }
  if (debug) for (PetscInt p = 0; p < Np; ++p ) PetscPrintf(PETSC_COMM_WORLD, "Particle %" PetscInt_FMT " integral %g %g eps %g\n", p, integral[p*2+0], integral[p*2+1], ctx->epsilon);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * A lot of this comes from the swarm in the default examples, but we don't have that here since we're pulling everything from subvectors
 * this should be fixed and use subswarms instead, but for now just use the application context since we aren't migrating particles in this step
 * regardless.
 */
static PetscErrorCode ComputeIntGradS(TS ts)
{
  PetscInt       Np, dim=1;
  DM             cellSwarm;
  PetscReal     *gradS, *velocity, *weights;
  Vec            sol;
  PetscInt       dbg=0;
  collisionData  *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, (PetscCtxRt)&user));
  PetscCall(TSGetDM(ts, &cellSwarm));
  PetscCall(TSGetSolution(ts, &sol));
  PetscCall(VecGetLocalSize(sol, &Np));
  PetscCall(DMSwarmGetField(cellSwarm, "w_q", NULL, NULL, (void**)&weights));
  Np /= dim;
  PetscCall(VecViewFromOptions(sol, NULL, "-collisions_solution_view"));
  PetscCall(VecViewFromOptions(user->dSdvp, NULL, "-collisions_dSdvp_view"));
  PetscCall(VecGetArray(sol, &velocity));
  PetscCall(VecGetArray(user->dSdvp, &gradS));
  switch (dim){
    case 1: PetscCall(ComputeGradS_1D(weights, Np, velocity, gradS, user));break;
    //case 2: PetscCall(ComputeGradS_2D(weights, species, Np, velocity, gradS, user));break;
    //case 3: PetscCall(ComputeGradS_3D(weights, species, Np, velocity, gradS, user));break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension%" PetscInt_FMT, dim);
  }
  if (dbg) for (PetscInt p = 0; p < Np; ++p) PetscPrintf(PETSC_COMM_WORLD, "gradient for particle %" PetscInt_FMT " %g %g\n", p, gradS[p*2+0], gradS[p*2+1]);
  PetscCall(DMSwarmRestoreField(cellSwarm, "w_q", NULL, NULL, (void**)&weights));
  PetscCall(VecRestoreArray(sol, &velocity));
  PetscCall(VecRestoreArray(user->dSdvp, &gradS));
  PetscCall(VecDuplicate(sol, &user->preStepVelocities));
  PetscCall(VecCopy(sol, user->preStepVelocities));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSCollisionFunction(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  collisionData     *user = (collisionData*)ctx;
  PetscInt           dbg  = 0;
  const PetscScalar *u;                   /* input solution vector */
  PetscScalar       *r;
  const PetscReal   *gradS, *weight;
  PetscReal          m0=user->mass;
  PetscInt           dim=1, d, Np;
  DM                 cellSwarm;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &cellSwarm));
  PetscCall(VecZeroEntries(R));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(VecGetArray(R, &r));
  PetscCall(VecViewFromOptions(U, NULL, "-sol_view"));
  PetscCall(VecGetArrayRead(U, &u));
  /* The collision data context holds these for now, it will be replaced with a subswarm. */
  PetscCall(DMSwarmGetField(cellSwarm, "w_q", NULL, NULL, (void**)&weight));
  PetscCall(VecGetArrayRead(user->dSdvp, &gradS));

  if (dbg) {PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Part  ppr     x        y\n"));}
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt q = 0; q < Np; ++q) {
      PetscReal GammaS[3] = {0., 0., 0.}, Q[9];
      PetscReal residual[3] = {0., 0.,0.};

      if (q == p) continue;

      DMPlex_WaxpyD_Internal(1, -1.0, (const PetscReal*)&gradS[q*dim], (const PetscReal*)&gradS[p*dim], GammaS);
      // This has 1/mw_p applied at the computation of \nabla_v_p S in ComputeGammaS(..)
      QCompute(dim, &u[p*dim], &u[q*dim], Q);
      residual[0] = GammaS[0] * PetscSign(u[p] - u[q]);
      //residual[0] = Q[0]*GammaS[0];//1D so this should just reduce to a sign
      //PetscPrintf(PETSC_COMM_WORLD, "Residual in loop: %g\n", residual[0]);
/*
      switch (dim) {
        case 1: DMPlex_MultAdd1DReal_Internal(Q, 1, GammaS, residual);break;
        case 2: DMPlex_MultAdd2DReal_Internal(Q, 1, GammaS, residual);break;
        case 3: DMPlex_MultAdd3DReal_Internal(Q, 1, GammaS, residual);break;
      }
 */
      for(d=0; d < dim; ++d) r[p] += residual[d] * weight[q];//All same species, assume collision frequency of 1, mass ratio to reference mass is 1
    }
    if (dbg) PetscPrintf(PETSC_COMM_WORLD, "Final %4" PetscInt_FMT " %10.8lf\n", p, r[p*dim+0]);
  }
  PetscCall(DMSwarmRestoreField(cellSwarm, "w_q", NULL, NULL, (void**)&weight));
  PetscCall(VecRestoreArrayRead(user->dSdvp, &gradS));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(R, &r));
  PetscCall(VecViewFromOptions(R, NULL, "-residual_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PostCollisions(TS ts){
  DM cellSwarm;
  Vec velocities;
  PetscInt Np;
  PetscReal preStepEnergy=0., postStepEnergy=0., tol = 1e-6;
  PetscReal *preStepVel, *postStepVel, *weights;
  collisionData *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetSolution(ts, &velocities));
  PetscCall(TSGetApplicationContext(ts, &user));
  PetscCall(TSGetDM(ts, &cellSwarm));
  PetscCall(DMSwarmGetField(cellSwarm, "w_q", NULL, NULL, (void**)&weights));
  // Ignoring the mass term, just compute v_i^2 and measure the difference
  PetscCall(VecGetLocalSize(velocities, &Np));
  PetscCall(VecGetArray(user->preStepVelocities, &preStepVel));
  PetscCall(VecGetArray(velocities, &postStepVel));
  for (PetscInt p = 0; p < Np; ++p){

      preStepEnergy += weights[p] * PetscSqr(preStepVel[p]);
      postStepEnergy += weights[p] * PetscSqr(postStepVel[p]);
  }
  PetscCall(VecRestoreArray(user->preStepVelocities, &preStepVel));
  PetscCall(VecRestoreArray(velocities, &postStepVel));

  PetscCall(DMSwarmRestoreField(cellSwarm, "w_q", NULL, NULL, (void**)&weights));
//  PetscPrintf(PETSC_COMM_WORLD, "Collision step energy delta: %g\n", postStepEnergy - preStepEnergy);
  PetscCheck(PetscAbsReal(postStepEnergy - preStepEnergy) < tol, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Collision energy growth %g too large. Prestep %g PostStep %g", PetscAbsReal(postStepEnergy - preStepEnergy), preStepEnergy, postStepEnergy);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
static PetscErrorCode ComputeS(DM sw, Vec U, PetscScalar *S, void *ctx)
{ PetscReal         *weight, *ent;//, *velocity;
  Vec                entropy, wVec;
  const PetscScalar *velocity;
  PetscInt          *species, Np, dim;
  PetscReal kHermite[6] = {-2.3506049736745, -1.3358490740137, -0.43607741192762, 0.43607741192762, 1.3358490740137, 2.3506049736745};
  PetscReal wHermite[6] = {0.0045300099055088, 0.15706732032286, 0.72462959522439, 0.72462959522439, 0.15706732032286, 0.0045300099055088};
  collisionData            *user = (collisionData*)ctx;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &wVec));
  PetscCall(VecDuplicate(wVec, &entropy));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &wVec));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void**)&weight));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void**)&species));
  PetscCall(VecGetArrayRead(U, &velocity));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecZeroEntries(entropy));
  PetscCall(VecGetArray(entropy, &ent));
  for (PetscInt p = 0; p < Np; ++p){
    *S = 0.;
    for (PetscInt i=0; i < 6; i++){
      for (PetscInt j=0; j < 6; j++) {
        PetscReal logsum = 0, kpx, kpy, dx, dy, SQRT2EPSM1, PI2EPSM1;
        for (PetscInt q = 0; q < Np; ++q) {

          if (species[p] != species[q]) continue;
          SQRT2EPSM1 = 1./sqrt(2.*user->epsilon[species[q]]);
          PI2EPSM1 = 1./(2*PETSC_PI * user->epsilon[species[q]]);
          kpx = kHermite[i] + velocity[p*dim + 0]*SQRT2EPSM1;
          kpy = kHermite[j] + velocity[p*dim + 1]*SQRT2EPSM1;
          dx = kpx - velocity[q*dim+0] * SQRT2EPSM1;
          dy = kpy - velocity[q*dim+1] * SQRT2EPSM1;
          logsum += weight[q] * PetscExpReal(-dx*dx - dy*dy)*PI2EPSM1;
        }
        *S += 1./(PETSC_PI) * weight[p] * wHermite[i] * wHermite[j] * (PetscLogReal(logsum));
      }
    }
    ent[p] = -*S;
  }
  PetscCall(VecRestoreArray(entropy, &ent));
  PetscCall(VecSum(entropy, S));
  PetscCall(VecRestoreArrayRead(U, &velocity));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void**)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void**)&species));
  //PetscPrintf(PETSC_COMM_WORLD, "Entropy: :%g\n", *F);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CalculateMomentsAndTemperatures(DM sw, PetscReal* momentum, PetscReal *KE, PetscReal* T)
{
  collisionData        *user;
  PetscInt       Np, p, dim, d, cStart, cEnd, s;
  PetscInt      *species, Ns;
  PetscReal     *velocities, *weights, v_0;
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
  v_0 = PetscSqrtReal((8 * BOLTZMANN_K * user->T[0])/(user->masses[0]*PETSC_PI));
  for (p = 0; p < Np; ++p){
    PetscReal v2 = 0.;

    for (d=0; d < dim; ++d) momentum[species[p]*dim+d] += velocities[p*dim+d] * weights[p];
    for (d=0; d < dim; ++d) v2 += PetscSqr(velocities[p*dim+d]);
    KE[species[p]] +=  weights[p] * v2;
  }
  for (s=0; s < Ns; ++s){
    PetscReal    dimensionalization, udotu=0., dimratio;
    dimratio = 2./dim;
    dimensionalization  = (user->mass/BOLTZMANN_K);
    if (user->regular) dimensionalization *= PetscSqr(v_0);//
    else dimensionalization *= PetscSqr(user->v0[0]);
    T[s] = KE[s];
    for (d = 0; d < dim; ++d) udotu += PetscSqr(momentum[s*dim+d]);
    T[s] -= udotu;
    T[s] *= dimratio/2. * dimensionalization/1.16045250061657e7;
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocities));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &weights));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **) &species));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Monitor(TS ts)
{
  collisionData    *user;
  DM         sw;
  PetscReal *T, *KE, *mom, totKE=0., time, v_0;
  PetscInt   s, Ns, dim, steps, idx;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void **) &user));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(PetscCalloc3(Ns, &T, Ns, &KE, dim*Ns, &mom));
  PetscCall(CalculateMomentsAndTemperatures(sw, mom, KE, T));

  if (steps % user->outputNum == 0) PetscPrintf(PETSC_COMM_WORLD, "time: %g\n", time);
  for (s = 0; s < Ns; ++s){
    totKE += KE[s];
    if (steps % user->outputNum == 0){
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "momentumx[%"PetscInt_FMT"]: %g momentumy[%"PetscInt_FMT"]: %g KE[%"PetscInt_FMT"]: %g T[%"PetscInt_FMT"]: %g\n", s, mom[s*dim], s, mom[s*dim+1], s, KE[s], s, T[s]));
    }
  }
  // Record the deviation in kinetic energy
  if (steps == 0) user->total_energy = totKE;
  if (steps % user->outputNum == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total Energy: %g\n", PetscAbsReal(totKE - user->total_energy)/user->total_energy));
  /* Recompute epsilon based on new temperatures */
  for (idx = 0; idx < Ns; ++idx) T[idx] *= 1.16045250061657e7;
  for (idx = 0; idx < Ns; ++idx) user->v0[idx] = PetscSqrtReal(BOLTZMANN_K * T[idx] / user->masses[idx]);
  v_0     = PetscSqrtReal((8 * BOLTZMANN_K * T[0])/(user->masses[0]*PETSC_PI));
  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] = 5.*user->v0[idx]/v_0;
  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] /= user->Np;// commented out the above to use the regular configuration
  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] = PetscPowReal(user->epsilon[idx], 1.98);
  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] *= 1.2;
  PetscCall(PetscFree3(T, KE, mom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CalculateMomentsAndTemperatures_Anisotropic(DM sw, PetscReal* momentum, PetscReal *KE, PetscReal* T)
{
  collisionData        *user;
  PetscInt       Np, p, dim, d, cStart, cEnd, s;
  PetscInt      *species, Ns;
  PetscReal     *velocities, *weights, v_0;
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
  PetscReal Tavg = 0.;
  // The first dim entries are for species zero, which the global distribution is normalized to.
  for (d = 0; d < dim; ++d) Tavg = user->Tavg;
  Tavg /= dim;
  // This constant is based on the initializing value
  v_0 = PetscSqrtReal((8 * BOLTZMANN_K * Tavg)/(user->masses[0]*PETSC_PI));
  for (p = 0; p < Np; ++p){

    for (d=0; d < dim; ++d) momentum[species[p]*dim+d] += velocities[p*dim+d] * weights[p];
    for (d=0; d < dim; ++d) KE[species[p]*dim + d] += weights[p] * PetscSqr(velocities[p*dim+d]);
  }
  for (s=0; s < Ns; ++s){
    PetscReal    dimensionalization, udotu[3]={0.,0.,0.};
    dimensionalization  = (user->masses[s]/BOLTZMANN_K);
    if (user->regular) dimensionalization *= PetscSqr(v_0);//
    else dimensionalization *= PetscSqr(user->v0[0]);
    for (d = 0; d < dim; ++d) T[s*dim + d] = KE[s*dim+d];

    for (d = 0; d < dim; ++d) udotu[d] += PetscSqr(momentum[s*dim+d]);
    for (d = 0; d < dim; ++d) T[s*dim+d] -= udotu[d];
    for (d = 0; d < dim; ++d) T[s*dim+d] *= dim/2. * dimensionalization/1.16045250061657e7;
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocities));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &weights));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **) &species));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Monitor_Anisotropic(TS ts)
{
  collisionData      *user;
  DM          sw;
  PetscScalar S;
  Vec         sol;
  PetscReal   *T, *KE, *mom, totKE=0., time;
  PetscInt    s, Ns, dim, steps, idx;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void **) &user));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(PetscCalloc3(Ns*dim, &T, Ns*dim, &KE, dim*Ns, &mom));
  PetscCall(CalculateMomentsAndTemperatures_Anisotropic(sw, mom, KE, T));
  if (steps % user->outputNum == 0) PetscPrintf(PETSC_COMM_WORLD, "time: %g\n", time*user->t_0);
  for (s = 0; s < Ns; ++s){
    totKE += KE[s];
    if (steps % user->outputNum == 0){
      for (PetscInt d = 0; d < dim; ++d){
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "T%"PetscInt_FMT"[%"PetscInt_FMT"]: %g\n", d, s, T[s*dim + d]));
      }
    }
  }
  if (steps == 0) user->total_energy = totKE;
  if (steps % user->outputNum == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total Energy: %g\n", PetscAbsReal(totKE - user->total_energy)/user->total_energy));
  /* Recompute epsilon based on new temperatures. Use the average temperature of the species to get a good epsilon */
  PetscReal Tavg[2]={0.,0.};
  for (idx = 0; idx < Ns; ++idx) {
    for (PetscInt d = 0; d < dim; ++d) Tavg[idx] += T[idx*dim + d];
    Tavg[idx] /= dim;
  }
  PetscReal v_0;
  for (idx = 0; idx < Ns; ++idx) user->v0[idx] = PetscSqrtReal(BOLTZMANN_K * Tavg[idx] / user->masses[idx]);
  //v_0     = PetscSqrtReal((8 * BOLTZMANN_K * Tavg[0])/(user->masses[0]*PETSC_PI));

  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] = 5.*user->v0[idx]/user->v0[0];

  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] /= user->Np;// commented out the above to use the regular configuration
  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] = PetscPowReal(user->epsilon[idx], 1.98);
  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] *= 1.2;
  if (user->run_nrl) {
    PetscReal          dt_real, dt;
    PetscCall(TSGetTimeStep(ts, &dt)); // dt for NEXT time step
    dt_real = dt * user->t_0;
    PetscCall(TSSetTimeStep(user->ts_nrl, dt_real));
    PetscCall(TSSetMaxSteps(user->ts_nrl, steps + 1)); // next step
    PetscCall(TSSolve(user->ts_nrl, NULL));
  }
  PetscCall(TSGetSolution(ts, &sol));
  PetscCall(ComputeS(sw, sol, &S, user));
  if (steps > 0) {
    if (steps % user->outputNum == 0){
      PetscPrintf(PETSC_COMM_WORLD, "deltaS: %.16g\n", S - user->S_init );
      PetscPrintf(PETSC_COMM_WORLD, "Final S: %.16g\n", S);
    }
  }
  else{
    PetscPrintf(PETSC_COMM_WORLD, "deltaS: 0.0\n" );
    user->S_init = S;
  }
  // Views will be logarithmic, so just update to the next data point
  PetscPrintf(PETSC_COMM_WORLD, "ES1: %g, ES2: %g\n", user->epsilon[0], user->epsilon[1]);
  PetscCall(PetscFree3(T, KE, mom));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  collisionData* user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocity));
  PetscCall(TSGetSolution(ts, &sol));
  PetscCall(VecGetLocalSize(sol, &n));
  PetscCall(VecGetArrayRead(sol, &u));
  for (idx = 0; idx < n; ++idx) velocity[idx] = u[idx];
  PetscCall(VecRestoreArrayRead(sol, &u));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity));
  if (!user->anisotropic) PetscCall(Monitor(ts));
  else PetscCall(Monitor_Anisotropic(ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
