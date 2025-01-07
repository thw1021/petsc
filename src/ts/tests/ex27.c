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

/* Some useful constants */
#define BOLTZMANN_K 1.380649e-23 /* J/K */
#define KEV_J 6.241506479963235e15 /*  */
#define LIGHT_C 299792458
#define EPSILON_NOUGHT 8.8542e-12
#define ELEMENTARY_CHARGE 1.602176e-19
#define ELECTRON_MASS 9.10938356e-31
#define PROTON_MASS 1.6726219e-27

typedef struct {
  TS        ts_nrl;                         /* Additional TS for NRL calculations */
  PetscInt  steps;                          /* Number of time steps */
  PetscReal step_size;                      /* Size of the time step */
  PetscReal gaussian_w;                     /* Width of quadrature evaulation on gaussian mollifiers */
  PetscReal epsilon[LANDAU_MAX_SPECIES];                        /* Gaussian regularization parameter */
  PetscReal t_0;                            /* time nondimensionalization */
  PetscInt  mass_units[LANDAU_MAX_SPECIES]; /* 0 for electron mass units, 1 for proton mass units */
  PetscReal masses[LANDAU_MAX_SPECIES];     /* Electron, Sr+ Mass [kg] */
  PetscReal T[LANDAU_MAX_SPECIES];        /* Electron, Ion Temperature [K] */
  PetscReal v0[LANDAU_MAX_SPECIES];         /* Species mean velocity in 1D */
  PetscReal n0[LANDAU_MAX_SPECIES];
  PetscReal charges[LANDAU_MAX_SPECIES];
  PetscReal total_energy;                    /* Cache total energy for computation */
  PetscBool regular;
  PetscBool anisotropic;                      /* maxwellians are anisotropic*/
  PetscInt  Np;
  PetscReal Tavg;
  PetscInt  Ns;
  PetscBool spitzer;                         // flag to test spitzer resistivity or not
  PetscReal E;                               // The E field in Connor-Hastie
  PetscInt  order;
  PetscReal qrad;
  PetscReal  lnLam;
  PetscBool  run_nrl;
  PetscReal  S_init;
  PetscInt   outputNum;
  PetscInt   dimension;
} AppCtx;

 /* CalculateE - Calculate the electric field  */
 /*  T        -- Electron temperature  */
 /*  n        -- Electron density  */
 /*  lnLambda --   */
 /*  eps0     --  */
 /*  E        -- output E, input \hat E */
static PetscReal CalculateE(PetscReal Tev, PetscReal n, PetscReal lnLambda, PetscReal eps0, PetscReal *E)
{
  PetscReal c,e,m;

  PetscFunctionBegin;
  c = 299792458.0;
  e = 1.602176e-19;
  m = 9.10938e-31;
  if (1) {
    double Ec, Ehat = *E, betath = PetscSqrtReal(2*Tev*e/(m*c*c)), j0 = Ehat * 7/(PetscSqrtReal(2)*2) * PetscPowReal(betath,3) * n * e * c;
    Ec = n*lnLambda*PetscPowReal(e,3) / (4*PETSC_PI*PetscPowReal(eps0,2)*m*c*c);
    *E = Ec;
    PetscPrintf(PETSC_COMM_WORLD, "CalculateE j0=%g Ec = %g\n",j0,Ec);
  } else {
    PetscReal Ed, vth;
    vth = PetscSqrtReal(8*Tev*e/(m*PETSC_PI));
    Ed =  n*lnLambda*PetscPowReal(e,3) / (4*PETSC_PI*PetscPowReal(eps0,2)*m*vth*vth);
    *E = Ed;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt   nn0=LANDAU_MAX_SPECIES, nT=LANDAU_MAX_SPECIES;
  PetscInt   nmu=LANDAU_MAX_SPECIES, nm=LANDAU_MAX_SPECIES;
  PetscInt   nc=LANDAU_MAX_SPECIES;
  PetscBool  nmuflg, Tflg, cflg;

  PetscFunctionBeginUser;
  options->gaussian_w       = -1.;
  options->step_size        = 0.1;
  options->steps            = 1;
  options->epsilon[0]       = 1.9;
  options->epsilon[1]       = 1.9;
  options->T[0]             = 5*1.16045250061657e7; /* 5kev converted to kelvin */
  options->T[1]             = 5*1.16045250061657e7; /* 5kev converted to kelvin */
  options->masses[0]        = ELECTRON_MASS;
  options->masses[1]        = ELECTRON_MASS;
  options->mass_units[0]    = 0;
  options->mass_units[1]    = 0;
  options->charges[0]       = -ELEMENTARY_CHARGE;
  options->charges[1]       = -ELEMENTARY_CHARGE;
  options->n0[0]            = 1.0e20;
  options->n0[1]            = 1.0e20;
  options->regular          = PETSC_FALSE;
  options->Np               = 20;
  options->Ns               = 2;
  options->spitzer          = PETSC_FALSE;
  options->E                = 0.1;
  options->anisotropic      = PETSC_FALSE;
  options->run_nrl          = PETSC_FALSE;
  options->lnLam            = 10.;
  options->outputNum        = 1;

  PetscOptionsBegin(comm, "", "Collision Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-steps", "max number of time steps to take", "ex27.c", options->steps, &options->steps, NULL));
  PetscCall(PetscOptionsInt("-order", "quadrature order", "ex27.c", options->order, &options->order, NULL));
  PetscCall(PetscOptionsInt("-Np", "In the regular case, particles per velocity dimension on a grid of Np^d", "ex27.c", options->Np, &options->Np, NULL));
  PetscCall(PetscOptionsInt("-output_step", "step to output monitor", "ex27.c", options->outputNum, &options->outputNum, NULL));
  PetscCall(PetscOptionsBool("-regular", "Layout particles in a grid", "ex27.c", options->regular, &options->regular, NULL));
  PetscCall(PetscOptionsBool("-run_nrl", "Compute relaxation from NRL", "ex27.c", options->run_nrl, &options->run_nrl, NULL));
  PetscCall(PetscOptionsBool("-anisotropic", "Anisotropic initialization", "ex27.c", options->anisotropic, &options->anisotropic, NULL));
  PetscCall(PetscOptionsBool("-spitzer", "Run the spitzer resititivity test", "ex27.c", options->spitzer, &options->spitzer, NULL));
  PetscCall(PetscOptionsReal("-E", "E field in Connor-Hastie", "ex27.c", options->E, &options->E, NULL));
  PetscCall(PetscOptionsReal("-step_size", "size of the time step", "ex27.c", options->step_size, &options->step_size, NULL));
  PetscCall(PetscOptionsReal("-gaussian_width", "Width of entropy gradient quadrature evaluation", "ex27.c", options->gaussian_w, &options->gaussian_w, NULL));
  PetscCall(PetscOptionsRealArray("-dm_swarm_number_density", "The non normalized number density of each species", "ex27.c", options->n0, &nn0, NULL));
  PetscCall(PetscOptionsRealArray("-dm_swarm_temperature", "The temperature of each species in KeV", "ex27.c", options->T, &nT, &Tflg));
  PetscCall(PetscOptionsIntArray("-dm_swarm_mass_units", "0 for electron 1 for proton mass", "ex27.c", options->mass_units, &nmu, &nmuflg));
  PetscCall(PetscOptionsRealArray("-dm_swarm_masses", "The mass of each species in multiples of fundamental mass units", "ex27.c", options->masses, &nm, NULL));
  PetscCall(PetscOptionsRealArray("-dm_swarm_charges", "The charge of each species in fundamental charge units (-1,2,3...)", "ex27.c", options->charges, &nc, &cflg));
  PetscOptionsEnd();

  /* If mass units were specified, get the mass array and compute masses */
  PetscInt dim;
  PetscCall(PetscOptionsGetInt(NULL, "", "-dm_plex_dim", &dim, NULL));
  if (nmuflg) {
    PetscInt idx;
    PetscCheck(nm == nmu, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Number of mass units and number of masses given are not equal.");
    for (idx = 0; idx < nmu; ++idx) options->masses[idx] = options->mass_units[idx] == 0 ? ELECTRON_MASS*options->masses[idx] : PROTON_MASS*options->masses[idx];
  }
  if (Tflg) {
    PetscInt idx;
    PetscReal v_0, Tavg[2]={0.,0.};
    for (idx = 0; idx < nT; ++idx) options->T[idx] *= 1.1604525e7;

    /* Calculate the average temperature ie (Tx+Ty+Tz)/3 */
    if (options->anisotropic) {
      for (idx = 0; idx < nT/dim; ++idx) Tavg[0] += options->T[idx];
      for (idx = dim; idx < nT; ++idx) Tavg[1] += options->T[idx];
      for (idx = 0; idx < nT/dim; ++idx) options->v0[idx] = PetscSqrtReal(BOLTZMANN_K * (Tavg[idx]/dim) / options->masses[idx]);
      v_0     = PetscSqrtReal((8 * BOLTZMANN_K * Tavg[0]/dim)/(options->masses[0]*PETSC_PI));
      options->Tavg = Tavg[0];
    }
    else{
      for (idx = 0; idx < nT; ++idx) options->v0[idx] = PetscSqrtReal(BOLTZMANN_K * options->T[idx] / options->masses[idx]);
      v_0     = PetscSqrtReal((8 * BOLTZMANN_K * options->T[0])/(options->masses[0]*PETSC_PI));
    }

    if (options->regular){
      if (options->anisotropic){
        for (idx = 0; idx < nT/dim; ++idx) options->epsilon[idx] = 5.*options->v0[idx]/v_0;
        for (idx = 0; idx < nT/dim; ++idx) options->epsilon[idx] /= options->Np;
      }
      else{
      /* this should only compute 1 epsilon per species, so we will choose is based on v_0x normalized to electron x thermal velocity */
        for (idx = 0; idx < nT; ++idx) options->epsilon[idx] = 5.*options->v0[idx]/v_0;
        for (idx = 0; idx < nT; ++idx) options->epsilon[idx] /= options->Np;
      }
    }
    else{
      for (idx = 0; idx < nT; ++idx) options->epsilon[idx] = 5.*options->v0[idx]/options->v0[0];
      for (idx = 0; idx < nT; ++idx) options->epsilon[idx] /= 10;
    }
    for (idx = 0; idx < nT; ++idx) options->epsilon[idx] = PetscPowReal(options->epsilon[idx], 1.98);
    for (idx = 0; idx < nT; ++idx) options->epsilon[idx] *= 1.2;
  }
  if (cflg) {
    PetscInt idx;
    for (idx = 0; idx < nc; ++idx) options->charges[idx] *= ELEMENTARY_CHARGE;
    for (idx = 0; idx < nc; ++idx) PetscPrintf(PETSC_COMM_WORLD, "charge of species %i: %g\n", idx, options->charges[idx]);

  }
  if (!options->anisotropic) for (PetscInt idx = 0; idx < nT; ++idx) PetscPrintf(PETSC_COMM_WORLD, "Initial target temperature of species %i: %g\n", idx, options->T[idx]/1.1604525e7);
  else {
    for (PetscInt idx = 0; idx < dim; ++idx) PetscPrintf(PETSC_COMM_WORLD, "Initial target temperature of species 0 in dimension %i: %g\n", idx, options->T[idx]/1.1604525e7);
    for (PetscInt idx = dim; idx < 2*dim; ++idx) PetscPrintf(PETSC_COMM_WORLD, "Initial target temperature of species 1 in dimension %i: %g\n", idx-dim, options->T[idx]/1.1604525e7);
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "masses[0]: %g, masses[1]: %g\n", options->masses[0], options->masses[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "epsilon[0]: %g, epsilon[1]: %g\n", options->epsilon[0], options->epsilon[1]));
  if (options->spitzer){
    CalculateE(options->T[0], options->n0[0], 10, EPSILON_NOUGHT, &options->E);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscInt       dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  user->dimension = dim;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmSetVelocitiesUniformCoordinates - Set velocity field uniformly in each dimension w/o an additional mesh

  Alteration of DMSwarmSetPointsUniformCoordinates. Assumes the field "velocity" as been registered by the user

  Input parameters:
+  sw - the DMSwarm
.  species - The species to set the velocities for
.  min - minimum coordinate values in the x, y, z directions (array of length dim)
.  max - maximum coordinate values in the x, y, z directions (array of length dim)
-  npoints - number of points in each spatial direction (array of length dim)

Note: Assumes equal number of particles per species
*/
PetscErrorCode DMSwarmSetVelocitiesUniformCoordinates(DM sw, PetscInt species, PetscReal min[], PetscReal max[], PetscInt npoints[], PetscReal* h)
{
  PetscInt           i, j, k, bs, b, n_estimate, p, Np, Ns, dim;
  PetscReal          dx[3];
  PetscInt           _npoints[] = {0, 0, 1};
  Vec                pos;
  PetscReal         *_pos;
  PetscReal         *velocity;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  // for cellwise initialization, this needs to be changed to do this based
  // on the cellwise size, not the total local size.
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  // total is over species. Different particle species can have different sized grids.
  Np = 1;
  for (PetscInt d = 0; d < dim; ++d) Np *= npoints[d];
  if (species < 0 || species > Ns) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Species specified out of range %" PetscInt_FMT, species);
  bs = dim;
  for (b = 0; b < bs; b++) {
    if (npoints[b] > 1) {
      dx[b] = (max[b] - min[b]) / ((PetscReal)(npoints[b] - 1));
    } else {
      dx[b] = 0.0;
    }
    _npoints[b] = npoints[b];
  }
  *h = dx[0];
  PetscCall(VecCreate(PETSC_COMM_SELF, &pos));
  PetscCall(VecSetSizes(pos, bs * Np, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(pos, bs));
  PetscCall(VecSetFromOptions(pos));
  PetscCall(VecGetArray(pos, &_pos));

  n_estimate = 0;
  for (k = 0; k < _npoints[2]; k++) {
    for (j = 0; j < _npoints[1]; j++) {
      for (i = 0; i < _npoints[0]; i++) {
        PetscReal xp[] = {0.0, 0.0, 0.0};
        PetscInt  ijk[3];

        ijk[0] = i;
        ijk[1] = j;
        ijk[2] = k;
        for (b = 0; b < bs; b++) xp[b] = min[b] + ijk[b] * dx[b];
        for (b = 0; b < bs; b++) _pos[bs * n_estimate + b] = xp[b];
        n_estimate++;

      }
    }
  }
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&velocity));
  for (p = 0; p < n_estimate; p++) {
    for (b = 0; b < bs; b++) {
      velocity[species*Np*bs + bs * p + b] = PetscRealPart(_pos[bs * p + b]);
      //PetscPrintf(PETSC_COMM_WORLD, "Offset for particles in configure %"PetscInt_FMT"\n", species*Np*bs + bs * p + b);
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&velocity));
  PetscCall(VecRestoreArray(pos, &_pos));
  PetscCall(VecDestroy(&pos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeWeights(PetscInt Np, PetscInt dim, PetscInt off, PetscReal h, PetscReal theta[], PetscReal* v, PetscReal* w){
  PetscReal scale;

  PetscFunctionBegin;
  scale = dim == 2 ? 0.25 : 0.125;
  PetscPrintf(PETSC_COMM_WORLD, "Np %"PetscInt_FMT" spacing %g\n", Np, h);
  for (PetscInt p = 0; p < Np; ++p) {
    PetscReal a, b, c, d, e, f, x, y, z;

    a = v[off*dim + p*dim] - h/2.;
    b = v[off*dim + p*dim] + h/2.;
    c = v[off*dim + p*dim + 1] - h/2.;
    d = v[off*dim + p*dim + 1] + h/2.;
    if (dim == 3) {
      e = v[off*dim + p*dim + 2] - h/2.;
      f = v[off*dim + p*dim + 2] + h/2.;
    }
    x = PetscErfReal(b/PetscSqrtReal(theta[0])) - PetscErfReal(a/PetscSqrtReal(theta[0]));
    y = PetscErfReal(d/PetscSqrtReal(theta[1])) - PetscErfReal(c/PetscSqrtReal(theta[1]));
    z = dim == 3 ? PetscErfReal(f/PetscSqrtReal(theta[2])) - PetscErfReal(e/PetscSqrtReal(theta[2])) : 1.;
    w[off + p] = scale * x * y * z;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureRegularSwarm(DM plex, AppCtx* user, DM* sw){
  PetscReal     *vals, *vels, theta_e[3], theta_i[3];
  PetscInt      *spec, Np, Ns = user->Ns, p, dim;
  PetscReal vt_e, vt_i, h_e, h_i, L_e, L_i, v_0;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(plex, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject) plex), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMGetDimension(plex, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject) plex), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, plex));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "gradS", dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  // Np on the input is the number of particles in each grid dimension, not the total.
  switch (dim){
    case 2:
      Np = user->Np * user->Np;
      break;
    case 3:
      Np = user->Np * user->Np * user->Np;
      break;
  }
  // num species doesn't get called right, use from cl options
  PetscCall(DMSwarmSetLocalSizes(*sw, Np * Ns, 0));
  PetscCall(DMSwarmSetNumSpecies(*sw, Ns));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **) &vals));
  PetscCall(DMSwarmGetField(*sw, "species", NULL, NULL, (void **) &spec));
  for (p = 0; p < Np; ++p){
    spec[p] = 0;
    spec[Np + p] = 1;
  }
  if (!user->anisotropic) v_0  = PetscSqrtReal((8 * BOLTZMANN_K * user->T[0])/(user->masses[0]*PETSC_PI));
  else{
    PetscReal Tavg = 0.;
    for (PetscInt d = 0; d < dim; ++d) Tavg += user->T[d];
    Tavg /= dim;
    v_0  = PetscSqrtReal((8 * BOLTZMANN_K * Tavg)/(user->masses[0]*PETSC_PI));
  }

  vt_e = user->v0[0]/v_0;
  vt_i = user->v0[1]/v_0; // normalize to electron thermal velocity


  // h = 2L/N in domain [-L, L]^2 and N = sqrt(Np)
  // 5v_t has been found to be a good domain (Fillipo's observation)
  // get h back from the uniform coordinate grid
  //h_e = 2*(5.*vt_e)*PetscPowReal(user->Np, -1./dim);// sqrt for 2d, cube root for 3d for x by y by z
  //h_i = 2*(5.*vt_i)*PetscPowReal(user->Np, -1./dim);// sqrt for 2d, cube root for 3d for x by y by z

  L_e = 5*vt_e;
  L_i = 5*vt_i;
  PetscReal dlow_e[3], dhigh_e[3], dlow_i[3], dhigh_i[3];
  PetscInt npoints[3];
  for (PetscInt d = 0; d < dim; ++d) {
    dlow_e[d] = -L_e; dhigh_e[d] = L_e;
    dlow_i[d] = -L_i; dhigh_i[d] = L_i;
    npoints[d] = user->Np;
  }
  // get n points from the input Np, in the regular case, square or cube it for total Np.
  PetscCall(DMSwarmSetVelocitiesUniformCoordinates(*sw, 0, dlow_e, dhigh_e, npoints, &h_e));//PetscCall(LayoutGrid_2D(user->Np, 0, h_e, L_e, vels));
  PetscCall(DMSwarmSetVelocitiesUniformCoordinates(*sw, 1, dlow_i, dhigh_i, npoints, &h_i));

  if (!user->anisotropic) {
    theta_e[0] = theta_e[1] = theta_e[2] = (2. * BOLTZMANN_K * user->T[0])/(user->masses[0] * PetscSqr(v_0));
    theta_i[0] = theta_i[1] = theta_i[2] = (2. * BOLTZMANN_K * user->T[1])/(user->masses[1] * PetscSqr(v_0));
  }
  else {
    for (PetscInt d = 0; d < dim; ++d) {
      theta_e[d] = (2. * BOLTZMANN_K * user->T[d])/(user->masses[0] * PetscSqr(v_0));
      theta_i[d] = (2. * BOLTZMANN_K * user->T[dim + d])/(user->masses[1] * PetscSqr(v_0));
    }
  }
  // this it total number need to change the name of Np here and user->np
  PetscCall(DMSwarmGetField(*sw, "velocity", NULL, NULL, (void **) &vels));
  PetscCall(ComputeWeights(Np, dim, 0, h_e, theta_e, vels, vals));
  PetscCall(ComputeWeights(Np, dim, Np, h_i, theta_i, vels, vals));
  PetscReal esum=0., isum=0.;
  for (p = 0; p < Np; ++p){
    esum += vals[p];
    isum += vals[Np+p];
  }
  PetscPrintf(PETSC_COMM_WORLD, "esum: %g, isum: %g\n", esum, isum);
  for (p = 0; p < Np; ++p){
    vals[p] /= esum;
    vals[Np+p] /= isum;
  }
  PetscCall(DMSwarmRestoreField(*sw, "species", NULL, NULL, (void **) &spec));
  PetscCall(DMSwarmRestoreField(*sw, "velocity", NULL, NULL, (void **) &vels));
  PetscCall(DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **) &vals));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Configure the swarm based on regular grid or Klemontovich representation. */
static PetscErrorCode SetupProb(MPI_Comm comm, DM* plex, DM* swarm, AppCtx* user)
{

  PetscFunctionBeginUser;
  PetscCall(CreateMesh(comm, plex, user));
  if (!user->regular) PetscCall(CreateSwarm(*plex, user, swarm));
  else PetscCall(ConfigureRegularSwarm(*plex, user, swarm));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  ComputeGradS - Compute grad_v dS_eps/df

  Input Parameters:
+ dim      - The dimension
. Np       - The number of particles
. velocity - The velocity field for all particles
. epsilon  - The regularization strength
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

static PetscErrorCode ComputeGradS(PetscReal* weight, PetscInt *species, PetscInt Np, PetscReal velocity[], PetscReal integral[], AppCtx *ctx)
{
  PetscInt  nHermite=6;
  PetscInt  debug = 0;

  PetscReal kHermite[6] = {-2.3506049736745, -1.3358490740137, -0.43607741192762, 0.43607741192762, 1.3358490740137, 2.3506049736745};
  PetscReal wHermite[6] = {0.0045300099055088, 0.15706732032286, 0.72462959522439, 0.72462959522439, 0.15706732032286, 0.0045300099055088};


  PetscFunctionBeginHot;
  #pragma omp parallel for
  for (PetscInt p = 0; p < Np; ++p){
    PetscInt start, end;
    PetscReal SQRT2EPSM1, PI2EPSM1, coeff;
    // Shift start and end so we only consider same species particles
    start = species[p] == 0 ? 0 : Np/2;
    end = species[p] == 0 ? Np/2 : Np;
    SQRT2EPSM1 = 1./PetscSqrtReal(2.*ctx->epsilon[species[p]]);
    PI2EPSM1 = 1./(2*PETSC_PI * ctx->epsilon[species[p]]);

    for (PetscInt d = 0; d < 2; ++d) integral[p*2+d] = 0.0;

    for (PetscInt i=0; i < nHermite; i++){
      for (PetscInt j=0; j < nHermite; j++) {
        PetscReal logsum = 0, kpx, kpy, dx, dy;

        for (PetscInt q = start; q < end; ++q) {

          if (species[p]!=species[q]) continue;

          kpx = kHermite[i] + velocity[p*2+0] * SQRT2EPSM1;
          kpy = kHermite[j] + velocity[p*2+1] * SQRT2EPSM1;
          dx = kpx - velocity[q*2+0] * SQRT2EPSM1;
          dy = kpy - velocity[q*2+1] * SQRT2EPSM1;
          logsum += weight[q] * PetscExpReal(-PetscSqr(dx) - PetscSqr(dy)) * PI2EPSM1;
        }
        logsum = wHermite[i]*wHermite[j]*(1. + PetscLogReal(logsum));
        integral[p*2+0] += logsum * kHermite[i];
        integral[p*2+1] += logsum * kHermite[j];
      }
    }
    coeff = -ctx->masses[0]/ctx->masses[species[p]] * PetscSqrtReal(2. * ctx->epsilon[species[p]]) / (PETSC_PI * ctx->epsilon[species[p]]);
    integral[p*2+0] *= coeff;
    integral[p*2+1] *= coeff;
  }
  if (debug) for (PetscInt p = 0; p < Np; ++p ) PetscPrintf(PETSC_COMM_WORLD, "Particle %"PetscInt_FMT" integral %g %g species %"PetscInt_FMT" eps %g\n", p, integral[p*2+0], integral[p*2+1], species[p], ctx->epsilon[species[p]]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeGradS_3D(PetscReal* weight, PetscInt *species, PetscInt Np, PetscReal velocity[], PetscReal integral[], AppCtx *ctx)
{
  PetscInt  nHermite=6;

  PetscReal kHermite[6] = {-2.3506049736745, -1.3358490740137, -0.43607741192762, 0.43607741192762, 1.3358490740137, 2.3506049736745};
  PetscReal wHermite[6] = {0.0045300099055088, 0.15706732032286, 0.72462959522439, 0.72462959522439, 0.15706732032286, 0.0045300099055088};

  PetscFunctionBeginHot;

  #pragma omp parallel for
  for (PetscInt p = 0; p < Np; ++p){
    PetscInt start, end;
    PetscReal SQRT2EPSM1, PI2EPSM1;
    PetscReal coeff;
    // Shift start and end so we only consider same species particles
    start = species[p] == 0 ? 0 : Np/2;
    end = species[p] == 0 ? Np/2 : Np;
    SQRT2EPSM1 = 1./PetscSqrtReal(2.*ctx->epsilon[species[p]]);
    PI2EPSM1 = 1./PetscPowReal(2*PETSC_PI * ctx->epsilon[species[p]], 1.5);
    for (PetscInt d = 0; d < 3; ++d) integral[p*3+d] = 0.0;

    for (PetscInt i=0; i < nHermite; i++) {
      for (PetscInt j=0; j < nHermite; j++) {
        for (PetscInt k=0; k < nHermite; k++) {
          PetscReal logsum = 0, kpx, kpy, kpz, dx, dy, dz;

          for (PetscInt q = start; q < end; ++q) {

            if (species[p]!= species[q]) continue;
            kpx = kHermite[i] + velocity[p*3+0] * SQRT2EPSM1;
            kpy = kHermite[j] + velocity[p*3+1] * SQRT2EPSM1;
            kpz = kHermite[k] + velocity[p*3+2] * SQRT2EPSM1;
            dx = kpx - velocity[q*3+0] * SQRT2EPSM1;
            dy = kpy - velocity[q*3+1] * SQRT2EPSM1;
            dz = kpz - velocity[q*3+2] * SQRT2EPSM1;
            logsum += weight[q] * PetscExpReal(-dx*dx - dy*dy - dz*dz) * PI2EPSM1;
          }
          logsum = wHermite[i]*wHermite[j]*wHermite[k]*(1. + PetscLogReal(logsum));
          // in 3v logsum should be whermite_ijk
          integral[p*3+0] += logsum * kHermite[i];
          integral[p*3+1] += logsum * kHermite[j];
          integral[p*3+2] += logsum * kHermite[k];
        }
      }
    }
    // the w_p term here is cancelled out by the \Gamma(S, p, \bar p) which has 1/mw_p
    coeff = -ctx->masses[0]/ctx->masses[species[p]] * PetscSqrtReal(2. * ctx->epsilon[species[p]]) / (PETSC_PI*ctx->epsilon[species[p]]);//should be to the 3/2 on the pi epsilon because of the 3rd dimension regularization
    integral[p*3+0] *= coeff;
    integral[p*3+1] *= coeff;
    integral[p*3+2] *= coeff;
  }
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

static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  PetscInt           dbg  = 0;
  DM                 sw;                  /* Particles */
  const PetscScalar *u;                   /* input solution vector */
  PetscScalar       *r;
  PetscReal         *gradS, *weight;
  PetscReal          lnLam=10., t0, nu_nd, m0=user->masses[0], v_0, nu_ee, nu_ei, nu_ii;
  PetscInt           dim, d, Np, *species, Ns;

  PetscFunctionBeginUser;

  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  /* Non dimensionalization of \nu, todo: nondimensionalization to be moved out of the solver into a part of swarm in future updates. */
  v_0 = PetscSqrtReal((8 * BOLTZMANN_K * user->T[0])/(user->masses[0]*PETSC_PI));

  //t0 = 8*PETSC_PI*PetscSqr(EPSILON_NOUGHT*m0/PetscSqr(user->charges[0]))/ lnLam/user->n0[0]*PetscPowReal(v_0,3);
  t0 = (8 * PETSC_PI * PetscSqr(m0) * PetscSqr(EPSILON_NOUGHT) * PetscPowReal(v_0, 3))/(PetscPowReal(user->charges[0], 4) * lnLam * user->n0[0]);
  user->t_0 = t0;
  nu_nd = t0*user->n0[0]/PetscPowReal(v_0,3.);

  nu_ee = nu_nd * (PetscPowReal(user->charges[0], 4) * lnLam/(8*PETSC_PI*PetscSqr(m0) * PetscSqr(EPSILON_NOUGHT)));
  nu_ei = nu_nd * (PetscSqr(user->charges[0]) * PetscSqr(user->charges[1]) * lnLam/(8*PETSC_PI*PetscSqr(m0) * PetscSqr(EPSILON_NOUGHT)));
  nu_ii = nu_nd * (PetscPowReal(user->charges[1], 4) * lnLam/(8*PETSC_PI*PetscSqr(m0) * PetscSqr(EPSILON_NOUGHT)));

  if (dbg) PetscPrintf(PETSC_COMM_WORLD, "nuee %g nuei %g nuii %g\n", nu_ee, nu_ei, nu_ii);

  PetscCall(VecZeroEntries(R));
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
  #pragma omp parallel for
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt q = 0; q < Np; ++q) {
      PetscReal GammaS[3] = {0., 0., 0.}, Q[9], nu;
      PetscReal residual[3] = {0., 0.,0.};

      if (q == p) continue;

      if ((species[p] == 0) & (species[q] == 0)) nu = nu_ee;
      if (species[p] != species[q]) nu = nu_ei;
      if ((species[p] == 1) & (species[q] == 1)) nu = nu_ii;

      DMPlex_WaxpyD_Internal(dim, -1.0, (const PetscReal*)&gradS[q*dim], (const PetscReal*)&gradS[p*dim], GammaS);
      // This has 1/mw_p applied at the computation of \nabla_v_p S in ComputeGammaS(..)
      QCompute(dim, &u[p*dim], &u[q*dim], Q);

      switch (dim) {
        case 2: DMPlex_MultAdd2DReal_Internal(Q, 1, GammaS, residual);break;
        case 3: DMPlex_MultAdd3DReal_Internal(Q, 1, GammaS, residual);break;
      }
      for(d=0; d < dim; ++d) r[p*dim+d] += residual[d] * nu * weight[q] * m0/user->masses[species[p]];
    }
    if (dbg) PetscPrintf(PETSC_COMM_WORLD, "Final %4" PetscInt_FMT " %10.8lf %10.8lf\n", p, r[p*dim+0], r[p*dim+1]);
  }
  PetscCall(DMSwarmRestoreField(sw, "gradS", NULL, NULL, (void **) &gradS));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **) &species));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &weight));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(R, &r));
  PetscCall(VecViewFromOptions(R, NULL, "-residual_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeIntGradS(TS ts)
{
  PetscInt       Np, dim, *species;
  PetscReal     *gradS, *velocity, *weights;
  DM             sw;
  Vec            sol;
  PetscInt       dbg=0;
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
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **) &species));
  switch (dim){
    case 2: PetscCall(ComputeGradS(weights, species, Np, velocity, gradS, user));break;
    case 3: PetscCall(ComputeGradS_3D(weights, species, Np, velocity, gradS, user));break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Do not support dimension%" PetscInt_FMT, dim);
  }
  if (dbg) for (PetscInt p = 0; p < Np; ++p) PetscPrintf(PETSC_COMM_WORLD, "gradient for particle %"PetscInt_FMT" %g %g\n", p, gradS[p*2+0], gradS[p*2+1]);
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **) &species));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &weights));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity));
  PetscCall(DMSwarmRestoreField(sw, "gradS", NULL, NULL, (void **) &gradS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeS(DM sw, Vec U, PetscScalar *S, void *ctx)
{ PetscReal         *weight, *ent;//, *velocity;
  Vec                entropy, wVec;
  const PetscScalar *velocity;
  PetscInt          *species, Np, dim;
  PetscReal kHermite[6] = {-2.3506049736745, -1.3358490740137, -0.43607741192762, 0.43607741192762, 1.3358490740137, 2.3506049736745};
  PetscReal wHermite[6] = {0.0045300099055088, 0.15706732032286, 0.72462959522439, 0.72462959522439, 0.15706732032286, 0.0045300099055088};
  AppCtx            *user = (AppCtx*)ctx;

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
  AppCtx        *user;
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
    dimensionalization  = (user->masses[s]/BOLTZMANN_K);
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
  AppCtx    *user;
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
  AppCtx        *user;
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
  AppCtx      *user;
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
  v_0     = PetscSqrtReal((8 * BOLTZMANN_K * Tavg[0])/(user->masses[0]*PETSC_PI));

  for (idx = 0; idx < Ns; ++idx) user->epsilon[idx] = 5.*user->v0[idx]/v_0;

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
  AppCtx* user;

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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Ripped from Mark */
static PetscReal n_cm3[2] = {0, 0};
typedef enum {
  E_PAR_IDX,
  E_PERP_IDX,
  I_PAR_IDX,
  I_PERP_IDX,
  NUM_TEMPS
} TemperatureIDX;
static PetscErrorCode   IsotropizationFunctionNRL(TS ts, PetscReal tdummy, Vec X, Vec F, void *actx)
{
  AppCtx         *ctx = (AppCtx *)actx; /* user-defined application context */
  PetscScalar       *f;
  const PetscScalar *x;
  const PetscReal    k_B = 1.6e-12, e_cgs = 4.8e-10, m_cgs[2] = {9.1094e-28, 9.1094e-28 * ctx->masses[1] / ctx->masses[0]}; // erg/eV, e, m as per NRL;
  PetscReal          AA, sqrtA, v_abT, vTe, t1, TeDiff, Te, Ti, Tdiff;

  PetscFunctionBeginUser;
  n_cm3[0] = n_cm3[1] = ctx->n0[0];
  PetscCall(VecGetArrayRead(X, &x));
  Te = PetscRealPart(2 * x[E_PERP_IDX] + x[E_PAR_IDX]) / 3, Ti = PetscRealPart(2 * x[I_PERP_IDX] + x[I_PAR_IDX]) / 3;
  v_abT = 1.8e-19 * PetscSqrtReal(m_cgs[0] * m_cgs[1]) * n_cm3[0] * ctx->lnLam * PetscPowReal(m_cgs[0] * Ti + m_cgs[1] * Te, -1.5);
  PetscCall(VecGetArray(F, &f));
  for (PetscInt ii = 0; ii < 2; ii++) {
    PetscReal tPerp = PetscRealPart(x[2 * ii + E_PERP_IDX]), tPar = PetscRealPart(x[2 * ii + E_PAR_IDX]);
    TeDiff = tPerp - tPar;
    AA     = tPerp / tPar - 1;
    if (AA < 0.){
      sqrtA = PetscSqrtReal(-AA);
      t1    = (-3 + (AA + 3) * PetscAtanhReal(sqrtA) / sqrtA) / PetscSqr(AA);
      //PetscReal vTeB = 8.2e-7 * n_cm3[0] * ctx->lnLam * PetscPowReal(Te, -1.5);
      vTe = PetscRealPart(2 * PetscSqrtReal(PETSC_PI / m_cgs[ii]) * PetscSqr(PetscSqr(e_cgs)) * n_cm3[0] * ctx->lnLam * PetscPowReal(k_B * x[E_PAR_IDX], -1.5)) * t1;
      t1  = vTe * TeDiff; // scaling form NRL that makes it work ???
    }
    else {
      sqrtA = PetscSqrtReal(AA);
      t1    = (-3 + (AA + 3) * PetscAtanReal(sqrtA) / sqrtA) / PetscSqr(AA);
      //PetscReal vTeB = 8.2e-7 * n_cm3[0] * ctx->lnLam * PetscPowReal(Te, -1.5);
      vTe = PetscRealPart(2 * PetscSqrtReal(PETSC_PI / m_cgs[ii]) * PetscSqr(PetscSqr(e_cgs)) * n_cm3[0] * ctx->lnLam * PetscPowReal(k_B * x[E_PAR_IDX], -1.5)) * t1;
      t1  = vTe * TeDiff; // scaling form NRL that makes it work ???
    }
    f[2 * ii + E_PAR_IDX]  = 2 * t1; // par
    f[2 * ii + E_PERP_IDX] = -t1;    // perp
    Tdiff                  = (ii == 0) ? (Ti - Te) : (Te - Ti);
    f[2 * ii + E_PAR_IDX] += v_abT * Tdiff;
    f[2 * ii + E_PERP_IDX] += v_abT * Tdiff;
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(F, &f));
  PetscCall(VecViewFromOptions(F, NULL, "-nrl_res_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode createVec_NRL(AppCtx *ctx, Vec *vec)
{
  PetscScalar *x;
  Vec          Temps;

  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, NUM_TEMPS, &Temps));
  PetscCall(VecGetArray(Temps, &x));
  for (PetscInt i = 0; i < 4; i++) x[i] = ctx->T[i];
  PetscCall(VecRestoreArray(Temps, &x));
  *vec = Temps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Monitor_NRL(TS ts_nrl)
{
  Vec sol;
  const PetscReal *solArr;
  PetscReal time;
  PetscInt  steps;
  AppCtx* user;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts_nrl, &steps));
  PetscCall(TSGetApplicationContext(ts_nrl, (void **) &user));
  if (steps % user->outputNum == 0){
    PetscCall(TSGetTime(ts_nrl, &time));
    PetscCall(TSGetSolution(ts_nrl, &sol));
    PetscCall(VecGetArrayRead(sol, &solArr));
    PetscPrintf(PETSC_COMM_WORLD, "NRL Time: %g\n", time);
    PetscPrintf(PETSC_COMM_WORLD, "NRLT_ex: %.8g, NRLT_ey: %.8g, NRLT_ix: %.8g, NRLT_iy: %.8g\n", solArr[0]/1.16045250061657e7, solArr[1]/1.16045250061657e7, solArr[2]/1.16045250061657e7, solArr[3]/1.16045250061657e7);
    PetscCall(VecRestoreArrayRead(sol, &solArr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createTS_NRL(AppCtx *ctx, Vec Temps)
{
  TSAdapt adapt;
  TS      ts;

  PetscFunctionBeginUser;
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  ctx->ts_nrl = ts; // 'data' is for applications (eg, monitors)
  PetscCall(TSSetApplicationContext(ts, ctx));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetRHSFunction(ts, NULL, IsotropizationFunctionNRL, ctx));
  PetscCall(TSSetSolution(ts, Temps));
  PetscCall(TSRKSetType(ts, TSRK2A));
  PetscCall(TSSetOptionsPrefix(ts, "nrl_"));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptSetType(adapt, TSADAPTNONE));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetMaxSteps(ts, 1));
  PetscCall(TSSetTime(ts, 0));
  PetscCall(TSSetPostStep(ts, Monitor_NRL));
  //PetscCall(TSMonitorSet(ts, Monitor_NRL, ctx, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc,char **argv)
{
  TS             ts;     /* nonlinear solver */
  DM             dm, sw; /* Velocity space mesh and Particle Swarm */
  Vec            u, v, vec_nrl;   /* problem vector */
  PetscInt       Np, dim;
  MPI_Comm       comm;
  AppCtx         user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  /* Initialize objects and set initial conditions */
  PetscCall(SetupProb(comm, &dm, &sw, &user));
  PetscCall(DMSetApplicationContext(sw, &user));
  PetscCall(DMSwarmVectorDefineField(sw, "velocity"));
  PetscCall(DMSwarmSetNumSpecies(sw, 2));
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 100000.0));
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
  PetscCall(PetscPrintf(comm, "Np: %"PetscInt_FMT"\n", Np));
  PetscCall(DMGetDimension(dm, &dim));

  if (user.run_nrl) {
    PetscReal momentum[6]={0., 0., 0., 0., 0., 0}, KE[6]={0., 0., 0., 0., 0., 0}, T[6]={0., 0., 0., 0., 0., 0};

    // make them match the calculated distribution
    PetscCall(CalculateMomentsAndTemperatures_Anisotropic(sw, momentum, KE, T));
    // x is treated parallel, y perp in 2d, in 3d (x+y) are treated as parallel w/ z perp.
    if (dim == 2) for (PetscInt i = 0; i < 4; ++i) user.T[i] = T[i]*1.16045250061657e7;
    else {
      // give it Txe and Txi in the first and 3rd, take average and give perp
      user.T[1] = ((T[0] + T[1])/2.)*1.16045250061657e7;
      user.T[0] = T[2]*1.16045250061657e7;
      user.T[3] = ((T[3] + T[4])/2.)*1.16045250061657e7;
      user.T[2] = T[5]*1.16045250061657e7;
    }
    PetscPrintf(PETSC_COMM_WORLD, "initial nrl temps: %g, %g, %g, %g\n", T[0], T[1], T[2], T[3]);
    createVec_NRL(&user, &vec_nrl);
    createTS_NRL(&user, vec_nrl);
    TSView(user.ts_nrl, PETSC_VIEWER_STDOUT_WORLD);
  }
  PetscCall(TSSetSolution(ts, u));
  VecViewFromOptions(u, NULL, "-ic_view");
  PetscCall(TSPostStep(ts));
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
  test:
    suffix: algoim_2v_equilibration
    requires: ks triangle algoim !single !complex
    args: -steps 10000000 -step_size 0.001 \                                                                                                                      -ts_type theta -ts_theta_theta 0.5\
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
    -dm_swarm_charges -1,1\
    -dm_swarm_temperature 0.3,0.25 -regular -order 3 -npls 10 -ts_adapt_type none -ts_max_snes_failures -1
  test:
    suffix: algoim_3v_equilibration
    requires: ks triangle algoim !single !complex
    args: -steps 10000000 -step_size 0.001 \                                                                                                                      -ts_type theta -ts_theta_theta 0.5\
    -dm_plex_simplex 0 -dm_plex_dim 3\
    -dm_plex_box_lower -1,-1,-1\
    -dm_plex_box_upper 1,1,1\
    -dm_plex_box_faces 1,1,1\
    -snes_mf\
    -dm_swarm_num_species 2\
    -dm_swarm_masses 1.,1.\
    -dm_swarm_mass_units 0,0\
    -dm_swarm_charges -1,1\
    -dm_swarm_temperature 0.3,0.25 -regular -order 3 -npls 10 -ts_adapt_type none -ts_max_snes_failures -1
TEST*/
