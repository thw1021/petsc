static char help[] = "Particle Basis Landau Example using nonlinear solve + Implicit Midpoint-like time stepping.";

#include <petscdmplex.h>
#include <petsc/private/petscfeimpl.h> /* For CoordinatesRefToReal() */
#include <petscdmswarm.h>
#include <petscts.h>
#include <petscdraw.h>
#include <petscviewer.h>
#include <petscmath.h>

typedef struct {
  PetscInt    particlesPerCell; /* The number of partices per cell */
  PetscInt    dim;              /* Topological mesh dimension */
  PetscReal   momentTol;        /* Tolerance for checking moment conservation */
  PetscBool   monitor;
  PetscBool   error;            /* Flag for printing the error */
  PetscReal   epsi;             /* gaussian regularization parameter */
  PetscBool   simplices;        /* True for simplices, false for tensor cells */
  PetscInt    ostep;            /* print the energy at each ostep time steps */
  PetscDraw   draw;             /* The draw object for histogram monitoring */
  PetscInt    max_step;         /* Number of time steps to take */
  PetscReal   step_size;        /* Size of each time step */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->monitor          = PETSC_FALSE;
  options->particlesPerCell = 1;
  options->momentTol        = 100.0*PETSC_MACHINE_EPSILON;
  options->ostep            = 100;
  options->dim              = 2;
  
  options->simplices        = PETSC_FALSE;
  options->epsi             = 0.64*pow(4,1.98);
  options->max_step         = 1;
  options->step_size        = 0.01;

  ierr = PetscOptionsBegin(comm, "", "Collision Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitor", "Flag to use the TS histogram monitor", "ex27.c", options->monitor, &options->monitor, NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsBool("-simplices", "True for simplices, falls for tensor cells", "ex27.c", options->simplices, &options->simplices, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex27.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "Topological mesh dimension", "ex27.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-output_step", "Number of time steps between output", "ex27.c", options->ostep, &options->ostep, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-max_step", "Maximum number of time steps, default=1", "ex27.c", options->max_step, &options->max_step, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-epsi", "Mollifier regularization parameter", "ex27.c", options->epsi, &options->epsi, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-step_size", "Time step dt, default=0.01", "ex27.c", options->step_size, &options->step_size, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Declaration of static functions */
static PetscErrorCode ComputeGradEFunctionalAtPoint(DM sw, PetscInt Np, PetscQuadrature quad, PetscReal *particle, PetscReal *field, PetscReal* integral,  void* ctx);
static PetscErrorCode ComputeAndApplyQForPPPrimePair(PetscReal* particle, PetscInt ppridx, const PetscReal *u, PetscReal *GammaS, PetscReal *particle_residual, PetscInt Np, PetscInt dim);
static PetscErrorCode ComputelnSumFromPoint(PetscReal* u, PetscReal *particle, PetscReal *sum_ptr, void* ctx);

/* Create the mesh for velocity space */
static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{

  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplices, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
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
  Np   = user->particlesPerCell;
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
  PetscReal v_val;
  for(c=cStart; c<cEnd; ++c){
    for(p=0; p<Np; ++p){
      for(d=0; d<dim; ++d){
        ierr = PetscRandomGetValueReal(rndv, &v_val);
        //velocity[dim*p+d] = d == 0 ? p : 0;
        velocity[p*dim+d] = v_val;
  
      }
    }
  }
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = PetscFree5(centroid, xi0, v0, J, invJ);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
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
  Np   = user->particlesPerCell;
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
  PetscInt       dim, cStart, cEnd, c, Np = user->particlesPerCell, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);

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


static PetscErrorCode RHSFunctionParticles(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  const PetscScalar *u;                   /* input solution vector */
  PetscScalar       *r;
  PetscQuadrature    quad;                /* for integral evaluation */
  PetscInt           dim, d, Np, p, ppr;  /* spatial dim, index tracking, no. particles, index tracking, pprime index, cell idx, cell idx, cell, quad points, quad idx */
  PetscReal          *velocity;           //Nc, *points, *q_weights,  
  DM                 sw;                  /* point tracking, problem topology */
  PetscErrorCode     ierr;
  AppCtx* user = (AppCtx*) ctx;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(R);CHKERRQ(ierr);
  
  /* Create quadrature for integral evaluation */  
  ierr = TSGetDM(ts, &sw);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMGetDimension(sw, &dim);
  ierr = DMSwarmGetField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U, &Np);CHKERRQ(ierr);
  ierr = VecGetArray(R, &r);
  ierr = VecGetArrayRead(U, &u);
  Np /= dim;

  ierr = PetscDTGaussTensorQuadrature(dim, 2*dim*Np,dim*Np, -1, 1, &quad);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Part  ppr     x        y\n");CHKERRQ(ierr);
  for(p = 0; p < Np; ++p){
    PetscReal particle[2], res[2]={0.,0.}, integral[2]={0.,0.}, ln_sum;
    
    for(d=0; d < dim; ++d) particle[d] = velocity[p*dim+d];
    ierr = ComputeGradEFunctionalAtPoint(sw, Np, quad, particle, velocity, integral, user);CHKERRQ(ierr);
    
    /* compute entropy integrals for p' to subtract from S_p */
    for(ppr = 0; ppr<Np; ++ppr){
      PetscReal pprime[2], integral_ppr[2]={0.,0.}, GammaS[2]={0.,0.}, particle_shift[2];
      
      if(ppr == p) continue;
      for(d=0; d < dim; ++d) pprime[d] = velocity[ppr*dim+d];
      ierr = ComputeGradEFunctionalAtPoint(sw, Np, quad, pprime, velocity, integral_ppr, user);CHKERRQ(ierr);
      
      for(d=0;d<dim;++d) GammaS[d] = integral[d] - integral_ppr[d];
      
      /* We are done with S so swap particle to be at the midpoint */
      for(d=0; d<dim; ++d) particle_shift[d] = u[p*dim+d];
      ierr = ComputeAndApplyQForPPPrimePair(particle_shift, ppr, u, GammaS, res, Np, dim);
      /* The affect had better be symmetric. */
      
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%4D %4D %10.8lf %10.8lf\n", p, ppr, res[0], res[1]);CHKERRQ(ierr);
      for(d=0; d<dim;++d) r[p*dim+d] += res[d];
    }
  }

  ierr = DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **) &velocity);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);
  ierr = VecViewFromOptions(R, NULL, "-residual_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode ComputeAndApplyQForPPPrimePair(PetscReal* particle, PetscInt ppridx, const PetscReal *u, PetscReal *GammaS, PetscReal *particle_residual, PetscInt Np, PetscInt dim){
  int p, d;
  PetscReal xiTxi, xicpy[2];

  /* Not hard, but looks gross */
  for(d=0; d<dim;++d) particle_residual[d] = 0.;
  PetscReal xi[2],mag_xi=0., xiTS=0.;
  for(d=0; d<dim;++d) xi[d] = particle[d] - u[ppridx*dim+d];
  for(d=0; d<dim;++d) mag_xi += xi[d]*xi[d];

  mag_xi = PetscSqrtReal(mag_xi);
  for(d=0; d<dim; ++d) xiTS += xi[d]*GammaS[d]; 
  for(d=0; d<dim; ++d) xi[d] *= (xiTS/(mag_xi*mag_xi));
  for(d=0; d<dim; ++d) particle_residual[d] += (1/mag_xi) *(GammaS[d] - xi[d]);
  

  /* check Q(xi)xi annihilates. */
  for(d=0; d<dim; ++d) xi[d] = particle[d]-u[ppridx*dim+d];
  for(d=0; d<dim; ++d) xiTxi += xi[d]*xi[d];
  for(d=0; d<dim; ++d) xicpy[d] = xi[d]*(xiTS/(mag_xi*mag_xi));
  for(d=0; d<dim; ++d) if((xi[d] - xicpy[d]) != 0) SETERRQ(PETSC_COMM_WORLD, 1, "Q failed to annhilate xi.\n");
  return(0);
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

#if 1
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
#endif 
#if 0
static PetscErrorCode InitializeSolve_TSFree(DM sw, Vec u){
  AppCtx        *user;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetApplicationContext(sw, (void **) &user);CHKERRQ(ierr);
  ierr = SetInitialCoordinates(sw);CHKERRQ(ierr);
  ierr = SetInitialConditions(sw, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
#endif 
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

/* Particle centered Gaussian and Gaussian gradient functions for quadrature point integration.
  Assume 2d normalization */
static PetscReal Gaussian(PetscReal* center, PetscInt dim, PetscReal* q, PetscReal epsi){
  return 1.0/(2.0*PETSC_PI*epsi) * exp(-((center[0]-q[0])*(center[0]-q[0])+(center[1]-q[1])*(center[1]-q[1]))/(2.0*epsi));
}

/* Evaluate and return the evaluation of a function composed of a collection of gaussians relative to the particle at a quadrature point. This function
is to be used for a gaussian centered around the particle with its full "weight" integrated with a quadrature rule (tensor quadrature) */
static void GradGaussian(PetscReal* center, PetscInt dim, PetscReal* q, PetscReal epsi, PetscReal* gradpsi){
  
  gradpsi[0] = (-1./epsi) * center[0] * Gaussian(center, dim, q, epsi);
  gradpsi[1] = (-1./epsi) * center[1] * Gaussian(center, dim, q, epsi);

}

/* This is the function used to evaluate a pointwise weighted gaussian in the fully discrete-in-velocity formulation */
static void mollifiedGaussian(PetscReal* center, PetscInt dim, PetscReal* q, PetscReal epsi, PetscReal *points, PetscInt Np, PetscReal* gradpsi){
  PetscInt p, d, c_e[2]={0.,0.};
  
  /* \grad S_i = \grad \psi(v_i - v) log \sum_k \psi(v-v_k)  */
  for(p=0; p<Np; ++p){
    PetscReal v_k[2], c[2]={0.,0.};
    for(d=0; d<dim; ++d) v_k[d] = points[p*dim+d];//-q[d];
    for(d=0; d<dim; ++d) gradpsi[d] += Gaussian(c, dim, v_k, epsi);
  }

  for(d=0; d<dim; ++d) gradpsi[d] = log(gradpsi[d]);
  for(d=0; d<dim; ++d) gradpsi[d] *= (-1./(epsi)) * (center[d]) * (Gaussian(c_e, dim, center, epsi));
  
}

/* ---------------------------- This whole section of code needs to be rethought so we don't have excessive N^2 operations ------------------- */

/* Integrate the gradient of the entropy functional for a particle. This should be optimized out */
static PetscErrorCode ComputeGradEFunctionalAtPoint(DM sw, PetscInt Np, PetscQuadrature quad, PetscReal *particle, PetscReal *field, PetscReal* integral,  void* ctx){
  PetscInt         dim, d, q, Nc, Nq, p;
  PetscReal        *xi0, *v0, *J, *invJ, *velocity, detJ;
  const PetscReal  *q_weights, *points;
  AppCtx*          user = (AppCtx*) ctx;
  DM               plex;
  PetscErrorCode   ierr;
  
  ierr = DMSwarmGetCellDM(sw, &plex);
  
  ierr = DMGetDimension(plex, &dim);
  ierr = PetscMalloc4(dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  ierr = DMPlexComputeCellGeometryFEM(plex, 0, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
  /* Assuming 2D here, easily extrapolated to 3D when it finally works */
  ierr = PetscQuadratureGetData(quad, &dim, &Nc, &Nq, &points, &q_weights);CHKERRQ(ierr);  
  for (q = 0; q < Nq; ++q) {
    PetscReal   integrand[2] = {0.,0.};
    PetscReal   w;
    PetscReal   q_point[2]={0.,0.}, x[2];
    for(d=0; d<dim;++d) q_point[d] = points[q*dim+d];
    CoordinatesRefToReal(dim, dim, xi0, v0, J, q_point, x);CHKERRQ(ierr);
    w = detJ*q_weights[q];
    
    //GradGaussian(particle, dim, q_point, user->epsi, integrand);
    mollifiedGaussian(particle, dim, q_point, user->epsi, field, Np, integrand);
    for(d=0; d<dim;++d)integral[d] += integrand[d];//*w;
    break;
  }
  ierr = PetscFree4(xi0, v0, J, invJ);
  return(0);
}

/* 
  Compute the summation of gaussian evaluations from the gaussian centered at the origin of the cell according to . This function is not needed
  but is kept around for diagnostics.
 */

static PetscErrorCode ComputelnSumFromPoint(PetscReal* u, PetscReal *particle, PetscReal *sum_ptr, void* ctx){
  PetscInt       Np, p, d;
  PetscReal      sum=0.;
  AppCtx*        user = (AppCtx*) ctx;
  Np=user->particlesPerCell;
  for(p=0;p<Np;++p){
    PetscReal pprime[2];
    for(d=0;d<user->dim;++d) pprime[d] = u[p*user->dim+d];
    sum += log(Gaussian(particle, user->dim, pprime, user->epsi));
  }
  *sum_ptr = sum;
  return(0);
}

/* ---------------------------- This whole section of code needs to be rethought so we don't have excessive N^2 operations ------------------- */

/* 
    Initialize a coordinate grid and randomly distribute particles in space and velocity. Peroform Eulerian and DG evaluations of the
    particle basis landau collision operator.
*/
int main(int argc,char **argv)
{
  SNES            snes;                       /* Non linear solve for time stepping */
  TS              ts;                       /* nonlinear solver */
  DM              dm, sw;                     /* Velocity space mesh and Particle Swarm */
  Vec             u, v;                       /* problem vector */
  MPI_Comm        comm;
  AppCtx          user;
  PetscErrorCode  ierr;
  #if 0
  PetscQuadrature quad;
  
  PetscReal       *xi0, *v0, *J, *invJ, detJ; /* Info for CoordsReftoReal */
  PetscInt        n, p, Np;
  #endif 

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  
  /* Initialize objects and set initial conditions */
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(sw, &user);CHKERRQ(ierr);
  ierr = DMSwarmVectorDefineField(sw, "velocity");CHKERRQ(ierr);
  //ierr = DMSwarmCreateGlobalVectorFromField(sw, "velocity", &u);CHKERRQ(ierr);
#if 0  
  ierr = SNESCreate(comm, &snes);
  ierr = SNESSetFromOptions(snes);

  ierr = InitializeSolve_TSFree(sw, u);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u, &Np);CHKERRQ(ierr);
  Np /= user.dim;
  /* Compute Cell Geometry for coords ref to real, single velocity space cell is assumed. */
  ierr = PetscMalloc4(dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  ierr = DMPlexComputeCellGeometryFEM(plex, 0, NULL, v, J, invJ, &detJ);CHKERRQ(ierr);
  ierr = PetscDTGaussTensorQuadrature(dim, dim*Np, -1, 1, &quad);CHKERRQ(ierr);
#endif   
  /* 
    The following statement depends on a fully discrete in velocity, or a semi-discrete in velocity case:

    Compute the time step for v^n+1 = 1(p,p')Q(v_i^n+1/2 - v_k^n+1/2)\Gamma(S^n, p, p')

    A Tensor Quadrature rule is used to evaluate the integrals for S = \int_R^d \grad \psi_\epsilon (v_p)

    the log of the summation for the term ln \sum_k \psi_\epsilon(v_p') is computed in relation to the gaussian centered at v_p

    The midpoint velocities are found using SNES for a non linear forward solve to compute the backwards evaluation for the midpoints.
  
  */

#if 1
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
  ierr = VecCopy(v, u);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v);CHKERRQ(ierr);
  ierr = TSComputeInitialCondition(ts, u);CHKERRQ(ierr);
  if(user.monitor) TSMonitorSet(ts, Monitor, &user, NULL);
  ierr = TSSetPostStep(ts, UpdateSwarm);CHKERRQ(ierr);
  ierr = TSGetSNES(ts, &snes);
  ierr = SNESSetFromOptions(snes);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
#endif 
  
  //ierr = PetscFree4(xi0, v0, J, invJ);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
/*
NOTE: implicit midpoint behaves better than a fully backwards euler time discretization
*/


/*TEST
   build:
     requires: triangle !single !complex
   test:
     suffix: euler
     args: -dim 2 -particles_per_cell 3 -output_step 5 -ts_type euler -dm_plex_box_dim 2 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1 -dm_plex_box_upper 1,1 -dm_view -monitor -output_step 1
   test:
     suffix: 1
     args: -dim 2 -particles_per_cell 3 -output_step 5 -ts_type theta -ts_theta_theta 1.0 -dm_plex_box_dim 2 -dm_plex_box_faces 1,1 -dm_plex_box_lower -2,-2 -dm_plex_box_upper 2,2 -dm_view -monitor -output_step 1
   test:
     suffix: 2
     args: -dim 2 -particles_per_cell 3 -output_step 5 -ts_type theta -ts_theta_theta 0.5 -snes_fd -dm_plex_box_dim 2 -dm_plex_box_faces 1,1 -dm_plex_box_lower -1,-1 -dm_plex_box_upper 1,1 -dm_view -monitor -output_step 1
TEST*/
