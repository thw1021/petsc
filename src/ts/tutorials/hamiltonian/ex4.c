static char help[] = "Two-level system for Landau Damping using Vlasov-Poisson equations\n";

/*
  Moment Equations:

    We will discretize the moment equations using finite elements, and we will project the moments into the finite element space We will use the PFAK method, which guarantees that our FE approximation is weakly equivalent to the true moment. The first moment, number density, is given by

      \int dx \phi_i n_f = \int dx \phi_i n_p
      \int dx \phi_i n_f = \int dx \phi_i \int dv f
      \int dx \phi_i n_f = \int dx \phi_i \int dv \sum_p w_p \delta(x - x_p) \delta(v - v_p)
      \int dx \phi_i n_f = \int dx \phi_i \sum_p w_p \delta(x - x_p)
                   M n_F = M_p w_p

    where

      (M_p){ip} = \phi_i(x_p)

    which is just a scaled version of the charge density. The second moment, momentum density, is given by

      \int dx \phi_i p_f = m \int dx \phi_i \int dv v f
      \int dx \phi_i p_f = m \int dx \phi_i \sum_p w_p \delta(x - x_p) v_p
                   M p_F = M_p v_p w_p

    And finally the third moment, pressure, is given by

      \int dx \phi_i pr_f = m \int dx \phi_i \int dv (v - u)^2 f
      \int dx \phi_i pr_f = m \int dx \phi_i \sum_p w_p \delta(x - x_p) (v_p - u)^2
                   M pr_F = M_p (v_p - u)^2 w_p
                          = M_p (v_p - p_F(x_p) / m n_F(x_p))^2 w_p
                          = M_p (v_p - (\sum_j p_F \phi_j(x_p)) / m (\sum_k n_F \phi_k(x_p)))^2 w_p

    Here we need all FEM basis functions \phi_i that see that particle p.

  To run the code with particles sinusoidally perturbed in x space use the test "pp_poisson_bsi_1d_4" or "pp_poisson_bsi_2d_4"
  According to Lukas, good damping results come at ~16k particles per cell

  Swarm CellDMs
  =============
  Name: "space"
  Fields: DMSwarmPICField_coor, "velocity"
  Coordinates: DMSwarmPICField_coor

  Name: "velocity"
  Fields: "w_q"
  Coordinates: "velocity"

  Name: "moments"
  Fields: "w_q"
  Coordinates: DMSwarmPICField_coor

  Name: "moment fields"
  Fields: "velocity"
  Coordinates: DMSwarmPICField_coor

  To visualize the maximum electric field use

    -efield_monitor

  To monitor velocity moments of the distribution use

    -ptof_pc_type lu -moments_monitor

  To monitor the particle positions in phase space use

    -positions_monitor

  To monitor the charge density, E field, and potential use

    -poisson_monitor

  To monitor the remapping field use

    -remap_uf_view draw

  To visualize the swarm distribution use

    -ts_monitor_hg_swarm

  To visualize the particles, we can use

    -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain 0 -ts_monitor_sp_swarm_phase 1 -draw_size 500,500

  To visualize FAS use

    -moment_field_monitor -fas_coarse_snes_monitor_fields draw
*/
#include <petsctao.h>
#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscbag.h>
#include <petscdraw.h>
#include <petsc/private/petscfeimpl.h> /* For interpolation */
#include <petsc/private/dmswarmimpl.h> /* For swarm debugging */
#include "petscdm.h"
#include "petscdmlabel.h"

PETSC_EXTERN PetscErrorCode stream(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode line(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);

const char *EMTypes[] = {"primal", "mixed", "coulomb", "none", "EMType", "EM_", NULL};
typedef enum {
  EM_PRIMAL,
  EM_MIXED,
  EM_COULOMB,
  EM_NONE
} EMType;

typedef enum {
  V0,
  X0,
  T0,
  M0,
  Q0,
  PHI0,
  POISSON,
  VLASOV,
  SIGMA,
  NUM_CONSTANTS
} ConstantType;

typedef enum {
  E_MONITOR_NONE,
  E_MONITOR_FULL,
  E_MONITOR_QUIET
} EMonitorType;
const char *const EMonitorTypes[] = {"NONE", "FULL", "QUIET", "EMonitorType", "E_MONITOR_", NULL};

typedef struct {
  DM          sw;                   // Borrowed ref to the DMSwarm
  DM          dm;                   // The DM for moment fields
  TS          ts;                   // The TS for moment fields
  KSP         ksp_fn;               // Solver for the mass finite element mass matrix
  KSP         ksp_fp;               // Solver for the momentum finite element mass matrix
  KSP         ksp_fe;               // Solver for the energy finite element mass matrix
  KSP         ksp_p;                // Solver for the particle mass matrix
  Vec         work_p;               // Particle work vector
  DM          dmN;                  // The DM for number density fields
  IS          isN;                  // The IS mapping dmN into dm
  Mat         MN;                   // The finite element mass matrix for number density
  Mat         IN;                   // The finite element interpolation matrix for number density into charge
  Vec         INscale;              // The finite element interpolation scale for number density into charge
  DM          dmP;                  // The DM for momentum density fields
  IS          isP;                  // The IS mapping dmP into dm
  Mat         MP;                   // The finite element mass matrix for momentum density
  DM          dmE;                  // The DM for energy density (pressure) fields
  IS          isE;                  // The IS mapping dmE into dm
  Mat         ME;                   // The finite element mass matrix for energy density (pressure)
  PetscBool   moment_field_monitor; // Flag to show moment field monitor
  PetscViewer viewerN;              // Number density viewer
  PetscViewer viewerP;              // Momentum density viewer
  PetscViewer viewerE;              // Energy density (pressure) viewer
  PetscViewer viewerNRes;           // Number density residual viewer
  PetscViewer viewerPRes;           // Momentum density residual viewer
  PetscViewer viewerERes;           // Energy density (pressure) residual viewer
  PetscDrawLG drawlgMomRes;         // Residuals for the moment equations
} MomStruct;

typedef struct {
  PetscScalar v0; /* Velocity scale, often the thermal velocity */
  PetscScalar t0; /* Time scale */
  PetscScalar x0; /* Space scale */
  PetscScalar m0; /* Mass scale */
  PetscScalar q0; /* Charge scale */
  PetscScalar kb;
  PetscScalar epsi0;
  PetscScalar phi0;          /* Potential scale */
  PetscScalar poissonNumber; /* Non-Dimensional Poisson Number */
  PetscScalar vlasovNumber;  /* Non-Dimensional Vlasov Number */
  PetscReal   sigma;         /* Nondimensional charge per length in x */
} Parameter;

typedef struct {
  PetscInt         s;    // Starting sample (we ignore some in the beginning)
  PetscInt         e;    // Ending sample
  PetscInt         el;   // Maximum sample
  PetscInt         per;  // Period of fitting
  const PetscReal *t;    // Time for each sample
  PetscReal       *Emax; // Emax for each sample
} EmaxCtx;

typedef struct {
  PetscBag     bag;               // Problem parameters
  PetscBool    error;             // Flag for printing the error
  PetscInt     remapFreq;         // Number of timesteps between remapping
  PetscBool    saveOutput;        // Flag to save screenshots in files
  EMonitorType efield_monitor;    // Flag to show electric field monitor
  PetscBool    moment_monitor;    // Flag to show distribution moment monitor
  PetscBool    positions_monitor; // Flag to show particle positins at each time step
  PetscBool    poisson_monitor;   // Flag to display charge, E field, and potential at each solve
  PetscBool    initial_monitor;   // Flag to monitor the initial conditions
  PetscInt     velocity_monitor;  // Cell to monitor the velocity distribution for
  PetscBool    perturbed_weights; // Uniformly sample x,v space with gaussian weights
  PetscInt     ostep;             // Print the energy at each ostep time steps
  PetscInt     numParticles;
  PetscReal    timeScale;              // Nondimensionalizing time scale
  PetscReal    charges[2];             // The charges of each species
  PetscReal    masses[2];              // The masses of each species
  PetscReal    thermal_energy[2];      // Thermal Energy (used to get other constants)
  PetscReal    cosine_coefficients[2]; // (alpha, k)
  PetscBool    optimizeVelGrid;        // Equalize weights in velocity bins
  PetscReal    gridVolume;             // Total volume of the mesh
  PetscReal    totalWeight;
  PetscReal    stepSize;
  PetscInt     steps;
  PetscReal    initVel;
  EMType       em;            // Type of electrostatic model
  SNES         snes;          // EM solver
  MomStruct    momCtx;        // The context for the moment problem
  DM           dmPot;         // The DM for potential
  Mat          fftPot;        // Fourier Transform operator for the potential
  Vec          fftX, fftY;    //   FFT vectors with phases added (complex parts)
  IS           fftReal;       //   The indices for real parts
  IS           isPot;         // The IS for potential, or NULL in primal
  Mat          M;             // The finite element mass matrix for potential
  PetscFEGeom *fegeom;        // Geometric information for the DM cells
  PetscDrawHG  drawhgic_x;    // Histogram of the particle weight in each X cell
  PetscDrawHG  drawhgic_v;    // Histogram of the particle weight in each X cell
  PetscDrawHG  drawhgcell_v;  // Histogram of the particle weight in a given cell
  PetscBool    validE;        // Flag to indicate E-field in swarm is valid
  PetscReal    drawlgEmin;    // The minimum lg(E) to plot
  PetscDrawLG  drawlgE;       // Logarithm of maximum electric field
  PetscDrawLG  drawlgEMom;    // Logarithm of maximum electric field for the moment equation
  PetscDrawSP  drawspE;       // Electric field at particle positions
  PetscDrawSP  drawspX;       // Particle positions
  PetscBool    drawPosGrid;   // Draw the remap grid on the particle positions view
  PetscViewer  viewerRho;     // Charge density viewer
  PetscViewer  viewerRhoHat;  // Charge density Fourier Transform viewer
  PetscViewer  viewerPhi;     // Potential viewer
  DM           swarm;         // The particle swarm
  PetscRandom  random;        // Used for particle perturbations
  PetscBool    two_stream;    // Flag for activating 2-stream setup
  PetscReal    two_stream_v0; // 2-stream beam velocity
  PetscBool    checkweights;  // Check weight normalization
  PetscInt     checkVRes;     // Flag to check/output velocity residuals for nightly tests
  PetscBool    checkLandau;   // Check the Landau damping result
  EmaxCtx      emaxCtx;       // Information for fit to decay profile
  PetscReal    C;             // The coefficient for Landau damping
  PetscReal    gamma;         // The damping rate for Landau damping
  PetscReal    omega;         // The perturbed oscillation frequency for Landau damping
  PetscReal    phi;           // The phase shift for Landau damping

  PetscLogEvent RhsXEvent, RhsVEvent, ESolveEvent, ETabEvent;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscInt d                      = 2;
  PetscInt maxSpecies             = 2;
  options->remapFreq              = 1;
  options->velocity_monitor       = -1;
  options->ostep                  = 100;
  options->timeScale              = 2.0e-14;
  options->charges[0]             = -1.0;
  options->charges[1]             = 1.0;
  options->masses[0]              = 1.0;
  options->masses[1]              = 1000.0;
  options->thermal_energy[0]      = 1.0;
  options->thermal_energy[1]      = 1.0;
  options->cosine_coefficients[0] = 0.01;
  options->cosine_coefficients[1] = 0.5;
  options->initVel                = 1;
  options->totalWeight            = 1.0;
  options->drawlgEmin             = -6;
  options->numParticles           = 32768;
  options->two_stream_v0          = 5.;
  options->emaxCtx.s              = 50;
  options->emaxCtx.el             = -1;
  options->emaxCtx.per            = 100;
  options->C                      = -1.;

  PetscOptionsBegin(comm, "", "Landau Damping and Two Stream options", "DMSWARM");
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", __FILE__, options->error, &options->error, NULL));
  PetscCall(PetscOptionsInt("-remap_freq", "Number", __FILE__, options->remapFreq, &options->remapFreq, NULL));
  PetscCall(PetscOptionsBool("-save_output", "Flag to save screenshots in files", "ex2.c", options->saveOutput, &options->saveOutput, NULL));
  PetscCall(PetscOptionsEnum("-efield_monitor", "Flag to record and plot log(max E) over time", __FILE__, EMonitorTypes, (PetscEnum)options->efield_monitor, (PetscEnum *)&options->efield_monitor, NULL));
  PetscCall(PetscOptionsReal("-efield_min_monitor", "Minimum E field to plot", __FILE__, options->drawlgEmin, &options->drawlgEmin, NULL));
  PetscCall(PetscOptionsBool("-moments_monitor", "Flag to show moments table", __FILE__, options->moment_monitor, &options->moment_monitor, NULL));
  PetscCall(PetscOptionsBool("-moment_field_monitor", "Flag to show moment fields", __FILE__, options->momCtx.moment_field_monitor, &options->momCtx.moment_field_monitor, NULL));
  PetscCall(PetscOptionsBool("-ics_monitor", "Flag to show initial condition histograms", __FILE__, options->initial_monitor, &options->initial_monitor, NULL));
  PetscCall(PetscOptionsBool("-positions_monitor", "The flag to show particle positions", __FILE__, options->positions_monitor, &options->positions_monitor, NULL));
  PetscCall(PetscOptionsBool("-draw_positions_grid", "Draw the remap grid over the particle positions", __FILE__, options->drawPosGrid, &options->drawPosGrid, NULL));
  PetscCall(PetscOptionsBool("-poisson_monitor", "The flag to show charges, Efield and potential solve", __FILE__, options->poisson_monitor, &options->poisson_monitor, NULL));
  PetscCall(PetscOptionsInt("-velocity_monitor", "Cell to show velocity histograms", __FILE__, options->velocity_monitor, &options->velocity_monitor, NULL));
  PetscCall(PetscOptionsBool("-two_stream", "Run two stream instability", __FILE__, options->two_stream, &options->two_stream, NULL));
  PetscCall(PetscOptionsReal("-two_stream_v0", "+/- Beam velocities for two stream instabilities test", __FILE__, options->two_stream_v0, &options->two_stream_v0, NULL));
  PetscCall(PetscOptionsBool("-perturbed_weights", "Flag to run uniform sampling with perturbed weights", __FILE__, options->perturbed_weights, &options->perturbed_weights, NULL));
  PetscCall(PetscOptionsBool("-check_weights", "Ensure all particle weights are positive", __FILE__, options->checkweights, &options->checkweights, NULL));
  PetscCall(PetscOptionsBool("-check_landau", "Check the decay from Landau damping", __FILE__, options->checkLandau, &options->checkLandau, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", __FILE__, options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsReal("-timeScale", "Nondimensionalizing time scale", __FILE__, options->timeScale, &options->timeScale, NULL));
  PetscCall(PetscOptionsInt("-check_vel_res", "Check particle velocity residuals for nightly tests", __FILE__, options->checkVRes, &options->checkVRes, NULL));
  PetscCall(PetscOptionsReal("-initial_velocity", "Initial velocity of perturbed particle", __FILE__, options->initVel, &options->initVel, NULL));
  PetscCall(PetscOptionsReal("-total_weight", "Total weight of all particles", __FILE__, options->totalWeight, &options->totalWeight, NULL));
  PetscCall(PetscOptionsRealArray("-cosine_coefficients", "Amplitude and frequency of cosine equation used in initialization", __FILE__, options->cosine_coefficients, &d, NULL));
  PetscCall(PetscOptionsBool("-opt_vel_grid", "Equalize weights in velocity bins", __FILE__, options->optimizeVelGrid, &options->optimizeVelGrid, NULL));
  PetscCall(PetscOptionsRealArray("-charges", "Species charges", __FILE__, options->charges, &maxSpecies, NULL));
  PetscCall(PetscOptionsEnum("-em_type", "Type of electrostatic solver", __FILE__, EMTypes, (PetscEnum)options->em, (PetscEnum *)&options->em, NULL));
  PetscCall(PetscOptionsInt("-emax_start_step", "First time step to use for Emax fits", __FILE__, options->emaxCtx.s, &options->emaxCtx.s, NULL));
  PetscCall(PetscOptionsInt("-emax_end_step", "Last time step to use for Emax fits", __FILE__, options->emaxCtx.el, &options->emaxCtx.el, NULL));
  PetscCall(PetscOptionsInt("-emax_solve_step", "Number of time steps between Emax fits", __FILE__, options->emaxCtx.per, &options->emaxCtx.per, NULL));
  PetscOptionsEnd();

  PetscCall(PetscLogEventRegister("RhsX", TS_CLASSID, &options->RhsXEvent));
  PetscCall(PetscLogEventRegister("RhsV", TS_CLASSID, &options->RhsVEvent));
  PetscCall(PetscLogEventRegister("ESolve", TS_CLASSID, &options->ESolveEvent));
  PetscCall(PetscLogEventRegister("ETab", TS_CLASSID, &options->ETabEvent));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupContext(DM dm, DM sw, AppCtx *ctx)
{
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  if (ctx->efield_monitor) {
    PetscDraw     draw;
    PetscDrawAxis axis;

    if (ctx->efield_monitor == E_MONITOR_FULL) {
      PetscCall(PetscDrawCreate(comm, NULL, "Max Electric Field", 0, 0, 400, 300, &draw));
      PetscCall(PetscDrawSetSave(draw, "ex4_Efield"));
      PetscCall(PetscDrawSetFromOptions(draw));
    } else {
      PetscCall(PetscDrawOpenNull(comm, &draw));
    }
    PetscCall(PetscDrawLGCreate(draw, 2, &ctx->drawlgE));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawLGGetAxis(ctx->drawlgE, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Max Electric Field", "time", "E_max"));
    PetscCall(PetscDrawLGSetLimits(ctx->drawlgE, 0., ctx->steps * ctx->stepSize, ctx->drawlgEmin, 0.));

    if (ctx->momCtx.moment_field_monitor) {
      if (ctx->efield_monitor == E_MONITOR_FULL) {
        PetscCall(PetscDrawCreate(comm, NULL, "Max Electric Field", 0, 800, 400, 300, &draw));
        PetscCall(PetscDrawSetSave(draw, "ex2_Efield_mom"));
        PetscCall(PetscDrawSetFromOptions(draw));
      } else {
        PetscCall(PetscDrawOpenNull(comm, &draw));
      }
      PetscCall(PetscDrawLGCreate(draw, 1, &ctx->drawlgEMom));
      PetscCall(PetscDrawDestroy(&draw));
      PetscCall(PetscDrawLGGetAxis(ctx->drawlgEMom, &axis));
      PetscCall(PetscDrawAxisSetLabels(axis, "Max Electric Field", "time", "E_max"));
      PetscCall(PetscDrawLGSetLimits(ctx->drawlgEMom, 0., ctx->steps * ctx->stepSize, ctx->drawlgEmin, 0.));
    }
  }

  if (ctx->initial_monitor) {
    PetscDraw     drawic_x, drawic_v;
    PetscDrawAxis axis1, axis2;
    PetscReal     dmboxlower[2], dmboxupper[2];
    PetscInt      dim, cStart, cEnd;

    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMGetBoundingBox(dm, dmboxlower, dmboxupper));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

    PetscCall(PetscDrawCreate(comm, NULL, "monitor_initial_conditions_x", 0, 300, 400, 300, &drawic_x));
    PetscCall(PetscDrawSetSave(drawic_x, "ex4_ic_x"));
    PetscCall(PetscDrawSetFromOptions(drawic_x));
    PetscCall(PetscDrawHGCreate(drawic_x, (int)dim, &ctx->drawhgic_x));
    PetscCall(PetscDrawHGCalcStats(ctx->drawhgic_x, PETSC_TRUE));
    PetscCall(PetscDrawHGGetAxis(ctx->drawhgic_x, &axis1));
    PetscCall(PetscDrawHGSetNumberBins(ctx->drawhgic_x, (int)(cEnd - cStart)));
    PetscCall(PetscDrawAxisSetLabels(axis1, "Initial X Distribution", "X", "weight"));
    PetscCall(PetscDrawAxisSetLimits(axis1, dmboxlower[0], dmboxupper[0], 0, 0));
    PetscCall(PetscDrawDestroy(&drawic_x));

    PetscCall(PetscDrawCreate(comm, NULL, "monitor_initial_conditions_v", 400, 300, 400, 300, &drawic_v));
    PetscCall(PetscDrawSetSave(drawic_v, "ex4_ic_v"));
    PetscCall(PetscDrawSetFromOptions(drawic_v));
    PetscCall(PetscDrawHGCreate(drawic_v, (int)dim, &ctx->drawhgic_v));
    PetscCall(PetscDrawHGCalcStats(ctx->drawhgic_v, PETSC_TRUE));
    PetscCall(PetscDrawHGGetAxis(ctx->drawhgic_v, &axis2));
    PetscCall(PetscDrawHGSetNumberBins(ctx->drawhgic_v, 21));
    PetscCall(PetscDrawAxisSetLabels(axis2, "Initial V_x Distribution", "V", "weight"));
    PetscCall(PetscDrawAxisSetLimits(axis2, -6, 6, 0, 0));
    PetscCall(PetscDrawDestroy(&drawic_v));
  }

  if (ctx->velocity_monitor >= 0) {
    DM            vdm;
    DMSwarmCellDM celldm;
    PetscDraw     drawcell_v;
    PetscDrawAxis axis;
    PetscReal     dmboxlower[2], dmboxupper[2];
    PetscInt      dim;
    char          title[PETSC_MAX_PATH_LEN];

    PetscCall(DMSwarmGetCellDMByName(sw, "velocity", &celldm));
    PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
    PetscCall(DMGetDimension(vdm, &dim));
    PetscCall(DMGetBoundingBox(vdm, dmboxlower, dmboxupper));

    PetscCall(PetscSNPrintf(title, PETSC_MAX_PATH_LEN, "Cell %" PetscInt_FMT ": Velocity Distribution", ctx->velocity_monitor));
    PetscCall(PetscDrawCreate(comm, NULL, title, 400, 300, 400, 300, &drawcell_v));
    PetscCall(PetscDrawSetSave(drawcell_v, "ex4_cell_v"));
    PetscCall(PetscDrawSetFromOptions(drawcell_v));
    PetscCall(PetscDrawHGCreate(drawcell_v, (int)dim, &ctx->drawhgcell_v));
    PetscCall(PetscDrawHGCalcStats(ctx->drawhgcell_v, PETSC_TRUE));
    PetscCall(PetscDrawHGGetAxis(ctx->drawhgcell_v, &axis));
    PetscCall(PetscDrawHGSetNumberBins(ctx->drawhgcell_v, 21));
    PetscCall(PetscDrawAxisSetLabels(axis, "V_x Distribution", "V", "weight"));
    PetscCall(PetscDrawAxisSetLimits(axis, dmboxlower[0], dmboxupper[0], 0, 0));
    PetscCall(PetscDrawDestroy(&drawcell_v));
  }

  if (ctx->positions_monitor) {
    PetscDraw     draw;
    PetscDrawAxis axis;

    PetscCall(PetscDrawCreate(comm, NULL, "Particle Position", 0, 0, 400, 300, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_pos"));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawSPCreate(draw, 10, &ctx->drawspX));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawSPSetDimension(ctx->drawspX, 1));
    PetscCall(PetscDrawSPGetAxis(ctx->drawspX, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Particles", "x", "v"));
    PetscCall(PetscDrawSPReset(ctx->drawspX));
  }
  if (ctx->poisson_monitor) {
    Vec           rho, rhohat, phi;
    PetscDraw     draw;
    PetscDrawAxis axis;

    PetscCall(PetscDrawCreate(comm, NULL, "Electric_Field", 0, 0, 400, 300, &draw));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_E_spatial"));
    PetscCall(PetscDrawSPCreate(draw, 10, &ctx->drawspE));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawSPSetDimension(ctx->drawspE, 1));
    PetscCall(PetscDrawSPGetAxis(ctx->drawspE, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Particles", "x", "E"));
    PetscCall(PetscDrawSPReset(ctx->drawspE));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Charge Density", 0, 0, 400, 300, &ctx->viewerRho));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ctx->viewerRho, "rho_"));
    PetscCall(PetscViewerDrawGetDraw(ctx->viewerRho, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_rho_spatial"));
    PetscCall(PetscViewerSetFromOptions(ctx->viewerRho));
    PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "rho", &rho));
    PetscCall(PetscObjectSetName((PetscObject)rho, "charge_density"));
    PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "rho", &rho));

    PetscInt dim, N;

    PetscCall(DMGetDimension(ctx->dmPot, &dim));
    if (dim == 1) {
      PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "rhohat", &rhohat));
      PetscCall(VecGetSize(rhohat, &N));
      PetscCall(MatCreateFFT(comm, dim, &N, MATFFTW, &ctx->fftPot));
      PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "rhohat", &rhohat));
      PetscCall(MatCreateVecs(ctx->fftPot, &ctx->fftX, &ctx->fftY));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &ctx->fftReal));
    }

    PetscCall(PetscViewerDrawOpen(comm, NULL, "rhohat: Charge Density FT", 0, 0, 400, 300, &ctx->viewerRhoHat));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ctx->viewerRhoHat, "rhohat_"));
    PetscCall(PetscViewerDrawGetDraw(ctx->viewerRhoHat, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_rho_ft"));
    PetscCall(PetscViewerSetFromOptions(ctx->viewerRhoHat));
    PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "rhohat", &rhohat));
    PetscCall(PetscObjectSetName((PetscObject)rhohat, "charge_density_ft"));
    PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "rhohat", &rhohat));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Potential", 400, 0, 400, 300, &ctx->viewerPhi));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ctx->viewerPhi, "phi_"));
    PetscCall(PetscViewerDrawGetDraw(ctx->viewerPhi, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_phi_spatial"));
    PetscCall(PetscViewerSetFromOptions(ctx->viewerPhi));
    PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "phi", &phi));
    PetscCall(PetscObjectSetName((PetscObject)phi, "potential"));
    PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "phi", &phi));
  }
  if (ctx->momCtx.moment_field_monitor) {
    MomStruct *ms = &ctx->momCtx;
    Vec        n, p, e;
    Vec        nres, pres, eres;
    PetscDraw  draw;

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Number Density", 400, 0, 400, 300, &ms->viewerN));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ms->viewerN, "n_"));
    PetscCall(PetscViewerDrawGetDraw(ms->viewerN, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_n_spatial"));
    PetscCall(PetscViewerSetFromOptions(ms->viewerN));
    PetscCall(DMGetNamedGlobalVector(ms->dmN, "n", &n));
    PetscCall(PetscObjectSetName((PetscObject)n, "Number Density"));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "n", &n));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Momentum Density", 800, 0, 400, 300, &ms->viewerP));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ms->viewerP, "p_"));
    PetscCall(PetscViewerDrawGetDraw(ms->viewerP, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_p_spatial"));
    PetscCall(PetscViewerSetFromOptions(ms->viewerP));
    PetscCall(DMGetNamedGlobalVector(ms->dmP, "p", &p));
    PetscCall(PetscObjectSetName((PetscObject)p, "Momentum Density"));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "p", &p));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Emergy Density (Pressure)", 1200, 0, 400, 300, &ms->viewerE));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ms->viewerE, "e_"));
    PetscCall(PetscViewerDrawGetDraw(ms->viewerE, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_e_spatial"));
    PetscCall(PetscViewerSetFromOptions(ms->viewerE));
    PetscCall(DMGetNamedGlobalVector(ms->dmE, "e", &e));
    PetscCall(PetscObjectSetName((PetscObject)e, "Energy Density (Pressure)"));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "e", &e));

    PetscDrawAxis axis;

    PetscCall(PetscDrawCreate(comm, NULL, "Moment Residual", 0, 320, 400, 300, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_moment_res"));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawLGCreate(draw, 3, &ms->drawlgMomRes));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawLGGetAxis(ms->drawlgMomRes, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Moment Residial", "time", "Residual Norm"));
    PetscCall(PetscDrawLGSetLimits(ms->drawlgMomRes, 0., ctx->steps * ctx->stepSize, -8, 0));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Number Density Residual", 400, 300, 400, 300, &ms->viewerNRes));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ms->viewerNRes, "nres_"));
    PetscCall(PetscViewerDrawGetDraw(ms->viewerNRes, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_nres_spatial"));
    PetscCall(PetscViewerSetFromOptions(ms->viewerNRes));
    PetscCall(DMGetNamedGlobalVector(ms->dmN, "nres", &nres));
    PetscCall(PetscObjectSetName((PetscObject)nres, "Number Density Residual"));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "nres", &nres));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Momentum Density Residual", 800, 300, 400, 300, &ms->viewerPRes));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ms->viewerPRes, "pres_"));
    PetscCall(PetscViewerDrawGetDraw(ms->viewerPRes, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_pres_spatial"));
    PetscCall(PetscViewerSetFromOptions(ms->viewerPRes));
    PetscCall(DMGetNamedGlobalVector(ms->dmP, "pres", &pres));
    PetscCall(PetscObjectSetName((PetscObject)pres, "Momentum Density Residual"));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "pres", &pres));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Energy Density Residual", 1200, 300, 400, 300, &ms->viewerERes));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ms->viewerERes, "eres_"));
    PetscCall(PetscViewerDrawGetDraw(ms->viewerERes, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_eres_spatial"));
    PetscCall(PetscViewerSetFromOptions(ms->viewerERes));
    PetscCall(DMGetNamedGlobalVector(ms->dmE, "eres", &eres));
    PetscCall(PetscObjectSetName((PetscObject)eres, "Energy Density Residual"));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "eres", &eres));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyContext(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscDrawHGDestroy(&ctx->drawhgic_x));
  PetscCall(PetscDrawHGDestroy(&ctx->drawhgic_v));
  PetscCall(PetscDrawHGDestroy(&ctx->drawhgcell_v));

  PetscCall(PetscDrawLGDestroy(&ctx->drawlgE));
  PetscCall(PetscDrawLGDestroy(&ctx->drawlgEMom));
  PetscCall(PetscDrawSPDestroy(&ctx->drawspE));
  PetscCall(PetscDrawSPDestroy(&ctx->drawspX));
  PetscCall(PetscViewerDestroy(&ctx->viewerRho));
  PetscCall(PetscViewerDestroy(&ctx->viewerRhoHat));
  PetscCall(MatDestroy(&ctx->fftPot));
  PetscCall(VecDestroy(&ctx->fftX));
  PetscCall(VecDestroy(&ctx->fftY));
  PetscCall(ISDestroy(&ctx->fftReal));
  PetscCall(PetscViewerDestroy(&ctx->viewerPhi));
  PetscCall(PetscViewerDestroy(&ctx->momCtx.viewerN));
  PetscCall(PetscViewerDestroy(&ctx->momCtx.viewerP));
  PetscCall(PetscViewerDestroy(&ctx->momCtx.viewerE));
  PetscCall(PetscViewerDestroy(&ctx->momCtx.viewerNRes));
  PetscCall(PetscViewerDestroy(&ctx->momCtx.viewerPRes));
  PetscCall(PetscViewerDestroy(&ctx->momCtx.viewerERes));
  PetscCall(PetscDrawLGDestroy(&ctx->momCtx.drawlgMomRes));

  PetscCall(PetscBagDestroy(&ctx->bag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckNonNegativeWeights(DM sw, AppCtx *ctx)
{
  const PetscScalar *w;
  PetscInt           Np;

  PetscFunctionBeginUser;
  if (!ctx->checkweights) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  for (PetscInt p = 0; p < Np; ++p) PetscCheck(w[p] >= 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Particle %" PetscInt_FMT " has negative weight %g", p, w[p]);
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_Dirichlet(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] += 0.5 * PetscSqr(u_x[d]);
}

static PetscErrorCode computeFieldEnergy(DM dm, Vec u, PetscReal *En)
{
  PetscDS        ds;
  const PetscInt field = 0;
  PetscInt       Nf;
  void          *ctx;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &ctx));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCheck(Nf == 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "We currently only support 1 field, not %" PetscInt_FMT, Nf);
  PetscCall(PetscDSSetObjective(ds, field, &f0_Dirichlet));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, En, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* compute inverse error functions with maximum error of 2.35793 ulp
   https://stackoverflow.com/questions/27229371/inverse-error-function-in-c */
static float erfinvf(float a)
{
  float p, r, t;

  t = fmaf(a, 0.0f - a, 1.0f);
  t = PetscLogReal(t);
  if (PetscAbsReal(t) > 6.125f) {    // maximum ulp error = 2.35793
    p = 3.03697567e-10f;             //  0x1.4deb44p-32
    p = fmaf(p, t, 2.93243101e-8f);  //  0x1.f7c9aep-26
    p = fmaf(p, t, 1.22150334e-6f);  //  0x1.47e512p-20
    p = fmaf(p, t, 2.84108955e-5f);  //  0x1.dca7dep-16
    p = fmaf(p, t, 3.93552968e-4f);  //  0x1.9cab92p-12
    p = fmaf(p, t, 3.02698812e-3f);  //  0x1.8cc0dep-9
    p = fmaf(p, t, 4.83185798e-3f);  //  0x1.3ca920p-8
    p = fmaf(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = fmaf(p, t, 8.40016484e-1f);  //  0x1.ae16a4p-1
  } else {                           // maximum ulp error = 2.35002
    p = 5.43877832e-9f;              //  0x1.75c000p-28
    p = fmaf(p, t, 1.43285448e-7f);  //  0x1.33b402p-23
    p = fmaf(p, t, 1.22774793e-6f);  //  0x1.499232p-20
    p = fmaf(p, t, 1.12963626e-7f);  //  0x1.e52cd2p-24
    p = fmaf(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = fmaf(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = fmaf(p, t, 2.31468678e-3f);  //  0x1.2f6400p-9
    p = fmaf(p, t, 1.15392581e-2f);  //  0x1.7a1e50p-7
    p = fmaf(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = fmaf(p, t, 8.86226892e-1f);  //  0x1.c5bf88p-1
  }
  r = a * p;
  return r;
}

/* To equalize weights, we use non-uniform velocity meshes
    TODO Only works in 1D right now
  We assume a simple normalized Gaussian distribution
    f(x) = 1/sqrt{2 pi} e^{-x^2/2}.
  If the mesh runs [-M, M] and has 2N bins, then the weight in each bin is
    \int^{(b + 1) x/N - M}_{b x/N - M} f(x) dx
  We want to transform x such that the weight is uniform in each bin. If we let
    u = 1/sqrt{2 pi} \int^x_0 ds e^{-s^2 / 2}
  so that
    du = 1/sqrt{2 pi} e^{-x^2 / 2} dx
  then our integral becomes
      \int^{(b + 1) x/N - M}_{b x/N - M} 1/sqrt{2 pi} e^{-x^2/2} sqrt{2 pi} e^{x^2 / 2} du
    = \int^{(b + 1) x/N - M}_{b x/N - M} du
  which will be uniform for uniform spacing in u. We can express u in terms of the error function
    u = 1/sqrt{2 pi} \int^x_0 ds e^{-s^2 / 2}
      = erf(x / sqrt{2})
  so that
    x = sqrt{2} inverf(u)
  Then
    Delta u = erf(M / sqrt{2}) / N
  and
    x_b = sqrt{2} inverf(b Delta u)
  Now if originally x_b = M (b/N - 1), then we can transform it using
    x -> sqrt{2} inverf((x/M + 1) N Delta u - 1).
*/
static PetscErrorCode OptimizeVelocityGrid(DM sw, DM vdm)
{
  DM           cdm;
  PetscSection s, gs;
  Vec          coordinates, coordinatesLocal;
  PetscScalar *al, *ag;
  PetscReal    vmin[3], vmax[3], du;
  PetscInt     dim, dimV, vStart, vEnd, N;
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCall(DMGetDimension(sw, &dim));
  // Optimize the velocity grid
  PetscCall(DMGetDimension(vdm, &dimV));
  PetscCall(DMGetCoordinateDM(vdm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &s));
  PetscCall(DMGetGlobalSection(cdm, &gs));
  PetscCall(DMPlexGetDepthStratum(vdm, 0, &vStart, &vEnd));
  PetscCheck(dimV == dim, comm, PETSC_ERR_ARG_WRONG, "The swarm velocity dimension %" PetscInt_FMT " != %" PetscInt_FMT " velocity grid dimension", dim, dimV);
  PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
  N  = (vEnd - vStart - 1) / 2;
  du = PetscErfReal(vmax[0] / PETSC_SQRT2) / N;

  PetscCall(DMGetCoordinatesLocal(vdm, &coordinatesLocal));
  PetscCall(DMGetCoordinates(vdm, &coordinates));
  PetscCall(VecGetArray(coordinatesLocal, &al));
  PetscCall(VecGetArray(coordinates, &ag));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscInt  loff, goff;
    PetscReal arg;

    PetscCall(PetscSectionGetOffset(s, v, &loff));
    PetscCall(PetscSectionGetOffset(gs, v, &goff));
    arg = (al[loff] / vmax[0] + 1.) * N * du - 1.;
    if (arg > -1. + PETSC_SQRT_MACHINE_EPSILON && arg < 1. - PETSC_SQRT_MACHINE_EPSILON) al[loff] = PETSC_SQRT2 * erfinvf(arg);
    if (goff < 0) continue;
    arg = (ag[goff] / vmax[0] + 1.) * N * du - 1.;
    if (arg > -1. + PETSC_SQRT_MACHINE_EPSILON && arg < 1. - PETSC_SQRT_MACHINE_EPSILON) ag[goff] = PETSC_SQRT2 * erfinvf(arg);
  }
  PetscCall(VecRestoreArray(coordinatesLocal, &al));
  PetscCall(VecRestoreArray(coordinates, &ag));
  PetscCall(PetscGridHashDestroy(&((DM_Plex *)vdm->data)->lbox));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode OptimizeRemapGrid(DM sw)
{
  DM            vdm, rdm, cdm;
  DMSwarmCellDM celldm;
  PetscSection  s, gs;
  Vec           coordinates, coordinatesLocal;
  PetscScalar  *al, *ag;
  PetscReal     vmin[3], vmax[3], rmin[6], rmax[6], du;
  PetscInt      dim, dimR, vStart, vEnd, N;
  MPI_Comm      comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCall(DMGetDimension(sw, &dim));

  PetscCall(DMSwarmGetCellDMByName(sw, "velocity", &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
  PetscCall(DMPlexGetDepthStratum(vdm, 0, &vStart, &vEnd));
  PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
  N  = (vEnd - vStart - 1) / 2;
  du = PetscErfReal(vmax[0] / PETSC_SQRT2) / N;

  // Optimize the remap grid
  PetscCall(DMSwarmGetCellDMByName(sw, "remap", &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &rdm));
  PetscCall(DMGetDimension(rdm, &dimR));
  PetscCall(DMGetCoordinateDM(rdm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &s));
  PetscCall(DMGetGlobalSection(cdm, &gs));
  PetscCall(DMPlexGetDepthStratum(rdm, 0, &vStart, &vEnd));
  PetscCheck(dimR == dim + dim, comm, PETSC_ERR_ARG_WRONG, "The swarm velocity dimension %" PetscInt_FMT "^2 != %" PetscInt_FMT " remap grid dimension", dim, dimR);
  PetscCall(DMGetBoundingBox(rdm, rmin, rmax));
  PetscCall(DMGetCoordinatesLocal(rdm, &coordinatesLocal));
  PetscCall(DMGetCoordinates(rdm, &coordinates));
  PetscCall(VecGetArray(coordinatesLocal, &al));
  PetscCall(VecGetArray(coordinates, &ag));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscInt  loff, goff;
    PetscReal arg;

    PetscCall(PetscSectionGetOffset(s, v, &loff));
    PetscCall(PetscSectionGetOffset(gs, v, &goff));
    arg = (al[loff + dim] / rmax[dim] + 1.) * N * du - 1.;
    if (arg > -1. + PETSC_SQRT_MACHINE_EPSILON && arg < 1. - PETSC_SQRT_MACHINE_EPSILON) al[loff + dim] = PETSC_SQRT2 * erfinvf(arg);
    if (goff < 0) continue;
    arg = (ag[goff + dim] / rmax[dim] + 1.) * N * du - 1.;
    if (arg > -1. + PETSC_SQRT_MACHINE_EPSILON && arg < 1. - PETSC_SQRT_MACHINE_EPSILON) ag[goff + dim] = PETSC_SQRT2 * erfinvf(arg);
  }
  PetscCall(VecRestoreArray(coordinatesLocal, &al));
  PetscCall(VecRestoreArray(coordinates, &ag));
  PetscCall(PetscGridHashDestroy(&((DM_Plex *)rdm->data)->lbox));
  // Need an API for this
  rdm->coordinates[1].dim = -1;
  PetscCall(DMLocalizeCoordinates(rdm));
  PetscCall(DMSwarmCellDMSetSort(celldm, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// To avoid particle loss, we enlarge the velocity and remap grids if necessary
static PetscErrorCode EnlargeVelocityGrids(DM sw)
{
  DM              vdm, rdm;
  DMSwarmCellDM   celldm;
  const PetscReal safetyFactor = 1.1;
  PetscReal      *v, pvmin[3] = {0., 0., 0.}, pvmax[3] = {0., 0., 0.}, vmin[3], vmax[3], rmin[6], rmax[6], vfact = 1., rfact = 1.;
  PetscInt        dim, Np, dimV, dimR;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  // Get velocity bounding box for particles
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt d = 0; d < dim; ++d) {
      pvmin[d] = PetscMin(pvmax[d], v[p * dim + d]);
      pvmax[d] = PetscMax(pvmax[d], v[p * dim + d]);
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  // Check for particles outside the velocity grid
  PetscCall(DMSwarmGetCellDMByName(sw, "velocity", &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
  PetscCall(DMGetDimension(vdm, &dimV));
  PetscCheck(dimV == dim, comm, PETSC_ERR_ARG_WRONG, "The swarm velocity dimension %" PetscInt_FMT " != %" PetscInt_FMT " velocity grid dimension", dim, dimV);
  PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
  for (PetscInt d = 0; d < dim; ++d) {
    vfact = PetscMax(vfact, pvmax[d] / vmax[d]);
    vfact = PetscMax(vfact, pvmin[d] / vmin[d]);
  }
  if (vfact > 1.) {
    Vec coordinates, coordinatesLocal;

    vfact *= safetyFactor;
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Expanding velocity grid by %g\n", vfact));
    PetscCall(DMGetCoordinatesLocal(vdm, &coordinatesLocal));
    PetscCall(DMGetCoordinates(vdm, &coordinates));
    PetscCall(VecScale(coordinatesLocal, vfact));
    PetscCall(VecScale(coordinates, vfact));
    PetscCall(PetscGridHashDestroy(&((DM_Plex *)vdm->data)->lbox));
    PetscCall(DMSwarmCellDMSetSort(celldm, NULL));
  }
  // Check for particles outside the remap grid
  PetscCall(DMSwarmGetCellDMByName(sw, "remap", &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &rdm));
  PetscCall(DMGetDimension(rdm, &dimR));
  PetscCheck(dimR == dim + dim, comm, PETSC_ERR_ARG_WRONG, "The swarm velocity dimension %" PetscInt_FMT "^2 != %" PetscInt_FMT " remap grid dimension", dim, dimR);
  PetscCall(DMGetBoundingBox(rdm, rmin, rmax));
  for (PetscInt d = 0; d < dim; ++d) {
    if (PetscAbsReal(rmax[dim + d] - vmax[d]) > PETSC_SMALL) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "vmin:"));
      for (PetscInt e = 0; e < dim; ++e) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %g", vmin[e]));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "vmax:"));
      for (PetscInt e = 0; e < dim; ++e) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %g", vmax[e]));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "rmin:"));
      for (PetscInt e = 0; e < dim; ++e) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %g", rmin[dim + e]));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "rmax:"));
      for (PetscInt e = 0; e < dim; ++e) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %g", rmax[dim + e]));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
    }
    PetscCheck(PetscAbsReal(rmax[dim + d] - vmax[d]) < PETSC_SMALL, comm, PETSC_ERR_PLIB, "Bounding boxes are different");
    PetscCheck(PetscAbsReal(rmin[dim + d] - vmin[d]) < PETSC_SMALL, comm, PETSC_ERR_PLIB, "Bounding boxes are different");
    rfact = PetscMax(rfact, pvmax[d] / rmax[dim + d]);
    rfact = PetscMax(rfact, pvmin[d] / rmin[dim + d]);
  }
  if (rfact > 1.) {
    Vec coordinates, coordinatesLocal;

    rfact *= safetyFactor;
    PetscCheck(PetscAbsReal(rfact - vfact) < PETSC_SMALL, comm, PETSC_ERR_PLIB, "Factors are different");
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Expanding remap grid by %g\n", rfact));
    PetscCall(DMGetCoordinatesLocal(rdm, &coordinatesLocal));
    PetscCall(DMGetCoordinates(rdm, &coordinates));
    for (PetscInt d = 0; d < dim; ++d) {
      PetscCall(VecStrideScale(coordinatesLocal, dim + d, rfact));
      PetscCall(VecStrideScale(coordinates, dim + d, rfact));
    }
    PetscCall(DMGetCellCoordinatesLocal(rdm, &coordinatesLocal));
    PetscCall(DMGetCellCoordinates(rdm, &coordinates));
    if (coordinatesLocal) {
      for (PetscInt d = 0; d < dim; ++d) {
        PetscCall(VecStrideScale(coordinatesLocal, dim + d, rfact));
        PetscCall(VecStrideScale(coordinates, dim + d, rfact));
      }
    }
    PetscCall(PetscGridHashDestroy(&((DM_Plex *)rdm->data)->lbox));
    PetscCall(DMSwarmCellDMSetSort(celldm, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode computeVelocityFEMMoments(DM sw, PetscReal moments[], AppCtx *ctx)
{
  DMSwarmCellDM celldm;
  DM            vdm;
  Vec           u[1];
  const char   *fields[1] = {"w_q"};

  PetscFunctionBegin;
  PetscCall(DMSwarmSetCellDMActive(sw, "velocity"));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
  PetscCall(DMGetGlobalVector(vdm, &u[0]));
  PetscCall(DMSwarmProjectFields(sw, vdm, 1, fields, u, SCATTER_FORWARD));
  PetscCall(DMPlexComputeMoments(vdm, u[0], moments));
  PetscCall(DMRestoreGlobalVector(vdm, &u[0]));
  PetscCall(DMSwarmSetCellDMActive(sw, "space"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_grad_phi2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.;
  for (PetscInt d = 0; d < dim; ++d) f0[0] += PetscSqr(u_x[uOff_x[0] + d * dim + d]);
}

static PetscReal ComputeEmax(PetscReal t, PetscReal C, PetscReal gamma, PetscReal omega, PetscReal phi)
{
  return C * PetscExpReal(-gamma * t) * PetscAbsReal(PetscCosReal(omega * t - phi));
}

// Our model is E_max(t) = C e^{-gamma t} |cos(omega t - phi)|
static PetscErrorCode ComputeEmaxResidual(Tao tao, Vec x, Vec res, void *Ctx)
{
  EmaxCtx           *ctx = (EmaxCtx *)Ctx;
  const PetscScalar *a;
  PetscScalar       *F;
  PetscReal          C, gamma, omega, phi;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  PetscCall(VecGetArray(res, &F));
  C     = PetscRealPart(a[0]);
  gamma = PetscRealPart(a[1]);
  omega = PetscRealPart(a[2]);
  phi   = PetscRealPart(a[3]);
  PetscCall(VecRestoreArrayRead(x, &a));
  for (PetscInt i = ctx->s; i < ctx->e; ++i) F[i - ctx->s] = PetscPowReal(10., ctx->Emax[i * 2 + 0]) - ComputeEmax(ctx->t[i * 2 + 0], C, gamma, omega, phi);

  PetscCall(VecRestoreArray(res, &F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The Jacobian of the residual J = dr(x)/dx
static PetscErrorCode ComputeEmaxJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *Ctx)
{
  EmaxCtx           *ctx = (EmaxCtx *)Ctx;
  const PetscScalar *a;
  PetscScalar       *jac;
  PetscReal          C, gamma, omega, phi;
  const PetscInt     n = ctx->e - ctx->s;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  C     = PetscRealPart(a[0]);
  gamma = PetscRealPart(a[1]);
  omega = PetscRealPart(a[2]);
  phi   = PetscRealPart(a[3]);
  PetscCall(VecRestoreArrayRead(x, &a));
  PetscCall(MatDenseGetArray(J, &jac));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscInt k = (i + ctx->s) * 2 + 0;

    jac[i * 4 + 0] = -PetscExpReal(-gamma * ctx->t[k]) * PetscAbsReal(PetscCosReal(omega * ctx->t[k] - phi));
    jac[i * 4 + 1] = C * ctx->t[k] * PetscExpReal(-gamma * ctx->t[k]) * PetscAbsReal(PetscCosReal(omega * ctx->t[k] - phi));
    jac[i * 4 + 2] = C * ctx->t[k] * PetscExpReal(-gamma * ctx->t[k]) * (PetscCosReal(omega * ctx->t[k] - phi) < 0. ? -1. : 1.) * PetscSinReal(omega * ctx->t[k] - phi);
    jac[i * 4 + 3] = -C * PetscExpReal(-gamma * ctx->t[k]) * (PetscCosReal(omega * ctx->t[k] - phi) < 0. ? -1. : 1.) * PetscSinReal(omega * ctx->t[k] - phi);
  }
  PetscCall(MatDenseRestoreArray(J, &jac));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal ComputeLogEmax(PetscReal t, PetscReal C, PetscReal gamma, PetscReal omega, PetscReal phi)
{
  return (PetscLog10Real(C) - gamma * t * PetscLog10Real(PETSC_E) + PetscLog10Real(PetscAbsReal(PetscCosReal(omega * t - phi))));
}

// Our model is log_10 E_max(t) = log_10 C - gamma t log_10 e + log_10 |cos(omega t - phi)|
static PetscErrorCode ComputeLogEmaxResidual(Tao tao, Vec x, Vec res, void *Ctx)
{
  EmaxCtx           *ctx = (EmaxCtx *)Ctx;
  const PetscScalar *a;
  PetscScalar       *F;
  PetscReal          C, gamma, omega, phi;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  PetscCall(VecGetArray(res, &F));
  C     = PetscRealPart(a[0]);
  gamma = PetscRealPart(a[1]);
  omega = PetscRealPart(a[2]);
  phi   = PetscRealPart(a[3]);
  PetscCall(VecRestoreArrayRead(x, &a));
  for (PetscInt i = ctx->s; i < ctx->e; ++i) {
    if (C < 0) {
      F[i - ctx->s] = 1e10;
      continue;
    }
    F[i - ctx->s] = ctx->Emax[i * 2 + 0] - ComputeLogEmax(ctx->t[i * 2 + 0], C, gamma, omega, phi);
  }
  PetscCall(VecRestoreArray(res, &F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The Jacobian of the residual J = dr(x)/dp
static PetscErrorCode ComputeLogEmaxJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *Ctx)
{
  EmaxCtx           *ctx = (EmaxCtx *)Ctx;
  const PetscScalar *a;
  PetscScalar       *jac;
  PetscReal          C, omega, phi;
  const PetscInt     n = ctx->e - ctx->s;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  C     = PetscRealPart(a[0]);
  omega = PetscRealPart(a[2]);
  phi   = PetscRealPart(a[3]);
  PetscCall(VecRestoreArrayRead(x, &a));
  PetscCall(MatDenseGetArray(J, &jac));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscInt k = (i + ctx->s) * 2 + 0;

    jac[0 * n + i] = -1. / (PetscLog10Real(PETSC_E) * C);
    jac[1 * n + i] = ctx->t[k] * PetscLog10Real(PETSC_E);
    jac[2 * n + i] = (PetscCosReal(omega * ctx->t[k] - phi) < 0. ? -1. : 1.) * ctx->t[k] * PetscSinReal(omega * ctx->t[k] - phi) / (PetscLog10Real(PETSC_E) * PetscAbsReal(PetscCosReal(omega * ctx->t[k] - phi)));
    jac[3 * n + i] = -(PetscCosReal(omega * ctx->t[k] - phi) < 0. ? -1. : 1.) * PetscSinReal(omega * ctx->t[k] - phi) / (PetscLog10Real(PETSC_E) * PetscAbsReal(PetscCosReal(omega * ctx->t[k] - phi)));
  }
  PetscCall(MatDenseRestoreArray(J, &jac));
  PetscCall(MatViewFromOptions(J, NULL, "-emax_jac_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Our model is E_max(t) = A t + b
static PetscErrorCode ComputeLogEmaxResidual_TwoStream(Tao tao, Vec x, Vec res, void *Ctx)
{
  EmaxCtx           *ctx = (EmaxCtx *)Ctx;
  const PetscScalar *a;
  PetscScalar       *F;
  PetscReal          A, b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  PetscCall(VecGetArray(res, &F));
  b = PetscRealPart(a[0]);
  A = PetscRealPart(a[1]);
  PetscCall(VecRestoreArrayRead(x, &a));
  for (PetscInt i = ctx->s; i < ctx->e; ++i) {
    F[i - ctx->s] = ctx->Emax[i] - (A * ctx->t[i] + b);
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%d: %g = %g - (%g * %g + %g)\n", i - ctx->s, F[i - ctx->s], ctx->Emax[i], A, ctx->t[i], b));
  }
  PetscCall(VecRestoreArray(res, &F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The Jacobian of the residual J = dr(x)/dp
static PetscErrorCode ComputeLogEmaxJacobian_TwoStream(Tao tao, Vec x, Mat J, Mat Jpre, void *user)
{
  EmaxCtx       *ctx = (EmaxCtx *)user;
  PetscScalar   *jac;
  const PetscInt n = ctx->e - ctx->s;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(J, &jac));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscInt k = i + ctx->s;

    jac[0 * n + i] = -1.;
    jac[1 * n + i] = -ctx->t[k];
  }
  PetscCall(MatDenseRestoreArray(J, &jac));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorEField(TS ts, PetscInt step, PetscReal t, Vec U, void *Ctx)
{
  AppCtx     *ctx = (AppCtx *)Ctx;
  DM          sw;
  PetscScalar intESq;
  PetscReal   xEmax[2], yEmax[2];
  PetscReal  *E, *x, *weight;
  PetscReal   Enorm = 0., lgEnorm, lgEmax, sum = 0., Emax = 0., chargesum = 0.;
  PetscReal   pmoments[4]; /* \int f, \int v f, \int v^2 f */
  PetscInt   *species, dim, Np, gNp;
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  if (step < 0 || !ctx->validE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetSize(sw, &gNp));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));

  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt d = 0; d < 1; ++d) {
      PetscReal temp = PetscAbsReal(E[p * dim + d]);
      if (temp > Emax) Emax = temp;
    }
    Enorm += PetscSqrtReal(E[p * dim] * E[p * dim]);
    sum += E[p * dim];
    chargesum += ctx->charges[0] * weight[p];
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &Emax, 1, MPIU_REAL, MPIU_MAX, comm));
  lgEnorm = Enorm != 0 ? PetscLog10Real(Enorm) : -16.;
  lgEmax  = Emax != 0 ? PetscLog10Real(Emax) : ctx->drawlgEmin;

  PetscDS ds;
  Vec     phi;

  PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "phi", &phi));
  PetscCall(DMGetDS(ctx->dmPot, &ds));
  PetscCall(PetscDSSetObjective(ds, 0, &f0_grad_phi2));
  PetscCall(DMPlexComputeIntegralFEM(ctx->dmPot, phi, &intESq, ctx));
  PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "phi", &phi));

  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  xEmax[0] = t;
  yEmax[0] = lgEmax;
  xEmax[1] = t;
  yEmax[1] = ctx->C < 0. ? lgEmax : ComputeLogEmax(t, ctx->C, ctx->gamma, ctx->omega, ctx->phi);
  PetscCall(PetscDrawLGAddPoint(ctx->drawlgE, xEmax, yEmax));
  if (ctx->efield_monitor == E_MONITOR_FULL) {
    PetscDraw draw;

    PetscCall(PetscDrawLGDraw(ctx->drawlgE));
    PetscCall(PetscDrawLGGetDraw(ctx->drawlgE, &draw));
    if (ctx->saveOutput) PetscCall(PetscDrawSave(draw));

    PetscCall(DMSwarmComputeMoments(sw, "velocity", "w_q", pmoments));
    PetscCall(PetscPrintf(comm, "E: %f\t%+e\t%e\t%f\t%20.15e\t%f\t%f\t%f\t%20.15e\t%20.15e\t%20.15e\t%" PetscInt_FMT "\t(%" PetscInt_FMT ")\n", (double)t, (double)sum, (double)Enorm, (double)lgEnorm, (double)Emax, (double)lgEmax, (double)chargesum, (double)pmoments[0], (double)pmoments[1], (double)pmoments[1 + dim], (double)PetscSqrtReal(intESq), gNp, step));
    PetscCall(DMViewFromOptions(sw, NULL, "-sw_efield_view"));
  }

  // Compute decay rate and frequency
  char title[PETSC_MAX_PATH_LEN];

  PetscCall(PetscDrawLGGetData(ctx->drawlgE, NULL, &ctx->emaxCtx.e, &ctx->emaxCtx.t, (const PetscReal **)&ctx->emaxCtx.Emax));
  if (ctx->emaxCtx.el >= 0) ctx->emaxCtx.e = PetscMin(ctx->emaxCtx.e, ctx->emaxCtx.el);
  if (!rank && (ctx->emaxCtx.e > ctx->emaxCtx.s + 1) && !(ctx->emaxCtx.e % ctx->emaxCtx.per)) {
    Tao          tao;
    Mat          J;
    Vec          x, r;
    PetscScalar *a;
    PetscInt     Npm    = ctx->two_stream ? 2 : 4;
    PetscBool    fitLog = PETSC_TRUE, debug = PETSC_FALSE;

    PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
    PetscCall(TaoSetOptionsPrefix(tao, "emax_"));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, Npm, &x));
    PetscCall(TaoSetSolution(tao, x));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, ctx->emaxCtx.e - ctx->emaxCtx.s, &r));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, ctx->emaxCtx.e - ctx->emaxCtx.s, Npm, NULL, &J));
    PetscCall(VecGetArray(x, &a));
    if (ctx->two_stream) {
      if (fitLog) PetscCall(TaoSetResidualRoutine(tao, r, ComputeLogEmaxResidual_TwoStream, &ctx->emaxCtx));
      if (fitLog) PetscCall(TaoSetJacobianResidualRoutine(tao, J, J, ComputeLogEmaxJacobian_TwoStream, &ctx->emaxCtx));
      a[0] = 1.;
      a[1] = -0.1;
    } else {
      if (fitLog) PetscCall(TaoSetResidualRoutine(tao, r, ComputeLogEmaxResidual, &ctx->emaxCtx));
      else PetscCall(TaoSetResidualRoutine(tao, r, ComputeEmaxResidual, &ctx->emaxCtx));
      PetscCall(VecDestroy(&r));
      if (fitLog) PetscCall(TaoSetJacobianResidualRoutine(tao, J, J, ComputeLogEmaxJacobian, &ctx->emaxCtx));
      else PetscCall(TaoSetJacobianResidualRoutine(tao, J, J, ComputeEmaxJacobian, &ctx->emaxCtx));
      PetscCall(MatDestroy(&J));
      a[0] = 0.02;
      a[1] = 0.15;
      a[2] = 1.4;
      a[3] = 0.45;
    }
    PetscCall(VecRestoreArray(x, &a));
    PetscCall(TaoSetFromOptions(tao));
    if (ctx->two_stream) {
      PetscCall(VecGetArray(x, &a));
      PetscCall(PetscLinearRegression(ctx->emaxCtx.e - ctx->emaxCtx.s, &ctx->emaxCtx.t[ctx->emaxCtx.s], &ctx->emaxCtx.Emax[ctx->emaxCtx.s], &a[1], &a[0]));
      PetscCall(VecRestoreArray(x, &a));
    } else PetscCall(TaoSolve(tao));
    if (debug) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "t = ["));
      for (PetscInt i = 0; i < ctx->emaxCtx.e; ++i) {
        if (i > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", ctx->emaxCtx.t[i * 2 + 0]));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "]\n"));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Emax = ["));
      for (PetscInt i = 0; i < ctx->emaxCtx.e; ++i) {
        if (i > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", ctx->emaxCtx.Emax[i * 2 + 0]));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "]\n"));
    }
    PetscCall(VecGetArray(x, &a));
    ctx->C     = a[0];
    ctx->gamma = a[1];
    ctx->omega = a[2];
    ctx->phi   = a[3];
    if (ctx->efield_monitor == E_MONITOR_FULL) {
      if (ctx->two_stream) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Emax Fit: A %g b %g\n", a[1], a[0]));
        PetscCall(PetscSNPrintf(title, PETSC_MAX_PATH_LEN, "Max Electric Field A %.4g b %.4g", a[1], a[0]));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Emax Fit: gamma %g omega %g C %g phi %g\n", a[1], a[2], a[0], a[3]));
        PetscCall(PetscSNPrintf(title, PETSC_MAX_PATH_LEN, "Max Electric Field gamma %.4g omega %.4g", a[1], a[2]));
      }
    }
    PetscCall(VecRestoreArray(x, &a));
    PetscCall(VecDestroy(&x));
    PetscCall(TaoDestroy(&tao));
    // Rewrite the line graph
    for (PetscInt i = 0; i < ctx->emaxCtx.e; ++i) ctx->emaxCtx.Emax[i * 2 + 1] = ComputeLogEmax(ctx->emaxCtx.t[i * 2 + 1], ctx->C, ctx->gamma, ctx->omega, ctx->phi);
  }
  if (ctx->efield_monitor == E_MONITOR_FULL && (ctx->emaxCtx.e > ctx->emaxCtx.s + 1) && !(ctx->emaxCtx.e % ctx->emaxCtx.per)) {
    PetscDraw     draw;
    PetscDrawAxis axis;

    PetscCall(PetscDrawLGGetDraw(ctx->drawlgE, &draw));
    PetscCall(PetscDrawSetTitle(draw, title));
    PetscCall(PetscDrawLGGetAxis(ctx->drawlgE, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, title, "time", "E_max"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorMomentEField(TS ts, PetscInt step, PetscReal t, Vec U, void *Ctx)
{
  AppCtx   *ctx = (AppCtx *)Ctx;
  DM        dm;
  Vec       phi, locE;
  PetscReal lgEmax;

  PetscFunctionBeginUser;
  if (step < 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &phi));
  // TODO Should map to local phi
  PetscCall(DMGetLocalVector(ctx->dmPot, &locE));
  PetscCall(DMPlexComputeClementInterpolant(ctx->dmPot, phi, locE));
  PetscCall(VecMax(locE, NULL, &lgEmax));
  lgEmax = PetscLog10Real(PetscAbsReal(lgEmax));
  PetscCall(DMRestoreLocalVector(ctx->dmPot, &locE));
  PetscCall(PetscDrawLGAddPoint(ctx->drawlgEMom, &t, &lgEmax));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time %g max Log|E| %g\n", t, lgEmax));
  if (ctx->efield_monitor == E_MONITOR_FULL) {
    PetscDraw draw;

    PetscCall(PetscDrawLGDraw(ctx->drawlgEMom));
    PetscCall(PetscDrawLGGetDraw(ctx->drawlgEMom, &draw));
    PetscCall(PetscDrawSave(draw));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorMoments(TS ts, PetscInt step, PetscReal t, Vec U, void *Ctx)
{
  AppCtx   *ctx = (AppCtx *)Ctx;
  DM        sw;
  PetscReal pmoments[4], fmoments[4]; /* \int f, \int v f, \int v^2 f */

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetDM(ts, &sw));

  PetscCall(DMSwarmComputeMoments(sw, "velocity", "w_q", pmoments));
  PetscCall(computeVelocityFEMMoments(sw, fmoments, ctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%f\t%f\t%f\t%f\t%f\t%f\t%f\n", (double)t, (double)pmoments[0], (double)pmoments[1], (double)pmoments[3], (double)fmoments[0], (double)fmoments[1], (double)fmoments[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, PetscCtx ctx)
{
  u[0] = 0.0;
  return PETSC_SUCCESS;
}

/*
    M_p w_p
      - Make M_p with "moments"
      - Get w_p from Swarm
    M_p v_p w_p
      - Get v_p from Swarm
      - pointwise multiply v_p and w_p
    M_p (v_p - (\sum_j p_F \phi_j(x_p)) / m (\sum_k n_F \phi_k(x_p)))^2 w_p
      - ProjectField(sw, {n, p} U, {v_p} A, tmp_p)
      - pointwise multiply tmp_p and w_p

  Projection works for swarms
    Fields are FE from the CellDM, and aux fields are the swarm fields
*/
static PetscErrorCode ComputeMomentFields_Swarm(TS ts)
{
  AppCtx    *ctx;
  MomStruct *ms;
  DM         sw;
  KSP        ksp, kspE;
  Mat        M_p, D_p, M_pe, D_pe;
  Vec        f, v, E, tmpMom;
  Vec        m, mold, mfluxold, mres, n, nrhs, nflux, nres, p, prhs, pflux, pres, e, erhs, eflux, eres;
  PetscReal  dt, t;
  PetscInt   Nts;

  PetscFunctionBegin;
  PetscCall(TSGetStepNumber(ts, &Nts));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &ctx));
  ms = &ctx->momCtx;
  PetscCall(DMSwarmSetCellDMActive(sw, "moment fields"));
  PetscCall(DMSwarmMigrate(sw, PETSC_FALSE));
  // TODO In higher dimensions, we will have to create different M_p and D_p for each field
  PetscCall(DMCreateMassMatrix(sw, ms->dmN, &M_p));
  PetscCall(DMCreateGradientMatrix(sw, ms->dmN, &D_p));
  PetscCall(DMCreateMassMatrix(sw, ms->dmE, &M_pe));
  PetscCall(DMCreateGradientMatrix(sw, ms->dmE, &D_pe));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "E_field", &E));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));

  PetscCall(MatViewFromOptions(ms->MN, NULL, "-mn_view"));
  PetscCall(MatViewFromOptions(ms->MP, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(ms->ME, NULL, "-me_view"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));

  PetscCall(DMGetGlobalVector(ms->dmN, &nrhs));
  PetscCall(DMGetGlobalVector(ms->dmN, &nflux));
  PetscCall(PetscObjectSetName((PetscObject)nrhs, "Weak number density"));
  PetscCall(DMGetNamedGlobalVector(ms->dmN, "n", &n));
  PetscCall(DMGetGlobalVector(ms->dmP, &prhs));
  PetscCall(DMGetGlobalVector(ms->dmP, &pflux));
  PetscCall(PetscObjectSetName((PetscObject)prhs, "Weak momentum density"));
  PetscCall(DMGetNamedGlobalVector(ms->dmP, "p", &p));
  PetscCall(DMGetGlobalVector(ms->dmE, &erhs));
  PetscCall(DMGetGlobalVector(ms->dmE, &eflux));
  PetscCall(PetscObjectSetName((PetscObject)erhs, "Weak energy density (pressure)"));
  PetscCall(DMGetNamedGlobalVector(ms->dmE, "e", &e));

  // Compute moments and fluxes
  PetscCall(VecDuplicate(f, &tmpMom));

  PetscCall(MatMultTranspose(M_p, f, nrhs));

  PetscCall(VecPointwiseMult(tmpMom, f, v));
  PetscCall(MatMultTranspose(M_p, tmpMom, prhs));
  PetscCall(MatMultTranspose(D_p, tmpMom, nflux));

  PetscCall(VecPointwiseMult(tmpMom, tmpMom, v));
  PetscCall(MatMultTranspose(M_pe, tmpMom, erhs));
  PetscCall(MatMultTranspose(D_p, tmpMom, pflux));

  PetscCall(VecPointwiseMult(tmpMom, tmpMom, v));
  PetscCall(MatMultTranspose(D_pe, tmpMom, eflux));

  PetscCall(VecPointwiseMult(tmpMom, f, E));
  PetscCall(MatMultTransposeAdd(M_p, tmpMom, pflux, pflux));

  PetscCall(VecPointwiseMult(tmpMom, v, E));
  PetscCall(VecScale(tmpMom, 2.));
  PetscCall(MatMultTransposeAdd(M_pe, tmpMom, eflux, eflux));

  PetscCall(VecDestroy(&tmpMom));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "E_field", &E));

  PetscCall(MatDestroy(&M_p));
  PetscCall(MatDestroy(&D_p));
  PetscCall(MatDestroy(&M_pe));
  PetscCall(MatDestroy(&D_pe));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "mom_proj_"));
  PetscCall(KSPSetOperators(ksp, ms->MN, ms->MN));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, nrhs, n));
  PetscCall(KSPSetOperators(ksp, ms->MP, ms->MP));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, prhs, p));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &kspE));
  PetscCall(KSPSetOptionsPrefix(kspE, "mom_proj_"));
  PetscCall(KSPSetOperators(kspE, ms->ME, ms->ME));
  PetscCall(KSPSetFromOptions(kspE));
  PetscCall(KSPSolve(kspE, erhs, e));
  PetscCall(KSPDestroy(&kspE));
  PetscCall(DMRestoreGlobalVector(ms->dmN, &nrhs));
  PetscCall(DMRestoreGlobalVector(ms->dmP, &prhs));
  PetscCall(DMRestoreGlobalVector(ms->dmE, &erhs));

  // Check moment residual
  // TODO Fix global2local here
  PetscReal res[3], logres[3];

  PetscCall(DMGetGlobalVector(ctx->momCtx.dm, &m));
  PetscCall(VecISCopy(m, ms->isN, SCATTER_FORWARD, n));
  PetscCall(VecISCopy(m, ms->isP, SCATTER_FORWARD, p));
  PetscCall(VecISCopy(m, ms->isE, SCATTER_FORWARD, e));
  PetscCall(DMGetNamedGlobalVector(ctx->momCtx.dm, "mold", &mold));
  PetscCall(DMGetNamedGlobalVector(ctx->momCtx.dm, "mfluxold", &mfluxold));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "n", &n));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "p", &p));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "e", &e));
  if (!Nts) goto end;

  // e = \Tr{\tau}
  // M_p w^{k+1} - M_p w^k - \Delta t D_p (w^k \vb{v}^k) = 0
  // M_p \vb{p}^{k+1} - M_p \vb{p}^k - \Delta t D_p \tau - e \Delta t M_p \left( n \vb{E} \right) = 0
  // M_p e^{k+1} - M_p e^k - \Delta t D_p \vb{Q} - 2 e \Delta t M_p \left( \vb{p} \cdot \vb{E} \right) = 0
  PetscCall(DMGetGlobalVector(ctx->momCtx.dm, &mres));
  PetscCall(VecCopy(mfluxold, mres));
  PetscCall(VecAXPBYPCZ(mres, 1. / dt, -1. / dt, -1., m, mold));

  PetscCall(DMGetNamedGlobalVector(ms->dmN, "nres", &nres));
  PetscCall(DMGetNamedGlobalVector(ms->dmP, "pres", &pres));
  PetscCall(DMGetNamedGlobalVector(ms->dmE, "eres", &eres));
  PetscCall(VecISCopy(mres, ms->isN, SCATTER_REVERSE, nres));
  PetscCall(VecISCopy(mres, ms->isP, SCATTER_REVERSE, pres));
  PetscCall(VecISCopy(mres, ms->isE, SCATTER_REVERSE, eres));
  PetscCall(VecNorm(nres, NORM_2, &res[0]));
  PetscCall(VecNorm(pres, NORM_2, &res[1]));
  PetscCall(VecNorm(eres, NORM_2, &res[2]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "Mass Residual: %g\n", (double)res[0]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "Momentum Residual: %g\n", (double)res[1]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "Energy Residual: %g\n", (double)res[2]));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "nres", &nres));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "pres", &pres));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "eres", &eres));
  PetscCall(DMRestoreGlobalVector(ctx->momCtx.dm, &mres));

  for (PetscInt i = 0; i < 3; ++i) logres[i] = PetscLog10Real(res[i]);
  PetscCall(PetscDrawLGAddCommonPoint(ms->drawlgMomRes, t, logres));
  PetscCall(PetscDrawLGDraw(ms->drawlgMomRes));
  {
    PetscDraw draw;

    PetscCall(PetscDrawLGGetDraw(ms->drawlgMomRes, &draw));
    if (ctx->saveOutput) PetscCall(PetscDrawSave(draw));
  }

  if (1) {
    Vec      mnew;
    PetscInt k;

    PetscCall(DMGetGlobalVector(ctx->momCtx.dm, &mnew));
    PetscCall(VecCopy(mold, mnew));
    PetscCall(TSGetMaxSteps(ctx->momCtx.ts, &k));
    PetscCall(TSSetTimeStep(ctx->momCtx.ts, dt));
    PetscCall(TSSetMaxSteps(ctx->momCtx.ts, k + 1));
    PetscCall(TSSolve(ctx->momCtx.ts, mnew));

    PetscCall(VecAXPY(mnew, -1, m));
    PetscCall(DMGetNamedGlobalVector(ms->dmN, "nres", &nres));
    PetscCall(DMGetNamedGlobalVector(ms->dmP, "pres", &pres));
    PetscCall(DMGetNamedGlobalVector(ms->dmE, "eres", &eres));
    PetscCall(VecISCopy(mnew, ms->isN, SCATTER_REVERSE, nres));
    PetscCall(VecISCopy(mnew, ms->isP, SCATTER_REVERSE, pres));
    PetscCall(VecISCopy(mnew, ms->isE, SCATTER_REVERSE, eres));
    PetscCall(VecNorm(nres, NORM_2, &res[0]));
    PetscCall(VecNorm(pres, NORM_2, &res[1]));
    PetscCall(VecNorm(eres, NORM_2, &res[2]));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "New Mass Residual: %g\n", (double)res[0]));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "New Momentum Residual: %g\n", (double)res[1]));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "New Energy Residual: %g\n", (double)res[2]));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "nres", &nres));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "pres", &pres));
    PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "eres", &eres));

    PetscCall(DMRestoreGlobalVector(ctx->momCtx.dm, &mnew));
  }
end:
  PetscCall(VecCopy(m, mold));
  PetscCall(DMRestoreGlobalVector(ctx->momCtx.dm, &m));
  PetscCall(DMRestoreNamedGlobalVector(ctx->momCtx.dm, "mold", &mold));
  PetscCall(VecISCopy(mfluxold, ms->isN, SCATTER_FORWARD, nflux));
  PetscCall(VecISCopy(mfluxold, ms->isP, SCATTER_FORWARD, pflux));
  PetscCall(VecISCopy(mfluxold, ms->isE, SCATTER_FORWARD, eflux));
  PetscCall(DMRestoreNamedGlobalVector(ctx->momCtx.dm, "mfluxold", &mfluxold));
  PetscCall(DMRestoreGlobalVector(ms->dmN, &nflux));
  PetscCall(DMRestoreGlobalVector(ms->dmP, &pflux));
  PetscCall(DMRestoreGlobalVector(ms->dmE, &eflux));
  PetscCall(DMSwarmSetCellDMActive(sw, "space"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeMomentFields_Plex(TS ts, Vec U, MomStruct *ms)
{
  Vec n, p, e;

  PetscFunctionBeginUser;
  PetscCall(DMGetNamedGlobalVector(ms->dmN, "n", &n));
  PetscCall(DMGetNamedGlobalVector(ms->dmP, "p", &p));
  PetscCall(DMGetNamedGlobalVector(ms->dmE, "e", &e));
  PetscCall(VecISCopy(U, ms->isN, SCATTER_REVERSE, n));
  PetscCall(VecISCopy(U, ms->isP, SCATTER_REVERSE, p));
  PetscCall(VecISCopy(U, ms->isE, SCATTER_REVERSE, e));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "n", &n));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "p", &p));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "e", &e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeMomentFields(TS ts, Vec U, void *ctx)
{
  DM        dm;
  PetscBool isplex;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(U, &dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isplex));
  if (isplex) PetscCall(ComputeMomentFields_Plex(ts, U, ctx));
  else PetscCall(ComputeMomentFields_Swarm(ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorMomentFields(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  MomStruct *ms = (MomStruct *)ctx;
  Vec        n, p, e;
  Vec        nres, pres, eres;

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ComputeMomentFields(ts, U, ms));

  PetscCall(DMGetNamedGlobalVector(ms->dmN, "n", &n));
  PetscCall(VecView(n, ms->viewerN));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "n", &n));

  PetscCall(DMGetNamedGlobalVector(ms->dmP, "p", &p));
  PetscCall(VecView(p, ms->viewerP));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "p", &p));

  PetscCall(DMGetNamedGlobalVector(ms->dmE, "e", &e));
  PetscCall(VecView(e, ms->viewerE));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "e", &e));

  PetscCall(DMGetNamedGlobalVector(ms->dmN, "nres", &nres));
  PetscCall(VecView(nres, ms->viewerNRes));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "nres", &nres));

  PetscCall(DMGetNamedGlobalVector(ms->dmP, "pres", &pres));
  PetscCall(VecView(pres, ms->viewerPRes));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "pres", &pres));

  PetscCall(DMGetNamedGlobalVector(ms->dmE, "eres", &eres));
  PetscCall(VecView(eres, ms->viewerERes));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "eres", &eres));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorMomentFields_Continuum(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  MomStruct *ms = (MomStruct *)ctx;
  Vec        n, p, e;

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetNamedGlobalVector(ms->dmN, "nres", &n));
  PetscCall(PetscObjectSetName((PetscObject)n, "Monitored Number Density"));
  PetscCall(DMGetNamedGlobalVector(ms->dmP, "pres", &p));
  PetscCall(PetscObjectSetName((PetscObject)p, "Monitored Momentum Density"));
  PetscCall(DMGetNamedGlobalVector(ms->dmE, "eres", &e));
  PetscCall(PetscObjectSetName((PetscObject)e, "Monitored Energy Density"));

  PetscCall(VecISCopy(U, ms->isN, SCATTER_REVERSE, n));
  PetscCall(VecISCopy(U, ms->isP, SCATTER_REVERSE, p));
  PetscCall(VecISCopy(U, ms->isE, SCATTER_REVERSE, e));

  PetscCall(VecView(n, ms->viewerNRes));
  PetscCall(VecView(p, ms->viewerPRes));
  PetscCall(VecView(e, ms->viewerERes));

  PetscCall(PetscObjectSetName((PetscObject)n, "Number Density Residual"));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmN, "nres", &n));
  PetscCall(PetscObjectSetName((PetscObject)p, "Momentum Density Residual"));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmP, "pres", &p));
  PetscCall(PetscObjectSetName((PetscObject)e, "Energy Density Residual"));
  PetscCall(DMRestoreNamedGlobalVector(ms->dmE, "eres", &e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorInitialConditions(TS ts, PetscInt step, PetscReal t, Vec U, void *Ctx)
{
  AppCtx    *ctx = (AppCtx *)Ctx;
  DM         sw;
  PetscDraw  drawic_x, drawic_v;
  PetscReal *weight, *pos, *vel;
  PetscInt   dim, Np;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS); /* -1 indicates interpolated solution */
  if (step == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));

    PetscCall(PetscDrawHGReset(ctx->drawhgic_x));
    PetscCall(PetscDrawHGGetDraw(ctx->drawhgic_x, &drawic_x));
    PetscCall(PetscDrawClear(drawic_x));
    PetscCall(PetscDrawFlush(drawic_x));

    PetscCall(PetscDrawHGReset(ctx->drawhgic_v));
    PetscCall(PetscDrawHGGetDraw(ctx->drawhgic_v, &drawic_v));
    PetscCall(PetscDrawClear(drawic_v));
    PetscCall(PetscDrawFlush(drawic_v));

    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&pos));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
    for (PetscInt p = 0; p < Np; ++p) {
      PetscCall(PetscDrawHGAddWeightedValue(ctx->drawhgic_x, pos[p * dim], weight[p]));
      PetscCall(PetscDrawHGAddWeightedValue(ctx->drawhgic_v, vel[p * dim], weight[p]));
    }
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&pos));
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));

    PetscCall(PetscDrawHGDraw(ctx->drawhgic_x));
    if (ctx->saveOutput) PetscCall(PetscDrawHGSave(ctx->drawhgic_x));
    PetscCall(PetscDrawHGDraw(ctx->drawhgic_v));
    if (ctx->saveOutput) PetscCall(PetscDrawHGSave(ctx->drawhgic_v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Right now, make the complete velocity histogram
PetscErrorCode MonitorVelocity(TS ts, PetscInt step, PetscReal t, Vec U, void *Ctx)
{
  AppCtx      *ctx = (AppCtx *)Ctx;
  DM           sw, dm;
  Vec          ks;
  PetscProbFn *cdf;
  PetscDraw    drawcell_v;
  PetscScalar *ksa;
  PetscReal   *weight, *vel;
  PetscInt    *pidx;
  PetscInt     dim, Npc, cStart, cEnd, cell = ctx->velocity_monitor;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));

  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), &ks));
  PetscCall(PetscObjectSetName((PetscObject)ks, "KS Statistic by Cell"));
  PetscCall(VecSetSizes(ks, cEnd - cStart, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(ks));
  switch (dim) {
  case 1:
    //cdf = PetscCDFMaxwellBoltzmann1D;
    cdf = PetscCDFGaussian1D;
    break;
  case 2:
    cdf = PetscCDFMaxwellBoltzmann2D;
    break;
  case 3:
    cdf = PetscCDFMaxwellBoltzmann3D;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported", dim);
  }

  PetscCall(PetscDrawHGReset(ctx->drawhgcell_v));
  PetscCall(PetscDrawHGGetDraw(ctx->drawhgcell_v, &drawcell_v));
  PetscCall(PetscDrawClear(drawcell_v));
  PetscCall(PetscDrawFlush(drawcell_v));

  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(VecGetArrayWrite(ks, &ksa));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    Vec          cellv, cellw;
    PetscScalar *cella, *cellaw;
    PetscReal    totWgt = 0.;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(VecCreate(PETSC_COMM_SELF, &cellv));
    PetscCall(VecSetBlockSize(cellv, dim));
    PetscCall(VecSetSizes(cellv, Npc * dim, Npc));
    PetscCall(VecSetFromOptions(cellv));
    PetscCall(VecCreate(PETSC_COMM_SELF, &cellw));
    PetscCall(VecSetSizes(cellw, Npc, Npc));
    PetscCall(VecSetFromOptions(cellw));
    PetscCall(VecGetArrayWrite(cellv, &cella));
    PetscCall(VecGetArrayWrite(cellw, &cellaw));
    for (PetscInt q = 0; q < Npc; ++q) {
      const PetscInt p = pidx[q];
      if (c == cell) PetscCall(PetscDrawHGAddWeightedValue(ctx->drawhgcell_v, vel[p * dim], weight[p]));
      for (PetscInt d = 0; d < dim; ++d) cella[q * dim + d] = vel[p * dim + d];
      cellaw[q] = weight[p];
      totWgt += weight[p];
    }
    PetscCall(VecRestoreArrayWrite(cellv, &cella));
    PetscCall(VecRestoreArrayWrite(cellw, &cellaw));
    PetscCall(VecScale(cellw, 1. / totWgt));
    PetscCall(PetscProbComputeKSStatisticWeighted(cellv, cellw, cdf, &ksa[c - cStart]));
    PetscCall(VecDestroy(&cellv));
    PetscCall(VecDestroy(&cellw));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
  }
  PetscCall(VecRestoreArrayWrite(ks, &ksa));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmSortRestoreAccess(sw));

  PetscReal minalpha, maxalpha;
  PetscInt  mincell, maxcell;

  PetscCall(VecFilter(ks, PETSC_SMALL));
  PetscCall(VecMin(ks, &mincell, &minalpha));
  PetscCall(VecMax(ks, &maxcell, &maxalpha));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "Step %" PetscInt_FMT ": Min/Max KS statistic %g/%g in cell %" PetscInt_FMT "/%" PetscInt_FMT "\n", step, minalpha, maxalpha, mincell, maxcell));
  PetscCall(VecViewFromOptions(ks, NULL, "-ks_view"));
  PetscCall(VecDestroy(&ks));

  PetscCall(PetscDrawHGDraw(ctx->drawhgcell_v));
  if (ctx->saveOutput) PetscCall(PetscDrawHGSave(ctx->drawhgcell_v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorPositions_2D(TS ts, PetscInt step, PetscReal t, Vec U, void *Ctx)
{
  AppCtx         *ctx = (AppCtx *)Ctx;
  DM              dm, sw;
  PetscDrawAxis   axis;
  char            title[1024];
  PetscScalar    *x, *v, *weight;
  PetscReal       lower[3], upper[3], speed;
  const PetscInt *s;
  PetscInt        dim, cStart, cEnd, c;

  PetscFunctionBeginUser;
  if (step > 0 && step % ctx->ostep == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMSwarmGetCellDM(sw, &dm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMGetBoundingBox(dm, lower, upper));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
    PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&s));
    PetscCall(DMSwarmSortGetAccess(sw));
    PetscCall(PetscDrawSPReset(ctx->drawspX));
    PetscCall(PetscDrawSPGetAxis(ctx->drawspX, &axis));
    PetscCall(PetscSNPrintf(title, 1024, "Step %" PetscInt_FMT " Time: %g", step, (double)t));
    PetscCall(PetscDrawAxisSetLabels(axis, title, "x", "v"));
    PetscCall(PetscDrawSPSetLimits(ctx->drawspX, lower[0], upper[0], lower[1], upper[1]));
    PetscCall(PetscDrawSPSetLimits(ctx->drawspX, lower[0], upper[0], -12, 12));
    for (c = 0; c < cEnd - cStart; ++c) {
      PetscInt *pidx, Npc, q;
      PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
      for (q = 0; q < Npc; ++q) {
        const PetscInt p = pidx[q];
        if (s[p] == 0) {
          speed = 0.;
          for (PetscInt d = 0; d < dim; ++d) speed += PetscSqr(v[p * dim + d]);
          speed = PetscSqrtReal(speed);
          if (dim == 1) {
            PetscCall(PetscDrawSPAddPointColorized(ctx->drawspX, &x[p * dim], &v[p * dim], &speed));
          } else {
            PetscCall(PetscDrawSPAddPointColorized(ctx->drawspX, &x[p * dim], &x[p * dim + 1], &speed));
          }
        } else if (s[p] == 1) {
          PetscCall(PetscDrawSPAddPoint(ctx->drawspX, &x[p * dim], &v[p * dim]));
        }
      }
      PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
    }
    PetscCall(PetscDrawSPDraw(ctx->drawspX, PETSC_TRUE));
    PetscDraw draw;
    PetscCall(PetscDrawSPGetDraw(ctx->drawspX, &draw));
    if (ctx->saveOutput) PetscCall(PetscDrawSave(draw));
    if (ctx->drawPosGrid) {
      DMSwarmCellDM celldm;
      DM            rdm;
      PetscInt      cStart, cEnd;

      PetscCall(DMSwarmGetCellDMByName(sw, "remap", &celldm));
      PetscCall(DMSwarmCellDMGetDM(celldm, &rdm));
      PetscCall(DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd));
      for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscScalar       *coords = NULL;
        const PetscScalar *coords_arr;
        PetscInt           numCoords;
        PetscBool          isDG;

        PetscCall(DMPlexGetCellCoordinates(rdm, c, &isDG, &numCoords, &coords_arr, &coords));
        PetscCall(DMPlexDrawCell(rdm, draw, PETSC_DETERMINE, -2, c, coords));
        PetscCall(DMPlexRestoreCellCoordinates(rdm, c, &isDG, &numCoords, &coords_arr, &coords));
      }
      PetscCall(PetscDrawFlush(draw));
      PetscCall(PetscDrawPause(draw));
    }
    PetscCall(DMSwarmSortRestoreAccess(sw));
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
    PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&s));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorPoisson(TS ts, PetscInt step, PetscReal t, Vec U, void *Ctx)
{
  AppCtx *ctx = (AppCtx *)Ctx;
  DM      dm, sw;

  PetscFunctionBeginUser;
  if (step > 0 && step % ctx->ostep == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMSwarmGetCellDM(sw, &dm));

    if (ctx->validE) {
      PetscScalar *x, *E, *weight;
      PetscReal    lower[3], upper[3], xval;
      PetscDraw    draw;
      PetscInt     dim, cStart, cEnd;

      PetscCall(DMGetDimension(dm, &dim));
      PetscCall(DMGetBoundingBox(dm, lower, upper));
      PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

      PetscCall(PetscDrawSPReset(ctx->drawspE));
      PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
      PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
      PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));

      PetscCall(DMSwarmSortGetAccess(sw));
      for (PetscInt c = 0; c < cEnd - cStart; ++c) {
        PetscReal Eavg = 0.0;
        PetscInt *pidx, Npc;

        PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
        for (PetscInt q = 0; q < Npc; ++q) {
          const PetscInt p = pidx[q];
          Eavg += E[p * dim];
        }
        Eavg /= Npc;
        xval = (c + 0.5) * ((upper[0] - lower[0]) / (cEnd - cStart));
        PetscCall(PetscDrawSPAddPoint(ctx->drawspE, &xval, &Eavg));
        PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
      }
      PetscCall(PetscDrawSPDraw(ctx->drawspE, PETSC_TRUE));
      PetscCall(PetscDrawSPGetDraw(ctx->drawspE, &draw));
      if (ctx->saveOutput) PetscCall(PetscDrawSave(draw));
      PetscCall(DMSwarmSortRestoreAccess(sw));
      PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
      PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
      PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
    }

    Vec rho, rhohat, phi;

    PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "rho", &rho));
    PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "rhohat", &rhohat));
    PetscCall(VecView(rho, ctx->viewerRho));
    PetscCall(VecISCopy(ctx->fftX, ctx->fftReal, SCATTER_FORWARD, rho));
    PetscCall(MatMult(ctx->fftPot, ctx->fftX, ctx->fftY));
    PetscCall(VecFilter(ctx->fftY, PETSC_SMALL));
    PetscCall(VecViewFromOptions(ctx->fftX, NULL, "-real_view"));
    PetscCall(VecViewFromOptions(ctx->fftY, NULL, "-fft_view"));
    PetscCall(VecISCopy(ctx->fftY, ctx->fftReal, SCATTER_REVERSE, rhohat));
    PetscCall(VecSetValue(rhohat, 0, 0., INSERT_VALUES)); // Remove large DC component
    PetscCall(VecView(rhohat, ctx->viewerRhoHat));
    PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "rho", &rho));
    PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "rhohat", &rhohat));

    PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "phi", &phi));
    PetscCall(VecView(phi, ctx->viewerPhi));
    PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "phi", &phi));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *ctx)
{
  PetscBag   bag;
  Parameter *p;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  PetscCall(PetscBagGetData(ctx->bag, &p));
  PetscCall(PetscBagSetName(ctx->bag, "par", "Vlasov-Poisson Parameters"));
  bag = ctx->bag;
  PetscCall(PetscBagRegisterScalar(bag, &p->v0, 1.0, "v0", "Velocity scale, m/s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->t0, 1.0, "t0", "Time scale, s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->x0, 1.0, "x0", "Space scale, m"));
  PetscCall(PetscBagRegisterScalar(bag, &p->v0, 1.0, "phi0", "Potential scale, kg*m^2/A*s^3"));
  PetscCall(PetscBagRegisterScalar(bag, &p->q0, 1.0, "q0", "Charge Scale, A*s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->m0, 1.0, "m0", "Mass Scale, kg"));
  PetscCall(PetscBagRegisterScalar(bag, &p->epsi0, 1.0, "epsi0", "Permittivity of Free Space, kg"));
  PetscCall(PetscBagRegisterScalar(bag, &p->kb, 1.0, "kb", "Boltzmann Constant, m^2 kg/s^2 K^1"));

  PetscCall(PetscBagRegisterScalar(bag, &p->sigma, 1.0, "sigma", "Charge per unit area, C/m^3"));
  PetscCall(PetscBagRegisterScalar(bag, &p->poissonNumber, 1.0, "poissonNumber", "Non-Dimensional Poisson Number"));
  PetscCall(PetscBagRegisterScalar(bag, &p->vlasovNumber, 1.0, "vlasovNumber", "Non-Dimensional Vlasov Number"));
  PetscCall(PetscBagSetFromOptions(bag));
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscBool         flg;

    PetscCall(PetscOptionsCreateViewer(comm, NULL, NULL, "-param_view", &viewer, &format, &flg));
    if (flg) {
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(PetscBagView(bag, viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  DMField         coordField;
  IS              cellIS;
  PetscQuadrature quad;
  PetscReal      *wt, *pt;
  PetscInt        cdim, cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "space"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  // Cache the mesh geometry
  PetscCall(DMGetCoordinateField(*dm, &coordField));
  PetscCheck(coordField, comm, PETSC_ERR_USER, "DM must have a coordinate field");
  PetscCall(DMGetCoordinateDim(*dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, cEnd - cStart, cStart, 1, &cellIS));
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &quad));
  PetscCall(PetscMalloc1(1, &wt));
  PetscCall(PetscMalloc1(2, &pt));
  wt[0] = 1.;
  pt[0] = -1.;
  pt[1] = -1.;
  PetscCall(PetscQuadratureSetData(quad, cdim, 1, 1, pt, wt));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, PETSC_FEGEOM_BASIC, &ctx->fegeom));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(ISDestroy(&cellIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void ion_f0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -constants[SIGMA];
}

static void laplacian_f1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void laplacian_g3(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

/*
   /  I   -grad\ / q \ = /0\
   \-div    0  / \phi/   \f/
*/
static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[d] += u[uOff[0] + d];
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d * dim + d] = u[uOff[1]];
}

static void f0_phi_backgroundCharge(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += constants[SIGMA];
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_x[uOff_x[0] + d * dim + d];
}

/* Boundary residual. Dirichlet boundary for u means u_bdy=p*n */
static void g0_qq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  for (PetscInt d = 0; d < dim; ++d) g0[d * dim + d] = 1.0;
}

static void g2_qphi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  for (PetscInt d = 0; d < dim; ++d) g2[d * dim + d] = 1.0;
}

static void g1_phiq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  for (PetscInt d = 0; d < dim; ++d) g1[d * dim + d] = 1.0;
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *ctx)
{
  PetscFE   fephi, feq;
  PetscDS   ds;
  PetscBool simplex;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  if (ctx->em == EM_MIXED) {
    DMLabel        label;
    const PetscInt id = 1;

    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "field_", PETSC_DETERMINE, &feq));
    PetscCall(PetscObjectSetName((PetscObject)feq, "field"));
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "potential_", PETSC_DETERMINE, &fephi));
    PetscCall(PetscObjectSetName((PetscObject)fephi, "potential"));
    PetscCall(PetscFECopyQuadrature(feq, fephi));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)feq));
    PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fephi));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&fephi));
    PetscCall(PetscFEDestroy(&feq));

    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMGetDS(dm, &ds));

    PetscCall(PetscDSSetResidual(ds, 0, f0_q, f1_q));
    PetscCall(PetscDSSetResidual(ds, 1, f0_phi_backgroundCharge, NULL));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_qq, NULL, NULL, NULL));
    PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_qphi, NULL));
    PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_phiq, NULL, NULL));

    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)zero, NULL, NULL, NULL));

  } else {
    MatNullSpace nullsp;
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "em_", PETSC_DETERMINE, &fephi));
    PetscCall(PetscObjectSetName((PetscObject)fephi, "potential"));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fephi));
    PetscCall(DMCreateDS(dm));
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetResidual(ds, 0, ion_f0, laplacian_f1));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, laplacian_g3));
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_TRUE, 0, NULL, &nullsp));
    PetscCall(PetscObjectCompose((PetscObject)fephi, "nullspace", (PetscObject)nullsp));
    PetscCall(MatNullSpaceDestroy(&nullsp));
    PetscCall(PetscFEDestroy(&fephi));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreatePoisson(DM dm, AppCtx *ctx)
{
  SNES         snes;
  Mat          J;
  MatNullSpace nullSpace;

  PetscFunctionBeginUser;
  PetscCall(CreateFEM(dm, ctx));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dm), &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "em_"));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, ctx));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_TRUE, 0, NULL, &nullSpace));
  PetscCall(MatSetNullSpace(J, nullSpace));
  PetscCall(MatNullSpaceDestroy(&nullSpace));
  PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
  PetscCall(MatDestroy(&J));
  if (ctx->em == EM_MIXED) {
    const PetscInt potential = 1;

    PetscCall(DMCreateSubDM(dm, 1, &potential, &ctx->isPot, &ctx->dmPot));
  } else {
    ctx->dmPot = dm;
    PetscCall(PetscObjectReference((PetscObject)ctx->dmPot));
  }
  PetscCall(DMCreateMassMatrix(ctx->dmPot, ctx->dmPot, &ctx->M));
  PetscCall(DMPlexCreateClosureIndex(dm, NULL));
  ctx->snes = snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Conservation of mass (m = 1)
// n_t + 1/ m div p = 0
static void f0_mass(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += u_t[uOff[0]];
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_x[uOff_x[1] + d * dim + d];
}

// Conservation of momentum (m = 1, e = 1)
// p_t + div \tau - e n (E + u \times B)
// p_t + div \tau - e (n E + 1 / m p \times B)
// IN 1D: p_t + div Tr \tau - e (n E + 1 / m p \times B)
static void f0_momentum(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar n = u[uOff[0]];

  for (PetscInt d = 0; d < dim; ++d) {
    const PetscScalar E = -a_x[d];

    f0[d] += u_t[uOff[1] + d];
    for (PetscInt e = 0; e < dim; ++e) f0[d] += u_x[uOff_x[2] + (e * dim + d) * dim + e];
    f0[d] -= n * E;
  }
}

// Conservation of energy (m = 1, e = 1)
// Tr \tau_t + 2 div Q - 2 e n u \cdot E
// Tr \tau_t + 2 div Q - 2 e / m p \cdot E
static void f0_energy(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += u_t[uOff[2]];
  for (PetscInt d = 0; d < dim; ++d) {
    const PetscScalar p = u[uOff[1] + d];
    const PetscScalar E = -a_x[d];

    f0[0] -= 2. * p * E;
  }
}

static PetscErrorCode SetupMomentProblem(DM dm, AppCtx *ctx)
{
  PetscDS       ds;
  PetscWeakForm wf;

  PetscFunctionBegin;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetWeakForm(ds, &wf));
#if 0
  PetscCall(PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 100, 0, f0_mass_exp, 0, NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, NULL, 0, 1, 100, 0, f0_momentum_exp, 0, NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, NULL, 0, 2, 100, 0, f0_energy_exp, 0, NULL));
#else
  PetscCall(PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 0, f0_mass, 0, NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, NULL, 0, 1, 0, 0, f0_momentum, 0, NULL));
  PetscCall(PetscWeakFormSetIndexResidual(wf, NULL, 0, 2, 0, 0, f0_energy, 0, NULL));
  ///PetscCall(PetscWeakFormSetIndexJacobianPreconditioner(wf, NULL, 0, 0, 0, 0, 0, g0_nn, 0, NULL, 0, NULL, 0, NULL));
  //PetscCall(PetscWeakFormSetIndexJacobianPreconditioner(wf, NULL, 0, 0, 1, 0, 0, NULL, 0, g1_np, 0, NULL, 0, NULL));
  //PetscCall(PetscWeakFormSetIndexJacobianPreconditioner(wf, NULL, 0, 1, 1, 0, 0, g0_pp, 0, NULL, 0, NULL, 0, NULL));
  //PetscCall(PetscWeakFormSetIndexJacobianPreconditioner(wf, NULL, 0, 1, 0, 0, 0, g0_pn, 0, NULL, 0, NULL, 0, NULL));
  //PetscCall(PetscWeakFormSetIndexJacobianPreconditioner(wf, NULL, 0, 1, 2, 0, 0, NULL, 0, g1_pe, 0, NULL, 0, NULL));
  //PetscCall(PetscWeakFormSetIndexJacobianPreconditioner(wf, NULL, 0, 2, 2, 0, 0, g0_ee, 0, NULL, 0, NULL, 0, NULL));
  //PetscCall(PetscWeakFormSetIndexJacobianPreconditioner(wf, NULL, 0, 2, 1, 0, 0, g0_ep, 0, NULL, 0, NULL, 0, NULL));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateParticleMassMatrix(DM sw, DM dm, Vec x[], Mat *M_p)
{
  PetscFunctionBeginUser;
  DM            dmc = sw;
  DM            dmf = dm;
  PetscSection  gsf;
  DMSwarmCellDM celldm;
  PetscInt      m, n, Np, bs;
  PetscInt      rStart, maxC = 0;
  void         *ctx;

  PetscCall(DMGetGlobalSection(dmf, &gsf));
  PetscCall(PetscSectionGetConstrainedStorageSize(gsf, &m));
  PetscCall(DMSwarmGetLocalSize(dmc, &Np));
  PetscCall(DMSwarmGetCellDMActive(dmc, &celldm));
  PetscCall(DMSwarmCellDMGetBlockSize(celldm, dmc, &bs));
  n = Np * bs;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dmc), M_p));
  PetscCall(MatSetSizes(*M_p, n, m, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(*M_p, dmc->mattype));
  PetscCall(DMGetApplicationContext(dmf, &ctx));

  PetscCall(DMSwarmPreallocateMassMatrix(dmc, dmf, *M_p, &rStart, &maxC, ctx));

  const char  **cnames;
  PetscScalar **coords;
  PetscInt     *cbs, Nfc;

  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &cnames));
  PetscCall(PetscMalloc2(Nfc, &cbs, Nfc, &coords));
  for (PetscInt c = 0; c < Nfc; ++c) {
    PetscCall(DMSwarmGetFieldInfo(sw, cnames[c], &cbs[c], NULL));
    PetscCall(VecGetArray(x[c], &coords[c]));
  }
  PetscCall(DMSwarmFillMassMatrix(dmc, dmf, *M_p, rStart, maxC, PETSC_TRUE, Nfc, cbs, coords, ctx));
  for (PetscInt c = 0; c < Nfc; ++c) PetscCall(VecRestoreArray(x[c], &coords[c]));
  PetscCall(PetscFree2(cbs, coords));

  PetscCall(MatAssemblyBegin(*M_p, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*M_p, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ProjectParticleToMoments_Private - Project a function defined on the particle basis to one on the finite element basis for the moment problem.

  Collective

  Input Parameters:
+ R - The `MATSHELL` implementing this restriction
. w - The `Vec` of particle weights
. x - The `Vec` of particle positions
- v - The `Vec` of particle velocities

  Output Parameter:
. f - The `Vec` from the moment basis

  Level: developer

  Note:
  We need to reform the particle mass matrix on each call, since it depends on particle positions

.seealso: [](ch_dm), `ProjectMomentsToParticle_Private()`
*/
static PetscErrorCode ProjectParticleToMoments_Private(Mat R, Vec w, Vec x, Vec v, Vec f)
{
  MomStruct *ms;
  Mat        M_p, M_pe;
  Vec        nrhs, prhs, erhs, n, p, e;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(R, &ms));

  PetscCall(DMSwarmSetCellDMActive(ms->sw, "moments"));
  PetscCall(CreateParticleMassMatrix(ms->sw, ms->dmN, &x, &M_p));
  PetscCall(DMSwarmSetCellDMActive(ms->sw, "space"));

  PetscCall(DMSwarmSetCellDMActive(ms->sw, "moments"));
  PetscCall(CreateParticleMassMatrix(ms->sw, ms->dmE, &x, &M_pe));
  PetscCall(DMSwarmSetCellDMActive(ms->sw, "space"));

  PetscCall(DMGetGlobalVector(ms->dmN, &nrhs));
  PetscCall(DMGetGlobalVector(ms->dmP, &prhs));
  PetscCall(DMGetGlobalVector(ms->dmE, &erhs));
  PetscCall(DMGetGlobalVector(ms->dmN, &n));
  PetscCall(DMGetGlobalVector(ms->dmP, &p));
  PetscCall(DMGetGlobalVector(ms->dmE, &e));
  // nrhs = M_p w_p
  PetscCall(MatMultTranspose(M_p, w, nrhs));
  // Solve M_f n = M_p w_p
  PetscCall(KSPSolve(ms->ksp_fn, nrhs, n));
  // prhs = M_p (w_p : v)
  PetscCall(VecPointwiseMult(ms->work_p, w, v));
  PetscCall(MatMultTranspose(M_p, ms->work_p, prhs));
  // Solve M_f p = M_p (w_p : v)
  PetscCall(KSPSolve(ms->ksp_fp, prhs, p));
  // erhs = M_p (w_p : v^2)
  PetscCall(VecPointwiseMult(ms->work_p, ms->work_p, v));
  PetscCall(MatMultTranspose(M_pe, ms->work_p, erhs));
  // Solve M_f e = M_p (w_p : v^2)
  PetscCall(KSPSolve(ms->ksp_fe, erhs, e));
  PetscCall(MatDestroy(&M_p));
  PetscCall(MatDestroy(&M_pe));
  // Assemble moments into combined vector
  PetscCall(VecISCopy(f, ms->isN, SCATTER_FORWARD, n));
  PetscCall(VecISCopy(f, ms->isP, SCATTER_FORWARD, p));
  PetscCall(VecISCopy(f, ms->isE, SCATTER_FORWARD, e));
  PetscCall(DMRestoreGlobalVector(ms->dmN, &nrhs));
  PetscCall(DMRestoreGlobalVector(ms->dmP, &prhs));
  PetscCall(DMRestoreGlobalVector(ms->dmE, &erhs));
  PetscCall(DMRestoreGlobalVector(ms->dmN, &n));
  PetscCall(DMRestoreGlobalVector(ms->dmP, &p));
  PetscCall(DMRestoreGlobalVector(ms->dmE, &e));
  if (ms->moment_field_monitor) PetscCall(MonitorMomentFields_Continuum(NULL, 1, 1.0, f, ms));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0];
  return PETSC_SUCCESS;
}

static PetscErrorCode quadratic(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscSqr(x[0]);
  return PETSC_SUCCESS;
}

/*
  ProjectMomentsToParticle_Private - Project a function defined on the finite element basis for the moment problem to one defined on the particle basis.

  Collective

  Input Parameters:
+ In - The `MATSHELL` implementing this restriction
. f  - The `Vec` from the moment basis
- xv - The `Vec` of particle positions and velocities

  Output Parameter:
. w - The `Vec` of particle weights

  Level: developer

  Note:
  We need to reform the particle mass matrix on each call, since it depends on particle positions
  $$
  \begin{align}
  b_{(ki)}     &= \begin{cases} n_i & k = 0\\ p_i & k = 1\\ \tau_i & k = 2\end{cases} \\
  A_{(ki)(lj)} &= d_{kj} \delta_{il}
  \end{align}
  $$

.seealso: [](ch_dm), `ProjectParticleToMoments_Private()`
*/
static PetscErrorCode ProjectMomentsToParticle_Private(Mat In, Vec f, Vec x[], Vec w)
{
  const PetscInt debug = 1;
  MomStruct     *ms;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)In, &comm));
  PetscCall(MatShellGetContext(In, &ms));

  Mat      IN;
  Vec      INscale;
  Vec      rhs;
  Vec      mrhs, n, p, e;
  VecType  vtype;
  IS       fis;
  PetscInt nx, Nx, rStart;

  // Create rhs
  PetscCall(DMGetGlobalVector(ms->dmN, &n));
  PetscCall(DMGetGlobalVector(ms->dmP, &p));
  PetscCall(DMGetGlobalVector(ms->dmE, &e));
  PetscCall(DMGetGlobalVector(ms->dmE, &mrhs));
  PetscCall(VecGetLocalSize(e, &nx));
  PetscCall(VecGetSize(e, &Nx));
  PetscCall(VecCreate(comm, &rhs));
  PetscCall(VecSetSizes(rhs, nx * 3, Nx * 3));
  PetscCall(VecGetType(e, &vtype));
  PetscCall(VecSetType(rhs, vtype));
  //   Separate solution into moments
  PetscCall(VecISCopy(f, ms->isN, SCATTER_REVERSE, n));
  PetscCall(VecISCopy(f, ms->isP, SCATTER_REVERSE, p));
  PetscCall(VecISCopy(f, ms->isE, SCATTER_REVERSE, e));
  //   Interpolate n and p into eneregy space and inject all into rhs
  PetscCall(VecGetOwnershipRange(rhs, &rStart, NULL));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, nx, rStart + nx * 2, 1, &fis));
  PetscCall(VecISCopy(rhs, fis, SCATTER_FORWARD, e));
  PetscCall(ISDestroy(&fis));
  PetscCall(DMCreateInterpolation(ms->dmN, ms->dmE, &IN, &INscale));
  PetscCall(MatMult(IN, n, mrhs));
  PetscCall(MatDestroy(&IN));
  PetscCall(VecDestroy(&INscale));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, nx, rStart + nx * 0, 1, &fis));
  PetscCall(VecISCopy(rhs, fis, SCATTER_FORWARD, mrhs));
  PetscCall(ISDestroy(&fis));
  PetscCall(DMCreateInterpolation(ms->dmP, ms->dmE, &IN, &INscale));
  PetscCall(MatMult(IN, p, mrhs));
  PetscCall(MatDestroy(&IN));
  PetscCall(VecDestroy(&INscale));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, nx, rStart + nx * 1, 1, &fis));
  PetscCall(VecISCopy(rhs, fis, SCATTER_FORWARD, mrhs));
  PetscCall(ISDestroy(&fis));
  if (debug) {
    PetscReal nnorm, pnorm, enorm, rnorm;

    PetscCall(VecNorm(n, NORM_2, &nnorm));
    PetscCall(VecNorm(p, NORM_2, &pnorm));
    PetscCall(VecNorm(e, NORM_2, &enorm));
    PetscCall(VecNorm(rhs, NORM_2, &rnorm));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "||n|| %g\n||p|| %g\n||e|| %g\n||rhs|| %g\n", nnorm, pnorm, enorm, rnorm));
  }
  PetscCall(DMRestoreGlobalVector(ms->dmN, &n));
  PetscCall(DMRestoreGlobalVector(ms->dmP, &p));
  PetscCall(DMRestoreGlobalVector(ms->dmE, &e));
  PetscCall(DMRestoreGlobalVector(ms->dmE, &mrhs));

  PetscSimplePointFn *funcs[1];
  DM                  vdm, rdm;
  Mat                 A;
  MatType             mtype;
  Vec                 Phi, momVec[3];
  PetscInt            Nv, nxv, Nxv;

  // Create system matrix
  PetscCall(DMSwarmSetCellDMActive(ms->sw, "moment_phase"));
  PetscCall(DMSwarmGetCellDM(ms->sw, &rdm));
  PetscCall(DMGetGlobalVector(rdm, &Phi));
  PetscCall(VecGetLocalSize(Phi, &nxv));
  PetscCall(VecGetSize(Phi, &Nxv));
  PetscCall(DMSwarmSetCellDMActive(ms->sw, "velocity"));
  PetscCall(DMSwarmGetCellDM(ms->sw, &vdm));
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, nx * 3, nxv, Nx * 3, Nxv));
  PetscCall(DMGetMatType(rdm, &mtype));
  PetscCall(MatSetType(A, mtype));
  PetscCall(DMGetGlobalVector(vdm, &momVec[0]));
  PetscCall(DMGetGlobalVector(vdm, &momVec[1]));
  PetscCall(DMGetGlobalVector(vdm, &momVec[2]));
  PetscCall(VecGetSize(momVec[0], &Nv));
  PetscCheck(Nx * Nv == Nxv, comm, PETSC_ERR_PLIB, "Nx %" PetscInt_FMT " * Nv %" PetscInt_FMT " != %" PetscInt_FMT " Nxv", Nx, Nv, Nxv);
  // Project moment multiplier
  funcs[0] = constant;
  PetscCall(DMProjectFunction(vdm, 0., funcs, NULL, INSERT_VALUES, momVec[0]));
  funcs[0] = linear;
  PetscCall(DMProjectFunction(vdm, 0., funcs, NULL, INSERT_VALUES, momVec[1]));
  funcs[0] = quadratic;
  PetscCall(DMProjectFunction(vdm, 0., funcs, NULL, INSERT_VALUES, momVec[2]));
  // TODO How do I map the numbering from dmN and vdm to rdm?
  //   In 1D P1, it is done because we use the vertex numbering
  for (PetscInt k = 0; k < 3; ++k) {
    for (PetscInt i = 0; i < Nx; ++i) {
      const PetscInt row = k * Nx + i;
      for (PetscInt j = 0; j < Nv; ++j) {
        const PetscInt col = j * Nx + i;
        PetscScalar    val;

        PetscCall(VecGetValues(momVec[k], 1, &j, &val));
        PetscCall(MatSetValue(A, row, col, val, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A, NULL, "-proj_A_view"));
  PetscCall(DMRestoreGlobalVector(vdm, &momVec[0]));
  PetscCall(DMRestoreGlobalVector(vdm, &momVec[1]));
  PetscCall(DMRestoreGlobalVector(vdm, &momVec[2]));
  PetscCall(DMSwarmSetCellDMActive(ms->sw, "space"));

  // Solve for Phi
  {
    KSP         ksp;
    const char *prefix;

    PetscCall(KSPCreate(PetscObjectComm((PetscObject)ms->sw), &ksp));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ms->sw, &prefix));
    PetscCall(KSPSetOptionsPrefix(ksp, prefix));
    PetscCall(KSPAppendOptionsPrefix(ksp, "ftor_"));
    PetscCall(KSPSetFromOptions(ksp));

    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, rhs, Phi));
    PetscCall(VecViewFromOptions(Phi, NULL, "-proj_phi_view"));
    PetscCall(KSPDestroy(&ksp));
    if (debug) {
      PetscReal pnorm;

      PetscCall(VecNorm(Phi, NORM_2, &pnorm));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "||Phi|| %g\n", pnorm));
    }
  }
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));

  Mat rM_p, rPM_p;

  // Solve for w
  PetscCall(DMSwarmSetCellDMActive(ms->sw, "moment_phase"));
  PetscCall(CreateParticleMassMatrix(ms->sw, rdm, x, &rM_p));
  PetscCall(MatViewFromOptions(rM_p, NULL, "-rM_p_view"));
  PetscCall(DMSwarmSetCellDMActive(ms->sw, "space"));
  // Solve M_p
  {
    KSP         ksp;
    PC          pc;
    const char *prefix;
    PetscBool   isBjacobi;

    PetscCall(KSPCreate(PetscObjectComm((PetscObject)ms->sw), &ksp));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ms->sw, &prefix));
    PetscCall(KSPSetOptionsPrefix(ksp, prefix));
    PetscCall(KSPAppendOptionsPrefix(ksp, "ftop_"));
    PetscCall(KSPSetFromOptions(ksp));

    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &isBjacobi));
    if (isBjacobi) {
      PetscCall(DMSwarmCreateMassMatrixSquare(ms->sw, rdm, &rPM_p));
    } else {
      rPM_p = rM_p;
      PetscCall(PetscObjectReference((PetscObject)rPM_p));
    }
    PetscCall(KSPSetOperators(ksp, rM_p, rPM_p));
    PetscCall(KSPSolveTranspose(ksp, Phi, w));
    if (debug) {
      PetscReal wnorm;

      PetscCall(VecNorm(w, NORM_2, &wnorm));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "||dw|| %g\n", wnorm));
    }
    PetscCall(VecViewFromOptions(w, NULL, "-proj_w_view"));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&rPM_p));
    PetscCall(MatDestroy(&rM_p));
  }
  PetscCall(DMRestoreGlobalVector(rdm, &Phi));

  if (ms->moment_field_monitor) PetscCall(MonitorMomentFields_Continuum(NULL, 1, 1.0, f, ms));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Restriction: This maps defects like F(x) - b
//   We use the particle-wise norm of the residual as the weight
//   We use the existing particle locations
static PetscErrorCode ptofrhs(Mat R, Vec rp, Vec rf)
{
  MomStruct         *ms;
  Vec                w, x, v;
  const PetscScalar *rpa;
  PetscScalar       *wa;
  PetscInt           dim, Np;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(R, &ms));
  PetscCall(DMSwarmCreateGlobalVectorFromField(ms->sw, DMSwarmPICField_coor, &x));
  PetscCall(DMSwarmCreateGlobalVectorFromField(ms->sw, "velocity", &v));
  PetscCall(VecDuplicate(x, &w));
  PetscCall(DMGetDimension(ms->sw, &dim));
  PetscCall(DMSwarmGetLocalSize(ms->sw, &Np));
  PetscCall(VecGetArrayRead(rp, &rpa));
  PetscCall(VecGetArrayWrite(w, &wa));
  for (PetscInt p = 0; p < Np; ++p) {
    PetscReal norm = 0.;
    for (PetscInt d = 0; d < dim * 2; ++d) norm += PetscSqr(PetscRealPart(rpa[p * dim * 2 + d]));
    wa[p] = PetscSqrtReal(norm);
  }
  PetscCall(VecRestoreArrayRead(rp, &rpa));
  PetscCall(VecRestoreArrayWrite(w, &wa));
  PetscCall(ProjectParticleToMoments_Private(R, w, x, v, rf));
  PetscCall(VecDestroy(&w));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(ms->sw, DMSwarmPICField_coor, &x));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(ms->sw, "velocity", &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Injection: This maps states
//   We use the existing particle weights
//   We use the particle locations from the input xp
static PetscErrorCode ptof(Mat R, Vec xp, Vec xf)
{
  MomStruct   *ms;
  TS           ts;
  DM           dm;
  Vec          w, x, v, X0, Xdot;
  PetscInt     dim;
  PetscClassId id;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(R, &ms));
  PetscCall(DMGetDimension(ms->sw, &dim));
  PetscCall(DMSwarmCreateGlobalVectorFromField(ms->sw, "w_q", &w));
  // Extract coordinates
  //   Can use VecStrideSubSetGather() for higher D
  PetscCall(VecDuplicate(w, &x));
  PetscCall(VecDuplicate(w, &v));
  PetscCall(VecSetBlockSize(xp, 2 * dim));
  PetscCall(VecStrideGather(xp, 0, x, INSERT_VALUES));
  PetscCall(VecStrideGather(xp, 1, v, INSERT_VALUES));
  PetscCall(ProjectParticleToMoments_Private(R, w, x, v, xf));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&v));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(ms->sw, "w_q", &w));
  // Set initial TS solution to be this iterate
  PetscCall(MatShellGetContext(R, &ms));
  PetscCall(DMSNESGetFunction(ms->dm, NULL, (void **)&ts));
  PetscCall(PetscObjectGetClassId((PetscObject)ts, &id));
  PetscCheck(id == TS_CLASSID, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid context for function evaluation");
  PetscCall(TSGetDM(ts, &dm));
  PetscCheck(dm == ms->dm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatched moment DM");
  PetscCall(TSDiscGradGetX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(VecCopy(xf, X0));
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, dm, &X0, &Xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Prolongation
//   We are updating the particle weight
static PetscErrorCode ftopadd(Mat In, Vec xf, Vec xpo, Vec xp)
{
  MomStruct *ms;
  Vec        w, dw, x[2];
  PetscInt   dim;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(In, &ms));
  PetscCall(DMGetDimension(ms->sw, &dim));
  PetscCall(DMSwarmCreateGlobalVectorFromField(ms->sw, "w_q", &w));
  PetscCall(VecDuplicate(w, &dw));
  PetscCall(VecDuplicate(w, &x[0]));
  PetscCall(VecDuplicate(w, &x[1]));
  PetscCall(VecSetBlockSize(xpo, 2 * dim));
  PetscCall(VecStrideGather(xpo, 0, x[0], INSERT_VALUES));
  PetscCall(VecStrideGather(xpo, 1, x[1], INSERT_VALUES));
  PetscCall(ProjectMomentsToParticle_Private(In, xf, x, dw));
  // TODO We could alternatively just use the full moments here, instead of the increments
  // TODO Check the sign here
  PetscCall(VecAXPY(w, 1., dw));
  PetscCall(VecDestroy(&dw));
  PetscCall(VecDestroy(&x[0]));
  PetscCall(VecDestroy(&x[1]));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(ms->sw, "w_q", &w));

  PetscCall(VecCopy(xpo, xp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateInterpolation_Hybrid(DM dmCoarse, DM dmFine, Mat *Inp, Vec *scaling)
{
  AppCtx  *user;
  Vec      vp, vf;
  PetscInt np, nf;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecificType(dmCoarse, DM_CLASSID, 1, DMPLEX);
  PetscValidHeaderSpecificType(dmFine, DM_CLASSID, 1, DMSWARM);
  PetscCall(PetscObjectGetComm((PetscObject)dmFine, &comm));
  PetscCall(DMGetLocalVector(dmFine, &vp));
  PetscCall(VecGetLocalSize(vp, &np));
  PetscCall(DMRestoreLocalVector(dmFine, &vp));
  PetscCall(DMGetLocalVector(dmCoarse, &vf));
  PetscCall(VecGetLocalSize(vf, &nf));
  PetscCall(DMRestoreLocalVector(dmCoarse, &vf));

  PetscCall(MatCreate(comm, Inp));
  PetscCall(MatSetType(*Inp, MATSHELL));
  PetscCall(MatSetSizes(*Inp, np, nf, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatShellSetOperation(*Inp, MATOP_MULT_ADD, (void (*)(void))ftopadd));
  PetscCall(DMGetApplicationContext(dmCoarse, &user));
  PetscCall(MatShellSetContext(*Inp, &user->momCtx));
  PetscCall(MatSetUp(*Inp));
  PetscCall(DMCreateGlobalVector(dmCoarse, scaling));
  PetscCall(VecSet(*scaling, 1.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateInjection_Hybrid(DM dmCoarse, DM dmFine, Mat *Inj)
{
  AppCtx  *user;
  Vec      vp, vf;
  PetscInt np, nf;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecificType(dmCoarse, DM_CLASSID, 1, DMPLEX);
  PetscValidHeaderSpecificType(dmFine, DM_CLASSID, 1, DMSWARM);
  PetscCall(PetscObjectGetComm((PetscObject)dmFine, &comm));
  PetscCall(DMGetLocalVector(dmFine, &vp));
  PetscCall(VecGetLocalSize(vp, &np));
  PetscCall(DMRestoreLocalVector(dmFine, &vp));
  PetscCall(DMGetLocalVector(dmCoarse, &vf));
  PetscCall(VecGetLocalSize(vf, &nf));
  PetscCall(DMRestoreLocalVector(dmCoarse, &vf));

  PetscCall(MatCreate(comm, Inj));
  PetscCall(MatSetType(*Inj, MATSHELL));
  PetscCall(MatSetSizes(*Inj, nf, np, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatShellSetOperation(*Inj, MATOP_MULT, (void (*)(void))ptof));
  PetscCall(DMGetApplicationContext(dmCoarse, &user));
  PetscCall(MatShellSetContext(*Inj, &user->momCtx));
  PetscCall(MatSetUp(*Inj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateRestriction_Hybrid(DM dmCoarse, DM dmFine, Mat *R)
{
  AppCtx  *ctx;
  Vec      vp, vf;
  PetscInt np, nf;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecificType(dmCoarse, DM_CLASSID, 1, DMPLEX);
  PetscValidHeaderSpecificType(dmFine, DM_CLASSID, 1, DMSWARM);
  PetscCall(PetscObjectGetComm((PetscObject)dmFine, &comm));
  PetscCall(DMGetLocalVector(dmFine, &vp));
  PetscCall(VecGetLocalSize(vp, &np));
  PetscCall(DMRestoreLocalVector(dmFine, &vp));
  PetscCall(DMGetLocalVector(dmCoarse, &vf));
  PetscCall(VecGetLocalSize(vf, &nf));
  PetscCall(DMRestoreLocalVector(dmCoarse, &vf));

  PetscCall(MatCreate(comm, R));
  PetscCall(MatSetType(*R, MATSHELL));
  PetscCall(MatSetSizes(*R, nf, np, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatShellSetOperation(*R, MATOP_MULT, (void (*)(void))ptofrhs));
  PetscCall(DMGetApplicationContext(dmCoarse, &ctx));
  PetscCall(MatShellSetContext(*R, &ctx->momCtx));
  PetscCall(MatSetUp(*R));

  // TODO Where does this go?
  Vec f;

  ctx->momCtx.sw = dmFine;
  PetscCall(DMSwarmCreateGlobalVectorFromField(dmFine, "w_q", &f));
  PetscCall(VecDuplicate(f, &ctx->momCtx.work_p));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(dmFine, "w_q", &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMomentFields(DM odm, AppCtx *ctx)
{
  MomStruct     *ms = &ctx->momCtx;
  DM             dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMClone(odm, &dm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, "mom_"));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "n_", PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "number density"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, dim, ct, "p_", PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "momentum density"));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "e_", PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "energy density"));
  PetscCall(DMSetField(dm, 2, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupMomentProblem(dm, ctx));

  dm->ops->createinterpolation = DMCreateInterpolation_Hybrid;
  dm->ops->createinjection     = DMCreateInjection_Hybrid;
  dm->ops->createrestriction   = DMCreateRestriction_Hybrid;
  ctx->momCtx.dm               = dm;
  PetscInt field;

  field = 0;
  PetscCall(DMCreateSubDM(ctx->momCtx.dm, 1, &field, &ms->isN, &ms->dmN));
  PetscCall(DMCreateMassMatrix(ms->dmN, ms->dmN, &ms->MN));
  PetscCall(DMCreateInterpolation(ms->dmN, ctx->dmPot, &ms->IN, &ms->INscale));
  field = 1;
  PetscCall(DMCreateSubDM(ctx->momCtx.dm, 1, &field, &ms->isP, &ms->dmP));
  PetscCall(DMCreateMassMatrix(ms->dmP, ms->dmP, &ms->MP));
  field = 2;
  PetscCall(DMCreateSubDM(ctx->momCtx.dm, 1, &field, &ms->isE, &ms->dmE));
  PetscCall(DMCreateMassMatrix(ms->dmE, ms->dmE, &ms->ME));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ms->ksp_fn));
  PetscCall(KSPSetOptionsPrefix(ms->ksp_fn, "mom_proj_"));
  PetscCall(KSPSetOperators(ms->ksp_fn, ms->MN, ms->MN));
  PetscCall(KSPSetFromOptions(ms->ksp_fn));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ms->ksp_fp));
  PetscCall(KSPSetOptionsPrefix(ms->ksp_fp, "mom_proj_"));
  PetscCall(KSPSetOperators(ms->ksp_fp, ms->MP, ms->MP));
  PetscCall(KSPSetFromOptions(ms->ksp_fp));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ms->ksp_fe));
  PetscCall(KSPSetOptionsPrefix(ms->ksp_fe, "mom_proj_"));
  PetscCall(KSPSetOperators(ms->ksp_fe, ms->ME, ms->ME));
  PetscCall(KSPSetFromOptions(ms->ksp_fe));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ms->ksp_p));
  PetscCall(KSPSetOptionsPrefix(ms->ksp_p, "mom_interp_"));
  PetscCall(KSPSetFromOptions(ms->ksp_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void mass(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += u[0];
}

static PetscErrorCode CheckNeutrality(DM dm, Vec rho, AppCtx *ctx)
{
  PetscDS     ds;
  Parameter  *param;
  PetscScalar totalCharge;

  PetscFunctionBegin;
  PetscCall(PetscBagGetData(ctx->bag, (void **)&param));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, 0, mass));
  PetscCall(DMPlexComputeIntegralFEM(dm, rho, &totalCharge, ctx));
  PetscCheck(PetscAbsReal(PetscRealPart(totalCharge) - ctx->gridVolume * param->sigma) < 1e-5 /*PETSC_SMALL*/, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Electron charge: %g ion charge: %g diff: %g", (double)PetscRealPart(totalCharge),
             (double)(ctx->gridVolume * param->sigma), (double)PetscAbsReal(PetscRealPart(totalCharge) - ctx->gridVolume * param->sigma));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeMomentEM(DM dm, PetscReal time, Vec X, Vec X_t, void *ctx)
{
  AppCtx    *user = (AppCtx *)ctx;
  MomStruct *ms   = &user->momCtx;
  DM         dmEM;
  Vec        n, rho, rhoRhs, phi;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(user->snes, &dmEM));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &phi));
  PetscCall(DMGetGlobalVector(ms->dmN, &n));
  PetscCall(VecISCopy(X, ms->isN, SCATTER_REVERSE, n));
  PetscCall(DMGetGlobalVector(dmEM, &rho));
  PetscCall(DMGetGlobalVector(dmEM, &rhoRhs));
  PetscCall(PetscObjectSetName((PetscObject)n, "Density"));
  PetscCall(VecViewFromOptions(n, NULL, "-em_debug_view"));
  PetscCall(MatMult(ms->IN, n, rho));
  PetscCall(PetscObjectSetName((PetscObject)rho, "Charge Density (Before)"));
  PetscCall(VecViewFromOptions(rho, NULL, "-em_debug_view"));
  PetscCall(VecPointwiseMult(rho, rho, ms->INscale));
  PetscCall(PetscObjectSetName((PetscObject)rho, "Charge Density"));
  PetscCall(VecViewFromOptions(rho, NULL, "-em_debug_view"));
#if 0
  PetscCall(CheckNeutrality(ms->dmN, n, user));
  PetscCall(CheckNeutrality(dmEM, rho, user));
#endif
  PetscCall(VecScale(rho, -1.0));
  PetscCall(MatMult(ms->MN, rho, rhoRhs));
  PetscCall(DMRestoreGlobalVector(ms->dmN, &n));
  PetscCall(DMRestoreGlobalVector(dmEM, &rho));
  PetscCall(VecSet(phi, 0.0));
  PetscCall(SNESSolve(user->snes, rhoRhs, phi));
  PetscCall(DMRestoreGlobalVector(dmEM, &rhoRhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMomentSolver(DM dm, TS *ts, AppCtx *user)
{
  Vec         phi;
  PetscBool   explicit = PETSC_FALSE, lumped = PETSC_FALSE;
  const char *prefix;

  PetscFunctionBeginUser;
  PetscCall(TSCreate(PETSC_COMM_WORLD, ts));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
  PetscCall(TSSetOptionsPrefix(*ts, prefix));
  PetscCall(TSSetDM(*ts, dm));
  PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, user));
  if (explicit) {
    PetscCall(DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFEM, user));
    if (lumped) PetscCall(DMTSCreateRHSMassMatrixLumped(dm));
    else PetscCall(DMTSCreateRHSMassMatrix(dm));
  } else {
    PetscCall(DMTSSetIFunctionPre(dm, ComputeMomentEM, user));
    PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, user));
    PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, user));
  }
  PetscCall(TSSetMaxSteps(*ts, 0));
  PetscCall(TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(*ts));

  PetscCall(DMCreateGlobalVector(user->dmPot, &phi));
  PetscCall(DMSetAuxiliaryVec(user->momCtx.dm, NULL, 0, 0, phi));
  PetscCall(VecDestroy(&phi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscPDFPertubedConstant2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1 + 0.01 * PetscCosReal(0.5 * x[0])) / (2 * PETSC_PI);
  p[1] = (1 + 0.01 * PetscCosReal(0.5 * x[1])) / (2 * PETSC_PI);
  return PETSC_SUCCESS;
}
PetscErrorCode PetscPDFPertubedConstant1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1. + 0.01 * PetscCosReal(0.5 * x[0])) / (2 * PETSC_PI);
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine1D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.0;
  const PetscReal k     = scale ? scale[1] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * x[0]));
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine2D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.;
  const PetscReal k     = scale ? scale[0] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * (x[0] + x[1])));
  return PETSC_SUCCESS;
}

static PetscErrorCode CreateVelocityDM(DM sw, DM *vdm)
{
  AppCtx        *user;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  const char    *prefix = "v";

  PetscFunctionBegin;
  PetscCall(DMCreate(PETSC_COMM_SELF, vdm));
  PetscCall(DMSetType(*vdm, DMPLEX));
  PetscCall(DMPlexSetOptionsPrefix(*vdm, prefix));
  PetscCall(DMSetFromOptions(*vdm));
  PetscCall(DMGetApplicationContext(sw, (void **)&user));
  if (user->optimizeVelGrid) PetscCall(OptimizeVelocityGrid(sw, *vdm));
  PetscCall(PetscObjectSetName((PetscObject)*vdm, "velocity"));
  PetscCall(DMViewFromOptions(*vdm, NULL, "-dm_view"));

  PetscCall(DMGetDimension(*vdm, &dim));
  PetscCall(DMPlexGetHeightStratum(*vdm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(*vdm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, prefix, PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "distribution"));
  PetscCall(DMSetField(*vdm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(*vdm));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  InitializeParticles_Centroid - Initialize a regular grid of particles.

  Input Parameters:
+ sw      - The `DMSWARM`
- force1D - Treat the spatial domain as 1D

  Notes:
  This functions sets the species, cellid, spatial coordinate, and velocity fields for all particles.

  It places one particle in the centroid of each cell in the implicit tensor product of the spatial
  and velocity meshes.
*/
static PetscErrorCode InitializeParticles_Centroid(DM sw)
{
  DM_Swarm     *swarm = (DM_Swarm *)sw->data;
  DMSwarmCellDM celldm;
  DM            xdm, vdm;
  PetscReal     vmin[3], vmax[3];
  PetscReal    *x, *v;
  PetscInt     *species, *cellid;
  PetscInt      dim, xcStart, xcEnd, vcStart, vcEnd, Ns, Np, Npc, debug;
  PetscBool     flg;
  MPI_Comm      comm;
  const char   *cellidname;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));

  PetscOptionsBegin(comm, "", "DMSwarm Options", "DMSWARM");
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(PetscOptionsInt("-dm_swarm_num_species", "The number of species", "DMSwarmSetNumSpecies", Ns, &Ns, &flg));
  if (flg) PetscCall(DMSwarmSetNumSpecies(sw, Ns));
  PetscCall(PetscOptionsBoundedInt("-dm_swarm_print_coords", "Debug output level for particle coordinate computations", "InitializeParticles", 0, &swarm->printCoords, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-dm_swarm_print_weights", "Debug output level for particle weight computations", "InitializeWeights", 0, &swarm->printWeights, NULL, 0));
  PetscOptionsEnd();
  debug = swarm->printCoords;

  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &xdm));
  PetscCall(DMPlexGetHeightStratum(xdm, 0, &xcStart, &xcEnd));

  PetscCall(DMSwarmGetCellDMByName(sw, "velocity", &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
  PetscCall(DMPlexGetHeightStratum(vdm, 0, &vcStart, &vcEnd));

  // One particle per centroid on the tensor product grid
  Npc = (vcEnd - vcStart) * Ns;
  Np  = (xcEnd - xcStart) * Npc;
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));
  if (debug) {
    PetscInt gNp, gNc, Nc = xcEnd - xcStart;
    PetscCallMPI(MPIU_Allreduce(&Np, &gNp, 1, MPIU_INT, MPIU_SUM, comm));
    PetscCall(PetscPrintf(comm, "Global Np = %" PetscInt_FMT "\n", gNp));
    PetscCallMPI(MPIU_Allreduce(&Nc, &gNc, 1, MPIU_INT, MPIU_SUM, comm));
    PetscCall(PetscPrintf(comm, "Global X-cells = %" PetscInt_FMT "\n", gNc));
    PetscCall(PetscPrintf(comm, "Global V-cells = %" PetscInt_FMT "\n", vcEnd - vcStart));
  }

  // Set species and cellid
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellidname));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmGetField(sw, cellidname, NULL, NULL, (void **)&cellid));
  for (PetscInt c = 0, p = 0; c < xcEnd - xcStart; ++c) {
    for (PetscInt s = 0; s < Ns; ++s) {
      for (PetscInt q = 0; q < Npc / Ns; ++q, ++p) {
        species[p] = s;
        cellid[p]  = c;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmRestoreField(sw, cellidname, NULL, NULL, (void **)&cellid));

  // Set particle coordinates
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
  PetscCall(DMGetCoordinatesLocalSetUp(xdm));
  for (PetscInt c = 0; c < xcEnd - xcStart; ++c) {
    const PetscInt cell = c + xcStart;
    PetscInt      *pidx, Npc;
    PetscReal      centroid[3], volume;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(DMPlexComputeCellGeometryFVM(xdm, cell, &volume, centroid, NULL));
    for (PetscInt s = 0; s < Ns; ++s) {
      for (PetscInt q = 0; q < Npc / Ns; ++q) {
        const PetscInt vcell = q + vcStart;
        const PetscInt p     = pidx[q * Ns + s];
        PetscReal      vcentroid[3];

        PetscCall(DMPlexComputeCellGeometryFVM(vdm, vcell, NULL, vcentroid, NULL));
        for (PetscInt d = 0; d < dim; ++d) {
          x[p * dim + d] = centroid[d];
          v[p * dim + d] = vcentroid[d];
        }
        if (debug > 1) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "Particle %4" PetscInt_FMT " ", p));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "  x: ("));
          for (PetscInt d = 0; d < dim; ++d) {
            if (d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", x[p * dim + d]));
          }
          PetscCall(PetscPrintf(PETSC_COMM_SELF, ") v:("));
          for (PetscInt d = 0; d < dim; ++d) {
            if (d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", v[p * dim + d]));
          }
          PetscCall(PetscPrintf(PETSC_COMM_SELF, ")\n"));
        }
      }
    }
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
  }
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  InitializeWeights - Compute weight for each local particle

  Input Parameters:
+ sw          - The `DMSwarm`
. totalWeight - The sum of all particle weights
. func        - The PDF for the particle spatial distribution
- param       - The PDF parameters

  Notes:
  The PDF for velocity is assumed to be a Gaussian

  The particle weights are returned in the `w_q` field of `sw`.
*/
static PetscErrorCode InitializeWeights(DM sw, PetscReal totalWeight, PetscProbFn *func, const PetscReal param[])
{
  AppCtx          *user;
  DM               xdm, vdm;
  DMSwarmCellDM    celldm;
  PetscScalar     *weight;
  PetscQuadrature  xquad;
  const PetscReal *xq, *xwq;
  const PetscInt   order = 5;
  PetscReal        xi0[3];
  PetscReal        xwtot = 0., pwtot = 0.;
  PetscInt         xNq;
  PetscInt         dim, Ns, xcStart, xcEnd, vcStart, vcEnd, debug = ((DM_Swarm *)sw->data)->printWeights;
  MPI_Comm         comm;
  PetscMPIInt      rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCall(DMGetApplicationContext(sw, (void **)&user));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &xdm));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMPlexGetHeightStratum(xdm, 0, &xcStart, &xcEnd));
  PetscCall(DMSwarmGetCellDMByName(sw, "velocity", &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
  PetscCall(DMPlexGetHeightStratum(vdm, 0, &vcStart, &vcEnd));

  // Setup Quadrature for spatial and velocity weight calculations
  PetscCall(PetscDTGaussTensorQuadrature(dim, 1, order, -1.0, 1.0, &xquad));
  PetscCall(PetscQuadratureGetData(xquad, NULL, NULL, &xNq, &xq, &xwq));
  for (PetscInt d = 0; d < dim; ++d) xi0[d] = -1.0;

  // Integrate the density function to get the weights of particles in each cell
  PetscCall(DMGetCoordinatesLocalSetUp(vdm));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  for (PetscInt c = xcStart; c < xcEnd; ++c) {
    PetscReal          xv0[3], xJ[9], xinvJ[9], xdetJ, xqr[3], xden, xw = 0.;
    PetscInt          *pidx, Npc;
    PetscInt           xNc;
    const PetscScalar *xarray;
    PetscScalar       *xcoords = NULL;
    PetscBool          xisDG;

    PetscCall(DMPlexGetCellCoordinates(xdm, c, &xisDG, &xNc, &xarray, &xcoords));
    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCheck(Npc == (vcEnd - vcStart) * Ns, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of particles %" PetscInt_FMT " in cell (rank %d) != %" PetscInt_FMT " number of velocity vertices", Npc, rank, (vcEnd - vcStart) * Ns);
    PetscCall(DMPlexComputeCellGeometryFEM(xdm, c, NULL, xv0, xJ, xinvJ, &xdetJ));
    for (PetscInt q = 0; q < xNq; ++q) {
      // Transform quadrature points from ref space to real space
      CoordinatesRefToReal(dim, dim, xi0, xv0, xJ, &xq[q * dim], xqr);
      // Get probability density at quad point
      //   No need to scale xqr since PDF will be periodic
      PetscCall((*func)(xqr, param, &xden));
      xw += xden * (xwq[q] * xdetJ);
    }
    xwtot += xw;
    if (debug) {
      IS              globalOrdering;
      const PetscInt *ordering;

      PetscCall(DMPlexGetCellNumbering(xdm, &globalOrdering));
      PetscCall(ISGetIndices(globalOrdering, &ordering));
      PetscCall(PetscSynchronizedPrintf(comm, "c:%" PetscInt_FMT " [x_a,x_b] = %1.15f,%1.15f -> cell weight = %1.15f\n", ordering[c], (double)PetscRealPart(xcoords[0]), (double)PetscRealPart(xcoords[0 + dim]), (double)xw));
      PetscCall(ISRestoreIndices(globalOrdering, &ordering));
    }
    // Set weights to be Gaussian in velocity cells
    for (PetscInt vc = vcStart; vc < vcEnd; ++vc) {
      const PetscInt     p  = pidx[vc * Ns + 0];
      PetscReal          vw = 0.;
      PetscInt           vNc;
      const PetscScalar *varray;
      PetscScalar       *vcoords = NULL;
      PetscBool          visDG;

      PetscCall(DMPlexGetCellCoordinates(vdm, vc, &visDG, &vNc, &varray, &vcoords));
      if (user->two_stream) {
        const PetscReal v_th  = 1.0;
        const PetscReal sigma = v_th * PetscSqrtReal(2.);
        const PetscReal v0    = user->two_stream_v0; // beam speed
        const PetscReal zP0   = vcoords[0] - v0;     // positive beam argument for bottom
        const PetscReal zP1   = vcoords[1] - v0;     // positive beam argument for top
        const PetscReal zN0   = vcoords[0] + v0;     // negative beam argument for bottom
        const PetscReal zN1   = vcoords[1] + v0;     // negative beam argument for top

        vw = 0.25 * ((PetscErfReal(zP1 / sigma) - PetscErfReal(zP0 / sigma)) + (PetscErfReal(zN1 / sigma) - PetscErfReal(zN0 / sigma)));
      } else {
        vw = 0.5 * (PetscErfReal(vcoords[1] / PetscSqrtReal(2.)) - PetscErfReal(vcoords[0] / PetscSqrtReal(2.)));
      }
      weight[p] = totalWeight * vw * xw;
      pwtot += weight[p];
      PetscCheck(weight[p] <= 10., PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Particle %" PetscInt_FMT " weight exceeded 1: %g, %g, %g", p, xw, vw, totalWeight);
      PetscCall(DMPlexRestoreCellCoordinates(vdm, vc, &visDG, &vNc, &varray, &vcoords));
      if (debug > 1) PetscCall(PetscPrintf(comm, "particle %" PetscInt_FMT ": %g, vw: %g xw: %g\n", p, weight[p], vw, xw));
    }
    PetscCall(DMPlexRestoreCellCoordinates(xdm, c, &xisDG, &xNc, &xarray, &xcoords));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
  }
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(PetscQuadratureDestroy(&xquad));

  if (debug) {
    PetscReal wtot[2] = {pwtot, xwtot}, gwtot[2];

    PetscCall(PetscSynchronizedFlush(comm, NULL));
    PetscCallMPI(MPIU_Allreduce(wtot, gwtot, 2, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(comm, "particle weight sum = %1.10f cell weight sum = %1.10f\n", (double)gwtot[0], (double)gwtot[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeParticles_PerturbedWeights(DM sw, AppCtx *ctx)
{
  PetscReal scale[2] = {ctx->cosine_coefficients[0], ctx->cosine_coefficients[1]};
  PetscInt  dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(InitializeParticles_Centroid(sw));
  PetscCall(InitializeWeights(sw, ctx->totalWeight, dim == 1 ? PetscPDFCosine1D : PetscPDFCosine2D, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeConstants(DM sw, AppCtx *ctx)
{
  DM         dm;
  PetscInt  *species;
  PetscReal *weight, totalCharge = 0., totalWeight = 0., gmin[3], gmax[3], global_charge, global_weight;
  PetscInt   Np, dim;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  for (PetscInt p = 0; p < Np; ++p) {
    totalWeight += weight[p];
    totalCharge += ctx->charges[species[p]] * weight[p];
  }
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  {
    Parameter *param;

    PetscCall(PetscBagGetData(ctx->bag, &param));
    switch (dim) {
    case 1:
      ctx->gridVolume = (gmax[0] - gmin[0]);
      break;
    case 2:
      ctx->gridVolume = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]);
      break;
    case 3:
      ctx->gridVolume = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]) * (gmax[2] - gmin[2]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT " not supported", dim);
    }
    PetscCallMPI(MPIU_Allreduce(&totalWeight, &global_weight, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPIU_Allreduce(&totalCharge, &global_charge, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dim = %" PetscInt_FMT "\ttotalWeight = %f, user->charges[species[0]] = %f\ttotalCharge = %f, Total Area = %f\n", dim, (double)global_weight, (double)ctx->charges[0], (double)global_charge, (double)ctx->gridVolume));
    param->sigma = PetscAbsReal(global_charge / (ctx->gridVolume));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sigma: %g\n", (double)param->sigma));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(x0,v0,t0,m0,q0,phi0): (%e, %e, %e, %e, %e, %e) - (P, V) = (%e, %e)\n", (double)param->x0, (double)param->v0, (double)param->t0, (double)param->m0, (double)param->q0, (double)param->phi0, (double)param->poissonNumber,
                          (double)param->vlasovNumber));
  }
  /* Setup Constants */
  {
    PetscDS    ds;
    Parameter *param;
    PetscCall(PetscBagGetData(ctx->bag, &param));
    PetscScalar constants[NUM_CONSTANTS];
    constants[SIGMA]   = param->sigma;
    constants[V0]      = param->v0;
    constants[T0]      = param->t0;
    constants[X0]      = param->x0;
    constants[M0]      = param->m0;
    constants[Q0]      = param->q0;
    constants[PHI0]    = param->phi0;
    constants[POISSON] = param->poissonNumber;
    constants[VLASOV]  = param->vlasovNumber;
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Cell DMs:
//   space:         fields      {coordinates, velocity}
//                  coordinates {coordinates}
//   velocity:      fields      {weight}
//                  coordinates {velocity}
//   moments:       fields      {weight}
//                  coordinates {coordinates}
//   moment fields: fields      {velocity}
//                  coordinates {coordinates}
static PetscErrorCode CreateSwarm(DM dm, AppCtx *ctx, DM *sw)
{
  DMSwarmCellDM celldm;
  DM            vdm;
  PetscReal     v0[2] = {1., 0.};
  PetscInt      dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSetApplicationContext(*sw, ctx));

  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "E_field", dim, PETSC_REAL));

  const char *fieldnames[2] = {DMSwarmPICField_coor, "velocity"};

  PetscCall(DMSwarmCellDMCreate(dm, 2, fieldnames, 1, fieldnames, &celldm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));

  const char *vfieldnames[2] = {"w_q"};

  PetscCall(CreateVelocityDM(*sw, &vdm));
  PetscCall(DMSwarmCellDMCreate(vdm, 1, vfieldnames, 1, &fieldnames[1], &celldm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));
  PetscCall(DMDestroy(&vdm));

  DM mdm;

  PetscCall(DMClone(dm, &mdm));
  PetscCall(PetscObjectSetName((PetscObject)mdm, "moments"));
  PetscCall(DMCopyDisc(dm, mdm));
  PetscCall(DMSwarmCellDMCreate(mdm, 1, vfieldnames, 1, fieldnames, &celldm));
  PetscCall(DMDestroy(&mdm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));

  DM mfdm;

  PetscCall(DMClone(dm, &mfdm));
  PetscCall(PetscObjectSetName((PetscObject)mfdm, "moment fields"));
  PetscCall(DMCopyDisc(dm, mfdm));
  PetscCall(DMSwarmCellDMCreate(mfdm, 1, &fieldnames[1], 1, fieldnames, &celldm));
  PetscCall(DMDestroy(&mfdm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));

  DM mpdm;

  PetscCall(DMCreate(PetscObjectComm((PetscObject)*sw), &mpdm));
  PetscCall(DMSetType(mpdm, DMPLEX));
  PetscCall(DMPlexSetOptionsPrefix(mpdm, "mp_"));
  PetscCall(DMSetFromOptions(mpdm));
  PetscCall(PetscObjectSetName((PetscObject)mpdm, "moment_phase"));
  PetscCall(DMViewFromOptions(mpdm, NULL, "-dm_view"));
  {
    PetscFE        fe;
    DMPolytopeType ct;
    PetscInt       mpdim, cStart;

    PetscCall(DMGetDimension(mpdm, &mpdim));
    PetscCall(DMPlexGetHeightStratum(mpdm, 0, &cStart, NULL));
    PetscCall(DMPlexGetCellType(mpdm, cStart, &ct));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, mpdim, 1, ct, "mp_", PETSC_DETERMINE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "moment_phase"));
    PetscCall(DMSetField(mpdm, 0, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(mpdm));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMSwarmCellDMCreate(mpdm, 1, vfieldnames, 2, fieldnames, &celldm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));
  PetscCall(DMDestroy(&mpdm));

  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSetUp(*sw));

  if (ctx->optimizeVelGrid) {
    DMSwarmCellDM celldm;
    DM            rdm;

    PetscCall(OptimizeRemapGrid(*sw));
    PetscCall(DMSwarmGetCellDMByName(*sw, "remap", &celldm));
    PetscCall(DMSwarmCellDMGetDM(celldm, &rdm));
    PetscCall(DMViewFromOptions(rdm, NULL, "-dm_view"));
  }

  PetscCall(DMSwarmSetCellDMActive(*sw, "space"));
  ctx->swarm = *sw;
  // TODO: This is redundant init since it is done in InitializeSolveAndSwarm, however DMSetUp() requires the local size be set
  if (ctx->perturbed_weights) {
    PetscCall(InitializeParticles_PerturbedWeights(*sw, ctx));
  } else {
    PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
    PetscCall(DMSwarmInitializeCoordinates(*sw));
    PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, v0));
  }
  // Create multilevel structure
  if (ctx->momCtx.dm) {
    PetscCall(DMSetCoarseDM(*sw, ctx->momCtx.dm));
    PetscCall(DMSetRefineLevel(*sw, 1));
  }

  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Coulomb(SNES snes, DM sw, PetscReal E[])
{
  AppCtx     *ctx;
  PetscReal  *coords;
  PetscInt   *species, dim, Np, Ns;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)snes), &size));
  PetscCheck(size == 1, PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "Coulomb code only works in serial");
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMGetApplicationContext(sw, &ctx));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  for (PetscInt p = 0; p < Np; ++p) {
    PetscReal *pcoord = &coords[p * dim];
    PetscReal  pE[3]  = {0., 0., 0.};

    /* Calculate field at particle p due to particle q */
    for (PetscInt q = 0; q < Np; ++q) {
      PetscReal *qcoord = &coords[q * dim];
      PetscReal  rpq[3], r, r3, q_q;

      if (p == q) continue;
      q_q = ctx->charges[species[q]] * 1.;
      for (PetscInt d = 0; d < dim; ++d) rpq[d] = pcoord[d] - qcoord[d];
      r = DMPlex_NormD_Internal(dim, rpq);
      if (r < PETSC_SQRT_MACHINE_EPSILON) continue;
      r3 = PetscPowRealInt(r, 3);
      for (PetscInt d = 0; d < dim; ++d) pE[d] += q_q * rpq[d] / r3;
    }
    for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = pE[d];
  }
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Primal(SNES snes, DM sw, Mat M_p, PetscReal E[])
{
  DM         dm;
  AppCtx    *ctx;
  PetscDS    ds;
  PetscFE    fe;
  KSP        ksp;
  Vec        rhoRhs;      // Weak charge density, \int phi_i rho
  Vec        rho;         // Charge density, M^{-1} rhoRhs
  Vec        phi, locPhi; // Potential
  Vec        f;           // Particle weights
  PetscReal *coords;
  PetscInt   dim, cStart, cEnd, Np;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(sw, &ctx));
  PetscCall(PetscLogEventBegin(ctx->ESolveEvent, snes, sw, 0, 0));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));

  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetGlobalVector(dm, &rhoRhs));
  PetscCall(PetscObjectSetName((PetscObject)rhoRhs, "Weak charge density"));
  PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "rho", &rho));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));

  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(ctx->M, NULL, "-m_view"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));

  PetscCall(MatMultTranspose(M_p, f, rhoRhs));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "em_proj_"));
  PetscCall(KSPSetOperators(ksp, ctx->M, ctx->M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhoRhs, rho));

  PetscCall(CheckNeutrality(dm, rho, ctx));
  PetscCall(VecScale(rhoRhs, -1.0));

  PetscCall(VecViewFromOptions(rhoRhs, NULL, "-rho_view"));
  PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "rho", &rho));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "phi", &phi));
  PetscCall(VecSet(phi, 0.0));
  PetscCall(SNESSolve(snes, rhoRhs, phi));
  PetscCall(DMRestoreGlobalVector(dm, &rhoRhs));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));

  PetscCall(DMGetLocalVector(dm, &locPhi));
  PetscCall(DMGlobalToLocalBegin(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "phi", &phi));
  PetscCall(PetscLogEventEnd(ctx->ESolveEvent, snes, sw, 0, 0));

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  PetscCall(PetscLogEventBegin(ctx->ETabEvent, snes, sw, 0, 0));
  PetscTabulation tab;
  PetscReal      *pcoord, *refcoord;
  PetscFEGeom    *chunkgeom = NULL;
  PetscInt        maxNcp    = 0;

  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt Ncp;

    PetscCall(DMSwarmSortGetNumberOfPointsPerCell(sw, c, &Ncp));
    maxNcp = PetscMax(maxNcp, Ncp);
  }
  PetscCall(DMGetWorkArray(dm, maxNcp * dim, MPIU_REAL, &refcoord));
  PetscCall(DMGetWorkArray(dm, maxNcp * dim, MPIU_REAL, &pcoord));
  // This can raise an FP_INEXACT in the dgemm inside
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall(PetscFECreateTabulation(fe, 1, maxNcp, refcoord, 1, &tab));
  PetscCall(PetscFPTrapPop());
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscScalar *clPhi = NULL;
    PetscInt    *points;
    PetscInt     Ncp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    for (PetscInt cp = 0; cp < Ncp; ++cp)
      for (PetscInt d = 0; d < dim; ++d) pcoord[cp * dim + d] = coords[points[cp] * dim + d];
    {
      PetscCall(PetscFEGeomGetChunk(ctx->fegeom, c - cStart, c - cStart + 1, &chunkgeom));
      for (PetscInt i = 0; i < Ncp; ++i) {
        const PetscReal x0[3] = {-1., -1., -1.};
        CoordinatesRealToRef(dim, dim, x0, chunkgeom->v, chunkgeom->invJ, &pcoord[dim * i], &refcoord[dim * i]);
      }
    }
    PetscCall(PetscFEComputeTabulation(fe, Ncp, refcoord, 1, tab));
    PetscCall(DMPlexVecGetClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    for (PetscInt cp = 0; cp < Ncp; ++cp) {
      const PetscReal *basisDer = tab->T[1];
      const PetscInt   p        = points[cp];

      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = 0.;
      PetscCall(PetscFEFreeInterpolateGradient_Static(fe, basisDer, clPhi, dim, chunkgeom->invJ, NULL, cp, &E[p * dim]));
      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] *= -1.0;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Ncp, &points));
  }
  PetscCall(DMRestoreWorkArray(dm, maxNcp * dim, MPIU_REAL, &pcoord));
  PetscCall(DMRestoreWorkArray(dm, maxNcp * dim, MPIU_REAL, &refcoord));
  PetscCall(PetscTabulationDestroy(&tab));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dm, &locPhi));
  PetscCall(PetscFEGeomRestoreChunk(ctx->fegeom, 0, 1, &chunkgeom));
  PetscCall(PetscLogEventEnd(ctx->ETabEvent, snes, sw, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Mixed(SNES snes, DM sw, Mat M_p, PetscReal E[])
{
  DM         dm;
  AppCtx    *ctx;
  PetscDS    ds;
  PetscFE    fe;
  KSP        ksp;
  Vec        rhoRhs, rhoRhsFull;   // Weak charge density, \int phi_i rho, and embedding in mixed problem
  Vec        rho;                  // Charge density, M^{-1} rhoRhs
  Vec        phi, locPhi, phiFull; // Potential and embedding in mixed problem
  Vec        f;                    // Particle weights
  PetscReal *coords;
  PetscInt   dim, cStart, cEnd, Np;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(sw, &ctx));
  PetscCall(PetscLogEventBegin(ctx->ESolveEvent, snes, sw, 0, 0));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));

  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetGlobalVector(ctx->dmPot, &rhoRhs));
  PetscCall(PetscObjectSetName((PetscObject)rhoRhs, "Weak charge density"));
  PetscCall(DMGetGlobalVector(dm, &rhoRhsFull));
  PetscCall(PetscObjectSetName((PetscObject)rhoRhsFull, "Weak charge density"));
  PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "rho", &rho));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));

  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(ctx->M, NULL, "-m_view"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));

  PetscCall(MatMultTranspose(M_p, f, rhoRhs));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "em_proj"));
  PetscCall(KSPSetOperators(ksp, ctx->M, ctx->M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhoRhs, rho));

  PetscCall(VecISCopy(rhoRhsFull, ctx->isPot, SCATTER_FORWARD, rhoRhs));
  //PetscCall(VecScale(rhoRhsFull, -1.0));

  PetscCall(VecViewFromOptions(rhoRhs, NULL, "-rho_view"));
  PetscCall(VecViewFromOptions(rhoRhsFull, NULL, "-rho_full_view"));
  PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "rho", &rho));
  PetscCall(DMRestoreGlobalVector(ctx->dmPot, &rhoRhs));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(DMGetGlobalVector(dm, &phiFull));
  PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "phi", &phi));
  PetscCall(VecSet(phiFull, 0.0));
  PetscCall(SNESSolve(snes, rhoRhsFull, phiFull));
  PetscCall(DMRestoreGlobalVector(dm, &rhoRhsFull));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));

  PetscCall(VecISCopy(phiFull, ctx->isPot, SCATTER_REVERSE, phi));
  PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "phi", &phi));

  PetscCall(DMGetLocalVector(dm, &locPhi));
  PetscCall(DMGlobalToLocalBegin(dm, phiFull, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(dm, phiFull, INSERT_VALUES, locPhi));
  PetscCall(DMRestoreGlobalVector(dm, &phiFull));
  PetscCall(PetscLogEventEnd(ctx->ESolveEvent, snes, sw, 0, 0));

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  PetscCall(PetscLogEventBegin(ctx->ETabEvent, snes, sw, 0, 0));
  PetscTabulation tab;
  PetscReal      *pcoord, *refcoord;
  PetscFEGeom    *chunkgeom = NULL;
  PetscInt        maxNcp    = 0;

  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt Ncp;

    PetscCall(DMSwarmSortGetNumberOfPointsPerCell(sw, c, &Ncp));
    maxNcp = PetscMax(maxNcp, Ncp);
  }
  PetscCall(DMGetWorkArray(dm, maxNcp * dim, MPIU_REAL, &refcoord));
  PetscCall(DMGetWorkArray(dm, maxNcp * dim, MPIU_REAL, &pcoord));
  PetscCall(PetscFECreateTabulation(fe, 1, maxNcp, refcoord, 1, &tab));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscScalar *clPhi = NULL;
    PetscInt    *points;
    PetscInt     Ncp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    for (PetscInt cp = 0; cp < Ncp; ++cp)
      for (PetscInt d = 0; d < dim; ++d) pcoord[cp * dim + d] = coords[points[cp] * dim + d];
    {
      PetscCall(PetscFEGeomGetChunk(ctx->fegeom, c - cStart, c - cStart + 1, &chunkgeom));
      for (PetscInt i = 0; i < Ncp; ++i) {
        const PetscReal x0[3] = {-1., -1., -1.};
        CoordinatesRealToRef(dim, dim, x0, chunkgeom->v, chunkgeom->invJ, &pcoord[dim * i], &refcoord[dim * i]);
      }
    }
    PetscCall(PetscFEComputeTabulation(fe, Ncp, refcoord, 1, tab));
    PetscCall(DMPlexVecGetClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    for (PetscInt cp = 0; cp < Ncp; ++cp) {
      const PetscInt p = points[cp];

      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = 0.;
      PetscCall(PetscFEInterpolateAtPoints_Static(fe, tab, clPhi, chunkgeom, cp, &E[p * dim]));
      PetscCall(PetscFEPushforward(fe, chunkgeom, 1, &E[p * dim]));
      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] *= -1.0;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Ncp, &points));
  }
  PetscCall(DMRestoreWorkArray(dm, maxNcp * dim, MPIU_REAL, &pcoord));
  PetscCall(DMRestoreWorkArray(dm, maxNcp * dim, MPIU_REAL, &refcoord));
  PetscCall(PetscTabulationDestroy(&tab));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dm, &locPhi));
  PetscCall(PetscFEGeomRestoreChunk(ctx->fegeom, 0, 1, &chunkgeom));
  PetscCall(PetscLogEventEnd(ctx->ETabEvent, snes, sw, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles(SNES snes, DM sw)
{
  AppCtx    *ctx;
  Mat        M_p;
  PetscReal *E;
  PetscInt   dim, Np;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(sw, DM_CLASSID, 2);
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetApplicationContext(sw, &ctx));

  PetscCall(DMSwarmSetCellDMActive(sw, "moments"));
  // TODO: Could share sort context with space cellDM
  PetscCall(DMSwarmMigrate(sw, PETSC_FALSE));
  PetscCall(DMCreateMassMatrix(sw, ctx->dmPot, &M_p));
  PetscCall(DMSwarmSetCellDMActive(sw, "space"));

  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(PetscArrayzero(E, Np * dim));
  ctx->validE = PETSC_TRUE;

  switch (ctx->em) {
  case EM_COULOMB:
    PetscCall(ComputeFieldAtParticles_Coulomb(snes, sw, E));
    break;
  case EM_PRIMAL:
    PetscCall(ComputeFieldAtParticles_Primal(snes, sw, M_p, E));
    break;
  case EM_MIXED:
    PetscCall(ComputeFieldAtParticles_Mixed(snes, sw, M_p, E));
    break;
  case EM_NONE:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No solver for electrostatic model %s", EMTypes[ctx->em]);
  }
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(MatDestroy(&M_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, PetscCtx ctx)
{
  DM                 sw;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscReal         *E, m_p = 1., q_p = -1.;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(ComputeFieldAtParticles(snes, sw));

  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(G, &g));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = u[(p * 2 + 1) * dim + d];
      g[(p * 2 + 1) * dim + d] = q_p * E[p * dim + d] / m_p;
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* J_{ij} = dF_i/dx_j
   J_p = (  0   1)
         (-w^2  0)
   TODO Now there is another term with w^2 from the electric field. I think we will need to invert the operator.
        Perhaps we can approximate the Jacobian using only the cellwise P-P gradient from Coulomb
*/
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, PetscCtx ctx)
{
  DM               sw;
  const PetscReal *coords, *vel;
  PetscInt         dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(MatGetOwnershipRange(J, &rStart, NULL));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    // TODO This is not right because dv/dx has the electric field in it
    PetscScalar vals[4] = {0., 1., -1., 0.};

    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(J, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionX(TS ts, PetscReal t, Vec V, Vec Xres, void *Ctx)
{
  AppCtx            *ctx = (AppCtx *)Ctx;
  DM                 sw;
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt           Np, p, d, dim;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->RhsXEvent, ts, 0, 0, 0));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(Xres, &Np));
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArray(Xres, &xres));
  Np /= dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) xres[p * dim + d] = v[p * dim + d];
  }
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArray(Xres, &xres));
  PetscCall(PetscLogEventEnd(ctx->RhsXEvent, ts, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionV(TS ts, PetscReal t, Vec X, Vec Vres, void *Ctx)
{
  DM                 sw;
  AppCtx            *ctx  = (AppCtx *)Ctx;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscReal         *E, m_p, q_p;
  PetscInt           Np, p, dim, d;
  Parameter         *param;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->RhsVEvent, ts, 0, 0, 0));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(ComputeFieldAtParticles(snes, sw));

  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(PetscBagGetData(ctx->bag, &param));
  m_p = ctx->masses[0] * param->m0;
  q_p = ctx->charges[0] * param->q0;
  PetscCall(VecGetLocalSize(Vres, &Np));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(Vres, &vres));
  Np /= dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) vres[p * dim + d] = q_p * E[p * dim + d] / m_p;
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  /*
    Synchronized, ordered output for parallel/sequential test cases.
    In the 1D (on the 2D mesh) case, every y component should be zero.
  */
  if (ctx->checkVRes) {
    PetscBool pr = ctx->checkVRes > 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt  step;

    PetscCall(TSGetStepNumber(ts, &step));
    if (pr) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "step: %" PetscInt_FMT "\n", step));
    for (PetscInt p = 0; p < Np; ++p) {
      if (pr) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Residual: %.12g %.12g\n", (double)PetscRealPart(vres[p * dim + 0]), (double)PetscRealPart(vres[p * dim + 1])));
      PetscCheck(PetscAbsScalar(vres[p * dim + 1]) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Y velocity should be 0., not %g", (double)PetscRealPart(vres[p * dim + 1]));
    }
    if (pr) PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  }
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(PetscLogEventEnd(ctx->RhsVEvent, ts, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Discrete Gradients Formulation: S, F, gradF (G) */
PetscErrorCode RHSJacobianS_Kinetic(TS ts, PetscReal t, Vec U, Mat S, PetscCtx ctx)
{
  PetscScalar vals[4] = {0., 1., -1., 0.};
  DM          sw;
  PetscInt    dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(MatGetOwnershipRange(S, &rStart, NULL));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(S, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSJacobianS_Moment(TS ts, PetscReal t, Vec U, Mat S, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSJacobianS(TS ts, PetscReal t, Vec U, Mat S, void *ctx)
{
  DM        dm;
  PetscBool isswarm, isplex;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(U, &dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMSWARM, &isswarm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isplex));
  if (isplex) PetscCall(RHSJacobianS_Moment(ts, t, U, S, ctx));
  else PetscCall(RHSJacobianS_Kinetic(ts, t, U, S, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSObjectiveF_Kinetic(TS ts, PetscReal t, Vec U, PetscScalar *F, void *Ctx)
{
  AppCtx            *ctx = (AppCtx *)Ctx;
  DM                 sw;
  Vec                phi;
  const PetscScalar *u;
  PetscInt           dim, Np, cStart, cEnd;
  PetscReal         *vel, *coords, m_p = 1.;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMPlexGetHeightStratum(ctx->dmPot, 0, &cStart, &cEnd));

  PetscCall(DMGetNamedGlobalVector(ctx->dmPot, "phi", &phi));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view_dg"));
  PetscCall(computeFieldEnergy(ctx->dmPot, phi, F));
  PetscCall(DMRestoreNamedGlobalVector(ctx->dmPot, "phi", &phi));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np /= 2 * dim;
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt *points;
    PetscInt  Ncp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    for (PetscInt cp = 0; cp < Ncp; ++cp) {
      const PetscInt  p  = points[cp];
      const PetscReal v2 = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 1) * dim], &u[(p * 2 + 1) * dim]);

      *F += 0.5 * m_p * v2;
    }
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Ncp, &points));
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSObjectiveF_Moment(TS ts, PetscReal t, Vec U, PetscScalar *F, void *ctx)
{
  PetscFunctionBeginUser;
  *F = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSObjectiveF(TS ts, PetscReal t, Vec U, PetscScalar *F, void *ctx)
{
  DM        dm;
  PetscBool isswarm, isplex;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(U, &dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMSWARM, &isswarm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isplex));
  if (isplex) PetscCall(RHSObjectiveF_Moment(ts, t, U, F, ctx));
  else PetscCall(RHSObjectiveF_Kinetic(ts, t, U, F, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* dF/dx = q E   dF/dv = v */
PetscErrorCode RHSFunctionG_Kinetic(TS ts, PetscReal t, Vec U, Vec G, PetscCtx ctx)
{
  DM                 sw;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscReal   *coords, *vel, *E;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscReal          m_p = 1., q_p = -1.;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(G, &g));

  PetscLogEvent COMPUTEFIELD;
  PetscCall(PetscLogEventRegister("COMPFIELDATPART", TS_CLASSID, &COMPUTEFIELD));
  PetscCall(PetscLogEventBegin(COMPUTEFIELD, 0, 0, 0, 0));
  PetscCall(ComputeFieldAtParticles(snes, sw));
  PetscCall(PetscLogEventEnd(COMPUTEFIELD, 0, 0, 0, 0));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = -(q_p / m_p) * E[p * dim + d];
      g[(p * 2 + 1) * dim + d] = m_p * u[(p * 2 + 1) * dim + d];
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IFunctionG_Moment(TS ts, PetscReal t, Vec U, Vec U_t, Vec G, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  DM      dm;

  PetscFunctionBeginUser;
  if (user->momCtx.moment_field_monitor) {
    DM  dm;
    Vec X0, Xdot;

    PetscCall(TSGetDM(ts, &dm));
    PetscCall(MonitorMomentFields_Continuum(ts, 1, t, U_t, &user->momCtx));
    PetscCall(MonitorMomentFields_Continuum(ts, 1, t, U, &user->momCtx));
    PetscCall(TSDiscGradGetX0AndXdot(ts, dm, &X0, &Xdot));
    PetscCall(MonitorMomentFields_Continuum(ts, 1, t, X0, &user->momCtx));
    PetscCall(MonitorMomentFields_Continuum(ts, 1, t, Xdot, &user->momCtx));
    PetscCall(TSDiscGradRestoreX0AndXdot(ts, dm, &X0, &Xdot));
  }
  PetscCall(VecGetDM(U, &dm));
  PetscCall(ComputeMomentEM(dm, t, U, NULL, ctx));
  {
    Vec locX, locX_t, locF;

    PetscCall(DMGetLocalVector(dm, &locX));
    PetscCall(DMGetLocalVector(dm, &locX_t));
    PetscCall(DMGetLocalVector(dm, &locF));
    PetscCall(VecZeroEntries(locX));
    PetscCall(VecZeroEntries(locX_t));
    PetscCall(DMPlexTSComputeBoundary(dm, t, locX, locX_t, ctx));
    PetscCall(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalEnd(dm, U, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalBegin(dm, U_t, INSERT_VALUES, locX_t));
    PetscCall(DMGlobalToLocalEnd(dm, U_t, INSERT_VALUES, locX_t));
    PetscCall(VecZeroEntries(locF));
    PetscCall(DMPlexTSComputeIFunctionFEM(dm, t, locX, locX_t, locF, ctx));
    PetscCall(VecZeroEntries(G));
    PetscCall(DMLocalToGlobalBegin(dm, locF, ADD_VALUES, G));
    PetscCall(DMLocalToGlobalEnd(dm, locF, ADD_VALUES, G));
    PetscCall(DMRestoreLocalVector(dm, &locX));
    PetscCall(DMRestoreLocalVector(dm, &locX_t));
    PetscCall(DMRestoreLocalVector(dm, &locF));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IJacobianG_Moment(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal shift, Mat J, Mat Jp, void *ctx)
{
  DM dm;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(U, &dm));
  {
    Vec locX, locX_t;

    PetscCall(DMGetLocalVector(dm, &locX));
    PetscCall(DMGetLocalVector(dm, &locX_t));
    PetscCall(VecZeroEntries(locX));
    PetscCall(VecZeroEntries(locX_t));
    PetscCall(DMPlexTSComputeBoundary(dm, t, locX, locX_t, ctx));
    PetscCall(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalEnd(dm, U, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalBegin(dm, U_t, INSERT_VALUES, locX_t));
    PetscCall(DMGlobalToLocalEnd(dm, U_t, INSERT_VALUES, locX_t));
    PetscCall(MatViewFromOptions(Jp, NULL, "-pre_view"));
    PetscCall(DMPlexTSComputeIJacobianFEM(dm, t, locX, locX_t, shift, J, Jp, ctx));
    PetscCall(DMRestoreLocalVector(dm, &locX));
    PetscCall(DMRestoreLocalVector(dm, &locX_t));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSFunctionG(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  DM        dm;
  PetscBool isswarm, isplex;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(U, &dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMSWARM, &isswarm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isplex));
  PetscCheck(!isplex, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "This should not be called with a Plex");
  PetscCall(RHSFunctionG_Kinetic(ts, t, U, G, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSolution(TS ts)
{
  DM       sw;
  Vec      u;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetBlockSize(u, dim));
  PetscCall(VecSetSizes(u, 2 * Np * dim, PETSC_DECIDE));
  PetscCall(VecSetUp(u));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetProblem(TS ts)
{
  AppCtx *ctx;
  DM      sw;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &ctx));
  // Define unified system for (X, V)
  {
    Mat      J;
    PetscInt dim, Np;

    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
    PetscCall(MatSetSizes(J, 2 * Np * dim, 2 * Np * dim, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSetBlockSize(J, 2 * dim));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSetUp(J));
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, ctx));
    PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, ctx));
    PetscCall(MatDestroy(&J));
  }
  /* Define split system for X and V */
  {
    Vec             u;
    IS              isx, isv, istmp;
    const PetscInt *idx;
    PetscInt        dim, Np, rstart;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(VecGetOwnershipRange(u, &rstart, NULL));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 0, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isx));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 1, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isv));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(TSRHSSplitSetIS(ts, "position", isx));
    PetscCall(TSRHSSplitSetIS(ts, "momentum", isv));
    PetscCall(ISDestroy(&isx));
    PetscCall(ISDestroy(&isv));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunctionX, ctx));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunctionV, ctx));
  }
  // Define symplectic formulation U_t = S . G, where G = grad F
  {
    PetscCall(TSDiscGradSetFormulation(ts, RHSJacobianS, RHSObjectiveF, RHSFunctionG, ctx));
    PetscCall(TSDiscGradSetImplicitFormulation(ts, IFunctionG_Moment, IJacobianG_Moment));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmTSRedistribute(TS ts)
{
  DM        sw;
  Vec       u;
  PetscReal t, maxt, dt;
  PetscInt  n, maxn;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetMaxTime(ts, &maxt));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &n));
  PetscCall(TSGetMaxSteps(ts, &maxn));

  PetscCall(TSReset(ts));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetTime(ts, t));
  PetscCall(TSSetMaxTime(ts, maxt));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetStepNumber(ts, n));
  PetscCall(TSSetMaxSteps(ts, maxn));

  PetscCall(CreateSolution(ts));
  PetscCall(SetProblem(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode line(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar x[], void *Ctx)
{
  DM        sw, cdm;
  PetscInt  Np;
  PetscReal low[2], high[2];
  AppCtx   *ctx = (AppCtx *)Ctx;

  sw = ctx->swarm;
  PetscCall(DMSwarmGetCellDM(sw, &cdm));
  // Get the bounding box so we can equally space the particles
  PetscCall(DMGetLocalBoundingBox(cdm, low, high));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  // shift it by h/2 so nothing is initialized directly on a boundary
  x[0] = ((high[0] - low[0]) / Np) * (p + 0.5);
  x[1] = 0.;
  return PETSC_SUCCESS;
}

/*
  InitializeSolveAndSwarm - Set the solution values to the swarm coordinates and velocities, and also possibly set the initial values.

  Input Parameters:
+ ts         - The TS
- useInitial - Flag to also set the initial conditions to the current coordinates and velocities and setup the problem

  Output Parameters:
. u - The initialized solution vector

  Level: advanced

.seealso: InitializeSolve()
*/
static PetscErrorCode InitializeSolveAndSwarm(TS ts, PetscBool useInitial)
{
  DM       sw;
  Vec      u, gc, gv;
  IS       isx, isv;
  PetscInt dim;
  AppCtx  *ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &ctx));
  PetscCall(DMGetDimension(sw, &dim));
  if (useInitial) {
    PetscReal v0[2] = {1., 0.};
    if (ctx->perturbed_weights) {
      PetscCall(InitializeParticles_PerturbedWeights(sw, ctx));
    } else {
      PetscCall(DMSwarmComputeLocalSizeFromOptions(sw));
      PetscCall(DMSwarmInitializeCoordinates(sw));
      PetscCall(DMSwarmInitializeVelocitiesFromOptions(sw, v0));
    }
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMSwarmTSRedistribute(ts));
  }
  PetscCall(DMSetUp(sw));
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
  PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(VecISCopy(u, isx, SCATTER_FORWARD, gc));
  PetscCall(VecISCopy(u, isv, SCATTER_FORWARD, gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  PetscFunctionBegin;
  PetscCall(TSSetSolution(ts, u));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MigrateParticles(TS ts)
{
  DM               sw, cdm;
  const PetscReal *L;
  AppCtx          *ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &ctx));
  PetscCall(DMViewFromOptions(sw, NULL, "-migrate_view_pre"));
  {
    Vec        u, gc, gv, position, momentum;
    IS         isx, isv;
    PetscReal *pos, *mom;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
    PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
    PetscCall(VecGetSubVector(u, isx, &position));
    PetscCall(VecGetSubVector(u, isv, &momentum));
    PetscCall(VecGetArray(position, &pos));
    PetscCall(VecGetArray(momentum, &mom));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(VecISCopy(u, isx, SCATTER_REVERSE, gc));
    PetscCall(VecISCopy(u, isv, SCATTER_REVERSE, gv));

    PetscCall(DMSwarmGetCellDM(sw, &cdm));
    PetscCall(DMGetPeriodicity(cdm, NULL, NULL, &L));
    PetscCheck(L, PetscObjectComm((PetscObject)cdm), PETSC_ERR_ARG_WRONG, "Mesh must be periodic");
    if ((L[0] || L[1]) >= 0.) {
      PetscReal *x, *v, upper[3], lower[3];
      PetscInt   Np, dim;

      PetscCall(DMSwarmGetLocalSize(sw, &Np));
      PetscCall(DMGetDimension(cdm, &dim));
      PetscCall(DMGetBoundingBox(cdm, lower, upper));
      PetscCall(VecGetArray(gc, &x));
      PetscCall(VecGetArray(gv, &v));
      for (PetscInt p = 0; p < Np; ++p) {
        for (PetscInt d = 0; d < dim; ++d) {
          if (pos[p * dim + d] < lower[d]) {
            x[p * dim + d] = pos[p * dim + d] + (upper[d] - lower[d]);
          } else if (pos[p * dim + d] > upper[d]) {
            x[p * dim + d] = pos[p * dim + d] - (upper[d] - lower[d]);
          } else {
            x[p * dim + d] = pos[p * dim + d];
          }
          PetscCheck(x[p * dim + d] >= lower[d] && x[p * dim + d] <= upper[d], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "p: %" PetscInt_FMT "x[%" PetscInt_FMT "] %g", p, d, (double)x[p * dim + d]);
          v[p * dim + d] = mom[p * dim + d];
        }
      }
      PetscCall(VecRestoreArray(gc, &x));
      PetscCall(VecRestoreArray(gv, &v));
    }
    PetscCall(VecRestoreArray(position, &pos));
    PetscCall(VecRestoreArray(momentum, &mom));
    PetscCall(VecRestoreSubVector(u, isx, &position));
    PetscCall(VecRestoreSubVector(u, isv, &momentum));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  }
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscInt step;

  PetscCall(TSGetStepNumber(ts, &step));
  if (!(step % ctx->remapFreq)) {
    // Monitor electric field before we destroy it
    PetscReal ptime;
    PetscInt  step;

    PetscCall(TSGetStepNumber(ts, &step));
    PetscCall(TSGetTime(ts, &ptime));
    if (ctx->efield_monitor) PetscCall(MonitorEField(ts, step, ptime, NULL, ctx));
    if (ctx->poisson_monitor) PetscCall(MonitorPoisson(ts, step, ptime, NULL, ctx));
    PetscCall(EnlargeVelocityGrids(sw));
    PetscCall(DMSwarmRemap(sw));
    ctx->validE = PETSC_FALSE;
  }
  // This MUST come last, since it recreates the subswarms and they must DMClone() the new swarm
  PetscCall(DMSwarmTSRedistribute(ts));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/snesimpl.h>

static PetscErrorCode DestroyMomentContext(MomStruct *user)
{
  PetscFunctionBegin;
  PetscCall(KSPDestroy(&user->ksp_fn));
  PetscCall(KSPDestroy(&user->ksp_fp));
  PetscCall(KSPDestroy(&user->ksp_fe));
  PetscCall(KSPDestroy(&user->ksp_p));
  PetscCall(VecDestroy(&user->work_p));
  PetscCall(DMDestroy(&user->dmN));
  PetscCall(ISDestroy(&user->isN));
  PetscCall(MatDestroy(&user->MN));
  PetscCall(MatDestroy(&user->IN));
  PetscCall(VecDestroy(&user->INscale));
  PetscCall(DMDestroy(&user->dmP));
  PetscCall(ISDestroy(&user->isP));
  PetscCall(MatDestroy(&user->MP));
  PetscCall(DMDestroy(&user->dmE));
  PetscCall(ISDestroy(&user->isE));
  PetscCall(MatDestroy(&user->ME));
  PetscCall(TSDestroy(&user->ts));
  PetscCall(DMDestroy(&user->dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sw;
  TS        ts;
  Vec       u;
  PetscReal dt;
  PetscInt  maxn;
  AppCtx    ctx;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscMemzero(&ctx, sizeof(ctx)));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &ctx.bag));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  PetscCall(CreatePoisson(dm, &ctx));
  PetscCall(CreateMomentFields(dm, &ctx));
  PetscCall(CreateMomentSolver(ctx.momCtx.dm, &ctx.momCtx.ts, &ctx));
  PetscCall(CreateSwarm(dm, &ctx, &sw));
  PetscCall(SetupParameters(PETSC_COMM_WORLD, &ctx));
  PetscCall(InitializeConstants(sw, &ctx));
  PetscCall(DMSetApplicationContext(sw, &ctx));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetTimeStep(ts, 0.00001));
  PetscCall(TSSetMaxSteps(ts, 100));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  if (ctx.efield_monitor) {
    PetscCall(TSMonitorSet(ts, MonitorEField, &ctx, NULL));
    PetscCall(TSMonitorSet(ctx.momCtx.ts, MonitorMomentEField, &ctx, NULL));
  }
  if (ctx.moment_monitor) PetscCall(TSMonitorSet(ts, MonitorMoments, &ctx, NULL));
  if (ctx.momCtx.moment_field_monitor) {
    PetscCall(TSMonitorSet(ts, MonitorMomentFields, &ctx.momCtx, NULL));
    PetscCall(TSMonitorSet(ctx.momCtx.ts, MonitorMomentFields, &ctx.momCtx, NULL));
  }
  if (ctx.initial_monitor) PetscCall(TSMonitorSet(ts, MonitorInitialConditions, &ctx, NULL));
  if (ctx.positions_monitor) PetscCall(TSMonitorSet(ts, MonitorPositions_2D, &ctx, NULL));
  if (ctx.poisson_monitor) PetscCall(TSMonitorSet(ts, MonitorPoisson, &ctx, NULL));
  if (ctx.velocity_monitor >= 0) PetscCall(TSMonitorSet(ts, MonitorVelocity, &ctx, NULL));

  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetMaxSteps(ts, &maxn));
  ctx.steps    = maxn;
  ctx.stepSize = dt;
  PetscCall(SetupContext(dm, sw, &ctx));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(TSSetPostStep(ts, MigrateParticles));
  PetscCall(CreateSolution(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(CheckNonNegativeWeights(sw, &ctx));
  PetscCall(TSSolve(ts, NULL));

  if (ctx.checkLandau) {
    // We should get a lookup table based on charge density and \hat k
    const PetscReal gammaEx = -0.15336;
    const PetscReal omegaEx = 1.4156;
    const PetscReal tol     = 1e-2;

    PetscCheck(PetscAbsReal((ctx.gamma - gammaEx) / gammaEx) < tol, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Invalid Landau gamma %g != %g", ctx.gamma, gammaEx);
    PetscCheck(PetscAbsReal((ctx.omega - omegaEx) / omegaEx) < tol, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Invalid Landau omega %g != %g", ctx.omega, omegaEx);
  }

  PetscCall(SNESDestroy(&ctx.snes));
  PetscCall(DMDestroy(&ctx.dmPot));
  PetscCall(ISDestroy(&ctx.isPot));
  PetscCall(MatDestroy(&ctx.M));
  PetscCall(PetscFEGeomDestroy(&ctx.fegeom));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(DestroyMomentContext(&ctx.momCtx));
  PetscCall(DestroyContext(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex double

  # This tests that we can compute the correct decay rate and frequency
  #   For gold runs, use -dm_plex_box_faces 160 -vdm_plex_box_faces 450 -remap_dm_plex_box_faces 80,150 -ts_max_steps 1000
  #                      -remap_freq 100 -emax_start_step 50 -emax_solve_step 100
  testset:
    args: -cosine_coefficients 0.01 -charges -1. -perturbed_weights -total_weight 1. \
          -dm_plex_dim 1 -dm_plex_box_faces 80 -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 \
            -dm_plex_box_bd periodic -dm_plex_hash_location \
          -vdm_plex_dim 1 -vdm_plex_box_faces 220 -vdm_plex_box_lower -6 -vdm_plex_box_upper 6 \
            -vpetscspace_degree 2 -vdm_plex_hash_location \
          -remap_freq 1 -dm_swarm_remap_type pfak -remap_dm_plex_dim 2 -remap_dm_plex_simplex 0 \
            -remap_dm_plex_box_faces 40,110 -remap_dm_plex_box_bd periodic,none \
            -remap_dm_plex_box_lower 0.,-6. -remap_dm_plex_box_upper 12.5664,6. \
            -remap_petscspace_degree 1 -remap_dm_plex_hash_location \
            -ftop_ksp_type lsqr -ftop_pc_type none -ftop_ksp_rtol 1.e-14 -ptof_pc_type lu \
          -em_type primal -em_petscspace_degree 1 -em_snes_atol 1.e-12 -em_snes_error_if_not_converged \
            -em_ksp_error_if_not_converged -em_pc_type svd -em_proj_pc_type lu \
          -ts_time_step 0.03 -ts_max_steps 2 -ts_max_time 100 \
          -emax_tao_type brgn -emax_tao_max_it 100 -emax_tao_brgn_regularization_type l2pure \
            -emax_tao_brgn_regularizer_weight 1e-5 -emax_tao_brgn_subsolver_tao_bnk_ksp_rtol 1e-12 \
            -emax_start_step 0 -emax_solve_step 1 \
          -output_step 1 -efield_monitor quiet

    test:
      suffix: landau_damping_1d_bs
      args: -ts_type basicsymplectic -ts_basicsymplectic_type 1

    test:
      suffix: landau_damping_1d_dg
      args: -ts_type discgrad -ts_discgrad_type average -snes_type qn

    # -info :snes
    # -em_snes_monitor -em_snes_converged_reason
    test:
      suffix: landau_damping_1d_dg_fas
      args: -dm_plex_box_faces 16 -vdm_plex_box_faces 16 -remap_dm_plex_box_faces 8,8 \
            -ts_type discgrad -ts_discgrad_type average -snes_type fas \
              -fas_levels_1_snes_max_it 50 -fas_levels_1_snes_atol 1e-10 -fas_levels_1_snes_norm_schedule always \
              -fas_coarse_snes_fd \
            -n_petscspace_degree 0 -p_petscspace_degree 1 -e_petscspace_degree 2 \
              -mom_snes_mf_operator -mom_ksp_type gmres -mom_pc_type lu \
            -mp_dm_plex_dim 2 -mp_dm_plex_simplex 0 \
              -mp_dm_plex_box_faces 16,16 -mp_dm_plex_box_bd periodic,none \
              -mp_dm_plex_box_lower 0.,-6. -mp_dm_plex_box_upper 12.5664,6. \
              -mp_petscspace_degree 2 -mp_dm_plex_hash_location \
              -ftor_ksp_type lsqr -ftor_pc_type none -ftor_ksp_rtol 1.e-14 \
            -mom_ts_monitor -mom_snes_monitor -mom_snes_converged_reason \
            -mom_ksp_monitor -mom_ksp_converged_reason -mom_ksp_error_if_not_converged \
            -mom_proj_ksp_monitor \
            -em_petscfe_default_quadrature_order 2 \
            -sw_view -snes_monitor -snes_converged_reason -snes_view \
            -fas_levels_1_snes_monitor -fas_levels_1_snes_converged_reason \
            -fas_coarse_snes_fd -fas_coarse_pc_type lu \
            -fas_coarse_snes_monitor -fas_coarse_snes_monitor_fields -fas_coarse_snes_converged_reason \
              -fas_coarse_snes_linesearch_monitor

    test:
      suffix: equilibrium_1d
      args: -dm_plex_box_faces 8 -vdm_plex_box_faces 8 -remap_dm_plex_box_faces 4,4 \
            -cosine_coefficients 0.0 -remap_freq 2 -opt_vel_grid

TEST*/
