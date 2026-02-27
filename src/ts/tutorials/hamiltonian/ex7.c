static char help[] = "Backward Semi-Lagrangian Vlasov-Poisson solver for Landau damping.\n\n";

#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <petscdraw.h>
#include <petsc/private/petscfeimpl.h>
#include "petscdm.h"
#include "petscdmlabel.h"

typedef struct {
  DM dmX;   /* 1D x-space DMPlex (periodic, MPI) */
  DM dmV;   /* 1D v-space DMPlex (bounded, SELF) */
  DM dmPot; /* 1D x-space DMPlex for phi(x), rho(x), E(x) */

  DM           dmScalar;
  PetscFEGeom *fegeomX;
  PetscFEGeom *fegeomV;
  PetscFEGeom *fegeomPot;

  Vec f; /* Global DG solution Vec */
  Vec f_xwork;
  Vec f_vwork;

  Vec  rho;
  Vec  phi;
  Vec  E_field;
  SNES snesPoisson;
  Mat  MassPot;
  KSP  kspMassPot;

  DM swX;
  DM swV;

  KSP kspMassX;
  KSP kspMassV;

  PetscInt   NvDOF;
  PetscInt   NxDOF_local;
  PetscReal *v_dof_coords;
  PetscInt  *v_cell_dofs;   /* v_cell_dofs[cv_local*NbV + b] = local DOF index for cell cv_local, basis b */
  PetscInt  *phys_to_local; /* phys_to_local[phys_cell_idx] = local cell index (0-based), or -1 if not owned */
  PetscReal *cell_x0;       /* cell_x0[cx_local] = left boundary of local cell cx_local */

  PetscInt  Nx;
  PetscInt  Nv;
  PetscReal x_max;
  PetscReal v_max;
  PetscReal alpha;
  PetscReal kwave;
  PetscReal sigma;
  PetscReal dt;
  PetscInt  steps;
  PetscReal t_final;
  PetscInt  ostep;

  PetscBool   efield_monitor;
  PetscBool   check_landau;
  PetscReal   gamma_measured;
  PetscReal   omega_measured;
  PetscDrawLG drawlgE;

  PetscLogEvent BSLStepEvent;
  PetscLogEvent PoissonEvent;
  PetscLogEvent RhoEvent;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->dmX            = NULL;
  options->dmV            = NULL;
  options->dmPot          = NULL;
  options->dmScalar       = NULL;
  options->fegeomX        = NULL;
  options->fegeomV        = NULL;
  options->fegeomPot      = NULL;
  options->f              = NULL;
  options->f_xwork        = NULL;
  options->f_vwork        = NULL;
  options->rho            = NULL;
  options->phi            = NULL;
  options->E_field        = NULL;
  options->snesPoisson    = NULL;
  options->MassPot        = NULL;
  options->kspMassPot     = NULL;
  options->swX            = NULL;
  options->swV            = NULL;
  options->kspMassX       = NULL;
  options->kspMassV       = NULL;
  options->v_dof_coords   = NULL;
  options->Nx             = 64;
  options->Nv             = 128;
  options->x_max          = 2.0 * PETSC_PI / 0.5;
  options->v_max          = 6.0;
  options->alpha          = 0.01;
  options->kwave          = 0.5;
  options->sigma          = 1.0;
  options->dt             = 0.1;
  options->steps          = 200;
  options->t_final        = 10.0;
  options->ostep          = 1;
  options->efield_monitor = PETSC_FALSE;
  options->check_landau   = PETSC_FALSE;
  options->drawlgE        = NULL;

  PetscOptionsBegin(comm, "", "BSL Vlasov-Poisson options", "DMSWARM");
  PetscCall(PetscOptionsInt("-Nx", "Number of x cells", __FILE__, options->Nx, &options->Nx, NULL));
  PetscCall(PetscOptionsInt("-Nv", "Number of v cells", __FILE__, options->Nv, &options->Nv, NULL));
  PetscCall(PetscOptionsReal("-alpha", "Perturbation amplitude", __FILE__, options->alpha, &options->alpha, NULL));
  PetscCall(PetscOptionsReal("-kwave", "Perturbation wave number", __FILE__, options->kwave, &options->kwave, NULL));
  PetscCall(PetscOptionsReal("-sigma", "Constant background ion charge density", __FILE__, options->sigma, &options->sigma, NULL));
  PetscCall(PetscOptionsReal("-v_max", "Velocity domain half-width", __FILE__, options->v_max, &options->v_max, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", __FILE__, options->steps, &options->steps, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", __FILE__, options->dt, &options->dt, NULL));
  PetscCall(PetscOptionsReal("-t_final", "Final simulation time", __FILE__, options->t_final, &options->t_final, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Output every N steps", __FILE__, options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsBool("-efield_monitor", "Log max E-field each step", __FILE__, options->efield_monitor, &options->efield_monitor, NULL));
  PetscCall(PetscOptionsBool("-check_landau", "Check Landau damping rate at end", __FILE__, options->check_landau, &options->check_landau, NULL));
  PetscOptionsEnd();

  options->x_max = 2.0 * PETSC_PI / options->kwave;

  PetscCall(PetscLogEventRegister("BSLStep", TS_CLASSID, &options->BSLStepEvent));
  PetscCall(PetscLogEventRegister("Poisson", TS_CLASSID, &options->PoissonEvent));
  PetscCall(PetscLogEventRegister("ChargeDensity", TS_CLASSID, &options->RhoEvent));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateXMesh(MPI_Comm comm, AppCtx *ctx)
{
  DMField         coordField;
  IS              cellIS;
  PetscQuadrature quad;
  PetscReal      *wt, *pt;
  PetscFE         cgfe, fe;
  PetscInt        dim, cdim, cStart, cEnd;
  DM              dm;

  PetscFunctionBeginUser;
  /* Set box bounds programmatically so the mesh has correct physical coordinates */
  {
    char buf[64];
    PetscCall(PetscSNPrintf(buf, sizeof(buf), "%g", (double)ctx->x_max));
    PetscCall(PetscOptionsSetValue(NULL, "-fx_dm_plex_box_upper", buf));
    PetscCall(PetscOptionsSetValue(NULL, "-fx_dm_plex_box_lower", "0"));
  }
  PetscCall(DMCreate(comm, &ctx->dmX));
  dm = ctx->dmX;
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetOptionsPrefix(dm, "fx_"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "x_space"));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, "fx_", PETSC_DETERMINE, &cgfe));
  {
    PetscSpace sp;
    PetscInt   k;
    PetscCall(PetscFEGetBasisSpace(cgfe, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &k, NULL));
    PetscCheck(k > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "CreateXMesh: CG FE degree is 0 (petscspace_degree not set?)");
  }
  PetscCall(PetscFECreateBrokenElement(cgfe, &fe));
  PetscCall(PetscFEDestroy(&cgfe));
  {
    PetscSpace sp;
    PetscInt   k;
    PetscCall(PetscFEGetBasisSpace(fe, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &k, NULL));
    PetscCheck(k > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "CreateXMesh: broken FE degree is 0 (petscspace_degree not set?)");
  }
  PetscCall(PetscObjectSetName((PetscObject)fe, "x-distribution"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));

  PetscCall(DMPlexCreateClosureIndex(dm, NULL));

  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, cEnd - cStart, cStart, 1, &cellIS));
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &quad));
  PetscCall(PetscMalloc1(1, &wt));
  PetscCall(PetscMalloc1(cdim, &pt));
  wt[0] = 1.0;
  for (PetscInt d = 0; d < cdim; ++d) pt[d] = -1.0;
  PetscCall(PetscQuadratureSetData(quad, cdim, 1, 1, pt, wt));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, PETSC_FEGEOM_BASIC, &ctx->fegeomX));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(ISDestroy(&cellIS));

  {
    Mat M;
    PC  pc;
    PetscCall(DMCreateMassMatrix(dm, dm, &M));
    PetscCall(KSPCreate(comm, &ctx->kspMassX));
    PetscCall(KSPSetOperators(ctx->kspMassX, M, M));
    PetscCall(KSPSetType(ctx->kspMassX, KSPPREONLY));
    PetscCall(KSPGetPC(ctx->kspMassX, &pc));
    PetscCall(PCSetType(pc, PCBJACOBI));
    PetscCall(KSPSetOptionsPrefix(ctx->kspMassX, "fx_mass_"));
    PetscCall(KSPSetFromOptions(ctx->kspMassX));
    PetscCall(KSPSetUp(ctx->kspMassX));
    PetscCall(MatDestroy(&M));
  }

  {
    PetscInt ncells_global;
    PetscInt ncells_local = cEnd - cStart;
    PetscCall(MPIU_Allreduce(&ncells_local, &ncells_global, 1, MPIU_INT, MPI_SUM, comm));
    ctx->Nx = ncells_global;
  }

  /* Build phys_to_local and cell_x0 mappings.
     DMPlex cell numbering is arbitrary; use FVM geometry to find each cell's
     physical position and map physical cell index -> local cell index. */
  {
    PetscReal h_x          = ctx->x_max / ctx->Nx;
    PetscInt  ncells_local = cEnd - cStart;

    PetscCall(PetscMalloc1(ctx->Nx, &ctx->phys_to_local));
    PetscCall(PetscMalloc1(ncells_local, &ctx->cell_x0));
    for (PetscInt i = 0; i < ctx->Nx; ++i) ctx->phys_to_local[i] = -1;

    for (PetscInt cx = cStart; cx < cEnd; ++cx) {
      PetscReal vol, centroid[3];
      PetscCall(DMPlexComputeCellGeometryFVM(dm, cx, &vol, centroid, NULL));
      PetscInt phys_idx = (PetscInt)(centroid[0] / h_x); /* floor: centroid of cell i is at (i+0.5)*h_x */
      if (phys_idx < 0) phys_idx = 0;
      if (phys_idx >= ctx->Nx) phys_idx = ctx->Nx - 1;
      PetscInt cx_local            = cx - cStart;
      ctx->phys_to_local[phys_idx] = cx_local;
      ctx->cell_x0[cx_local]       = centroid[0] - vol * 0.5;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateVMesh(MPI_Comm comm, AppCtx *ctx)
{
  DMField         coordField;
  IS              cellIS;
  PetscQuadrature quad;
  PetscReal      *wt, *pt;
  PetscFE         fe;
  PetscInt        dim, cdim, cStart, cEnd;
  DM              dm;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(PETSC_COMM_SELF, &ctx->dmV));
  dm = ctx->dmV;
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetOptionsPrefix(dm, "fv_"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "v_space"));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, "fv_", PETSC_DETERMINE, &fe));
  {
    PetscSpace sp;
    PetscInt   k;
    PetscCall(PetscFEGetBasisSpace(fe, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &k, NULL));
    PetscCheck(k > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "CreateVMesh: FE degree is 0 (petscspace_degree not set?)");
  }
  PetscCall(PetscObjectSetName((PetscObject)fe, "v-distribution"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));

  PetscCall(DMPlexCreateClosureIndex(dm, NULL));

  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, cEnd - cStart, cStart, 1, &cellIS));
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &quad));
  PetscCall(PetscMalloc1(1, &wt));
  PetscCall(PetscMalloc1(cdim, &pt));
  wt[0] = 1.0;
  for (PetscInt d = 0; d < cdim; ++d) pt[d] = -1.0;
  PetscCall(PetscQuadratureSetData(quad, cdim, 1, 1, pt, wt));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, PETSC_FEGEOM_BASIC, &ctx->fegeomV));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(ISDestroy(&cellIS));

  {
    Mat M;
    PC  pc;
    PetscCall(DMCreateMassMatrix(dm, dm, &M));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &ctx->kspMassV));
    PetscCall(KSPSetOperators(ctx->kspMassV, M, M));
    PetscCall(KSPSetType(ctx->kspMassV, KSPPREONLY));
    PetscCall(KSPGetPC(ctx->kspMassV, &pc));
    PetscCall(PCSetType(pc, PCLU));
    PetscCall(KSPSetOptionsPrefix(ctx->kspMassV, "fv_mass_"));
    PetscCall(KSPSetFromOptions(ctx->kspMassV));
    PetscCall(KSPSetUp(ctx->kspMassV));
    PetscCall(MatDestroy(&M));
  }
  ctx->Nv = cEnd - cStart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void ion_f0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = constants[0];
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

static PetscErrorCode CreatePotentialMeshAndPoisson(MPI_Comm comm, AppCtx *ctx)
{
  PetscFE         fephi;
  PetscDS         ds;
  DMField         coordField;
  IS              cellIS;
  PetscQuadrature quad;
  PetscReal      *wt, *pt;
  PetscInt        dim, cStart, cEnd, cdim;
  Mat             J;

  PetscFunctionBeginUser;
  /* Clone dmX so dmPot has the SAME topology and partition.
     Using DMCreate+DMSetFromOptions independently would give a different
     partition in MPI, making cell indices incompatible between dmX and dmPot. */
  PetscCall(DMClone(ctx->dmX, &ctx->dmPot));
  PetscCall(PetscObjectSetName((PetscObject)ctx->dmPot, "potential"));

  PetscCall(DMGetDimension(ctx->dmPot, &dim));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, "x_", PETSC_DETERMINE, &fephi));
  {
    PetscSpace sp;
    PetscInt   k;
    PetscCall(PetscFEGetBasisSpace(fephi, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &k, NULL));
    PetscCheck(k > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "CreatePotentialMesh: FE degree is 0 (petscspace_degree not set?)");
  }
  PetscCall(PetscObjectSetName((PetscObject)fephi, "potential"));
  PetscCall(DMSetField(ctx->dmPot, 0, NULL, (PetscObject)fephi));
  PetscCall(DMCreateDS(ctx->dmPot));
  PetscCall(PetscFEDestroy(&fephi));

  PetscCall(DMGetDS(ctx->dmPot, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, ion_f0, laplacian_f1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, laplacian_g3));
  {
    PetscScalar sigma_const = (PetscScalar)ctx->sigma;
    PetscCall(PetscDSSetConstants(ds, 1, &sigma_const));
  }

  PetscCall(DMCreateMatrix(ctx->dmPot, &J));
  PetscCall(MatSetOption(J, MAT_SYMMETRIC, PETSC_TRUE));
  {
    MatNullSpace nullSpace;
    PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullSpace));
    PetscCall(MatSetNullSpace(J, nullSpace));
    PetscCall(MatNullSpaceDestroy(&nullSpace));
  }

  PetscCall(SNESCreate(comm, &ctx->snesPoisson));
  PetscCall(SNESSetDM(ctx->snesPoisson, ctx->dmPot));
  PetscCall(DMPlexSetSNESLocalFEM(ctx->dmPot, PETSC_FALSE, ctx));
  PetscCall(SNESSetJacobian(ctx->snesPoisson, J, J, NULL, NULL));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESSetOptionsPrefix(ctx->snesPoisson, "em_"));
  PetscCall(SNESSetFromOptions(ctx->snesPoisson));
  PetscCall(DMSetUp(ctx->dmPot));

  PetscCall(DMCreateGlobalVector(ctx->dmPot, &ctx->rho));
  PetscCall(DMCreateGlobalVector(ctx->dmPot, &ctx->phi));
  PetscCall(DMCreateGlobalVector(ctx->dmPot, &ctx->E_field));

  PetscCall(DMCreateMassMatrix(ctx->dmPot, ctx->dmPot, &ctx->MassPot));
  {
    PC pc;
    PetscCall(KSPCreate(comm, &ctx->kspMassPot));
    PetscCall(KSPSetOperators(ctx->kspMassPot, ctx->MassPot, ctx->MassPot));
    /* Use CG+Jacobi: PCBJACOBI ignores off-diagonal blocks at partition
       boundaries for CG mass matrix, giving wrong results in parallel. */
    PetscCall(KSPSetType(ctx->kspMassPot, KSPCG));
    PetscCall(KSPGetPC(ctx->kspMassPot, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetTolerances(ctx->kspMassPot, 1e-10, 1e-12, PETSC_DEFAULT, 200));
    PetscCall(KSPSetOptionsPrefix(ctx->kspMassPot, "rho_mass_"));
    PetscCall(KSPSetFromOptions(ctx->kspMassPot));
    PetscCall(KSPSetUp(ctx->kspMassPot));
  }

  PetscCall(DMPlexCreateClosureIndex(ctx->dmPot, NULL));

  PetscCall(DMPlexGetHeightStratum(ctx->dmPot, 0, &cStart, &cEnd));
  PetscCall(DMGetCoordinateField(ctx->dmPot, &coordField));
  PetscCall(DMGetCoordinateDim(ctx->dmPot, &cdim));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, cEnd - cStart, cStart, 1, &cellIS));
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &quad));
  PetscCall(PetscMalloc1(1, &wt));
  PetscCall(PetscMalloc1(cdim, &pt));
  wt[0] = 1.0;
  for (PetscInt d = 0; d < cdim; ++d) pt[d] = -1.0;
  PetscCall(PetscQuadratureSetData(quad, cdim, 1, 1, pt, wt));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, PETSC_FEGEOM_BASIC, &ctx->fegeomPot));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(ISDestroy(&cellIS));

  ctx->dmScalar = ctx->dmPot;
  PetscCall(PetscObjectReference((PetscObject)ctx->dmScalar));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AllocateF(AppCtx *ctx)
{
  PetscSection sectionV;
  PetscInt     nlocalX, nlocalV;

  PetscFunctionBeginUser;
  /* NxDOF_local: owned DOF count from the global Vec (no ghosts).
     f is a PETSC_COMM_WORLD Vec — each rank owns NvDOF * NxDOF_local entries.
     The kspMassX and f_xwork operate on the same owned layout. */
  PetscCall(DMCreateGlobalVector(ctx->dmX, &ctx->f_xwork));
  PetscCall(VecGetLocalSize(ctx->f_xwork, &nlocalX));
  ctx->NxDOF_local = nlocalX;

  PetscCall(DMGetLocalSection(ctx->dmV, &sectionV));
  PetscCall(PetscSectionGetStorageSize(sectionV, &nlocalV));
  ctx->NvDOF = nlocalV;

  /* f is a global (PETSC_COMM_WORLD) Vec: local size NvDOF * NxDOF_local per rank */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ctx->f));
  PetscCall(VecSetSizes(ctx->f, ctx->NvDOF * ctx->NxDOF_local, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(ctx->f));
  PetscCall(PetscObjectSetName((PetscObject)ctx->f, "f_dist"));

  PetscCall(DMCreateGlobalVector(ctx->dmV, &ctx->f_vwork));

  PetscCall(PetscMalloc1(ctx->NvDOF, &ctx->v_dof_coords));

  /* Build v_cell_dofs: for each v-cell and each basis function, store the local DOF index.
     DMPlexGetClosureIndices with useClPerm=PETSC_TRUE returns indices in the order that
     matches the FE basis tabulation, so v_cell_dofs[cv_local*NbV + b] is the local DOF
     index for cell cv_local, basis function b. This works for any polynomial degree. */
  {
    PetscFE  feV;
    PetscInt NbV, cStartV, cEndV;
    PetscCall(DMGetField(ctx->dmV, 0, NULL, (PetscObject *)&feV));
    PetscCall(PetscFEGetDimension(feV, &NbV));
    PetscCall(DMPlexGetHeightStratum(ctx->dmV, 0, &cStartV, &cEndV));
    PetscCall(PetscMalloc1(ctx->Nv * NbV, &ctx->v_cell_dofs));
    for (PetscInt cv = cStartV; cv < cEndV; ++cv) {
      PetscInt numIndices, *indices;
      PetscCall(DMPlexGetClosureIndices(ctx->dmV, sectionV, sectionV, cv, PETSC_TRUE, &numIndices, &indices, NULL, NULL));
      for (PetscInt b = 0; b < NbV; ++b) ctx->v_cell_dofs[(cv - cStartV) * NbV + b] = indices[b];
      PetscCall(DMPlexRestoreClosureIndices(ctx->dmV, sectionV, sectionV, cv, PETSC_TRUE, &numIndices, &indices, NULL, NULL));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupXSwarm(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(PETSC_COMM_WORLD, &ctx->swX));
  PetscCall(DMSetType(ctx->swX, DMSWARM));
  PetscCall(DMSetDimension(ctx->swX, 1));
  PetscCall(DMSwarmSetType(ctx->swX, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(ctx->swX, ctx->dmX));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swX, "v_value", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swX, "f_val", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swX, "quad_weight", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swX, "iv_dof", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swX, "ix_src", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swX, "x_init", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(ctx->swX));
  PetscCall(DMSetFromOptions(ctx->swX));
  PetscCall(DMSetUp(ctx->swX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupVSwarm(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(PETSC_COMM_SELF, &ctx->swV));
  PetscCall(DMSetType(ctx->swV, DMSWARM));
  PetscCall(DMSetDimension(ctx->swV, 1));
  PetscCall(DMSwarmSetType(ctx->swV, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(ctx->swV, ctx->dmV));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swV, "x_value", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swV, "E_value", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swV, "f_val", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swV, "quad_weight", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(ctx->swV, "ixdof_local", 1, PETSC_INT));
  PetscCall(DMSwarmFinalizeFieldRegister(ctx->swV));
  PetscCall(DMSetFromOptions(ctx->swV));
  PetscCall(DMSetUp(ctx->swV));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeF(AppCtx *ctx)
{
  PetscInt     cStartX, cEndX, cStartV, cEndV;
  PetscInt     cx, cv, bx, bv;
  PetscScalar *f_array;
  PetscInt     NbX, NbV;
  PetscReal    h_x, h_v, v_start;
  PetscReal    refNodes[3] = {-1.0, 0.0, 1.0};

  PetscFunctionBeginUser;
  PetscCall(VecGetArray(ctx->f, &f_array));
  PetscCall(DMPlexGetHeightStratum(ctx->dmX, 0, &cStartX, &cEndX));
  PetscCall(DMPlexGetHeightStratum(ctx->dmV, 0, &cStartV, &cEndV));

  {
    PetscFE fe;
    PetscCall(DMGetField(ctx->dmX, 0, NULL, (PetscObject *)&fe));
    PetscCall(PetscFEGetDimension(fe, &NbX));
    PetscCall(DMGetField(ctx->dmV, 0, NULL, (PetscObject *)&fe));
    PetscCall(PetscFEGetDimension(fe, &NbV));
  }

  h_x     = ctx->x_max / ctx->Nx;
  h_v     = 2.0 * ctx->v_max / ctx->Nv;
  v_start = -ctx->v_max;

  /* First pass: set v_dof_coords for all CG nodes using the cell-to-DOF mapping */
  for (cv = cStartV; cv < cEndV; ++cv) {
    PetscInt iv = cv - cStartV;
    for (bv = 0; bv < NbV; ++bv) {
      PetscReal xi_v     = (NbV == 1) ? 0.0 : refNodes[bv];
      PetscReal v_center = v_start + (iv + 0.5) * h_v;
      PetscReal v_val    = v_center + xi_v * (h_v * 0.5);
      PetscInt  iv_dof   = ctx->v_cell_dofs[iv * NbV + bv]; /* actual local DOF index */
      if (iv_dof >= 0 && iv_dof < ctx->NvDOF) ctx->v_dof_coords[iv_dof] = v_val;
    }
  }

  /* Second pass: initialize f array using the cell-to-DOF mapping */
  for (cv = cStartV; cv < cEndV; ++cv) {
    PetscInt iv = cv - cStartV;
    for (bv = 0; bv < NbV; ++bv) {
      PetscInt  iv_dof = ctx->v_cell_dofs[iv * NbV + bv]; /* actual local DOF index */
      PetscReal v_val  = ctx->v_dof_coords[iv_dof];

      for (cx = cStartX; cx < cEndX; ++cx) {
        PetscInt ix = cx - cStartX;
        for (bx = 0; bx < NbX; ++bx) {
          PetscReal xi_x        = (NbX == 1) ? 0.0 : refNodes[bx % 2];
          PetscReal x_val       = ctx->cell_x0[ix] + (xi_x + 1.0) * h_x * 0.5;
          PetscReal f0          = (1.0 + ctx->alpha * PetscCosReal(ctx->kwave * x_val)) / PetscSqrtReal(2.0 * PETSC_PI) * PetscExpReal(-0.5 * v_val * v_val);
          PetscInt  ixdof_local = ix * NbX + bx;
          PetscInt  idx         = iv_dof * ctx->NxDOF_local + ixdof_local;
          f_array[idx]          = (PetscScalar)f0;
        }
      }
    }
  }
  PetscCall(VecRestoreArray(ctx->f, &f_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeChargeDensity(AppCtx *ctx)
{
  PetscScalar     *f_array, *rho_dg_array;
  PetscReal       *v_basis_integrals;
  PetscInt         cStartV, cEndV, NbV, NqV;
  PetscInt         cStartX, cEndX;
  PetscFE          feV;
  PetscQuadrature  quadV;
  const PetscReal *wqV, *xiqV;
  PetscTabulation  tabV;
  Vec              rhs_global, rhs_local;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(ctx->NvDOF, &v_basis_integrals));
  PetscCall(DMPlexGetHeightStratum(ctx->dmV, 0, &cStartV, &cEndV));
  PetscCall(DMGetField(ctx->dmV, 0, NULL, (PetscObject *)&feV));
  PetscCall(PetscFEGetDimension(feV, &NbV));
  PetscCall(PetscFEGetQuadrature(feV, &quadV));
  PetscCall(PetscQuadratureGetData(quadV, NULL, NULL, &NqV, &xiqV, &wqV));
  PetscCall(PetscFECreateTabulation(feV, 1, NqV, xiqV, 0, &tabV));

  /* Uniform mesh cheat for detJ */
  PetscReal h_v  = 2.0 * ctx->v_max / ctx->Nv;
  PetscReal detJ = h_v * 0.5;

  /* Integrate each v-basis function over its support */
  for (PetscInt cv = cStartV; cv < cEndV; ++cv) {
    for (PetscInt bv = 0; bv < NbV; ++bv) {
      PetscReal sum = 0.0;
      for (PetscInt q = 0; q < NqV; ++q) { sum += tabV->T[0][q * NbV + bv] * wqV[q] * detJ; }
      PetscInt iv_dof = ctx->v_cell_dofs[(cv - cStartV) * NbV + bv]; /* actual local DOF index */
      if (iv_dof >= 0) v_basis_integrals[iv_dof] += sum;
    }
  }
  PetscCall(PetscTabulationDestroy(&tabV));

  PetscCall(PetscMalloc1(ctx->NxDOF_local, &rho_dg_array));
  for (PetscInt i = 0; i < ctx->NxDOF_local; i++) rho_dg_array[i] = 0.0;

  PetscCall(VecGetArray(ctx->f, &f_array));
  for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
    PetscReal weight = v_basis_integrals[iv];
    PetscInt  offset = iv * ctx->NxDOF_local;
    for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix) { rho_dg_array[ix] += f_array[offset + ix] * weight; }
  }
  PetscCall(VecRestoreArray(ctx->f, &f_array));
  PetscCall(PetscFree(v_basis_integrals));
  PetscCall(DMGetGlobalVector(ctx->dmPot, &rhs_global));
  PetscCall(VecZeroEntries(rhs_global));
  PetscCall(DMGetLocalVector(ctx->dmPot, &rhs_local));
  PetscCall(VecZeroEntries(rhs_local));

  PetscCall(DMPlexGetHeightStratum(ctx->dmX, 0, &cStartX, &cEndX));
  {
    PetscFE          feX, fePot;
    PetscInt         NbX, NbPot, Nq;
    PetscQuadrature  quad;
    const PetscReal *wq, *xiq;
    PetscTabulation  tabX, tabPot;
    PetscSection     sectionPot;

    PetscCall(DMGetField(ctx->dmX, 0, NULL, (PetscObject *)&feX));
    PetscCall(PetscFEGetDimension(feX, &NbX));
    PetscCall(DMGetField(ctx->dmPot, 0, NULL, (PetscObject *)&fePot));
    PetscCall(PetscFEGetDimension(fePot, &NbPot));
    PetscCall(DMGetLocalSection(ctx->dmPot, &sectionPot));
    PetscCall(PetscFEGetQuadrature(fePot, &quad));
    PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, &xiq, &wq));
    PetscCall(PetscFECreateTabulation(feX, 1, Nq, xiq, 0, &tabX));
    PetscCall(PetscFECreateTabulation(fePot, 1, Nq, xiq, 0, &tabPot));

    PetscReal h_x  = ctx->x_max / ctx->Nx;
    PetscReal detJ = h_x * 0.5;

    for (PetscInt cx = cStartX; cx < cEndX; ++cx) {
      PetscScalar *rhs_elem;
      PetscCall(PetscMalloc1(NbPot, &rhs_elem));
      for (int i = 0; i < NbPot; i++) rhs_elem[i] = 0.0;
      PetscScalar *rho_cell_coeffs = &rho_dg_array[(cx - cStartX) * NbX];

      for (PetscInt q = 0; q < Nq; ++q) {
        PetscScalar rho_val = 0.0;
        for (PetscInt bx = 0; bx < NbX; ++bx) { rho_val += rho_cell_coeffs[bx] * tabX->T[0][q * NbX + bx]; }
        for (PetscInt b_pot = 0; b_pot < NbPot; ++b_pot) { rhs_elem[b_pot] += rho_val * tabPot->T[0][q * NbPot + b_pot] * wq[q] * detJ; }
      }
      PetscCall(DMPlexVecSetClosure(ctx->dmPot, sectionPot, rhs_local, cx, rhs_elem, ADD_VALUES));
      PetscCall(PetscFree(rhs_elem));
    }
    PetscCall(PetscTabulationDestroy(&tabX));
    PetscCall(PetscTabulationDestroy(&tabPot));
  }
  PetscCall(DMLocalToGlobal(ctx->dmPot, rhs_local, ADD_VALUES, rhs_global));
  PetscCall(DMRestoreLocalVector(ctx->dmPot, &rhs_local));
  PetscCall(KSPSolve(ctx->kspMassPot, rhs_global, ctx->rho));
  PetscCall(DMRestoreGlobalVector(ctx->dmPot, &rhs_global));
  PetscCall(PetscFree(rho_dg_array));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolvePoisson(DM dmPot, Vec rho, Vec phi, AppCtx *ctx)
{
  Vec                 rho_rhs;
  SNESConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(rho, &rho_rhs));

  /* Solve: -phi'' = rho - sigma.
     Weak form: (grad phi, grad psi) + (sigma, psi) = (rho, psi).
     LHS is handled by DS residual F(phi) with f0 = sigma.
     RHS is b = (rho, psi) = M * rho.
     SNESSolve solves F(phi) = b.
  */
  PetscCall(MatMult(ctx->MassPot, rho, rho_rhs));

  PetscCall(VecZeroEntries(phi));
  PetscCall(SNESSolve(ctx->snesPoisson, rho_rhs, phi));

  PetscCall(SNESGetConvergedReason(ctx->snesPoisson, &reason));
  if (reason < 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "WARNING: SNESPoisson failed to converge, reason: %d\n", reason));

  PetscCall(VecDestroy(&rho_rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdvectX(Vec f, Vec f_out, PetscReal s, AppCtx *ctx)
{
  PetscInt     p, np, cStartX, cEndX;
  PetscScalar *fa;
  PetscReal   *coords, *v_vals, *weights, *x_inits;
  PetscInt    *iv_dofs, *ix_srcs;
  PetscInt     dim = 1;
  PetscScalar *f_out_arr;

  PetscFunctionBeginUser;

  /* 1. Populate swX */
  PetscCall(DMSwarmSetLocalSizes(ctx->swX, 0, 0));
  PetscCall(DMPlexGetHeightStratum(ctx->dmX, 0, &cStartX, &cEndX));

  PetscFE          feX;
  PetscQuadrature  quadX;
  PetscInt         NqX, NbX;
  const PetscReal *xq, *wq;
  PetscCall(DMGetField(ctx->dmX, 0, NULL, (PetscObject *)&feX));
  PetscCall(PetscFEGetQuadrature(feX, &quadX));
  PetscCall(PetscQuadratureGetData(quadX, NULL, NULL, &NqX, &xq, &wq));
  PetscCall(PetscFEGetDimension(feX, &NbX));

  PetscInt n_particles = (cEndX - cStartX) * NqX * ctx->NvDOF;
  PetscCall(DMSwarmSetLocalSizes(ctx->swX, n_particles, -1));

  PetscCall(DMSwarmGetField(ctx->swX, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(ctx->swX, "v_value", NULL, NULL, (void **)&v_vals));
  PetscCall(DMSwarmGetField(ctx->swX, "quad_weight", NULL, NULL, (void **)&weights));
  PetscCall(DMSwarmGetField(ctx->swX, "iv_dof", NULL, NULL, (void **)&iv_dofs));
  PetscCall(DMSwarmGetField(ctx->swX, "ix_src", NULL, NULL, (void **)&ix_srcs));
  PetscCall(DMSwarmGetField(ctx->swX, "x_init", NULL, NULL, (void **)&x_inits));

  PetscInt  p_idx = 0;
  PetscReal h_x   = ctx->x_max / ctx->Nx;

  for (PetscInt cx = cStartX; cx < cEndX; ++cx) {
    PetscReal vol, centroid[3];
    PetscCall(DMPlexComputeCellGeometryFVM(ctx->dmX, cx, &vol, centroid, NULL));
    PetscReal detJ = vol * 0.5;
    PetscReal x0   = centroid[0] - vol * 0.5;

    for (PetscInt q = 0; q < NqX; ++q) {
      PetscReal xi     = xq[q];
      PetscReal x_phys = x0 + (xi + 1.0) * (h_x * 0.5);

      for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
        PetscReal v      = ctx->v_dof_coords[iv];
        PetscReal x_foot = x_phys - v * s;

        /* Periodic Wrap */
        if (x_foot < 0) x_foot += ctx->x_max;
        if (x_foot >= ctx->x_max) x_foot -= ctx->x_max;
        while (x_foot < 0) x_foot += ctx->x_max;
        while (x_foot >= ctx->x_max) x_foot -= ctx->x_max;

        coords[p_idx * dim] = x_foot;
        v_vals[p_idx]       = v;
        weights[p_idx]      = wq[q] * detJ;
        iv_dofs[p_idx]      = iv;
        ix_srcs[p_idx]      = (PetscInt)((x_phys / h_x) + 0.0); /* Physical cell index */
        if (ix_srcs[p_idx] >= ctx->Nx) ix_srcs[p_idx] = ctx->Nx - 1;
        if (ix_srcs[p_idx] < 0) ix_srcs[p_idx] = 0;
        x_inits[p_idx] = x_phys; /* Store return coordinate */
        p_idx++;
      }
    }
  }

  PetscCall(DMSwarmRestoreField(ctx->swX, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(ctx->swX, "v_value", NULL, NULL, (void **)&v_vals));
  PetscCall(DMSwarmRestoreField(ctx->swX, "quad_weight", NULL, NULL, (void **)&weights));
  PetscCall(DMSwarmRestoreField(ctx->swX, "iv_dof", NULL, NULL, (void **)&iv_dofs));
  PetscCall(DMSwarmRestoreField(ctx->swX, "ix_src", NULL, NULL, (void **)&ix_srcs));
  PetscCall(DMSwarmRestoreField(ctx->swX, "x_init", NULL, NULL, (void **)&x_inits));

  /* 2. Migrate Forward */
  PetscCall(DMSwarmMigrate(ctx->swX, PETSC_TRUE));

  /* 3. Interpolate */
  PetscCall(DMSwarmGetLocalSize(ctx->swX, &np));
  PetscCall(DMSwarmGetField(ctx->swX, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(ctx->swX, "iv_dof", NULL, NULL, (void **)&iv_dofs));
  PetscCall(DMSwarmGetField(ctx->swX, "f_val", NULL, NULL, (void **)&fa));

  const PetscScalar *f_arr_in;
  PetscCall(VecGetArrayRead(f, &f_arr_in));

  PetscTabulation T;

  for (p = 0; p < np; ++p) {
    PetscReal x  = coords[p * dim];
    PetscInt  iv = iv_dofs[p];

    /* Use phys_to_local to map physical cell index -> local cell index.
        This handles arbitrary DM partitioning where local cell numbering
        does not match physical order. */
    PetscInt cx_phys = (PetscInt)(x / h_x);
    if (cx_phys >= ctx->Nx) cx_phys = ctx->Nx - 1;
    if (cx_phys < 0) cx_phys = 0;
    PetscInt  cx_local = ctx->phys_to_local[cx_phys];
    PetscReal x0       = ctx->cell_x0[cx_local];

    PetscReal       xi       = (x - x0) / (h_x * 0.5) - 1.0;
    const PetscReal xi_pt[1] = {xi};
    PetscCall(PetscFECreateTabulation(feX, 1, 1, xi_pt, 0, &T));

    PetscScalar val    = 0.0;
    PetscInt    offset = iv * ctx->NxDOF_local + cx_local * NbX;
    for (PetscInt b = 0; b < NbX; ++b) { val += f_arr_in[offset + b] * T->T[0][b]; }
    fa[p] = val;
    PetscCall(PetscTabulationDestroy(&T));
  }

  PetscCall(VecRestoreArrayRead(f, &f_arr_in));

  /* 4. Reset Coordinates and Migrate Back */
  PetscCall(DMSwarmGetField(ctx->swX, "x_init", NULL, NULL, (void **)&x_inits));
  for (p = 0; p < np; ++p) { coords[p * dim] = x_inits[p]; }
  PetscCall(DMSwarmRestoreField(ctx->swX, "x_init", NULL, NULL, (void **)&x_inits));

  PetscCall(DMSwarmRestoreField(ctx->swX, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(ctx->swX, "iv_dof", NULL, NULL, (void **)&iv_dofs));
  PetscCall(DMSwarmRestoreField(ctx->swX, "f_val", NULL, NULL, (void **)&fa));

  PetscCall(DMSwarmMigrate(ctx->swX, PETSC_TRUE));

  /* 5. Assemble RHS & Solve (Locally) */
  PetscCall(DMSwarmGetLocalSize(ctx->swX, &np));
  PetscCall(DMSwarmGetField(ctx->swX, "f_val", NULL, NULL, (void **)&fa));
  PetscCall(DMSwarmGetField(ctx->swX, "quad_weight", NULL, NULL, (void **)&weights));
  PetscCall(DMSwarmGetField(ctx->swX, "ix_src", NULL, NULL, (void **)&ix_srcs));
  PetscCall(DMSwarmGetField(ctx->swX, "iv_dof", NULL, NULL, (void **)&iv_dofs));
  PetscCall(DMSwarmGetField(ctx->swX, "x_init", NULL, NULL, (void **)&x_inits));

  /* We need tabX for the quadrature points (source basis) */
  PetscTabulation tabX;
  PetscCall(PetscFECreateTabulation(feX, 1, NqX, xq, 0, &tabX));

  /* Use f_out as temporary storage? No, we need row-by-row solve.
     We can solve directly into f_out if we form RHS row-by-row.
     Let's use f_out for storage of the result.
     We need an array for RHS. Since we iterate particles, we can't easily form one row.
     Particles are mixed.
     Memory efficient: Accumulate ALL RHS into a big array (size of f).
     Then iterate rows and solve.
  */
  PetscScalar *rhs_all;
  PetscCall(PetscCalloc1(ctx->NvDOF * ctx->NxDOF_local, &rhs_all));

  /* Accumulate RHS */
  for (p = 0; p < np; ++p) {
    PetscInt    cx   = ix_srcs[p]; /* This is just cell index now */
    PetscInt    iv   = iv_dofs[p];
    PetscReal   w    = weights[p];
    PetscScalar fval = fa[p];

    /* We need to know WHICH quad point this particle came from?
        Ah, we stored `ix_src = cx`. We lost `q`!
        BUT Migration scrambles order.
        We NEED to store `q` or `x_init` to identify basis.
        We have `x_init`!
        We can infer `xi` from `x_init`.
     */

    /* Recover xi from x_init using cell_x0 (physical cell index stored in ix_srcs) */
    PetscInt  cx_local_src = ctx->phys_to_local[cx];
    PetscReal x0           = ctx->cell_x0[cx_local_src];

    PetscReal xi_recon = (x_inits[p] - x0) / (h_x * 0.5) - 1.0;

    /* Tabulate at xi_recon */
    /* Optimization: This is expensive per particle. Storing q would be better.
        But for now, correctness.
     */
    PetscTabulation T_src;
    const PetscReal xi_pt[1] = {xi_recon};
    PetscCall(PetscFECreateTabulation(feX, 1, 1, xi_pt, 0, &T_src));

    /* Accumulate to RHS for this cell (cx_local_src is the local cell index) */
    PetscInt offset = iv * ctx->NxDOF_local + cx_local_src * NbX;
    for (PetscInt b = 0; b < NbX; ++b) { rhs_all[offset + b] += fval * T_src->T[0][b] * w; }
    PetscCall(PetscTabulationDestroy(&T_src));
  }

  PetscCall(DMSwarmRestoreField(ctx->swX, "f_val", NULL, NULL, (void **)&fa));
  PetscCall(DMSwarmRestoreField(ctx->swX, "quad_weight", NULL, NULL, (void **)&weights));
  PetscCall(DMSwarmRestoreField(ctx->swX, "ix_src", NULL, NULL, (void **)&ix_srcs));
  PetscCall(DMSwarmRestoreField(ctx->swX, "iv_dof", NULL, NULL, (void **)&iv_dofs));
  PetscCall(DMSwarmRestoreField(ctx->swX, "x_init", NULL, NULL, (void **)&x_inits));
  PetscCall(PetscTabulationDestroy(&tabX));

  /* 6. Solve Mass Matrix Per Row */
  PetscCall(VecGetArray(f_out, &f_out_arr));

  Vec f_rhs_row, f_sol_row;
  PetscCall(VecDuplicate(ctx->f_xwork, &f_rhs_row));
  PetscCall(VecDuplicate(ctx->f_xwork, &f_sol_row));

  for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
    /* Copy RHS row to vector */
    PetscScalar *rhs_ptr;
    PetscCall(VecGetArray(f_rhs_row, &rhs_ptr));
    PetscInt offset = iv * ctx->NxDOF_local;
    for (PetscInt i = 0; i < ctx->NxDOF_local; ++i) { rhs_ptr[i] = rhs_all[offset + i]; }
    PetscCall(VecRestoreArray(f_rhs_row, &rhs_ptr));

    /* Solve */
    PetscCall(KSPSolve(ctx->kspMassX, f_rhs_row, f_sol_row));

    /* Copy solution to f_out */
    const PetscScalar *sol_ptr;
    PetscCall(VecGetArrayRead(f_sol_row, &sol_ptr));
    for (PetscInt i = 0; i < ctx->NxDOF_local; ++i) { f_out_arr[offset + i] = sol_ptr[i]; }
    PetscCall(VecRestoreArrayRead(f_sol_row, &sol_ptr));
  }

  PetscCall(VecDestroy(&f_rhs_row));
  PetscCall(VecDestroy(&f_sol_row));
  PetscCall(VecRestoreArray(f_out, &f_out_arr));
  PetscCall(PetscFree(rhs_all));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdvectV(Vec f, Vec f_out, PetscReal s, Vec E_field, AppCtx *ctx)
{
  PetscInt           p, np, cStartV, cEndV;
  PetscScalar       *fa;
  PetscReal         *coords, *E_vals, *weights;
  PetscInt          *ix_dofs;
  PetscInt           dim = 1;
  PetscScalar       *f_out_arr;
  const PetscScalar *f_in_arr;

  PetscFunctionBeginUser;

  /* 1. Evaluate E-field at each x-DOF physical coordinate */
  PetscFE      feX, fePot;
  PetscInt     NbX, NbPot;
  PetscSection sectionPot;
  Vec          E_local;
  PetscReal    h_x = ctx->x_max / ctx->Nx;
  PetscReal   *E_at_xdof;

  PetscCall(DMGetField(ctx->dmX, 0, NULL, (PetscObject *)&feX));
  PetscCall(PetscFEGetDimension(feX, &NbX));
  PetscCall(DMGetField(ctx->dmPot, 0, NULL, (PetscObject *)&fePot));
  PetscCall(PetscFEGetDimension(fePot, &NbPot));
  PetscCall(DMGetLocalSection(ctx->dmPot, &sectionPot));
  PetscCall(DMGetLocalVector(ctx->dmPot, &E_local));
  PetscCall(DMGlobalToLocal(ctx->dmPot, E_field, INSERT_VALUES, E_local));

  PetscCall(PetscMalloc1(ctx->NxDOF_local, &E_at_xdof));
  {
    PetscInt cStartX, cEndX;
    PetscCall(DMPlexGetHeightStratum(ctx->dmX, 0, &cStartX, &cEndX));
    for (PetscInt cx = cStartX; cx < cEndX; ++cx) {
      PetscInt  cx_local = cx - cStartX;
      PetscReal x0       = ctx->cell_x0[cx_local];
      /* Get E closure on dmPot cell cx */
      PetscScalar *E_coefs = NULL;
      PetscCall(DMPlexVecGetClosure(ctx->dmPot, sectionPot, E_local, cx, NULL, &E_coefs));
      for (PetscInt bx = 0; bx < NbX; ++bx) {
        /* Physical coordinate of this x-DOF */
        PetscReal xi_ref = (NbX == 1) ? 0.0 : (bx == 0 ? -1.0 : 1.0);
        PetscReal x_phys = x0 + (xi_ref + 1.0) * h_x * 0.5;
        (void)x_phys; /* used for documentation; evaluate E via basis */
        /* Evaluate E at xi_ref using fePot basis */
        const PetscReal xi_pt[1] = {xi_ref};
        PetscTabulation T;
        PetscCall(PetscFECreateTabulation(fePot, 1, 1, xi_pt, 0, &T));
        PetscScalar E_val = 0.0;
        for (PetscInt b = 0; b < NbPot; ++b) E_val += E_coefs[b] * T->T[0][b];
        E_at_xdof[cx_local * NbX + bx] = PetscRealPart(E_val);
        PetscCall(PetscTabulationDestroy(&T));
      }
      PetscCall(DMPlexVecRestoreClosure(ctx->dmPot, sectionPot, E_local, cx, NULL, &E_coefs));
    }
  }
  PetscCall(DMRestoreLocalVector(ctx->dmPot, &E_local));

  /* 2. Populate swV */
  PetscFE          feV;
  PetscQuadrature  quadV;
  PetscInt         NqV, NbV;
  const PetscReal *vq, *wq;

  PetscCall(DMGetField(ctx->dmV, 0, NULL, (PetscObject *)&feV));
  PetscCall(PetscFEGetQuadrature(feV, &quadV));
  PetscCall(PetscQuadratureGetData(quadV, NULL, NULL, &NqV, &vq, &wq));
  PetscCall(PetscFEGetDimension(feV, &NbV));
  PetscCall(DMPlexGetHeightStratum(ctx->dmV, 0, &cStartV, &cEndV));

  PetscInt Nv_local_cells = cEndV - cStartV;
  PetscInt n_particles    = ctx->NxDOF_local * Nv_local_cells * NqV;

  PetscCall(DMSwarmSetLocalSizes(ctx->swV, n_particles, -1));

  PetscCall(DMSwarmGetField(ctx->swV, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(ctx->swV, "E_value", NULL, NULL, (void **)&E_vals));
  PetscCall(DMSwarmGetField(ctx->swV, "quad_weight", NULL, NULL, (void **)&weights));
  PetscCall(DMSwarmGetField(ctx->swV, "ixdof_local", NULL, NULL, (void **)&ix_dofs));

  PetscInt  p_idx = 0;
  PetscReal h_v   = 2.0 * ctx->v_max / ctx->Nv;

  /* Loop over x-DOFs */
  for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix) {
    PetscReal E = E_at_xdof[ix];

    /* Loop over v-cells */
    for (PetscInt cv = cStartV; cv < cEndV; ++cv) {
      PetscReal v_center = -ctx->v_max + (cv + 0.5) * h_v;
      PetscReal detJ     = h_v * 0.5;

      for (PetscInt q = 0; q < NqV; ++q) {
        PetscReal xi     = vq[q];
        PetscReal v_phys = v_center + xi * (h_v * 0.5);

        /* Trace */
        PetscReal v_foot = v_phys - E * s;

        /* Store particle */
        coords[p_idx * dim] = v_foot;
        E_vals[p_idx]       = E;
        weights[p_idx]      = wq[q] * detJ;
        ix_dofs[p_idx]      = ix;
        p_idx++;
      }
    }
  }

  PetscCall(PetscFree(E_at_xdof));

  PetscCall(DMSwarmRestoreField(ctx->swV, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(ctx->swV, "E_value", NULL, NULL, (void **)&E_vals));
  PetscCall(DMSwarmRestoreField(ctx->swV, "quad_weight", NULL, NULL, (void **)&weights));
  PetscCall(DMSwarmRestoreField(ctx->swV, "ixdof_local", NULL, NULL, (void **)&ix_dofs));

  /* 3. Interpolate */
  PetscCall(DMSwarmGetLocalSize(ctx->swV, &np));
  PetscCall(DMSwarmGetField(ctx->swV, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(ctx->swV, "ixdof_local", NULL, NULL, (void **)&ix_dofs));
  PetscCall(DMSwarmGetField(ctx->swV, "f_val", NULL, NULL, (void **)&fa));

  PetscCall(VecGetArrayRead(f, &f_in_arr));

  /* Optimization: Batch tabulation */
  /* We compute xi for all particles first */
  PetscReal *xi_batch;
  PetscInt  *cv_batch;
  PetscCall(PetscMalloc1(np, &xi_batch));
  PetscCall(PetscMalloc1(np, &cv_batch));

  for (p = 0; p < np; ++p) {
    PetscReal v = coords[p * dim];

    /* Locate cell (uniform mesh) */
    PetscReal v_normalized = (v + ctx->v_max) / h_v;
    PetscInt  cv           = (PetscInt)PetscFloorReal(v_normalized);
    if (cv >= ctx->Nv) cv = ctx->Nv - 1;
    if (cv < 0) cv = 0;

    PetscReal xi = (v - (-ctx->v_max + (cv + 0.5) * h_v)) / (h_v * 0.5);
    if (xi < -1.0) xi = -1.0;
    if (xi > 1.0) xi = 1.0;

    xi_batch[p] = xi;
    cv_batch[p] = cv;
  }

  PetscTabulation T;
  PetscCall(PetscFECreateTabulation(feV, 1, np, xi_batch, 0, &T));

  for (p = 0; p < np; ++p) {
    PetscReal v  = coords[p * dim];
    PetscInt  ix = ix_dofs[p];
    PetscInt  cv = cv_batch[p];

    /* Check physical bounds */
    if (PetscAbsReal(v) > ctx->v_max) {
      fa[p] = 0.0;
    } else {
      PetscScalar val = 0.0;
      for (PetscInt b = 0; b < NbV; ++b) {
        PetscInt v_dof = ctx->v_cell_dofs[cv * NbV + b]; /* actual local DOF index */
        PetscInt idx   = v_dof * ctx->NxDOF_local + ix;
        val += f_in_arr[idx] * T->T[0][p * NbV + b];
      }
      fa[p] = val;
    }
  }

  PetscCall(PetscTabulationDestroy(&T));
  PetscCall(PetscFree(xi_batch));
  PetscCall(PetscFree(cv_batch));
  PetscCall(VecRestoreArrayRead(f, &f_in_arr));

  /* 4. Assemble RHS */
  PetscScalar *rhs_all;
  PetscCall(PetscCalloc1(ctx->NvDOF * ctx->NxDOF_local, &rhs_all));

  PetscCall(DMSwarmGetField(ctx->swV, "quad_weight", NULL, NULL, (void **)&weights));

  /* Use quadrature tabulation for test functions */
  PetscTabulation tabV;
  PetscCall(PetscFECreateTabulation(feV, 1, NqV, vq, 0, &tabV));

  p_idx = 0;
  for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix) {
    for (PetscInt cv = cStartV; cv < cEndV; ++cv) {
      for (PetscInt q = 0; q < NqV; ++q) {
        PetscReal   w    = weights[p_idx];
        PetscScalar fval = fa[p_idx];

        for (PetscInt b = 0; b < NbV; ++b) {
          PetscInt v_dof = ctx->v_cell_dofs[(cv - cStartV) * NbV + b]; /* actual local DOF index */
          PetscInt idx   = v_dof * ctx->NxDOF_local + ix;
          rhs_all[idx] += fval * tabV->T[0][q * NbV + b] * w;
        }
        p_idx++;
      }
    }
  }

  PetscCall(PetscTabulationDestroy(&tabV));

  PetscCall(DMSwarmRestoreField(ctx->swV, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(ctx->swV, "ixdof_local", NULL, NULL, (void **)&ix_dofs));
  PetscCall(DMSwarmRestoreField(ctx->swV, "f_val", NULL, NULL, (void **)&fa));
  PetscCall(DMSwarmRestoreField(ctx->swV, "quad_weight", NULL, NULL, (void **)&weights));

  /* 5. Solve */
  Vec f_rhs_col, f_sol_col;
  PetscCall(VecDuplicate(ctx->f_vwork, &f_rhs_col));
  PetscCall(VecDuplicate(ctx->f_vwork, &f_sol_col));

  PetscCall(VecGetArray(f_out, &f_out_arr));

  for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix) {
    PetscScalar *rhs_ptr;
    PetscCall(VecGetArray(f_rhs_col, &rhs_ptr));
    for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) { rhs_ptr[iv] = rhs_all[iv * ctx->NxDOF_local + ix]; }
    PetscCall(VecRestoreArray(f_rhs_col, &rhs_ptr));

    PetscCall(KSPSolve(ctx->kspMassV, f_rhs_col, f_sol_col));

    const PetscScalar *sol_ptr;
    PetscCall(VecGetArrayRead(f_sol_col, &sol_ptr));
    for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) { f_out_arr[iv * ctx->NxDOF_local + ix] = sol_ptr[iv]; }
    PetscCall(VecRestoreArrayRead(f_sol_col, &sol_ptr));
  }

  PetscCall(VecRestoreArray(f_out, &f_out_arr));
  PetscCall(VecDestroy(&f_rhs_col));
  PetscCall(VecDestroy(&f_sol_col));
  PetscCall(PetscFree(rhs_all));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   ComputeElectricField  (Task T10)
   Computes E = -grad(phi) by projecting -grad(phi) onto E_field basis.
   Solves M * E = ( -grad(phi), psi )
   ======================================================================== */
static PetscErrorCode ComputeElectricField(AppCtx *ctx)
{
  Vec              rhs_E, rhs_local;
  PetscInt         cStart, cEnd, Nb, Nq;
  PetscFE          fe;
  PetscQuadrature  quad;
  const PetscReal *wq, *xiq;
  PetscTabulation  tab;
  PetscSection     section;

  PetscFunctionBeginUser;
  PetscCall(DMGetGlobalVector(ctx->dmPot, &rhs_E));
  PetscCall(VecZeroEntries(rhs_E));
  PetscCall(DMGetLocalVector(ctx->dmPot, &rhs_local));
  PetscCall(VecZeroEntries(rhs_local));

  PetscCall(DMPlexGetHeightStratum(ctx->dmPot, 0, &cStart, &cEnd));
  PetscCall(DMGetField(ctx->dmPot, 0, NULL, (PetscObject *)&fe));
  PetscCall(PetscFEGetDimension(fe, &Nb));
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, &xiq, &wq));
  PetscCall(PetscFECreateTabulation(fe, 1, Nq, xiq, 1, &tab)); /* Need gradient (order 1) */
  PetscCall(DMGetLocalSection(ctx->dmPot, &section));

  Vec phi_local;
  PetscCall(DMGetLocalVector(ctx->dmPot, &phi_local));
  PetscCall(DMGlobalToLocal(ctx->dmPot, ctx->phi, INSERT_VALUES, phi_local));

  PetscReal h_x  = ctx->x_max / ctx->Nx;
  PetscReal detJ = h_x * 0.5;
  PetscReal invJ = 2.0 / h_x;

  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscScalar *phi_coefs = NULL;
    PetscCall(DMPlexVecGetClosure(ctx->dmPot, section, phi_local, c, NULL, &phi_coefs));

    PetscScalar *rhs_elem;
    PetscCall(PetscMalloc1(Nb, &rhs_elem));
    for (int i = 0; i < Nb; i++) rhs_elem[i] = 0.0;

    for (PetscInt q = 0; q < Nq; ++q) {
      /* Compute grad(phi) at q */
      /* phi(x) = sum phi_b * N_b(xi) */
      /* dphi/dx = sum phi_b * (dN_b/dxi * dxi/dx) */
      /* dxi/dx = invJ */
      PetscScalar dphi_dx = 0.0;
      for (PetscInt b = 0; b < Nb; ++b) { dphi_dx += phi_coefs[b] * tab->T[1][q * Nb + b] * invJ; }

      /* E = -dphi/dx */
      PetscScalar E_val = -dphi_dx;

      /* Test against psi_i */
      for (PetscInt i = 0; i < Nb; ++i) { rhs_elem[i] += E_val * tab->T[0][q * Nb + i] * wq[q] * detJ; }
    }

    PetscCall(DMPlexVecSetClosure(ctx->dmPot, section, rhs_local, c, rhs_elem, ADD_VALUES));
    PetscCall(PetscFree(rhs_elem));
    PetscCall(DMPlexVecRestoreClosure(ctx->dmPot, section, phi_local, c, NULL, &phi_coefs));
  }

  PetscCall(DMRestoreLocalVector(ctx->dmPot, &phi_local));
  PetscCall(PetscTabulationDestroy(&tab));

  PetscCall(DMLocalToGlobal(ctx->dmPot, rhs_local, ADD_VALUES, rhs_E));
  PetscCall(DMRestoreLocalVector(ctx->dmPot, &rhs_local));

  /* Solve M * E = rhs_E */
  PetscCall(KSPSolve(ctx->kspMassPot, rhs_E, ctx->E_field));

  PetscCall(DMRestoreGlobalVector(ctx->dmPot, &rhs_E));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   ComputeMoments
   Computes phase-space moments of f:
     m[0] = integral f dv dx          (total mass / charge density integral)
     m[1] = integral v*f dv dx        (momentum)
     m[2] = integral v^2*f dv dx      (energy, kinetic)
   Uses the same v-basis-integral pattern as ComputeChargeDensity but with
   v-weighted integrals.  The x-integral is just a sum over x-DOFs times h_x/NbX
   (uniform mesh: each DG DOF covers h_x/NbX of physical length).
   ======================================================================== */
static PetscErrorCode ComputeMoments(AppCtx *ctx, PetscReal *m0_out, PetscReal *m1_out, PetscReal *m2_out)
{
  PetscInt         cStartV, cEndV, NbV, NqV;
  PetscFE          feV;
  PetscQuadrature  quadV;
  const PetscReal *wqV, *xiqV;
  PetscTabulation  tabV;
  PetscReal        h_v   = 2.0 * ctx->v_max / ctx->Nv;
  PetscReal        detJV = h_v * 0.5;
  PetscReal        h_x   = ctx->x_max / ctx->Nx;
  PetscReal       *v_w0, *v_w1, *v_w2; /* per-DOF integrals of phi_b, v*phi_b, v^2*phi_b */
  PetscScalar     *f_array;
  PetscReal        m0_local = 0.0, m1_local = 0.0, m2_local = 0.0;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetHeightStratum(ctx->dmV, 0, &cStartV, &cEndV));
  PetscCall(DMGetField(ctx->dmV, 0, NULL, (PetscObject *)&feV));
  PetscCall(PetscFEGetDimension(feV, &NbV));
  PetscCall(PetscFEGetQuadrature(feV, &quadV));
  PetscCall(PetscQuadratureGetData(quadV, NULL, NULL, &NqV, &xiqV, &wqV));
  PetscCall(PetscFECreateTabulation(feV, 1, NqV, xiqV, 0, &tabV));

  PetscCall(PetscCalloc1(ctx->NvDOF, &v_w0));
  PetscCall(PetscCalloc1(ctx->NvDOF, &v_w1));
  PetscCall(PetscCalloc1(ctx->NvDOF, &v_w2));

  /* Build per-DOF integrals: integral phi_b dv, v*phi_b dv, v^2*phi_b dv */
  for (PetscInt cv = cStartV; cv < cEndV; ++cv) {
    PetscReal v_cell_center = -ctx->v_max + (cv - cStartV + 0.5) * h_v;
    for (PetscInt bv = 0; bv < NbV; ++bv) {
      PetscReal s0 = 0.0, s1 = 0.0, s2 = 0.0;
      for (PetscInt q = 0; q < NqV; ++q) {
        /* physical v at this quadrature point */
        PetscReal v_phys  = v_cell_center + xiqV[q] * detJV;
        PetscReal phi_bq  = tabV->T[0][q * NbV + bv];
        PetscReal contrib = phi_bq * wqV[q] * detJV;
        s0 += contrib;
        s1 += v_phys * contrib;
        s2 += v_phys * v_phys * contrib;
      }
      PetscInt iv_dof = ctx->v_cell_dofs[(cv - cStartV) * NbV + bv];
      if (iv_dof >= 0) {
        v_w0[iv_dof] += s0;
        v_w1[iv_dof] += s1;
        v_w2[iv_dof] += s2;
      }
    }
  }
  PetscCall(PetscTabulationDestroy(&tabV));

  /* x-integral: each DG x-DOF represents a physical length of h_x / NbX
     (uniform mesh, NbX DOFs per cell of width h_x).
     We use the DG DOF value directly as the cell-average approximation. */
  {
    PetscFE  feX;
    PetscInt NbX;
    PetscCall(DMGetField(ctx->dmX, 0, NULL, (PetscObject *)&feX));
    PetscCall(PetscFEGetDimension(feX, &NbX));
    PetscReal dx_per_dof = h_x / NbX;

    PetscCall(VecGetArray(ctx->f, &f_array));
    for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
      PetscReal w0 = v_w0[iv], w1 = v_w1[iv], w2 = v_w2[iv];
      PetscInt  offset = iv * ctx->NxDOF_local;
      for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix) {
        PetscReal fval = PetscRealPart(f_array[offset + ix]);
        m0_local += fval * w0 * dx_per_dof;
        m1_local += fval * w1 * dx_per_dof;
        m2_local += fval * w2 * dx_per_dof;
      }
    }
    PetscCall(VecRestoreArray(ctx->f, &f_array));
  }

  PetscCall(PetscFree(v_w0));
  PetscCall(PetscFree(v_w1));
  PetscCall(PetscFree(v_w2));

  /* Sum across MPI ranks (each rank owns a subset of x-DOFs) */
  PetscReal moments_local[3] = {m0_local, m1_local, m2_local};
  PetscReal moments_global[3];
  PetscCallMPI(MPIU_Allreduce(moments_local, moments_global, 3, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)ctx->f)));
  *m0_out = moments_global[0];
  *m1_out = moments_global[1];
  *m2_out = moments_global[2];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   BSLStep_Strang  (Task TR8)
   Step 1: AdvectX(dt/2)
   Step 2: E-field update (rho, Poisson, E)
   Step 3: AdvectV(dt, E)
   Step 4: AdvectX(dt/2)
   ======================================================================== */
static PetscErrorCode BSLStep_Strang(AppCtx *ctx, PetscReal dt)
{
  PetscFunctionBeginUser;

  /* Step 1: AdvectX(dt/2) */
  PetscCall(AdvectX(ctx->f, ctx->f, dt * 0.5, ctx));

  /* Step 2: E-field update */
  PetscCall(ComputeChargeDensity(ctx));
  PetscCall(SolvePoisson(ctx->dmPot, ctx->rho, ctx->phi, ctx));
  PetscCall(ComputeElectricField(ctx));

  /* Step 3: AdvectV(dt) */
  PetscCall(AdvectV(ctx->f, ctx->f, dt, ctx->E_field, ctx));

  /* Step 4: AdvectX(dt/2) */
  PetscCall(AdvectX(ctx->f, ctx->f, dt * 0.5, ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyContext(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFEGeomDestroy(&ctx->fegeomX));
  PetscCall(PetscFEGeomDestroy(&ctx->fegeomV));
  PetscCall(PetscFEGeomDestroy(&ctx->fegeomPot));
  PetscCall(PetscDrawLGDestroy(&ctx->drawlgE));
  PetscCall(VecDestroy(&ctx->f));
  PetscCall(VecDestroy(&ctx->f_xwork));
  PetscCall(VecDestroy(&ctx->f_vwork));
  PetscCall(VecDestroy(&ctx->rho));
  PetscCall(VecDestroy(&ctx->phi));
  PetscCall(VecDestroy(&ctx->E_field));
  PetscCall(MatDestroy(&ctx->MassPot));
  PetscCall(KSPDestroy(&ctx->kspMassPot));
  PetscCall(KSPDestroy(&ctx->kspMassX));
  PetscCall(KSPDestroy(&ctx->kspMassV));
  PetscCall(SNESDestroy(&ctx->snesPoisson));
  PetscCall(DMDestroy(&ctx->swX));
  PetscCall(DMDestroy(&ctx->swV));
  PetscCall(DMDestroy(&ctx->dmX));
  PetscCall(DMDestroy(&ctx->dmV));
  PetscCall(DMDestroy(&ctx->dmPot));
  PetscCall(DMDestroy(&ctx->dmScalar));
  PetscCall(PetscFree(ctx->v_dof_coords));
  PetscCall(PetscFree(ctx->v_cell_dofs));
  PetscCall(PetscFree(ctx->phys_to_local));
  PetscCall(PetscFree(ctx->cell_x0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx ctx;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateXMesh(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateVMesh(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreatePotentialMeshAndPoisson(PETSC_COMM_WORLD, &ctx));
  PetscCall(AllocateF(&ctx));
  PetscCall(InitializeF(&ctx));
  PetscCall(SetupXSwarm(&ctx));
  PetscCall(SetupVSwarm(&ctx));

  /* Initial Diagnostics */
  PetscCall(ComputeChargeDensity(&ctx));
  {
    PetscReal rho_sum;
    PetscCall(VecSum(ctx.rho, &rho_sum));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial rho sum = %g\n", (double)rho_sum));
  }
  PetscCall(SolvePoisson(ctx.dmPot, ctx.rho, ctx.phi, &ctx));
  PetscCall(ComputeElectricField(&ctx));

  /* Main Loop */
  PetscReal *E_history = NULL;
  if (ctx.check_landau) { PetscCall(PetscMalloc1(ctx.steps, &E_history)); }

  for (PetscInt step = 0; step < ctx.steps; ++step) {
    PetscCall(BSLStep_Strang(&ctx, ctx.dt));

    if (step % ctx.ostep == 0 || ctx.check_landau) {
      PetscReal E_max, E_norm, E_sum;
      PetscCall(VecNorm(ctx.E_field, NORM_INFINITY, &E_max));

      if (step % ctx.ostep == 0) {
        PetscReal t      = (step + 1) * ctx.dt;
        PetscReal lgEmax = E_max > 0 ? PetscLog10Real(E_max) : -16.0;
        PetscReal m0, m1, m2;
        PetscCall(VecNorm(ctx.E_field, NORM_2, &E_norm));
        PetscCall(VecSum(ctx.E_field, &E_sum));
        PetscReal lgEnorm = E_norm > 0 ? PetscLog10Real(E_norm) : -16.0;
        PetscCall(ComputeMoments(&ctx, &m0, &m1, &m2));
        /* Match ex4.c column layout:
              col1=t  col2=Esum  col3=Enorm  col4=lgEnorm  col5=Emax  col6=lgEmax
              col7=chargesum(=m0)  col8=m0  col9=m1  col10=m2  col11=(step) */
        if (step == 0)
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time            Sum E           |E|              log(|E|) "
                                                  "      E_max                   log(E_max)      sum(q)   "
                                                  "part:    moment-0                moment-1                moment-2 "
                                                  "(V^2)       #step\n"));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "E: %f\t%+e\t%e\t%f\t%20.15e\t%f\t%f\t%20.15e\t%20.15e\t%20.15e\t(%" PetscInt_FMT ")\n", (double)t, (double)E_sum, (double)E_norm, (double)lgEnorm, (double)E_max, (double)lgEmax, (double)m0, (double)m0, (double)m1, (double)m2, step));
      }
      if (ctx.check_landau) { E_history[step] = E_max; }
    }
  }

  /* Landau Damping Rate Check */
  if (ctx.check_landau) {
    /* Use local maxima (envelope peaks) for regression on log(E) vs t.
        The E-field oscillates at the plasma frequency while the envelope decays,
        so using all points (including near-zero troughs) corrupts the regression. */
    PetscReal sum_t = 0, sum_logE = 0, sum_t_logE = 0, sum_t2 = 0;
    PetscInt  n_points = 0;

    /* Skip first few steps to avoid initial transient, then use all peaks */
    PetscInt start_step = 2;

    for (PetscInt s = start_step + 1; s < ctx.steps - 1; ++s) {
      PetscReal val = E_history[s];
      /* Local maximum: larger than both neighbours */
      if (val > E_history[s - 1] && val > E_history[s + 1] && val > 1e-16) {
        PetscReal t    = (s + 1) * ctx.dt;
        PetscReal logE = PetscLogReal(val);
        sum_t += t;
        sum_logE += logE;
        sum_t_logE += t * logE;
        sum_t2 += t * t;
        n_points++;
      }
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nLandau Damping Check:\n"));
    if (n_points > 1) {
      PetscReal gamma    = (n_points * sum_t_logE - sum_t * sum_logE) / (n_points * sum_t2 - sum_t * sum_t);
      ctx.gamma_measured = gamma;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Envelope peaks used: %d\n", (int)n_points));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Measured gamma = %g\n", (double)gamma));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Theoretical gamma (k=0.5) ~ -0.1533\n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Not enough envelope peaks (%d) to fit gamma\n", (int)n_points));
    }
    PetscCall(PetscFree(E_history));
  }

  PetscCall(DestroyContext(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex double

  # dmPot is cloned from dmX so -x_dm_plex_box_* options are no longer needed.
  # Only -x_petscspace_degree is required to set the CG FE degree for the potential.
  testset:
    args: -fx_dm_plex_dim 1 -fv_dm_plex_dim 1 \
          -fx_dm_plex_box_faces 32 -fv_dm_plex_box_faces 64 -fx_dm_plex_box_bd periodic -fv_dm_plex_box_bd none \
          -fx_dm_plex_box_upper 12.5664 -fv_dm_plex_box_lower -6.0 -fv_dm_plex_box_upper 6.0 \
          -fx_petscspace_degree 1 -fv_petscspace_degree 1 \
          -x_petscspace_degree 1 \
          -em_snes_type ksponly -em_ksp_type cg -em_pc_type gamg \
          -em_mg_coarse_ksp_type preonly -em_mg_coarse_pc_type svd

    test:
      suffix: 0
      args: -steps 10 -output_step 1

    test:
      suffix: mpi
      nsize: 2
      args: -steps 10 -output_step 1

    test:
      suffix: landau
      args: -steps 200 -output_step 10 -check_landau

TEST*/
