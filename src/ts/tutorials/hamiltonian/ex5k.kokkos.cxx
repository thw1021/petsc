static char help[] = "SLDG Vlasov-Poisson solver for Landau damping using DMDA and Kokkos-ready flat arrays.\n"
                     "Lukas Einkemmer, High performance computing aspects of a dimension independent\n"
                     "semi-Lagrangian discontinuous Galerkin code, Computer Physics Communications,\n"
                     "Volume 202, 2016.\n\n";

/*
  Phases 0-5: Scaffold, AppCtx, DMDA meshes, Poisson KSP, geometry precomputation,
               distribution function allocation and initialization.
  Phases 6+:  x-advection (SLDG), v-advection (DG), time-stepping (to be added).
*/

/* petscvec_kokkos.hpp must be included FIRST before any other PETSc header
   when using Kokkos, to avoid the petsccomplexlib redefinition error. */
#include <petscvec_kokkos.hpp>
#include <petscconf.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscmath.h>
#include <petscdt.h>

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  #include <Kokkos_Core.hpp>
  #include <petsc_kokkos.hpp>
  #include <petscdmda_kokkos.hpp>
#endif

/*
  TODO: When enabling Kokkos GPU offload, change DefaultMemorySpace to
        Kokkos::DefaultExecutionSpace::memory_space and uncomment View types.
  using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using ScalarView1D = Kokkos::View<PetscScalar *, DefaultMemorySpace>;
  using ScalarView2D = Kokkos::View<PetscScalar **, Kokkos::LayoutRight, DefaultMemorySpace>;
  using IntView1D    = Kokkos::View<PetscInt *,    DefaultMemorySpace>;
*/

/* ========================================================================
   AppCtx — all simulation state
   ======================================================================== */
typedef struct {
  /* Grid DMs */
  DM daX;   /* 1D periodic DMDA for x-space, MPI-distributed */
  DM daV;   /* 1D non-periodic DMDA for v-space, PETSC_COMM_SELF */
  DM daPot; /* 1D periodic DMDA for phi/rho/E, clone of daX with dof=1 */
  DM daF;   /* 1D periodic DMDA for one iv-slice of f: dof=NbX, same stencil as daX */

  /* Distribution function */
  Vec f; /* Global Vec, local size NvDOF * NxDOF_local */
         /* Layout: f_arr[iv_dof * NxDOF_local + ix_dof] */
         /* iv_dof = cv * NbV + bv,  ix_dof = cx_local * NbX + bx */

  /* Potential / field */
  Vec rho;
  Vec phi;
  Vec E_field;
  KSP kspPoisson;
  Mat Jac;

  /* Precomputed geometry (host arrays, Kokkos-ready) */
  PetscReal *xi_x_nodes;   /* [NbX] GLL reference coords of x-DOF nodes */
  PetscReal *xi_v_nodes;   /* [NbV] GLL reference coords of v-DOF nodes */
  PetscReal *v_dof_coords; /* [NvDOF] physical v-coord of each v-DOF */
  PetscReal *v_basis_int;  /* [NvDOF] integral of v-basis over its support */
  PetscReal *x_basis_int;  /* [NbX] integral of x-basis: gll_wts_x[bx] * h_x/2 */

  /* SLDG matrices for x-advection (precomputed per velocity DOF) */
  PetscReal *A_sldg; /* [NvDOF * NbX * NbX] same-cell overlap */
  PetscReal *B_sldg; /* [NvDOF * NbX * NbX] neighbor-cell overlap */
  PetscInt  *n_shift; /* [NvDOF] integer cell shift per v-DOF */

  /* Quadrature for v-advection */
  PetscReal *vq_pts;    /* [NqV] Gauss quadrature points in [-1,1] */
  PetscReal *vq_wts;    /* [NqV] Gauss quadrature weights */
  PetscReal *tabV;      /* [NqV * NbV] v-basis tabulation at vq_pts */
  PetscReal *MassV_inv; /* [NvDOF] diagonal mass inverse */

  /* Sizes */
  PetscInt NxLocal;      /* local number of x-cells */
  PetscInt NxDOF_local;  /* local x-DOFs = NxLocal * NbX */
  PetscInt NvDOF;        /* total v-DOFs = Nv * NbV */
  PetscInt xs;           /* global x-cell start index for this rank */
  PetscInt Nx;           /* global number of x-cells */
  PetscInt Nv;           /* number of v-cells (same on all ranks) */
  PetscInt NbX;          /* basis functions per x-cell (= degree_x + 1) */
  PetscInt NbV;          /* basis functions per v-cell (= degree_v + 1) */
  PetscInt NqV;          /* quadrature points for v-advection */

  /* Physical parameters */
  PetscReal x_max;
  PetscReal v_max;
  PetscReal h_x;
  PetscReal h_v;
  PetscReal alpha;
  PetscReal kwave;
  PetscReal sigma;
  PetscReal dt;
  PetscInt  steps;
  PetscInt  ostep;
  PetscBool check_landau;
  PetscInt  degree_x; /* polynomial degree for x-space DG */
  PetscInt  degree_v; /* polynomial degree for v-space DG */

  PetscLogEvent AdvectXEvent;
  PetscLogEvent AdvectVEvent;
  PetscLogEvent PoissonEvent;
  PetscLogEvent RhoEvent;
} AppCtx;

/* ========================================================================
   Lagrange basis helper
   ======================================================================== */
static inline PetscReal EvalLagrangeBasis(PetscInt nb, const PetscReal *nodes, PetscInt b, PetscReal xi)
{
  PetscReal val = 1.0;
  for (PetscInt j = 0; j < nb; ++j)
    if (j != b) val *= (xi - nodes[j]) / (nodes[b] - nodes[j]);
  return val;
}

/* ========================================================================
   Phase 1 — ProcessOptions
   ======================================================================== */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscFunctionBeginUser;

  /* Null-initialize all pointers */
  ctx->daX        = NULL;
  ctx->daV        = NULL;
  ctx->daPot      = NULL;
  ctx->daF        = NULL;
  ctx->f          = NULL;
  ctx->rho        = NULL;
  ctx->phi        = NULL;
  ctx->E_field    = NULL;
  ctx->kspPoisson = NULL;
  ctx->Jac        = NULL;

  ctx->xi_x_nodes   = NULL;
  ctx->xi_v_nodes   = NULL;
  ctx->v_dof_coords = NULL;
  ctx->v_basis_int  = NULL;
  ctx->x_basis_int  = NULL;
  ctx->A_sldg       = NULL;
  ctx->B_sldg       = NULL;
  ctx->n_shift      = NULL;
  ctx->vq_pts       = NULL;
  ctx->vq_wts       = NULL;
  ctx->tabV         = NULL;
  ctx->MassV_inv    = NULL;

  /* Defaults */
  ctx->Nx          = 64;
  ctx->Nv          = 128;
  ctx->v_max       = 6.0;
  ctx->alpha       = 0.01;
  ctx->kwave       = 0.5;
  ctx->sigma       = 1.0;
  ctx->dt          = 0.1;
  ctx->steps       = 200;
  ctx->ostep       = 10;
  ctx->degree_x    = 1;
  ctx->degree_v    = 1;
  ctx->check_landau = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "SLDG Vlasov-Poisson options", "TS");
  PetscCall(PetscOptionsInt("-Nx", "Number of x cells", __FILE__, ctx->Nx, &ctx->Nx, NULL));
  PetscCall(PetscOptionsInt("-Nv", "Number of v cells", __FILE__, ctx->Nv, &ctx->Nv, NULL));
  PetscCall(PetscOptionsReal("-v_max", "Velocity domain half-width", __FILE__, ctx->v_max, &ctx->v_max, NULL));
  PetscCall(PetscOptionsReal("-alpha", "Perturbation amplitude", __FILE__, ctx->alpha, &ctx->alpha, NULL));
  PetscCall(PetscOptionsReal("-kwave", "Perturbation wave number", __FILE__, ctx->kwave, &ctx->kwave, NULL));
  PetscCall(PetscOptionsReal("-sigma", "Background ion charge density", __FILE__, ctx->sigma, &ctx->sigma, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", __FILE__, ctx->dt, &ctx->dt, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", __FILE__, ctx->steps, &ctx->steps, NULL));
  PetscCall(PetscOptionsInt("-ostep", "Output every N steps", __FILE__, ctx->ostep, &ctx->ostep, NULL));
  PetscCall(PetscOptionsInt("-degree_x", "Polynomial degree for x-space DG", __FILE__, ctx->degree_x, &ctx->degree_x, NULL));
  PetscCall(PetscOptionsInt("-degree_v", "Polynomial degree for v-space DG", __FILE__, ctx->degree_v, &ctx->degree_v, NULL));
  PetscCall(PetscOptionsBool("-check_landau", "Check Landau damping rate at end", __FILE__, ctx->check_landau, &ctx->check_landau, NULL));
  PetscOptionsEnd();

  /* Derived quantities */
  ctx->x_max = 2.0 * PETSC_PI / ctx->kwave;
  ctx->h_x   = ctx->x_max / ctx->Nx;
  ctx->h_v   = 2.0 * ctx->v_max / ctx->Nv;
  ctx->NbX   = ctx->degree_x + 1;
  ctx->NbV   = ctx->degree_v + 1;

  /* Register log events */
  PetscCall(PetscLogEventRegister("AdvectX",      TS_CLASSID, &ctx->AdvectXEvent));
  PetscCall(PetscLogEventRegister("AdvectV",      TS_CLASSID, &ctx->AdvectVEvent));
  PetscCall(PetscLogEventRegister("Poisson",      TS_CLASSID, &ctx->PoissonEvent));
  PetscCall(PetscLogEventRegister("ChargeDensity",TS_CLASSID, &ctx->RhoEvent));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 2 — CreateXMesh + CreateVMesh
   ======================================================================== */
static PetscErrorCode CreateXMesh(MPI_Comm comm, AppCtx *ctx)
{
  PetscFunctionBeginUser;
  /* 1D periodic DMDA for x-space.
     dof = NbX: one DG coefficient per basis function per cell.
     stencil width = ceil(v_max * dt / h_x) + 1 for SLDG ghost cells. */
  PetscInt sw = (PetscInt)(ctx->v_max * ctx->dt / ctx->h_x) + 2;
  PetscCall(DMDACreate1d(comm, DM_BOUNDARY_PERIODIC, ctx->Nx, ctx->NbX, sw, NULL, &ctx->daX));
  PetscCall(DMSetFromOptions(ctx->daX));
  PetscCall(DMSetUp(ctx->daX));
  PetscCall(DMDAGetCorners(ctx->daX, &ctx->xs, NULL, NULL, &ctx->NxLocal, NULL, NULL));
  ctx->NxDOF_local = ctx->NxLocal * ctx->NbX;

  /* daF: same topology as daX, used for per-iv ghost exchange in AdvectX.
     dof=NbX and stencil width sw are identical to daX. */
  PetscCall(DMDACreate1d(comm, DM_BOUNDARY_PERIODIC, ctx->Nx, ctx->NbX, sw, NULL, &ctx->daF));
  PetscCall(DMSetUp(ctx->daF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateVMesh(MPI_Comm comm, AppCtx *ctx)
{
  PetscFunctionBeginUser;
  (void)comm; /* v-space is replicated on PETSC_COMM_SELF on every rank */
  /* 1D non-periodic DMDA for v-space on PETSC_COMM_SELF.
     Each rank owns the full v-space. */
  PetscCall(DMDACreate1d(PETSC_COMM_SELF, DM_BOUNDARY_NONE, ctx->Nv, ctx->NbV, 0, NULL, &ctx->daV));
  PetscCall(DMSetUp(ctx->daV));
  ctx->NvDOF = ctx->Nv * ctx->NbV;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 3 — SetupPoisson
   ======================================================================== */
static PetscErrorCode SetupPoisson(MPI_Comm comm, AppCtx *ctx)
{
  MatNullSpace nullsp;
  PC           pc;
  PetscInt     xs, xm;
  PetscReal    h2inv;

  PetscFunctionBeginUser;

  /* 1D periodic DMDA for potential (dof=1, one value per cell) */
  PetscCall(DMDACreate1d(comm, DM_BOUNDARY_PERIODIC, ctx->Nx, 1, 1, NULL, &ctx->daPot));
  PetscCall(DMSetUp(ctx->daPot));

  /* Create global Vecs from daPot */
  PetscCall(DMCreateGlobalVector(ctx->daPot, &ctx->rho));
  PetscCall(DMCreateGlobalVector(ctx->daPot, &ctx->phi));
  PetscCall(DMCreateGlobalVector(ctx->daPot, &ctx->E_field));
  PetscCall(PetscObjectSetName((PetscObject)ctx->rho,     "rho"));
  PetscCall(PetscObjectSetName((PetscObject)ctx->phi,     "phi"));
  PetscCall(PetscObjectSetName((PetscObject)ctx->E_field, "E_field"));

  /* Build the periodic 1D Laplacian matrix: -phi'' = rho
     Using second-order finite differences on cell-centered values.
     Diagonal: -2/h^2,  off-diagonal: 1/h^2 (periodic). */
  PetscCall(DMCreateMatrix(ctx->daPot, &ctx->Jac));
  PetscCall(DMDAGetCorners(ctx->daPot, &xs, NULL, NULL, &xm, NULL, NULL));

  h2inv = 1.0 / (ctx->h_x * ctx->h_x);

  {
    MatStencil  row, col[3];
    PetscScalar vals[3];
    for (PetscInt i = xs; i < xs + xm; ++i) {
      row.i = i; row.c = 0;

      /* For periodic DMDA, MatSetValuesStencil handles wrap-around automatically.
         Use i-1 and i+1 directly — no manual modular arithmetic needed. */
      col[0].i = i;     col[0].c = 0;
      col[1].i = i - 1; col[1].c = 0;
      col[2].i = i + 1; col[2].c = 0;

      vals[0] =  2.0 * h2inv;
      vals[1] = -1.0 * h2inv;
      vals[2] = -1.0 * h2inv;

      PetscCall(MatSetValuesStencil(ctx->Jac, 1, &row, 3, col, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(ctx->Jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->Jac, MAT_FINAL_ASSEMBLY));

  /* Fix the constant null space */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullsp));
  PetscCall(MatSetNullSpace(ctx->Jac, nullsp));
  PetscCall(MatNullSpaceDestroy(&nullsp));

  /* Set up KSP: CG + GAMG with SVD coarse solver for robustness on
     the 1D periodic Laplacian (GAMG can produce indefinite PC otherwise). */
  PetscCall(KSPCreate(comm, &ctx->kspPoisson));
  PetscCall(KSPSetOptionsPrefix(ctx->kspPoisson, "poisson_"));
  PetscCall(KSPSetOperators(ctx->kspPoisson, ctx->Jac, ctx->Jac));
  PetscCall(KSPSetType(ctx->kspPoisson, KSPCG));
  PetscCall(KSPGetPC(ctx->kspPoisson, &pc));
  PetscCall(PCSetType(pc, PCGAMG));
  /* Force SVD on the GAMG coarse level to handle the null space robustly */
  PetscCall(PetscOptionsSetValue(NULL, "-poisson_mg_coarse_ksp_type", "preonly"));
  PetscCall(PetscOptionsSetValue(NULL, "-poisson_mg_coarse_pc_type", "svd"));
  PetscCall(KSPSetTolerances(ctx->kspPoisson, 1e-10, 1e-12, PETSC_DEFAULT, 200));
  PetscCall(KSPSetFromOptions(ctx->kspPoisson));
  PetscCall(KSPSetUp(ctx->kspPoisson));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 4 — PrecomputeGeometry
   ======================================================================== */
static PetscErrorCode PrecomputeGeometry(AppCtx *ctx)
{
  PetscReal *gll_wts_x, *gll_wts_v;

  PetscFunctionBeginUser;

  /* Step 1: GLL nodes for x and v DOFs */
  PetscCall(PetscMalloc1(ctx->NbX, &ctx->xi_x_nodes));
  PetscCall(PetscMalloc1(ctx->NbX, &gll_wts_x));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(ctx->NbX, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, ctx->xi_x_nodes, gll_wts_x));

  PetscCall(PetscMalloc1(ctx->NbV, &ctx->xi_v_nodes));
  PetscCall(PetscMalloc1(ctx->NbV, &gll_wts_v));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(ctx->NbV, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, ctx->xi_v_nodes, gll_wts_v));

  /* Step 2: Physical v-DOF coordinates */
  PetscCall(PetscMalloc1(ctx->NvDOF, &ctx->v_dof_coords));
  // TODO: Kokkos::parallel_for over (cv, bv)
  for (PetscInt cv = 0; cv < ctx->Nv; ++cv)
    for (PetscInt bv = 0; bv < ctx->NbV; ++bv)
      ctx->v_dof_coords[cv * ctx->NbV + bv] =
        -ctx->v_max + cv * ctx->h_v + (ctx->xi_v_nodes[bv] + 1.0) * (ctx->h_v * 0.5);

  /* Step 3: Gauss quadrature for v-advection (NqV = NbV + 2 points) */
  ctx->NqV = ctx->NbV + 2;
  PetscCall(PetscMalloc1(ctx->NqV, &ctx->vq_pts));
  PetscCall(PetscMalloc1(ctx->NqV, &ctx->vq_wts));
  PetscCall(PetscDTGaussQuadrature(ctx->NqV, -1.0, 1.0, ctx->vq_pts, ctx->vq_wts));

  /* Step 4: Tabulate v-basis at quadrature points */
  PetscCall(PetscMalloc1(ctx->NqV * ctx->NbV, &ctx->tabV));
  // TODO: Kokkos::parallel_for over (q, bv)
  for (PetscInt q = 0; q < ctx->NqV; ++q)
    for (PetscInt bv = 0; bv < ctx->NbV; ++bv)
      ctx->tabV[q * ctx->NbV + bv] = EvalLagrangeBasis(ctx->NbV, ctx->xi_v_nodes, bv, ctx->vq_pts[q]);

  /* Step 5: v-basis integrals for charge density computation */
  PetscCall(PetscCalloc1(ctx->NvDOF, &ctx->v_basis_int));
  // TODO: Kokkos::parallel_for over (cv, bv)
  for (PetscInt cv = 0; cv < ctx->Nv; ++cv) {
    for (PetscInt bv = 0; bv < ctx->NbV; ++bv) {
      PetscReal sum = 0.0;
      for (PetscInt q = 0; q < ctx->NqV; ++q)
        sum += ctx->tabV[q * ctx->NbV + bv] * ctx->vq_wts[q] * (ctx->h_v * 0.5);
      ctx->v_basis_int[cv * ctx->NbV + bv] = sum;
    }
  }

  /* Step 6: x-basis integrals for charge density and moments
     x_basis_int[bx] = gll_wts_x[bx] * h_x/2 (integral of x-basis over cell) */
  PetscCall(PetscMalloc1(ctx->NbX, &ctx->x_basis_int));
  for (PetscInt bx = 0; bx < ctx->NbX; ++bx)
    ctx->x_basis_int[bx] = gll_wts_x[bx] * (ctx->h_x * 0.5);

  /* Step 7: v-mass matrix inverse (diagonal, GLL mass matrix is diagonal) */
  PetscCall(PetscMalloc1(ctx->NvDOF, &ctx->MassV_inv));
  // TODO: Kokkos::parallel_for over (cv, bv)
  for (PetscInt cv = 0; cv < ctx->Nv; ++cv)
    for (PetscInt bv = 0; bv < ctx->NbV; ++bv)
      ctx->MassV_inv[cv * ctx->NbV + bv] = 1.0 / (gll_wts_v[bv] * ctx->h_v * 0.5);

  PetscCall(PetscFree(gll_wts_x));
  PetscCall(PetscFree(gll_wts_v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 5 — AllocateF + InitializeF
   ======================================================================== */
static PetscErrorCode AllocateF(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  /* f is a global Vec with local size NvDOF * NxDOF_local */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ctx->f));
  PetscCall(VecSetSizes(ctx->f, ctx->NvDOF * ctx->NxDOF_local, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(ctx->f));
  PetscCall(PetscObjectSetName((PetscObject)ctx->f, "f_dist"));
  /* TODO: When adding Kokkos GPU: VecSetType(ctx->f, VECKOKKOS); */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeF(AppCtx *ctx)
{
  PetscScalar *f_arr;

  PetscFunctionBeginUser;
  PetscCall(VecGetArray(ctx->f, &f_arr));

  // TODO: When adding Kokkos, replace with Kokkos::parallel_for over (iv, ix)
  for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
    PetscReal v = ctx->v_dof_coords[iv];
    for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix) {
      PetscInt  cx_local    = ix / ctx->NbX;
      PetscInt  bx          = ix % ctx->NbX;
      PetscReal x_cell_left = (ctx->xs + cx_local) * ctx->h_x;
      PetscReal x           = x_cell_left + (ctx->xi_x_nodes[bx] + 1.0) * (ctx->h_x * 0.5);
      PetscReal f0          = (1.0 + ctx->alpha * PetscCosReal(ctx->kwave * x))
                              / PetscSqrtReal(2.0 * PETSC_PI) * PetscExpReal(-0.5 * v * v);
      f_arr[iv * ctx->NxDOF_local + ix] = (PetscScalar)f0;
    }
  }

  PetscCall(VecRestoreArray(ctx->f, &f_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 6 — ComputeChargeDensity
   ======================================================================== */
static PetscErrorCode ComputeChargeDensity(AppCtx *ctx)
{
  const PetscScalar *f_arr;
  PetscScalar       *rho_arr;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->RhoEvent, 0, 0, 0, 0));

  /* rho[cx_local] = (1/h_x) * sum_{iv} sum_{bx} f[iv*NxDOF_local + cx_local*NbX + bx]
                     * v_basis_int[iv] * gll_wts_x[bx] * h_x/2
     The factor 1/h_x * h_x/2 = 1/2 cancels with the x-basis integral normalization.
     x_basis_int[bx] = gll_wts_x[bx] * h_x/2 (precomputed in PrecomputeGeometry).
     rho = (1/h_x) * integral_x integral_v f dv dx
         = sum_{iv} v_basis_int[iv] * sum_{bx} f[iv,bx] * gll_wts_x[bx] * (h_x/2) / h_x
         = sum_{iv} v_basis_int[iv] * sum_{bx} f[iv,bx] * gll_wts_x[bx] / 2 */

  PetscCall(VecGetArrayRead(ctx->f, &f_arr));
  PetscCall(VecGetArray(ctx->rho, &rho_arr));

  for (PetscInt cx_local = 0; cx_local < ctx->NxLocal; ++cx_local) {
    PetscReal rho_val = 0.0;
    for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
      PetscReal w = ctx->v_basis_int[iv];
      for (PetscInt bx = 0; bx < ctx->NbX; ++bx) {
        PetscReal fval = PetscRealPart(f_arr[iv * ctx->NxDOF_local + cx_local * ctx->NbX + bx]);
        /* x_basis_int[bx] = gll_wts_x[bx] * h_x/2, divide by h_x to get cell average */
        rho_val += fval * w * ctx->x_basis_int[bx];
      }
    }
    rho_arr[cx_local] = (PetscScalar)(rho_val / ctx->h_x);
  }

  PetscCall(VecRestoreArrayRead(ctx->f, &f_arr));
  PetscCall(VecRestoreArray(ctx->rho, &rho_arr));

  PetscCall(PetscLogEventEnd(ctx->RhoEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 7 — SolvePoisson + ComputeElectricField
   ======================================================================== */
static PetscErrorCode SolvePoisson(AppCtx *ctx)
{
  Vec rhs;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->PoissonEvent, 0, 0, 0, 0));

  /* Solve -phi'' = rho - sigma
     RHS = rho - sigma (subtract background ion density). */
  PetscCall(VecDuplicate(ctx->rho, &rhs));
  PetscCall(VecCopy(ctx->rho, rhs));
  PetscCall(VecShift(rhs, -(PetscScalar)ctx->sigma));

  PetscCall(VecZeroEntries(ctx->phi));
  PetscCall(KSPSolve(ctx->kspPoisson, rhs, ctx->phi));

  PetscCall(VecDestroy(&rhs));
  PetscCall(PetscLogEventEnd(ctx->PoissonEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeElectricField(AppCtx *ctx)
{
  /* E = -dphi/dx using central finite differences on the periodic DMDA.
     E[i] = -(phi[i+1] - phi[i-1]) / (2 * h_x)
     phi_local: ghost-aware local vector, indexed by global cell index.
     E_field:   global Vec, accessed via VecGetArray (0-based local indexing). */
  PetscScalar   *phi_arr, *E_arr;
  DMDALocalInfo  info;
  Vec            phi_local;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(ctx->daPot, &info));

  PetscCall(DMGetLocalVector(ctx->daPot, &phi_local));
  PetscCall(DMGlobalToLocal(ctx->daPot, ctx->phi, INSERT_VALUES, phi_local));

  PetscCall(DMDAVecGetArray(ctx->daPot, phi_local, &phi_arr));
  PetscCall(VecGetArray(ctx->E_field, &E_arr));

  PetscReal inv2hx = 1.0 / (2.0 * ctx->h_x);
  for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
    PetscInt cx_local = i - info.xs;
    E_arr[cx_local] = -(PetscScalar)((PetscRealPart(phi_arr[i+1]) - PetscRealPart(phi_arr[i-1])) * inv2hx);
  }

  PetscCall(DMDAVecRestoreArray(ctx->daPot, phi_local, &phi_arr));
  PetscCall(VecRestoreArray(ctx->E_field, &E_arr));
  PetscCall(DMRestoreLocalVector(ctx->daPot, &phi_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 8 — AdvectX (SLDG L2 projection with DMDA ghost exchange)

   Uses the SLDG overlap-integral approach: for each velocity DOF, compute
   the fractional cell shift s = v*dt/h_x, decompose into integer shift n
   and fractional part alpha.  Build overlap matrices A and B via Gauss
   quadrature on the two sub-intervals, then apply:
     f_new[i] = (1/M[i]) * sum_j (A[i,j]*f[src_A,j] + B[i,j]*f[src_B,j])
   where M[i] = x_basis_int[i] is the diagonal GLL mass matrix entry.

   Ghost exchange: instead of MPI_Allgather, we copy each iv-slice into a
   global Vec on daF, call DMGlobalToLocal, and read ghost cells via
   DMDAVecGetArrayDOF.  daF has the same periodic topology and stencil width
   as daX, so ghost cells cover all source cells needed by the SLDG stencil.
   ======================================================================== */
static PetscErrorCode AdvectX(Vec f, Vec f_out, PetscReal dt, AppCtx *ctx)
{
  const PetscScalar *f_arr;
  PetscScalar       *f_out_arr;
  Vec                f_tmp = NULL;
  PetscBool          inplace;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->AdvectXEvent, 0, 0, 0, 0));

  /* If f and f_out alias the same Vec, use a temporary output buffer */
  inplace = (PetscBool)(f == f_out);
  if (inplace) {
    PetscCall(VecDuplicate(f, &f_tmp));
    f_out = f_tmp;
  }

  PetscInt NqSLDG = 2 * ctx->NbX + 2;
  PetscReal *qa, *wa, *qb, *wb;
  PetscCall(PetscMalloc1(NqSLDG, &qa));
  PetscCall(PetscMalloc1(NqSLDG, &wa));
  PetscCall(PetscMalloc1(NqSLDG, &qb));
  PetscCall(PetscMalloc1(NqSLDG, &wb));

  /* Allocate a global Vec on daF for one iv-slice, and a local Vec with ghosts */
  Vec        f_iv_global, f_iv_local;
  PetscReal **f_iv_arr; /* ghost-aware 2D array: f_iv_arr[cell][dof] */
  PetscCall(DMCreateGlobalVector(ctx->daF, &f_iv_global));
  PetscCall(DMCreateLocalVector(ctx->daF, &f_iv_local));

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(ctx->daF, &info));

  PetscCall(VecGetArrayRead(f, &f_arr));
  PetscCall(VecGetArray(f_out, &f_out_arr));

  /* A and B matrices: max NbX=4 (degree 3), so NbX*NbX <= 16 */
  PetscReal A[16], B[16];

  for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
    PetscReal v   = ctx->v_dof_coords[iv];
    PetscReal s   = v * dt / ctx->h_x;
    PetscInt  n   = (PetscInt)PetscFloorReal(s);
    PetscReal alp = s - (PetscReal)n;  /* in [0,1) */

    /* Snap near-integer shifts: if alp ≈ 1, treat as n+1 with alp=0 */
    if (alp > 1.0 - 1e-14) { n++; alp = 0.0; }

    PetscReal split = 1.0 - 2.0 * alp; /* split point in [-1,1] */

    /* Compute A and B matrices for this iv and dt */
    for (PetscInt k = 0; k < ctx->NbX * ctx->NbX; ++k) A[k] = B[k] = 0.0;

    /* Handle degenerate case: alp == 0 (pure integer shift) */
    if (alp < 1e-14) {
      /* Pure integer shift: A = lumped reference mass matrix = diag(w_i), B = 0.
         This ensures f_new = A*f / w_i = f_old (identity). */
      for (PetscInt i = 0; i < ctx->NbX; ++i)
        A[i * ctx->NbX + i] = ctx->x_basis_int[i] / (ctx->h_x * 0.5);
    } else {
      PetscCall(PetscDTGaussQuadrature(NqSLDG, -1.0, split, qa, wa));
      PetscCall(PetscDTGaussQuadrature(NqSLDG, split, 1.0, qb, wb));

      /* A[i,j] = integral_{-1}^{split} phi_i(xi) * phi_j(xi + 2*alp) dxi */
      for (PetscInt q = 0; q < NqSLDG; ++q) {
        PetscReal xi = qa[q], xi2 = xi + 2.0 * alp;
        for (PetscInt i = 0; i < ctx->NbX; ++i) {
          PetscReal phi_i = EvalLagrangeBasis(ctx->NbX, ctx->xi_x_nodes, i, xi);
          for (PetscInt j = 0; j < ctx->NbX; ++j)
            A[i * ctx->NbX + j] += phi_i * EvalLagrangeBasis(ctx->NbX, ctx->xi_x_nodes, j, xi2) * wa[q];
        }
      }
      /* B[i,j] = integral_{split}^{1} phi_i(xi) * phi_j(xi + 2*alp - 2) dxi */
      for (PetscInt q = 0; q < NqSLDG; ++q) {
        PetscReal xi = qb[q], xi2 = xi + 2.0 * alp - 2.0;
        for (PetscInt i = 0; i < ctx->NbX; ++i) {
          PetscReal phi_i = EvalLagrangeBasis(ctx->NbX, ctx->xi_x_nodes, i, xi);
          for (PetscInt j = 0; j < ctx->NbX; ++j)
            B[i * ctx->NbX + j] += phi_i * EvalLagrangeBasis(ctx->NbX, ctx->xi_x_nodes, j, xi2) * wb[q];
        }
      }
    }

    /* Copy this iv-slice into the global Vec on daF, then ghost-exchange */
    {
      PetscScalar *gv_arr;
      PetscCall(VecGetArray(f_iv_global, &gv_arr));
      for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix)
        gv_arr[ix] = f_arr[iv * ctx->NxDOF_local + ix];
      PetscCall(VecRestoreArray(f_iv_global, &gv_arr));
    }
    PetscCall(DMGlobalToLocal(ctx->daF, f_iv_global, INSERT_VALUES, f_iv_local));
    PetscCall(DMDAVecGetArrayDOF(ctx->daF, f_iv_local, &f_iv_arr));

    /* Apply SLDG update for each local x-cell.
       f_iv_arr[cell][bx] gives ghost-aware access: cell ranges from
       info.xs - info.sw to info.xs + info.xm + info.sw - 1. */
    for (PetscInt cx_local = 0; cx_local < ctx->NxLocal; ++cx_local) {
      PetscInt cx_global = ctx->xs + cx_local;
      /* Source cell for A: cx_global - n (periodic, covered by ghosts) */
      PetscInt src_A = cx_global - n;
      /* Source cell for B: cx_global - n - 1 (periodic, covered by ghosts) */
      PetscInt src_B = cx_global - n - 1;

      for (PetscInt i = 0; i < ctx->NbX; ++i) {
        PetscReal val = 0.0;
        for (PetscInt j = 0; j < ctx->NbX; ++j) {
          val += A[i * ctx->NbX + j] * PetscRealPart(f_iv_arr[src_A][j]);
          val += B[i * ctx->NbX + j] * PetscRealPart(f_iv_arr[src_B][j]);
        }
        /* Apply reference mass matrix inverse: overlap integrals are in
           reference coordinates [-1,1], so divide by GLL weight w_i only
           (not w_i * h_x/2).  x_basis_int[i] = w_i * h_x/2. */
        PetscReal w_i = ctx->x_basis_int[i] / (ctx->h_x * 0.5);
        f_out_arr[iv * ctx->NxDOF_local + cx_local * ctx->NbX + i] =
          (PetscScalar)(val / w_i);
      }
    }

    PetscCall(DMDAVecRestoreArrayDOF(ctx->daF, f_iv_local, &f_iv_arr));
  }

  PetscCall(PetscFree(qa)); PetscCall(PetscFree(wa));
  PetscCall(PetscFree(qb)); PetscCall(PetscFree(wb));
  PetscCall(VecRestoreArrayRead(f, &f_arr));
  PetscCall(VecRestoreArray(f_out, &f_out_arr));
  PetscCall(VecDestroy(&f_iv_global));
  PetscCall(VecDestroy(&f_iv_local));

  /* Copy result back if in-place */
  if (inplace) {
    PetscCall(VecCopy(f_tmp, f));
    PetscCall(VecDestroy(&f_tmp));
  }

  PetscCall(PetscLogEventEnd(ctx->AdvectXEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 9 — AdvectV (semi-Lagrangian DG L2 projection in velocity)

   L2 projection of f(v - E*dt, x) onto the DG basis using Gauss quadrature.
   For each quadrature point in the output v-cell, trace the characteristic
   back by E*dt, interpolate f at the foot point via Lagrange interpolation
   in the source v-cell, then accumulate the RHS.  Finally apply the diagonal
   GLL mass matrix inverse.
   ======================================================================== */
static PetscErrorCode AdvectV(Vec f, Vec f_out, PetscReal dt, Vec E_field, AppCtx *ctx)
{
  const PetscScalar *f_arr, *E_arr;
  PetscScalar       *f_out_arr;
  Vec                f_tmp = NULL;
  PetscBool          inplace;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->AdvectVEvent, 0, 0, 0, 0));

  /* If f and f_out alias the same Vec, use a temporary output buffer */
  inplace = (PetscBool)(f == f_out);
  if (inplace) {
    PetscCall(VecDuplicate(f, &f_tmp));
    f_out = f_tmp;
  }

  PetscCall(VecGetArrayRead(f, &f_arr));
  PetscCall(VecGetArrayRead(E_field, &E_arr));
  PetscCall(VecGetArray(f_out, &f_out_arr));

  /* Allocate RHS array: rhs[iv * NxDOF_local + ix] */
  PetscReal *rhs;
  PetscCall(PetscCalloc1(ctx->NvDOF * ctx->NxDOF_local, &rhs));

  for (PetscInt cx_local = 0; cx_local < ctx->NxLocal; ++cx_local) {
    PetscReal E = PetscRealPart(E_arr[cx_local]);

    for (PetscInt cv = 0; cv < ctx->Nv; ++cv) {
      PetscReal v_cell_left = -ctx->v_max + cv * ctx->h_v;

      for (PetscInt q = 0; q < ctx->NqV; ++q) {
        PetscReal xi     = ctx->vq_pts[q];
        PetscReal v_phys = v_cell_left + (xi + 1.0) * (ctx->h_v * 0.5);
        PetscReal v_foot = v_phys - E * dt;

        /* Locate foot cell (clamp to boundary for non-periodic v) */
        PetscReal v_norm  = (v_foot + ctx->v_max) / ctx->h_v;
        PetscInt  cv_foot = (PetscInt)PetscFloorReal(v_norm);
        if (cv_foot < 0) cv_foot = 0;
        if (cv_foot >= ctx->Nv) cv_foot = ctx->Nv - 1;

        PetscReal xi_foot = (v_foot - (-ctx->v_max + cv_foot * ctx->h_v)) / (ctx->h_v * 0.5) - 1.0;
        if (xi_foot < -1.0) xi_foot = -1.0;
        if (xi_foot >  1.0) xi_foot =  1.0;

        /* Interpolate f at foot point for each x-DOF in this x-cell */
        for (PetscInt bx = 0; bx < ctx->NbX; ++bx) {
          PetscInt ix = cx_local * ctx->NbX + bx;
          PetscReal f_val = 0.0;
          for (PetscInt bv = 0; bv < ctx->NbV; ++bv) {
            PetscInt iv_dof = cv_foot * ctx->NbV + bv;
            f_val += PetscRealPart(f_arr[iv_dof * ctx->NxDOF_local + ix])
                   * EvalLagrangeBasis(ctx->NbV, ctx->xi_v_nodes, bv, xi_foot);
          }

          /* Accumulate RHS: rhs[iv_dof, ix] += f_val * phi_bv(xi_q) * wq * h_v/2 */
          for (PetscInt bv = 0; bv < ctx->NbV; ++bv) {
            PetscInt iv_dof = cv * ctx->NbV + bv;
            rhs[iv_dof * ctx->NxDOF_local + ix] +=
              f_val * ctx->tabV[q * ctx->NbV + bv] * ctx->vq_wts[q] * (ctx->h_v * 0.5);
          }
        }
      }
    }
  }

  /* Apply mass matrix inverse (diagonal for GLL basis) */
  for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv)
    for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix)
      f_out_arr[iv * ctx->NxDOF_local + ix] = (PetscScalar)(rhs[iv * ctx->NxDOF_local + ix] * ctx->MassV_inv[iv]);

  PetscCall(PetscFree(rhs));
  PetscCall(VecRestoreArrayRead(f, &f_arr));
  PetscCall(VecRestoreArrayRead(E_field, &E_arr));
  PetscCall(VecRestoreArray(f_out, &f_out_arr));

  /* Copy result back if in-place */
  if (inplace) {
    PetscCall(VecCopy(f_tmp, f));
    PetscCall(VecDestroy(&f_tmp));
  }

  PetscCall(PetscLogEventEnd(ctx->AdvectVEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 10 — ComputeMoments
   ======================================================================== */
static PetscErrorCode ComputeMoments(AppCtx *ctx, PetscReal *m0_out, PetscReal *m1_out, PetscReal *m2_out)
{
  const PetscScalar *f_arr;
  PetscReal m0_local = 0.0, m1_local = 0.0, m2_local = 0.0;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(ctx->f, &f_arr));

  for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
    PetscReal w0 = ctx->v_basis_int[iv];
    PetscInt  cv = iv / ctx->NbV;
    PetscInt  bv = iv % ctx->NbV;
    PetscReal v_cell_left = -ctx->v_max + cv * ctx->h_v;
    PetscReal w1 = 0.0, w2 = 0.0;
    /* Compute v-weighted integrals using quadrature */
    for (PetscInt q = 0; q < ctx->NqV; ++q) {
      PetscReal v_phys = v_cell_left + (ctx->vq_pts[q] + 1.0) * (ctx->h_v * 0.5);
      PetscReal phi_bv = ctx->tabV[q * ctx->NbV + bv];
      w1 += v_phys * phi_bv * ctx->vq_wts[q] * (ctx->h_v * 0.5);
      w2 += v_phys * v_phys * phi_bv * ctx->vq_wts[q] * (ctx->h_v * 0.5);
    }
    for (PetscInt cx_local = 0; cx_local < ctx->NxLocal; ++cx_local) {
      for (PetscInt bx = 0; bx < ctx->NbX; ++bx) {
        PetscInt  ix   = cx_local * ctx->NbX + bx;
        PetscReal fval = PetscRealPart(f_arr[iv * ctx->NxDOF_local + ix]);
        /* x_basis_int[bx] = gll_wts_x[bx] * h_x/2 */
        PetscReal dx   = ctx->x_basis_int[bx];
        m0_local += fval * w0 * dx;
        m1_local += fval * w1 * dx;
        m2_local += fval * w2 * dx;
      }
    }
  }
  PetscCall(VecRestoreArrayRead(ctx->f, &f_arr));

  PetscReal loc[3] = {m0_local, m1_local, m2_local}, glob[3];
  PetscCallMPI(MPIU_Allreduce(loc, glob, 3, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)ctx->f)));
  *m0_out = glob[0]; *m1_out = glob[1]; *m2_out = glob[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 10 — BSLStep_Strang (Strang splitting: X/2, V, X/2)
   ======================================================================== */
static PetscErrorCode BSLStep_Strang(AppCtx *ctx, PetscReal dt)
{
  PetscFunctionBeginUser;
  PetscCall(AdvectX(ctx->f, ctx->f, dt * 0.5, ctx));
  PetscCall(ComputeChargeDensity(ctx));
  PetscCall(SolvePoisson(ctx));
  PetscCall(ComputeElectricField(ctx));
  PetscCall(AdvectV(ctx->f, ctx->f, dt, ctx->E_field, ctx));
  PetscCall(AdvectX(ctx->f, ctx->f, dt * 0.5, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 10 — DestroyContext
   ======================================================================== */
static PetscErrorCode DestroyContext(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&ctx->f));
  PetscCall(VecDestroy(&ctx->rho));
  PetscCall(VecDestroy(&ctx->phi));
  PetscCall(VecDestroy(&ctx->E_field));
  PetscCall(KSPDestroy(&ctx->kspPoisson));
  PetscCall(MatDestroy(&ctx->Jac));
  PetscCall(DMDestroy(&ctx->daX));
  PetscCall(DMDestroy(&ctx->daV));
  PetscCall(DMDestroy(&ctx->daPot));
  PetscCall(DMDestroy(&ctx->daF));
  PetscCall(PetscFree(ctx->xi_x_nodes));
  PetscCall(PetscFree(ctx->xi_v_nodes));
  PetscCall(PetscFree(ctx->v_dof_coords));
  PetscCall(PetscFree(ctx->v_basis_int));
  PetscCall(PetscFree(ctx->x_basis_int));
  PetscCall(PetscFree(ctx->vq_pts));
  PetscCall(PetscFree(ctx->vq_wts));
  PetscCall(PetscFree(ctx->tabV));
  PetscCall(PetscFree(ctx->MassV_inv));
  PetscCall(PetscFree(ctx->A_sldg));
  PetscCall(PetscFree(ctx->B_sldg));
  PetscCall(PetscFree(ctx->n_shift));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   main — full time loop
   ======================================================================== */
int main(int argc, char **argv)
{
  AppCtx ctx;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Phase 1 */
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));

  /* Phase 2 */
  PetscCall(CreateXMesh(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateVMesh(PETSC_COMM_WORLD, &ctx));

  /* Phase 3 */
  PetscCall(SetupPoisson(PETSC_COMM_WORLD, &ctx));

  /* Phase 4 */
  PetscCall(PrecomputeGeometry(&ctx));

  /* Phase 5 */
  PetscCall(AllocateF(&ctx));
  PetscCall(InitializeF(&ctx));

  /* Initial field solve */
  PetscCall(ComputeChargeDensity(&ctx));
  PetscCall(SolvePoisson(&ctx));
  PetscCall(ComputeElectricField(&ctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "SLDG Landau damping: Nx=%" PetscInt_FMT " Nv=%" PetscInt_FMT
    " degree_x=%" PetscInt_FMT " degree_v=%" PetscInt_FMT
    " alpha=%g kwave=%g dt=%g steps=%" PetscInt_FMT "\n",
    ctx.Nx, ctx.Nv, ctx.degree_x, ctx.degree_v,
    (double)ctx.alpha, (double)ctx.kwave, (double)ctx.dt, ctx.steps));

  /* E-field history for Landau check */
  PetscReal *E_history = NULL;
  if (ctx.check_landau) PetscCall(PetscMalloc1(ctx.steps + 1, &E_history));

  /* Print header */
  if (ctx.ostep > 0)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "%-12s %-20s %-20s %-12s %-20s %-12s %-20s %-20s %-20s\n",
      "Time", "Sum_E", "|E|", "log|E|", "E_max", "log(E_max)", "m0", "m1", "m2"));

  /* Main time loop */
  for (PetscInt step = 0; step < ctx.steps; ++step) {
    PetscCall(BSLStep_Strang(&ctx, ctx.dt));

    PetscReal E_max;
    PetscCall(VecNorm(ctx.E_field, NORM_INFINITY, &E_max));

    if (ctx.check_landau) E_history[step] = E_max;

    if (ctx.ostep > 0 && step % ctx.ostep == 0) {
      PetscReal t = (step + 1) * ctx.dt;
      PetscReal m0, m1, m2;
      PetscReal E_norm;
      PetscScalar E_sum_sc;
      PetscCall(VecNorm(ctx.E_field, NORM_2, &E_norm));
      PetscCall(VecSum(ctx.E_field, &E_sum_sc));
      PetscReal E_sum = PetscRealPart(E_sum_sc);
      PetscCall(ComputeMoments(&ctx, &m0, &m1, &m2));
      PetscReal lgEmax  = E_max  > 0 ? PetscLog10Real(E_max)  : -16.0;
      PetscReal lgEnorm = E_norm > 0 ? PetscLog10Real(E_norm) : -16.0;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "E: %f\t%+e\t%e\t%f\t%20.15e\t%f\t%20.15e\t%20.15e\t%20.15e\t(%" PetscInt_FMT ")\n",
        (double)t, (double)E_sum, (double)E_norm, (double)lgEnorm,
        (double)E_max, (double)lgEmax, (double)m0, (double)m1, (double)m2, step));
    }
  }

  /* Landau damping rate check */
  if (ctx.check_landau) {
    /* Find first minimum, fit log-linear to envelope peaks */
    PetscReal gamma_theory = (PetscAbsReal(ctx.alpha - 0.5) < 0.1) ? -0.286 : -0.1533;
    PetscInt first_min = ctx.steps - 1;
    for (PetscInt s = 1; s < ctx.steps - 1; ++s)
      if (E_history[s] < E_history[s-1] && E_history[s] < E_history[s+1]) { first_min = s; break; }

    PetscReal sum_t = 0, sum_logE = 0, sum_t_logE = 0, sum_t2 = 0;
    PetscInt  n_pts = 0;
    PetscReal last_peak = PETSC_MAX_REAL;
    for (PetscInt s = first_min + 1; s < ctx.steps - 1; ++s) {
      PetscReal val = E_history[s];
      if (val > E_history[s-1] && val > E_history[s+1] && val > 1e-16) {
        if (val > last_peak) break;
        last_peak = val;
        PetscReal t = (s + 1) * ctx.dt;
        sum_t += t; sum_logE += PetscLogReal(val);
        sum_t_logE += t * PetscLogReal(val); sum_t2 += t * t;
        n_pts++;
      }
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nLandau Damping Check:\n"));
    if (n_pts > 1) {
      PetscReal gamma = (n_pts * sum_t_logE - sum_t * sum_logE) / (n_pts * sum_t2 - sum_t * sum_t);
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Measured gamma = %g  (theory: %g)\n",
        (double)gamma, (double)gamma_theory));
    } else
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Not enough peaks to fit gamma\n"));
    PetscCall(PetscFree(E_history));
  }

  PetscCall(DestroyContext(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex double

  testset:
    args: -Nx 32 -Nv 64 -degree_x 1 -degree_v 1 -dt 0.1 -steps 10 -ostep 1

    test:
      suffix: 0
      args:

    test:
      suffix: mpi
      nsize: 2
      args:

  test:
    suffix: landau
    args: -Nx 64 -Nv 128 -degree_x 1 -degree_v 1 -dt 0.1 -steps 200 -ostep 10 -check_landau

TEST*/
