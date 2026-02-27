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

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
using DevSpace     = Kokkos::DefaultExecutionSpace::memory_space;
using ScalarView1D = Kokkos::View<PetscScalar *, DevSpace>;
using RealView1D   = Kokkos::View<PetscReal *, DevSpace>;
using IntView1D    = Kokkos::View<PetscInt *, DevSpace>;
using RealView2D   = Kokkos::View<PetscReal **, Kokkos::LayoutRight, DevSpace>;
#endif

/* ========================================================================
   AppCtx -- all simulation state
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
  Vec f_iv_global; /* per-iv slice global Vec for AdvectX ghost exchange */
  Vec f_iv_local;  /* per-iv slice local Vec with ghosts for AdvectX */
  Vec rhs_poisson; /* reusable RHS Vec for Poisson solve */
  KSP kspPoisson;
  Mat Jac;

  /* Precomputed geometry (host arrays, Kokkos-ready) */
  PetscReal *xi_x_nodes;   /* [NbX] GLL reference coords of x-DOF nodes */
  PetscReal *xi_v_nodes;   /* [NbV] GLL reference coords of v-DOF nodes */
  PetscReal *v_dof_coords; /* [NvDOF] physical v-coord of each v-DOF */
  PetscReal *v_basis_int;  /* [NvDOF] integral of v-basis over its support */
  PetscReal *x_basis_int;  /* [NbX] integral of x-basis: gll_wts_x[bx] * h_x/2 */

  /* SLDG matrices for x-advection (precomputed per velocity DOF) */
  PetscReal *A_sldg;  /* [NvDOF * NbX * NbX] same-cell overlap */
  PetscReal *B_sldg;  /* [NvDOF * NbX * NbX] neighbor-cell overlap */
  PetscInt  *n_shift; /* [NvDOF] integer cell shift per v-DOF */

  /* Quadrature for v-advection */
  PetscReal *vq_pts; /* [NqV] Gauss quadrature points in [-1,1] */
  PetscReal *vq_wts; /* [NqV] Gauss quadrature weights */
  PetscReal *tabV;   /* [NqV * NbV] v-basis tabulation at vq_pts */

  /* Sizes */
  PetscInt NxLocal;     /* local number of x-cells */
  PetscInt NxDOF_local; /* local x-DOFs = NxLocal * NbX */
  PetscInt NvDOF;       /* total v-DOFs = Nv * NbV */
  PetscInt xs;          /* global x-cell start index for this rank */
  PetscInt Nx;          /* global number of x-cells */
  PetscInt Nv;          /* number of v-cells (same on all ranks) */
  PetscInt NbX;         /* basis functions per x-cell (= degree_x + 1) */
  PetscInt NbV;         /* basis functions per v-cell (= degree_v + 1) */
  PetscInt NqV;         /* quadrature points for v-advection */

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

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  /* Persistent device copies of geometry (allocated once in PrecomputeGeometry) */
  RealView1D d_xi_v_nodes;   /* [NbV] */
  RealView1D d_xi_x_nodes;   /* [NbX] */
  RealView1D d_v_dof_coords; /* [NvDOF] */
  RealView1D d_v_basis_int;  /* [NvDOF] */
  RealView1D d_x_basis_int;  /* [NbX] */
  RealView1D d_tabV;         /* [NqV * NbV] */
  RealView1D d_vq_pts;       /* [NqV] */
  RealView1D d_vq_wts;       /* [NqV] */
  RealView1D d_A_sldg;       /* [NvDOF * NbX * NbX] */
  RealView1D d_B_sldg;       /* [NvDOF * NbX * NbX] */
  IntView1D  d_n_shift;      /* [NvDOF] */
  /* Ghost-extended f buffer for AdvectX (allocated once, reused each iv) */
  RealView1D d_fiv_ghost; /* [(NxLocal + 2*sw) * NbX] */
#endif
} AppCtx;

/* ========================================================================
   Lagrange basis helper
   ======================================================================== */
#if !defined(KOKKOS_INLINE_FUNCTION)
  #define KOKKOS_INLINE_FUNCTION inline
#endif
KOKKOS_INLINE_FUNCTION static PetscReal EvalLagrangeBasis(PetscInt nb, const PetscReal *nodes, PetscInt b, PetscReal xi)
{
  PetscReal val = 1.0;
  for (PetscInt j = 0; j < nb; ++j)
    if (j != b) val *= (xi - nodes[j]) / (nodes[b] - nodes[j]);
  return val;
}

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
/* ========================================================================
   Device-callable Gauss-Legendre quadrature helpers
   Ported from src/dm/dt/interface/dt.c (PetscDTGaussJacobiQuadrature_Newton_Internal
   and helpers).  All functions are KOKKOS_INLINE_FUNCTION so they can be
   called from device kernels.  No PETSc error codes -- pure arithmetic.

   For Gauss-Legendre (alpha=beta=0) the weight prefactor a6 = 2 exactly,
   so lgamma is not needed.
   ======================================================================== */

/* Three-term recurrence coefficients for J^{a,b}_n:
   J^{a,b}_n(x) = (cnm1 + cnm1x*x)*J^{a,b}_{n-1}(x) - cnm2*J^{a,b}_{n-2}(x) */
KOKKOS_INLINE_FUNCTION static void KokkosDTJacobiRecurrence(int n, double a, double b, double &cnm1, double &cnm1x, double &cnm2)
{
  if (n == 1) {
    cnm1  = (a - b) * 0.5;
    cnm1x = (a + b + 2.) * 0.5;
    cnm2  = 0.;
  } else {
    double _2n  = n + n;
    double _d   = _2n * (n + a + b) * (_2n + a + b - 2.);
    double _n1  = (_2n + a + b - 1.) * (a * a - b * b);
    double _n1x = (_2n + a + b - 1.) * (_2n + a + b) * (_2n + a + b - 2.);
    double _n2  = 2. * (n + a - 1.) * (n + b - 1.) * (_2n + a + b);
    cnm1        = _n1 / _d;
    cnm1x       = _n1x / _d;
    cnm2        = _n2 / _d;
  }
}

/* Evaluate Jacobi polynomial P_n^{a,b}(x) via three-term recurrence */
KOKKOS_INLINE_FUNCTION static double KokkosDTComputeJacobi(double a, double b, int n, double x)
{
  if (n == 0) return 1.0;
  double cnm1, cnm1x, cnm2;
  KokkosDTJacobiRecurrence(1, a, b, cnm1, cnm1x, cnm2);
  double pn2 = 1., pn1 = cnm1 + cnm1x * x, P = 0.;
  if (n == 1) return pn1;
  for (int k = 2; k <= n; ++k) {
    KokkosDTJacobiRecurrence(k, a, b, cnm1, cnm1x, cnm2);
    P   = (cnm1 + cnm1x * x) * pn1 - cnm2 * pn2;
    pn2 = pn1;
    pn1 = P;
  }
  return P;
}

/* Evaluate k-th derivative of P_n^{a,b}(x) */
KOKKOS_INLINE_FUNCTION static double KokkosDTComputeJacobiDerivative(double a, double b, int n, double x, int k)
{
  if (k > n) return 0.0;
  double nP = KokkosDTComputeJacobi(a + k, b + k, n - k, x);
  for (int i = 0; i < k; ++i) nP *= (a + b + n + 1. + i) * 0.5;
  return nP;
}

/* Compute npoints Gauss-Legendre quadrature points/weights on [a,b].
   x[] and w[] must be caller-allocated arrays of length npoints.
   Uses Newton iteration; for GL alpha=beta=0 so a6=2 exactly.
   NqSLDG <= 10 for degree_x <= 3, so stack allocation is safe. */
KOKKOS_INLINE_FUNCTION static void KokkosDTGaussQuadrature(int npoints, double a, double b, double *x, double *w)
{
  const int maxIter = 100;
  /* eps = exp(0.75 * log(machine_epsilon_double)) ~ 1.2e-11 */
  const double eps = 1.2e-11;
  /* For Gauss-Legendre (alpha=beta=0): a6 = 2^1 * Gamma(n+1)^2 / (Gamma(2n+1)*Gamma(1)) = 2 */
  const double a6 = 2.0;

  for (int k = 0; k < npoints; ++k) {
    /* Chebyshev initial guess (Karniadakis & Sherwin, alpha=beta=0) */
    double r = PetscCosReal(PETSC_PI * (1. - (4. * k + 3.) / (4. * npoints + 2.)));
    if (k > 0) r = 0.5 * (r + x[k - 1]);
    for (int j = 0; j < maxIter; ++j) {
      double s = 0.0;
      for (int i = 0; i < k; ++i) s += 1.0 / (r - x[i]);
      double f     = KokkosDTComputeJacobi(0., 0., npoints, r);
      double fp    = KokkosDTComputeJacobiDerivative(0., 0., npoints, r, 1);
      double delta = f / (fp - f * s);
      r -= delta;
      if (PetscAbsReal(delta) < eps) break;
    }
    x[k]      = r;
    double dP = KokkosDTComputeJacobiDerivative(0., 0., npoints, x[k], 1);
    w[k]      = a6 / (1.0 - x[k] * x[k]) / (dP * dP);
  }
  /* Symmetrize (alpha == beta case) */
  for (int i = 0; i < (npoints + 1) / 2; ++i) {
    int    j  = npoints - 1 - i;
    double xi = x[i], xj = x[j], wi = w[i], wj = w[j];
    x[i] = (xi - xj) * 0.5;
    x[j] = (xj - xi) * 0.5;
    w[i] = w[j] = (wi + wj) * 0.5;
  }
  /* Shift from [-1,1] to [a,b] */
  if (a != -1. || b != 1.)
    for (int i = 0; i < npoints; ++i) {
      x[i] = (x[i] + 1.) * ((b - a) * 0.5) + a;
      w[i] *= (b - a) * 0.5;
    }
}
#endif /* PETSC_HAVE_KOKKOS_KERNELS */

/* ========================================================================
   Phase 1 -- ProcessOptions
   ======================================================================== */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscFunctionBeginUser;
  /* Null-initialize all pointers */
  ctx->daX         = NULL;
  ctx->daV         = NULL;
  ctx->daPot       = NULL;
  ctx->daF         = NULL;
  ctx->f           = NULL;
  ctx->rho         = NULL;
  ctx->phi         = NULL;
  ctx->E_field     = NULL;
  ctx->f_iv_global = NULL;
  ctx->f_iv_local  = NULL;
  ctx->rhs_poisson = NULL;
  ctx->kspPoisson  = NULL;
  ctx->Jac         = NULL;

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

  /* Defaults */
  ctx->Nx           = 64;
  ctx->Nv           = 128;
  ctx->v_max        = 6.0;
  ctx->alpha        = 0.01;
  ctx->kwave        = 0.5;
  ctx->sigma        = 1.0;
  ctx->dt           = 0.1;
  ctx->steps        = 200;
  ctx->ostep        = 10;
  ctx->degree_x     = 1;
  ctx->degree_v     = 1;
  ctx->check_landau = PETSC_FALSE;

  /* Derived/deferred integer fields -- zero until set by later phases.
     AppCtx is stack-allocated (not zero-initialized), so these must be
     set explicitly here to avoid uninitialized-memory bugs on platforms
     (e.g. Frontier/HIP) where the stack is not zeroed.  In particular,
     NqV is read by the PrecomputeGeometry guard at the first d_tabV
     allocation before NqV is assigned its real value (NbV+2). */
  ctx->NqV         = 0;
  ctx->NxLocal     = 0;
  ctx->NxDOF_local = 0;
  ctx->NvDOF       = 0;
  ctx->xs          = 0;

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
  PetscCall(PetscLogEventRegister("AdvectX", TS_CLASSID, &ctx->AdvectXEvent));
  PetscCall(PetscLogEventRegister("AdvectV", TS_CLASSID, &ctx->AdvectVEvent));
  PetscCall(PetscLogEventRegister("Poisson", TS_CLASSID, &ctx->PoissonEvent));
  PetscCall(PetscLogEventRegister("ChargeDensity", TS_CLASSID, &ctx->RhoEvent));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 2 -- CreateXMesh + CreateVMesh
   ======================================================================== */
static PetscErrorCode CreateXMesh(MPI_Comm comm, AppCtx *ctx)
{
  PetscFunctionBeginUser;
  /* 1D periodic DMDA for x-space.
     dof = NbX: one DG coefficient per basis function per cell.
     stencil width = ceil(v_max * (dt/2) / h_x) + 1 for SLDG ghost cells.
     The maximum shift per half-step is v_max * (dt/2) / h_x. */
  PetscInt sw = (PetscInt)(ctx->v_max * ctx->dt * 0.5 / ctx->h_x) + 2;
  PetscCall(DMDACreate1d(comm, DM_BOUNDARY_PERIODIC, ctx->Nx, ctx->NbX, sw, NULL, &ctx->daX));
  PetscCall(DMSetFromOptions(ctx->daX));
  PetscCall(DMSetUp(ctx->daX));
  PetscCall(DMDAGetCorners(ctx->daX, &ctx->xs, NULL, NULL, &ctx->NxLocal, NULL, NULL));
  ctx->NxDOF_local = ctx->NxLocal * ctx->NbX;

  /* daF: same topology as daX, used for per-iv ghost exchange in AdvectX.
     dof=NbX and stencil width sw are identical to daX. */
  PetscCall(DMDACreate1d(comm, DM_BOUNDARY_PERIODIC, ctx->Nx, ctx->NbX, sw, NULL, &ctx->daF));
  PetscCall(DMSetFromOptions(ctx->daF));
  PetscCall(DMSetUp(ctx->daF));
  /* Pre-allocate per-iv slice Vecs for AdvectX ghost exchange (reused every step) */
  PetscCall(DMCreateGlobalVector(ctx->daF, &ctx->f_iv_global));
  PetscCall(DMGetLocalVector(ctx->daF, &ctx->f_iv_local));
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
   Phase 3 -- SetupPoisson
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
  PetscCall(DMSetFromOptions(ctx->daPot));
  PetscCall(DMSetUp(ctx->daPot));

  /* Create global Vecs from daPot */
  PetscCall(DMCreateGlobalVector(ctx->daPot, &ctx->rho));
  PetscCall(DMCreateGlobalVector(ctx->daPot, &ctx->phi));
  PetscCall(DMCreateGlobalVector(ctx->daPot, &ctx->E_field));
  PetscCall(PetscObjectSetName((PetscObject)ctx->rho, "rho"));
  PetscCall(PetscObjectSetName((PetscObject)ctx->phi, "phi"));
  PetscCall(PetscObjectSetName((PetscObject)ctx->E_field, "E_field"));
  /* Pre-allocate reusable RHS Vec for Poisson solve */
  PetscCall(DMCreateGlobalVector(ctx->daPot, &ctx->rhs_poisson));
  PetscCall(PetscObjectSetName((PetscObject)ctx->rhs_poisson, "rhs_poisson"));

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
      row.i = i;
      row.c = 0;

      /* For periodic DMDA, MatSetValuesStencil handles wrap-around automatically.
         Use i-1 and i+1 directly -- no manual modular arithmetic needed. */
      col[0].i = i;
      col[0].c = 0;
      col[1].i = i - 1;
      col[1].c = 0;
      col[2].i = i + 1;
      col[2].c = 0;

      vals[0] = 2.0 * h2inv;
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
   Phase 4 -- PrecomputeGeometry
   ======================================================================== */
static PetscErrorCode PrecomputeGeometry(AppCtx *ctx)
{
  PetscReal *gll_wts_x, *gll_wts_v;

  PetscFunctionBeginUser;
#if defined(PETSC_HAVE_KOKKOS)
  PetscCall(PetscKokkosInitializeCheck());
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  /* Allocate all persistent device Views up front so computation steps can write
     directly into them, avoiding a redundant device->host->device round-trip. */
  ctx->d_xi_v_nodes   = RealView1D("d_xi_v_nodes", ctx->NbV);
  ctx->d_xi_x_nodes   = RealView1D("d_xi_x_nodes", ctx->NbX);
  ctx->d_v_dof_coords = RealView1D("d_v_dof_coords", ctx->NvDOF);
  ctx->d_v_basis_int  = RealView1D("d_v_basis_int", ctx->NvDOF);
  ctx->d_x_basis_int  = RealView1D("d_x_basis_int", ctx->NbX);
  ctx->d_tabV         = RealView1D("d_tabV", ctx->NqV > 0 ? ctx->NqV * ctx->NbV : 1);
  ctx->d_vq_pts       = RealView1D("d_vq_pts", ctx->NqV > 0 ? ctx->NqV : 1);
  ctx->d_vq_wts       = RealView1D("d_vq_wts", ctx->NqV > 0 ? ctx->NqV : 1);
  ctx->d_A_sldg       = RealView1D("d_A_sldg", ctx->NvDOF * ctx->NbX * ctx->NbX);
  ctx->d_B_sldg       = RealView1D("d_B_sldg", ctx->NvDOF * ctx->NbX * ctx->NbX);
  ctx->d_n_shift      = IntView1D("d_n_shift", ctx->NvDOF);
  {
    PetscInt sw;
    PetscCall(DMDAGetInfo(ctx->daF, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &sw, NULL, NULL, NULL, NULL));
    ctx->d_fiv_ghost = RealView1D("d_fiv_ghost", (ctx->NxLocal + 2 * sw) * ctx->NbX);
  }
#endif
  /* Step 1: GLL nodes for x and v DOFs */
  PetscCall(PetscMalloc1(ctx->NbX, &ctx->xi_x_nodes));
  PetscCall(PetscMalloc1(ctx->NbX, &gll_wts_x));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(ctx->NbX, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, ctx->xi_x_nodes, gll_wts_x));

  PetscCall(PetscMalloc1(ctx->NbV, &ctx->xi_v_nodes));
  PetscCall(PetscMalloc1(ctx->NbV, &gll_wts_v));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(ctx->NbV, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, ctx->xi_v_nodes, gll_wts_v));

  /* Step 2: Physical v-DOF coordinates */
  PetscCall(PetscMalloc1(ctx->NvDOF, &ctx->v_dof_coords));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  {
    /* Write directly into persistent ctx->d_v_dof_coords; deep_copy back to host. */
    Kokkos::deep_copy(ctx->d_xi_v_nodes, Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->xi_v_nodes, ctx->NbV));
    RealView1D      d_xi_v = ctx->d_xi_v_nodes;
    RealView1D      d_vdof = ctx->d_v_dof_coords;
    const PetscReal v_max = ctx->v_max, h_v = ctx->h_v;
    const PetscInt  NbV = ctx->NbV, NvDOF = ctx->NvDOF;
    Kokkos::parallel_for(
      "v_dof_coords", Kokkos::RangePolicy<>(0, NvDOF), KOKKOS_LAMBDA(PetscInt idx) {
        PetscInt cv = idx / NbV, bv = idx % NbV;
        d_vdof(idx) = -v_max + cv * h_v + (d_xi_v(bv) + 1.0) * (h_v * 0.5);
      });
    Kokkos::deep_copy(Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->v_dof_coords, ctx->NvDOF), d_vdof);
  }
#else
  for (PetscInt cv = 0; cv < ctx->Nv; ++cv)
    for (PetscInt bv = 0; bv < ctx->NbV; ++bv) ctx->v_dof_coords[cv * ctx->NbV + bv] = -ctx->v_max + cv * ctx->h_v + (ctx->xi_v_nodes[bv] + 1.0) * (ctx->h_v * 0.5);
#endif

  /* Step 3: Gauss quadrature for v-advection (NqV = NbV + 2 points) */
  ctx->NqV = ctx->NbV + 2;
  PetscCall(PetscMalloc1(ctx->NqV, &ctx->vq_pts));
  PetscCall(PetscMalloc1(ctx->NqV, &ctx->vq_wts));
  PetscCall(PetscDTGaussQuadrature(ctx->NqV, -1.0, 1.0, ctx->vq_pts, ctx->vq_wts));

  /* Step 4: Tabulate v-basis at quadrature points */
  PetscCall(PetscMalloc1(ctx->NqV * ctx->NbV, &ctx->tabV));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  {
    /* Resize persistent Views now that NqV is known (was allocated with NqV>0 guard above). */
    ctx->d_tabV   = RealView1D("d_tabV", ctx->NqV * ctx->NbV);
    ctx->d_vq_pts = RealView1D("d_vq_pts", ctx->NqV);
    ctx->d_vq_wts = RealView1D("d_vq_wts", ctx->NqV);
    /* Upload vq_pts/vq_wts (host-computed) and xi_v_nodes to device. */
    Kokkos::deep_copy(ctx->d_vq_pts, Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->vq_pts, ctx->NqV));
    Kokkos::deep_copy(ctx->d_vq_wts, Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->vq_wts, ctx->NqV));
    /* Write directly into persistent ctx->d_tabV; deep_copy back to host. */
    RealView1D     d_xi_v = ctx->d_xi_v_nodes; /* already uploaded in Step 2 */
    RealView1D     d_vq   = ctx->d_vq_pts;
    RealView1D     d_tab  = ctx->d_tabV;
    const PetscInt NbV = ctx->NbV, NqV = ctx->NqV;
    Kokkos::parallel_for(
      "tabV", Kokkos::RangePolicy<>(0, NqV * NbV), KOKKOS_LAMBDA(PetscInt idx) {
        PetscInt q = idx / NbV, bv = idx % NbV;
        d_tab(idx) = EvalLagrangeBasis(NbV, d_xi_v.data(), bv, d_vq(q));
      });
    Kokkos::deep_copy(Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->tabV, ctx->NqV * ctx->NbV), d_tab);
  }
#else
  for (PetscInt q = 0; q < ctx->NqV; ++q)
    for (PetscInt bv = 0; bv < ctx->NbV; ++bv) ctx->tabV[q * ctx->NbV + bv] = EvalLagrangeBasis(ctx->NbV, ctx->xi_v_nodes, bv, ctx->vq_pts[q]);
#endif

  /* Step 5: v-basis integrals for charge density computation */
  PetscCall(PetscCalloc1(ctx->NvDOF, &ctx->v_basis_int));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  {
    /* Use already-on-device ctx->d_tabV and ctx->d_vq_wts; write into ctx->d_v_basis_int. */
    RealView1D      d_tab = ctx->d_tabV;
    RealView1D      d_wts = ctx->d_vq_wts;
    RealView1D      d_vbi = ctx->d_v_basis_int;
    const PetscInt  NbV = ctx->NbV, NqV = ctx->NqV, NvDOF = ctx->NvDOF;
    const PetscReal h_v = ctx->h_v;
    Kokkos::parallel_for(
      "v_basis_int", Kokkos::RangePolicy<>(0, NvDOF), KOKKOS_LAMBDA(PetscInt idx) {
        PetscInt  bv  = idx % NbV;
        PetscReal sum = 0.0;
        for (PetscInt q = 0; q < NqV; ++q) sum += d_tab(q * NbV + bv) * d_wts(q) * (h_v * 0.5);
        d_vbi(idx) = sum;
      });
    Kokkos::deep_copy(Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->v_basis_int, ctx->NvDOF), d_vbi);
  }
#else
  for (PetscInt cv = 0; cv < ctx->Nv; ++cv) {
    for (PetscInt bv = 0; bv < ctx->NbV; ++bv) {
      PetscReal sum = 0.0;
      for (PetscInt q = 0; q < ctx->NqV; ++q) sum += ctx->tabV[q * ctx->NbV + bv] * ctx->vq_wts[q] * (ctx->h_v * 0.5);
      ctx->v_basis_int[cv * ctx->NbV + bv] = sum;
    }
  }
#endif

  /* Step 6: x-basis integrals for charge density and moments
     x_basis_int[bx] = gll_wts_x[bx] * h_x/2 (integral of x-basis over cell) */
  PetscCall(PetscMalloc1(ctx->NbX, &ctx->x_basis_int));
  for (PetscInt bx = 0; bx < ctx->NbX; ++bx) ctx->x_basis_int[bx] = gll_wts_x[bx] * (ctx->h_x * 0.5);

  /* Step 7 (removed): v-mass matrix inverse was computed here but is never used.
     The nodal AdvectV update does not require MassV_inv. */

  /* Step 8: Precompute SLDG overlap matrices A_sldg, B_sldg, and integer shifts n_shift.
     For each velocity DOF iv, compute the fractional shift s = v_dof_coords[iv]*dt/h_x,
     decompose into integer n and fractional alp in [0,1), then build the NbX x NbX
     overlap matrices A and B via Gauss quadrature on [-1, split] and [split, 1].
     NqSLDG = 2*NbX+2 quadrature points; NbX <= 4 so NqSLDG <= 10 -- safe for stack.

     Convention: A_sldg and B_sldg store A[i,j]/w_i and B[i,j]/w_i where
     w_i = gll_wts_x[i] is the GLL weight (= x_basis_int[i] / (h_x/2)).
     This way the apply step is simply f_new[i] = sum_j (A[i,j]*f_A[j] + B[i,j]*f_B[j])
     with no division, and the pure-integer-shift case gives A[i,i]/w_i = w_i/w_i = 1. */
  PetscCall(PetscMalloc1(ctx->NvDOF * ctx->NbX * ctx->NbX, &ctx->A_sldg));
  PetscCall(PetscMalloc1(ctx->NvDOF * ctx->NbX * ctx->NbX, &ctx->B_sldg));
  PetscCall(PetscMalloc1(ctx->NvDOF, &ctx->n_shift));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  {
    const PetscInt  NvDOF  = ctx->NvDOF;
    const PetscInt  NbX    = ctx->NbX;
    const PetscInt  NqSLDG = 2 * NbX + 2;
    const PetscReal h_x    = ctx->h_x;
    const PetscReal dt     = ctx->dt;
    /* Upload xi_x_nodes and x_basis_int (host-computed) to device. */
    Kokkos::deep_copy(ctx->d_xi_x_nodes, Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->xi_x_nodes, NbX));
    Kokkos::deep_copy(ctx->d_x_basis_int, Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->x_basis_int, NbX));
    /* Use already-on-device ctx->d_v_dof_coords; write directly into persistent ctx->d_A/B/n_shift. */
    RealView1D d_vdc  = ctx->d_v_dof_coords;
    RealView1D d_xi_x = ctx->d_xi_x_nodes;
    RealView1D d_xbi  = ctx->d_x_basis_int;
    RealView1D d_A    = ctx->d_A_sldg;
    RealView1D d_B    = ctx->d_B_sldg;
    IntView1D  d_ns   = ctx->d_n_shift;
    Kokkos::parallel_for(
      "sldg_precompute", Kokkos::RangePolicy<>(0, NvDOF), KOKKOS_LAMBDA(PetscInt iv) {
        PetscReal s   = d_vdc(iv) * (dt * 0.5) / h_x;
        PetscInt  n   = (PetscInt)PetscFloorReal(s);
        PetscReal alp = s - (PetscReal)n;
        if (alp > 1.0 - 1e-14) {
          n++;
          alp = 0.0;
        }
        d_ns(iv) = n;

        PetscReal *A = &d_A(iv * NbX * NbX);
        PetscReal *B = &d_B(iv * NbX * NbX);
        for (PetscInt k = 0; k < NbX * NbX; ++k) A[k] = B[k] = 0.0;

        if (alp < 1e-14) {
          /* Pure integer shift: A[i,i] = w_i/w_i = 1, B = 0 */
          for (PetscInt i = 0; i < NbX; ++i) A[i * NbX + i] = 1.0;
        } else {
          PetscReal split = 1.0 - 2.0 * alp;
          /* Stack-allocate quadrature arrays (NqSLDG <= 10) */
          PetscReal qa[10], wa[10], qb[10], wb[10];
          KokkosDTGaussQuadrature(NqSLDG, -1.0, split, qa, wa);
          KokkosDTGaussQuadrature(NqSLDG, split, 1.0, qb, wb);

          /* Build raw overlap integrals, then divide each row i by w_i */
          for (PetscInt q = 0; q < NqSLDG; ++q) {
            PetscReal xi = qa[q], xi2 = xi + 2.0 * alp;
            for (PetscInt i = 0; i < NbX; ++i) {
              PetscReal phi_i = EvalLagrangeBasis(NbX, d_xi_x.data(), i, xi);
              for (PetscInt j = 0; j < NbX; ++j) A[i * NbX + j] += phi_i * EvalLagrangeBasis(NbX, d_xi_x.data(), j, xi2) * wa[q];
            }
          }
          for (PetscInt q = 0; q < NqSLDG; ++q) {
            PetscReal xi = qb[q], xi2 = xi + 2.0 * alp - 2.0;
            for (PetscInt i = 0; i < NbX; ++i) {
              PetscReal phi_i = EvalLagrangeBasis(NbX, d_xi_x.data(), i, xi);
              for (PetscInt j = 0; j < NbX; ++j) B[i * NbX + j] += phi_i * EvalLagrangeBasis(NbX, d_xi_x.data(), j, xi2) * wb[q];
            }
          }
          /* Divide each row i by w_i = x_basis_int[i] / (h_x/2) */
          for (PetscInt i = 0; i < NbX; ++i) {
            PetscReal inv_wi = (h_x * 0.5) / d_xbi(i);
            for (PetscInt j = 0; j < NbX; ++j) {
              A[i * NbX + j] *= inv_wi;
              B[i * NbX + j] *= inv_wi;
            }
          }
        }
      });
    Kokkos::deep_copy(Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->A_sldg, NvDOF * NbX * NbX), d_A);
    Kokkos::deep_copy(Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->B_sldg, NvDOF * NbX * NbX), d_B);
    Kokkos::deep_copy(Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ctx->n_shift, NvDOF), d_ns);
  }
#else
  {
    PetscInt   NqSLDG = 2 * ctx->NbX + 2;
    PetscReal *qa, *wa, *qb, *wb;
    PetscCall(PetscMalloc1(NqSLDG, &qa));
    PetscCall(PetscMalloc1(NqSLDG, &wa));
    PetscCall(PetscMalloc1(NqSLDG, &qb));
    PetscCall(PetscMalloc1(NqSLDG, &wb));
    for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
      PetscReal s   = ctx->v_dof_coords[iv] * (ctx->dt * 0.5) / ctx->h_x;
      PetscInt  n   = (PetscInt)PetscFloorReal(s);
      PetscReal alp = s - (PetscReal)n;
      if (alp > 1.0 - 1e-14) {
        n++;
        alp = 0.0;
      }
      ctx->n_shift[iv] = n;
      PetscReal *A     = &ctx->A_sldg[iv * ctx->NbX * ctx->NbX];
      PetscReal *B     = &ctx->B_sldg[iv * ctx->NbX * ctx->NbX];
      for (PetscInt k = 0; k < ctx->NbX * ctx->NbX; ++k) A[k] = B[k] = 0.0;
      if (alp < 1e-14) {
        for (PetscInt i = 0; i < ctx->NbX; ++i) A[i * ctx->NbX + i] = 1.0;
      } else {
        PetscReal split = 1.0 - 2.0 * alp;
        PetscCall(PetscDTGaussQuadrature(NqSLDG, -1.0, split, qa, wa));
        PetscCall(PetscDTGaussQuadrature(NqSLDG, split, 1.0, qb, wb));

        for (PetscInt q = 0; q < NqSLDG; ++q) {
          PetscReal xi = qa[q], xi2 = xi + 2.0 * alp;
          for (PetscInt i = 0; i < ctx->NbX; ++i) {
            PetscReal phi_i = EvalLagrangeBasis(ctx->NbX, ctx->xi_x_nodes, i, xi);
            for (PetscInt j = 0; j < ctx->NbX; ++j) A[i * ctx->NbX + j] += phi_i * EvalLagrangeBasis(ctx->NbX, ctx->xi_x_nodes, j, xi2) * wa[q];
          }
        }
        for (PetscInt q = 0; q < NqSLDG; ++q) {
          PetscReal xi = qb[q], xi2 = xi + 2.0 * alp - 2.0;
          for (PetscInt i = 0; i < ctx->NbX; ++i) {
            PetscReal phi_i = EvalLagrangeBasis(ctx->NbX, ctx->xi_x_nodes, i, xi);
            for (PetscInt j = 0; j < ctx->NbX; ++j) B[i * ctx->NbX + j] += phi_i * EvalLagrangeBasis(ctx->NbX, ctx->xi_x_nodes, j, xi2) * wb[q];
          }
        }
        /* Divide each row i by w_i = x_basis_int[i] / (h_x/2) */
        for (PetscInt i = 0; i < ctx->NbX; ++i) {
          PetscReal inv_wi = (ctx->h_x * 0.5) / ctx->x_basis_int[i];
          for (PetscInt j = 0; j < ctx->NbX; ++j) {
            A[i * ctx->NbX + j] *= inv_wi;
            B[i * ctx->NbX + j] *= inv_wi;
          }
        }
      }
    }
    PetscCall(PetscFree(qa));
    PetscCall(PetscFree(wa));
    PetscCall(PetscFree(qb));
    PetscCall(PetscFree(wb));
  }
#endif

  PetscCall(PetscFree(gll_wts_x));
  PetscCall(PetscFree(gll_wts_v));
  /* All persistent device Views were allocated and populated in the steps above.
     No redundant host->device upload needed here. */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 5 -- AllocateF + InitializeF
   ======================================================================== */
static PetscErrorCode AllocateF(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  /* f is a global Vec with local size NvDOF * NxDOF_local */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ctx->f));
  PetscCall(VecSetSizes(ctx->f, ctx->NvDOF * ctx->NxDOF_local, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(ctx->f));
  PetscCall(PetscObjectSetName((PetscObject)ctx->f, "f_dist"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeF(AppCtx *ctx)
{
  PetscScalar *f_ptr;
  PetscMemType mtype;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayAndMemType(ctx->f, &f_ptr, &mtype));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  if (mtype == PETSC_MEMTYPE_DEVICE) {
    const PetscInt  NvDOF     = ctx->NvDOF;
    const PetscInt  NxDOF_loc = ctx->NxDOF_local;
    const PetscInt  NbX       = ctx->NbX;
    const PetscInt  xs        = ctx->xs;
    const PetscReal h_x       = ctx->h_x;
    const PetscReal alpha     = ctx->alpha;
    const PetscReal kwave     = ctx->kwave;
    ScalarView1D    d_f(f_ptr, (size_t)NvDOF * NxDOF_loc);
    RealView1D      d_vdc  = ctx->d_v_dof_coords;
    RealView1D      d_xi_x = ctx->d_xi_x_nodes;
    Kokkos::parallel_for(
      "InitializeF", Kokkos::RangePolicy<>(0, NvDOF * NxDOF_loc), KOKKOS_LAMBDA(PetscInt idx) {
        PetscInt  iv          = idx / NxDOF_loc;
        PetscInt  ix          = idx % NxDOF_loc;
        PetscReal v           = d_vdc(iv);
        PetscInt  cx_local    = ix / NbX;
        PetscInt  bx          = ix % NbX;
        PetscReal x_cell_left = (xs + cx_local) * h_x;
        PetscReal x           = x_cell_left + (d_xi_x(bx) + 1.0) * (h_x * 0.5);
        PetscReal f0          = (1.0 + alpha * PetscCosReal(kwave * x)) / PetscSqrtReal(2.0 * PETSC_PI) * PetscExpReal(-0.5 * v * v);
        d_f(idx)              = (PetscScalar)f0;
      });
  } else
#endif
  {
    for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
      PetscReal v = ctx->v_dof_coords[iv];
      for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix) {
        PetscInt  cx_local                = ix / ctx->NbX;
        PetscInt  bx                      = ix % ctx->NbX;
        PetscReal x_cell_left             = (ctx->xs + cx_local) * ctx->h_x;
        PetscReal x                       = x_cell_left + (ctx->xi_x_nodes[bx] + 1.0) * (ctx->h_x * 0.5);
        PetscReal f0                      = (1.0 + ctx->alpha * PetscCosReal(ctx->kwave * x)) / PetscSqrtReal(2.0 * PETSC_PI) * PetscExpReal(-0.5 * v * v);
        f_ptr[iv * ctx->NxDOF_local + ix] = (PetscScalar)f0;
      }
    }
  }
  PetscCall(VecRestoreArrayAndMemType(ctx->f, &f_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 6 -- ComputeChargeDensity
   ======================================================================== */
static PetscErrorCode ComputeChargeDensity(AppCtx *ctx)
{
  PetscScalar *f_ptr, *rho_ptr;
  PetscMemType mtype_f, mtype_rho;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->RhoEvent, 0, 0, 0, 0));

  /* rho[cx_local] = (1/h_x) * sum_{iv} sum_{bx} f[iv*NxDOF_local + cx_local*NbX + bx]
                     * v_basis_int[iv] * gll_wts_x[bx] * h_x/2
     The factor 1/h_x * h_x/2 = 1/2 cancels with the x-basis integral normalization.
     x_basis_int[bx] = gll_wts_x[bx] * h_x/2 (precomputed in PrecomputeGeometry).
     rho = (1/h_x) * integral_x integral_v f dv dx
         = sum_{iv} v_basis_int[iv] * sum_{bx} f[iv,bx] * gll_wts_x[bx] * (h_x/2) / h_x
         = sum_{iv} v_basis_int[iv] * sum_{bx} f[iv,bx] * gll_wts_x[bx] / 2 */

  PetscCall(VecGetArrayAndMemType(ctx->f, &f_ptr, &mtype_f));
  PetscCall(VecGetArrayAndMemType(ctx->rho, &rho_ptr, &mtype_rho));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  if (mtype_f == PETSC_MEMTYPE_DEVICE) {
    const PetscInt  NxLocal   = ctx->NxLocal;
    const PetscInt  NvDOF     = ctx->NvDOF;
    const PetscInt  NbX       = ctx->NbX;
    const PetscInt  NxDOF_loc = ctx->NxDOF_local;
    const PetscReal h_x       = ctx->h_x;
    ScalarView1D    d_f(f_ptr, (size_t)NvDOF * NxDOF_loc);
    ScalarView1D    d_rho(rho_ptr, NxLocal);
    RealView1D      d_vbi = ctx->d_v_basis_int;
    RealView1D      d_xbi = ctx->d_x_basis_int;
    Kokkos::parallel_for(
      "ComputeChargeDensity", Kokkos::RangePolicy<>(0, NxLocal), KOKKOS_LAMBDA(PetscInt cx_local) {
        PetscReal rho_val = 0.0;
        for (PetscInt iv = 0; iv < NvDOF; ++iv) {
          PetscReal w = d_vbi(iv);
          for (PetscInt bx = 0; bx < NbX; ++bx) rho_val += PetscRealPart(d_f(iv * NxDOF_loc + cx_local * NbX + bx)) * w * d_xbi(bx);
        }
        d_rho(cx_local) = (PetscScalar)(rho_val / h_x);
      });
  } else
#endif
  {
    for (PetscInt cx_local = 0; cx_local < ctx->NxLocal; ++cx_local) {
      PetscReal rho_val = 0.0;
      for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
        PetscReal w = ctx->v_basis_int[iv];
        for (PetscInt bx = 0; bx < ctx->NbX; ++bx) {
          PetscReal fval = PetscRealPart(f_ptr[iv * ctx->NxDOF_local + cx_local * ctx->NbX + bx]);
          /* x_basis_int[bx] = gll_wts_x[bx] * h_x/2, divide by h_x to get cell average */
          rho_val += fval * w * ctx->x_basis_int[bx];
        }
      }
      rho_ptr[cx_local] = (PetscScalar)(rho_val / ctx->h_x);
    }
  }
  PetscCall(VecRestoreArrayAndMemType(ctx->f, &f_ptr));
  PetscCall(VecRestoreArrayAndMemType(ctx->rho, &rho_ptr));

  PetscCall(PetscLogEventEnd(ctx->RhoEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 7 -- SolvePoisson + ComputeElectricField
   ======================================================================== */
static PetscErrorCode SolvePoisson(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->PoissonEvent, 0, 0, 0, 0));

  /* Solve -phi'' = rho - sigma
     RHS = rho - sigma (subtract background ion density).
     Use pre-allocated rhs_poisson to avoid VecDuplicate/VecDestroy each call. */
  PetscCall(VecCopy(ctx->rho, ctx->rhs_poisson));
  PetscCall(VecShift(ctx->rhs_poisson, -(PetscScalar)ctx->sigma));

  PetscCall(VecZeroEntries(ctx->phi));
  PetscCall(KSPSolve(ctx->kspPoisson, ctx->rhs_poisson, ctx->phi));

  PetscCall(PetscLogEventEnd(ctx->PoissonEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeElectricField(AppCtx *ctx)
{
  /* E = -dphi/dx using central finite differences on the periodic DMDA.
     E[i] = -(phi[i+1] - phi[i-1]) / (2 * h_x)
     phi_local: ghost-aware local vector, indexed by global cell index.
     E_field:   global Vec, accessed via VecGetArrayAndMemType (device pointer). */
  DMDALocalInfo info;
  Vec           phi_local;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(ctx->daPot, &info));

  PetscCall(DMGetLocalVector(ctx->daPot, &phi_local));
  PetscCall(DMGlobalToLocal(ctx->daPot, ctx->phi, INSERT_VALUES, phi_local));

  {
    PetscScalar    *phi_ptr, *E_ptr;
    PetscMemType    mtype_phi, mtype_E;
    const PetscInt  sw     = info.sw;
    const PetscReal inv2hx = 1.0 / (2.0 * ctx->h_x);
    PetscCall(VecGetArrayAndMemType(phi_local, &phi_ptr, &mtype_phi));
    PetscCall(VecGetArrayAndMemType(ctx->E_field, &E_ptr, &mtype_E));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    if (mtype_phi == PETSC_MEMTYPE_DEVICE) {
      /* phi_local is a local Vec: its flat array is [ghost_left | owned | ghost_right].
         Length = n_phi = NxLocal + 2*sw.  Element [sw + cx_local] is the owned cell. */
      const PetscInt NxLocal = info.xm;
      const PetscInt n_phi   = NxLocal + 2 * sw;
      ScalarView1D   d_phi(phi_ptr, n_phi);
      ScalarView1D   d_E(E_ptr, NxLocal);
      Kokkos::parallel_for("ComputeElectricField", Kokkos::RangePolicy<>(0, NxLocal), KOKKOS_LAMBDA(PetscInt cx_local) { d_E(cx_local) = -(PetscScalar)((PetscRealPart(d_phi(sw + cx_local + 1)) - PetscRealPart(d_phi(sw + cx_local - 1))) * inv2hx); });
    } else
#endif
    {
      for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
        PetscInt cx_local = i - info.xs;
        /* phi_ptr for a local Vec is offset so that phi_ptr[i] is the global cell i value */
        E_ptr[cx_local] = -(PetscScalar)((PetscRealPart(phi_ptr[sw + cx_local + 1]) - PetscRealPart(phi_ptr[sw + cx_local - 1])) * inv2hx);
      }
    }
    PetscCall(VecRestoreArrayAndMemType(phi_local, &phi_ptr));
    PetscCall(VecRestoreArrayAndMemType(ctx->E_field, &E_ptr));
  }

  PetscCall(DMRestoreLocalVector(ctx->daPot, &phi_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 8 -- AdvectX (SLDG L2 projection with DMDA ghost exchange)

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
static PetscErrorCode AdvectX(Vec f, Vec f_out, AppCtx *ctx)
{
  Vec       f_tmp = NULL;
  PetscBool inplace;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->AdvectXEvent, 0, 0, 0, 0));

  /* If f and f_out alias the same Vec, use a temporary output buffer */
  inplace = (PetscBool)(f == f_out);
  if (inplace) {
    PetscCall(VecDuplicate(f, &f_tmp));
    f_out = f_tmp;
  }

  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(ctx->daF, &info));

  /* Use precomputed A_sldg, B_sldg, n_shift from PrecomputeGeometry Step 8.
     A_sldg[iv*NbX*NbX + i*NbX + j] = overlap_integral[i,j] / w_i (already divided).
     The outer iv loop stays on CPU because DMGlobalToLocal is an MPI collective.
     The inner cx_local loop is offloaded to Kokkos.
     dt is already baked into n_shift/A_sldg/B_sldg at PrecomputeGeometry time. */
  {
    PetscScalar   *f_ptr, *fout_ptr;
    PetscMemType   mtype_f, mtype_fout;
    const PetscInt sw = info.sw;
    PetscCall(VecGetArrayAndMemType(f, &f_ptr, &mtype_f));
    PetscCall(VecGetArrayAndMemType(f_out, &fout_ptr, &mtype_fout));

    /* TODO: This loop issues NvDOF sequential DMGlobalToLocal (MPI collective) calls.
       For better scalability, pack all iv-slices into a single Vec with
       dof = NvDOF * NbX and do one DMGlobalToLocal per AdvectX call,
       then dispatch all iv kernels in a single 2D Kokkos launch. */
    for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
      PetscInt n = ctx->n_shift[iv];

      /* Copy this iv-slice into f_iv_global, then ghost-exchange */
      {
        PetscScalar *gv_ptr;
        PetscMemType mtype_gv;
        PetscCall(VecGetArrayAndMemType(ctx->f_iv_global, &gv_ptr, &mtype_gv));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
        if (mtype_f == PETSC_MEMTYPE_DEVICE) {
          const PetscInt NxDOF_loc = ctx->NxDOF_local;
          ScalarView1D   d_f(f_ptr, (size_t)ctx->NvDOF * NxDOF_loc);
          ScalarView1D   d_gv(gv_ptr, NxDOF_loc);
          auto           d_f_iv = Kokkos::subview(d_f, Kokkos::make_pair((size_t)(iv * NxDOF_loc), (size_t)((iv + 1) * NxDOF_loc)));
          Kokkos::deep_copy(d_gv, d_f_iv);
        } else
#endif
          for (PetscInt ix = 0; ix < ctx->NxDOF_local; ++ix) gv_ptr[ix] = f_ptr[iv * ctx->NxDOF_local + ix];
        PetscCall(VecRestoreArrayAndMemType(ctx->f_iv_global, &gv_ptr));
      }
      PetscCall(DMGlobalToLocal(ctx->daF, ctx->f_iv_global, INSERT_VALUES, ctx->f_iv_local));

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
      if (mtype_f == PETSC_MEMTYPE_DEVICE) {
        const PetscInt NxLocal   = ctx->NxLocal;
        const PetscInt NbX       = ctx->NbX;
        const PetscInt NxDOF_loc = ctx->NxDOF_local;
        const PetscInt xs        = ctx->xs;
        const PetscInt n_cells   = NxLocal + 2 * sw;
        ScalarView1D   d_f(f_ptr, (size_t)ctx->NvDOF * NxDOF_loc);
        ScalarView1D   d_fout(fout_ptr, (size_t)ctx->NvDOF * NxDOF_loc);
        /* Get device pointer to ghost-extended local Vec */
        {
          PetscScalar *local_ptr;
          PetscMemType mtype_local;
          PetscCall(VecGetArrayAndMemType(ctx->f_iv_local, &local_ptr, &mtype_local));
          /* local_ptr points to [ghost_left | owned | ghost_right], length = n_cells * NbX */
          ScalarView1D d_local(local_ptr, n_cells * NbX);
          /* Copy into persistent ghost buffer (PetscReal, not PetscScalar).
             Capture d_fiv_ghost by value (View is a reference-counted handle). */
          RealView1D d_fiv_ghost_copy = ctx->d_fiv_ghost;
          Kokkos::parallel_for("AdvectX_copy_ghost", Kokkos::RangePolicy<>(0, n_cells * NbX), KOKKOS_LAMBDA(PetscInt k) { d_fiv_ghost_copy(k) = PetscRealPart(d_local(k)); });
          PetscCall(VecRestoreArrayAndMemType(ctx->f_iv_local, &local_ptr));
        }

        /* Subview into persistent d_A_sldg, d_B_sldg -- zero-copy */
        auto d_A = Kokkos::subview(ctx->d_A_sldg, Kokkos::make_pair(iv * NbX * NbX, (iv + 1) * NbX * NbX));
        auto d_B = Kokkos::subview(ctx->d_B_sldg, Kokkos::make_pair(iv * NbX * NbX, (iv + 1) * NbX * NbX));
        /* Subview into d_fout for this iv-slice */
        auto       d_fout_iv   = Kokkos::subview(d_fout, Kokkos::make_pair((size_t)(iv * NxDOF_loc), (size_t)((iv + 1) * NxDOF_loc)));
        RealView1D d_fiv_ghost = ctx->d_fiv_ghost;

        Kokkos::parallel_for(
          "AdvectX_cx", Kokkos::RangePolicy<>(0, NxLocal), KOKKOS_LAMBDA(PetscInt cx_local) {
            PetscInt cx_global = xs + cx_local;
            PetscInt off_A     = (cx_global - n - (xs - sw)) * NbX;
            PetscInt off_B     = (cx_global - n - 1 - (xs - sw)) * NbX;
            for (PetscInt i = 0; i < NbX; ++i) {
              PetscReal val = 0.0;
              for (PetscInt j = 0; j < NbX; ++j) {
                val += d_A(i * NbX + j) * d_fiv_ghost(off_A + j);
                val += d_B(i * NbX + j) * d_fiv_ghost(off_B + j);
              }
              d_fout_iv(cx_local * NbX + i) = (PetscScalar)val;
            }
          });
      } else
#endif
      {
        PetscReal      **f_iv_arr; /* ghost-aware 2D array: f_iv_arr[cell][dof] */
        const PetscReal *A = &ctx->A_sldg[iv * ctx->NbX * ctx->NbX];
        const PetscReal *B = &ctx->B_sldg[iv * ctx->NbX * ctx->NbX];
        PetscCall(DMDAVecGetArrayDOF(ctx->daF, ctx->f_iv_local, &f_iv_arr));
        /* Apply SLDG update for each local x-cell. */
        for (PetscInt cx_local = 0; cx_local < ctx->NxLocal; ++cx_local) {
          PetscInt cx_global = ctx->xs + cx_local;
          PetscInt src_A     = cx_global - n;
          PetscInt src_B     = cx_global - n - 1;
          for (PetscInt i = 0; i < ctx->NbX; ++i) {
            PetscReal val = 0.0;
            for (PetscInt j = 0; j < ctx->NbX; ++j) {
              val += A[i * ctx->NbX + j] * PetscRealPart(f_iv_arr[src_A][j]);
              val += B[i * ctx->NbX + j] * PetscRealPart(f_iv_arr[src_B][j]);
            }
            fout_ptr[iv * ctx->NxDOF_local + cx_local * ctx->NbX + i] = (PetscScalar)val;
          }
        }
        PetscCall(DMDAVecRestoreArrayDOF(ctx->daF, ctx->f_iv_local, &f_iv_arr));
      }
    }

    PetscCall(VecRestoreArrayAndMemType(f, &f_ptr));
    PetscCall(VecRestoreArrayAndMemType(f_out, &fout_ptr));
  }

  /* Copy result back if in-place */
  if (inplace) {
    PetscCall(VecCopy(f_tmp, f));
    PetscCall(VecDestroy(&f_tmp));
  }

  PetscCall(PetscLogEventEnd(ctx->AdvectXEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 9 -- AdvectV (semi-Lagrangian nodal interpolation in velocity)

   For each GLL node (cv, bv) in the output v-space, trace the characteristic
   back by E*dt to find the foot point v_foot = v_node - E*dt, then
   interpolate f at v_foot using Lagrange interpolation in the source v-cell.
   This is consistent with the GLL (lumped) mass matrix used in MassV_inv:
   the nodal update f_new[cv,bv] = f(v_node[cv,bv] - E*dt) avoids the
   mismatch between an exact-quadrature RHS and a lumped mass inverse that
   causes poor accuracy for Q1 (degree=1).
   ======================================================================== */
static PetscErrorCode AdvectV(Vec f, Vec f_out, PetscReal dt, Vec E_field, AppCtx *ctx)
{
  Vec           f_tmp = NULL;
  Vec           E_local;
  DMDALocalInfo info;
  PetscBool     inplace;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ctx->AdvectVEvent, 0, 0, 0, 0));

  /* If f and f_out alias the same Vec, use a temporary output buffer */
  inplace = (PetscBool)(f == f_out);
  if (inplace) {
    PetscCall(VecDuplicate(f, &f_tmp));
    f_out = f_tmp;
  }

  /* Ghost-exchange E_field so we can interpolate E at each GLL node position.
     daPot has stencil width sw=1, so E_local has one ghost cell on each side. */
  PetscCall(DMDAGetLocalInfo(ctx->daPot, &info));
  PetscCall(DMGetLocalVector(ctx->daPot, &E_local));
  PetscCall(DMGlobalToLocal(ctx->daPot, E_field, INSERT_VALUES, E_local));

  {
    PetscScalar *f_ptr, *Eloc_ptr, *fout_ptr;
    PetscMemType mtype_f, mtype_Eloc, mtype_fout;
    PetscCall(VecGetArrayAndMemType(f, &f_ptr, &mtype_f));
    PetscCall(VecGetArrayAndMemType(E_local, &Eloc_ptr, &mtype_Eloc));
    PetscCall(VecGetArrayAndMemType(f_out, &fout_ptr, &mtype_fout));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    if (mtype_f == PETSC_MEMTYPE_DEVICE) {
      const PetscInt  NxLocal   = ctx->NxLocal;
      const PetscInt  NxDOF_loc = ctx->NxDOF_local;
      const PetscInt  NvDOF     = ctx->NvDOF;
      const PetscInt  sw        = info.sw;
      const PetscInt  n_Eloc    = NxLocal + 2 * sw;
      const PetscInt  Nv        = ctx->Nv;
      const PetscInt  NbX       = ctx->NbX;
      const PetscInt  NbV       = ctx->NbV;
      const PetscReal v_max     = ctx->v_max;
      const PetscReal h_v       = ctx->h_v;
      RealView1D      d_xi_x    = ctx->d_xi_x_nodes;
      ScalarView1D    d_f(f_ptr, (size_t)NvDOF * NxDOF_loc);
      ScalarView1D    d_Eloc(Eloc_ptr, n_Eloc);
      ScalarView1D    d_fout(fout_ptr, (size_t)NvDOF * NxDOF_loc);
      RealView1D      d_xi_v = ctx->d_xi_v_nodes;
      /* Parallelize over (iv, cx_local) pairs -- no write conflicts. */
      Kokkos::parallel_for(
        "AdvectV_nodal", Kokkos::RangePolicy<>(0, NvDOF * NxLocal), KOKKOS_LAMBDA(PetscInt idx) {
          PetscInt iv       = idx / NxLocal;
          PetscInt cx_local = idx % NxLocal;
          PetscInt cv       = iv / NbV;
          PetscInt bv       = iv % NbV;
          /* Physical v-coordinate of this GLL node */
          PetscReal v_node = -v_max + cv * h_v + (d_xi_v(bv) + 1.0) * (h_v * 0.5);

          /* Ghost-extended E: d_Eloc[sw + cx_local] is the owned cell center. */
          PetscReal E_c  = PetscRealPart(d_Eloc(sw + cx_local));
          PetscReal E_lm = PetscRealPart(d_Eloc(sw + cx_local - 1));
          PetscReal E_rp = PetscRealPart(d_Eloc(sw + cx_local + 1));

          for (PetscInt bx = 0; bx < NbX; ++bx) {
            /* Linearly interpolate E at GLL node position x_{cx,bx}.
               xi_x[bx] in [-1,1]: t = xi_x[bx]/2 in [-0.5,0.5].
               For t>=0: E_bx = E_c*(1-t) + E_rp*t
               For t< 0: E_bx = E_lm*(-t) + E_c*(1+t) */
            PetscReal t      = d_xi_x(bx) * 0.5;
            PetscReal E_bx   = (t >= 0.0) ? (E_c * (1.0 - t) + E_rp * t) : (E_lm * (-t) + E_c * (1.0 + t));
            PetscReal v_foot = v_node - E_bx * dt;

            PetscReal v_norm  = (v_foot + v_max) / h_v;
            PetscInt  cv_foot = (PetscInt)PetscFloorReal(v_norm);
            if (cv_foot < 0) cv_foot = 0;
            if (cv_foot >= Nv) cv_foot = Nv - 1;

            PetscReal xi_foot = (v_foot - (-v_max + cv_foot * h_v)) / (h_v * 0.5) - 1.0;
            if (xi_foot < -1.0) xi_foot = -1.0;
            if (xi_foot > 1.0) xi_foot = 1.0;

            PetscInt  ix    = cx_local * NbX + bx;
            PetscReal f_val = 0.0;
            for (PetscInt bv2 = 0; bv2 < NbV; ++bv2) {
              PetscInt iv_src = cv_foot * NbV + bv2;
              f_val += PetscRealPart(d_f(iv_src * NxDOF_loc + ix)) * EvalLagrangeBasis(NbV, d_xi_v.data(), bv2, xi_foot);
            }
            d_fout(iv * NxDOF_loc + ix) = (PetscScalar)f_val;
          }
        });
    } else
#endif
    {
      for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
        PetscInt  cv     = iv / ctx->NbV;
        PetscInt  bv     = iv % ctx->NbV;
        PetscReal v_node = -ctx->v_max + cv * ctx->h_v + (ctx->xi_v_nodes[bv] + 1.0) * (ctx->h_v * 0.5);

        for (PetscInt cx_local = 0; cx_local < ctx->NxLocal; ++cx_local) {
          /* Ghost-extended E: Eloc_ptr[sw + cx_local] is the owned cell center. */
          PetscReal E_c  = PetscRealPart(Eloc_ptr[sw + cx_local]);
          PetscReal E_lm = PetscRealPart(Eloc_ptr[sw + cx_local - 1]);
          PetscReal E_rp = PetscRealPart(Eloc_ptr[sw + cx_local + 1]);

          for (PetscInt bx = 0; bx < ctx->NbX; ++bx) {
            /* Linearly interpolate E at GLL node position x_{cx,bx}.
               t = xi_x[bx]/2 in [-0.5,0.5].
               For t>=0: E_bx = E_c*(1-t) + E_rp*t
               For t< 0: E_bx = E_lm*(-t) + E_c*(1+t) */
            PetscReal t      = ctx->xi_x_nodes[bx] * 0.5;
            PetscReal E_bx   = (t >= 0.0) ? (E_c * (1.0 - t) + E_rp * t) : (E_lm * (-t) + E_c * (1.0 + t));
            PetscReal v_foot = v_node - E_bx * dt;

            PetscReal v_norm  = (v_foot + ctx->v_max) / ctx->h_v;
            PetscInt  cv_foot = (PetscInt)PetscFloorReal(v_norm);
            if (cv_foot < 0) cv_foot = 0;
            if (cv_foot >= ctx->Nv) cv_foot = ctx->Nv - 1;

            PetscReal xi_foot = (v_foot - (-ctx->v_max + cv_foot * ctx->h_v)) / (ctx->h_v * 0.5) - 1.0;
            if (xi_foot < -1.0) xi_foot = -1.0;
            if (xi_foot > 1.0) xi_foot = 1.0;

            PetscInt  ix    = cx_local * ctx->NbX + bx;
            PetscReal f_val = 0.0;
            for (PetscInt bv2 = 0; bv2 < ctx->NbV; ++bv2) {
              PetscInt iv_src = cv_foot * ctx->NbV + bv2;
              f_val += PetscRealPart(f_ptr[iv_src * ctx->NxDOF_local + ix]) * EvalLagrangeBasis(ctx->NbV, ctx->xi_v_nodes, bv2, xi_foot);
            }
            fout_ptr[iv * ctx->NxDOF_local + ix] = (PetscScalar)f_val;
          }
        }
      }
    }
    PetscCall(VecRestoreArrayAndMemType(f, &f_ptr));
    PetscCall(VecRestoreArrayAndMemType(E_local, &Eloc_ptr));
    PetscCall(VecRestoreArrayAndMemType(f_out, &fout_ptr));
  }
  PetscCall(DMRestoreLocalVector(ctx->daPot, &E_local));

  /* Copy result back if in-place */
  if (inplace) {
    PetscCall(VecCopy(f_tmp, f));
    PetscCall(VecDestroy(&f_tmp));
  }

  PetscCall(PetscLogEventEnd(ctx->AdvectVEvent, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 10 -- ComputeMoments
   ======================================================================== */
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
struct MomTriple {
  PetscReal                   r0, r1, r2;
  KOKKOS_INLINE_FUNCTION      MomTriple() : r0(0.0), r1(0.0), r2(0.0) { }
  KOKKOS_INLINE_FUNCTION      MomTriple(const MomTriple &) = default;
  KOKKOS_INLINE_FUNCTION void operator+=(const MomTriple &rhs)
  {
    r0 += rhs.r0;
    r1 += rhs.r1;
    r2 += rhs.r2;
  }
};
namespace Kokkos
{
template <>
struct reduction_identity<MomTriple> {
  KOKKOS_INLINE_FUNCTION static MomTriple sum() { return MomTriple(); }
};
} // namespace Kokkos
#endif

static PetscErrorCode ComputeMoments(AppCtx *ctx, PetscReal *m0_out, PetscReal *m1_out, PetscReal *m2_out)
{
  PetscReal    m0_local = 0.0, m1_local = 0.0, m2_local = 0.0;
  PetscScalar *f_ptr;
  PetscMemType mtype_f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayAndMemType(ctx->f, &f_ptr, &mtype_f));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  if (mtype_f == PETSC_MEMTYPE_DEVICE) {
    const PetscInt  NvDOF     = ctx->NvDOF;
    const PetscInt  NbV       = ctx->NbV;
    const PetscInt  NbX       = ctx->NbX;
    const PetscInt  NqV       = ctx->NqV;
    const PetscInt  NxDOF_loc = ctx->NxDOF_local;
    const PetscReal h_v       = ctx->h_v;
    const PetscReal v_max     = ctx->v_max;
    ScalarView1D    d_f(f_ptr, (size_t)NvDOF * NxDOF_loc);
    RealView1D      d_vbi = ctx->d_v_basis_int;
    RealView1D      d_xbi = ctx->d_x_basis_int;
    RealView1D      d_tab = ctx->d_tabV;
    RealView1D      d_vqp = ctx->d_vq_pts;
    RealView1D      d_vqw = ctx->d_vq_wts;
    /* Per-iv moment weights: w0[iv], w1[iv], w2[iv] */
    Kokkos::View<PetscReal *, DevSpace> d_w0("d_w0", NvDOF);
    Kokkos::View<PetscReal *, DevSpace> d_w1("d_w1", NvDOF);
    Kokkos::View<PetscReal *, DevSpace> d_w2("d_w2", NvDOF);
    /* Step 1: precompute per-iv weights w0, w1, w2 */
    Kokkos::parallel_for(
      "moments_weights", Kokkos::RangePolicy<>(0, NvDOF), KOKKOS_LAMBDA(PetscInt iv) {
        PetscInt  cv          = iv / NbV;
        PetscInt  bv          = iv % NbV;
        PetscReal v_cell_left = -v_max + cv * h_v;
        PetscReal w0 = d_vbi(iv), w1 = 0.0, w2 = 0.0;
        for (PetscInt q = 0; q < NqV; ++q) {
          PetscReal v_phys = v_cell_left + (d_vqp(q) + 1.0) * (h_v * 0.5);
          PetscReal phi_bv = d_tab(q * NbV + bv);
          PetscReal wq     = d_vqw(q) * (h_v * 0.5);
          w1 += v_phys * phi_bv * wq;
          w2 += v_phys * v_phys * phi_bv * wq;
        }
        d_w0(iv) = w0;
        d_w1(iv) = w1;
        d_w2(iv) = w2;
      });
    /* Step 2: single parallel_reduce pass over flat index using MomTriple reducer.
       One pass over d_f instead of three, reducing memory traffic by 3x. */
    MomTriple result;
    Kokkos::parallel_reduce(
      "moments_all", Kokkos::RangePolicy<>(0, NvDOF * NxDOF_loc),
      KOKKOS_LAMBDA(PetscInt idx, MomTriple &lsum) {
        PetscInt  iv   = idx / NxDOF_loc;
        PetscInt  ix   = idx % NxDOF_loc;
        PetscInt  bx   = ix % NbX;
        PetscReal fval = PetscRealPart(d_f(idx)) * d_xbi(bx);
        lsum.r0 += fval * d_w0(iv);
        lsum.r1 += fval * d_w1(iv);
        lsum.r2 += fval * d_w2(iv);
      },
      result);
    m0_local = result.r0;
    m1_local = result.r1;
    m2_local = result.r2;
  } else
#endif
  {
    for (PetscInt iv = 0; iv < ctx->NvDOF; ++iv) {
      PetscReal w0          = ctx->v_basis_int[iv];
      PetscInt  cv          = iv / ctx->NbV;
      PetscInt  bv          = iv % ctx->NbV;
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
          PetscReal fval = PetscRealPart(f_ptr[iv * ctx->NxDOF_local + ix]);
          /* x_basis_int[bx] = gll_wts_x[bx] * h_x/2 */
          PetscReal dx = ctx->x_basis_int[bx];
          m0_local += fval * w0 * dx;
          m1_local += fval * w1 * dx;
          m2_local += fval * w2 * dx;
        }
      }
    }
  }
  PetscCall(VecRestoreArrayAndMemType(ctx->f, &f_ptr));

  PetscReal loc[3] = {m0_local, m1_local, m2_local}, glob[3];
  PetscCallMPI(MPIU_Allreduce(loc, glob, 3, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)ctx->f)));
  *m0_out = glob[0];
  *m1_out = glob[1];
  *m2_out = glob[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 10 -- BSLStep_Strang (Strang splitting: X/2, V, X/2)
   ======================================================================== */
static PetscErrorCode BSLStep_Strang(AppCtx *ctx, PetscReal dt)
{
  PetscFunctionBeginUser;
  PetscCall(AdvectX(ctx->f, ctx->f, ctx));
  PetscCall(ComputeChargeDensity(ctx));
  PetscCall(SolvePoisson(ctx));
  PetscCall(ComputeElectricField(ctx));
  PetscCall(AdvectV(ctx->f, ctx->f, dt, ctx->E_field, ctx));
  PetscCall(AdvectX(ctx->f, ctx->f, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   Phase 10 -- DestroyContext
   ======================================================================== */
static PetscErrorCode DestroyContext(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&ctx->f));
  PetscCall(VecDestroy(&ctx->rho));
  PetscCall(VecDestroy(&ctx->phi));
  PetscCall(VecDestroy(&ctx->E_field));
  /* Release pre-allocated AdvectX/Poisson Vecs before destroying daF/daPot */
  PetscCall(VecDestroy(&ctx->f_iv_global));
  PetscCall(DMRestoreLocalVector(ctx->daF, &ctx->f_iv_local));
  PetscCall(VecDestroy(&ctx->rhs_poisson));
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
  PetscCall(PetscFree(ctx->A_sldg));
  PetscCall(PetscFree(ctx->B_sldg));
  PetscCall(PetscFree(ctx->n_shift));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  /* Release device Views while Kokkos is still alive (before PetscFinalize).
     Assigning a default-constructed View decrements the reference count to zero
     and frees the device allocation immediately. */
  ctx->d_xi_v_nodes   = RealView1D();
  ctx->d_xi_x_nodes   = RealView1D();
  ctx->d_v_dof_coords = RealView1D();
  ctx->d_v_basis_int  = RealView1D();
  ctx->d_x_basis_int  = RealView1D();
  ctx->d_tabV         = RealView1D();
  ctx->d_vq_pts       = RealView1D();
  ctx->d_vq_wts       = RealView1D();
  ctx->d_A_sldg       = RealView1D();
  ctx->d_B_sldg       = RealView1D();
  ctx->d_n_shift      = IntView1D();
  ctx->d_fiv_ghost    = RealView1D();
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================
   main -- full time loop
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

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "SLDG Landau damping: Nx=%" PetscInt_FMT " Nv=%" PetscInt_FMT " degree_x=%" PetscInt_FMT " degree_v=%" PetscInt_FMT " alpha=%g kwave=%g dt=%g steps=%" PetscInt_FMT "\n", ctx.Nx, ctx.Nv, ctx.degree_x, ctx.degree_v,
                        (double)ctx.alpha, (double)ctx.kwave, (double)ctx.dt, ctx.steps));

  /* E-field history for Landau check */
  PetscReal *E_history = NULL;
  if (ctx.check_landau) PetscCall(PetscMalloc1(ctx.steps + 1, &E_history));

  /* Print header */
  if (ctx.ostep > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-12s %-20s %-20s %-12s %-20s %-12s %-20s %-20s %-20s\n", "Time", "Sum_E", "|E|", "log|E|", "E_max", "log(E_max)", "m0", "m1", "m2"));

  /* Main time loop */
  for (PetscInt step = 0; step < ctx.steps; ++step) {
    PetscCall(BSLStep_Strang(&ctx, ctx.dt));

    PetscReal E_max;
    PetscCall(VecNorm(ctx.E_field, NORM_INFINITY, &E_max));

    if (ctx.check_landau) E_history[step] = E_max;

    if (ctx.ostep > 0 && step % ctx.ostep == 0) {
      PetscReal   t = (step + 1) * ctx.dt;
      PetscReal   m0, m1, m2;
      PetscReal   E_norm;
      PetscScalar E_sum_sc;
      PetscCall(VecNorm(ctx.E_field, NORM_2, &E_norm));
      PetscCall(VecSum(ctx.E_field, &E_sum_sc));
      PetscReal E_sum = PetscRealPart(E_sum_sc);
      PetscCall(ComputeMoments(&ctx, &m0, &m1, &m2));
      PetscReal lgEmax  = E_max > 0 ? PetscLog10Real(E_max) : -16.0;
      PetscReal lgEnorm = E_norm > 0 ? PetscLog10Real(E_norm) : -16.0;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "E: %f\t%+e\t%e\t%f\t%20.15e\t%f\t%20.15e\t%20.15e\t%20.15e\t(%" PetscInt_FMT ")\n", (double)t, (double)E_sum, (double)E_norm, (double)lgEnorm, (double)E_max, (double)lgEmax, (double)m0, (double)m1, (double)m2, step));
    }
  }

  /* Landau damping rate check */
  if (ctx.check_landau) {
    /* Find first minimum, fit log-linear to envelope peaks */
    PetscReal gamma_theory = (PetscAbsReal(ctx.alpha - 0.5) < 0.1) ? -0.286 : -0.1533;
    PetscInt  first_min    = ctx.steps - 1;
    for (PetscInt s = 1; s < ctx.steps - 1; ++s)
      if (E_history[s] < E_history[s - 1] && E_history[s] < E_history[s + 1]) {
        first_min = s;
        break;
      }

    PetscReal sum_t = 0, sum_logE = 0, sum_t_logE = 0, sum_t2 = 0;
    PetscInt  n_pts     = 0;
    PetscReal last_peak = PETSC_MAX_REAL;
    for (PetscInt s = first_min + 1; s < ctx.steps - 1; ++s) {
      PetscReal val = E_history[s];
      if (val > E_history[s - 1] && val > E_history[s + 1] && val > 1e-16) {
        if (val > last_peak) break;
        last_peak   = val;
        PetscReal t = (s + 1) * ctx.dt;
        sum_t += t;
        sum_logE += PetscLogReal(val);
        sum_t_logE += t * PetscLogReal(val);
        sum_t2 += t * t;
        n_pts++;
      }
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nLandau Damping Check:\n"));
    if (n_pts > 1) {
      PetscReal gamma = (n_pts * sum_t_logE - sum_t * sum_logE) / (n_pts * sum_t2 - sum_t * sum_t);
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Measured gamma = %g  (theory: %g)\n", (double)gamma, (double)gamma_theory));
    } else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Not enough peaks to fit gamma\n"));
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
    args: -Nx 64 -Nv 128 -degree_x 1 -degree_v 1 -dt 0.1 -steps 100 -ostep 5 -check_landau

  test:
    suffix: landau_kokkos
    nsize: 4
    requires: kokkos_kernels
    args: -Nx 16 -Nv 32 -degree_x 2 -degree_v 2 -dt 0.1 -steps 100 -ostep 5 -check_landau -dm_mat_type aijkokkos -mat_type aijkokkos -dm_vec_type kokkos -vec_type kokkos

TEST*/
