/*TEST
  test:
    suffix: 1
    args: -petscspace_degree 2 -dm_plex_box_faces 8,8 -ksp_type cg -pc_type gamg
    filter: grep "L2 error"
TEST*/

/*
  ex_kokkos_fe.kokkos.cxx -- Tutorial: GPU-resident FEM assembly with PetscFEKokkosMaps

  Demonstrates the GPU-resident finite element assembly API:

    PetscFEKokkosSetUp(dm, &maps, J)          -- build maps + geometry once at setup
    DMPlexSNESComputeResidualFEM_Kokkos<f0, f1>(snes, X, F, &maps)
    DMPlexSNESComputeJacobianFEM_Kokkos<G0, G1, G2, G3>(snes, X, J, Jp, &maps)
    PetscFEKokkosMapsDestroy(&maps)           -- frees maps and cached geometry

  Geometry lifecycle (Landau-style setup/reset pattern):
    PetscFEKokkosSetUp calls PetscFEKokkosSetUpGeometry internally.
    For moving meshes or AMR, call PetscFEKokkosResetGeometry + PetscFEKokkosSetUpGeometry
    explicitly to rebuild device geometry without rebuilding the full maps.

  Problem: 2D Poisson equation  -grad^2u = f  on [0,1]^2
    Manufactured solution:  u_exact(x,y) = sin(pi*x) sin(pi*y)
    Source term:            f(x,y)       = 2*pi^2 sin(pi*x) sin(pi*y)

  Key design principle:
    CUDA cannot call host function pointers from device kernels.
    Solution: write physics callbacks as KOKKOS_INLINE_FUNCTION and pass them
    as C++ template parameters.  nvcc_wrapper inlines them at compile time.

  Usage:
    ./ex_kokkos_fe -petscspace_degree 2 -dm_plex_box_faces 8,8 \
                   -ksp_type cg -pc_type gamg

  Expected output (P2, 8x8 mesh):
    L2 error: ~4.9e-04

  Build:
    export PETSC_DIR=/path/to/petsc_cld
    export PETSC_ARCH=arch-macosx-gnu-kokkos-O-3d
    make ex_kokkos_fe

  See petscfekokkos.h for the full API reference.
*/

static char help[] = "Tutorial: GPU-resident FEM assembly with PetscFEKokkosMaps\n"
                     "  Problem: -Laplacian(u) = f on [0,1]^2\n"
                     "  Manufactured solution: u = sin(pi*x)*sin(pi*y)\n"
                     "Options:\n"
                     "  -petscspace_degree <k>     FE polynomial degree (default: 1)\n"
                     "  -dm_plex_box_faces <Nx,Ny> mesh resolution (default: 4,4)\n"
                     "  -ksp_type cg -pc_type gamg recommended solver\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscfe.h>
#include <Kokkos_Core.hpp>
#include <petscfekokkos.h>

/* =========================================================================
   Step 1: Define physics callbacks as KOKKOS_INLINE_FUNCTION.

   These MUST be device-callable (KOKKOS_INLINE_FUNCTION) so that
   nvcc_wrapper can inline them into the GPU kernel at compile time.

   Signature matches PetscPointFn (petscdstypes.h) -- use the
   PETSCFE_KOKKOS_POINT_ARGS / PETSCFE_KOKKOS_JAC_POINT_ARGS macros.
   ========================================================================= */

/* f0: source term  f0 = -2pi^2 sin(pix) sin(piy)
 * (negative because PETSc convention: F(u) = 0, so f0 = -f_rhs) */
KOKKOS_INLINE_FUNCTION
static void f0_poisson(PETSCFE_KOKKOS_POINT_ARGS, PetscScalar f0[])
{
  /* assumes dim == 2 */
  /* Cast to PetscScalar for complex-build safety: Kokkos::sin returns PetscReal. */
  f0[0] = (PetscScalar)(-2.0 * PETSC_PI * PETSC_PI * Kokkos::sin(PETSC_PI * x[0]) * Kokkos::sin(PETSC_PI * x[1]));
}

/* f1: flux term  f1[d] = du/dx_d  (Laplacian weak form) */
KOKKOS_INLINE_FUNCTION
static void f1_poisson(PETSCFE_KOKKOS_POINT_ARGS, PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/* g3: Jacobian  g3[i*dim+j] = delta_{ij}  (identity tensor for Laplacian) */
KOKKOS_INLINE_FUNCTION
static void g3_poisson(PETSCFE_KOKKOS_JAC_POINT_ARGS, PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

/* =========================================================================
   Step 2: Manufactured solution (HOST callback -- not device-callable).
   Registered with PetscDSSetExactSolution and DMAddBoundary.
   ========================================================================= */
static PetscErrorCode u_exact(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCheck(dim == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Only dim==2 supported");
  *u = PetscSinReal(PETSC_PI * x[0]) * PetscSinReal(PETSC_PI * x[1]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* =========================================================================
   Step 3: SetupDiscretization -- attach PETSCFEKOKKOS to DM.
   ========================================================================= */
static PetscErrorCode SetupDiscretization(DM dm)
{
  PetscFE        fe;
  PetscDS        ds;
  DMLabel        label;
  PetscInt       dim;
  PetscBool      simplex;
  DM             plex;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &simplex));
  PetscCall(DMDestroy(&plex));

  /* Create scalar Lagrange FE space; degree set via -petscspace_degree */
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe));
  PetscCall(PetscFESetType(fe, PETSCFEKOKKOS));
  PetscCall(PetscObjectSetName((PetscObject)fe, "u"));

  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));

  /* Register DS metadata (exact solution, BCs).  No host residual callbacks
   * are needed: the GPU template path (DMPlexSNESComputeResidualFEM_Kokkos)
   * bypasses the PetscDS callback table entirely.
   * PetscDSSetResidual(ds, 0, NULL, NULL) is called only to satisfy any
   * internal PetscDS consistency checks that expect field 0 to be registered. */
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, NULL, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 0, u_exact, NULL));

  /* Dirichlet BC: u = u_exact on all boundary faces */
  PetscCall(DMCreateLabel(dm, "marker"));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))u_exact, NULL, NULL, NULL));

  /* Propagate discretization to coarse DMs (for GAMG) */
  {
    DM cdm = dm;
    while (cdm) {
      PetscCall(DMCopyDisc(dm, cdm));
      PetscCall(DMGetCoarseDM(cdm, &cdm));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* =========================================================================
   main
   ========================================================================= */
int main(int argc, char **argv)
{
  DM                dm;
  SNES              snes;
  Vec               u;
  Mat               J;
  PetscReal         error;
  PetscFEKokkosMaps kokkos_ctx; /* Step 4: declare the GPU assembly maps */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Create mesh */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* Attach FE discretization */
  PetscCall(SetupDiscretization(dm));

  /* Create SNES */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));

  /* Step 5: Register GPU-resident residual callback.
   *
   * DMPlexSNESComputeResidualFEM_Kokkos<f0, f1> is a drop-in replacement
   * for DMPlexSNESComputeResidualFEM.  The template parameters are the
   * KOKKOS_INLINE_FUNCTION callbacks defined above.
   *
   * The context pointer (&kokkos_ctx) is passed as the void* ctx argument.
   * It must be set up via PetscFEKokkosSetUp before the first SNES solve. */
  PetscCall(SNESSetFunction(snes, NULL, DMPlexSNESComputeResidualFEM_Kokkos<f0_poisson, f1_poisson>, &kokkos_ctx));

  PetscCall(SNESSetFromOptions(snes));

  /* Step 6: PetscFEKokkosSetUp -- build assembly maps, stage to device,
   * preallocate COO matrix, and build element geometry once.
   *
   * Must be called AFTER SNESSetFromOptions (which triggers DMSetUp ->
   * Kokkos::initialize).  Internally calls:
   *   PetscFEKokkosCreateMaps + PetscFEKokkosStageMaps
   *   + PetscFEKokkosPreallocateCOO + PetscFEKokkosSetUpGeometry
   *
   * Geometry is built once here and reused across all residual/Jacobian
   * evaluations.  For moving meshes or AMR, call PetscFEKokkosResetGeometry
   * + PetscFEKokkosSetUpGeometry explicitly to rebuild without recreating maps. */
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(PetscFEKokkosSetUp(dm, &kokkos_ctx, J));

  /* Step 7: Register GPU-resident Jacobian callback.
   *
   * DMPlexSNESComputeJacobianFEM_Kokkos<G0,G1,G2,G3> takes all four
   * Jacobian callbacks.  Pass nullptr for unused terms (g0, g1, g2 are
   * zero for the Laplacian; only g3 is non-zero). */
  PetscCall(SNESSetJacobian(snes, J, J, DMPlexSNESComputeJacobianFEM_Kokkos<nullptr, nullptr, nullptr, g3_poisson>, &kokkos_ctx));
  PetscCall(MatDestroy(&J));

  /* Solve */
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  PetscCall(VecSet(u, 0.0));
  PetscCall(SNESSolve(snes, NULL, u));

  /* ---- Compute L2 error ---- */
  {
    PetscErrorCode (*exactFuncs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *) = {u_exact};
    PetscCall(DMComputeL2Diff(dm, 0.0, exactFuncs, NULL, u, &error));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error: %g\n", (double)error));

  /* Step 8: PetscFEKokkosMapsDestroy -- free host arrays.
   * Device Kokkos::Views are reference-counted and freed automatically. */
  PetscCall(PetscFEKokkosMapsDestroy(&kokkos_ctx));

  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
