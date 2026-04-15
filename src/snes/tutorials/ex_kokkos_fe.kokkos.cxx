static char help[] = "Tutorial: GPU-resident FEM assembly with PetscFEKokkosMaps\n"
                     "  Problem: -Laplacian(u) = f on [0,1]^d, d=2 or 3\n"
                     "  Manufactured solution: u = prod_{i} sin(pi*x_i)\n"
                     "Options:\n"
                     "  -petscspace_degree <k>          FE polynomial degree (default: 1)\n"
                     "  -dm_plex_box_faces <Nx[,Ny,Nz]> mesh resolution (default: 4,4)\n"
                     "  -dm_plex_dim <d>                spatial dimension (default: 2)\n"
                     "  -ksp_type cg -pc_type gamg      recommended solver\n";

/* GPU-resident FEM assembly tutorial using PetscFEKokkosMaps and template callbacks */

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscfe.h>
#include <Kokkos_Core.hpp>
#include <petscfekokkos.h>

/* Physics callbacks: device-callable via KOKKOS_INLINE_FUNCTION, passed as template parameters */

/* f0: source term  f0 = -dim * pi^2 * prod_{d=0}^{dim-1} sin(pi*x[d])
 * Manufactured solution: u = prod sin(pi*x[d])
 * Laplacian: -grad^2 u = dim * pi^2 * prod sin(pi*x[d])
 * PETSc convention: F(u) = 0, so f0 = -f_rhs */
KOKKOS_INLINE_FUNCTION
static void f0_poisson(PETSC_POINT_ARGS, PetscScalar f0[])
{
  PetscReal prod = 1.0;
  for (PetscInt d = 0; d < dim; ++d) prod *= Kokkos::sin(PETSC_PI * x[d]);
  f0[0] = (PetscScalar)(-(PetscReal)dim * PETSC_PI * PETSC_PI * prod);
}

/* f1: flux term  f1[d] = du/dx_d  (Laplacian weak form) */
KOKKOS_INLINE_FUNCTION
static void f1_poisson(PETSC_POINT_ARGS, PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/* g3: Jacobian  g3[i*dim+j] = delta_{ij}  (identity tensor for Laplacian) */
KOKKOS_INLINE_FUNCTION
static void g3_poisson(PETSC_JAC_POINT_ARGS, PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

/* Manufactured solution (host callback for PetscDSSetExactSolution and DMAddBoundary) */
static PetscErrorCode u_exact(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscReal prod = 1.0;

  PetscFunctionBeginUser;
  for (PetscInt d = 0; d < dim; ++d) prod *= PetscSinReal(PETSC_PI * x[d]);
  *u = prod;
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

  /* Register DS metadata (exact solution, BCs) */
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

int main(int argc, char **argv)
{
  DM                dm;
  SNES              snes;
  Vec               u;
  Mat               J;
  PetscReal         error;
  PetscFEKokkosMaps maps;

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

  /* Register GPU-resident residual callback; &maps is the void* ctx */
  PetscCall(SNESSetFunction(snes, NULL, DMPlexSNESComputeResidualFEM_Kokkos<f0_poisson, f1_poisson>, &maps));

  PetscCall(SNESSetFromOptions(snes));

  /* Build assembly maps, stage to device, preallocate COO, build geometry.
     Must be called after SNESSetFromOptions (which triggers Kokkos::initialize). */
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(PetscFEKokkosSetUp(dm, &maps, J));

  /* Register GPU-resident Jacobian callback; nullptr for unused g0, g1, g2 terms */
  PetscCall(SNESSetJacobian(snes, J, J, (DMPlexSNESComputeJacobianFEM_Kokkos<nullptr, nullptr, nullptr, g3_poisson, true>), &maps));
  PetscCall(MatDestroy(&J));

  /* Solve */
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  PetscCall(VecSet(u, 0.0));
  PetscCall(SNESSolve(snes, NULL, u));

  /* Compute L2 error */
  {
    PetscErrorCode (*exactFuncs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *) = {u_exact};
    PetscCall(DMComputeL2Diff(dm, 0.0, exactFuncs, NULL, u, &error));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error: %g\n", (double)error));

  /* Free assembly maps and cached geometry */
  PetscCall(PetscFEKokkosMapsDestroy(&maps));

  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST
  testset:
    requires: kokkos_kernels
    nsize: 4
    args: -petscspace_degree 2 -dm_mat_type aijkokkos -dm_vec_type kokkos -ksp_type cg -pc_type gamg
    filter: grep "L2 error"
    test:
      suffix: 1
      requires: triangle
      args: -dm_plex_simplex 1 -dm_plex_box_faces 8,8
    test:
      suffix: 2
      args: -dm_plex_simplex 0 -dm_plex_box_faces 8,8
    test:
      suffix: 3
      requires: triangle ctetgen
      args: -dm_plex_dim 3 -dm_plex_simplex 1 -dm_plex_box_faces 2,2,2
    test:
      suffix: 4
      args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2
TEST*/
