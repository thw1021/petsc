static char help[] = "Tutorial: GPU-resident FEM assembly with PetscFEKokkosMaps\n"
                     "  Problem: -Laplacian(u) = f on [0,1]^d, d=2 or 3\n"
                     "  Manufactured solution: u = prod_{i} sin(pi*x_i)\n"
                     "Options:\n"
                     "  -petscspace_degree <k>          FE polynomial degree (default: 1)\n"
                     "  -dm_plex_box_faces <Nx[,Ny,Nz]> mesh resolution (default: 4,4)\n"
                     "  -dm_plex_dim <d>                spatial dimension (default: 2)\n"
                     "  -use_kokkos_maps <bool>         use GPU-resident template path (default: true)\n"
                     "  -ksp_type cg -pc_type gamg      recommended solver\n";

/* GPU-resident FEM assembly tutorial using PetscFEKokkosMaps and template callbacks */

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscfe.h>
#include <Kokkos_Core.hpp>
#include <petscfekokkos.h>

/* Residual and Jacobian callbacks.  KOKKOS_INLINE_FUNCTION marks them __host__ __device__
   so they serve both as template parameters for the GPU-resident path and as plain
   PetscPointFn / PetscPointJacFn pointers for the standard fallback path. */

/* f0: source term  f0 = -dim * pi^2 * prod sin(pi*x[d]) */
KOKKOS_INLINE_FUNCTION
static void f0_poisson(PETSC_POINT_ARGS, PetscScalar f0[])
{
  PetscReal prod = 1.0;
  for (PetscInt d = 0; d < dim; ++d) prod *= Kokkos::sin(PETSC_PI * x[d]);
  f0[0] = (PetscScalar)(-(PetscReal)dim * PETSC_PI * PETSC_PI * prod);
}

/* f1: flux term  f1[d] = du/dx_d */
KOKKOS_INLINE_FUNCTION
static void f1_poisson(PETSC_POINT_ARGS, PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/* g3: Jacobian  g3[i*dim+j] = delta_{ij} */
KOKKOS_INLINE_FUNCTION
static void g3_poisson(PETSC_JAC_POINT_ARGS, PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

/* Manufactured solution for exact BCs and error computation */
static PetscErrorCode u_exact(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscReal prod = 1.0;

  PetscFunctionBeginUser;
  for (PetscInt d = 0; d < dim; ++d) prod *= PetscSinReal(PETSC_PI * x[d]);
  *u = prod;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, PetscBool use_kokkos_maps)
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

  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe));
  PetscCall(PetscFESetType(fe, PETSCFEKOKKOS));
  PetscCall(PetscObjectSetName((PetscObject)fe, "u"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));

  PetscCall(DMGetDS(dm, &ds));
  if (use_kokkos_maps) {
    /* Template path: residual/Jacobian registered later via SNESSetFunction/Jacobian */
    PetscCall(PetscDSSetResidual(ds, 0, NULL, NULL));
  } else {
    /* Fallback path: register callbacks so DMPlexSNESComputeResidualFEM
       invokes PetscFEIntegrateResidual_Kokkos via the PetscFEOps table */
    PetscCall(PetscDSSetResidual(ds, 0, f0_poisson, f1_poisson));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_poisson));
  }
  PetscCall(PetscDSSetExactSolution(ds, 0, u_exact, NULL));

  PetscCall(DMCreateLabel(dm, "marker"));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))u_exact, NULL, NULL, NULL));

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
  DM                 dm;
  SNES               snes;
  Vec                u;
  Mat                J;
  PetscReal          error;
  PetscFEKokkosMaps *maps            = NULL;
  PetscBool          use_kokkos_maps = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_kokkos_maps", &use_kokkos_maps, NULL));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(SetupDiscretization(dm, use_kokkos_maps));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));

  if (use_kokkos_maps) {
    /* GPU-resident template path */
    PetscCall(PetscFEKokkosMapsCreate(&maps));
    PetscCall(SNESSetFunction(snes, NULL, DMPlexSNESComputeResidualFEM_Kokkos<f0_poisson, f1_poisson>, maps));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(DMCreateMatrix(dm, &J));
    PetscCall(PetscFEKokkosSetUp(dm, maps, J));
    PetscCall(SNESSetJacobian(snes, J, J, (DMPlexSNESComputeJacobianFEM_Kokkos<nullptr, nullptr, nullptr, g3_poisson, true>), maps));
    PetscCall(MatDestroy(&J));
  } else {
    /* Fallback path: standard PETSc FEM assembly via PetscFEOps.
       PetscFEIntegrateResidual_Kokkos is called internally because the FE
       type is PETSCFEKOKKOS.  No PetscFEKokkosMaps needed. */
    PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
    PetscCall(DMCreateMatrix(dm, &J));
    PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
    PetscCall(MatDestroy(&J));
    PetscCall(SNESSetFromOptions(snes));
  }

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  PetscCall(VecSet(u, 0.0));
  PetscCall(SNESSolve(snes, NULL, u));

  {
    PetscErrorCode (*exactFuncs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *) = {u_exact};
    PetscCall(DMComputeL2Diff(dm, 0.0, exactFuncs, NULL, u, &error));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error: %g\n", (double)error));

  if (use_kokkos_maps) PetscCall(PetscFEKokkosMapsDestroy(&maps));
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST
  testset:
    requires: kokkos_kernels
    nsize: {{1 3}}
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
  # Fallback path: exercises PetscFEIntegrateResidual_Kokkos via the standard
  # DMPlexSNESComputeResidualFEM route (PetscFEOps table, no PetscFEKokkosMaps).
  testset:
    requires: kokkos
    nsize: {{1 3}}
    args: -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -dm_mat_type aij -dm_vec_type standard -ksp_type cg -pc_type jacobi -use_kokkos_maps 0
    filter: grep "L2 error"
    test:
      suffix: fallback_2d_quad
      output_file: output/ex_kokkos_fe_fallback_2d_quad.out
    test:
      suffix: fallback_2d_quad_deg2
      args: -petscspace_degree 2
      output_file: output/ex_kokkos_fe_fallback_2d_quad_deg2.out
    test:
      suffix: fallback_3d_hex
      args: -dm_plex_dim 3 -dm_plex_box_faces 2,2,2
      output_file: output/ex_kokkos_fe_fallback_3d_hex.out
TEST*/
