static char help[] = "3D linear elasticity (Nc=3) with Kokkos GPU-resident FEM assembly.\n"
                     "Stripped-down clone of ex56.c to test the Kokkos template kernel path\n"
                     "for vector FE (Nc>1), exercising g3 Jacobian assembly and f1 residual.\n"
                     "\n"
                     "Options:\n"
                     "  -use_kokkos_maps <bool>  use GPU-resident template path (default: true)\n"
                     "  -max_conv_its <n>        refinement levels for convergence test (default: 1)\n"
                     "  -mu     <val>            Lame mu     (default: 0.4)\n"
                     "  -lambda <val>            Lame lambda (default: 0.4)\n"
                     "  -ksp_type cg -pc_type gamg  recommended solver\n";

/*
  3D linear elasticity on [0,1]^3 with manufactured solution body force.
  Matches ex56.c run_type=1 (elasticity convergence test).

  Strong form:
    -div(sigma(u)) = f   in Omega
    u = 0                on boundary

  where sigma_ij = mu*(u_{i,j} + u_{j,i}) + lambda * tr(grad u) * delta_ij

  Manufactured body force f0 (same as f0_u_x4 in ex56.c):
    f0[comp] = 1e5 * prod_{i=0}^{dim-1} (x[i]^4 - x[i]^2)
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscfe.h>
#include <Kokkos_Core.hpp>
#include <petscfekokkos.h>

/* Material constants -- set from options in main() via a context struct */
typedef struct {
  PetscReal mu;
  PetscReal lambda;
} ElastCtx;

/* Global constants used by KOKKOS_INLINE_FUNCTION callbacks.
   These must be compile-time accessible on device; we use constexpr globals
   initialised to the ex56 defaults and overridden at runtime only for the
   fallback path (where the callbacks are plain function pointers). */
static PetscReal s_mu     = 0.4;
static PetscReal s_lambda = 0.4;

/* f0: manufactured body force  f0[comp] = 1e5 * prod_i (x_i^4 - x_i^2)
   For elasticity Nc == dim, so we loop over dim components. */
KOKKOS_INLINE_FUNCTION
static void f0_elast(PETSC_POINT_ARGS, PetscScalar f0[])
{
  PetscReal prod = 1.0;
  for (PetscInt i = 0; i < dim; ++i) prod *= x[i] * x[i] * x[i] * x[i] - x[i] * x[i];
  for (PetscInt comp = 0; comp < dim; ++comp) f0[comp] = (PetscScalar)(1e5 * prod);
}

/* f1: stress tensor  sigma_ij = mu*(u_{i,j}+u_{j,i}) + lambda*tr(grad u)*delta_ij
   f1 is stored as f1[i*dim + j] = sigma_ij */
KOKKOS_INLINE_FUNCTION
static void f1_elast(PETSC_POINT_ARGS, PetscScalar f1[])
{
  PetscReal trace = 0.0;
  for (PetscInt i = 0; i < dim; ++i) trace += PetscRealPart(u_x[i * dim + i]);
  for (PetscInt i = 0; i < dim; ++i) {
    for (PetscInt j = 0; j < dim; ++j) f1[i * dim + j] = (PetscScalar)(s_mu * (PetscRealPart(u_x[i * dim + j]) + PetscRealPart(u_x[j * dim + i])));
    f1[i * dim + i] += (PetscScalar)(s_lambda * trace);
  }
}

/* g3: elasticity tensor  C_ijkl = lambda*delta_ij*delta_kl + mu*(delta_ik*delta_jl + delta_il*delta_jk)
   Stored as g3[((i*dim+j)*dim+k)*dim+l] */
KOKKOS_INLINE_FUNCTION
static void g3_elast(PETSC_JAC_POINT_ARGS, PetscScalar g3[])
{
  for (PetscInt i = 0; i < dim; ++i) {
    for (PetscInt j = 0; j < dim; ++j) {
      for (PetscInt k = 0; k < dim; ++k) {
        for (PetscInt l = 0; l < dim; ++l) {
          const PetscInt idx = ((i * dim + j) * dim + k) * dim + l;
          if (i == j && k == l) g3[idx] += (PetscScalar)s_lambda;
          if (i == k && j == l) g3[idx] += (PetscScalar)s_mu;
          if (i == l && j == k) g3[idx] += (PetscScalar)s_mu;
        }
      }
    }
  }
}

static PetscErrorCode SetupDiscretization(DM dm, PetscBool use_kokkos_maps)
{
  PetscFE        fe;
  PetscDS        ds;
  DMLabel        label;
  PetscInt       dim;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  /* 3-component (vector) Lagrange element on hexahedra */
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, PETSC_FALSE, NULL, -1, &fe));
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
    /* Fallback path: standard PETSc FEM assembly via PetscFEOps table */
    PetscCall(PetscDSSetResidual(ds, 0, f0_elast, f1_elast));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_elast));
  }

  /* Zero Dirichlet on all boundary faces */
  PetscCall(DMCreateLabel(dm, "marker"));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, NULL, NULL, NULL, NULL));

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
  PetscReal         mdisp[8];
  PetscInt          sizes[8];
  PetscFEKokkosMaps maps;
  PetscBool         use_kokkos_maps = PETSC_TRUE;
  PetscInt          max_conv_its    = 1, iter;
  MPI_Comm          comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_kokkos_maps", &use_kokkos_maps, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &s_mu, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &s_lambda, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_conv_its", &max_conv_its, NULL));
  PetscCheck(max_conv_its > 0 && max_conv_its < 8, comm, PETSC_ERR_USER, "Bad number of iterations for convergence test (%" PetscInt_FMT ")", max_conv_its);

  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(SetupDiscretization(dm, use_kokkos_maps));

  for (iter = 0; iter < max_conv_its; iter++) {
    PetscCall(SNESCreate(comm, &snes));
    PetscCall(SNESSetDM(snes, dm));

    if (use_kokkos_maps) {
      /* GPU-resident template path.
         PetscFEKokkosPreallocateCOO requires MATAIJKOKKOS: pass -dm_mat_type aijkokkos
         and -dm_vec_type kokkos on the command line (or set them in the options file). */
      PetscCall(SNESSetFunction(snes, NULL, DMPlexSNESComputeResidualFEM_Kokkos<f0_elast, f1_elast>, &maps));
      PetscCall(SNESSetFromOptions(snes));
      PetscCall(DMCreateMatrix(dm, &J));
      PetscCall(PetscFEKokkosSetUp(dm, &maps, J));
      PetscCall(SNESSetJacobian(snes, J, J, (DMPlexSNESComputeJacobianFEM_Kokkos<nullptr, nullptr, nullptr, g3_elast, true>), &maps));
      PetscCall(MatDestroy(&J));
    } else {
      /* Fallback path: standard PETSc FEM assembly via PetscFEOps */
      PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
      PetscCall(DMCreateMatrix(dm, &J));
      PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
      PetscCall(MatDestroy(&J));
      PetscCall(SNESSetFromOptions(snes));
    }

    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(PetscObjectSetName((PetscObject)u, "u"));
    PetscCall(VecSet(u, 0.0));
    PetscCall(VecGetSize(u, &sizes[iter]));
    PetscCall(SNESSolve(snes, NULL, u));

    PetscCall(VecNorm(u, NORM_INFINITY, &mdisp[iter]));

    if (use_kokkos_maps) PetscCall(PetscFEKokkosMapsDestroy(&maps));
    PetscCall(VecDestroy(&u));
    PetscCall(SNESDestroy(&snes));

    /* Refine for next iteration */
    if (iter + 1 < max_conv_its) {
      DM newdm;
      PetscCall(DMRefine(dm, comm, &newdm));
      PetscCall(DMDestroy(&dm));
      dm = newdm;
      PetscCall(DMSetFromOptions(dm));
      PetscCall(SetupDiscretization(dm, use_kokkos_maps));
    }
  }
  PetscCall(DMDestroy(&dm));

  /* Print convergence results */
  if (max_conv_its == 1) PetscCall(PetscPrintf(comm, "Max displacement: %g\n", (double)mdisp[0]));
  else {
    PetscCall(PetscPrintf(comm, "%" PetscInt_FMT ") N=%12" PetscInt_FMT ", max displ=%9.7e\n", (PetscInt)0, sizes[0], (double)mdisp[0]));
    for (iter = 1; iter < max_conv_its; iter++) {
      PetscReal rate = PetscLogReal(PetscAbs(mdisp[iter - 1] - mdisp[iter]) / PetscAbs(mdisp[iter] - mdisp[PetscMin(iter + 1, max_conv_its - 1)]));
      if (iter + 1 < max_conv_its) rate /= PetscLogReal(2.0);
      else rate = 0.0; /* no rate for last level */
      PetscCall(PetscPrintf(comm, "%" PetscInt_FMT ") N=%12" PetscInt_FMT ", max displ=%9.7e, disp diff=%9.2e, rate=%3.2g\n", iter, sizes[iter], (double)mdisp[iter], (double)(mdisp[iter] - mdisp[iter - 1]), (double)rate));
    }
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST
  testset:
    requires: kokkos_kernels !single
    nsize: 3
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -dm_plex_box_faces 2,2,1 -petscspace_degree 2 -snes_max_it 1 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-10 -ksp_norm_type unpreconditioned -pc_type gamg -pc_gamg_coarse_eq_limit 10 -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.001 -mg_levels_ksp_max_it 2 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -snes_type ksponly
    filter: grep "Max displacement"
    test:
      suffix: kokkos_maps
      args: -dm_mat_type aijkokkos -dm_vec_type kokkos -use_kokkos_maps 1
      output_file: output/ex56k_1.out
    test:
      suffix: fallback
      args: -dm_mat_type aij -dm_vec_type standard -use_kokkos_maps 0
      output_file: output/ex56k_1.out
  testset:
    requires: kokkos_kernels !single
    suffix: conv
    nsize: {{1 3}}
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -dm_plex_box_faces 2,2,1 -petscspace_degree 2 -max_conv_its 3 -snes_max_it 1 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-10 -ksp_norm_type unpreconditioned -pc_type gamg -pc_gamg_coarse_eq_limit 10 -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.001 -mg_levels_ksp_max_it 2 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -snes_type ksponly -dm_mat_type aijkokkos -dm_vec_type kokkos -use_kokkos_maps 1
    filter: grep "max displ"
    output_file: output/ex56k_conv.out
TEST*/
