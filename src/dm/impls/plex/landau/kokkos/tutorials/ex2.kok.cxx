static char help[] = "Landau test\n\n";

#define PETSC_SKIP_CXX_COMPLEX_FIX
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <petscts.h>

#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[])
{
  DM             dm;
  Vec            X;
  PetscErrorCode ierr;
  PetscInt       dim = 2;
  TS             ts;
  Mat            J;
  PetscDS        prob;
  LandCtx        *ctx;
  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  Kokkos::initialize(argc, argv);
  ierr = PetscOptionsGetInt(NULL,NULL, "-ex2_dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  ierr = DMPlexLandCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &dm); CHKERRQ(ierr);
  ierr = DMPlexLandCreateMassMatrix(dm, X, NULL); CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 0, 0.0);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-ex2_dm_view");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-ex2_dm_view_sources");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-ex2_dm_view_diff");CHKERRQ(ierr);
  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,DMPlexLandIFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,DMPlexLandIJacobian,NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts, ctx);CHKERRQ(ierr);
  ierr = MatSetOption(J, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
  if (1) {
    PetscLogStage stage;
    Vec           vec;
    PetscRandom   rctx;
    ierr = PetscLogStageRegister("Warmup", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecDuplicate(X,&vec);CHKERRQ(ierr);
    ierr = VecSetRandom(vec,rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
    ierr = DMPlexLandIJacobian(ts,0.0,vec,vec,1.0,J,J,ctx);CHKERRQ(ierr);
    ierr = DMPlexLandFormLandau_Internal(X,ctx->J,dim,(void*)ctx);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  ierr = VecViewFromOptions(X,NULL,"-ex2_x_vec_view");CHKERRQ(ierr);
  /* go */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  /* clean up */
  ierr = DMPlexLandDestroyVelocitySpace(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  Kokkos::finalize();
  ierr = PetscFinalize();
}

/*TEST

  test:
    requires: kokkos
    suffix: 0
    args: -dm_land_device_type kokkos -petscspace_degree 4 -petscspace_poly_tensor 1 -dm_land_type p4est -info :dm,tsadapt -dm_land_ion_masses 2,4 -dm_land_ion_charges 1,18 -dm_land_thermal_temps 5,5,.5 -dm_land_n 1.00018,1,1e-5 -dm_land_n_0 1e20 -ex2_ts_monitor -ex2_snes_rtol 1.e-6 -ex2_snes_monitor -ex2_snes_converged_reason -ex2_ts_type arkimex -ex2_ts_arkimex_type 1bee -ex2_ts_max_snes_failures -1 -ex2_ts_rtol 1e-4 -ex2_ts_dt 1.e-6 -ex2_ts_max_time 1 -ex2_ts_adapt_clip .5,1.25 -ex2_ts_adapt_scale_solve_failed 0.75 -ex2_ts_adapt_time_step_increase_delay 5 -ex2_ts_max_steps 1 -ex2_pc_type lu -ex2_ksp_type preonly -dm_land_amr_levels_max 7 -dm_land_domain_radius 5 -dm_land_amr_re_levels 0 -dm_land_re_radius 1 -dm_land_amr_z_refine1 1 -dm_land_amr_z_refine2 0 -dm_land_amr_post_refine 0 -dm_land_z_radius1 .1 -dm_land_z_radius2 .1

TEST*/
