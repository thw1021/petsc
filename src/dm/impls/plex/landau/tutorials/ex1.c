
static char help[] = "Landau collision operator driver\n\n";

#include <petsc/private/dmpleximpl.h>
#include <petscts.h>

int main(int argc, char **argv)
{
  DM             dm;
  Vec            X,X_0;
  PetscErrorCode ierr;
  PetscInt       dim=2;
  TS             ts;
  Mat            J;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  SNESLineSearch linesearch;
  PetscReal      time;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  ierr = DMPlexLandCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &dm); CHKERRQ(ierr);
  ierr = DMPlexLandCreateMassMatrix(dm, X, NULL); CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&X_0);CHKERRQ(ierr);
  ierr = VecCopy(X,X_0);CHKERRQ(ierr);
  ierr = DMPlexLandPrintNorms(X,0);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 0, 0.0);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(X,NULL,"-vec_view");CHKERRQ(ierr);
  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(ts, "ex1_");CHKERRQ(ierr);  /* should get this from the dm or give it to the dm */
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes, "ex1_");CHKERRQ(ierr);  /* should get this from the dm or give it to the dm */
  ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,DMPlexLandIFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,DMPlexLandIJacobian,NULL);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "ex1_");CHKERRQ(ierr);  /* should get this from the dm or give it to the dm */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(pc, "ex1_");CHKERRQ(ierr);  /* should get this from the dm or give it to the dm */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = DMPlexLandPrintNorms(X,1);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 1, time);CHKERRQ(ierr);
  ierr = VecViewFromOptions(X,NULL,"-vec_view");CHKERRQ(ierr);
  ierr = VecAXPY(X,-1,X_0);CHKERRQ(ierr);
  /* clean up */
  ierr = DMPlexLandDestroyVelocitySpace(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&X_0);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

  test:
    suffix: 0
    requires: p4est !complex
    args: -petscspace_degree 4 -petscspace_poly_tensor 1 -dm_land_type p4est -info :dm,tsadapt -dm_land_ion_masses 2,4 -dm_land_ion_charges 1,18 -dm_land_thermal_temps 5,5,.5 -dm_land_n 1.00018,1,1e-5 -dm_land_n_0 1e20 -ex1_ts_monitor -ex1_snes_rtol 1.e-6 -ex1_snes_monitor -ex1_snes_converged_reason -ex1_ts_type arkimex -ex1_ts_arkimex_type 1bee -ex1_ts_max_snes_failures -1 -ex1_ts_rtol 1e-4 -ex1_ts_dt 1.e-6 -ex1_ts_max_time 1 -ex1_ts_adapt_clip .5,1.25 -ex1_ts_adapt_scale_solve_failed 0.75 -ex1_ts_adapt_time_step_increase_delay 5 -ex1_ts_max_steps 1 -ex1_pc_type lu -ex1_ksp_type preonly -dm_land_amr_levels_max 7 -dm_land_domain_radius 5 -dm_land_amr_re_levels 0 -dm_land_re_radius 1 -dm_land_amr_z_refine1 1 -dm_land_amr_z_refine2 0 -dm_land_amr_post_refine 0 -dm_land_z_radius1 .1 -dm_land_z_radius2 .1 -dm_refine 1

TEST*/
