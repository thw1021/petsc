
static char help[] = "Landau collision operator driver\n\n";

#include <petsc/private/dmpleximpl.h>
#include <petscts.h>

int main(int argc, char **argv)
{
  DM             dm;
  Vec            X,X_0;
  PetscErrorCode ierr;
  PetscInt       dim;
  TS             ts;
  Mat            J;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  SNESLineSearch linesearch;
  PetscScalar    time;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  dim = 2;
  ierr = DMPlexFPCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &dm); CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&X_0);CHKERRQ(ierr);
  ierr = VecCopy(X,X_0);CHKERRQ(ierr);
  ierr = DMPlexFPPrintNorms(X,0);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 0, 0.0);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(X,NULL,"-vec_view");CHKERRQ(ierr);
  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(ts, "fp_");CHKERRQ(ierr);  /* should get this from the dm or give it to the dm */
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes, "fp_");CHKERRQ(ierr);  /* should get this from the dm or give it to the dm */
  ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FPLandIFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FPLandIJacobian,NULL);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "fp_");CHKERRQ(ierr);  /* should get this from the dm or give it to the dm */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(pc, "fp_");CHKERRQ(ierr);  /* should get this from the dm or give it to the dm */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = DMPlexFPPrintNorms(X,1);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 1, time);CHKERRQ(ierr);
  ierr = VecViewFromOptions(X,NULL,"-vec_view");CHKERRQ(ierr);
  ierr = VecAXPY(X,-1,X_0);CHKERRQ(ierr);
  /* clean up */
  ierr = DMPlexFPDestroyPhaseSpace(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&X_0);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

  test:
    suffix: 0
    requires: p4est
    args: -petscspace_degree 4 -mass_petscspace_degree 4 -petscspace_poly_tensor 1 -mass_petscspace_poly_tensor 1 -dm_type p4est -info :dm,tsadapt -ion_masses 2,40 -ion_charges 1,18 -thermal_temps 5,5,.005 -n 1.00018,1,1e-5 -n_0 1e20 -fp_ts_monitor -fp_snes_rtol 1.e-6 -fp_snes_monitor -fp_snes_converged_reason -fp_ts_type arkimex -fp_ts_arkimex_type 1bee -fp_ts_max_snes_failures -1 -fp_ts_rtol 1e-4 -fp_ts_dt 1.e-6 -fp_ts_max_time 1 -fp_ts_adapt_clip .5,1.25 -fp_ts_adapt_scale_solve_failed 0.75 -fp_ts_adapt_time_step_increase_delay 5 -fp_ts_max_steps 1 -fp_pc_type lu -fp_ksp_type preonly -amr_levels_max 17 -domain_radius 5 -amr_re_levels 0 -re_radius 1 -amr_z_refine1 1 -amr_z_refine2 0 -amr_post_refine 0 -r0_radius1 .1 -r0_radius2 .1

TEST*/
