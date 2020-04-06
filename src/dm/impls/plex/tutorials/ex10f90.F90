! test phase space (Maxwellian) mesh construction (serial)
!
!run:
!	-${MPIEXEC} ....
!	-@${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py *.h5
!
!
! Contributed by Mark Adams
program DMPlexTestFPInterface
  use petscts
  use petscdmplex
#include <petsc/finclude/petscts.h>
#include <petsc/finclude/petscdmplex.h>
  implicit none
  external FPLandIFunction
  external FPLandIJacobian
  DM             dm
  PetscInt       dim
  PetscInt       ii
  PetscErrorCode ierr
  TS             ts
  Vec            X,X_0
  Mat            J
  SNES           snes
  KSP            ksp
  PC             pc
  SNESLineSearch linesearch
  PetscScalar    mone

  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  if (ierr .ne. 0) then
     print*,'Unable to initialize PETSc'
     stop
  endif
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Create mesh (DM), read in parameters, create and add f_0 (X)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  dim = 2
  call DMPlexFPCreateVelocitySpace(PETSC_COMM_SELF, dim, '', X, dm, ierr); CHKERRQ(ierr)
  call DMSetUp(dm,ierr);CHKERRQ(ierr)
  call VecDuplicate(X,X_0,ierr);CHKERRQ(ierr)
  call VecCopy(X,X_0,ierr)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  View
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ii = 0
  call DMPlexFPPrintNorms(X,ii,ierr);CHKERRQ(ierr)
  mone = 0;
  call DMSetOutputSequenceNumber(dm, ii, mone, ierr);CHKERRQ(ierr);
  call PetscObjectViewFromOptions(dm,PETSC_NULL_VEC,'-dm_view',ierr);CHKERRQ(ierr)
  call PetscObjectViewFromOptions(X,PETSC_NULL_VEC,'-vec_view',ierr);CHKERRQ(ierr)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !    Create timestepping solver context
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call TSCreate(PETSC_COMM_SELF,ts,ierr);CHKERRQ(ierr)
  call PetscObjectSetOptionsPrefix(ts, 'fp_', ierr);CHKERRQ(ierr) ! should get this from the dm or give it to the dm
  call TSSetDM(ts,dm,ierr);CHKERRQ(ierr)
  call TSGetSNES(ts,snes,ierr);CHKERRQ(ierr)
  call PetscObjectSetOptionsPrefix(snes, 'fp_', ierr);CHKERRQ(ierr) ! should get this from the dm or give it to the dm
  call SNESGetLineSearch(snes,linesearch,ierr);CHKERRQ(ierr)
  call SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC,ierr);CHKERRQ(ierr)
  call DMCreateMatrix(dm, J, ierr);CHKERRQ(ierr);
  call TSSetIFunction(ts,PETSC_NULL_VEC,FPLandIFunction,PETSC_NULL_VEC,ierr);CHKERRQ(ierr)
  call TSSetIJacobian(ts,J,J,FPLandIJacobian,PETSC_NULL_VEC,ierr);CHKERRQ(ierr)
  call TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER,ierr);CHKERRQ(ierr)

  call SNESGetKSP(snes,ksp,ierr);CHKERRQ(ierr)
  call PetscObjectSetOptionsPrefix(ksp, 'fp_', ierr);CHKERRQ(ierr) ! should get this from the dm or give it to the dm
  call KSPGetPC(ksp,pc,ierr);CHKERRQ(ierr)
  call PetscObjectSetOptionsPrefix(pc, 'fp_', ierr);CHKERRQ(ierr) ! should get this from the dm or give it to the dm

  call TSSetFromOptions(ts,ierr);CHKERRQ(ierr)
  call TSSetSolution(ts,X,ierr);CHKERRQ(ierr)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Solve nonlinear system
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call TSSolve(ts,X,ierr);CHKERRQ(ierr)
  ! call TSGetSolution(ts, X, ierr);CHKERRQ(ierr);
  ii = 1
  call DMPlexFPPrintNorms(X,ii,ierr);CHKERRQ(ierr)
  call TSGetTime(ts, mone, ierr);CHKERRQ(ierr);
  call DMSetOutputSequenceNumber(dm, ii, mone, ierr);CHKERRQ(ierr);
  call PetscObjectViewFromOptions(X,PETSC_NULL_VEC,'-vec_view',ierr);CHKERRQ(ierr)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  remove f_0
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  mone = -1.
  call VecAXPY(X,mone,X_0,ierr);CHKERRQ(ierr)
  call DMPlexFPDestroyPhaseSpace(dm, ierr);CHKERRQ(ierr)
  call TSDestroy(ts, ierr);CHKERRQ(ierr)
  call MatDestroy(J, ierr);CHKERRQ(ierr)
  call VecDestroy(X, ierr);CHKERRQ(ierr)
  call VecDestroy(X_0, ierr);CHKERRQ(ierr)
  call PetscFinalize(ierr)
end program DMPlexTestFPInterface

!/*TEST
!  build:
!    requires: define(PETSC_USING_F90FREEFORM)
!
!  test:
!    suffix: 0
!    requires: p4est
!    args: -petscspace_degree 4 -mass_petscspace_degree 4 -petscspace_poly_tensor 1 -mass_petscspace_poly_tensor 1 -dm_type p4est -info :dm,tsadapt -ion_masses 2,40 -ion_charges 1,18 -thermal_temps 5,5,.005 -n 1.00018,1,1e-5 -n_0 1e20 -fp_ts_monitor -fp_snes_rtol 1.e-6 -fp_snes_monitor -fp_snes_converged_reason -fp_ts_type arkimex -fp_ts_arkimex_type 1bee -fp_ts_max_snes_failures -1 -fp_ts_rtol 1e-4 -fp_ts_dt 1.e-6 -fp_ts_max_time 1 -fp_ts_adapt_clip .5,1.25 -fp_ts_adapt_scale_solve_failed 0.75 -fp_ts_adapt_time_step_increase_delay 5 -fp_ts_max_steps 1 -fp_pc_type lu -fp_ksp_type preonly -amr_levels_max 17 -domain_radius 5 -amr_re_levels 0 -re_radius 1 -amr_z_refine1 1 -amr_z_refine2 0 -amr_post_refine 0 -r0_radius1 .1 -r0_radius2 .1
!
!TEST*/
