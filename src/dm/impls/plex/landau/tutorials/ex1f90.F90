! test phase space (Maxwellian) mesh construction (serial)
!
!run:
!	-${MPIEXEC} ....
!	-@${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py *.h5
!
!
! Contributed by Mark Adams
program DMPlexTestLandInterface
  use petscts
  use petscdmplex
#include <petsc/finclude/petscts.h>
#include <petsc/finclude/petscdmplex.h>
  implicit none
  external LandIFunction
  external LandIJacobian
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
  PetscReal      mone
  PetscScalar    scalar
  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  if (ierr .ne. 0) then
     print*,'Unable to initialize PETSc'
     stop
  endif
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Create mesh (DM), read in parameters, create and add f_0 (X)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  dim = 2
  call DMPlexLandCreateVelocitySpace(PETSC_COMM_SELF, dim, '', X, J, dm, ierr); CHKERRA(ierr)
  call DMSetUp(dm,ierr);CHKERRA(ierr)
  call VecDuplicate(X,X_0,ierr);CHKERRA(ierr)
  call VecCopy(X,X_0,ierr)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  View
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ii = 0
  call DMPlexLandPrintNorms(X,ii,ierr);CHKERRA(ierr)
  mone = 0;
  call DMSetOutputSequenceNumber(dm, ii, mone, ierr);CHKERRA(ierr);
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !    Create timestepping solver context
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call TSCreate(PETSC_COMM_SELF,ts,ierr);CHKERRA(ierr)
  call TSSetOptionsPrefix(ts, 'land_', ierr);CHKERRA(ierr) ! should get this from the dm or give it to the dm
  call TSSetDM(ts,dm,ierr);CHKERRA(ierr)
  call TSGetSNES(ts,snes,ierr);CHKERRA(ierr)
  call SNESSetOptionsPrefix(snes, 'land_', ierr);CHKERRA(ierr) ! should get this from the dm or give it to the dm
  call SNESGetLineSearch(snes,linesearch,ierr);CHKERRA(ierr)
  call SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC,ierr);CHKERRA(ierr)
  call TSSetIFunction(ts,PETSC_NULL_VEC,LandIFunction,PETSC_NULL_VEC,ierr);CHKERRA(ierr)
  call TSSetIJacobian(ts,J,J,LandIJacobian,PETSC_NULL_VEC,ierr);CHKERRA(ierr)
  call TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER,ierr);CHKERRA(ierr)

  call SNESGetKSP(snes,ksp,ierr);CHKERRA(ierr)
  call KSPSetOptionsPrefix(ksp, 'land_', ierr);CHKERRA(ierr) ! should get this from the dm or give it to the dm
  call KSPGetPC(ksp,pc,ierr);CHKERRA(ierr)
  call PCSetOptionsPrefix(pc, 'land_', ierr);CHKERRA(ierr) ! should get this from the dm or give it to the dm

  call TSSetFromOptions(ts,ierr);CHKERRA(ierr)
  call TSSetSolution(ts,X,ierr);CHKERRA(ierr)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Solve nonlinear system
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call TSSolve(ts,X,ierr);CHKERRA(ierr)
  ii = 1
  call DMPlexLandPrintNorms(X,ii,ierr);CHKERRA(ierr)
  call TSGetTime(ts, mone, ierr);CHKERRA(ierr);
  call DMSetOutputSequenceNumber(dm, ii, mone, ierr);CHKERRA(ierr);
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  remove f_0
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  scalar = -1.
  call VecAXPY(X,scalar,X_0,ierr);CHKERRA(ierr)
  call DMPlexLandDestroyVelocitySpace(dm, ierr);CHKERRA(ierr)
  call TSDestroy(ts, ierr);CHKERRA(ierr)
  call VecDestroy(X, ierr);CHKERRA(ierr)
  call VecDestroy(X_0, ierr);CHKERRA(ierr)
  call PetscFinalize(ierr)
end program DMPlexTestLandInterface

!/*TEST
!  build:
!    requires: define(PETSC_USING_F90FREEFORM)
!
!  test:
!    suffix: 0
!    requires: p4est !complex
!    args: -petscspace_degree 4 -petscspace_poly_tensor 1 -dm_type p4est -info :dm,tsadapt -ion_masses 2,4 -ion_charges 1,8 -thermal_temps 5,5,.5 -n 1.00018,1,1e-5 -n_0 1e20 -land_ts_monitor -land_snes_rtol 1.e-6 -land_snes_monitor -land_snes_converged_reason -land_ts_type arkimex -land_ts_arkimex_type 1bee -land_ts_max_snes_failures -1 -land_ts_rtol 1e-4 -land_ts_dt 1.e-6 -land_ts_max_time 1 -land_ts_adapt_clip .5,1.25 -land_ts_adapt_scale_solve_failed 0.75 -land_ts_adapt_time_step_increase_delay 5 -land_ts_max_steps 1 -land_pc_type lu -land_ksp_type preonly -amr_levels_max 7 -domain_radius 5 -amr_re_levels 0 -re_radius 1 -amr_z_refine1 1 -amr_z_refine2 0 -amr_post_refine 0 -z_radius1 .1 -z_radius2 .1
!
!TEST*/
