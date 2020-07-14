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
  external DMPlexLandIFunction
  external DMPlexLandIJacobian
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
  call DMPlexLandCreateMassMatrix(dm, X, PETSC_NULL_MAT, ierr); CHKERRA(ierr)
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
  call TSSetOptionsPrefix(ts, 'ex1f90_', ierr);CHKERRA(ierr) ! should get this from the dm or give it to the dm
  call TSSetDM(ts,dm,ierr);CHKERRA(ierr)
  call TSGetSNES(ts,snes,ierr);CHKERRA(ierr)
  call SNESSetOptionsPrefix(snes, 'ex1f90_', ierr);CHKERRA(ierr) ! should get this from the dm or give it to the dm
  call SNESGetLineSearch(snes,linesearch,ierr);CHKERRA(ierr)
  call SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC,ierr);CHKERRA(ierr)
  call TSSetIFunction(ts,PETSC_NULL_VEC,DMPlexLandIFunction,PETSC_NULL_VEC,ierr);CHKERRA(ierr)
  call TSSetIJacobian(ts,J,J,DMPlexLandIJacobian,PETSC_NULL_VEC,ierr);CHKERRA(ierr)
  call TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER,ierr);CHKERRA(ierr)

  call SNESGetKSP(snes,ksp,ierr);CHKERRA(ierr)
  call KSPSetOptionsPrefix(ksp, 'ex1f90_', ierr);CHKERRA(ierr) ! should get this from the dm or give it to the dm
  call KSPGetPC(ksp,pc,ierr);CHKERRA(ierr)
  call PCSetOptionsPrefix(pc, 'ex1f90_', ierr);CHKERRA(ierr) ! should get this from the dm or give it to the dm

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
!    args: -petscspace_degree 4 -petscspace_poly_tensor 1 -dm_land_type p4est -info :dm,tsadapt -dm_land_ion_masses 2,4 -dm_land_ion_charges 1,8 -dm_land_thermal_temps 5,5,.5 -dm_land_n 1.00018,1,1e-5 -dm_land_n_0 1e20 -ex1f90_ts_monitor -ex1f90_snes_rtol 1.e-6 -ex1f90_snes_monitor -ex1f90_snes_converged_reason -ex1f90_ts_type arkimex -ex1f90_ts_arkimex_type 1bee -ex1f90_ts_max_snes_failures -1 -ex1f90_ts_rtol 1e-4 -ex1f90_ts_dt 1.e-6 -ex1f90_ts_max_time 1 -ex1f90_ts_adapt_clip .5,1.25 -ex1f90_ts_adapt_scale_solve_failed 0.75 -ex1f90_ts_adapt_time_step_increase_delay 5 -ex1f90_ts_max_steps 1 -ex1f90_pc_type lu -ex1f90_ksp_type preonly -dm_land_amr_levels_max 7 -dm_land_domain_radius 5 -dm_land_amr_re_levels 0 -dm_land_re_radius 1 -dm_land_amr_z_refine1 1 -dm_land_amr_z_refine2 0 -dm_land_amr_post_refine 0 -dm_land_z_radius1 .1 -dm_land_z_radius2 .1 -dm_land_device_type cpu
!
!TEST*/
