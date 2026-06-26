!
!  Description: Test setting several callback functions from Fortran.
!
#include <petsc/finclude/petsc.h>
module ex2fmodule
  use petscvec
  use petscsnesdef
  use petscvec
  use petscmat
  implicit none

contains
!
! ------------------------------------------------------------------------
!
!  FormFunction - Evaluates nonlinear function, F(x).
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameter:
!  f - function vector
!
  subroutine FormFunction(snes, x, f, dummy, ierr)
    SNES snes
    Vec x, f
    PetscErrorCode, intent(out) :: ierr
    integer dummy(*)

!  Declarations for use with local arrays
    PetscScalar, pointer :: lx_v(:), lf_v(:)

!  Get pointers to vector data.
!    - VecGetArray() returns a pointer to the data array.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.

    PetscCall(VecGetArrayRead(x, lx_v, ierr))
    PetscCall(VecGetArray(f, lf_v, ierr))

!  Compute function

    lf_v(1) = lx_v(1)*lx_v(1) + lx_v(1)*lx_v(2) - 3.0
    lf_v(2) = lx_v(1)*lx_v(2) + lx_v(2)*lx_v(2) - 6.0

!  Restore vectors

    PetscCall(VecRestoreArrayRead(x, lx_v, ierr))
    PetscCall(VecRestoreArray(f, lf_v, ierr))

  end

! ---------------------------------------------------------------------
!
!  FormJacobian - Evaluates Jacobian matrix.
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameters:
!  A - Jacobian matrix
!  B - optionally different matrix used to construct the preconditioner
!
  subroutine FormJacobian(snes, X, jac, B, dummy, ierr)

    SNES snes
    Vec X
    Mat jac, B
    PetscScalar A(4)
    PetscErrorCode, intent(out) :: ierr
    PetscInt idx(2)
    integer dummy(*)

!  Declarations for use with local arrays

    PetscScalar, pointer :: lx_v(:)

!  Get pointer to vector data

    PetscCall(VecGetArrayRead(x, lx_v, ierr))

!  Compute Jacobian entries and insert into matrix.
!   - Since this is such a small problem, we set all entries for
!     the matrix at once.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C (as set here in the array idx).

    idx = [0, 1]
    A = [2.0*lx_v(1) + lx_v(2), lx_v(1), lx_v(2), lx_v(1) + 2.0*lx_v(2)]
    PetscCall(MatSetValues(B, 2_PETSC_INT_KIND, idx, 2_PETSC_INT_KIND, idx, A, INSERT_VALUES, ierr))

!  Restore vector

    PetscCall(VecRestoreArrayRead(x, lx_v, ierr))

!  Assemble matrix

    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY, ierr))
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY, ierr))
    if (B /= jac) then
      PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY, ierr))
      PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY, ierr))
    end if

  end

! ---------------------------------------------------------------------
!
!  MonitorDummy - Does nothing, used to test setting several callback functions in Fortran
!
  subroutine MonitorDummy(snes, its, norm, mctx, ierr)
    SNES, intent(in)  :: snes
    PetscInt, intent(in)  :: its
    PetscReal, intent(in)  :: norm
    integer, intent(in)  :: mctx
    PetscErrorCode, intent(out) :: ierr
    ierr = 0
  end subroutine MonitorDummy

end module

program main
  use petsc
  use ex2fmodule
  implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     snes        - nonlinear solver
!     x, r        - solution, residual vectors
!     J           - Jacobian matrix
!     its         - iterations for convergence
!
  SNES snes
  Vec x, r
  Mat J
  PetscErrorCode ierr
  PetscInt its
  PetscMPIInt size, rank
  PetscScalar, parameter :: pfive = 0.5
  PetscReal, parameter :: tol = 1.e-4
  double precision threshold, oldthreshold

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(PetscLogNestedBegin(ierr))
  threshold = 1.0
  PetscCallA(PetscLogSetThreshold(threshold, oldthreshold, ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))
  PetscCheckA(size == 1, PETSC_COMM_SELF, PETSC_ERR_WRONG_MPI_SIZE, 'Uniprocessor example')

! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(SNESCreate(PETSC_COMM_WORLD, snes, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix and vector data structures; set corresponding routines
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, 2_PETSC_INT_KIND, x, ierr))
  PetscCallA(VecDuplicate(x, r, ierr))

  PetscCallA(MatCreate(PETSC_COMM_SELF, J, ierr))
  PetscCallA(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2_PETSC_INT_KIND, 2_PETSC_INT_KIND, ierr))
  PetscCallA(MatSetFromOptions(J, ierr))
  PetscCallA(MatSetUp(J, ierr))

  PetscCallA(SNESSetFunction(snes, r, FormFunction, 0, ierr))

!  Uncomment the following line to set the Jacobian evaluation routine
!  PetscCallA(SNESSetJacobian(snes, J, J, FormJacobian, 0, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Test setting two more callback functions
  PetscCallA(SNESMonitorSet(snes, MonitorDummy, 0, PETSC_NULL_FUNCTION, ierr))

  PetscCallA(SNESSetFromOptions(snes, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(VecSet(x, pfive, ierr))
  PetscCallA(SNESSolve(snes, PETSC_NULL_VEC, x, ierr))

! View solver converged reason; we could instead use the option -snes_converged_reason
  PetscCallA(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(SNESGetIterationNumber(snes, its, ierr))
  if (rank == 0) then
    write (6, 100) its
  end if
100 format('Number of SNES iterations = ', i5)

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(r, ierr))
  PetscCallA(MatDestroy(J, ierr))
  PetscCallA(SNESDestroy(snes, ierr))
  PetscCallA(PetscFinalize(ierr))
end
!/*TEST
!
!   test:
!      args: -snes_type composite -snes_composite_type additiveoptimal -snes_composite_sneses anderson,nrichardson
!      requires: !single
!
!TEST*/
