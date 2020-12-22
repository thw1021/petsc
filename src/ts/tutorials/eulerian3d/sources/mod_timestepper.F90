module timestepper

    use petscsys
    use petscts
    use petscdmplex
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscts.h>
#include <petsc/finclude/petscdmplex.h>

    use input_parameter
    use finitevolume ! >>> includes "petsc_wrapping/wrapper_petsc.h90" already <<<
    use geometry
    use output

    implicit none

    ! Manage TS object
    TS :: timeS

    contains

        subroutine initTS

            PetscErrorCode :: ierr
            TSAdapt        :: adapt

            call PetscPrintf(PETSC_COMM_WORLD, "Initializing temporal discretization...\n", ierr); CHKERRA(ierr)

            ! Generate the TS object
            call TSCreate(PETSC_COMM_WORLD, timeS, ierr); CHKERRA(ierr)

            ! Set up the actual time-stepping integrator wanted by the user
            if (RKorder == 1) then
                call TSSetType(timeS, TSEULER, ierr);               CHKERRA(ierr)
            else if (RKorder == 2) then
                call TSSetType(timeS, TSSSP, ierr);                 CHKERRA(ierr)
                call TSSSPSetType(timeS, TSSSPRKS2, ierr);          CHKERRA(ierr)
                call TSSSPSetNumStages(timeS, RKstep, ierr);        CHKERRA(ierr)
            else if (RKorder == 3) then
                call TSSetType(timeS, TSSSP, ierr);                 CHKERRA(ierr)
                call TSSSPSetType(timeS, TSSSPRKS3, ierr);          CHKERRA(ierr)
                call TSSSPSetNumStages(timeS, RKstep**2, ierr);     CHKERRA(ierr)
            else if (RKorder == 4) then
                call TSSetType(timeS, TSSSP, ierr);                 CHKERRA(ierr)
                call TSSSPSetType(timeS, TSSSPRK104, ierr);         CHKERRA(ierr)
            end if

            ! Connect time and spatial discretization
            call TSSetDM(timeS, dm, ierr);                          CHKERRA(ierr)

            ! Link time step and solution output (add a Monitor to the TS)
            call TSMonitorSet(timeS, iterMonitor, context, PETSC_NULL_FUNCTION, ierr);       CHKERRA(ierr)

            ! Manage BC in TS
            call DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, context, ierr);           CHKERRA(ierr)

            ! Compute RHS in TS
            call DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, context, ierr);  CHKERRA(ierr)

            ! How to compute time step at each iteration
            ! call TSAdaptRegister("CFL-Adapt", TSAdaptCreate_MyCFLAdapt, ierr) ;  CHKERRA(ierr)

            ! Finalize TS
            call TSSetMaxTime(timeS, tfinal, ierr);                             CHKERRA(ierr)
            call TSSetExactFinalTime(timeS, TS_EXACTFINALTIME_MATCHSTEP, ierr); CHKERRA(ierr)
            call TSSetMaxSteps(timeS, maxiter, ierr);                           CHKERRA(ierr)

            call PetscPrintf(PETSC_COMM_WORLD, "Done !\n", ierr); CHKERRA(ierr)

        end subroutine initTS

        subroutine marchTime

            PetscErrorCode         :: ierr
            PetscReal              :: ftime
            PetscInt               :: nsteps
            character(len=MSTRLEN) :: filename
            PetscViewer            :: vtkViewer

            call PetscPrintf(PETSC_COMM_WORLD, "Starting computation...\n", ierr); CHKERRA(ierr)

            ! Set time step
            call TSSetTimeStep(timeS, 1.d-3, ierr); CHKERRA(ierr)

            ! Finalize setting up time stepper
            call TSSetFromOptions(timeS, ierr);     CHKERRA(ierr)
            if (debug) then
                ! Show in terminal
                call PetscPrintf(PETSC_COMM_WORLD, ":: [DEBUG] Visualizing TS in console ::\n", ierr); CHKERRA(ierr)
                call TSView(timeS, PETSC_VIEWER_STDOUT_WORLD, ierr);                                   CHKERRA(ierr)
                call PetscPrintf(PETSC_COMM_WORLD, ":: [DEBUG] Visualizing DM in console ::\n", ierr); CHKERRA(ierr)
                call DMView(dm, PETSC_VIEWER_STDOUT_WORLD, ierr);                                      CHKERRA(ierr)
            end if

            ! Solve the problem
            call TSSolve(timeS, sol, ierr);                                                            CHKERRA(ierr)

            ! Get back the solver final time and final iteration number
            call TSGetSolveTime(timeS, ftime, ierr);                                                   CHKERRA(ierr)
            call TSGetStepNumber(timeS, nsteps, ierr);                                                 CHKERRA(ierr)

            ! VTK viewer for the final solution
            call PetscViewerCreate(PETSC_COMM_WORLD, vtkViewer, ierr);                                 CHKERRA(ierr)
            call PetscViewerSetType(vtkViewer, PETSCVIEWERVTK, ierr);                                  CHKERRA(ierr)
            write(filename,'(A,I5.5,A)') "solution_", nsteps, ".vtu"
            call PetscViewerFileSetName(vtkViewer, trim(filename), ierr);                              CHKERRA(ierr)
            call VecView(sol, vtkViewer, ierr);                                                        CHKERRA(ierr)
            call PetscViewerDestroy(vtkViewer, ierr);                                                  CHKERRA(ierr)

            call PetscPrintf(PETSC_COMM_WORLD, "Done !\n", ierr); CHKERRA(ierr)
        end subroutine marchTime

end module timestepper