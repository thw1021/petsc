module finitevolume

    use petscsys
    use petscdmplex
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdmplex.h>

    use input_parameter
    use geometry
    use riemannsolver
    use boundary

    implicit none

    ! Interface to access PetscFVSetComponentName which is missing
    ! in PETSc fortran interface
#include "petsc_wrapping/wrapper_petsc.h90"

    ! Prepare a "context" for the computations (for now, leave it empty)
    PetscScalar, dimension(0:nvar-1) :: context
    ! Manage FV object
    PetscFV                          :: fvM
    ! Manage Discrete System
    PetscDS                          :: prob

    contains

        subroutine initFV

            PetscErrorCode :: ierr

            call PetscPrintf(PETSC_COMM_WORLD, "Initializing spatial discretization:\n", ierr); CHKERRA(ierr)

            ! Generate the FV object
            call PetscFVCreate(PETSC_COMM_WORLD, fvM, ierr)          ; CHKERRA(ierr)
            call PetscFVSetFromOptions(fvM, ierr)                    ; CHKERRA(ierr)

            ! Set the number of degrees of freedom per cell
            call PetscFVSetNumComponents(fvM, nvar, ierr)            ; CHKERRA(ierr)

            ! Set the number of dimensions of the problem
            call PetscFVSetSpatialDimension(fvM, ndim, ierr)         ; CHKERRA(ierr)

            ! Set the names of 1) the FV handler & 2) the components for the outputs
            call PetscObjectSetName(fvM, "FV solver", ierr)          ; CHKERRA(ierr)
            call PetscFVSetComponentName(fvM, 0, "Density", ierr)    ; CHKERRA(ierr)
            call PetscFVSetComponentName(fvM, 1, "X-Momentum", ierr) ; CHKERRA(ierr)
            call PetscFVSetComponentName(fvM, 2, "Y-Momentum", ierr) ; CHKERRA(ierr)
            call PetscFVSetComponentName(fvM, 3, "Z-Momentum", ierr) ; CHKERRA(ierr)
            call PetscFVSetComponentName(fvM, 4, "Energy", ierr)     ; CHKERRA(ierr)

            ! Set the type of the FV object : least-square type for future MUSCL
            call PetscFVSetType(fvM, "leastsquares", ierr)           ; CHKERRA(ierr)

            ! Start setting up space discretization
            if (IDinit == 0) then
                ! First order - no need to compute gradient
                call PetscFVSetComputeGradients(fvM, PETSC_FALSE, ierr); CHKERRA(ierr)
            end if

            ! Link fvM to dm
            call DMAddField(dm, PETSC_NULL_DMLABEL, fvM, ierr)       ; CHKERRA(ierr)
            ! Create discrete system
            call DMCreateDS(dm, ierr)                                ; CHKERRA(ierr)
            ! Store in prob
            call DMGetDS(dm, prob, ierr)                             ; CHKERRA(ierr)
            ! Get Riemann solver
            call PetscDSSetRiemannSolver(prob, 0, RSChoice, ierr)    ; CHKERRA(ierr)
            ! Setup Boundary Conditions
            call setupBC(prob, context);
            ! Finish setting up DS
            call PetscDSSetFromOptions(prob, ierr)                   ; CHKERRA(ierr)

            if (debug) then
                ! Show in terminal
                call PetscPrintf(PETSC_COMM_WORLD, ":: [DEBUG] Visualizing FVM in console ::\n", ierr); CHKERRA(ierr)
                call PetscFVView(fvM, PETSC_VIEWER_STDOUT_WORLD, ierr)   ; CHKERRA(ierr)
                call PetscPrintf(PETSC_COMM_WORLD, ":: [DEBUG] Visualizing DS in console ::\n", ierr); CHKERRA(ierr)
                call PetscDSView(prob, PETSC_VIEWER_STDOUT_WORLD, ierr)  ; CHKERRA(ierr)
            end if

        end subroutine initFV

end module finitevolume