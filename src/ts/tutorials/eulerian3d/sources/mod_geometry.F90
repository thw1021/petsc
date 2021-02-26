module geometry

    use petscsys
    use petscdmplex
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdmplex.h>

    use input_parameter

    implicit none

    ! Discretization manager to store unstructured mesh
    DM :: dm
    ! Solution vector
    Vec :: sol

    contains

        subroutine initmesh

            PetscErrorCode :: ierr
            DM             :: dmDist, dmGhost
            integer        :: overlap
            PetscViewer    :: vtkViewer

            ! Number of neighbours taken into account in MP communications(1 - Order 1; 2 - Order 2)
            overlap = 1

            call PetscPrintf(PETSC_COMM_WORLD, "Initializing mesh...\n", ierr)          ; CHKERRA(ierr)

            ! Force DMPlex to use gmsh marker
            ! call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-dm_plex_gmsh_use_marker", "true", ierr); CHKERRA(ierr)

            ! Read mesh from file name 'meshname'
            call DMPlexCreateFromFile(PETSC_COMM_WORLD, meshname, PETSC_TRUE, dm, ierr); CHKERRA(ierr)

            ! Distribute on processors
            ! Start with connectivity
            call DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE, ierr)                 ; CHKERRA(ierr)

            ! Distribute on processors
            call DMPlexDistribute(dm, overlap, PETSC_NULL_SF, dmDist, ierr)             ; CHKERRA(ierr)

            ! Security check
            if (dmDist /= PETSC_NULL_DM) then
                ! Destroy previous dm
                call DMDestroy(dm, ierr)                                                ; CHKERRA(ierr)
                ! Replace with dmDist
                dm = dmDist
            end if

            ! Finalize setup of the object
            call DMSetFromOptions(dm, ierr)                                             ; CHKERRA(ierr)

            ! Boundary condition with ghost cells
            call DMPlexConstructGhostCells(dm, PETSC_NULL_CHARACTER, PETSC_NULL_INTEGER, dmGhost, ierr); CHKERRA(ierr)

            ! Security check
            if (dmGhost /= PETSC_NULL_DM) then
                ! Destroy previous dm
                call DMDestroy(dm, ierr)                                                ; CHKERRA(ierr)
                ! Replace with dmGhost
                dm = dmGhost
            end if

            if (debug) then
                ! Show in terminal
                call PetscPrintf(PETSC_COMM_WORLD, ":: [DEBUG] Visualizing DM in console ::\n", ierr); CHKERRA(ierr)
                call DMView(dm, PETSC_VIEWER_STDOUT_WORLD, ierr)                        ; CHKERRA(ierr)
                ! VTK viewer
                call PetscViewerCreate(PETSC_COMM_WORLD, vtkViewer, ierr)               ; CHKERRA(ierr)
                call PetscViewerSetType(vtkViewer, PETSCVIEWERVTK, ierr)                ; CHKERRA(ierr)
                call PetscViewerFileSetName(vtkViewer, "debug_initmesh.vtk", ierr)      ; CHKERRA(ierr)
                call DMView(dm, vtkViewer, ierr)                                        ; CHKERRA(ierr)
                call PetscViewerDestroy(vtkViewer, ierr)                                ; CHKERRA(ierr)
            end if

            call PetscPrintf(PETSC_COMM_WORLD, "Done !\n", ierr)                        ; CHKERRA(ierr)

        end subroutine initmesh

end module geometry