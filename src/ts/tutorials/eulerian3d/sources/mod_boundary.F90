module boundary

    use petscsys
    use petscdmplex
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdmplex.h>

    use input_parameter

    implicit none

    PetscInt                               :: nSlipwall, nOutflow, nFreestream, nBC
    PetscInt, dimension(:), allocatable    :: slipwall_ids, outflow_ids, freestream_ids
    PetscReal, dimension(:,:), allocatable :: freestream_params

    contains

    subroutine setupBC(prob, ctx)

        PetscDS                   :: prob
        PetscScalar, dimension(:) :: ctx
        PetscErrorCode            :: ierr
        ! Number of process
        PetscInt                  :: rank
        integer                   :: dummyint, iBC, iFS, index
        character(len=MSTRLEN)    :: string
        logical                   :: file_exists

        ! Put in rank the number of processus
        call MPI_COMM_RANK(PETSC_COMM_WORLD, rank, ierr) ; CHKERRA(ierr)

        ! Only rank 0 will read
        if (rank == 0) then
            ! open mesh.bc
            inquire(file="mesh.bc", exist=file_exists)
            ! If it exists, read it and store the data
            if (file_exists) then
                ! Count number of each BC
                nSlipwall   = 0
                nOutflow    = 0
                nFreestream = 0
                open(112, file="mesh.bc")
                read(112,*) nBC
                do iBC = 1, nBC
                    read(112,*) dummyint, string
                    if (trim(string) == "slipwall") then
                        nSlipwall   = nSlipwall + 1
                    else if (trim(string) == "outflowsupersonic") then
                        nOutflow    = nOutflow + 1
                    else if (trim(string) == "freestream") then
                        nFreestream = nFreestream + 1
                    end if
                end do
                close(112)
                ! Debug messages
                if (debug) then
                    call PetscPrintf(PETSC_COMM_WORLD, ":: [DEBUG] Numbers of boundary conditions found ::\n", ierr); CHKERRA(ierr)
                    if (rank == 0) then
                        write(*,"(A,X,I0,X,A)") ":: [DEBUG]", nSlipwall, "slipwall's ::"
                        write(*,"(A,X,I0,X,A)") ":: [DEBUG]", nOutflow, "outflow's ::"
                        write(*,"(A,X,I0,X,A)") ":: [DEBUG]", nFreestream, "freestream's ::"
                    end if
                end if
                ! Allocate the BC id list
                allocate(slipwall_ids(0:nSlipwall-1))
                allocate(outflow_ids(0:nOutflow-1))
                allocate(freestream_ids(0:nFreestream-1))
                allocate(freestream_params(0:nFreestream-1, 0:nvar-1))
                ! Fill in the list
                nSlipwall   = 0
                nOutflow    = 0
                nFreestream = 0
                open(112, file="mesh.bc")
                read(112,*) nBC
                do iBC = 1, nBC
                    read(112,*) dummyint, string
                    if (trim(string) == "slipwall") then
                        slipwall_ids(nSlipwall)     = dummyint
                        nSlipwall   = nSlipwall + 1
                    else if (trim(string) == "outflowsupersonic") then
                        outflow_ids(nOutflow)       = dummyint
                        nOutflow    = nOutflow + 1
                    else if (trim(string) == "freestream") then
                        freestream_ids(nFreestream) = dummyint
                        nFreestream = nFreestream + 1
                    end if
                end do
                if (nFreestream >= 1) then
                    do iFS = 0, nFreestream-1
                        read(112,*) (freestream_params(iFS, index), index = 0, nvar-1)
                    end do
                end if
                close(112)
            else
                write(*,'(A,A,A)') 'File ', "mesh.bc", ' does not exist'
                stop
            end if
        end if ! (rank == 0)

        ! Now broadcast to the other MPI processes
        call MPI_BCAST(nSlipwall, 1, MPI_INT, 0, PETSC_COMM_WORLD, ierr)    ; CHKERRA(ierr)
        call MPI_BCAST(nOutflow, 1, MPI_INT, 0, PETSC_COMM_WORLD, ierr)     ; CHKERRA(ierr)
        call MPI_BCAST(nFreestream, 1, MPI_INT, 0, PETSC_COMM_WORLD, ierr)  ; CHKERRA(ierr)
        if (rank /= 0) then
            allocate(slipwall_ids(0:nSlipwall-1))
            allocate(outflow_ids(0:nOutflow-1))
            allocate(freestream_ids(0:nFreestream-1))
            allocate(freestream_params(0:nFreestream-1, 0:nvar-1))
        end if
        call MPI_BCAST(slipwall_ids, nSlipwall, MPI_INT, 0, PETSC_COMM_WORLD, ierr)     ; CHKERRA(ierr)
        call MPI_BCAST(outflow_ids, nOutflow, MPI_INT, 0, PETSC_COMM_WORLD, ierr)       ; CHKERRA(ierr)
        call MPI_BCAST(freestream_ids, nFreestream, MPI_INT, 0, PETSC_COMM_WORLD, ierr) ; CHKERRA(ierr)
        if (nFreestream >= 1) then
            do iFS = 0, nFreestream-1
                call MPI_BCAST(freestream_params(iFS, 0:nvar-1), nvar, MPI_DOUBLE, 0, PETSC_COMM_WORLD, ierr) ; CHKERRA(ierr)
            end do
        end if

        ! Now tell PETSc which IDs have to be treated with which bc function
        if (nSlipwall > 0) then
            ! 10 stands for 'BC_DM_NATURAL_RIEMANN', i.e. a BC using ghost cells for the Riemann problem
            call PetscDSAddBoundary(prob, 10, "slipwall", "Face Sets", 0, 0, &
            &                       PETSC_NULL_INTEGER, boundary_SlipWall, PETSC_NULL_FUNCTION, &
            &                       nSlipwall, slipwall_ids, ctx, ierr)    ; CHKERRA(ierr)
        end if
        if (nOutflow > 0) then
            call PetscDSAddBoundary(prob, 10, "outflowsupersonic", "Face Sets", 0, 0, &
            &                       PETSC_NULL_INTEGER, boundary_Outflow, PETSC_NULL_FUNCTION, &
            &                       nOutflow, outflow_ids, ctx, ierr)      ; CHKERRA(ierr)
        end if
        if (nFreestream > 0) then
            call PetscDSAddBoundary(prob, 10, "freestream", "Face Sets", 0, 0, &
            &                       PETSC_NULL_INTEGER, boundary_Freestream, PETSC_NULL_FUNCTION, &
            &                       nFreestream, freestream_ids, ctx, ierr); CHKERRA(ierr)
        end if

    end subroutine setupBC

    subroutine boundary_SlipWall(time, c, n, int_cVar, ghost_cVar, ierr)

        PetscErrorCode, intent(out)                 :: ierr
        PetscReal, dimension(0:nvar-1), intent(out) :: ghost_cVar
        PetscInt, dimension(0:ndim-1)               :: c, n ! coordinate and normal
        PetscReal                                   :: time
        PetscScalar, dimension(0:nvar-1)            :: int_cVar

        PetscReal, dimension(0:ndim-1)  :: nn
        PetscReal                       :: norm, ghost_pressure, vn, vx_ghost, vy_ghost, vz_ghost
        PetscInt                        :: idim

        ! Normalize the normal n
        norm = 0.d0
        do idim = 0, ndim-1
            nn(idim) = n(idim)
            norm     = norm + nn(idim)*nn(idim)
        end do
        norm = sqrt(norm)
        nn   = nn/norm

        ! Copy the density
        ghost_cVar(0) = int_cVar(0)

        ! Normal velocity at interface
        vn = nn(0)*int_cVar(1)/int_cVar(0) + nn(1)*int_cVar(2)/int_cVar(0) + nn(2)*int_cVar(3)/int_cVar(0)

        vx_ghost = int_cVar(1)/int_cVar(0) - vn*nn(0)
        vy_ghost = int_cVar(2)/int_cVar(0) - vn*nn(1)
        vz_ghost = int_cVar(3)/int_cVar(0) - vn*nn(2)

        ! Cancel normal velocity
        ghost_cVar(1) =  vx_ghost * ghost_cVar(0)
        ghost_cVar(2) =  vy_ghost * ghost_cVar(0)
        ghost_cVar(3) =  vz_ghost * ghost_cVar(0)

        ! Copy pressure
        ghost_pressure = (gamma-1)*(int_cVar(4) - 0.5d0*(int_cVar(1)**2 + int_cVar(2)**2 + int_cVar(3)**2)/int_cVar(0))

        ! Reconstruct the energy
        ghost_cVar(4)  =  ghost_pressure/(gamma-1) + 0.5d0*(vx_ghost**2 + vy_ghost**2 + vz_ghost**2)*ghost_cvar(0)

    end subroutine boundary_SlipWall

    subroutine boundary_Outflow(time, c, n, int_cVar, ghost_cVar, ierr)

        PetscErrorCode, intent(out)                 :: ierr
        PetscReal, dimension(0:nvar-1), intent(out) :: ghost_cVar
        PetscInt, dimension(0:ndim-1)               :: c, n ! coordinate and normal
        PetscReal                                   :: time
        PetscScalar, dimension(0:nvar-1)            :: int_cVar

        ! Copy all variables
        ghost_cVar = int_cVar

    end subroutine boundary_Outflow

    subroutine boundary_Freestream(time, c, n, int_cVar, ghost_cVar, ierr)

        PetscErrorCode, intent(out)                 :: ierr
        PetscReal, dimension(0:nvar-1), intent(out) :: ghost_cVar
        PetscInt, dimension(0:ndim-1)               :: c, n ! coordinate and normal
        PetscReal                                   :: time
        PetscScalar, dimension(0:nvar-1)            :: int_cVar

        ghost_cVar(0) = freestream_params(0,0)
        ghost_cVar(1) = freestream_params(0,0) * freestream_params(0,1)
        ghost_cVar(2) = freestream_params(0,0) * freestream_params(0,2)
        ghost_cVar(3) = freestream_params(0,0) * freestream_params(0,3)
        ghost_cVar(4) = freestream_params(0,4)/(gamma - 1.0d0) &
        &               + 0.5d0 * (freestream_params(0,1)**2+freestream_params(0,2)**2+freestream_params(0,3)**2) * freestream_params(0,0)

    end subroutine boundary_Freestream

end module boundary