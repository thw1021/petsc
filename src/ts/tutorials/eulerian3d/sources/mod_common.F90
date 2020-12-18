module common

    use petscsys
#include <petsc/finclude/petscsys.h>

    implicit none

    contains

        subroutine hello

            PetscErrorCode ierr;

            call PetscPrintf(PETSC_COMM_WORLD, "███████ ██    ██ ██      ███████ ██████  ██  █████  ███    ██ ██████  ██████ \n", ierr); CHKERRA(ierr)
            call PetscPrintf(PETSC_COMM_WORLD, "██      ██    ██ ██      ██      ██   ██ ██ ██   ██ ████   ██      ██ ██   ██\n", ierr); CHKERRA(ierr)
            call PetscPrintf(PETSC_COMM_WORLD, "█████   ██    ██ ██      █████   ██████  ██ ███████ ██ ██  ██  █████  ██   ██\n", ierr); CHKERRA(ierr)
            call PetscPrintf(PETSC_COMM_WORLD, "██      ██    ██ ██      ██      ██   ██ ██ ██   ██ ██  ██ ██      ██ ██   ██\n", ierr); CHKERRA(ierr)
            call PetscPrintf(PETSC_COMM_WORLD, "███████  ██████  ███████ ███████ ██   ██ ██ ██   ██ ██   ████ ██████  ██████ \n", ierr); CHKERRA(ierr)

        end subroutine hello

end module common