program eulerian3D

    use petscsys
#include <petsc/finclude/petscsys.h>

    use common
    use geometry
    use finitevolume
    use timestepper
    use solution

    implicit none

    PetscErrorCode :: ierr

    call PetscInitialize(PETSC_NULL_CHARACTER,ierr)


    ! Prepare mesh
    call initmesh

    ! Space discretizaion
    call initFV

    ! Initialize time discretization
    call initTS

    ! Initialize solution
    call initCond

    ! Do the time loop
    call marchTime

    call PetscFinalize(ierr)

end program eulerian3D