program eulerian3D

    use petscsys
#include <petsc/finclude/petscsys.h>

    use common
    use geometry
    use finitevolume
    use timestepper

    implicit none

    PetscErrorCode :: ierr

    call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

    ! Prepare mesh
    call initmesh

    ! Space discretizaion
    call initFV

    ! Initialize time discretization
    call initTS

    call PetscFinalize(ierr)

end program eulerian3D
