        module petsctaodef
        use petsckspdef

#include <../src/tao/f90-mod/ftn-auto-interfaces/petscall.h>
        end module petsctaodef

        module petsctao
        use petscts
        use petsctaodef

#include <../src/tao/f90-mod/ftn-auto-interfaces/petscall.h90>

        contains

#include <../src/tao/f90-mod/ftn-auto-interfaces/petscall.hf90>

        end module petsctao

! The all encompassing petsc module

        module petscdef
        use petsctaodef
        end module petscdef

        module petsc
        use petsctao
        use petscao
        use petscpf
        use petscdmplex
        use petscdmswarm
        use petscdmnetwork
        use petscdmda
        use petscdmcomposite
        use petscdmforest
        use petsccharacteristic
        end module petsc
