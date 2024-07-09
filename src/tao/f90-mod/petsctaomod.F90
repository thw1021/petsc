        module petsctaodef
        use petsckspdef
#include <petsc/finclude/petsctao.h>
#include <petsc/finclude/petsctaolinesearch.h>
#include <../src/tao/f90-mod/ftn-auto-interfaces/petsctao.h>
#include <../src/tao/f90-mod/ftn-auto-interfaces/petsctaolinesearch.h>
        end module petsctaodef

        module petsctao
        use petsctaodef
        use petscksp

#include <../src/tao/f90-mod/petsctao.h90>
#include <../src/tao/f90-mod/ftn-auto-interfaces/petsctao.h90>
#include <../src/tao/f90-mod/ftn-auto-interfaces/petsctaolinesearch.h90>

        contains

#include <../src/tao/f90-mod/ftn-auto-interfaces/petsctao.hf90>
#include <../src/tao/f90-mod/ftn-auto-interfaces/petsctaolinesearch.hf90>

        end module petsctao

! The all encompassing petsc module

        module petscdef
        use petscdmdef
        use petsctsdef
        use petsctaodef
        end module petscdef

        module petsc
        use petscdm
        use petscdmswarm
        use petscdmplex
        use petscdmnetwork
        use petscdmda
        use petscdmcomposite
        use petscdmforest
        use petscts
        use petsctao
        end module petsc
