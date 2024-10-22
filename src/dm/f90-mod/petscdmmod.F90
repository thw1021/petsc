        module petscdmdef
        use petscvecdef
        use petscmatdef
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscall.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscspace.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdualspace.h>
        end module petscdmdef
!     ----------------------------------------------

        module petscdm
        use petscmat
        use petscdmdef
#include <../src/dm/f90-mod/petscdm.h90>
#include <../src/dm/f90-mod/petscdt.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscall.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscspace.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdualspace.h90>

        contains

#include <../src/dm/f90-mod/ftn-auto-interfaces/petscall.hf90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscspace.hf90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdualspace.hf90>
        end module petscdm

!     ----------------------------------------------

        module petscdmdadef
        use petscdmdef
        use petscaodef
        use petscpfdef
#include <petsc/finclude/petscao.h>
#include <petsc/finclude/petscdmda.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmda.h>

        end module petscdmdadef

        module petscdmda
        use petscdm
        use petscdmdadef

#include <../src/dm/f90-mod/petscdmda.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmda.h90>
        end module petscdmda

!     ----------------------------------------------

        module petscdmplex
        use petscdm
        use petscdmdef
#include <petsc/finclude/petscfv.h>
#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscdmplextransform.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscfv.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmplex.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmplextransform.h>

#include <../src/dm/f90-mod/petscdmplex.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscfv.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmplex.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmplextransform.h90>

        contains

#include <../src/dm/f90-mod/ftn-auto-interfaces/petscfv.hf90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmplex.hf90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmplextransform.hf90>

        end module petscdmplex

!     ----------------------------------------------

        module petscdmstag
        use petscdmdef
#include <petsc/finclude/petscdmstag.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmstag.h>

#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmstag.h90>
        end module petscdmstag

!     ----------------------------------------------

        module petscdmswarm
        use petscdm
        use petscdmdef
#include <petsc/finclude/petscdmswarm.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmswarm.h>

#include <../src/dm/f90-mod/petscdmswarm.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmswarm.h90>

        contains

#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmswarm.hf90>
        end module petscdmswarm

!     ----------------------------------------------

        module petscdmcomposite
        use petscdm
#include <petsc/finclude/petscdmcomposite.h>

#include <../src/dm/f90-mod/petscdmcomposite.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmcomposite.h90>
        end module petscdmcomposite

!     ----------------------------------------------

        module petscdmforest
        use petscdm
#include <petsc/finclude/petscdmforest.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmforest.h>

#include <../src/dm/f90-mod/petscdmforest.h90>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmforest.h90>
        end module petscdmforest

!     ----------------------------------------------

        module petscdmnetwork
        use petscdm
#include <petsc/finclude/petscdmnetwork.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmnetwork.h>

#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmnetwork.h90>

        contains

#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmnetwork.hf90>
        end module petscdmnetwork

!     ----------------------------------------------

        module petscdmadaptor
        use petscdm
        use petscdmdef
!        use petscsnes
#include <petsc/finclude/petscdmadaptor.h>
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmadaptor.h>

!#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmadaptor.h90>

        contains

!#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmadaptor.hf90>
        end module petscdmadaptor
