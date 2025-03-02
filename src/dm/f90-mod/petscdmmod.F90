        module petscdmdef
        use petscvecdef
        use petscmatdef
#include <../ftn/dm/petscall.h>
#include <../ftn/dm/petscspace.h>
#include <../ftn/dm/petscdualspace.h>
        end module petscdmdef
!     ----------------------------------------------

        module petscdm
        use petscmat
        use petscdmdef
#include <../src/dm/f90-mod/petscdm.h90>
#include <../src/dm/f90-mod/petscdt.h90>
#include <../ftn/dm/petscall.h90>
#include <../ftn/dm/petscspace.h90>
#include <../ftn/dm/petscdualspace.h90>

        contains

#include <../ftn/dm/petscall.hf90>
#include <../ftn/dm/petscspace.hf90>
#include <../ftn/dm/petscdualspace.hf90>
        end module petscdm

!     ----------------------------------------------

        module petscdmdadef
        use petscdmdef
        use petscaodef
        use petscpfdef
#include <petsc/finclude/petscao.h>
#include <petsc/finclude/petscdmda.h>
#include <../ftn/dm/petscdmda.h>

        end module petscdmdadef

        module petscdmda
        use petscdm
        use petscdmdadef

#include <../src/dm/f90-mod/petscdmda.h90>
#include <../ftn/dm/petscdmda.h90>

!        contains
!
!#include <../ftn/dm/petscdmda.hf90>
        end module petscdmda

!     ----------------------------------------------

        module petscdmplex
        use petscdm
        use petscdmdef
#include <petsc/finclude/petscfv.h>
#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscdmplextransform.h>
#include <../src/dm/f90-mod/petscdmplex.h90>
#include <../ftn/dm/petscfv.h>
#include <../ftn/dm/petscdmplex.h>
#include <../ftn/dm/petscdmplextransform.h>

#include <../ftn/dm/petscfv.h90>
#include <../ftn/dm/petscdmplex.h90>
#include <../ftn/dm/petscdmplextransform.h90>

        contains

#include <../ftn/dm/petscfv.hf90>
#include <../ftn/dm/petscdmplex.hf90>
#include <../ftn/dm/petscdmplextransform.hf90>
        end module petscdmplex

!     ----------------------------------------------

        module petscdmstag
        use petscdmdef
#include <petsc/finclude/petscdmstag.h>
#include <../ftn/dm/petscdmstag.h>

#include <../ftn/dm/petscdmstag.h90>
        end module petscdmstag

!     ----------------------------------------------

        module petscdmswarm
        use petscdm
        use petscdmdef
#include <petsc/finclude/petscdmswarm.h>
#include <../ftn/dm/petscdmswarm.h>

#include <../src/dm/f90-mod/petscdmswarm.h90>
#include <../ftn/dm/petscdmswarm.h90>

        contains

#include <../ftn/dm/petscdmswarm.hf90>
        end module petscdmswarm

!     ----------------------------------------------

        module petscdmcomposite
        use petscdm
#include <petsc/finclude/petscdmcomposite.h>

#include <../src/dm/f90-mod/petscdmcomposite.h90>
#include <../ftn/dm/petscdmcomposite.h90>
        end module petscdmcomposite

!     ----------------------------------------------

        module petscdmforest
        use petscdm
#include <petsc/finclude/petscdmforest.h>
#include <../ftn/dm/petscdmforest.h>
#include <../ftn/dm/petscdmforest.h90>

!      contain
!
!#include <../ftn/dm/petscdmforest.hf90>
        end module petscdmforest

!     ----------------------------------------------

        module petscdmnetwork
        use petscdm
#include <petsc/finclude/petscdmnetwork.h>
#include <../ftn/dm/petscdmnetwork.h>

#include <../ftn/dm/petscdmnetwork.h90>

        contains

#include <../ftn/dm/petscdmnetwork.hf90>
        end module petscdmnetwork

!     ----------------------------------------------

        module petscdmadaptor
        use petscdm
        use petscdmdef
!        use petscsnes
#include <petsc/finclude/petscdmadaptor.h>
#include <../ftn/dm/petscdmadaptor.h>

!#include <../ftn/dm/petscdmadaptor.h90>

        contains

!#include <../ftn/dm/petscdmadaptor.hf90>
        end module petscdmadaptor
