        module petscisdef
        use petscsysdef
#include <petsc/finclude/petscis.h>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscis.h>
#include <petsc/finclude/petscsf.h>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsf.h>
#include <petsc/finclude/petscsection.h>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsection.h>

        end module

!     Needed by Fortran stub petscsfgetgraph_()
      subroutine F90Array1dCreateSFNode(array,start,len,ptr)
      use petscisdef
      implicit none
      PetscInt start,len
      PetscSFNode, target :: array(start:start+len-1)
      PetscSFNode, pointer :: ptr(:)
      ptr => array
      end subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: F90Array1dCreateSFNode
#endif

      subroutine F90Array1dDestroySFNode(ptr)
      use petscisdef
      implicit none
      PetscSFNode, pointer :: ptr(:)
      nullify(ptr)
      end subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: F90Array1dDestroySFNode
#endif

!     ----------------------------------------------

        module petscis
        use petscisdef
        use petscsys

      interface PetscSFRestoreRemoteOffsets
      subroutine PetscSFRestoreRemoteOffsets(ptr)
      use petscisdef
      implicit none
      PetscInt, pointer :: ptr(:)
      end subroutine PetscSFRestoreRemoteOffsets
      end interface

#include <../src/vec/f90-mod/petscis.h90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsf.h90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsection.h90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscis.h90>

        contains

#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsf.hf90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsection.hf90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscis.hf90>

      end module

!     ----------------------------------------------

        module petscvecdef
        use petscisdef
#include <petsc/finclude/petscvec.h>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscvec.h>
        end module

!     ----------------------------------------------

        module petscvec
        use petscis
        use petscvecdef

#include <../src/vec/f90-mod/petscvec.h90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscvec.h90>

        contains

#include <../src/vec/f90-mod/ftn-auto-interfaces/petscvec.hf90>

      end module

!     ----------------------------------------------

        module  petscaodef
        use petscsys
        use petscvecdef
#include <petsc/finclude/petscao.h>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscao.h>
        end module

!     ----------------------------------------------

        module petscao
        use petscsys
        use petscaodef
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscao.h90>
        contains
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscao.hf90>
      end module

!     ----------------------------------------------

        module  petscpfdef
        use petscsys
        use petscvecdef
#include <petsc/finclude/petscpf.h>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscpf.h>
        end module

!     ----------------------------------------------

        module petscpf
        use petscsys
        use petscpfdef
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscpf.h90>
        contains
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscpf.hf90>
      end module
