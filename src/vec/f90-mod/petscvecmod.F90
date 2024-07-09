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

#include <../src/vec/f90-mod/petscis.h90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsf.h90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsection.h90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscis.h90>

        contains

#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsf.hf90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscsection.hf90>
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscis.hf90>

!     The following F90 interfaces are deprecated

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscLayoutGetRanges90
#endif
        Subroutine PetscLayoutGetRanges90(a,b,z)
          PetscLayout a
          PetscInt, pointer :: b(:)
          PetscErrorCode  z
          call  PetscLayoutGetRanges(a,b,z)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::ISGetIndicesF90
#endif
        Subroutine ISGetIndicesF90(i,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          IS       i
          call ISGetIndices(i,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::ISRestoreIndicesF90
#endif
        Subroutine ISRestoreIndicesF90(i,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode ierr
          IS      i
          call ISRestoreIndices(i,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: ISBlockGetIndicesF90
#endif
        Subroutine ISBlockGetIndicesF90(i,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          IS       i
          call ISBlockGetIndices(i,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: ISBlockRestoreIndicesF90
#endif
        Subroutine ISBlockRestoreIndicesF90(i,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode ierr
          IS      i
          call ISBlockRestoreIndices(i,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: ISColoringGetISF90
#endif
        Subroutine ISColoringGetISF90(ic,mode,n,isa,ierr)
          IS, pointer :: isa(:)
          PetscInt     n
          PetscCopyMode mode
          PetscErrorCode ierr
          ISColoring ic
          call ISColoringGetIS(ic,mode,n,isa,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: ISColoringRestoreISF90
#endif
        Subroutine ISColoringRestoreISF90(ic,mode,isa,ierr)
          IS, pointer :: isa(:)
          PetscErrorCode     ierr
          PetscCopyMode mode
          ISColoring ic
          call ISColoringRestoreIS(ic,mode,isa,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: PetscSectionGetConstraintIndicesF90
#endif
      Subroutine PetscSectionGetConstraintIndicesF90(s,p,a,ierr)
          PetscInt p
          PetscInt, pointer :: a(:)
          PetscErrorCode  ierr
          PetscSection       s
          call PetscSectionGetConstraintIndices(s,p,a,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: PetscSectionRestoreConstraintIndicesF90
#endif
        Subroutine PetscSectionRestoreConstraintIndicesF90(s,p,a,ierr)
          PetscInt p
          PetscInt, pointer :: a(:)
          PetscErrorCode  ierr
          PetscSection       s
          call PetscSectionRestoreConstraintIndices(s,p,a,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: PetscSectionSetConstraintIndicesF90
#endif
        Subroutine PetscSectionSetConstraintIndicesF90(s,p,a,ierr)
          PetscInt p
          PetscInt, pointer :: a(:)
          PetscErrorCode  ierr
          PetscSection       s
          call PetscSectionSetConstraintIndices(s,p,a,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscSectionGetFieldConstraintIndicesF90
#endif
        Subroutine PetscSectionGetFieldConstraintIndicesF90(s,p,f,a,ierr)
          PetscSection      :: s
          PetscInt          :: p
          PetscInt          :: f
          PetscInt, pointer :: a(:)
          PetscErrorCode    :: ierr
          call PetscSectionGetFieldConstraintIndices(s,p,f,a,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: PetscSectionRestoreFieldConstraintIndicesF90
#endif
        Subroutine PetscSectionRestoreFieldConstraintIndicesF90(s,p,f,a,ierr)
          PetscSection      :: s
          PetscInt          :: p
          PetscInt          :: f
          PetscInt, pointer :: a(:)
          PetscErrorCode    :: ierr
          call PetscSectionRestoreFieldConstraintIndices(s,p,f,a,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: PetscSectionSetFieldConstraintIndicesF90
#endif
        Subroutine PetscSectionSetFieldConstraintIndicesF90(s,p,f,a,ierr)
          PetscSection      :: s
          PetscInt          :: p
          PetscInt          :: f
          PetscInt          :: a(*)
          PetscErrorCode    :: ierr
          call PetscSectionSetFieldConstraintIndices(s,p,f,a,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: PetscSFDistributeSectionF90
#endif
        Subroutine PetscSFDistributeSectionF90(sf,rootsection,array,leafsection,ierr)
          PetscSF sf
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          PetscSection rootsection,leafsection
          call PetscSFDistributeSection(sf,rootsection,array,leafsection,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: PetscSFCreateSectionSFF90
#endif
        Subroutine PetscSFCreateSectionSFF90(pointsf,rootsection,array,leafsection,sf,ierr)
          PetscSF pointsf
          PetscSF sf
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          PetscSection rootsection,leafsection
          call PetscSFCreateSectionSF(pointsf,rootsection,array,leafsection,sf,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: PetscSFCreateRemoteOffsetsF90
#endif
        Subroutine PetscSFCreateRemoteOffsetsF90(sf,rootsection,leafsection,array,ierr)
          PetscSF sf
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          PetscSection rootsection,leafsection
          call PetscSFCreateRemoteOffsets(sf,rootsection,leafsection,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: ISLocalToGlobalMappingGetIndicesF90
#endif
        Subroutine ISLocalToGlobalMappingGetIndicesF90(i,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          ISLocalToGlobalMapping       i
          call ISLocalToGlobalMappingGetIndices(i,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: ISLocalToGlobalMappingRestoreIndicesF90
#endif
        Subroutine ISLocalToGlobalMappingRestoreIndicesF90(i,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          ISLocalToGlobalMapping       i
          call ISLocalToGlobalMappingRestoreIndices(i,array,ierr)
        End Subroutine

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

!     deprecated naming convention

        Subroutine VecDuplicateVecsF90(v,m,vs,ierr)
          Vec, pointer :: vs(:)
          PetscInt m
          PetscErrorCode ierr
          Vec     v
          call VecDuplicateVecs(v,m,vs,ierr)
        End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecDuplicateVecsF90
#endif

        Subroutine VecDestroyVecsF90(m,vs,ierr)
          Vec, pointer :: vs(:)
          PetscInt m
          PetscErrorCode ierr
          call VecDestroyVecs(m,vs,ierr)
       End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecDestroyVecsF90
#endif

      Subroutine VecGetArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
          call VecGetArray(v,array,ierr)
        End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecGetArrayF90
#endif

      Subroutine VecRestoreArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
          call VecRestoreArray(v,array,ierr)
        End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecRestoreArrayF90
#endif

        Subroutine VecGetArrayReadF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
         call VecGetArrayRead(v,array,ierr)
        End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecGetArrayReadF90
#endif

        Subroutine VecRestoreArrayReadF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
          call VecRestoreArrayRead(v,array,ierr)
        End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecRestoreArrayReadF90
#endif

        Subroutine VecGetValuesSectionF90(v,s,p,va,ierr)
          PetscScalar, pointer :: va(:)
          PetscErrorCode ierr
          Vec     v
          PetscSection s
          PetscInt p
          call VecGetValuesSection(v,s,p,va,ierr)
        End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecGetValuesSectionF90
#endif

        Subroutine VecRestoreValuesSectionF90(v,s,p,va,ierr)
          PetscScalar, pointer :: va(:)
          PetscErrorCode ierr
          Vec     v
          PetscSection s
          PetscInt p
          call VecRestoreValuesSection(v,s,p,va,ierr)
        End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecRestoreValuesSectionF90
#endif

        Subroutine VecSetValuesSectionF90(v,s,p,va,mode,ierr)
          InsertMode mode
          PetscScalar va(*)
          PetscErrorCode ierr
          Vec     v
          PetscSection s
          PetscInt p
          call VecSetValuesSection(v,s,p,va,mode,ierr)
       End Subroutine
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecSetValuesSectionF90
#endif

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
