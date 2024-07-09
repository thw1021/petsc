        module petscisdefdummy
        use petscsysdef
#include <../src/vec/f90-mod/petscis.h>
#include <../src/vec/f90-mod/petscislocaltoglobalmapping.h>
        end module petscisdefdummy

        module petscisdef
        use petscisdefdummy
        interface operator(.ne.)
          function isnotequal(A,B)
            import tIs
            logical isnotequal
            type(tIS), intent(in) :: A,B
          end function
          function petscsfnotequal(A,B)
            import tPetscSF
            logical petscsfnotequal
            type(tPetscSF), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function isequals(A,B)
            import tIs
            logical isequals
            type(tIS), intent(in) :: A,B
          end function
          function petscsfequals(A,B)
            import tPetscSF
            logical petscsfequals
            type(tPetscSF), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function isnotequal(A,B)
          use petscisdefdummy, only: tIS
          logical isnotequal
          type(tIS), intent(in) :: A,B
          isnotequal = (A%v .ne. B%v)
        end function

        function isequals(A,B)
          use petscisdefdummy, only: tIS
          logical isequals
          type(tIS), intent(in) :: A,B
          isequals = (A%v .eq. B%v)
        end function

        function petscsfnotequal(A,B)
          use petscisdefdummy, only: tPetscSF
          logical petscsfnotequal
          type(tPetscSF), intent(in) :: A,B
          petscsfnotequal = (A%v .ne. B%v)
        end function

        function petscsfequals(A,B)
          use petscisdefdummy, only: tPetscSF
          logical petscsfequals
          type(tPetscSF), intent(in) :: A,B
          petscsfequals = (A%v .eq. B%v)
        end function

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::isnotequal
!DEC$ ATTRIBUTES DLLEXPORT::petscsfnotequal
!DEC$ ATTRIBUTES DLLEXPORT::isequals
!DEC$ ATTRIBUTES DLLEXPORT::petscsfequals
#endif

        module  petscaodef
        use petscisdef
        use petscsysdef
#include <../src/vec/f90-mod/petscao.h>
        end module

        module petscvecdefdummy
        use petscisdef
        use petscaodef
#include <../src/vec/f90-mod/petscvec.h>
        end module

        module petscvecdef
        use petscvecdefdummy
        interface operator(.ne.)
          function vecnotequal(A,B)
            import tVec
            logical vecnotequal
            type(tVec), intent(in) :: A,B
          end function
          function vecscatternotequal(A,B)
            import tVecScatter
            logical vecscatternotequal
            type(tVecScatter), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function vecequals(A,B)
            import tVec
            logical vecequals
            type(tVec), intent(in) :: A,B
          end function
          function vecscatterequals(A,B)
            import tVecScatter
            logical vecscatterequals
            type(tVecScatter), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function vecnotequal(A,B)
          use petscvecdefdummy, only: tVec
          logical vecnotequal
          type(tVec), intent(in) :: A,B
          vecnotequal = (A%v .ne. B%v)
        end function

        function vecequals(A,B)
          use petscvecdefdummy, only: tVec
          logical vecequals
          type(tVec), intent(in) :: A,B
          vecequals = (A%v .eq. B%v)
        end function

        function vecscatternotequal(A,B)
          use petscvecdefdummy, only: tVecScatter
          logical vecscatternotequal
          type(tVecScatter), intent(in) :: A,B
          vecscatternotequal = (A%v .ne. B%v)
        end function

        function vecscatterequals(A,B)
          use petscvecdefdummy, only: tVecScatter
          logical vecscatterequals
          type(tVecScatter), intent(in) :: A,B
          vecscatterequals = (A%v .eq. B%v)
        end function

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::vecnotequal
!DEC$ ATTRIBUTES DLLEXPORT::vecscatternotequal
!DEC$ ATTRIBUTES DLLEXPORT::vecequals
!DEC$ ATTRIBUTES DLLEXPORT::vecscatterequals
#endif

        module petscis
        use petscisdef
        use petscsys
#include <../src/vec/f90-mod/petscis.h90>
        interface
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscis.h90>
        end interface

        contains

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
          PetscInt, pointer :: a(:)
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

        module petscao
        use petscis
        use petscaodef
        end module

        module petscvec
        use petscvecdef
        use petscis
        use petscao
#include <../src/vec/f90-mod/petscvec.h90>
        interface
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscvec.h90>
        end interface

        contains

!     deprecated naming convention

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecDuplicateVecsF90
#endif
        Subroutine VecDuplicateVecsF90(v,m,vs,ierr)
          Vec, pointer :: vs(:)
          PetscInt m
          PetscErrorCode ierr
          Vec     v
          call VecDuplicateVecs(v,m,vs,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecDestroyVecsF90
#endif
        Subroutine VecDestroyVecsF90(m,vs,ierr)
          Vec, pointer :: vs(:)
          PetscInt m
          PetscErrorCode ierr
          call VecDestroyVecs(m,vs,ierr)
       End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecGetArrayF90
#endif
      Subroutine VecGetArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
          call VecGetarray(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecRestoreArrayF90
#endif
        Subroutine VecRestoreArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
          call VecRestoreArray(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecGetArrayReadF90
#endif
        Subroutine VecGetArrayReadF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
         call VecGetArrayRead(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecRestoreArrayReadF90
#endif
        Subroutine VecRestoreArrayReadF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
          call VecRestoreArrayRead(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecGetValuesSectionF90
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
!DEC$ ATTRIBUTES DLLEXPORT:: VecRestoreValuesSectionF90
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
!DEC$ ATTRIBUTES DLLEXPORT:: VecSetValuesSectionF90
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
!DEC$ ATTRIBUTES DLLEXPORT:: VecGetOwnershipRanges
#endif
       Subroutine VecGetOwnershipRanges(v,b,z)
          Vec v
          PetscInt, pointer :: b(:)
          PetscErrorCode  z
          PetscLayout a
          call VecGetLayout(v,a,z)
          call PetscLayoutGetRanges(a,b,z)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: VecRestoreOwnershipRanges
#endif
       Subroutine VecRestoreOwnershipRanges(v,b,z)
          Vec v
          PetscInt, pointer :: b(:)
          PetscErrorCode  z
          PetscLayout a
          call VecGetLayout(v,a,z)
          call PetscLayoutRestoreRanges(a,b,z)
        End Subroutine
      end module

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: F90ArraySFNodeCreate
#endif
      subroutine F90ArraySFNodeCreate(array,n,ptr)
      use petscis, only: PetscSFNode
      implicit none
      PetscInt n,array(2*n)
      type(PetscSFNode), pointer :: ptr(:)
      PetscInt i
      allocate(ptr(n))
      do i=1,n
        ptr(i)%rank  = array(2*i-1)
        ptr(i)%index = array(2*i)
      enddo

      end subroutine
