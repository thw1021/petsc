        module petscmatdefdummy
        use petscvecdef
#include <../src/mat/f90-mod/petscmat.h>
        end module petscmatdefdummy

        module petscmatdef
        use petscmatdefdummy
        interface operator(.ne.)
          function matnotequal(A,B)
            import tMat
            logical matnotequal
            type(tMat), intent(in) :: A,B
          end function
          function matfdcoloringnotequal(A,B)
            import tMatFDColoring
            logical matfdcoloringnotequal
            type(tMatFDColoring), intent(in) :: A,B
          end function
          function matnullspacenotequal(A,B)
            import tMatNullSpace
            logical matnullspacenotequal
            type(tMatNullSpace), intent(in) :: A,B
            end function
      end interface operator (.ne.)
        interface operator(.eq.)
          function matequals(A,B)
            import tMat
            logical matequals
            type(tMat), intent(in) :: A,B
          end function
          function matfdcoloringequals(A,B)
            import tMatFDColoring
            logical matfdcoloringequals
            type(tMatFDColoring), intent(in) :: A,B
          end function
           function matnullspaceequals(A,B)
            import tMatNullSpace
            logical matnullspaceequals
            type(tMatNullSpace), intent(in) :: A,B
            end function
          end interface operator (.eq.)
        end module

        function matnotequal(A,B)
          use petscmatdefdummy, only: tMat
          implicit none
          logical matnotequal
          type(tMat), intent(in) :: A,B
          matnotequal = (A%v .ne. B%v)
        end function

       function matequals(A,B)
          use petscmatdefdummy, only: tMat
          implicit none
          logical matequals
          type(tMat), intent(in) :: A,B
          matequals = (A%v .eq. B%v)
        end function

        function matfdcoloringnotequal(A,B)
          use petscmatdefdummy, only: tMatFDColoring
          implicit none
          logical matfdcoloringnotequal
          type(tMatFDColoring), intent(in) :: A,B
          matfdcoloringnotequal = (A%v .ne. B%v)
        end function

        function matfdcoloringequals(A,B)
          use petscmatdefdummy, only: tMatFDColoring
          implicit none
          logical matfdcoloringequals
          type(tMatFDColoring), intent(in) :: A,B
          matfdcoloringequals = (A%v .eq. B%v)
        end function

        function matnullspacenotequal(A,B)
          use petscmatdefdummy, only: tMatNullSpace
          implicit none
          logical matnullspacenotequal
          type(tMatNullSpace), intent(in) :: A,B
          matnullspacenotequal = (A%v .ne. B%v)
        end function

        function matnullspaceequals(A,B)
          use petscmatdefdummy, only: tMatNullSpace
          implicit none
          logical matnullspaceequals
          type(tMatNullSpace), intent(in) :: A,B
          matnullspaceequals = (A%v .eq. B%v)
        end function

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::matnotequal
!DEC$ ATTRIBUTES DLLEXPORT::matequals
!DEC$ ATTRIBUTES DLLEXPORT::matfdcoloringnotequal
!DEC$ ATTRIBUTES DLLEXPORT::matfdcoloringequals
!DEC$ ATTRIBUTES DLLEXPORT::matnullspacenotequal
!DEC$ ATTRIBUTES DLLEXPORT::matnullspaceequals
#endif
        module petscmat
        use petscmatdef
        use petscvec
#include <../src/mat/f90-mod/petscmat.h90>
        interface
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmat.h90>
        end interface

        contains

!       deprecated API

        Subroutine MatFDColoringGetPerturbedColumnsF90(i,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          MatFDColoring       i
          call MatFDColoringGetPerturbedColumns(i,array,ierr)
         End Subroutine

        Subroutine MatFDColoringRestorePerturbedColumnsF90(i,array,ierr)
           PetscInt, pointer :: array(:)
           PetscErrorCode ierr
           MatFDColoring      i
           call MatFDColoringRestorePerturbedColumns(i,array,ierr)
         End Subroutine

        Subroutine MatGetRowIJF90(v,sh,sym,bl,n,ia,ja,d,ierr)
          PetscInt, pointer :: ia(:), ja(:)
          PetscInt  n,sh
          PetscBool  sym,bl,d
          PetscErrorCode ierr
          Mat     v
          call MatGetRowIJ(v,sh,sym,bl,n,ia,ja,d,ierr)
        End Subroutine

        Subroutine MatRestoreRowIJF90(v,s,sy,b,n,ia,ja,d,ierr)
          PetscInt, pointer :: ia(:), ja(:)
          PetscInt  n,s
          PetscBool  sy,b,d
          PetscErrorCode ierr
          Mat     v
          call MatRestoreRowIJ(v,s,sy,b,n,ia,ja,d,ierr)
        End Subroutine

        Subroutine MatDenseGetArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArray(v,array,ierr)
        End Subroutine

        Subroutine MatDenseRestoreArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArray(v,array,ierr)
        End Subroutine

        Subroutine MatDenseGetArrayReadF90(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArrayRead(v,array,ierr)
        End Subroutine

        Subroutine MatDenseRestoreArrayReadF90(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArrayRead(v,array,ierr)
        End Subroutine

        Subroutine MatDenseGetArrayWriteF90(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArrayWrite(v,array,ierr)
        End Subroutine

        Subroutine MatDenseRestoreArrayWriteF90(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArrayWrite(v,array,ierr)
        End Subroutine

        Subroutine MatDenseGetColumnF90(v,col,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          PetscInt col
          call MatDenseGetColumn(v,col,array,ierr)
        End Subroutine

        Subroutine MatDenseRestoreColumnF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreColumn(v,array,ierr)
        End Subroutine

        Subroutine MatSeqAIJGetArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatSeqAIJGetArray(v,array,ierr)
        End Subroutine

        Subroutine MatMPIAIJGetSeqAIJF90(a,b,c,d,ierr)
          PetscInt, pointer :: d(:)
          PetscErrorCode ierr
          Mat     a,b,c
          call MatMPIAIJGetSeqAIJ(a,b,c,d,ierr)
        End Subroutine

        Subroutine MatMPIAIJRestoreSeqAIJF90(a,b,c,d,ierr)
          PetscInt, pointer :: d(:)
          PetscErrorCode ierr
          Mat     a,b,c
          call MatMPIAIJRestoreSeqAIJ(a,b,c,d,ierr)
        End Subroutine

        Subroutine MatSeqAIJRestoreArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatSeqAIJRestoreArray(v,array,ierr)
        End Subroutine

        Subroutine MatGetGhostsF90(v,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatGetGhosts(v,array,ierr)
        End Subroutine

      end module
