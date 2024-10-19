        module petscmatdef
        use petscvecdef
#include "petsc/finclude/petscmat.h"
#include "petsc/finclude/petscmatcoarsen.h"
#include "petsc/finclude/petscpartitioner.h"
#include "petsc/finclude/petscmathypre.h"
#include "petsc/finclude/petscmathtool.h"
#include "petsc/finclude/petscmatelemental.h"
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmat.h>
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmatcoarsen.h>
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscpartitioner.h>

        end module

!     ----------------------------------------------

        module petscmat
        use petscmatdef
        use petscvec

#include <../src/mat/f90-mod/petscmat.h90>
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmat.h90>
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmatcoarsen.h90>
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscpartitioner.h90>

!     deprecated functions

        interface MatDenseGetArrayF90
          module procedure MatDenseGetArrayF901d, MatDenseGetArrayF902d
        end interface

        interface  MatDenseRestoreArrayF90
          module procedure MatDenseRestoreArrayF901d, MatDenseRestoreArrayF902d
        end interface

        interface MatDenseGetArrayReadF90
          module procedure MatDenseGetArrayReadF901d, MatDenseGetArrayReadF902d
        end interface

        interface  MatDenseRestoreArrayReadF90
          module procedure MatDenseRestoreArrayWriteF901d, MatDenseRestoreArrayWriteF902d
        end interface

        interface MatDenseGetArrayWriteF90
          module procedure MatDenseGetArrayWriteF901d, MatDenseGetArrayWriteF902d
        end interface

        interface  MatDenseRestoreArrayWriteF90
          module procedure MatDenseRestoreArrayWriteF901d, MatDenseRestoreArrayWriteF902d
        end interface

        contains

#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmat.hf90>
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmatcoarsen.hf90>
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscpartitioner.hf90>

!     deprecated functions

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatFDColoringGetPerturbedColumnsF90
#endif
        Subroutine MatFDColoringGetPerturbedColumnsF90(i,array,ierr)
          PetscInt, pointer :: array(:)
          PetscErrorCode  ierr
          MatFDColoring       i
          call MatFDColoringGetPerturbedColumns(i,array,ierr)
         End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatFDColoringRestorePerturbedColumnsF90
#endif
        Subroutine MatFDColoringRestorePerturbedColumnsF90(i,array,ierr)
          PetscInt, pointer :: array(:)
           PetscErrorCode ierr
           MatFDColoring      i
           call MatFDColoringRestorePerturbedColumns(i,array,ierr)
         End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatGetRowIJF90
#endif
        Subroutine MatGetRowIJF90(v,sh,sym,bl,n,ia,ja,d,ierr)
          PetscInt, pointer :: ia(:), ja(:)
          PetscInt  n,sh
          PetscBool  sym,bl,d
          PetscErrorCode ierr
          Mat     v
          call MatGetRowIJ(v,sh,sym,bl,n,ia,ja,d,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatRestoreRowIJF90
#endif
        Subroutine MatRestoreRowIJF90(v,s,sy,b,n,ia,ja,d,ierr)
          PetscInt, pointer :: ia(:), ja(:)
          PetscInt  n,s
          PetscBool  sy,b,d
          PetscErrorCode ierr
          Mat     v
          call MatRestoreRowIJ(v,s,sy,b,n,ia,ja,d,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayF901d
#endif
        Subroutine MatDenseGetArrayF901d(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArray(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayF901d
#endif
        Subroutine MatDenseRestoreArrayF901d(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArray(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayReadF901d
#endif
        Subroutine MatDenseGetArrayReadF901d(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArrayRead(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayReadF901d
#endif
        Subroutine MatDenseRestoreArrayReadF901d(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArrayRead(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayWriteF901d
#endif
        Subroutine MatDenseGetArrayWriteF901d(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArrayWrite(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayWriteF901d
#endif
        Subroutine MatDenseRestoreArrayWriteF901d(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArrayWrite(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayF902d
#endif
        Subroutine MatDenseGetArrayF902d(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArray(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayF902d
#endif
        Subroutine MatDenseRestoreArrayF902d(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArray(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayReadF902d
#endif
        Subroutine MatDenseGetArrayReadF902d(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArrayRead(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayReadF90
#endif
        Subroutine MatDenseRestoreArrayReadF902d(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArrayRead(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetArrayWriteF90
#endif
        Subroutine MatDenseGetArrayWriteF902d(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseGetArrayWrite(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreArrayWriteF90
#endif
        Subroutine MatDenseRestoreArrayWriteF902d(v,array,ierr)
          PetscScalar, pointer :: array(:,:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreArrayWrite(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseGetColumnF90
#endif
        Subroutine MatDenseGetColumnF90(v,col,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          PetscInt col
          call MatDenseGetColumn(v,col,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatDenseRestoreColumnF90
#endif
        Subroutine MatDenseRestoreColumnF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatDenseRestoreColumn(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatSeqAIJGetArrayF90
#endif
        Subroutine MatSeqAIJGetArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatSeqAIJGetArray(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatMPIAIJGetSeqAIJF90
#endif
        Subroutine MatMPIAIJGetSeqAIJF90(a,b,c,d,ierr)
          PetscInt, pointer :: d(:)
          PetscErrorCode ierr
          Mat     a,b,c
          call MatMPIAIJGetSeqAIJ(a,b,c,d,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatMPIAIJRestoreSeqAIJF90
#endif
        Subroutine MatMPIAIJRestoreSeqAIJF90(a,b,c,d,ierr)
          PetscInt, pointer :: d(:)
          PetscErrorCode ierr
          Mat     a,b,c
          call MatMPIAIJRestoreSeqAIJ(a,b,c,d,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatSeqAIJRestoreArrayF90
#endif
     Subroutine MatSeqAIJRestoreArrayF90(v,array,ierr)
          PetscScalar, pointer :: array(:)
          PetscErrorCode ierr
          Mat     v
          call MatSeqAIJRestoreArray(v,array,ierr)
        End Subroutine

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: MatGetGhostsF90
#endif
        Subroutine MatGetGhostsF90(v,b,array,ierr)
        PetscInt, pointer :: array(:)
        PetscInt b
          PetscErrorCode ierr
          Mat     v
          call MatGetGhosts(v,b,array,ierr)
        End Subroutine

        end module
