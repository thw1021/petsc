
        module petscmpi
#include <petscconf.h>
#include "petsc/finclude/petscsys.h"
#if defined(PETSC_HAVE_MPIUNI)
        use mpiuni
#define PETSC_AVOID_MPIF_H
#elif defined(PETSC_HAVE_MPI_F90MODULE)
        use mpi
#define PETSC_AVOID_MPIF_H
#endif

#if !defined(PETSC_AVOID_MPIF_H)
#if defined(PETSC_HAVE_MPIUNI)
#include "petsc/mpiuni/mpif.h"
#else
!
!  This code is extremely fragile; it assumes the format of the mpif.h file has
!  a particular structure that does not change with MPI implementation versions. But since
!  mpif.h is a bit of a deadwater and PETSC_PROMOTE_FORTRAN_INTEGER is
!  rarely used it is maybe ok to include fragile code
!
#if defined(PETSC_HAVE_MPICH_NUMVERSION) && defined(PETSC_PROMOTE_FORTRAN_INTEGER)
#define INTEGER integer4
#define MPI_STATUS_IGNORE(A) mpi_status_ignore(5)
#define MPI_STATUSES_IGNORE(B,C) mpi_statuses_ignore(5,1)
#elif defined(PETSC_HAVE_OMPI_MAJOR_VERSION)  && defined(PETSC_PROMOTE_FORTRAN_INTEGER)
#define integer integer4
#define INTEGER integer4
#endif
#include "mpif.h"
#if defined(PETSC_HAVE_MPICH_NUMVERSION) && defined(PETSC_PROMOTE_FORTRAN_INTEGER)
#undef INTEGER
#undef MPI_STATUS_IGNORE
#undef MPI_STATUSES_IGNORE
#elif defined(PETSC_HAVE_OMPI_MAJOR_VERSION) && defined(PETSC_PROMOTE_FORTRAN_INTEGER)
#undef integer
#undef INTEGER
#endif
#endif
#endif

!    This module uses mpi, but only let out the stuff that you need from it
        private
        public:: MPIU_REAL, MPIU_SUM, MPIU_SCALAR, MPIU_INTEGER
        public:: PETSC_COMM_WORLD, PETSC_COMM_SELF
!
! ----------------------------------------------------------------------------
!    BEGIN PETSc aliases for MPI_ constants
!
!   These values for __float128 are handled in the common block (below)
!     and transmitted from the C code
!
#if !defined(PETSC_USE_REAL___FLOAT128)
#if defined (PETSC_USE_REAL_SINGLE)
      integer4, parameter :: MPIU_REAL = MPI_REAL
#else
      integer4, parameter :: MPIU_REAL = MPI_DOUBLE_PRECISION
#endif

      integer4, parameter :: MPIU_SUM = MPI_SUM

#if defined(PETSC_USE_COMPLEX)
#if defined (PETSC_USE_REAL_SINGLE)
      integer4, parameter :: MPIU_SCALAR = MPI_COMPLEX
#else
      integer4, parameter :: MPIU_SCALAR = MPI_DOUBLE_COMPLEX
#endif
#else
#if defined (PETSC_USE_REAL_SINGLE)
      integer4, parameter :: MPIU_SCALAR = MPI_REAL
#else
      integer4, parameter :: MPIU_SCALAR = MPI_DOUBLE_PRECISION
#endif
#endif
#endif

#if defined(PETSC_USE_64BIT_INDICES)
      integer4, parameter :: MPIU_INTEGER = MPI_INTEGER8
#else
      integer4, parameter :: MPIU_INTEGER = MPI_INTEGER
#endif

      MPI_Comm PETSC_COMM_WORLD
      MPI_Comm PETSC_COMM_SELF
      common /petscfortran9/ PETSC_COMM_WORLD
      common /petscfortran10/ PETSC_COMM_SELF
      data   PETSC_COMM_WORLD /0/
      data   PETSC_COMM_SELF /0/
        end module


        module petscsysdefdummy
        use petscmpi
#include <../src/sys/f90-mod/petscsys.h>
#include <../src/sys/f90-mod/petscdraw.h>
#include <../src/sys/f90-mod/petscviewer.h>
#include <../src/sys/f90-mod/petscbag.h>
#include <../src/sys/f90-mod/petscerror.h>
#include <../src/sys/f90-mod/petsclog.h>
        end module petscsysdefdummy

        module petscsysdef
        use petscsysdefdummy
        interface operator(.ne.)
          function petscviewernotequal(A,B)
            import tPetscViewer
            logical petscviewernotequal
            type(tPetscViewer), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function petscviewerequals(A,B)
            import tPetscViewer
            logical petscviewerequals
            type(tPetscViewer), intent(in) :: A,B
          end function
        end interface operator (.eq.)

        interface operator(.ne.)
        function petscrandomnotequal(A,B)
          import tPetscRandom
          logical petscrandomnotequal
          type(tPetscRandom), intent(in) :: A,B
        end function
        end interface operator (.ne.)
        interface operator(.eq.)
        function petscrandomequals(A,B)
          import tPetscRandom
          logical petscrandomequals
          type(tPetscRandom), intent(in) :: A,B
        end function
        end interface operator (.eq.)

        Interface petscbinaryread
        subroutine petscbinaryreadcomplex(fd,data,num,count,type,z)
          integer fd
          PetscComplex data(*)
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadreal(fd,data,num,count,type,z)
          integer fd
          PetscReal data(*)
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadint(fd,data,num,count,type,z)
          integer fd
          PetscInt data(*)
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadcomplex1(fd,data,num,count,type,z)
          integer fd
          PetscComplex data
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadreal1(fd,data,num,count,type,z)
          integer fd
          PetscReal data
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadint1(fd,data,num,count,type,z)
          integer fd
          PetscInt data
          PetscInt num
          PetscInt count
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadcomplexcnt(fd,data,num,count,type,z)
          integer fd
          PetscComplex data(*)
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadrealcnt(fd,data,num,count,type,z)
          integer fd
          PetscReal data(*)
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadintcnt(fd,data,num,count,type,z)
          integer fd
          PetscInt data(*)
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadcomplex1cnt(fd,data,num,count,type,z)
          integer fd
          PetscComplex data
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadreal1cnt(fd,data,num,count,type,z)
          integer fd
          PetscReal data
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinaryreadint1cnt(fd,data,num,count,type,z)
          integer fd
          PetscInt data
          PetscInt num
          PetscInt count(1)
          PetscDataType type
          PetscErrorCode z
        end subroutine
        end Interface

        Interface petscbinarywrite
        subroutine petscbinarywritecomplex(fd,data,num,type,z)
          integer fd
          PetscComplex data(*)
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywritereal(fd,data,num,type,z)
          integer fd
          PetscReal data(*)
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywriteint(fd,data,num,type,z)
          integer fd
          PetscInt data(*)
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywritecomplex1(fd,data,num,type,z)
          integer fd
          PetscComplex data
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywritereal1(fd,data,num,type,z)
          integer fd
          PetscReal data
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        subroutine petscbinarywriteint1(fd,data,num,type,z)
          integer fd
          PetscInt data
          PetscInt num
          PetscDataType type
          PetscErrorCode z
        end subroutine
        end Interface

        end module

        function petscviewernotequal(A,B)
          use petscsysdefdummy, only: tPetscViewer
          logical petscviewernotequal
          type(tPetscViewer), intent(in) :: A,B
          petscviewernotequal = (A%v .ne. B%v)
        end function
        function petscviewerequals(A,B)
          use petscsysdefdummy, only: tPetscViewer
          logical petscviewerequals
          type(tPetscViewer), intent(in) :: A,B
          petscviewerequals = (A%v .eq. B%v)
        end function

        function petscrandomnotequal(A,B)
          use petscsysdefdummy, only: tPetscRandom
          logical petscrandomnotequal
          type(tPetscRandom), intent(in) :: A,B
          petscrandomnotequal = (A%v .ne. B%v)
        end function
        function petscrandomequals(A,B)
          use petscsysdefdummy, only: tPetscRandom
          logical petscrandomequals
          type(tPetscRandom), intent(in) :: A,B
          petscrandomequals = (A%v .eq. B%v)
        end function
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::petscviewernotequal
!DEC$ ATTRIBUTES DLLEXPORT::petscviewerequals
!DEC$ ATTRIBUTES DLLEXPORT::petscrandomnotequal
!DEC$ ATTRIBUTES DLLEXPORT::petscrandomequals
#endif
        module petscsys
        use iso_c_binding
        use petscsysdef
        PetscChar(80) PETSC_NULL_CHARACTER = ''
        PetscInt PETSC_NULL_INTEGER(1)
        PetscFortranDouble PETSC_NULL_DOUBLE(1)
        PetscScalar PETSC_NULL_SCALAR(1)
        PetscReal PETSC_NULL_REAL(1)
        PetscBool PETSC_NULL_BOOL
!
!
!
!
!     Basic math constants
!
        PetscReal PETSC_PI
        PetscReal PETSC_MAX_REAL
        PetscReal PETSC_MIN_REAL
        PetscReal PETSC_MACHINE_EPSILON
        PetscReal PETSC_SQRT_MACHINE_EPSILON
        PetscReal PETSC_SMALL
        PetscReal PETSC_INFINITY
        PetscReal PETSC_NINFINITY

!
#include <../src/sys/f90-mod/petscsys.h90>
        interface
#include <../src/sys/f90-mod/ftn-auto-interfaces/petscsys.h90>
        end interface

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_COMM_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_COMM_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_CHARACTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DOUBLE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SCALAR
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_BOOL
#if defined(PETSC_USE_REAL___FLOAT128)
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_REAL
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_SCALAR
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_SUM
#endif
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_PI
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MAX_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MIN_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MACHINE_EPSILON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SQRT_MACHINE_EPSILON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SMALL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_INFINITY
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NINFINITY
#endif
        end module

        subroutine PetscSetCOMM(c1,c2)
        use petscsys, only: PETSC_COMM_WORLD,PETSC_COMM_SELF
        implicit none
        MPI_Comm c1,c2

        PETSC_COMM_WORLD    = c1
        PETSC_COMM_SELF     = c2
        return
        end

        subroutine PetscGetCOMM(c1)
        use petscsys, only: PETSC_COMM_WORLD
        implicit none
        MPI_Comm c1

        c1 = PETSC_COMM_WORLD
        return
        end

        subroutine PetscSetModuleBlock()
        use petscsys, only: PETSC_NULL_CHARACTER,PETSC_NULL_INTEGER,&
             PETSC_NULL_SCALAR,PETSC_NULL_DOUBLE,PETSC_NULL_REAL,&
             PETSC_NULL_BOOL,PETSC_NULL_FUNCTION
        implicit none

        call PetscSetFortranBasePointers(PETSC_NULL_CHARACTER,            &
     &     PETSC_NULL_INTEGER,PETSC_NULL_SCALAR,                        &
     &     PETSC_NULL_DOUBLE,PETSC_NULL_REAL,                           &
     &     PETSC_NULL_BOOL,PETSC_NULL_FUNCTION)

        return
        end

#if defined(PETSC_USE_REAL___FLOAT128)
        subroutine PetscSetModuleBlockMPI(freal,fscalar,fsum)
        use petscsys
        implicit none

        integer freal,fscalar,fsum

        MPIU_REAL   = freal
        MPIU_SCALAR = fscalar
        MPIU_SUM    = fsum
        return
        end
#endif

        subroutine PetscSetModuleBlockNumeric(pi,maxreal,minreal,eps,       &
     &     seps,small,pinf,pninf)
        use petscsys, only: PETSC_PI,PETSC_MAX_REAL,PETSC_MIN_REAL,&
             PETSC_MACHINE_EPSILON,PETSC_SQRT_MACHINE_EPSILON,&
             PETSC_SMALL,PETSC_INFINITY,PETSC_NINFINITY
        implicit none

        PetscReal pi,maxreal,minreal,eps,seps
        PetscReal small,pinf,pninf

        PETSC_PI = pi
        PETSC_MAX_REAL = maxreal
        PETSC_MIN_REAL = minreal
        PETSC_MACHINE_EPSILON = eps
        PETSC_SQRT_MACHINE_EPSILON = seps
        PETSC_SMALL = small
        PETSC_INFINITY = pinf
        PETSC_NINFINITY = pninf

        return
        end


