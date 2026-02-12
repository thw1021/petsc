!
!  Manually maintained part of the base include file for Fortran use of PETSc.
!  Note: This file should contain only define statements
!
#if !defined (PETSCSYSBASEDEF_H)
#define PETSCSYSBASEDEF_H
#include "petscconf.h"
#if defined (PETSC_HAVE_MPIUNI)
#include "petsc/mpiuni/mpiunifdef.h"
#endif
#include "petscversion.h"

!
#define integer8 integer(kind=C_INT64_T)
#define integer4 integer(kind=C_INT32_T)
#define integer2 integer(kind=C_INT16_T)
#define integer1 integer(kind=C_INT8_T)
#define PetscBool logical(kind=C_BOOL)

#if (PETSC_SIZEOF_VOID_P == 8)
#define PetscOffset integer8
#define PetscFortranAddr integer8
#else
#define PetscOffset integer4
#define PetscFortranAddr integer4
#endif

#if defined(PETSC_USE_64BIT_INDICES)
#define PetscInt integer8
#else
#define PetscInt integer4
#endif
#define PetscInt64 integer8

#if defined(PETSC_USE_64BIT_BLAS_INDICES)
#define PetscBLASInt integer8
#else
#define PetscBLASInt integer4
#endif
#define PetscCuBLASInt integer4
#define PetscHipBLASInt integer4

!
#define PetscSizeT integer(kind=C_SIZE_T)
!
#if defined(PETSC_USE_MPI_F08)
#define MPIU_Comm type(MPI_Comm)
#define MPIU_Group type(MPI_Group)
#define MPIU_Datatype type(MPI_Datatype)
#define MPIU_Op type(MPI_Op)
#define MPIU_Request type(MPI_Request)
#define MPIU_Status type(MPI_Status)
#else
#define MPIU_Comm integer4
#define MPIU_Group integer4
#define MPIU_Datatype integer4
#define MPIU_Op integer4
#define MPIU_Status integer4
#define MPIU_Request integer4
#endif
!
#define PetscEnum integer4
#define PetscVoid PetscFortranAddr
!
#define PetscFortranFloat real(kind=C_FLOAT)
#define PetscFortranDouble real(kind=C_DOUBLE)
#define PetscFortranLongDouble real(kind=C_FLOAT128)

#if defined(PETSC_USE_REAL_SINGLE)
#define FORTRAN_REAL_KIND C_FLOAT
#define FORTRAN_CMPLX_KIND C_FLOAT_COMPLEX
#elif defined(PETSC_USE_REAL_DOUBLE)
#define FORTRAN_REAL_KIND C_DOUBLE
#define FORTRAN_CMPLX_KIND C_DOUBLE_COMPLEX
#elif defined(PETSC_USE_REAL___FLOAT128)
#define FORTRAN_REAL_KIND C_FLOAT128
#define FORTRAN_CMPLX_KIND C_FLOAT128_COMPLEX
#endif

#define PetscReal real(kind=FORTRAN_REAL_KIND)
#define PetscComplex complex(kind=FORTRAN_CMPLX_KIND)
#define PetscIntToReal(a) real(a,kind=FORTRAN_REAL_KIND)

!
! Macros for templating between real and complex
!
#define PetscRealPart(a) real(a,kind=FORTRAN_REAL_KIND)
#if defined(PETSC_USE_COMPLEX)
#define PetscScalar PetscComplex
#define PetscConj(a) conjg(cmplx(a,kind=FORTRAN_CMPLX_KIND))
#define PetscImaginaryPart(a) aimag(cmplx(a,kind=FORTRAN_CMPLX_KIND))
#define PetscFloatToScalar cmplx(a,kind=FORTRAN_CMPLX_KIND)
#else
#define PetscScalar PetscReal
#define PetscConj(a) real(a,kind=FORTRAN_REAL_KIND)
#define PetscImaginaryPart(a) real(0.0,kind=FORTRAN_REAL_KIND)
#define PetscFloatToScalar real(a,kind=FORTRAN_REAL_KIND)
#endif

#undef FORTRAN_REAL_KIND
#undef FORTRAN_CMPLX_KIND

#define PetscReal2d type(tPetscReal2d)

#define PETSC_FORTRAN_TYPE_INITIALIZE -2
#define PetscObjectIsNull(obj) (obj%v == 0 .or. obj%v ==  PETSC_FORTRAN_TYPE_INITIALIZE .or. obj%v == -3)
#define PetscObjectNullify(obj) obj%v = PETSC_FORTRAN_TYPE_INITIALIZE
!
!     Macros for error checking
!
#define SETERRQ(c, ierr, s)  call PetscError(c, ierr, PETSC_ERROR_INITIAL, s); return
#define SETERRA(c, ierr, s)  call PetscError(c, ierr, PETSC_ERROR_INITIAL, s); call MPIU_Abort(c, ierr)
#if defined(PETSC_HAVE_FORTRAN_FREE_LINE_LENGTH_NONE)
#define CHKERRQ(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr,__LINE__,__FILE__);return;endif
#define CHKERRA(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr,__LINE__,__FILE__);call MPIU_Abort(PETSC_COMM_SELF,ierr);endif
#define CHKERRMPI(ierr) if (ierr .ne. 0) then;call PetscErrorMPI(ierr,__LINE__,__FILE__);return;endif
#define CHKERRMPIA(ierr) if (ierr .ne. 0) then;call PetscErrorMPI(ierr,__LINE__,__FILE__);call MPIU_Abort(PETSC_COMM_SELF,ierr);endif
#else
#define CHKERRQ(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr);return;endif
#define CHKERRA(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr);call MPIU_Abort(PETSC_COMM_SELF,ierr);endif
#define CHKERRMPI(ierr) if (ierr .ne. 0) then;call PetscErrorMPI(ierr);return;endif
#define CHKERRMPIA(ierr) if (ierr .ne. 0) then;call PetscErrorMPI(ierr);call MPIU_Abort(PETSC_COMM_SELF,ierr);endif
#endif
#define CHKMEMQ call chkmemfortran(__LINE__,__FILE__,ierr)
#define PetscCall(func) call func; CHKERRQ(ierr)
#define PetscCallMPI(func) call func; CHKERRMPI(ierr)
#define PetscCallA(func) call func; CHKERRA(ierr)
#define PetscCallMPIA(func) call func; CHKERRMPIA(ierr)
#define PetscCheckA(err, c, ierr, s) if (.not.(err)) then; SETERRA(c, ierr, s); endif
#define PetscCheck(err, c, ierr, s) if (.not.(err)) then; SETERRQ(c, ierr, s); endif

#if !defined(PetscFlush)
#if defined(PETSC_HAVE_FORTRAN_FLUSH)
#define PetscFlush(a)    flush(a)
#elif defined(PETSC_HAVE_FORTRAN_FLUSH_)
#define PetscFlush(a)    flush_(a)
#else
#define PetscFlush(a)
#endif
#endif

#define PetscEnumCase(e) case(e%v)

#define PetscObjectSpecificCast(sp,ob) sp%v = ob%v

#endif
