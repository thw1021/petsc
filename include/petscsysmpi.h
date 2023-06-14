#if !defined(PETSCSYSMPI_H)
#define PETSCSYSMPI_H

#include <petscsystypes.h>

/* ========================================================================== */

/*
    Defines the interface to MPI allowing the use of all MPI functions.

    PETSc does not use the C++ binding of MPI at ALL. The following flag
    makes sure the C++ bindings are not included. The C++ bindings REQUIRE
    putting mpi.h before ANY C++ include files, we cannot control this
    with all PETSc users. Users who want to use the MPI C++ bindings can include
    mpicxx.h directly in their code
*/
#if !defined(MPICH_SKIP_MPICXX)
  #define MPICH_SKIP_MPICXX 1
#endif
#if !defined(OMPI_SKIP_MPICXX)
  #define OMPI_SKIP_MPICXX 1
#endif
#if defined(PETSC_HAVE_MPIUNI)
  #include <petsc/mpiuni/mpi.h>
#else
  #include <mpi.h>
#endif

/*
    Need to put stdio.h AFTER mpi.h for MPICH2 with C++ compiler
    see the top of mpicxx.h in the MPICH2 distribution.
*/
#include <stdio.h>

/*
   Perform various sanity checks that the correct mpi.h is being included at compile time.
   This usually happens because
      * either an unexpected mpi.h is in the default compiler path (i.e. in /usr/include) or
      * an extra include path -I/something (which contains the unexpected mpi.h) is being passed to the compiler
*/
#if defined(PETSC_HAVE_MPIUNI)
  #ifndef MPIUNI_H
    #error "PETSc was configured with --with-mpi=0 but now appears to be compiling using a different mpi.h"
  #endif
#elif defined(PETSC_HAVE_I_MPI_NUMVERSION)
  #if !defined(I_MPI_NUMVERSION)
    #error "PETSc was configured with I_MPI but now appears to be compiling using a non-I_MPI mpi.h"
  #elif I_MPI_NUMVERSION != PETSC_HAVE_I_MPI_NUMVERSION
    #error "PETSc was configured with one I_MPI mpi.h version but now appears to be compiling using a different I_MPI mpi.h version"
  #endif
#elif defined(PETSC_HAVE_MVAPICH2_NUMVERSION)
  #if !defined(MVAPICH2_NUMVERSION)
    #error "PETSc was configured with MVAPICH2 but now appears to be compiling using a non-MVAPICH2 mpi.h"
  #elif MVAPICH2_NUMVERSION != PETSC_HAVE_MVAPICH2_NUMVERSION
    #error "PETSc was configured with one MVAPICH2 mpi.h version but now appears to be compiling using a different MVAPICH2 mpi.h version"
  #endif
#elif defined(PETSC_HAVE_MPICH_NUMVERSION)
  #if !defined(MPICH_NUMVERSION) || defined(MVAPICH2_NUMVERSION) || defined(I_MPI_NUMVERSION)
    #error "PETSc was configured with MPICH but now appears to be compiling using a non-MPICH mpi.h"
  #elif (MPICH_NUMVERSION / 100000000 != PETSC_HAVE_MPICH_NUMVERSION / 100000000) || (MPICH_NUMVERSION / 100000 < PETSC_HAVE_MPICH_NUMVERSION / 100000) || (MPICH_NUMVERSION / 100000 == PETSC_HAVE_MPICH_NUMVERSION / 100000 && MPICH_NUMVERSION % 100000 / 1000 < PETSC_HAVE_MPICH_NUMVERSION % 100000 / 1000)
    #error "PETSc was configured with one MPICH mpi.h version but now appears to be compiling using a different MPICH mpi.h version"
  #endif
#elif defined(PETSC_HAVE_OMPI_MAJOR_VERSION)
  #if !defined(OMPI_MAJOR_VERSION)
    #error "PETSc was configured with OpenMPI but now appears to be compiling using a non-OpenMPI mpi.h"
  #elif (OMPI_MAJOR_VERSION != PETSC_HAVE_OMPI_MAJOR_VERSION) || (OMPI_MINOR_VERSION < PETSC_HAVE_OMPI_MINOR_VERSION) || (OMPI_MINOR_VERSION == PETSC_HAVE_OMPI_MINOR_VERSION && OMPI_RELEASE_VERSION < PETSC_HAVE_OMPI_RELEASE_VERSION)
    #error "PETSc was configured with one OpenMPI mpi.h version but now appears to be compiling using a different OpenMPI mpi.h version"
  #endif
#elif defined(PETSC_HAVE_MSMPI_VERSION)
  #if !defined(MSMPI_VER)
    #error "PETSc was configured with MSMPI but now appears to be compiling using a non-MSMPI mpi.h"
  #elif (MSMPI_VER != PETSC_HAVE_MSMPI_VERSION)
    #error "PETSc was configured with one MSMPI mpi.h version but now appears to be compiling using a different MSMPI mpi.h version"
  #endif
#elif defined(OMPI_MAJOR_VERSION) || defined(MPICH_NUMVERSION) || defined(MSMPI_VER)
  #error "PETSc was configured with undetermined MPI - but now appears to be compiling using any of OpenMPI, MS-MPI or a MPICH variant"
#endif

/* MSMPI on 32-bit Microsoft Windows requires this yukky hack - that breaks MPI standard compliance */
#if !defined(MPIAPI)
  #define MPIAPI
#endif

PETSC_EXTERN MPI_Datatype MPIU_ENUM PETSC_ATTRIBUTE_MPI_TYPE_TAG(PetscEnum);
PETSC_EXTERN MPI_Datatype MPIU_BOOL PETSC_ATTRIBUTE_MPI_TYPE_TAG(PetscBool);

/*MC
   MPIU_INT - Portable MPI datatype corresponding to `PetscInt` independent of the precision of `PetscInt`

   Level: beginner

   Note:
   In MPI calls that require an MPI datatype that matches a `PetscInt` or array of `PetscInt` values, pass this value.

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_COUNT`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`
M*/

PETSC_EXTERN MPI_Datatype MPIU_FORTRANADDR;

#if defined(PETSC_USE_64BIT_INDICES)
  #define MPIU_INT MPIU_INT64
#else
  #define MPIU_INT MPI_INT
#endif

/*MC
   MPIU_COUNT - Portable MPI datatype corresponding to `PetscCount` independent of the precision of `PetscCount`

   Level: beginner

   Note:
   In MPI calls that require an MPI datatype that matches a `PetscCount` or array of `PetscCount` values, pass this value.

  Developer Note:
  It seems MPI_AINT is unsigned so this may be the wrong choice here since `PetscCount` is signed

.seealso: `PetscReal`, `PetscScalar`, `PetscComplex`, `PetscInt`, `MPIU_INT`, `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`
M*/
#define MPIU_COUNT MPI_AINT

/*
    For the rare cases when one needs to send a size_t object with MPI
*/
PETSC_EXTERN MPI_Datatype MPIU_SIZE_T PETSC_ATTRIBUTE_MPI_TYPE_TAG(size_t);

/*MC
    PETSC_COMM_WORLD - the equivalent of the `MPI_COMM_WORLD` communicator which represents
           all the processes that PETSc knows about.

   Level: beginner

   Notes:
   By default `PETSC_COMM_WORLD` and `MPI_COMM_WORLD` are identical unless you wish to
          run PETSc on ONLY a subset of `MPI_COMM_WORLD`. In that case create your new (smaller)
          communicator, call it, say comm, and set `PETSC_COMM_WORLD` = comm BEFORE calling
          PetscInitialize(), but after `MPI_Init()` has been called.

          The value of `PETSC_COMM_WORLD` should never be USED/accessed before `PetscInitialize()`
          is called because it may not have a valid value yet.

.seealso: `PETSC_COMM_SELF`
M*/
PETSC_EXTERN MPI_Comm PETSC_COMM_WORLD;

/*MC
    PETSC_COMM_SELF - This is always `MPI_COMM_SELF`

   Level: beginner

   Notes:
   Do not USE/access or set this variable before `PetscInitialize()` has been called.

.seealso: `PETSC_COMM_WORLD`
M*/
#define PETSC_COMM_SELF MPI_COMM_SELF

/*MC
    PETSC_MPI_THREAD_REQUIRED - the required threading support used if PETSc initializes
           MPI with `MPI_Init_thread()`.

   Level: beginner

   Notes:
   By default `PETSC_MPI_THREAD_REQUIRED` equals `MPI_THREAD_FUNNELED`.

.seealso: `PetscInitialize()`
M*/
PETSC_EXTERN PetscMPIInt PETSC_MPI_THREAD_REQUIRED;

PETSC_EXTERN PetscBool PetscBeganMPI;

PETSC_EXTERN PetscErrorCode PetscCommDuplicate(MPI_Comm, MPI_Comm *, int *);
PETSC_EXTERN PetscErrorCode PetscCommDestroy(MPI_Comm *);
PETSC_EXTERN PetscErrorCode PetscCommGetComm(MPI_Comm, MPI_Comm *);
PETSC_EXTERN PetscErrorCode PetscCommRestoreComm(MPI_Comm, MPI_Comm *);

#define MPIU_PETSCLOGDOUBLE  MPI_DOUBLE
#define MPIU_2PETSCLOGDOUBLE MPI_2DOUBLE_PRECISION

/*
   These are MPI operations for MPI_Allreduce() etc
*/
PETSC_EXTERN MPI_Op MPIU_MAXSUM_OP;
#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
PETSC_EXTERN MPI_Op MPIU_SUM;
PETSC_EXTERN MPI_Op MPIU_MAX;
PETSC_EXTERN MPI_Op MPIU_MIN;
#else
  #define MPIU_SUM MPI_SUM
  #define MPIU_MAX MPI_MAX
  #define MPIU_MIN MPI_MIN
#endif
PETSC_EXTERN MPI_Op         Petsc_Garbage_SetIntersectOp;

#if (defined(PETSC_HAVE_REAL___FLOAT128) && !defined(PETSC_SKIP_REAL___FLOAT128)) || (defined(PETSC_HAVE_REAL___FP16) && !defined(PETSC_SKIP_REAL___FP16))
/*MC
    MPIU_SUM___FP16___FLOAT128 - MPI_Op that acts as a replacement for `MPI_SUM` with
    custom `MPI_Datatype` `MPIU___FLOAT128`, `MPIU___COMPLEX128`, and `MPIU___FP16`.

   Level: advanced

   Developer Note:
   This should be unified with `MPIU_SUM`

.seealso: `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_COMPLEX`
M*/
PETSC_EXTERN MPI_Op MPIU_SUM___FP16___FLOAT128;
#endif
PETSC_EXTERN PetscErrorCode PetscMaxSum(MPI_Comm, const PetscInt[], PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode MPIULong_Send(void *, PetscInt, MPI_Datatype, PetscMPIInt, PetscMPIInt, MPI_Comm) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(1, 3);
PETSC_EXTERN PetscErrorCode MPIULong_Recv(void *, PetscInt, MPI_Datatype, PetscMPIInt, PetscMPIInt, MPI_Comm) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(1, 3);

/*MC
    MPI_Comm - the basic object used by MPI to determine which processes are involved in a
        communication

   Level: beginner

   Note:
   This manual page is a place-holder because MPICH does not have a manual page for `MPI_Comm`

.seealso: `PETSC_COMM_WORLD`, `PETSC_COMM_SELF`
M*/

#if defined(PETSC_HAVE_MPIIO)
PETSC_EXTERN PetscErrorCode MPIU_File_write_all(MPI_File, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(2, 4);
PETSC_EXTERN PetscErrorCode MPIU_File_read_all(MPI_File, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(2, 4);
PETSC_EXTERN PetscErrorCode MPIU_File_write_at(MPI_File, MPI_Offset, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(3, 5);
PETSC_EXTERN PetscErrorCode MPIU_File_read_at(MPI_File, MPI_Offset, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(3, 5);
PETSC_EXTERN PetscErrorCode MPIU_File_write_at_all(MPI_File, MPI_Offset, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(3, 5);
PETSC_EXTERN PetscErrorCode MPIU_File_read_at_all(MPI_File, MPI_Offset, void *, PetscMPIInt, MPI_Datatype, MPI_Status *) PETSC_ATTRIBUTE_MPI_POINTER_WITH_TYPE(3, 5);
#endif

#endif // #define PETSCSYSMPI_H
