#if !defined(PETSCLOGDEFAULT_H)
  #define PETSCLOGDEFAULT_H

  #include <petsc/private/loghandlerimpl.h> /*I "petscsys.h" I*/
  #include <petsc/private/logimpl.h> /*I "petscsys.h" I*/

PETSC_INTERN PetscErrorCode _PetscLogHandlerCreate_Default(MPI_Comm comm, PetscLogHandler *);

PETSC_INTERN PetscErrorCode _PetscLogHandlerDefaultSetTrace(PetscLogHandler, FILE *);
PETSC_INTERN PetscErrorCode _PetscLogHandlerDefaultGetEventPerfInfo(PetscLogHandler, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **);
PETSC_INTERN PetscErrorCode _PetscLogHandlerDefaultSetLogActions(PetscLogHandler, PetscBool);
PETSC_INTERN PetscErrorCode _PetscLogHandlerDefaultSetLogObjects(PetscLogHandler, PetscBool);
PETSC_INTERN PetscErrorCode _PetscLogHandlerDefaultLogObjectState(PetscLogHandler, PetscObject, const char[], va_list);
PETSC_INTERN PetscErrorCode _PetscLogHandlerDefaultGetNumObjects(PetscLogHandler, PetscInt *);
#endif // #define PETSCLOGDEFAULT_H
