#if !defined(PETSCLOGDEFAULT_H)
  #define PETSCLOGDEFAULT_H

  #include <petsc/private/logimpl.h> /*I "petscsys.h" I*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Default(PetscLogHandler *);
PETSC_INTERN PetscErrorCode PetscLogView_Default(PetscLogHandler, PetscViewer);

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetTrace(PetscLogHandler, FILE *);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscLogHandler, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogActions(PetscLogHandler, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogObjects(PetscLogHandler, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultLogObjectState(PetscLogHandler, PetscObject, const char[], va_list);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetNumObjects(PetscLogHandler, PetscInt *);
#endif // #define PETSCLOGDEFAULT_H
