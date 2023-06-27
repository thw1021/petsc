#if !defined(PETSCLOGDEFAULT_H)
  #define PETSCLOGDEFAULT_H

  #include <petsc/private/logimpl.h> /*I "petscsys.h" I*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Default(PetscLogHandlerEntry *);
PETSC_INTERN PetscErrorCode PetscLogView_Default(PetscLogHandlerEntry, PetscViewer);

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetTrace(PetscLogHandlerEntry, FILE *);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscLogHandlerEntry, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogActions(PetscLogHandlerEntry, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogObjects(PetscLogHandlerEntry, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultLogObjectState(PetscLogHandlerEntry, PetscObject, const char[], va_list);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetNumObjects(PetscLogHandlerEntry, PetscInt *);
#endif // #define PETSCLOGDEFAULT_H
