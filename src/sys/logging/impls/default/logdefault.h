#if !defined(PETSCLOGDEFAULT_H)
#define PETSCLOGDEFAULT_H

#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/

typedef struct _n_PetscStageLog *PetscStageLog;

PETSC_INTERN PetscErrorCode PetscLogGetDefaultHandler(PetscStageLog *);

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscStageLog, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **);

PETSC_INTERN PetscErrorCode PetscStageLogDuplicate(PetscStageLog, PetscStageLog *);


#endif // #define PETSCLOGDEFAULT_H
