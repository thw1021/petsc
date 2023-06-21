#if !defined(PLOG_H)
#define PLOG_H
#include <petscsys.h>
#include <petsc/private/logimpl.h>

PETSC_INTERN PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *);
PETSC_INTERN PetscErrorCode PetscEventRegLogDestroy(PetscEventRegLog);
PETSC_INTERN PetscErrorCode PetscEventRegLogGetName(PetscEventRegLog, PetscLogEvent, const char **);

PETSC_INTERN PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *);
PETSC_INTERN PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog);

PETSC_INTERN PetscErrorCode PetscStageRegLogCreate(PetscStageRegLog *);
PETSC_INTERN PetscErrorCode PetscStageRegLogDestroy(PetscStageRegLog);
PETSC_INTERN PetscErrorCode PetscStageRegLogGetName(PetscStageRegLog, PetscLogStage, const char **);

#endif // define PLOG_H
