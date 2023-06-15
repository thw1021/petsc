
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/

PETSC_INTERN PetscErrorCode PetscStageRegLogCreate(PetscStageRegLog *stageLog)
{
  PetscStageRegLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));
  l->num_entries = 0;
  l->max_entries = 8;
  PetscCall(PetscMalloc1(l->max_entries, &l->array));
  *stageLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscStageRegLogEnsureSize(PetscStageRegLog stage_log, int new_size)
{
  PetscStageRegInfo blank_entry;

  PetscFunctionBegin;
  PetscCall(PetscMemzero(&blank_entry, sizeof(blank_entry)));
  PetscCall(PetscLogResizableArrayEnsureSize(stage_log,new_size,blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscStageRegLogInsert(PetscStageRegLog stage_log, const char sname[], int *stage)
{
  PetscStageRegInfo *stage_info;
  PetscFunctionBegin;
  PetscValidCharPointer(sname, 2);
  PetscValidIntPointer(stage, 3);
  for (int s = 0; s < stage_log->num_entries; s++) {
    PetscBool same;

    PetscCall(PetscStrcmp(stage_log->array[s].name, sname, &same));
    PetscCheck(!same, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Duplicate stage name given: %s", sname);
  }
  *stage = stage_log->num_entries;
  PetscCall(PetscStageRegLogEnsureSize(stage_log, stage_log->num_entries + 1));
  stage_info = &(stage_log->array[stage_log->num_entries++]);
  PetscCall(PetscMemzero(stage_info, sizeof(*stage_info)));
  PetscCall(PetscStrallocpy(sname, &stage_info->name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogGetEventLog(PetscEventRegLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog, 1);
  if (!petsc_log_registry) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *eventLog = petsc_log_registry->events;;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogGetClassLog(PetscClassRegLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 1);
  if (!petsc_log_registry) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *classLog = petsc_log_registry->classes;;
  PetscFunctionReturn(PETSC_SUCCESS);
}

