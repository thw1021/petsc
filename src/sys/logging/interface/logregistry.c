
#include <petsc/private/logimpl.h> /*I "petsclog.h" I*/

PETSC_INTERN PetscErrorCode PetscLogRegistryCreate(PetscLogRegistry *registry_p)
{
  PetscLogRegistry registry;

  PetscFunctionBegin;
  PetscCall(PetscNew(registry_p));
  registry = *registry_p;
  PetscCall(PetscEventRegLogCreate(&registry->events));
  PetscCall(PetscClassRegLogCreate(&registry->classes));
  PetscCall(PetscStageRegLogCreate(&registry->stages));
  PetscCall(PetscSpinlockCreate(&registry->lock));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryDestroy(PetscLogRegistry registry)
{
  PetscFunctionBegin;

  PetscCall(PetscEventRegLogDestroy(registry->events));
  PetscCall(PetscClassRegLogDestroy(registry->classes));
  PetscCall(PetscStageRegLogDestroy(registry->stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryStageRegister(PetscLogRegistry registry, const char sname[], int *stage)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryLock(registry));
  PetscCall(PetscStageRegLogInsert(registry->stages, sname, stage));
  PetscCall(PetscLogRegistryUnlock(registry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogRegistryEventRegister(PetscLogRegistry registry, const char name[], PetscClassId classid, PetscLogEvent *event)
{
  PetscFunctionBegin;
  *event = PETSC_DECIDE;
  PetscCall(PetscEventRegLogGetEvent(registry->events, name, event));
  if (*event > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscEventRegLogRegister(registry->events, name, classid, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryLock(PetscLogRegistry registry)
{
  PetscFunctionBegin;
  PetscCall(PetscSpinlockLock(&registry->lock));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryUnlock(PetscLogRegistry registry)
{
  PetscFunctionBegin;
  PetscCall(PetscSpinlockLock(&registry->lock));
  PetscFunctionReturn(PETSC_SUCCESS);
}
