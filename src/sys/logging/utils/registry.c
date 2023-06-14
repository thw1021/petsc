
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
  registry->bt_num_stages = registry->stages->max_entries;
  registry->bt_num_events = registry->events->max_entries;

  PetscCall(PetscIntStackCreate(&registry->stage_stack));
  PetscCall(PetscBTCreate(registry->bt_num_stages * registry->bt_num_events, &registry->inactive));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryDestroy(PetscLogRegistry registry)
{
  PetscFunctionBegin;

  PetscCall(PetscEventRegLogDestroy(registry->events));
  PetscCall(PetscClassRegLogDestroy(registry->classes));
  PetscCall(PetscStageRegLogDestroy(registry->stages));
  PetscCall(PetscIntStackDestroy(registry->stage_stack));
  PetscCall(PetscBTDestroy(&registry->inactive));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryStageRegister(PetscLogRegistry registry, const char sname[], int *stage)
{
  PetscFunctionBegin;
  PetscCall(PetscStageRegLogInsert(registry->stages, sname, stage));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryStagePush(PetscLogRegistry registry, PetscLogStage stage)
{
  PetscFunctionBegin;
  PetscCheck(stage >= 0 && stage < registry->stages->num_entries, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, registry->stages->num_entries);

  /* Activate the stage */
  PetscCall(PetscIntStackPush(registry->stage_stack, stage));
  registry->current_stage = stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryStagePop(PetscLogRegistry registry)
{
  int       curStage;
  PetscBool empty;

  PetscFunctionBegin;
  /* Record flops/time of current stage */
  PetscCall(PetscIntStackPop(registry->stage_stack, &curStage));
  PetscCall(PetscIntStackEmpty(registry->stage_stack, &empty));
  if (!empty) {
    /* Subtract current quantities so that we obtain the difference when we pop */
    PetscCall(PetscIntStackTop(registry->stage_stack, &registry->current_stage));
  } else registry->current_stage = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
