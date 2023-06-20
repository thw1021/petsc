
#include <petsc/private/logimpl.h> /*I "petsclog.h" I*/
#include "plog.h"

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
  PetscCall(PetscStageRegLogInsert(registry->stages, sname, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryGetEvent(PetscLogRegistry registry, const char name[], PetscLogEvent *event)
{
  PetscFunctionBegin;
  PetscCall(PetscEventRegLogGetEvent(registry->events, name, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryEventRegister(PetscLogRegistry registry, const char name[], PetscClassId classid, PetscLogEvent *event)
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

PETSC_INTERN PetscErrorCode PetscLogRegistryStageSetVisible(PetscLogRegistry registry, PetscLogStage stage)
{
  PetscFunctionBegin;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Given a list of strings on each process, create a global numbering.  Order them by their order on the first process, then the remaining by their order on the second process, etc.
// The expectation is that most processes have the same names in the same order so it shouldn't take too many rounds to figure out
static PetscErrorCode PetscLogGlobalNamesCreate_Internal(MPI_Comm comm, PetscInt num_names_local, const char **names, PetscInt *num_names_global_p, PetscInt **global_index_to_local_index_p, PetscInt **local_index_to_global_index_p, const char ***global_names_p)
{
  PetscMPIInt size, rank;
  PetscInt    num_names_global          = 0;
  PetscInt    num_names_local_remaining = num_names_local;
  PetscBool  *local_name_seen;
  PetscInt   *global_index_to_local_index = NULL;
  PetscInt   *local_index_to_global_index = NULL;
  PetscInt    max_name_len                = 0;
  char       *str_buffer;
  char      **global_names = NULL;
  PetscMPIInt p;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    PetscCall(PetscMalloc1(num_names_local, &global_index_to_local_index));
    PetscCall(PetscMalloc1(num_names_local, &local_index_to_global_index));
    PetscCall(PetscMalloc1(num_names_local, &global_names));
    for (PetscInt i = 0; i < num_names_local; i++) {
      global_index_to_local_index[i] = i;
      local_index_to_global_index[i] = i;
      PetscCall(PetscStrallocpy(names[i], &global_names[i]));
    }
    *num_names_global_p            = num_names_local;
    *global_index_to_local_index_p = global_index_to_local_index;
    *local_index_to_global_index_p = local_index_to_global_index;
    *global_names_p                = (const char **)global_names;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscCalloc1(num_names_local, &local_name_seen));
  PetscCall(PetscMalloc1(num_names_local, &local_index_to_global_index));

  for (PetscInt i = 0; i < num_names_local; i++) {
    size_t i_len;
    PetscCall(PetscStrlen(names[i], &i_len));
    max_name_len = PetscMax(max_name_len, (PetscInt)i_len);
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &max_name_len, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscCalloc1(max_name_len + 1, &str_buffer));

  p = 0;
  while (p < size) {
    PetscInt my_loc, next_loc;
    PetscInt num_to_add;

    my_loc = num_names_local_remaining > 0 ? rank : PETSC_MPI_INT_MAX;
    PetscCallMPI(MPIU_Allreduce(&my_loc, &next_loc, 1, MPIU_INT, MPI_MIN, comm));
    if (next_loc == PETSC_MPI_INT_MAX) break;
    PetscAssert(next_loc >= p, comm, PETSC_ERR_PLIB, "Failed invariant, expected increasing next process");
    p          = next_loc;
    num_to_add = (rank == p) ? num_names_local_remaining : -1;
    PetscCallMPI(MPI_Bcast(&num_to_add, 1, MPIU_INT, p, comm));
    {
      PetscInt  new_num_names_global = num_names_global + num_to_add;
      PetscInt *new_global_index_to_local_index;
      char    **new_global_names;

      PetscCall(PetscMalloc1(new_num_names_global, &new_global_index_to_local_index));
      PetscCall(PetscArraycpy(new_global_index_to_local_index, global_index_to_local_index, num_names_global));
      for (PetscInt i = num_names_global; i < new_num_names_global; i++) new_global_index_to_local_index[i] = -1;
      PetscCall(PetscFree(global_index_to_local_index));
      global_index_to_local_index = new_global_index_to_local_index;

      PetscCall(PetscCalloc1(new_num_names_global, &new_global_names));
      PetscCall(PetscArraycpy(new_global_names, global_names, num_names_global));
      PetscCall(PetscFree(global_names));
      global_names = new_global_names;
    }

    if (rank == p) {
      for (PetscInt s = 0; s < num_names_local; s++) {
        if (local_name_seen[s]) continue;
        local_name_seen[s] = PETSC_TRUE;
        PetscCall(PetscArrayzero(str_buffer, max_name_len + 1));
        PetscCall(PetscStrallocpy(names[s], &global_names[num_names_global]));
        PetscCall(PetscStrncpy(str_buffer, names[s], max_name_len + 1));
        PetscCallMPI(MPI_Bcast(str_buffer, max_name_len + 1, MPI_CHAR, p, comm));
        local_index_to_global_index[s]                  = num_names_global;
        global_index_to_local_index[num_names_global++] = s;
        num_names_local_remaining--;
      }
    } else {
      for (PetscInt i = 0; i < num_to_add; i++) {
        PetscInt s;
        PetscCallMPI(MPI_Bcast(str_buffer, max_name_len + 1, MPI_CHAR, p, comm));
        PetscCall(PetscStrallocpy(str_buffer, &global_names[num_names_global]));
        for (s = 0; s < num_names_local; s++) {
          PetscBool same;

          if (local_name_seen[s]) continue;
          PetscCall(PetscStrncmp(names[s], str_buffer, max_name_len + 1, &same));
          if (same) {
            local_name_seen[s]                            = PETSC_TRUE;
            global_index_to_local_index[num_names_global] = s;
            local_index_to_global_index[s]                = num_names_global;
            num_names_local_remaining--;
            break;
          }
        }
        if (s == num_names_local) {
          global_index_to_local_index[num_names_global] = -1; // this name is not present on this process
        }
        num_names_global++;
      }
    }
  }

  PetscCall(PetscFree(str_buffer));
  PetscCall(PetscFree(local_name_seen));
  *num_names_global_p            = num_names_global;
  *global_index_to_local_index_p = global_index_to_local_index;
  *local_index_to_global_index_p = local_index_to_global_index;
  *global_names_p                = (const char **)global_names;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesCreate(MPI_Comm comm, PetscInt num_names_local, const char **local_names, PetscLogGlobalNames *global_names_p)
{
  PetscLogGlobalNames global_names;

  PetscFunctionBegin;
  PetscCall(PetscNew(&global_names));
  PetscCall(PetscLogGlobalNamesCreate_Internal(comm, num_names_local, local_names, &global_names->count, &global_names->global_to_local, &global_names->local_to_global, &global_names->names));
  *global_names_p = global_names;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesDestroy(PetscLogGlobalNames *global_names_p)
{
  PetscLogGlobalNames global_names;

  PetscFunctionBegin;
  global_names    = *global_names_p;
  *global_names_p = NULL;
  PetscCall(PetscFree(global_names->global_to_local));
  PetscCall(PetscFree(global_names->local_to_global));
  for (PetscInt i = 0; i < global_names->count; i++) { PetscCall(PetscFree(global_names->names[i])); }
  PetscCall(PetscFree(global_names->names));
  PetscCall(PetscFree(global_names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryCreateGlobalStageNames(MPI_Comm comm, PetscLogRegistry registry, PetscLogGlobalNames *global_names_p)
{
  PetscInt     num_stages_local = registry->stages->num_entries;
  const char **names;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(num_stages_local, &names));
  for (PetscInt i = 0; i < num_stages_local; i++) names[i] = registry->stages->array[i].name;
  PetscCall(PetscLogGlobalNamesCreate(comm, num_stages_local, names, global_names_p));
  PetscCall(PetscFree(names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryCreateGlobalEventNames(MPI_Comm comm, PetscLogRegistry registry, PetscLogGlobalNames *global_names_p)
{
  PetscInt     num_events_local;
  const char **names;

  PetscFunctionBegin;
  num_events_local = registry->events->num_entries;
  PetscCall(PetscMalloc1(num_events_local, &names));
  for (PetscInt i = 0; i < num_events_local; i++) names[i] = registry->events->array[i].name;
  PetscCall(PetscLogGlobalNamesCreate(comm, num_events_local, names, global_names_p));
  PetscCall(PetscFree(names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

