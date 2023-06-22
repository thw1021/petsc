
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/
#include <petscdevice.h>
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  #include <../src/sys/perfstubs/timer.h>
#endif
#include <../src/sys/logging/impls/default/logdefault.h>

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
/* Note: these functions do not have prototypes in a public directory, so they are considered "internal" and not exported. */

/*
  PetscEventPerfInfoAdd - Add data in eventInfo to outInfo

  Not collective

  Input Parameter:
. eventInfo - The input `PetscEventPerfInfo`

  Output Parameter:
. outInfo   - The output `PetscEventPerfInfo`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfInfoClear()`
*/
PetscErrorCode PetscEventPerfInfoAdd(const PetscEventPerfInfo *eventInfo, PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->count += eventInfo->count;
  outInfo->time += eventInfo->time;
  outInfo->time2 += eventInfo->time2;
  outInfo->flops += eventInfo->flops;
  outInfo->flops2 += eventInfo->flops2;
  outInfo->numMessages += eventInfo->numMessages;
  outInfo->messageLength += eventInfo->messageLength;
  outInfo->numReductions += eventInfo->numReductions;
#if defined(PETSC_HAVE_DEVICE)
  outInfo->CpuToGpuCount += eventInfo->CpuToGpuCount;
  outInfo->GpuToCpuCount += eventInfo->GpuToCpuCount;
  outInfo->CpuToGpuSize += eventInfo->CpuToGpuSize;
  outInfo->GpuToCpuSize += eventInfo->GpuToCpuSize;
  outInfo->GpuFlops += eventInfo->GpuFlops;
  outInfo->GpuTime += eventInfo->GpuTime;
#endif
  outInfo->memIncrease += eventInfo->memIncrease;
  outInfo->mallocSpace += eventInfo->mallocSpace;
  outInfo->mallocIncreaseEvent += eventInfo->mallocIncreaseEvent;
  outInfo->mallocIncrease += eventInfo->mallocIncrease;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPE)
// TODO: MPE?
  #include <mpe.h>
PETSC_INTERN PetscErrorCode PetscLogMPEGetRGBColor(const char *[]);
PetscErrorCode              PetscLogEventBeginMPE(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(MPE_Log_event(petsc_stageLog->eventLog->array[event].mpe_id_begin, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndMPE(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(MPE_Log_event(petsc_stageLog->eventLog->array[event].mpe_id_end, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*--------------------------------------------- Registration Functions ----------------------------------------------*/

/*---------------------------------------------- Activation Functions -----------------------------------------------*/
/*
  PetscEventPerfLogDeactivatePush - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivatePush(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogDeactivatePop(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivatePop()`
*/
PetscErrorCode PetscEventPerfLogDeactivatePush(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].depth++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscEventPerfLogDeactivatePop - Indicates that a particular event should  be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivatePush(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogDeactivatePop(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivatePush()`
*/
PetscErrorCode PetscEventPerfLogDeactivatePop(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].depth--;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/


