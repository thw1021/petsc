
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


