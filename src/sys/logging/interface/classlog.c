
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/

/*@C
  PetscClassRegLogCreate - This creates a `PetscClassRegLog` object.

  Not Collective

  Input Parameter:
. classLog - The `PetscClassRegLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscClassRegLogDestroy()`, `PetscStageLogCreate()`
@*/
PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *classLog)
{
  PetscFunctionBegin;
  PetscCall(PetscLogResizableArrayCreate(classLog,128));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscClassRegLogDestroy - This destroys a `PetscClassRegLog` object.

  Not Collective

  Input Parameter:
. classLog - The `PetscClassRegLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscClassRegLogCreate()`
@*/
PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog classLog)
{
  int c;

  PetscFunctionBegin;
  for (c = 0; c < classLog->num_entries; c++) PetscCall(PetscClassRegInfoDestroy(&classLog->array[c]));
  PetscCall(PetscFree(classLog->array));
  PetscCall(PetscFree(classLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscClassRegInfoDestroy - This destroys a `PetscClassRegInfo` object.

  Not Collective

  Input Parameter:
. c - The PetscClassRegInfo

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscStageLogDestroy()`, `EventLogDestroy()`
@*/
PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *c)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(c->name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscClassRegLogEnsureSize(PetscClassRegLog class_log, int new_size)
{
  PetscClassRegInfo blank_entry;
  PetscFunctionBegin;
  PetscCall(PetscMemzero(&blank_entry, sizeof(blank_entry)));
  PetscCall(PetscLogResizableArrayEnsureSize(class_log,new_size,blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscClassPerfLogCreate - This creates a `PetscClassPerfLog` object.

  Not Collective

  Input Parameter:
. classLog - The `PetscClassPerfLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscClassPerfLogDestroy()`, `PetscStageLogCreate()`
@*/
PetscErrorCode PetscClassPerfLogCreate(PetscClassPerfLog *classLog)
{
  PetscFunctionBegin;
  PetscCall(PetscLogResizableArrayCreate(classLog,128));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscClassPerfLogDestroy - This destroys a `PetscClassPerfLog` object.

  Not Collective

  Input Parameter:
. classLog - The `PetscClassPerfLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscClassPerfLogCreate()`
@*/
PetscErrorCode PetscClassPerfLogDestroy(PetscClassPerfLog classLog)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(classLog->array));
  PetscCall(PetscFree(classLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ General Functions -------------------------------------------------*/
/*@C
  PetscClassPerfInfoClear - This clears a `PetscClassPerfInfo` object.

  Not Collective

  Input Parameter:
. classInfo - The `PetscClassPerfInfo`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscClassPerfLogCreate()`
@*/
PetscErrorCode PetscClassPerfInfoClear(PetscClassPerfInfo *classInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(classInfo, sizeof(*classInfo)));
  classInfo->id = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscClassPerfLogEnsureSize - This ensures that a `PetscClassPerfLog` is at least of a certain size.

  Not Collective

  Input Parameters:
+ classLog - The `PetscClassPerfLog`
- size     - The size

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscClassPerfLogCreate()`
@*/
PetscErrorCode PetscClassPerfLogEnsureSize(PetscClassPerfLog classLog, int size)
{
  PetscClassPerfInfo blank_entry;

  PetscFunctionBegin;
  PetscCall(PetscClassPerfInfoClear(&blank_entry));
  PetscCall(PetscLogResizableArrayEnsureSize(classLog,size,blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
/*@C
  PetscClassRegLogRegister - Registers a class for logging operations in an application code.

  Not Collective

  Input Parameters:
+ classLog - The `PetscClassRegLog`
- cname    - The name associated with the class

  Output Parameter:
.  classid   - The classid

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscClassIdRegister()`
@*/
PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog classLog, const char cname[], PetscClassId classid)
{
  PetscClassRegInfo *classInfo;
  int                c = classLog->num_entries;

  PetscFunctionBegin;
  PetscValidCharPointer(cname, 2);
  PetscCall(PetscClassRegLogEnsureSize(classLog, c + 1));
  classInfo = &(classLog->array[c]);
  classLog->num_entries++;
  PetscCall(PetscStrallocpy(cname, &(classInfo->name)));
  classInfo->classid = classid;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
/*@C
  PetscClassRegLogGetClass - This function returns the class corresponding to a given classid.

  Not Collective

  Input Parameters:
+ classLog - The `PetscClassRegLog`
- classid  - The cookie

  Output Parameter:
. oclass   - The class id

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscClassIdRegister()`, `PetscLogObjCreateDefault()`, `PetscLogObjDestroyDefault()`
@*/
PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog classLog, PetscClassId classid, int *oclass)
{
  int c;

  PetscFunctionBegin;
  PetscValidIntPointer(oclass, 3);
  for (c = 0; c < classLog->num_entries; c++) {
    if (classLog->array[c].classid == classid) break;
  }
  PetscCheck(c < classLog->num_entries, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid object classid %d\nThis could happen if you compile with PETSC_HAVE_DYNAMIC_LIBRARIES, but link with static libraries.", classid);
  *oclass = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

