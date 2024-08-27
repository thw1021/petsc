#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/

/*@
   DMTaoPythonSetType - Initialize a `DMTao` object implemented in Python.

   Collective

   Input Parameters:
+  dmta   - The `DMTao` context to contain `DMTao` context
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -dmtao_python_type <pyname> - python class

   Level: intermediate

.seealso: `DMTaoCreate()`, `DMTaoSetType()`, `DMTAOPYTHON`, `PetscPythonInitialize()`
@*/
PetscErrorCode DMTaoPythonSetType(DMTao dmtao, const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmtao, DMTAO_CLASSID, 1);
  PetscAssertPointer(pyname, 2);
  PetscTryMethod(dmtao, "DMTaoPythonSetType_C", (DMTao, const char[]), (dmtao, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMTaoPythonGetType - Get the type of a `DMTao` object implemented in Python.

   Not Collective

   Input Parameter:
.  dmtao - the `DMTao` context

   Output Parameter:
.  pyname - full dotted Python name [package].module[.{class|function}]

   Level: intermediate

.seealso: `DMTaoCreate()`, `DMTaoSetType()`, `DMTAOPYTHON`, `PetscPythonInitialize()`, `DMTaoPythonSetType()`
@*/
PetscErrorCode DMTaoPythonGetType(DMTao dmtao, const char *pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmtao, DMTAO_CLASSID, 1);
  PetscAssertPointer(pyname, 2);
  PetscUseMethod(dmtao, "DMTaoPythonGetType_C", (DMTao, const char *[]), (dmtao, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}
