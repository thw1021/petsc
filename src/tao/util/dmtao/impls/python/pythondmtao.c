#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/

/*@
   DMTaoPythonSetType - Initialize a `DMTao` object implemented in Python.

   Collective

   Input Parameters:
+  dm     - The `DM` context to contain `DMTao` context
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -dmtao_python_type <pyname> - python class

   Level: intermediate

.seealso: `DMTaoCreate()`, `DMTaoSetType()`, `DMTAOPYTHON`, `PetscPythonInitialize()`
@*/
PetscErrorCode DMTaoPythonSetType(DM dm, const char pyname[])
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(pyname, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscTryMethod(tdm, "TaoPythonSetType_C", (DMTao, const char[]), (tdm, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMTaoPythonGetType - Get the type of a `DMTao` object implemented in Python.

   Not Collective

   Input Parameter:
.  dm - the `DM` context

   Output Parameter:
.  pyname - full dotted Python name [package].module[.{class|function}]

   Level: intermediate

.seealso: `DMTaoCreate()`, `DMTaoSetType()`, `DMTAOPYTHON`, `PetscPythonInitialize()`, `DMTaoPythonSetType()`
@*/
PetscErrorCode DMTaoPythonGetType(DM dm, const char *pyname[])
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(pyname, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscUseMethod(dm, "DMTaoPythonGetType_C", (DMTao, const char *[]), (tdm, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}
