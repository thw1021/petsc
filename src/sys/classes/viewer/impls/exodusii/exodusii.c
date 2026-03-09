#include <petsc/private/viewerimpl.h>
#include <petsc/private/viewerexodusiiimpl.h>

/*@C
  PETSC_VIEWER_EXODUSII_ - Creates an `PETSCVIEWEREXODUSII` `PetscViewer` shared by all processors in a communicator.

  Collective; No Fortran Support

  Input Parameter:
. comm - the MPI communicator to share the `PETSCVIEWEREXODUSII` `PetscViewer`

  Level: intermediate

  Note:
  Unlike almost all other PETSc routines, `PETSC_VIEWER_EXODUSII_()` does not return
  an error code.  The GLVIS PetscViewer is usually used in the form
.vb
  XXXView(XXX object, PETSC_VIEWER_EXODUSII_(comm));
.ve

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewer`, `PetscViewerExodusIIOpen()`, `PetscViewerType`, `PetscViewerCreate()`, `PetscViewerDestroy()`
@*/
PetscViewer PETSC_VIEWER_EXODUSII_(MPI_Comm comm)
{
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscCallNull(PetscViewerExodusIIOpen(comm, "mesh.exo", FILE_MODE_WRITE, &viewer));
  PetscCallNull(PetscObjectRegisterDestroy((PetscObject)viewer));
  PetscFunctionReturn(viewer);
}

/*@
  PetscViewerExodusIISetZonalVariable - Sets the number of zonal variables in an ExodusII file

  Collective;

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- num    - the number of zonal variables in the ExodusII file

  Level: intermediate

  Notes:
  The ExodusII API does not allow changing the number of variables in a file so this function will return an error
  if called twice, called on a read-only file, or called on file for which the number of variables has already been specified

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIIGetZonalVariable()`
@*/
PetscErrorCode PetscViewerExodusIISetZonalVariable(PetscViewer viewer, PetscExodusIIInt num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIISetZonalVariable_C", (PetscViewer, PetscExodusIIInt), (viewer, num));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIISetNodalVariable - Sets the number of nodal variables in an ExodusII file

  Collective;

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- num    - the number of nodal variables in the ExodusII file

  Level: intermediate

  Notes:
  The ExodusII API does not allow changing the number of variables in a file so this function will return an error
  if called twice, called on a read-only file, or called on file for which the number of variables has already been specified

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIIGetNodalVariable()`
@*/
PetscErrorCode PetscViewerExodusIISetNodalVariable(PetscViewer viewer, PetscExodusIIInt num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIISetNodalVariable_C", (PetscViewer, PetscExodusIIInt), (viewer, num));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetZonalVariable - Gets the number of zonal variables in an ExodusII file

  Collective

  Input Parameters:
. viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`

  Output Parameter:
. num - the number variables in the ExodusII file

  Level: intermediate

  Notes:
  The number of variables in the ExodusII file is cached in the viewer

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIIsetZonalVariable()`
@*/
PetscErrorCode PetscViewerExodusIIGetZonalVariable(PetscViewer viewer, PetscExodusIIInt *num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIIGetZonalVariable_C", (PetscViewer, PetscExodusIIInt *), (viewer, num));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetNodalVariable - Gets the number of nodal variables in an ExodusII file

  Collective

  Input Parameters:
. viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`

  Output Parameter:
. num - the number variables in the ExodusII file

  Level: intermediate

  Notes:
  This function gets the number of nodal variables and saves it in the address of num.

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIISetNodalVariable()`
@*/
PetscErrorCode PetscViewerExodusIIGetNodalVariable(PetscViewer viewer, PetscExodusIIInt *num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIIGetNodalVariable_C", (PetscViewer, PetscExodusIIInt *), (viewer, num));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIISetZonalVariableName - Sets the name of a zonal variable.

  Collective;

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
. idx    - the index for which you want to save the name
- name   - string containing the name characters

  Level: intermediate

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIIGetZonalVariableName()`
@*/
PetscErrorCode PetscViewerExodusIISetZonalVariableName(PetscViewer viewer, PetscExodusIIInt idx, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIISetZonalVariableName_C", (PetscViewer, PetscExodusIIInt, const char[]), (viewer, idx, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIISetNodalVariableName - Sets the name of a nodal variable.

  Collective;

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
. idx    - the index for which you want to save the name
- name   - string containing the name characters

  Level: intermediate

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIIGetNodalVariableName()`
@*/
PetscErrorCode PetscViewerExodusIISetNodalVariableName(PetscViewer viewer, PetscExodusIIInt idx, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIISetNodalVariableName_C", (PetscViewer, PetscExodusIIInt, const char[]), (viewer, idx, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetZonalVariableName - Gets the name of a zonal variable.

  Collective;

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- idx    - the index for which you want to get the name

  Output Parameter:
. name - pointer to the string containing the name characters

  Level: intermediate

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIISetZonalVariableName()`
@*/
PetscErrorCode PetscViewerExodusIIGetZonalVariableName(PetscViewer viewer, PetscExodusIIInt idx, const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIIGetZonalVariableName_C", (PetscViewer, PetscExodusIIInt, const char *[]), (viewer, idx, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetNodalVariableName - Gets the name of a nodal variable.

  Collective;

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- idx    - the index for which you want to save the name

  Output Parameter:
. name - string array containing name characters

  Level: intermediate

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIISetNodalVariableName()`
@*/
PetscErrorCode PetscViewerExodusIIGetNodalVariableName(PetscViewer viewer, PetscExodusIIInt idx, const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIIGetNodalVariableName_C", (PetscViewer, PetscExodusIIInt, const char *[]), (viewer, idx, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerExodusIISetZonalVariableNames - Sets the names of all nodal variables

  Collective; No Fortran Support

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- names  - an array of string names to be set, the strings are copied into the `PetscViewer`

  Level: intermediate

  Notes:
  This function allows users to set multiple zonal variable names at a time.

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIIGetZonalVariableNames()`
@*/
PetscErrorCode PetscViewerExodusIISetZonalVariableNames(PetscViewer viewer, const char *const names[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIISetZonalVariableNames_C", (PetscViewer, const char *const[]), (viewer, names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerExodusIISetNodalVariableNames - Sets the names of all nodal variables.

  Collective; No Fortran Support

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- names  - an array of string names to be set, the strings are copied into the `PetscViewer`

  Level: intermediate

  Notes:
  This function allows users to set multiple nodal variable names at a time.

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIIGetNodalVariableNames()`
@*/
PetscErrorCode PetscViewerExodusIISetNodalVariableNames(PetscViewer viewer, const char *const names[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIISetNodalVariableNames_C", (PetscViewer, const char *const[]), (viewer, names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerExodusIIGetZonalVariableNames - Gets the names of all zonal variables.

  Collective; No Fortran Support

  Input Parameters:
+ viewer  - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- numVars - the number of zonal variable names to retrieve

  Output Parameter:
. varNames - returns an array of char pointers where the zonal variable names are

  Level: intermediate

  Notes:
  This function returns a borrowed pointer which should not be freed.

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIISetZonalVariableNames()`
@*/
PetscErrorCode PetscViewerExodusIIGetZonalVariableNames(PetscViewer viewer, PetscExodusIIInt *numVars, const char *const *varNames[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIIGetZonalVariableNames_C", (PetscViewer, PetscExodusIIInt *, const char *const *[]), (viewer, numVars, varNames));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerExodusIIGetNodalVariableNames - Gets the names of all nodal variables.

  Collective; No Fortran Support

  Input Parameters:
+ viewer  - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- numVars - the number of nodal variable names to retrieve

  Output Parameter:
. varNames - returns an array of char pointers where the nodal variable names are

  Level: intermediate

  Notes:
  This function returns a borrowed pointer which should not be freed.

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`, `PetscViewerExodusIIOpen()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewerExodusIISetNodalVariableNames()`
@*/
PetscErrorCode PetscViewerExodusIIGetNodalVariableNames(PetscViewer viewer, PetscExodusIIInt *numVars, const char *const *varNames[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIIGetNodalVariableNames_C", (PetscViewer, PetscExodusIIInt *, const char *const *[]), (viewer, numVars, varNames));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetNodalVariableIndex - return the location of a nodal variable in an ExodusII file given its name

  Collective

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- name   - the name of the result

  Output Parameter:
. varIndex - the location of the variable in the exodus file or -1 if the variable is not found

  Level: beginner

  Notes:
  The exodus variable index is obtained by comparing the name argument to the
  names of zonal variables declared in the exodus file. For instance if name is "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all variables of type obj_type.

.seealso: `PetscViewerExodusIISetNodalVariable()`, `PetscViewerExodusIIGetNodalVariable()`, `PetscViewerExodusIISetNodalVariableName()`, `PetscViewerExodusIISetNodalVariableNames()`, `PetscViewerExodusIIGetNodalVariableName()`, `PetscViewerExodusIIGetNodalVariableNames()`
@*/
PetscErrorCode PetscViewerExodusIIGetNodalVariableIndex(PetscViewer viewer, const char name[], PetscExodusIIInt *varIndex)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIIGetNodalVariableIndex(_C", (PetscViewer, const char[], PetscExodusIIInt *), (viewer, name, varIndex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetZonalVariableIndex - return the location of a zonal variable in an ExodusII file given its name

  Collective

  Input Parameters:
+ viewer - a `PetscViewer` of type `PETSCVIEWEREXODUSII`
- name   - the name of the result

  Output Parameter:
. varIndex - the location of the variable in the exodus file or -1 if the variable is not found

  Level: beginner

  Notes:
  The exodus variable index is obtained by comparing the name argument to the
  names of zonal variables declared in the exodus file. For instance if name is "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all variables of type obj_type.

.seealso: `PetscViewerExodusIISetNodalVariable()`, `PetscViewerExodusIIGetNodalVariable()`, `PetscViewerExodusIISetNodalVariableName()`, `PetscViewerExodusIISetNodalVariableNames()`, `PetscViewerExodusIIGetNodalVariableName()`, `PetscViewerExodusIIGetNodalVariableNames()`
@*/
PetscErrorCode PetscViewerExodusIIGetZonalVariableIndex(PetscViewer viewer, const char name[], int *varIndex)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerExodusIIGetZonalVariableIndex_C", (PetscViewer, const char[], int *), (viewer, name, varIndex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetId - Get the file id of the `PETSCVIEWEREXODUSII` file

  Logically Collective

  Input Parameter:
. viewer - the `PetscViewer`

  Output Parameter:
. exoid - The ExodusII file id

  Level: intermediate

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`
@*/
PetscErrorCode PetscViewerExodusIIGetId(PetscViewer viewer, int *exoid)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerExodusIIGetId_C", (PetscViewer, int *), (viewer, exoid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIISetOrder - Set the elements order in the ExodusII file.

  Collective

  Input Parameters:
+ viewer - the `PETSCVIEWEREXODUSII` viewer
- order  - elements order

  Output Parameter:

  Level: beginner

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerExodusIIGetId()`, `PetscViewerExodusIIGetOrder()`
@*/
PetscErrorCode PetscViewerExodusIISetOrder(PetscViewer viewer, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerExodusIISetOrder_C", (PetscViewer, PetscInt), (viewer, order));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIGetOrder - Get the elements order in the ExodusII file.

  Collective

  Input Parameters:
+ viewer - the `PETSCVIEWEREXODUSII` viewer
- order  - elements order

  Output Parameter:

  Level: beginner

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerExodusIIGetId()`, `PetscViewerExodusIISetOrder()`
@*/
PetscErrorCode PetscViewerExodusIIGetOrder(PetscViewer viewer, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerExodusIIGetOrder_C", (PetscViewer, PetscInt *), (viewer, order));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerExodusIIOpen - Opens a file for ExodusII input/output.

  Collective

  Input Parameters:
+ comm - MPI communicator
. name - name of file
- type - type of file
.vb
    FILE_MODE_WRITE - create new file for binary output
    FILE_MODE_READ - open existing file for binary input
    FILE_MODE_APPEND - open existing file for binary output
.ve

  Output Parameter:
. exo - `PETSCVIEWEREXODUSII` `PetscViewer` for Exodus II input/output to use with the specified file

  Level: beginner

.seealso: `PETSCVIEWEREXODUSII`, `PetscViewer`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `DMLoad()`, `PetscFileMode`, `PetscViewerSetType()`, `PetscViewerFileSetMode()`, `PetscViewerFileSetName()`
@*/
PetscErrorCode PetscViewerExodusIIOpen(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *exo)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, exo));
  PetscCall(PetscViewerSetType(*exo, PETSCVIEWEREXODUSII));
  PetscCall(PetscViewerFileSetMode(*exo, type));
  PetscCall(PetscViewerFileSetName(*exo, name));
  PetscCall(PetscViewerSetFromOptions(*exo));
  PetscFunctionReturn(PETSC_SUCCESS);
}
