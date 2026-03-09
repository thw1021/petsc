#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

#include <netcdf.h>
#include <exodusII.h>

#include <petsc/private/viewerimpl.h>
#include <petsc/private/viewerexodusiiimpl.h>

static PetscErrorCode PetscViewerView_ExodusII(PetscViewer v, PetscViewer viewer)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)v->data;

  PetscFunctionBegin;
  if (exo->filename) PetscCall(PetscViewerASCIIPrintf(viewer, "Filename:    %s\n", exo->filename));
  if (exo->exoid) PetscCall(PetscViewerASCIIPrintf(viewer, "exoid:       %" PetscExodusIIInt_FMT "\n", exo->exoid));
  if (exo->btype) PetscCall(PetscViewerASCIIPrintf(viewer, "IO Mode:     %d\n", exo->btype));
  if (exo->order) PetscCall(PetscViewerASCIIPrintf(viewer, "Mesh order:  %" PetscInt_FMT "\n", exo->order));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Number of nodal variables:  %" PetscExodusIIInt_FMT "\n", exo->numNodalVariables));
  for (int i = 0; i < exo->numNodalVariables; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "   %d: %s\n", i, exo->nodalVariableNames[i]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Number of zonal variables:  %" PetscExodusIIInt_FMT "\n", exo->numZonalVariables));
  for (int i = 0; i < exo->numZonalVariables; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "   %d: %s\n", i, exo->zonalVariableNames[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFlush_ExodusII(PetscViewer v)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)v->data;

  PetscFunctionBegin;
  if (exo->exoid >= 0) PetscCallExternal(ex_update, exo->exoid);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerSetFromOptions_ExodusII(PetscViewer v, PetscOptionItems PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "ExodusII PetscViewer Options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerDestroy_ExodusII(PetscViewer viewer)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  if (exo->exoid >= 0) PetscCallExternal(ex_close, exo->exoid);
  for (PetscInt i = 0; i < exo->numZonalVariables; i++) PetscCall(PetscFree(exo->zonalVariableNames[i]));
  PetscCall(PetscFree(exo->zonalVariableNames));
  for (PetscInt i = 0; i < exo->numNodalVariables; i++) PetscCall(PetscFree(exo->nodalVariableNames[i]));
  PetscCall(PetscFree(exo->nodalVariableNames));
  PetscCall(PetscFree(exo->filename));
  PetscCall(PetscFree(exo));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetId_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetOrder_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIISetOrder_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIISetZonalVariable_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIISetNodalVariable_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetZonalVariable_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIISetNodalVariable_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIISetZonalVariableName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIISetNodalVariableName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetZonalVariableName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetNodalVariableName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIISetZonalVariableNames_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIISetNodalVariableNames_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetZonalVariableNames_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetNodalVariableNames_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetNodalVariableIndex_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerExodusIIGetZonalVariableIndex_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetName_ExodusII(PetscViewer viewer, const char name[])
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;
  PetscMPIInt           rank;
  PetscExodusIIInt      CPU_word_size, IO_word_size, EXO_mode;
  MPI_Info              mpi_info = MPI_INFO_NULL;
  PetscExodusIIFloat    EXO_version;

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);

  PetscFunctionBegin;
  if (exo->exoid >= 0) {
    PetscCallExternal(ex_close, exo->exoid);
    exo->exoid = -1;
  }
  if (exo->filename) PetscCall(PetscFree(exo->filename));
  PetscCall(PetscStrallocpy(name, &exo->filename));
  switch (exo->btype) {
  case FILE_MODE_READ:
    EXO_mode = EX_READ;
    break;
  case FILE_MODE_APPEND:
  case FILE_MODE_UPDATE:
  case FILE_MODE_APPEND_UPDATE:
    /* Will fail if the file does not already exist */
    EXO_mode = EX_WRITE;
    break;
  case FILE_MODE_WRITE:
    /*
      exodus only allows writing geometry upon file creation, so we will let DMView create the file.
    */
    PetscFunctionReturn(PETSC_SUCCESS);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  }
#if defined(PETSC_USE_64BIT_INDICES)
  EXO_mode += EX_ALL_INT64_API;
#endif
  exo->exoid = ex_open_par(name, EXO_mode, &CPU_word_size, &IO_word_size, &EXO_version, PETSC_COMM_WORLD, mpi_info);
  PetscCheck(exo->exoid >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open_par failed for %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetName_ExodusII(PetscViewer viewer, const char *name[])
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  *name = exo->filename;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetMode_ExodusII(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  exo->btype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetMode_ExodusII(PetscViewer viewer, PetscFileMode *type)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  *type = exo->btype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerExodusIIGetId_ExodusII(PetscViewer viewer, PetscExodusIIInt *exoid)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  *exoid = exo->exoid;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerExodusIIGetOrder_ExodusII(PetscViewer viewer, PetscInt *order)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  *order = exo->order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerExodusIISetOrder_ExodusII(PetscViewer viewer, PetscInt order)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  exo->order = order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIISetZonalVariable_ExodusII(PetscViewer viewer, PetscExodusIIInt num)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;
  MPI_Comm              comm;
  PetscExodusIIInt      exoid = -1;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCheck(exo->numZonalVariables == -1, comm, PETSC_ERR_SUP, "The number of zonal variables has already been set to %d and cannot be overwritten", exo->numZonalVariables);
  PetscCheck((exo->btype != FILE_MODE_READ) && (exo->btype != FILE_MODE_UNDEFINED), comm, PETSC_ERR_FILE_WRITE, "Cannot set the number of variables because the file is not writable");

  exo->numZonalVariables = num;
  PetscCall(PetscMalloc1(num, &exo->zonalVariableNames));
  for (int i = 0; i < num; i++) exo->zonalVariableNames[i] = NULL;
  PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
  PetscCallExternal(ex_put_variable_param, exoid, EX_ELEM_BLOCK, num);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIISetNodalVariable_ExodusII(PetscViewer viewer, PetscExodusIIInt num)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;
  MPI_Comm              comm;
  PetscExodusIIInt      exoid = -1;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCheck(exo->numNodalVariables == -1, comm, PETSC_ERR_SUP, "The number of nodal variables has already been set to %d and cannot be overwritten", exo->numNodalVariables);
  PetscCheck((exo->btype != FILE_MODE_READ) && (exo->btype != FILE_MODE_UNDEFINED), comm, PETSC_ERR_FILE_WRITE, "Cannot set the number of variables because the file is not writable");

  exo->numNodalVariables = num;
  PetscCall(PetscMalloc1(num, &exo->nodalVariableNames));
  for (int i = 0; i < num; i++) exo->nodalVariableNames[i] = NULL;
  PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
  PetscCallExternal(ex_put_variable_param, exoid, EX_NODAL, num);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIIGetZonalVariable_ExodusII(PetscViewer viewer, PetscExodusIIInt *num)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;
  MPI_Comm              comm;
  PetscExodusIIInt      exoid = -1;

  PetscFunctionBegin;
  if (exo->numZonalVariables > -1) {
    *num = exo->numZonalVariables;
  } else {
    PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
    PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
    PetscCheck(exoid > 0, comm, PETSC_ERR_FILE_OPEN, "Exodus file is not open");
    PetscCallExternal(ex_get_variable_param, exoid, EX_ELEM_BLOCK, num);
    exo->numZonalVariables = *num;
    PetscCall(PetscMalloc1(*num, &exo->zonalVariableNames));
    for (int i = 0; i < *num; i++) exo->zonalVariableNames[i] = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIIGetNodalVariable_ExodusII(PetscViewer viewer, PetscExodusIIInt *num)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;
  MPI_Comm              comm;
  PetscExodusIIInt      exoid = -1;

  PetscFunctionBegin;
  if (exo->numNodalVariables > -1) {
    *num = exo->numNodalVariables;
  } else {
    PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
    PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
    PetscCheck(exoid > 0, comm, PETSC_ERR_FILE_OPEN, "Exodus file is not open");
    PetscCallExternal(ex_get_variable_param, exoid, EX_NODAL, num);
    exo->numNodalVariables = *num;
    PetscCall(PetscMalloc1(*num, &exo->nodalVariableNames));
    for (int i = 0; i < *num; i++) exo->nodalVariableNames[i] = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIISetZonalVariableName_ExodusII(PetscViewer viewer, PetscExodusIIInt idx, const char name[])
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  PetscCheck((idx >= 0) && (idx < exo->numZonalVariables), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Variable index out of range. Was PetscViewerExodusIISetZonalVariable called?");
  PetscCall(PetscStrallocpy(name, (char **)&exo->zonalVariableNames[idx]));
  PetscCallExternal(ex_put_variable_name, exo->exoid, EX_ELEM_BLOCK, idx + 1, name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIISetNodalVariableName_ExodusII(PetscViewer viewer, PetscExodusIIInt idx, const char name[])
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  PetscCheck((idx >= 0) && (idx < exo->numNodalVariables), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Variable index out of range. Was PetscViewerExodusIISetNodalVariable called?");
  PetscCall(PetscStrallocpy(name, (char **)&exo->nodalVariableNames[idx]));
  PetscCallExternal(ex_put_variable_name, exo->exoid, EX_NODAL, idx + 1, name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIIGetZonalVariableName_ExodusII(PetscViewer viewer, PetscExodusIIInt idx, const char *name[])
{
  PetscViewer_ExodusII *exo   = (PetscViewer_ExodusII *)viewer->data;
  PetscExodusIIInt      exoid = -1;
  char                  tmpName[MAX_NAME_LENGTH + 1];

  PetscFunctionBegin;
  PetscCheck(idx >= 0 && idx < exo->numZonalVariables, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Variable index out of range. Was PetscViewerExodusIISetZonalVariable called?");
  if (!exo->zonalVariableNames[idx]) {
    PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
    PetscCallExternal(ex_get_variable_name, exoid, EX_ELEM_BLOCK, idx + 1, tmpName);
    PetscCall(PetscStrallocpy(tmpName, (char **)&exo->zonalVariableNames[idx]));
  }
  *name = exo->zonalVariableNames[idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIIGetNodalVariableName_ExodusII(PetscViewer viewer, PetscExodusIIInt idx, const char *name[])
{
  PetscViewer_ExodusII *exo   = (PetscViewer_ExodusII *)viewer->data;
  PetscExodusIIInt      exoid = -1;
  char                  tmpName[MAX_NAME_LENGTH + 1];

  PetscFunctionBegin;
  PetscCheck((idx >= 0) && (idx < exo->numNodalVariables), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Variable index out of range. Was PetscViewerExodusIISetNodalVariable called?");
  if (!exo->nodalVariableNames[idx]) {
    PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
    PetscCallExternal(ex_get_variable_name, exoid, EX_NODAL, idx + 1, tmpName);
    PetscCall(PetscStrallocpy(tmpName, (char **)&exo->nodalVariableNames[idx]));
  }
  *name = exo->nodalVariableNames[idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIISetZonalVariableNames_ExodusII(PetscViewer viewer, const char *const names[])
{
  PetscExodusIIInt      numNames;
  PetscExodusIIInt      exoid = -1;
  PetscViewer_ExodusII *exo   = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, &numNames));
  PetscCheck(numNames >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of zonal variables not set. Was PetscViewerExodusIISetZonalVariable called?");

  PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
  for (int i = 0; i < numNames; i++) {
    PetscCall(PetscStrallocpy(names[i], &exo->zonalVariableNames[i]));
    PetscCallExternal(ex_put_variable_name, exoid, EX_ELEM_BLOCK, i + 1, names[i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIISetNodalVariableNames_ExodusII(PetscViewer viewer, const char *const names[])
{
  PetscExodusIIInt      numNames;
  PetscExodusIIInt      exoid = -1;
  PetscViewer_ExodusII *exo   = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &numNames));
  PetscCheck(numNames >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of nodal variables not set. Was PetscViewerExodusIISetNodalVariable called?");

  PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
  for (int i = 0; i < numNames; i++) {
    PetscCall(PetscStrallocpy(names[i], &exo->nodalVariableNames[i]));
    PetscCallExternal(ex_put_variable_name, exoid, EX_NODAL, i + 1, names[i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIIGetZonalVariableNames_ExodusII(PetscViewer viewer, PetscExodusIIInt *numVars, const char *const *varNames[])
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;
  PetscExodusIIInt      idx;
  char                  tmpName[MAX_NAME_LENGTH + 1];
  PetscExodusIIInt      exoid = -1;

  PetscFunctionBegin;
  PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, numVars));
  /*
    Cache variable names if necessary
  */
  for (idx = 0; idx < *numVars; idx++) {
    if (!exo->zonalVariableNames[idx]) {
      PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
      PetscCallExternal(ex_get_variable_name, exoid, EX_ELEM_BLOCK, idx + 1, tmpName);
      PetscCall(PetscStrallocpy(tmpName, (char **)&exo->zonalVariableNames[idx]));
    }
  }
  *varNames = (const char *const *)exo->zonalVariableNames;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIIGetNodalVariableNames_ExodusII(PetscViewer viewer, PetscExodusIIInt *numVars, const char *const *varNames[])
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;
  PetscExodusIIInt      idx;
  char                  tmpName[MAX_NAME_LENGTH + 1];
  PetscExodusIIInt      exoid = -1;

  PetscFunctionBegin;
  PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, numVars));
  /*
    Cache variable names if necessary
  */
  for (idx = 0; idx < *numVars; idx++) {
    if (!exo->nodalVariableNames[idx]) {
      PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
      PetscCallExternal(ex_get_variable_name, exoid, EX_NODAL, idx + 1, tmpName);
      PetscCall(PetscStrallocpy(tmpName, (char **)&exo->nodalVariableNames[idx]));
    }
  }
  *varNames = (const char *const *)exo->nodalVariableNames;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIIGetNodalVariableIndex_ExodusII(PetscViewer viewer, const char name[], PetscExodusIIInt *varIndex)
{
  PetscExodusIIInt   num_vars = 0, i, j;
  char               ext_name[MAX_STR_LENGTH + 1];
  const char *const *var_names;
  const int          num_suffix = 5;
  char              *suffix[5];
  PetscBool          flg;

  PetscFunctionBegin;
  suffix[0] = (char *)"";
  suffix[1] = (char *)"_X";
  suffix[2] = (char *)"_XX";
  suffix[3] = (char *)"_1";
  suffix[4] = (char *)"_11";
  *varIndex = -1;

  PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, &num_vars, &var_names));
  for (i = 0; i < num_vars; ++i) {
    for (j = 0; j < num_suffix; ++j) {
      PetscCall(PetscStrncpy(ext_name, name, MAX_STR_LENGTH));
      PetscCall(PetscStrlcat(ext_name, suffix[j], MAX_STR_LENGTH));
      PetscCall(PetscStrcasecmp(ext_name, var_names[i], &flg));
      if (flg) *varIndex = i;
    }
    if (flg) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerExodusIIGetZonalVariableIndex_ExodusII(PetscViewer viewer, const char name[], int *varIndex)
{
  PetscExodusIIInt   num_vars = 0, i, j;
  char               ext_name[MAX_STR_LENGTH + 1];
  const char *const *var_names;
  const int          num_suffix = 5;
  char              *suffix[5];
  PetscBool          flg;

  PetscFunctionBegin;
  suffix[0] = (char *)"";
  suffix[1] = (char *)"_X";
  suffix[2] = (char *)"_XX";
  suffix[3] = (char *)"_1";
  suffix[4] = (char *)"_11";
  *varIndex = -1;

  PetscCall(PetscViewerExodusIIGetZonalVariableNames(viewer, &num_vars, &var_names));
  for (i = 0; i < num_vars; ++i) {
    for (j = 0; j < num_suffix; ++j) {
      PetscCall(PetscStrncpy(ext_name, name, MAX_STR_LENGTH));
      PetscCall(PetscStrlcat(ext_name, suffix[j], MAX_STR_LENGTH));
      PetscCall(PetscStrcasecmp(ext_name, var_names[i], &flg));
      if (flg) *varIndex = i;
    }
    if (flg) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCVIEWEREXODUSII - A viewer that writes to an Exodus II file

  Level: beginner

.seealso: `PetscViewerExodusIIOpen()`, `PetscViewerCreate()`, `PETSCVIEWERBINARY`, `PETSCVIEWERHDF5`, `DMView()`,
          `PetscViewerFileSetName()`, `PetscViewerFileSetMode()`, `PetscViewerFormat`, `PetscViewerType`, `PetscViewerSetType()`
M*/
PETSC_EXTERN PetscErrorCode PetscViewerCreate_ExodusII(PetscViewer v)
{
  PetscViewer_ExodusII *exo;

  PetscFunctionBegin;
  PetscCall(PetscNew(&exo));

  v->data                 = (void *)exo;
  v->ops->destroy         = PetscViewerDestroy_ExodusII;
  v->ops->setfromoptions  = PetscViewerSetFromOptions_ExodusII;
  v->ops->view            = PetscViewerView_ExodusII;
  v->ops->flush           = PetscViewerFlush_ExodusII;
  exo->btype              = FILE_MODE_UNDEFINED;
  exo->filename           = 0;
  exo->exoid              = -1;
  exo->numNodalVariables  = -1;
  exo->numZonalVariables  = -1;
  exo->nodalVariableNames = NULL;
  exo->zonalVariableNames = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetName_C", PetscViewerFileSetName_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetName_C", PetscViewerFileGetName_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetMode_C", PetscViewerFileSetMode_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetMode_C", PetscViewerFileGetMode_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetId_C", PetscViewerExodusIIGetId_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIISetOrder_C", PetscViewerExodusIISetOrder_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetOrder_C", PetscViewerExodusIIGetOrder_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIISetZonalVariable_C", PetscViewerExodusIISetZonalVariable_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIISetNodalVariable_C", PetscViewerExodusIISetNodalVariable_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetZonalVariable_C", PetscViewerExodusIIGetZonalVariable_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIISetNodalVariable_C", PetscViewerExodusIISetNodalVariable_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIISetZonalVariableName_C", PetscViewerExodusIISetZonalVariableName_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIISetNodalVariableName_C", PetscViewerExodusIISetNodalVariableName_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetZonalVariableName_C", PetscViewerExodusIIGetZonalVariableName_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetNodalVariableName_C", PetscViewerExodusIIGetNodalVariableName_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIISetZonalVariableNames_C", PetscViewerExodusIISetZonalVariableNames_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIISetNodalVariableNames_C", PetscViewerExodusIISetNodalVariableNames_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetZonalVariableNames_C", PetscViewerExodusIIGetZonalVariableNames_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetNodalVariableNames_C", PetscViewerExodusIIGetNodalVariableNames_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetNodalVariableIndex_C", PetscViewerExodusIIGetNodalVariableIndex_ExodusII));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerExodusIIGetZonalVariableIndex_C", PetscViewerExodusIIGetZonalVariableIndex_ExodusII));
  PetscFunctionReturn(PETSC_SUCCESS);
}
