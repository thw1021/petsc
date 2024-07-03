#include "petscdmplex.h"
#include "petscsys.h"
static char help[] = "Test PetscViewer_ExodusII\n\n";

#include <petsc.h>
#include <exodusII.h>
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/
#include <petsc/private/viewerimpl.h>
#include <petsc/private/viewerexodusiiimpl.h>

/*
  EXOGetVarIndex_Internal - Locate a result in an exodus file based on its name

  Collective

  Input Parameters:
+ exoid    - the exodus id of a file (obtained from ex_open or ex_create for instance)
. obj_type - the type of entity for instance EX_NODAL, EX_ELEM_BLOCK
- name     - the name of the result

  Output Parameter:
. varIndex - the location in the exodus file of the result

  Level: beginner

  Notes:
  The exodus variable index is obtained by comparing the name argument to the
  names of zonal variables declared in the exodus file. For instance if name is "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all variables of type obj_type.

.seealso: `DMPlexView_ExodusII_Internal()`, `VecViewPlex_ExodusII_Nodal_Internal()`, `VecLoadNodal_PlexEXO()`, `VecLoadZonal_PlexEXO()`
*/
static PetscErrorCode EXOGetVarIndex_Internal2(PetscViewer viewer, int exoid, ex_entity_type obj_type, const char name[], int *varIndex)
{
  int       num_vars = 0, i, j;
  char      ext_name[MAX_STR_LENGTH + 1];
  char     *var_name = NULL; /* previously char var_name[MAX_STR_LENGTH + 1]; */
  char     **var_names;
  const int num_suffix = 5;
  char     *suffix[5];
  PetscBool flg;

  PetscFunctionBegin;
  suffix[0] = (char *)"";
  suffix[1] = (char *)"_X";
  suffix[2] = (char *)"_XX";
  suffix[3] = (char *)"_1";
  suffix[4] = (char *)"_11";
  *varIndex = -1;

  /* Get Variable Number from file - replacing PetscCallExternal(ex_get_variable_param, exoid, obj_type, &num_vars); */
  if (obj_type == EX_NODAL) {
    PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &num_vars));
    // This does not work and I am not sure I understand why
    // It would be OK if var_names was declared as char     *var_names[10]
    // but we can't do this...
    PetscCall(PetscCalloc1(num_vars,&var_names));
    PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, num_vars, var_names));
    for (i = 0; i < num_vars; ++i) {
      PetscCall(PetscViewerExodusIIGetNodalVariableName(viewer, i, &var_name));
      for (j = 0; j < num_suffix; ++j) {
        PetscCall(PetscStrncpy(ext_name, name, MAX_STR_LENGTH));
        PetscCall(PetscStrlcat(ext_name, suffix[j], MAX_STR_LENGTH));
        PetscCall(PetscStrcasecmp(ext_name, var_name, &flg));
        if (flg) *varIndex = i;
      }
      if (flg) break;
    }
  } else if (obj_type == EX_ELEM_BLOCK) {
    PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, &num_vars));
    for (i = 0; i < num_vars; ++i) {
      PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, i, &var_name));
      for (j = 0; j < num_suffix; ++j) {
        PetscCall(PetscStrncpy(ext_name, name, MAX_STR_LENGTH));
        PetscCall(PetscStrlcat(ext_name, suffix[j], MAX_STR_LENGTH));
        PetscCall(PetscStrcasecmp(ext_name, var_name, &flg));
        if (flg) *varIndex = i;
      }
      if (flg) break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **argv)
{
  DM          dm;
  char        ifilename[PETSC_MAX_PATH_LEN], ofilename[PETSC_MAX_PATH_LEN];
  PetscInt    nNodalVar = 4;
  PetscInt    nZonalVar = 3;
  PetscInt    order     = 1;
  PetscViewer viewer;
  const char *nodalVarName[4] = {"U_x", "U_y", "Alpha", "Beta"};
  const char *zonalVarName[3] = {"Sigma_11", "Sigma_12", "Sigma_22"};
  char      **varNames;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PetscViewer_ExodusII test", "ex96");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex96", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscCall(PetscOptionsString("-o", "Filename to write", "ex96", ofilename, ofilename, sizeof(ofilename), NULL));
  PetscOptionsEnd();

#ifdef PETSC_USE_DEBUG
  PetscCallExternal(ex_opts,EX_VERBOSE + EX_DEBUG);
#endif

  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, NULL, PETSC_TRUE, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "ex96"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_WRITE, &viewer));
  
  // PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));

  /* Save the geometry to the file, erasing all previous content */
  PetscCall(PetscViewerExodusIISetOrder(viewer, order));
  PetscCall(DMView(dm, viewer));
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerFlush(viewer));

  /* TESTING VARIABLE NUMBER */
  PetscCall(PetscViewerExodusIISetZonalVariable(viewer, nZonalVar));
  nZonalVar = -1;
  PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, &nZonalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of zonal variables: %d\n", nZonalVar));

  PetscCall(PetscViewerExodusIISetNodalVariable(viewer, nNodalVar));
  nNodalVar = -1;
  PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &nNodalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of nodal variables: %d\n", nNodalVar));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nAfter PetscViewerExodusIISet[Nodal/Zonal]Variable calls: \n"));
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));

  for (int i = 0; i < nNodalVar; i++){
    PetscCall(PetscViewerExodusIISetNodalVariableName(viewer, i, nodalVarName[i]));
  }
  for (int i = 0; i < nZonalVar; i++){
    PetscCall(PetscViewerExodusIISetZonalVariableName(viewer, i, zonalVarName[i]));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nAfter PetscViewerExodusIISet[Nodal/Zonal]VariableName calls: \n"));
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));

  /*
    TO DO: write test of PetscViewerExodusIISef[Nodal/Zonal]VariableNames
  */

  /* 
    Test of PetscViewerExodusIIGet[Nodal/Zonal]VariableName
  */
  char *name;
  int   numZVars,numNVars;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n testing *names setters\n"));
  PetscCall(PetscViewerExodusIISetZonalVariableNames(viewer, zonalVarName));
  PetscCall(PetscViewerExodusIISetNodalVariableNames(viewer, nodalVarName));
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerDestroy(&viewer));

  /* 
    Test of PetscViewerExodusIIGet[Nodal/Zonal]VariableName
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n\nReopenning the output file in Read-only mode\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Testing PetscViewerExodusIIGet[Nodal/Zonal]VariableName\n"));
  PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_APPEND, &viewer));
  PetscCall(PetscViewerExodusIISetOrder(viewer, order));
  PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, &numZVars));
  PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &numNVars));

  for (int i = 0; i < numZVars; i++){
    PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer,i,&name));
    PetscPrintf(PETSC_COMM_WORLD,"   Read zonal variable %d: %s\n",i,name);
  }
  for (int i = 0; i < numNVars; i++){
    PetscCall(PetscViewerExodusIIGetNodalVariableName(viewer,i,&name));
    PetscPrintf(PETSC_COMM_WORLD,"   Read nodal variable %d: %s\n",i,name);
  }
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n\nReopenning the output file in Read-only mode\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Testing PetscViewerExodusIIGet[Nodal/Zonal]VariableNames\n"));
  PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_APPEND, &viewer));
  PetscCall(PetscViewerExodusIISetOrder(viewer, order));
  PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, &numZVars));
  PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &numNVars));

  PetscCall(PetscCalloc1(numZVars,&varNames));
  PetscCall(PetscViewerExodusIIGetZonalVariableNames(viewer, numZVars, varNames));
  for (int i = 0; i < numZVars; i++){
    PetscPrintf(PETSC_COMM_WORLD,"   Read zonal variable %d: %s\n",i,varNames[i]);
  }
  PetscFree(varNames);
  PetscCall(PetscCalloc1(numNVars,&varNames));
  PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, numNVars, varNames));
  for (int i = 0; i < numNVars; i++){
    PetscPrintf(PETSC_COMM_WORLD,"   Read nodal variable %d: %s\n",i,varNames[i]);
  }
  PetscFree(varNames);

  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/* TEST
  build:
    requires: !complex
  testset:
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -o test.exo -dm_view
    nsize: 1
    test:
      suffix: 0
      args:
TEST */
