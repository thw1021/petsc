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
static PetscErrorCode EXOGetVarIndex_Internal(PetscViewer viewer, int exoid, ex_entity_type obj_type, const char name[], int *varIndex)
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
    // PetscCall(PetscMalloc1(num_vars,*var_names));
    // PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, num_vars, var_names));
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
  char       *tmpName = NULL;

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
  TO DO: Test the Names variants
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n testing *names functions\n"));
  PetscCall(PetscViewerExodusIISetZonalVariableNames(viewer, zonalVarName));
  PetscCall(PetscViewerExodusIISetNodalVariableNames(viewer, nodalVarName));
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));


  int exoid = -1;
  int idx;
  PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
  PetscCall(EXOGetVarIndex_Internal(viewer,exoid,EX_NODAL,"Alpha",&idx));
  PetscPrintf(PETSC_COMM_WORLD,"   %s idx %d\n", "Alpha",idx);
  PetscCall(EXOGetVarIndex_Internal(viewer,exoid,EX_NODAL,"Beta",&idx));
  PetscPrintf(PETSC_COMM_WORLD,"   %s idx %d\n", "Beta",idx);
  PetscCall(EXOGetVarIndex_Internal(viewer,exoid,EX_NODAL,"Delta",&idx));
  PetscPrintf(PETSC_COMM_WORLD,"   %s idx %d\n", "Delta",idx);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Zonal variables: \n"));
  for (int i = 0; i < nZonalVar; i++){
    PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, i, &tmpName));
    PetscCall(EXOGetVarIndex_Internal(viewer,exoid,EX_ELEM_BLOCK,tmpName,&idx));
    PetscPrintf(PETSC_COMM_WORLD,"   %d: %s location %d\n",i, tmpName,idx);
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Nodal variables: \n"));
  for (int i = 0; i < nNodalVar; i++){
    PetscCall(PetscViewerExodusIIGetNodalVariableName(viewer, i, &tmpName));
    PetscCall(EXOGetVarIndex_Internal(viewer,exoid,EX_NODAL,tmpName,&idx));
    PetscPrintf(PETSC_COMM_WORLD,"   %d: %s location %d\n",i, tmpName,idx);
  }

/*
  varIdx = 0;
  strcpy(varName, "zvar1");
  PetscCall(PetscViewerExodusIISetZonalVariableName(viewer, varIdx, varName));
  PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, varIdx, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of zonal variable %d: %s\n", varIdx, varName));

  varIdx = 1;
  strcpy(varName, "zvar2");
  PetscCall(PetscViewerExodusIISetZonalVariableName(viewer, varIdx, varName));
  PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, varIdx, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of zonal variable %d: %s\n", varIdx, varName));

  varIdx = 2;
  strcpy(varName, "zvar3");
  PetscCall(PetscViewerExodusIISetZonalVariableName(viewer, varIdx, varName));
  PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, varIdx, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of zonal variable %d: %s\n", varIdx, varName));
  */

  /* TESTING NODAL VARIABLE NUMBER & NAME*/
  //nNodalVar = 3; /*TOTAL IN TEST*/
  /*
  PetscCall(PetscViewerExodusIISetNodalVariable(viewer, nNodalVar));
  PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &nNodalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of nodal variables: %d\n", nNodalVar));

  varIdx = 0;
  strcpy(varName, "nvar0");
  PetscCall(PetscViewerExodusIISetNodalVariableName(viewer, varIdx, varName));
  PetscCall(PetscViewerExodusIIGetNodalVariableName(viewer, varIdx, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of nodal variable %d: %s\n", varIdx, varName));

  varIdx = 1;
  strcpy(varName, "nvar1");
  PetscCall(PetscViewerExodusIISetNodalVariableName(viewer, varIdx, varName));
  PetscCall(PetscViewerExodusIIGetNodalVariableName(viewer, varIdx, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of nodal variable %d: %s\n", varIdx, varName));

  varIdx = 2;
  strcpy(varName, "nvar2");
  PetscCall(PetscViewerExodusIISetNodalVariableName(viewer, varIdx, varName));
  PetscCall(PetscViewerExodusIIGetNodalVariableName(viewer, varIdx, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of nodal variable %d: %s\n", varIdx, varName));
  */
  // PetscCall(PetscViewerExodusIISetZonalVariable(viewer, nZonalVar));
  // // PetscCall(PetscViewerExodusIISetZonalVariable(viewer, nZonalVar));

  // PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, &nZonalVar));

  // PetscCall(PetscViewerExodusIISetNodalVariable(viewer, nNodalVar));
  // PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &nNodalVar));

  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of nodal variables: %d\n", nNodalVar));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of zonal variables: %d\n", nZonalVar));

  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nMultiple String Names Test\n"));
  // /* TESTING ZONAL MULTIPLE STRINGS SETTER */
  // const char *test1names[] = {"Name1", "Name2", "Name3"}; // Example names
  // PetscCall(PetscViewerExodusIISetZonalVariableNames(viewer, test1names));

  // /* Allocate memory for namesFromFunction to store the retrieved names */
  // char *namesFromFunction[3];
  // PetscCall(PetscMalloc1(3, &namesFromFunction));
  // for (int i = 0; i < 3; i++) {
  //   PetscCall(PetscMalloc1(256, &namesFromFunction[i])); // Allocate memory for each string
  // }

  // PetscCall(PetscViewerExodusIIGetZonalVariableNames(viewer, 3, namesFromFunction));

  // /* Check if names were correctly retrieved and copied */
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nZonal Variable Names:\n"));
  // for (int i = 0; i < 3; i++) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d: %s\n", i, namesFromFunction[i])); }

  // /* Free allocated memory */
  // for (int i = 0; i < 3; i++) { PetscCall(PetscFree(namesFromFunction[i])); }

  // /* TESTING ZONAL MULTIPLE STRINGS SETTER */
  // PetscCall(PetscViewerExodusIISetNodalVariableNames(viewer, test1names));

  // /* Allocate memory for namesFromFunction to store the retrieved names */
  // char *namesFromFunctionZ[3];
  // PetscCall(PetscMalloc1(3, &namesFromFunctionZ));
  // for (int i = 0; i < 3; i++) {
  //   PetscCall(PetscMalloc1(256, &namesFromFunctionZ[i])); // Allocate memory for each string
  // }

  // PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, 3, namesFromFunctionZ));

  // /* Check if names were correctly retrieved and copied */
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNodal Variable Names:\n"));
  // for (int i = 0; i < 3; i++) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d: %s\n", i, namesFromFunctionZ[i])); }

  // /* Free allocated memory */
  // for (int i = 0; i < 3; i++) { PetscCall(PetscFree(namesFromFunctionZ[i])); }

  // /* NODAL */
  // //const char *test2names[] = {"humidity", "porosity", "saturation"};
  // //PetscCall(PetscViewerExodusIISetNodalVariableNames(viewer, test2names));

  // //PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, 3, namesFromFunction1));
  // /* Check if names were correctly retrieved and copied */
  // //PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNodal Variable Names:\n"));
  // //for (int i = 0; i < 3; i++) {
  // //  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d: %s\n", i, namesFromFunction1[i]));
  // //}
  // //PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of nodal variables: %d\n", nNodalVar));
  // //PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of zonal variables: %d\n", nZonalVar));

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
