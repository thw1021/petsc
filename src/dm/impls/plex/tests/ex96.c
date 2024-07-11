#include "petscdmplex.h"
#include "petscsys.h"
static char help[] = "Test PetscViewer_ExodusII\n\n";

#include <petsc.h>
#include <exodusII.h>
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/
#include <petsc/private/viewerimpl.h>
#include <petsc/private/viewerexodusiiimpl.h>

int main(int argc, char **argv)
{
  DM          dm;
  char        ifilename[PETSC_MAX_PATH_LEN], ofilename[PETSC_MAX_PATH_LEN];
  PetscInt    nNodalVar = 4;
  PetscInt    nZonalVar = 3;
  PetscInt    order     = 1;
  PetscViewer viewer;
  int exoid = -1;
  int index = -1;
  const char *nodalVarName[4] = {"U_x", "U_y", "Alpha", "Beta"};
  const char *zonalVarName[3] = {"Sigma_11", "Sigma_12", "Sigma_22"};
  const char *testNames[3] = {"U", "Alpha", "Gamma"};
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

  /* Testing Variable Number*/
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
    TO DO: write test of PetscViewerExodusIISet[Nodal/Zonal]VariableNames
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
  PetscCall(PetscViewerDestroy(&viewer));   /* Destroyed Viewer */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n\nReopenning the output file in Read-only mode\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Testing PetscViewerExodusIIGet[Nodal/Zonal]VariableNames\n"));
  PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_APPEND, &viewer));
  PetscCall(PetscViewerExodusIISetOrder(viewer, order));
  PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, &numZVars));
  PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &numNVars));

  PetscCall(PetscCalloc1(numZVars,&varNames)); /* Memory must be allocated for the name array before getter function is invoked. */
  PetscCall(PetscViewerExodusIIGetZonalVariableNames(viewer, numZVars, varNames));
  for (int i = 0; i < numZVars; i++){
    PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
    EXOGetVarIndex_Internal(viewer, exoid, EX_ELEM_BLOCK, varNames[i], &index);
    PetscPrintf(PETSC_COMM_WORLD,"   Read zonal variable %d: %s, index in file %d\n",i,varNames[i], index);
  }
  for (int i = 0; i < 3; i++)
  {
    EXOGetVarIndex_Internal(viewer, exoid, EX_ELEM_BLOCK, testNames[i], &index);
    PetscPrintf(PETSC_COMM_WORLD,"   Read zonal variable %d: %s, index in file %d\n",i, testNames[i], index);
  }

  /* Free allocated memory for zonal variable names */
  for (int i = 0; i < numZVars; i++) {
      PetscFree(varNames[i]);
  }
  PetscFree(varNames);

  PetscCall(PetscCalloc1(numNVars,&varNames));
  PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, numNVars, varNames));
  for (int i = 0; i < numNVars; i++){
    PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
    EXOGetVarIndex_Internal(viewer, exoid, EX_NODAL, varNames[i], &index);
    PetscPrintf(PETSC_COMM_WORLD,"   Read nodal variable %d: %s, index in file %d\n",i, varNames[i], index);
  }
  for (int i = 0; i < 3; i++)
  {
    EXOGetVarIndex_Internal(viewer, exoid, EX_NODAL, testNames[i], &index);
    PetscPrintf(PETSC_COMM_WORLD,"   Read nodal variable %d: %s, index in file %d\n",i, testNames[i], index);
  }

  /* Free allocated memory for nodal variable names */
  for (int i = 0; i < numNVars; i++) {
      PetscFree(varNames[i]);
  }
  PetscFree(varNames);

  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: 1
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -o test.exo

TEST*/
