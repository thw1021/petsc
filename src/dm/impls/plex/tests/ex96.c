static char help[] = "Test PetscViewer_ExodusII\n\n";

#include <petsc.h>

int main(int argc, char **argv)
{
  DM             dm;
  char           ifilename[PETSC_MAX_PATH_LEN], ofilename[PETSC_MAX_PATH_LEN];
  int            nNodalVar, nZonalVar, rank;
  char           varName[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PetscViewer_ExodusII test", "ex96");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex96", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscCall(PetscOptionsString("-o", "Filename to write", "ex96", ofilename, ofilename, sizeof(ofilename), NULL));
  PetscOptionsEnd();

  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, NULL, PETSC_TRUE, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "ex96"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));


  PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_WRITE, &viewer));

  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));

  /* ZONAL TESTING */
  nZonalVar = 3;
  PetscCall(PetscViewerExodusIISetZonalVariableNumber(viewer, nZonalVar));
  nZonalVar = 2;
  PetscCall(PetscViewerExodusIIGetZonalVariableNumber(viewer, &nZonalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of zonal variables: %d\n", nZonalVar));

  // Testing PetscViewerExodusIISetZonalVariableName and PetscViewerExodusIIGetZonalVariableName
  rank = 0;
  strcpy(varName, "var1");
  PetscCall(PetscViewerExodusIISetZonalVariableName(viewer, rank, varName));
  PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, rank, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of zonal variable %d: %s\n", rank, varName));

  rank = 1;
  strcpy(varName, "var2");
  PetscCall(PetscViewerExodusIISetZonalVariableName(viewer, rank, varName));
  PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, rank, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of zonal variable %d: %s\n", rank, varName));

  rank = 2;
  strcpy(varName, "var3");
  PetscCall(PetscViewerExodusIISetZonalVariableName(viewer, rank, varName));
  PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, rank, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of zonal variable %d: %s\n", rank, varName));

  /* NODAL TESTING */
  nNodalVar = 2;
  PetscCall(PetscViewerExodusIISetNodalVariableNumber(viewer, nNodalVar));
  nNodalVar = 5;
  PetscCall(PetscViewerExodusIIGetNodalVariableNumber(viewer, &nNodalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of nodal variables: %d\n", nNodalVar));

  rank = 0;
  strcpy(varName, "0nodalvar");
  PetscCall(PetscViewerExodusIISetNodalVariableName(viewer, rank, varName));
  PetscCall(PetscViewerExodusIIGetNodalVariableName(viewer, rank, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of nodal variable %d: %s\n", rank, varName));

  rank = 1;
  strcpy(varName, "1nodalvar");
  PetscCall(PetscViewerExodusIISetNodalVariableName(viewer, rank, varName));
  PetscCall(PetscViewerExodusIIGetNodalVariableName(viewer, rank, varName));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of nodal variable %d: %s\n", rank, varName));

  /* TESTING THE MULTIPLE STRINGS SETTER */
  // Initialize the array with 3 strings
  nZonalVar = 3;
  PetscCall(PetscViewerExodusIISetZonalVariableNumber(viewer, nZonalVar));
  char names[3][PETSC_MAX_PATH_LEN] = {
    "variable1",
    "variable2",
    "variable3"
  };
  /*Testing the Setter*/
  PetscCall(PetscViewerExodusIISetZonalVariableNames(viewer, 3, names));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nMultiple string names\n"));
  for (int i = 0; i < nZonalVar; i++) {
    PetscCall(PetscViewerExodusIIGetZonalVariableName(viewer, i, varName));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Name of zonal variable %d: %s\n", i, varName));
  }
  PetscCall(PetscViewerExodusIIGetZonalVariableNumber(viewer, &nZonalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of zonal variables: %d\n", nZonalVar));
/*Testing the Getter*/
  
  char test2names[3][PETSC_MAX_PATH_LEN] = {
    "pressure",
    "temperature",
    "density"
  }; 
  
  PetscCall(PetscViewerExodusIISetZonalVariableNames(viewer, 3, test2names));

    // Create a buffer to hold zonal variable names
    char namesFromFunction[3][PETSC_MAX_PATH_LEN];

    // Call the function to retrieve zonal variable names
    PetscCall(PetscViewerExodusIIGetZonalVariableNames(viewer, 3, namesFromFunction));

    // Check if names were correctly retrieved and copied
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nZonal Variable Names:\n"));
    for (int i = 0; i < 3; i++) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d: %s\n", i, namesFromFunction[i]));
    }

  
nNodalVar = 5;
PetscCall(PetscViewerExodusIISetNodalVariableNumber(viewer, nNodalVar));

  char test3names[3][PETSC_MAX_PATH_LEN] = 
  {"humidity",
  "porosity",
  "saturation"
  };

  PetscCall(PetscViewerExodusIISetNodalVariableNames(viewer, 3, test3names));
  PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, 3, namesFromFunction));

  // Check if names were correctly retrieved and copied
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNodal Variable Names:\n"));
    for (int i = 0; i < 3; i++) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d: %s\n", i, namesFromFunction[i]));
    }

  PetscCall(PetscViewerExodusIIGetNodalVariableNumber(viewer, &nNodalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of nodal variables: %d\n", nNodalVar));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}


/* TEST
  build:
    requires: !complex
  testset:
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -dm_view
    nsize: 1
    test:
      suffix: 0
      args:
TEST */