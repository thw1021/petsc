static char help[] = "Test PetscViewer_ExodusII\n\n";

#include <petsc.h>
#include <exodusII.h>

int main(int argc, char **argv)
{
  DM   dm;
  char ifilename[PETSC_MAX_PATH_LEN], ofilename[PETSC_MAX_PATH_LEN];
  int  nNodalVar = 3;
  int  nZonalVar = 3;
  /*varIdx*/
  PetscViewer viewer;

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

  /* TESTING ZONAL VARIABLE NUMBER & NAME */
  //nZonalVar = 3; /*TOTAL IN TEST*/
  /*
  PetscCall(PetscViewerExodusIISetZonalVariableNumber(viewer, nZonalVar));
  nZonalVar = 2;
  PetscCall(PetscViewerExodusIIGetZonalVariableNumber(viewer, &nZonalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of zonal variables: %d\n", nZonalVar));

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
  PetscCall(PetscViewerExodusIISetNodalVariableNumber(viewer, nNodalVar));
  PetscCall(PetscViewerExodusIIGetNodalVariableNumber(viewer, &nNodalVar));
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
  PetscCall(PetscViewerExodusIISetZonalVariableNumber(viewer, nZonalVar));
  PetscCall(PetscViewerExodusIIGetZonalVariableNumber(viewer, &nZonalVar));

  PetscCall(PetscViewerExodusIISetNodalVariableNumber(viewer, nNodalVar));
  PetscCall(PetscViewerExodusIIGetNodalVariableNumber(viewer, &nNodalVar));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of nodal variables: %d\n", nNodalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of zonal variables: %d\n", nZonalVar));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nMultiple String Names Test\n"));
    /* TESTING ZONAL MULTIPLE STRINGS SETTER */
    const char *test1names[] = {"Name1", "Name2", "Name3"}; // Example names
    PetscCall(PetscViewerExodusIISetZonalVariableNames(viewer, test1names));

    /* Allocate memory for namesFromFunction to store the retrieved names */
    char *namesFromFunction[3];
    PetscCall(PetscMalloc1(3, &namesFromFunction));
    for (int i = 0; i < 3; i++) {
        PetscCall(PetscMalloc1(256, &namesFromFunction[i])); // Allocate memory for each string
    }

    PetscCall(PetscViewerExodusIIGetZonalVariableNames(viewer, 3, namesFromFunction));

    /* Check if names were correctly retrieved and copied */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nZonal Variable Names:\n"));
    for (int i = 0; i < 3; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d: %s\n", i, namesFromFunction[i]));
    }

    /* Free allocated memory */
    for (int i = 0; i < 3; i++) {
        PetscCall(PetscFree(namesFromFunction[i]));
    }

/* TESTING ZONAL MULTIPLE STRINGS SETTER */
PetscCall(PetscViewerExodusIISetNodalVariableNames(viewer, test1names));

    /* Allocate memory for namesFromFunction to store the retrieved names */
    char *namesFromFunctionZ[3];
    PetscCall(PetscMalloc1(3, &namesFromFunctionZ));
    for (int i = 0; i < 3; i++) {
        PetscCall(PetscMalloc1(256, &namesFromFunctionZ[i])); // Allocate memory for each string
    }

    PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, 3, namesFromFunctionZ));

    /* Check if names were correctly retrieved and copied */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNodal Variable Names:\n"));
    for (int i = 0; i < 3; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d: %s\n", i, namesFromFunctionZ[i]));
    }

    /* Free allocated memory */
    for (int i = 0; i < 3; i++) {
        PetscCall(PetscFree(namesFromFunctionZ[i]));
    }

/* NODAL */
//const char *test2names[] = {"humidity", "porosity", "saturation"};
//PetscCall(PetscViewerExodusIISetNodalVariableNames(viewer, test2names));

//PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, 3, namesFromFunction1));
/* Check if names were correctly retrieved and copied */
//PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNodal Variable Names:\n"));
//for (int i = 0; i < 3; i++) {
  //  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d: %s\n", i, namesFromFunction1[i]));
//}
//PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of nodal variables: %d\n", nNodalVar));
//PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNumber of zonal variables: %d\n", nZonalVar));

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
