static char help[] = "Test PetscViewer_ExodusII\n\n";

#include <petsc.h>

int main(int argc, char **argv)
{
  DM             dm;
  char           ifilename[PETSC_MAX_PATH_LEN],ofilename[PETSC_MAX_PATH_LEN];
  int            nNodalVar,nZonalVar;
  char           varName[PETSC_MAX_PATH_LEN];
  PetscViewer   viewer;


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

  nZonalVar = 3;
  PetscCall(PetscViewerExodusIISetZonalVariableNumber(viewer,nZonalVar));
  nZonalVar = -1;
  PetscCall(PetscViewerExodusIIGetZonalVariableNumber(viewer,&nZonalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of zonal variable: %d\n",nZonalVar));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));


  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: !complex
  testset:
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -dm_view
    nsize: 1
    test:
      suffix: 0
      args:
TEST*/
