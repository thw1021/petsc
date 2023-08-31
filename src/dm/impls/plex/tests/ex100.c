static char help[] = "Tests DMLoadFromFile() on a CGNS file\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM          dm;
  char        filename[256];
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", (char *)&filename, sizeof(filename), NULL));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMLoadFromFile(dm, filename));
  PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERCGNS));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer, filename));
  PetscCall(DMLoad(dm, viewer));
  PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: cgns
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/tut21.cgns

TEST*/
