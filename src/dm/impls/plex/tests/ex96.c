static char help[] = "Test DMPlexVecGetClosure / DMPlexVecRestoreClosure\n\n";

#include <petsc.h>

int main(int argc, char **argv) {
  DM           dm, pdm;
  PetscSection section;
  char         ifilename[PETSC_MAX_PATH_LEN];
  PetscInt     pStart, pEnd;
  PetscScalar *cval;
  PetscInt     clSize;
  Vec          v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FEM Layout Options", "ex96");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex96", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscOptionsEnd();

  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, NULL, PETSC_TRUE, &dm));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));

  PetscCall(DMPlexDistribute(dm, 0, NULL, &pdm));
  if (pdm) {
    PetscCall(DMDestroy(&dm));
    dm = pdm;
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* create a section */
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));

  /* initialize the section storage with a single value at point 0 */
  PetscCall(PetscSectionSetDof(section, pStart, 1));

  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscObjectViewFromOptions((PetscObject)section, NULL, "-dm_section_view"));

  PetscCall(DMGetLocalVector(dm, &v));
  PetscCall(VecSet(v, -1.0));
  PetscCall(VecViewFromOptions(v, NULL, "-dm_vec_view"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Point %" PetscInt_FMT "\n", pStart));
  cval = NULL;
  PetscCall(DMPlexVecGetClosure(dm, section, v, pStart, &clSize, &cval));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "clSize %" PetscInt_FMT "\n", clSize));
  PetscCall(PetscRealView(clSize, cval, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(DMPlexVecRestoreClosure(dm, section, v, pStart, &clSize, &cval));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Point %" PetscInt_FMT "\n", pStart + 1));
  cval = NULL;
  PetscCall(DMPlexVecGetClosure(dm, section, v, pStart + 1, &clSize, &cval));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "clSize %" PetscInt_FMT "\n", clSize));
  if (clSize > 0) PetscCall(PetscRealView(clSize, cval, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(DMPlexVecRestoreClosure(dm, section, v, pStart + 1, &clSize, &cval));

  PetscCall(PetscSectionDestroy(&section));
  PetscCall(DMRestoreLocalVector(dm, &v));
  PetscCall(DMDestroy(&dm));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(VecSetSizes(v, PETSC_DECIDE, 0));
  PetscCall(PetscObjectSetName((PetscObject)v, "U"));
  PetscCall(VecSetFromOptions(v));
  PetscCall(VecViewFromOptions(v, NULL, "-dm_vec_view"));
  PetscCall(VecGetSize(v, &clSize));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Vec size: %" PetscInt_FMT "\n", clSize));
  PetscCall(VecGetArray(v, &cval));
  if (clSize > 0) PetscCall(PetscRealView(clSize, cval, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(VecRestoreArray(v, &cval));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: exodusii pnetcdf !complex
  testset:
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/SquareFaceSet.exo -dm_view -dm_section_view -dm_vec_view
    nsize: 1

    test:
      suffix: 0
      args:

TEST*/
