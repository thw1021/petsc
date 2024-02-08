static char help[] = "Create a 2D DMBF object\n";

#include <petsc.h>
#include <petscdmbf.h>
#include <petscdmforest.h>

enum {
  CELLDATA_X,
  CELLDATA_Y,
  CELLDATA_U,
  CELLDATA_N_
};

#define CELLDATA_D_ 2

static const PetscInt CELLDATA_SHAPE[CELLDATA_N_ * CELLDATA_D_] = {
  /* X */ 16, 1,
  /* Y */ 16, 1,
  /* U */ 16, 16};

int main(int argc, char **argv)
{
  const char     _name[] = "2D-DMBF";
  DM             dm;
  DMType         dmtype;
  Vec            v;
  Mat            A;
  PetscErrorCode init_ierr;

  // initialize Petsc
  PetscFunctionBeginUser;
  init_ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (PETSC_SUCCESS != init_ierr) { return init_ierr; }

  // begin main
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Begin\n", _name));

  // create DM
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Create DM\n", _name));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, "bf"));
  // set DM options
  PetscCall(DMSetDimension(dm, 2));
  PetscCall(DMSetFromOptions(dm));
  // print DM type
  PetscCall(DMGetType(dm, &dmtype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] DM type = %s\n", _name, dmtype));

  // set cell data shapes
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Set cell data shape\n", _name));
  PetscCall(DMBFSetCellDataShape(dm, CELLDATA_SHAPE, CELLDATA_N_, CELLDATA_D_));
  // check cell data shapes
  {
    PetscInt *shapeElements = PETSC_NULLPTR;
    PetscInt  n, d, i, j;

    PetscCall(DMBFGetCellDataShape(dm, &shapeElements, &n, &d));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Shape elements dim=(%" PetscInt_FMT ",%" PetscInt_FMT ")\n", _name, n, d));
    for (i = 0; i < n; i++) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s]   %" PetscInt_FMT ": ", _name, i));
      for (j = 0; j < d; j++) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " ", shapeElements[i * d + j])); }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    }
    if (shapeElements) { PetscCall(PetscFree(shapeElements)); }
  }

  // setup DM
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Set up DM\n", _name));
  PetscCall(DMSetUp(dm));

  // print info
  {
    PetscInt dim, n, N, ng;

    PetscCall(DMBFGetInfo(dm, &dim, &n, &N, &ng));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Info about the DM\n", _name));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] - dimension              = %" PetscInt_FMT "\n", _name, dim));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] - number of local cells  = %" PetscInt_FMT "\n", _name, n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] - number of global cells = %" PetscInt_FMT "\n", _name, N));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] - number of ghost cells  = %" PetscInt_FMT "\n", _name, ng));
  }

  // create derived objects
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Create vectors\n", _name));
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Create matrices\n", _name));
  PetscCall(DMSetMatType(dm, MATSHELL));
  PetscCall(DMCreateMatrix(dm, &A));

  // destroy objects
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Destroy\n", _name));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&dm));

  // end main
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] End\n", _name));

  // finalize Petsc
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: p4est

  testset:

    test:
      args: -dm_forest_topology unit -dm_forest_initial_refinement 2
      suffix: unit

    test:
      args: -dm_forest_topology brick -dm_p4est_brick_size 4,5
      suffix: brick45

    test:
      args: -dm_forest_topology shell2d
      suffix: shell
TEST*/
