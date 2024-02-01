static char help[] = "Create a 3D DMBF object\n";

#include <petsc.h>
#include <petscdmbf.h>
#include <petscdmforest.h>

enum {
  CELLDATA_X,
  CELLDATA_Y,
  CELLDATA_Z,
  CELLDATA_U,
  CELLDATA_N_
};

#define CELLDATA_D_ 3

static const PetscInt CELLDATA_SHAPE[CELLDATA_N_ * CELLDATA_D_] = {
  /* X */ 16, 1,  0,
  /* Y */ 16, 1,  0,
  /* Z */ 16, 1,  0,
  /* U */ 16, 16, 16};

int main(int argc, char **argv)
{
  const char     _name[] = "3D-DMBF";
  DM             dm;
  DMType         dmtype;
  Vec            v;
  Mat            A;
  PetscErrorCode ierr;

  // initialize Petsc
  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (ierr) return ierr;

  PetscPrintf(PETSC_COMM_WORLD, "[%s] Begin\n", _name);

  // create DM
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Create DM\n", _name);
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);
  CHKERRQ(ierr);
  ierr = DMSetType(dm, "bf");
  CHKERRQ(ierr);
  // set DM options
  ierr = DMSetDimension(dm, 3);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);
  CHKERRQ(ierr);
  // print DM type
  ierr = DMGetType(dm, &dmtype);
  CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "[%s] DM type = %s\n", _name, dmtype);

  // set cell data shapes
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Set cell data shape\n", _name);
  ierr = DMBFSetCellDataShape(dm, CELLDATA_SHAPE, CELLDATA_N_, CELLDATA_D_);
  CHKERRQ(ierr);
  // check cell data shapes
  {
    PetscInt *shapeElements = PETSC_NULLPTR;
    PetscInt  n, d, i, j;

    ierr = DMBFGetCellDataShape(dm, &shapeElements, &n, &d);
    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] Shape elements dim=(%d,%d)\n", _name, n, d);
    for (i = 0; i < n; i++) {
      PetscPrintf(PETSC_COMM_WORLD, "[%s]   %d: ", _name, i);
      for (j = 0; j < d; j++) { PetscPrintf(PETSC_COMM_WORLD, "%d ", shapeElements[i * d + j]); }
      PetscPrintf(PETSC_COMM_WORLD, "\n");
    }
    if (shapeElements) {
      ierr = PetscFree(shapeElements);
      CHKERRQ(ierr);
    }
  }

  // setup DM
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Set up DM\n", _name);
  ierr = DMSetUp(dm);
  CHKERRQ(ierr);

  // print info
  {
    PetscInt dim, n, N, ng;

    ierr = DMBFGetInfo(dm, &dim, &n, &N, &ng);
    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] Info about the DM\n", _name);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] - dimension              = %" PetscInt_FMT "\n", _name, dim);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] - number of local cells  = %" PetscInt_FMT "\n", _name, n);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] - number of global cells = %" PetscInt_FMT "\n", _name, N);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] - number of ghost cells  = %" PetscInt_FMT "\n", _name, ng);
  }

  // create derived objects
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Create vectors\n", _name);
  ierr = DMCreateGlobalVector(dm, &v);
  CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Create matrices\n", _name);
  ierr = DMSetMatType(dm, MATSHELL);
  CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &A);
  CHKERRQ(ierr);

  // destroy objects
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Destroy\n", _name);
  ierr = MatDestroy(&A);
  CHKERRQ(ierr);
  ierr = VecDestroy(&v);
  CHKERRQ(ierr);
  ierr = DMDestroy(&dm);
  CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD, "[%s] End\n", _name);

  // finalize Petsc
  ierr = PetscFinalize();
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST
  testset:
    requires: p4est !single

    test:
      args: -dm_forest_topology unit -dm_forest_initial_refinement 2
      suffix: unit

    test:
      args: -dm_forest_topology brick -dm_p4est_brick_size 4,5,6
      suffix: brick456

    test:
      args: -dm_forest_topology shell
      suffix: shell
TEST*/
