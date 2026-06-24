#ifdef PETSC_HAVE_GRID
  #ifndef PETSCGRID_H
    #define PETSCGRID_H
  #endif
//#include <../../arch-darwin-c-complex-debug/include/Grid/Grid.h>

#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/

typedef enum {
  GRID_LATTICE_COLD,
  GRID_LATTICE_TEPID,
  GRID_LATTICE_HOT,
  GRID_LATTICE_FILE,
  GRID_LATTICE_NUM_TYPES,
} GRID_LOAD_TYPE;

PETSC_EXTERN PetscErrorCode PetscSetGauge_Grid(DM, GRID_LOAD_TYPE, PetscBool, PetscInt, PetscBool, PetscScalar[], const char *);
PETSC_EXTERN PetscErrorCode PetscCheckDwfWithGrid(DM, Mat, Vec, Vec, const char *);
PETSC_EXTERN PetscErrorCode PetscCheckWilsonWithGrid(DM, Mat, Vec, Vec, const char *);
PETSC_EXTERN PetscErrorCode PetscGridSetUpWilson(DM, PetscBool, Mat*);
PETSC_EXTERN PetscErrorCode PetscGridSetUpDdwf(DM, PetscBool, Mat*);
PETSC_EXTERN PetscErrorCode PetscInitializeGrid(int, char **);
PETSC_EXTERN PetscErrorCode PetscFinalizeGrid();
#endif
