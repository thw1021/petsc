#pragma once

#include <petscviewerexodusii.h>

#if defined(PETSC_HAVE_EXODUSII)

typedef struct {
  char         *filename;
  PetscFileMode btype;
  int           exoid;
  PetscInt      order; /* the "order" of the mesh, used to construct tri6, tetra10 cells */
  
  int numNodalVariables;
  int numGlobalVariables;
  int numZonalVariables;
  char *nodalVariableNames;
  char *globalVariableNames;
  char *zonalVariableNames;
} PetscViewer_ExodusII;

#endif
