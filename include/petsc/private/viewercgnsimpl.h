#pragma once

#include <petsc/private/viewerimpl.h>
#include <cgnstypes.h>

#define CGNS_MAX_DIM 3

typedef struct {
  char           *filename_template;
  char           *filename;
  PetscFileMode   btype;
  int             file_num;
  const PetscInt *node_l2g;
  int             base, zone;
  PetscInt        num_local_nodes;
  /* A range of indices is half-open, i.e., [Start, End). An unstructured grid uses only Start[0] and End[0] (consistent with CGNS convention). */
  PetscInt       nStart[CGNS_MAX_DIM], nEnd[CGNS_MAX_DIM]; /* range of indices of local nodes */
  PetscInt       eStart[CGNS_MAX_DIM], eEnd[CGNS_MAX_DIM]; /* range of indices of local elements */
  PetscScalar   *nodal_field;
  PetscSegBuffer output_steps;
  PetscSegBuffer output_times;
  PetscInt       batch_size;
} PetscViewer_CGNS;

#define PetscCallCGNS(ierr) \
  do { \
    int _cgns_ier = (ierr); \
    PetscCheck(!_cgns_ier, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", _cgns_ier, cg_get_error()); \
  } while (0)

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSCheckBatch_Internal(PetscViewer);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSFileOpen_Internal(PetscViewer, PetscInt);
