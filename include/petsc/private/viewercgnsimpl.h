#pragma once

#include <petsc/private/viewerimpl.h>
#include <cgnstypes.h>

typedef struct {
  char           *filename_template;
  char           *filename;
  PetscFileMode   btype;
  int             file_num;
  const PetscInt *node_l2g;
  int             base, zone;
  PetscInt        num_local_nodes, nStart, nEnd;
  PetscInt        eStart, eEnd;
  PetscScalar    *nodal_field;
  PetscSegBuffer  output_steps;
  PetscSegBuffer  output_times;
  PetscInt        batch_size;
} PetscViewer_CGNS;

#define PetscCallCGNS(ierr) \
  do { \
    int _cgns_ier = (ierr); \
    PetscCheck(!_cgns_ier, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", _cgns_ier, cg_get_error()); \
  } while (0)

#if CG_SIZEOF_SIZE == 32
  // cgsize_t is defined as int
  #define MPIU_CGSIZE     MPI_INT
  #define PetscCGSize_FMT "d"
#else
  // cgsize_t is defined as long
  #define MPIU_CGSIZE     MPI_LONG
  #define PetscCGSize_FMT "ld"
#endif

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSCheckBatch_Internal(PetscViewer);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerCGNSFileOpen_Internal(PetscViewer, PetscInt);
