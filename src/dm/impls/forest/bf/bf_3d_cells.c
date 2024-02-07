#include "bf_3d_cells.h"
#include "bf_3d_iterate.h"

#if defined(PETSC_HAVE_P4EST)

  /* convert to p8est for 3D domains */
  #if defined(PETSC_HAVE_MPIUNI)
    #undef MPI_SUCCESS
  #endif
  #if defined(PETSC_HAVE_P4EST)
    #include <p4est_to_p8est.h>
  #endif
  #if defined(PETSC_HAVE_MPIUNI)
    #define MPI_SUCCESS 0
  #endif

  /* rename generic functions that are the same for 2D and 3D */
  #define DMBF_XD_P4estCreate      DMBF_3D_P4estCreate
  #define DMBF_XD_P4estDestroy     DMBF_3D_P4estDestroy
  #define DMBF_XD_GhostCreate      DMBF_3D_GhostCreate
  #define DMBF_XD_GhostDestroy     DMBF_3D_GhostDestroy
  #define DMBF_XD_P4estMeshCreate  DMBF_3D_P4estMeshCreate
  #define DMBF_XD_P4estMeshDestroy DMBF_3D_P4estMeshDestroy

  #define DM_BF_XD_Topology DM_BF_3D_Topology
  #define DM_BF_XD_Cells    DM_BF_3D_Cells
  #define _p_DM_BF_XD_Cells _p_DM_BF_3D_Cells

  #define DMBF_XD_TopologyGetConnectivity DMBF_3D_TopologyGetConnectivity
  #define DMBF_XD_IterateSetUpP4estCells  DMBF_3D_IterateSetUpP4estCells
  #define DMBF_XD_IterateCopyP4estCells   DMBF_3D_IterateCopyP4estCells

  #define DMBF_XD_CellsCreate  DMBF_3D_CellsCreate
  #define DMBF_XD_CellsDestroy DMBF_3D_CellsDestroy
  #define DMBF_XD_CellsClone   DMBF_3D_CellsClone

  #define DMBF_XD_CellsCoarsen      DMBF_3D_CellsCoarsen
  #define DMBF_XD_CellsRefine       DMBF_3D_CellsRefine
  #define DMBF_XD_CellsAmrAdapt     DMBF_3D_CellsAmrAdapt
  #define DMBF_XD_CellsAmrPartition DMBF_3D_CellsAmrPartition
  #define DMBF_XD_CellsAmrFinalize  DMBF_3D_CellsAmrFinalize

  #define DMBF_XD_GetSizes                DMBF_3D_GetSizes
  #define DMBF_XD_GetLocalToGlobalIndices DMBF_3D_GetLocalToGlobalIndices
  #define DMBF_XD_CellsGetP4est           DMBF_3D_CellsGetP4est
  #define DMBF_XD_CellsGetGhost           DMBF_3D_CellsGetGhost
  #define DMBF_XD_CellsGetP4estMesh       DMBF_3D_CellsGetP4estMesh

  /* include generic functions */
  #include "bf_xd_cells.h"

#endif /* defined(PETSC_HAVE_P4EST) */
