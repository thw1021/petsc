#include "bf_2d_cells.h"
#include "bf_2d_iterate.h"

#if defined(PETSC_HAVE_P4EST)

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_P4estCreate             DMBF_2D_P4estCreate
#define DMBF_XD_P4estDestroy            DMBF_2D_P4estDestroy
#define DMBF_XD_GhostCreate             DMBF_2D_GhostCreate
#define DMBF_XD_GhostDestroy            DMBF_2D_GhostDestroy
#define DMBF_XD_P4estMeshCreate         DMBF_2D_P4estMeshCreate
#define DMBF_XD_P4estMeshDestroy        DMBF_2D_P4estMeshDestroy

#define DM_BF_XD_Topology               DM_BF_2D_Topology
#define DM_BF_XD_Cells                  DM_BF_2D_Cells
#define _p_DM_BF_XD_Cells               _p_DM_BF_2D_Cells

#define DMBF_XD_TopologyGetConnectivity DMBF_2D_TopologyGetConnectivity
#define DMBF_XD_IterateSetUpP4estCells  DMBF_2D_IterateSetUpP4estCells
#define DMBF_XD_IterateCopyP4estCells   DMBF_2D_IterateCopyP4estCells

#define DMBF_XD_CellsCreate             DMBF_2D_CellsCreate
#define DMBF_XD_CellsDestroy            DMBF_2D_CellsDestroy
#define DMBF_XD_CellsClone              DMBF_2D_CellsClone

#define DMBF_XD_CellsCoarsen            DMBF_2D_CellsCoarsen
#define DMBF_XD_CellsRefine             DMBF_2D_CellsRefine
#define DMBF_XD_CellsAmrAdapt           DMBF_2D_CellsAmrAdapt
#define DMBF_XD_CellsAmrPartition       DMBF_2D_CellsAmrPartition
#define DMBF_XD_CellsAmrFinalize        DMBF_2D_CellsAmrFinalize

#define DMBF_XD_GetSizes                DMBF_2D_GetSizes
#define DMBF_XD_GetLocalToGlobalIndices DMBF_2D_GetLocalToGlobalIndices
#define DMBF_XD_CellsGetP4est           DMBF_2D_CellsGetP4est
#define DMBF_XD_CellsGetGhost           DMBF_2D_CellsGetGhost
#define DMBF_XD_CellsGetP4estMesh       DMBF_2D_CellsGetP4estMesh

/* include generic functions */
#include "bf_xd_cells.h"

#endif /* defined(PETSC_HAVE_P4EST) */
