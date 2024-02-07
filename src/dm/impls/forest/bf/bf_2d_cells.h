#if !defined(PETSCDMBF_2D_CELLS_H)
  #define PETSCDMBF_2D_CELLS_H

  #include "bf_2d_topology.h"

typedef struct _p_DM_BF_2D_Cells DM_BF_2D_Cells;

PETSC_EXTERN PetscErrorCode DMBF_2D_CellsCreate(DM, DM_BF_2D_Topology *, DM_BF_2D_Cells **, PetscErrorCode (*)(DM, void *));
PETSC_EXTERN PetscErrorCode DMBF_2D_CellsDestroy(DM, DM_BF_2D_Cells *);
PETSC_EXTERN PetscErrorCode DMBF_2D_CellsClone(DM_BF_2D_Cells *, DM_BF_2D_Cells **, DM);

PETSC_EXTERN PetscErrorCode DMBF_2D_CellsCoarsen(DM_BF_2D_Cells *, DM_BF_2D_Cells **, DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMBF_2D_CellsRefine(DM_BF_2D_Cells *, DM_BF_2D_Cells **, DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMBF_2D_CellsAmrAdapt(DM_BF_2D_Cells *, DM_BF_2D_Cells **, DM, DM_BF_AmrOps *, PetscInt, PetscInt, const DM_BF_Shape *);
PETSC_EXTERN PetscErrorCode DMBF_2D_CellsAmrPartition(DM_BF_2D_Cells *);
PETSC_EXTERN PetscErrorCode DMBF_2D_CellsAmrFinalize(DM, DM_BF_2D_Cells *, DM_BF_Cell *, const DM_BF_Shape *);

PETSC_EXTERN PetscErrorCode DMBF_2D_GetSizes(DM, DM_BF_2D_Cells *, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMBF_2D_GetLocalToGlobalIndices(DM, DM_BF_2D_Cells *, PetscInt *, PetscInt *toIdx);

PETSC_EXTERN PetscErrorCode DMBF_2D_CellsGetP4est(DM_BF_2D_Cells *, void *);
PETSC_EXTERN PetscErrorCode DMBF_2D_CellsGetGhost(DM_BF_2D_Cells *, void *);
PETSC_EXTERN PetscErrorCode DMBF_2D_CellsGetP4estMesh(DM_BF_2D_Cells *, void *);

#endif /* defined(PETSCDMBF_2D_CELLS_H) */
