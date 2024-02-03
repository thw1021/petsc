#include <petscdmbf.h> /*I "petscdmbf.h" I*/
#include <petsc/private/dmbfimpl.h>

#if defined(PETSC_HAVE_P4EST)

  #include "bf_xd.h"

static inline DM_BF_Cell *_p_getCellPtr(const DM_BF_Cell *cells, size_t cellSize, const p4est_t *p4est, p4est_topidx_t treeid, p4est_locidx_t quadid, int8_t is_ghost)
{
  if (!is_ghost) {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, treeid);

    return (DM_BF_Cell *)(((char *)cells) + cellSize * ((size_t)(tree->quadrants_offset + quadid)));
  } else {
    return (DM_BF_Cell *)(((char *)cells) + cellSize * ((size_t)(p4est->local_num_quadrants + quadid)));
  }
}

static void _p_getInfo(/*IN */ p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t treeid, p4est_locidx_t quadid, int8_t is_ghost,
                       /*OUT*/ DM_BF_Cell *cell)
{
  const p4est_qcoord_t qlength = P4EST_QUADRANT_LEN(quad->level);
  double               vertex1[3], vertex2[3];

  /* get vertex coordinates of opposite corners */
  p4est_qcoord_to_vertex(p4est->connectivity, treeid, quad->x, quad->y,
  #if defined(P4_TO_P8)
                         quad->z,
  #endif
                         vertex1);
  p4est_qcoord_to_vertex(p4est->connectivity, treeid, quad->x + qlength, quad->y + qlength,
  #if defined(P4_TO_P8)
                         quad->z + qlength,
  #endif
                         vertex2);
  /* set cell data */
  if (!is_ghost) {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, treeid);
    cell->indexLocal   = (PetscInt)(tree->quadrants_offset + quadid);
    cell->indexGlobal  = cell->indexLocal + (PetscInt)p4est->global_first_quadrant[p4est->mpirank];
  } else {
    cell->indexLocal  = (PetscInt)(p4est->local_num_quadrants + quadid);
    cell->indexGlobal = -1;
  }
  cell->level     = (PetscInt)quad->level;
  cell->corner[0] = (PetscReal)vertex1[0];
  cell->corner[1] = (PetscReal)vertex1[1];
  cell->corner[2] = (PetscReal)vertex1[2];
  //TODO set all 4/8 corners
  //TODO set volume
  cell->sidelength[0] = (PetscReal)(vertex2[0] - vertex1[0]);
  cell->sidelength[1] = (PetscReal)(vertex2[1] - vertex1[1]);
  cell->sidelength[2] = (PetscReal)(vertex2[2] - vertex1[2]);
  //TODO set side lengths to NAN if warped geometry
}

static void _p_getVecView(/*IN    */ const PetscScalar **vecViewRead, PetscInt nVecsRead, PetscScalar **vecViewReadWrite, PetscInt nVecsReadWrite, size_t cellDof,
                          /*IN/OUT*/ DM_BF_Cell *cell)
{
  PetscInt i;

  for (i = 0; i < nVecsRead; i++) { cell->vecViewRead[i] = &vecViewRead[i][cell->indexLocal * cellDof]; }
  for (i = 0; i < nVecsReadWrite; i++) { cell->vecViewReadWrite[i] = &vecViewReadWrite[i][cell->indexLocal * cellDof]; }
}

/***************************************
 * CELL SETUP
 **************************************/

typedef enum {
  SET_DMBF_CELLS,
  SET_P4EST_CELLS
} DM_BF_SetUpMode;

typedef struct _p_DM_BF_SetUpCtx {
  DM_BF_SetUpMode mode;
  /* DM-specifc info (required) */
  DM                 dm;
  DM_BF_Cell        *cells;
  const DM_BF_Shape *memory;
} DM_BF_SetUpCtx;

static void _p_iterSetUp(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_SetUpCtx *iterCtx = ctx;
  DM_BF_Cell     *cell    = PETSC_NULLPTR;

  /* get cell */
  switch (iterCtx->mode) {
  case SET_DMBF_CELLS:
    cell = _p_getCellPtr(iterCtx->cells, iterCtx->memory->size, info->p4est, info->treeid, info->quadid, 0 /*!ghost*/);
    break;
  case SET_P4EST_CELLS:
    cell = info->quad->p.user_data;
    break;
  }
  /* get cell info */
  _p_getInfo(info->p4est, info->quad, info->treeid, info->quadid, 0, cell);
  CHKERRV(DMBFCellInitialize(cell, iterCtx->memory));
  /* assign cell to forest quadrant */
  switch (iterCtx->mode) {
  case SET_DMBF_CELLS:
    info->quad->p.user_data = cell;
    break;
  case SET_P4EST_CELLS:
    break;
  }
}

  #if !defined(DMBF_XD_IterateSetUpCells)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateSetUpCells(DM dm, DM_BF_Cell *cells, const DM_BF_Shape *cellMemoryShape)
{
  DM_BF_SetUpCtx iterCtx;
  PetscErrorCode ierr;
  p4est_t       *p4est;
  p4est_ghost_t *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  /* set iterator context */
  iterCtx.mode   = (cells ? SET_DMBF_CELLS : SET_P4EST_CELLS);
  iterCtx.dm     = dm;
  iterCtx.cells  = cells;
  iterCtx.memory = cellMemoryShape;
  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  if (p4est->local_num_quadrants) {
    PetscCheck(cells || p4est->user_data_pool, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Cell data allocations do not exist");
    PetscCheck(!cells || !p4est->user_data_pool, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Unclear which cell data allocation should be used");
    PetscCheck(cells || cellMemoryShape->size == p4est->data_size, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "p4est data size mismatch: is %d, should be %d", (int)p4est->data_size, (int)cellMemoryShape->size);
  } else {
    PetscCheck(!cells, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Cell data allocations exist but should not");
  }
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterSetUp, NULL, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterSetUp, NULL, NULL));
  #endif
  p4est->data_size = cellMemoryShape->size;
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_IterateSetUpP4estCells)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateSetUpP4estCells(DM dm, const DM_BF_Shape *cellMemoryShape)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  CHKERRQ(DMBF_XD_IterateSetUpCells(dm, PETSC_NULLPTR, cellMemoryShape));
  PetscFunctionReturn(0);
}

static void _p_iterCopy(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_SetUpCtx *iterCtx = ctx;
  DM_BF_Cell     *cell    = _p_getCellPtr(iterCtx->cells, iterCtx->memory->size, info->p4est, info->treeid, info->quadid, 0 /*!ghost*/);

  CHKERRV(PetscMemcpy(cell, info->quad->p.user_data, iterCtx->memory->size));
}

  #if !defined(DMBF_XD_IterateCopyP4estCells)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateCopyP4estCells(DM dm, DM_BF_Cell *cells, const DM_BF_Shape *cellMemoryShape)
{
  DM_BF_SetUpCtx iterCtx;
  PetscErrorCode ierr;
  p4est_t       *p4est;
  p4est_ghost_t *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  /* set iterator context */
  iterCtx.dm     = dm;
  iterCtx.cells  = cells;
  iterCtx.memory = cellMemoryShape;
  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  PetscCheck(p4est->user_data_pool, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "p4est has no user data memory");
  PetscCheck(p4est->data_size == cellMemoryShape->size, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "p4est data size mismatch: is %d, should be %d", (int)p4est->data_size, (int)cellMemoryShape->size);
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterCopy, NULL, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterCopy, NULL, NULL));
  #endif
  PetscFunctionReturn(0);
}

/***************************************
 * CELL DATA
 **************************************/

  #define _p_getCellDataPtr(cell, size) (((char *)(cell)) + (size))

typedef struct _p_DM_BF_SetCellDataIterCtx {
  /* DM-specifc info (required) */
  DM          dm;
  DM_BF_Cell *cells;
  size_t      cellSize, cellOffsetDataRead, cellOffsetDataReadWrite;
  /* iterator-specific info */
  const PetscInt     *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt            nValsPerElemRead, nValsPerElemReadWrite;
  const PetscScalar **vecViewRead, **vecViewReadWrite;
} DM_BF_SetCellDataIterCtx;

static void _p_iterSetCellData(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_SetCellDataIterCtx *iterCtx = ctx;
  DM_BF_Cell               *cell    = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, info->treeid, info->quadid, 0 /*!ghost*/);
  PetscScalar              *data;
  PetscInt                  i, j, di;

  /* set cell data for reading */
  if (iterCtx->vecViewRead) {
    data = (PetscScalar *)_p_getCellDataPtr(cell, iterCtx->cellOffsetDataRead);
    di   = 0;
    for (i = 0; i < iterCtx->nValsPerElemRead; i++) {
      for (j = 0; j < iterCtx->valsPerElemRead[i]; j++) {
        data[di] = iterCtx->vecViewRead[i][iterCtx->valsPerElemRead[i] * cell->indexLocal + j];
        di++;
      }
    }
  }
  /* set cell data for reading & writing */
  if (iterCtx->vecViewReadWrite) {
    data = (PetscScalar *)_p_getCellDataPtr(cell, iterCtx->cellOffsetDataReadWrite);
    di   = 0;
    for (i = 0; i < iterCtx->nValsPerElemReadWrite; i++) {
      for (j = 0; j < iterCtx->valsPerElemReadWrite[i]; j++) {
        data[di] = iterCtx->vecViewReadWrite[i][iterCtx->valsPerElemReadWrite[i] * cell->indexLocal + j];
        di++;
      }
    }
  }
}

  #if !defined(DMBF_XD_IterateSetCellData)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateSetCellData(DM dm, DM_BF_Cell *cells, size_t cellSize, size_t cellOffsetDataRead, size_t cellOffsetDataReadWrite, const PetscInt *valsPerElemRead, PetscInt nValsPerElemRead, const PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF_SetCellDataIterCtx iterCtx;
  PetscInt                 i;
  PetscErrorCode           ierr;
  p4est_t                 *p4est;
  p4est_ghost_t           *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  /* set iterator context */
  iterCtx.dm                      = dm;
  iterCtx.cells                   = cells;
  iterCtx.cellSize                = cellSize;
  iterCtx.cellOffsetDataRead      = cellOffsetDataRead;
  iterCtx.cellOffsetDataReadWrite = cellOffsetDataReadWrite;
  iterCtx.valsPerElemRead         = valsPerElemRead;
  iterCtx.nValsPerElemRead        = nValsPerElemRead;
  iterCtx.valsPerElemReadWrite    = valsPerElemReadWrite;
  iterCtx.nValsPerElemReadWrite   = nValsPerElemReadWrite;
  if (vecRead) {
    ierr = PetscMalloc1(nValsPerElemRead, &iterCtx.vecViewRead);
    CHKERRQ(ierr);
    for (i = 0; i < nValsPerElemRead; i++) {
      ierr = VecGetArrayRead(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
    }
  } else {
    iterCtx.vecViewRead = PETSC_NULLPTR;
  }
  if (vecReadWrite) {
    ierr = PetscMalloc1(nValsPerElemReadWrite, &iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
    for (i = 0; i < nValsPerElemReadWrite; i++) {
      ierr = VecGetArrayRead(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
    }
  } else {
    iterCtx.vecViewReadWrite = PETSC_NULLPTR;
  }
  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterSetCellData, NULL, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterSetCellData, NULL, NULL));
  #endif
  /* clear iterator context */
  if (vecRead) {
    for (i = 0; i < nValsPerElemRead; i++) {
      ierr = VecRestoreArrayRead(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);
    CHKERRQ(ierr);
  }
  if (vecReadWrite) {
    for (i = 0; i < nValsPerElemReadWrite; i++) {
      ierr = VecRestoreArrayRead(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct _p_DM_BF_SetCellFieldsIterCtx {
  /* DM-specifc info (required) */
  DM          dm;
  DM_BF_Cell *cells;
  size_t      cellSize, cellOffsetDataRead, cellOffsetDataReadWrite;
  /* iterator-specific info */
  const PetscInt     *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt            nValsPerElemRead, nValsPerElemReadWrite;
  const PetscScalar **vecViewRead, **vecViewReadWrite;
  const PetscInt     *fieldsRead, *fieldsReadWrite;
  PetscInt           *cellOffsetsRead, *cellOffsetsReadWrite;
  PetscInt            nFieldsRead, nFieldsReadWrite;
} DM_BF_SetCellFieldsIterCtx;

static void _p_iterSetCellFields(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_SetCellFieldsIterCtx *iterCtx = ctx;
  DM_BF_Cell                 *cell    = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, info->treeid, info->quadid, 0 /*!ghost*/);
  PetscScalar                *data;
  PetscInt                    i, di, fn;

  /* set cell data for reading */
  if (iterCtx->vecViewRead) {
    data = (PetscScalar *)_p_getCellDataPtr(cell, iterCtx->cellOffsetDataRead);
    for (i = 0; i < iterCtx->nFieldsRead; i++) {
      fn = iterCtx->fieldsRead[i];
      for (di = iterCtx->cellOffsetsRead[i]; di < iterCtx->valsPerElemRead[fn]; di++) { data[di] = iterCtx->vecViewRead[i][iterCtx->valsPerElemRead[fn] * cell->indexLocal + di]; }
    }
  }
  /* set cell data for reading & writing */
  if (iterCtx->vecViewReadWrite) {
    data = (PetscScalar *)_p_getCellDataPtr(cell, iterCtx->cellOffsetDataReadWrite);
    for (i = 0; i < iterCtx->nFieldsReadWrite; i++) {
      fn = iterCtx->fieldsReadWrite[i];
      for (di = iterCtx->cellOffsetsReadWrite[i]; di < iterCtx->valsPerElemReadWrite[fn]; di++) { data[di] = iterCtx->vecViewReadWrite[i][iterCtx->valsPerElemReadWrite[fn] * cell->indexLocal + di]; }
    }
  }
}

  #if !defined(DMBF_XD_IterateSetCellFields)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateSetCellFields(DM dm, DM_BF_Cell *cells, size_t cellSize, size_t cellOffsetDataRead, size_t cellOffsetDataReadWrite, const PetscInt *valsPerElemRead, PetscInt nValsPerElemRead, const PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite, Vec *vecRead, Vec *vecReadWrite, PetscInt nFieldsRead, PetscInt *fieldsRead, PetscInt nFieldsReadWrite, PetscInt *fieldsReadWrite)
{
  DM_BF_SetCellFieldsIterCtx iterCtx;
  PetscInt                   i, j, fn, di;
  PetscErrorCode             ierr;
  p4est_t                   *p4est;
  p4est_ghost_t             *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  /* set iterator context */
  iterCtx.dm                      = dm;
  iterCtx.cells                   = cells;
  iterCtx.cellSize                = cellSize;
  iterCtx.cellOffsetDataRead      = cellOffsetDataRead;
  iterCtx.cellOffsetDataReadWrite = cellOffsetDataReadWrite;
  iterCtx.valsPerElemRead         = valsPerElemRead;
  iterCtx.nValsPerElemRead        = nValsPerElemRead;
  iterCtx.valsPerElemReadWrite    = valsPerElemReadWrite;
  iterCtx.nValsPerElemReadWrite   = nValsPerElemReadWrite;
  iterCtx.fieldsRead              = fieldsRead;
  iterCtx.nFieldsRead             = nFieldsRead;
  iterCtx.fieldsReadWrite         = fieldsReadWrite;
  iterCtx.nFieldsReadWrite        = nFieldsReadWrite;
  if (vecRead) {
    ierr = PetscMalloc1(nFieldsRead, &iterCtx.vecViewRead);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(nFieldsRead, &iterCtx.cellOffsetsRead);
    CHKERRQ(ierr);
    for (i = 0; i < nFieldsRead; i++) {
      ierr = VecGetArrayRead(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
      fn = fieldsRead[i];
      for (j = 0, di = 0; j < fn; j++) { di += valsPerElemRead[j]; }
      iterCtx.cellOffsetsRead[i] = di;
    }
  } else {
    iterCtx.vecViewRead     = PETSC_NULLPTR;
    iterCtx.cellOffsetsRead = PETSC_NULLPTR;
  }
  if (vecReadWrite) {
    ierr = PetscMalloc1(nFieldsReadWrite, &iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(nFieldsReadWrite, &iterCtx.cellOffsetsReadWrite);
    CHKERRQ(ierr);
    for (i = 0; i < nFieldsReadWrite; i++) {
      ierr = VecGetArrayRead(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
      fn = fieldsReadWrite[i];
      for (j = 0, di = 0; j < fn; j++) { di += valsPerElemReadWrite[j]; }
      iterCtx.cellOffsetsReadWrite[i] = di;
    }
  } else {
    iterCtx.vecViewReadWrite     = PETSC_NULLPTR;
    iterCtx.cellOffsetsReadWrite = PETSC_NULLPTR;
  }
  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterSetCellFields, NULL, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterSetCellFields, NULL, NULL));
  #endif
  /* clear iterator context */
  if (vecRead) {
    for (i = 0; i < nFieldsRead; i++) {
      ierr = VecRestoreArrayRead(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);
    CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellOffsetsRead);
    CHKERRQ(ierr);
  }
  if (vecReadWrite) {
    for (i = 0; i < nFieldsReadWrite; i++) {
      ierr = VecRestoreArrayRead(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellOffsetsReadWrite);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct _p_DM_BF_GetCellDataIterCtx {
  /* DM-specifc info (required) */
  DM          dm;
  DM_BF_Cell *cells;
  size_t      cellSize, cellOffsetDataRead, cellOffsetDataReadWrite;
  /* iterator-specific info */
  const PetscInt *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt        nValsPerElemRead, nValsPerElemReadWrite;
  PetscScalar   **vecViewRead, **vecViewReadWrite;
} DM_BF_GetCellDataIterCtx;

static void _p_iterGetCellData(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_GetCellDataIterCtx *iterCtx = ctx;
  DM_BF_Cell               *cell    = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, info->treeid, info->quadid, 0 /*!ghost*/);
  const PetscScalar        *data;
  PetscInt                  i, j, di;

  /* get cell data for reading */
  if (iterCtx->vecViewRead) {
    data = (const PetscScalar *)_p_getCellDataPtr(cell, iterCtx->cellOffsetDataRead);
    di   = 0;
    for (i = 0; i < iterCtx->nValsPerElemRead; i++) {
      for (j = 0; j < iterCtx->valsPerElemRead[i]; j++) {
        iterCtx->vecViewRead[i][iterCtx->valsPerElemRead[i] * cell->indexLocal + j] = data[di];
        di++;
      }
    }
  }
  /* get cell data for reading & writing */
  if (iterCtx->vecViewReadWrite) {
    data = (const PetscScalar *)_p_getCellDataPtr(cell, iterCtx->cellOffsetDataReadWrite);
    di   = 0;
    for (i = 0; i < iterCtx->nValsPerElemReadWrite; i++) {
      for (j = 0; j < iterCtx->valsPerElemReadWrite[i]; j++) {
        iterCtx->vecViewReadWrite[i][iterCtx->valsPerElemReadWrite[i] * cell->indexLocal + j] = data[di];
        di++;
      }
    }
  }
}

  #if !defined(DMBF_XD_IterateGetCellData)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateGetCellData(DM dm, DM_BF_Cell *cells, size_t cellSize, size_t cellOffsetDataRead, size_t cellOffsetDataReadWrite, const PetscInt *valsPerElemRead, PetscInt nValsPerElemRead, const PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF_GetCellDataIterCtx iterCtx;
  PetscInt                 i;
  PetscErrorCode           ierr;
  p4est_t                 *p4est;
  p4est_ghost_t           *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  /* set iterator context */
  iterCtx.dm                      = dm;
  iterCtx.cells                   = cells;
  iterCtx.cellSize                = cellSize;
  iterCtx.cellOffsetDataRead      = cellOffsetDataRead;
  iterCtx.cellOffsetDataReadWrite = cellOffsetDataReadWrite;
  iterCtx.valsPerElemRead         = valsPerElemRead;
  iterCtx.nValsPerElemRead        = nValsPerElemRead;
  iterCtx.valsPerElemReadWrite    = valsPerElemReadWrite;
  iterCtx.nValsPerElemReadWrite   = nValsPerElemReadWrite;
  if (vecRead) {
    ierr = PetscMalloc1(nValsPerElemRead, &iterCtx.vecViewRead);
    CHKERRQ(ierr);
    for (i = 0; i < nValsPerElemRead; i++) {
      ierr = VecGetArray(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
    }
  } else {
    iterCtx.vecViewRead = PETSC_NULLPTR;
  }
  if (vecReadWrite) {
    ierr = PetscMalloc1(nValsPerElemReadWrite, &iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
    for (i = 0; i < nValsPerElemReadWrite; i++) {
      ierr = VecGetArray(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
    }
  } else {
    iterCtx.vecViewReadWrite = PETSC_NULLPTR;
  }
  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterGetCellData, NULL, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterGetCellData, NULL, NULL));
  #endif
  /* clear iterator context */
  if (vecRead) {
    for (i = 0; i < nValsPerElemRead; i++) {
      ierr = VecRestoreArray(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);
    CHKERRQ(ierr);
  }
  if (vecReadWrite) {
    for (i = 0; i < nValsPerElemReadWrite; i++) {
      ierr = VecRestoreArray(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct _p_DM_BF_GetCellFieldsIterCtx {
  /* DM-specifc info (required) */
  DM          dm;
  DM_BF_Cell *cells;
  size_t      cellSize, cellOffsetDataRead, cellOffsetDataReadWrite;
  /* iterator-specific info */
  const PetscInt *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt        nValsPerElemRead, nValsPerElemReadWrite;
  PetscScalar   **vecViewRead, **vecViewReadWrite;
  const PetscInt *fieldsRead, *fieldsReadWrite;
  PetscInt       *cellOffsetsRead, *cellOffsetsReadWrite;
  PetscInt        nFieldsRead, nFieldsReadWrite;
} DM_BF_GetCellFieldsIterCtx;

static void _p_iterGetCellFields(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_GetCellFieldsIterCtx *iterCtx = ctx;
  DM_BF_Cell                 *cell    = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, info->treeid, info->quadid, 0 /*!ghost*/);
  PetscScalar                *data;
  PetscInt                    i, di, fn;

  /* set cell data for reading */
  if (iterCtx->vecViewRead) {
    data = (PetscScalar *)_p_getCellDataPtr(cell, iterCtx->cellOffsetDataRead);
    for (i = 0; i < iterCtx->nFieldsRead; i++) {
      fn = iterCtx->fieldsRead[i];
      for (di = iterCtx->cellOffsetsRead[i]; di < iterCtx->valsPerElemRead[fn]; di++) { iterCtx->vecViewRead[i][iterCtx->valsPerElemRead[fn] * cell->indexLocal + di] = data[di]; }
    }
  }
  /* set cell data for reading & writing */
  if (iterCtx->vecViewReadWrite) {
    data = (PetscScalar *)_p_getCellDataPtr(cell, iterCtx->cellOffsetDataReadWrite);
    for (i = 0; i < iterCtx->nFieldsReadWrite; i++) {
      fn = iterCtx->fieldsReadWrite[i];
      for (di = iterCtx->cellOffsetsReadWrite[i]; di < iterCtx->valsPerElemReadWrite[fn]; di++) { iterCtx->vecViewReadWrite[i][iterCtx->valsPerElemReadWrite[fn] * cell->indexLocal + di] = data[di]; }
    }
  }
}

  #if !defined(DMBF_XD_IterateGetCellFields)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateGetCellFields(DM dm, DM_BF_Cell *cells, size_t cellSize, size_t cellOffsetDataRead, size_t cellOffsetDataReadWrite, const PetscInt *valsPerElemRead, PetscInt nValsPerElemRead, const PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite, Vec *vecRead, Vec *vecReadWrite, PetscInt nFieldsRead, PetscInt *fieldsRead, PetscInt nFieldsReadWrite, PetscInt *fieldsReadWrite)
{
  DM_BF_GetCellFieldsIterCtx iterCtx;
  PetscInt                   i, j, fn, di;
  PetscErrorCode             ierr;
  p4est_t                   *p4est;
  p4est_ghost_t             *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  /* set iterator context */
  iterCtx.dm                      = dm;
  iterCtx.cells                   = cells;
  iterCtx.cellSize                = cellSize;
  iterCtx.cellOffsetDataRead      = cellOffsetDataRead;
  iterCtx.cellOffsetDataReadWrite = cellOffsetDataReadWrite;
  iterCtx.valsPerElemRead         = valsPerElemRead;
  iterCtx.nValsPerElemRead        = nValsPerElemRead;
  iterCtx.valsPerElemReadWrite    = valsPerElemReadWrite;
  iterCtx.nValsPerElemReadWrite   = nValsPerElemReadWrite;
  iterCtx.fieldsRead              = fieldsRead;
  iterCtx.nFieldsRead             = nFieldsRead;
  iterCtx.fieldsReadWrite         = fieldsReadWrite;
  iterCtx.nFieldsReadWrite        = nFieldsReadWrite;
  if (vecRead) {
    ierr = PetscMalloc1(nFieldsRead, &iterCtx.vecViewRead);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(nFieldsRead, &iterCtx.cellOffsetsRead);
    CHKERRQ(ierr);
    for (i = 0; i < nFieldsRead; i++) {
      ierr = VecGetArray(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
      fn = fieldsRead[i];
      for (j = 0, di = 0; j < fn; j++) { di += valsPerElemRead[j]; }
      iterCtx.cellOffsetsRead[i] = di;
    }
  } else {
    iterCtx.vecViewRead     = PETSC_NULLPTR;
    iterCtx.cellOffsetsRead = PETSC_NULLPTR;
  }
  if (vecReadWrite) {
    ierr = PetscMalloc1(nFieldsReadWrite, &iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(nFieldsReadWrite, &iterCtx.cellOffsetsReadWrite);
    CHKERRQ(ierr);
    for (i = 0; i < nFieldsReadWrite; i++) {
      ierr = VecGetArray(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
      fn = fieldsReadWrite[i];
      for (j = 0, di = 0; j < fn; j++) { di += valsPerElemReadWrite[j]; }
      iterCtx.cellOffsetsReadWrite[i] = di;
    }
  } else {
    iterCtx.vecViewReadWrite     = PETSC_NULLPTR;
    iterCtx.cellOffsetsReadWrite = PETSC_NULLPTR;
  }
  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterGetCellFields, NULL, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterGetCellFields, NULL, NULL));
  #endif
  /* clear iterator context */
  if (vecRead) {
    for (i = 0; i < nFieldsRead; i++) {
      ierr = VecRestoreArray(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);
    CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellOffsetsRead);
    CHKERRQ(ierr);
  }
  if (vecReadWrite) {
    for (i = 0; i < nFieldsReadWrite; i++) {
      ierr = VecRestoreArray(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellOffsetsReadWrite);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/***************************************
 * GHOST CELLS
 **************************************/

  #if !defined(DMBF_XD_IterateGhostExchange)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateGhostExchange(DM dm, DM_BF_Cell *cells, size_t cellSize)
{
  DM_BF_Cell    *ghostCells;
  PetscErrorCode ierr;
  p4est_t       *p4est;
  p4est_ghost_t *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  ghostCells = _p_getCellPtr(cells, cellSize, p4est, -1 /*no tree id*/, 0 /*quadid*/, 1 /*ghost*/);
  PetscCallP4est(p4est_ghost_exchange_data, (p4est, ghost, ghostCells));
  PetscFunctionReturn(0);
}

/***************************************
 * CELL ITERATORS
 **************************************/

typedef struct _p_DM_BF_CellIterCtx {
  /* DM-specifc info (required) */
  DM          dm;
  DM_BF_Cell *cells;
  size_t      cellSize;
  size_t      cellDof;
  /* iterator-specific info */
  PetscErrorCode (*iterCell)(DM, DM_BF_Cell *, void *);
  void               *userIterCtx;
  const PetscScalar **vecViewRead, **cellVecViewRead;
  PetscScalar       **vecViewReadWrite, **cellVecViewReadWrite;
  PetscInt            nVecsRead, nVecsReadWrite;
} DM_BF_CellIterCtx;

static void _p_iterVolume(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_CellIterCtx *iterCtx = ctx;
  DM_BF_Cell        *cell    = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, info->treeid, info->quadid, 0 /*!ghost*/);
  PetscErrorCode     ierr;

  /* assign vector view to cell */
  cell->vecViewRead      = iterCtx->cellVecViewRead;
  cell->vecViewReadWrite = iterCtx->cellVecViewReadWrite;
  /* get vector view */
  _p_getVecView(iterCtx->vecViewRead, iterCtx->nVecsRead, iterCtx->vecViewReadWrite, iterCtx->nVecsReadWrite, iterCtx->cellDof, cell);
  /* call cell function */
  ierr = iterCtx->iterCell(iterCtx->dm, cell, iterCtx->userIterCtx);
  CHKERRV(ierr);
  /* remove vector view from cell */
  cell->vecViewRead      = PETSC_NULLPTR;
  cell->vecViewReadWrite = PETSC_NULLPTR;
}

  #if !defined(DMBF_XD_IterateOverCellsVectors)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateOverCellsVectors(DM dm, DM_BF_Cell *cells, size_t cellSize, PetscErrorCode (*iterCell)(DM, DM_BF_Cell *, void *), void *userIterCtx, Vec *vecRead, PetscInt nVecsRead, Vec *vecReadWrite, PetscInt nVecsReadWrite)
{
  DM_BF_CellIterCtx iterCtx;
  PetscInt          blockSize[3] = {1, 1, 1};
  PetscInt          dim, n, N, i;
  PetscErrorCode    ierr;
  p4est_t          *p4est;
  p4est_ghost_t    *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidFunction(iterCell, 2);
  if (nVecsRead) PetscAssertPointer(vecRead, 6);
  if (nVecsReadWrite) PetscAssertPointer(vecReadWrite, 8);
  /* calculate number of entries per cell */
  ierr = DMBFGetInfo(dm, &dim, &n, &N, PETSC_NULLPTR);
  CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  iterCtx.cellDof = 1;
  for (i = 0; i < dim; i++) { iterCtx.cellDof *= (size_t)blockSize[i]; }
  /* set iterator context */
  iterCtx.dm             = dm;
  iterCtx.cells          = cells;
  iterCtx.cellSize       = cellSize;
  iterCtx.iterCell       = iterCell;
  iterCtx.userIterCtx    = userIterCtx;
  iterCtx.nVecsRead      = nVecsRead;
  iterCtx.nVecsReadWrite = nVecsReadWrite;
  if (0 < iterCtx.nVecsRead) {
    ierr = PetscMalloc1(iterCtx.nVecsRead, &iterCtx.cellVecViewRead);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(iterCtx.nVecsRead, &iterCtx.vecViewRead);
    CHKERRQ(ierr);
    for (i = 0; i < iterCtx.nVecsRead; i++) {
      ierr = VecGetArrayRead(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
    }
  }
  if (0 < iterCtx.nVecsReadWrite) {
    ierr = PetscMalloc1(iterCtx.nVecsReadWrite, &iterCtx.cellVecViewReadWrite);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(iterCtx.nVecsReadWrite, &iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
    for (i = 0; i < iterCtx.nVecsReadWrite; i++) {
      ierr = VecGetArray(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
    }
  }
  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterVolume, NULL, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, _p_iterVolume, NULL, NULL));
  #endif
  /* clear iterator context */
  if (0 < iterCtx.nVecsRead) {
    for (i = 0; i < iterCtx.nVecsRead; i++) {
      ierr = VecRestoreArrayRead(vecRead[i], &iterCtx.vecViewRead[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);
    CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellVecViewRead);
    CHKERRQ(ierr);
  }
  if (0 < iterCtx.nVecsReadWrite) {
    for (i = 0; i < iterCtx.nVecsReadWrite; i++) {
      ierr = VecRestoreArray(vecReadWrite[i], &iterCtx.vecViewReadWrite[i]);
      CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);
    CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellVecViewReadWrite);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/***************************************
 * FACE ITERATORS
 **************************************/

typedef struct _p_DM_BF_FaceIterCtx {
  /* DM-specifc info (required) */
  DM          dm;
  DM_BF_Cell *cells;
  size_t      cellSize;
  /* iterator-specific info */
  PetscErrorCode (*iterFace)(DM, DM_BF_Face *, void *);
  void      *userIterCtx;
  DM_BF_Face face;
} DM_BF_FaceIterCtx;

static void _p_iterFace(p4est_iter_face_info_t *info, void *ctx)
{
  DM_BF_FaceIterCtx *iterCtx    = ctx;
  DM_BF_Face        *face       = &iterCtx->face;
  const PetscBool    isBoundary = (1 == info->sides.elem_count);
  PetscInt           i;

  #if defined(PETSC_USE_DEBUG)
  face->cellL[0] = PETSC_NULLPTR;
  face->cellL[1] = PETSC_NULLPTR;
  face->cellL[2] = PETSC_NULLPTR;
  face->cellL[3] = PETSC_NULLPTR;
  face->cellR[0] = PETSC_NULLPTR;
  face->cellR[1] = PETSC_NULLPTR;
  face->cellR[2] = PETSC_NULLPTR;
  face->cellR[3] = PETSC_NULLPTR;
  #endif

  /* get cell and vector data */
  if (isBoundary) {
    p4est_iter_face_side_t *side = p4est_iter_fside_array_index_int(&info->sides, 0);

    face->boundary = (DM_BF_FaceBoundary)side->face;
    face->dir      = (DM_BF_FaceDir)side->face;
    if (DM_BF_FACEBOUNDARY_XPOS == face->boundary || DM_BF_FACEBOUNDARY_YPOS == face->boundary || DM_BF_FACEBOUNDARY_ZPOS == face->boundary) {
      face->nCellsL  = 1;
      face->nCellsR  = 0;
      face->cellL[0] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, side->treeid, side->is.full.quadid, 0 /*!ghost*/);
    } else if (DM_BF_FACEBOUNDARY_XNEG == face->boundary || DM_BF_FACEBOUNDARY_YNEG == face->boundary || DM_BF_FACEBOUNDARY_ZNEG == face->boundary) {
      face->nCellsL  = 0;
      face->nCellsR  = 1;
      face->cellR[0] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, side->treeid, side->is.full.quadid, 0 /*!ghost*/);
    } else {
      SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Type of boundary face is unknown");
    }
  } else { /* !isBoundary */
    p4est_iter_face_side_t *sideL = p4est_iter_fside_array_index_int(&info->sides, 0);
    p4est_iter_face_side_t *sideR = p4est_iter_fside_array_index_int(&info->sides, 1);

    face->boundary = DM_BF_FACEBOUNDARY_NONE;
    face->dir      = (DM_BF_FaceDir)sideL->face;
    face->nCellsL  = (sideL->is_hanging ? P4EST_HALF : 1);
    face->nCellsR  = (sideR->is_hanging ? P4EST_HALF : 1);
    if (!(1 <= face->nCellsL && 1 <= face->nCellsR && (face->nCellsL + face->nCellsR) <= (P4EST_HALF + 1))) { SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatch of number of left and right cells"); }
    // set pointers to cells
    if (sideL->is_hanging) {
      for (i = 0; i < face->nCellsL; i++) { face->cellL[i] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, sideL->treeid, sideL->is.hanging.quadid[i], sideL->is.hanging.is_ghost[i]); }
    } else {
      face->cellL[0] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, sideL->treeid, sideL->is.full.quadid, sideL->is.full.is_ghost);
    }
    if (sideR->is_hanging) {
      for (i = 0; i < face->nCellsR; i++) { face->cellR[i] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, sideR->treeid, sideR->is.hanging.quadid[i], sideR->is.hanging.is_ghost[i]); }
    } else {
      face->cellR[0] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, sideR->treeid, sideR->is.full.quadid, sideR->is.full.is_ghost);
    }
  }
  /* call face function */
  CHKERRV(iterCtx->iterFace(iterCtx->dm, face, iterCtx->userIterCtx));
}

  #if !defined(DMBF_XD_IterateOverFaces)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateOverFaces(DM dm, DM_BF_Cell *cells, size_t cellSize, PetscErrorCode (*iterFace)(DM, DM_BF_Face *, void *), void *userIterCtx)
{
  DM_BF_FaceIterCtx iterCtx;
  PetscErrorCode    ierr;
  p4est_t          *p4est;
  p4est_ghost_t    *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidFunction(iterFace, 2);
  /* set iterator context */
  iterCtx.dm          = dm;
  iterCtx.cells       = cells;
  iterCtx.cellSize    = cellSize;
  iterCtx.iterFace    = iterFace;
  iterCtx.userIterCtx = userIterCtx;
  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, NULL, _p_iterFace, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, NULL, _p_iterFace, NULL));
  #endif
  PetscFunctionReturn(0);
}

typedef struct _p_DM_BF_FVMatAssemblyIterCtx {
  /* DM-specifc info (required) */
  DM          dm;
  DM_BF_Cell *cells;
  size_t      cellSize;
  /* iterator-specific info */
  PetscErrorCode (*iterFace)(DM, DM_BF_Face *, PetscReal *, void *);
  PetscReal *cellCoeff;
  PetscInt  *rowIndices;
  PetscInt  *colIndices;
  Mat        M;
  void      *userIterCtx;
  DM_BF_Face face;
} DM_BF_FVMatAssemblyIterCtx;

static void _p_iterFVMatAssembly(p4est_iter_face_info_t *info, void *ctx)
{
  DM_BF_FVMatAssemblyIterCtx *iterCtx      = ctx;
  DM_BF_Face                 *face         = &iterCtx->face;
  const PetscBool             isBoundary   = (1 == info->sides.elem_count);
  PetscInt                    blockSize[3] = {1, 1, 1};
  PetscInt                    i, j, k = 0, len = 0, bs, idx;

  #if defined(PETSC_USE_DEBUG)
  face->cellL[0] = PETSC_NULLPTR;
  face->cellL[1] = PETSC_NULLPTR;
  face->cellL[2] = PETSC_NULLPTR;
  face->cellL[3] = PETSC_NULLPTR;
  face->cellR[0] = PETSC_NULLPTR;
  face->cellR[1] = PETSC_NULLPTR;
  face->cellR[2] = PETSC_NULLPTR;
  face->cellR[3] = PETSC_NULLPTR;
  #endif

  CHKERRV(DMBFGetBlockSize(iterCtx->dm, blockSize)); /* set indices of values to set in matrix */
  bs = blockSize[0] * blockSize[1] * blockSize[2];

  /* get cell and vector data */
  if (isBoundary) {
    p4est_iter_face_side_t *side = p4est_iter_fside_array_index_int(&info->sides, 0);

    face->nCellsL  = 1;
    face->nCellsR  = 0;
    face->cellL[0] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, side->treeid, side->is.full.quadid, 0 /*!ghost*/);
    len++;
    idx = face->cellL[0]->indexLocal;
    for (j = 0; j < bs; j++, k++) {
      iterCtx->rowIndices[k] = bs * idx + j;
      iterCtx->colIndices[k] = bs * idx + j;
    }
  } else { /* !isBoundary */
    p4est_iter_face_side_t *sideL = p4est_iter_fside_array_index_int(&info->sides, 0);
    p4est_iter_face_side_t *sideR = p4est_iter_fside_array_index_int(&info->sides, 1);

    //TODO copy code from above
    face->nCellsL = (sideL->is_hanging ? 2 : 1);                                               //TODO only 2D
    face->nCellsR = (sideR->is_hanging ? 2 : 1);                                               //TODO only 2D
    if (!(1 <= face->nCellsL && 1 <= face->nCellsR && (face->nCellsL + face->nCellsR) <= 3)) { //TODO only 2D
      //TODO error
    }
    if (sideL->is_hanging) {
      for (i = 0; i < face->nCellsL; i++) {
        face->cellL[i] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, sideL->treeid, sideL->is.hanging.quadid[i], sideL->is.hanging.is_ghost[i]);
        len++;
        if (!sideL->is.hanging.is_ghost[i]) {
          idx = face->cellL[i]->indexLocal;
        } else {
          idx = info->p4est->local_num_quadrants + sideL->is.hanging.quadid[i];
        }
        for (j = 0; j < bs; j++, k++) {
          iterCtx->colIndices[k] = bs * idx + j;
          if (sideL->is.hanging.is_ghost[i]) {
            iterCtx->rowIndices[k] = -1;
          } else {
            iterCtx->rowIndices[k] = bs * idx + j;
          }
        }
      }
    } else {
      face->cellL[0] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, sideL->treeid, sideL->is.full.quadid, sideL->is.full.is_ghost);
      len++;
      if (!sideL->is.full.is_ghost) {
        idx = face->cellL[0]->indexLocal;
      } else {
        idx = info->p4est->local_num_quadrants + sideL->is.full.quadid;
      }
      for (j = 0; j < bs; j++, k++) {
        iterCtx->colIndices[k] = bs * idx + j;
        if (sideL->is.full.is_ghost) {
          iterCtx->rowIndices[k] = -1;
        } else {
          iterCtx->rowIndices[k] = bs * idx + j;
        }
      }
    }
    if (sideR->is_hanging) {
      for (i = 0; i < face->nCellsR; i++) {
        face->cellR[i] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, sideR->treeid, sideR->is.hanging.quadid[i], sideR->is.hanging.is_ghost[i]);
        len++;
        if (!sideR->is.hanging.is_ghost[i]) {
          idx = face->cellR[i]->indexLocal;
        } else {
          idx = info->p4est->local_num_quadrants + sideR->is.hanging.quadid[i];
        }
        for (j = 0; j < bs; j++, k++) {
          iterCtx->colIndices[k] = bs * idx + j;
          if (sideR->is.hanging.is_ghost[i]) {
            iterCtx->rowIndices[k] = -1;
          } else {
            iterCtx->rowIndices[k] = bs * idx + j;
          }
        }
      }
    } else {
      face->cellR[0] = _p_getCellPtr(iterCtx->cells, iterCtx->cellSize, info->p4est, sideR->treeid, sideR->is.full.quadid, sideR->is.full.is_ghost);
      len++;
      if (!sideR->is.full.is_ghost) {
        idx = face->cellR[0]->indexLocal;
      } else {
        idx = info->p4est->local_num_quadrants + sideR->is.full.quadid;
      }
      for (j = 0; j < bs; j++, k++) {
        iterCtx->colIndices[k] = bs * idx + j;
        if (sideR->is.full.is_ghost) {
          iterCtx->rowIndices[k] = -1;
        } else {
          iterCtx->rowIndices[k] = bs * idx + j;
        }
      }
    }
  }
  /* call face function */
  CHKERRV(iterCtx->iterFace(iterCtx->dm, face, iterCtx->cellCoeff, iterCtx->userIterCtx));
  CHKERRV(MatSetValuesLocal(iterCtx->M, bs * len, iterCtx->rowIndices, bs * len, iterCtx->colIndices, iterCtx->cellCoeff, ADD_VALUES));
}

  #if !defined(DMBF_XD_IterateFVMatAssembly)
static
  #endif
  PetscErrorCode
  DMBF_XD_IterateFVMatAssembly(DM dm, DM_BF_Cell *cells, size_t cellSize, Mat M, PetscErrorCode (*iterFace)(DM, DM_BF_Face *, PetscReal *, void *), void *userIterCtx)
{
  DM_BF_FVMatAssemblyIterCtx iterCtx;
  PetscErrorCode             ierr;
  p4est_t                   *p4est;
  p4est_ghost_t             *ghost;
  PetscInt                   blockSize[3] = {1, 1, 1}, bs;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidFunction(iterFace, 2);
  /* set iterator context */
  iterCtx.dm          = dm;
  iterCtx.cells       = cells;
  iterCtx.cellSize    = cellSize;
  iterCtx.iterFace    = iterFace;
  iterCtx.userIterCtx = userIterCtx;
  iterCtx.M           = M;

  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  bs = blockSize[0] * blockSize[1] * blockSize[2];
  PetscMalloc1(bs * bs * 3 * 3, &iterCtx.cellCoeff);
  CHKERRQ(ierr); /* TODO In 3D, 3 should be 5 */
  PetscMalloc1(bs * 3, &iterCtx.rowIndices);
  CHKERRQ(ierr);
  PetscMalloc1(bs * 3, &iterCtx.colIndices);
  CHKERRQ(ierr);

  /* run iterator */
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm, &ghost);
  CHKERRQ(ierr);
  #if defined(P4_TO_P8)
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, NULL, _p_iterFVMatAssembly, NULL, NULL));
  #else
  PetscCallP4est(p4est_iterate, (p4est, ghost, &iterCtx, NULL, _p_iterFVMatAssembly, NULL));
  #endif

  /* destroy */
  ierr = PetscFree(iterCtx.cellCoeff);
  CHKERRQ(ierr);
  ierr = PetscFree(iterCtx.rowIndices);
  CHKERRQ(ierr);
  ierr = PetscFree(iterCtx.colIndices);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
