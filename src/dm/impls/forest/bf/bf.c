#include <petscdmbf.h> /*I "petscdmbf.h" I*/
#include <petsc/private/dmbfimpl.h>
#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/
#include <petsc/private/dmimpl.h>       /*I "petscdm.h" I*/
#include "bf_2d_topology.h"
#include "bf_3d_topology.h"
#include "bf_2d_cells.h"
#include "bf_3d_cells.h"
#include "bf_2d_iterate.h"
#include "bf_3d_iterate.h"
#include "bf_2d_vtu.h"
#include "bf_3d_vtu.h"
#include "../p4est/petsc_p4est_package.h"

/******************************************************************************
 * PRIVATE STRUCTURES
 *****************************************************************************/

typedef struct _p_DM_BF {
  /* forest-of-tree objects sequence: topology -> cells -> nodes */
  void *ftTopology;
  void *ftCells;
  void *ftNodes;
  /* DMBF cells */
  DM_BF_Cell *cells;
  PetscBool   ownedCellsSetUpCalled;
  PetscBool   ghostCellsSetUpCalled;
  DM_BF_Shape cellMemoryShape;
  /* [option] cell data shapes */
  DM_BF_Shape cellDataShape;
  /* [option] variable cell data */
  size_t cellDataVSize;
  /* [option] blocks within a cell TODO deprecated */
  PetscInt blockSize[3];
  /* [option] settings for cell data TODO deprecated */
  PetscInt *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt  nValsPerElemRead, nValsPerElemReadWrite;
  PetscInt  valsPerElemReadTotal, valsPerElemReadWriteTotal;
  /* setup user functions */
  PetscErrorCode (*setUpUserFnAfterP4estTopology)(DM, void *);
  PetscErrorCode (*setUpUserFnAfterP4estCells)(DM, void *);
  PetscErrorCode (*setUpUserFnAfterP4estNodes)(DM, void *);
  /* local to global scatter object, for parallel communication of data between local and global vectors */
  VecScatter ltog;
  /* AMR callback functions */
  DM_BF_AmrOps *amrOps;
} DM_BF;

/******************************************************************************
 * PRIVATE FUNCTIONS WITHOUT ERROR CHECKING
 *****************************************************************************/

#define _p_comm(dm) PetscObjectComm((PetscObject)(dm))

#define _p_SETERRQ_UNREACHABLE(dm) SETERRQ(PetscObjectComm((PetscObject)(dm)), PETSC_ERR_SUP, "Unreachable code")

static inline DM_BF *_p_getBF(DM dm)
{
  return (DM_BF *)((DM_Forest *)dm->data)->data;
}

static PetscInt _p_dim(DM dm)
{
  PetscInt dim;

  CHKERRABORT(_p_comm(dm), DMGetDimension(dm, &dim));
  return dim;
}

static PetscInt _p_nCells(DM dm)
{
  PetscInt dim, n;

  CHKERRABORT(_p_comm(dm), DMBFGetInfo(dm, &dim, &n, PETSC_NULLPTR, PETSC_NULLPTR));
  return n;
}

/******************************************************************************
 * PRIVATE FUNCTION DEFINITIONS
 *****************************************************************************/

static PetscErrorCode DMForestDestroy_BF(DM);
static PetscErrorCode DMClone_BF(DM, DM *);

static PetscErrorCode DMCreateLocalVector_BF(DM, Vec *);
static PetscErrorCode DMCreateGlobalVector_BF(DM, Vec *);
static PetscErrorCode DMLocalToGlobalBegin_BF(DM, Vec, InsertMode, Vec);
static PetscErrorCode DMLocalToGlobalEnd_BF(DM, Vec, InsertMode, Vec);
static PetscErrorCode DMGlobalToLocalBegin_BF(DM, Vec, InsertMode, Vec);
static PetscErrorCode DMGlobalToLocalEnd_BF(DM, Vec, InsertMode, Vec);

static PetscErrorCode DMCreateMatrix_BF(DM, Mat *);

static PetscErrorCode DMCoarsen_BF(DM, MPI_Comm, DM *);
static PetscErrorCode DMRefine_BF(DM, MPI_Comm, DM *);
static PetscErrorCode DMView_BF(DM, PetscViewer);
static PetscErrorCode VecView_BF(Vec, PetscViewer);

/******************************************************************************
 * PRIVATE & PUBLIC FUNCTIONS
 *****************************************************************************/

/***************************************
 * CHECKING
 **************************************/

static PetscErrorCode DMBFCheck(DM dm)
{
  PetscBool isCorrectDM;
  DM_BF    *bf;

  PetscFunctionBegin;
  /* check type of DM */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm, DMBF, &isCorrectDM));
  PetscCheck(isCorrectDM, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Type of DM is %s, but has to be %s", ((PetscObject)dm)->type_name, DMBF);
  /* check cells */
  bf = _p_getBF(dm);
  PetscCheck(bf->cells || !_p_nCells(dm), _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  PetscCheck(bf->ownedCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Owned cells not set up");
  PetscCheck(bf->ghostCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Ghost cells not set up");
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_DEBUG)
  #define DMBFCheckDebug(dm) DMBFCheck((dm))
#else
  #define DMBFCheckDebug(dm) ((void)(0))
#endif

/***************************************
 * CELL MEMORY
 **************************************/

#define _p_bytesAlign(a)        (((a) + (PETSC_MEMALIGN - 1)) & ~(PETSC_MEMALIGN - 1))
#define _p_bytesAlignPadding(a) ((((a) + (PETSC_MEMALIGN - 1)) & ~(PETSC_MEMALIGN - 1)) - (a))

static PetscErrorCode DMBF_CellMemoryShapeSetUp(DM dm)
{
  DM_BF         *bf;
  PetscBool      isSetUp;
  const size_t   nCell = DMBF_CELLMEMIDX_DATA;
  size_t         nData, dim, *elements, *pad, i, j, e, s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  /* set dimensions */
  ierr = DMBFShapeIsSetUp(&bf->cellDataShape, &isSetUp);
  CHKERRQ(ierr);
  if (isSetUp) {
    ierr = DMBFShapeCheckValid(&bf->cellDataShape);
    CHKERRQ(ierr);
    nData = bf->cellDataShape.n;
    dim   = bf->cellDataShape.dim + 1;
  } else {
    nData = 0;
    dim   = 2;
  }
  /* create elements and padding */
  ierr = PetscMalloc1((nCell + nData) * dim, &elements);
  CHKERRQ(ierr);
  ierr = PetscMalloc1((nCell + nData), &pad);
  CHKERRQ(ierr);
  /* set zero all elements */
  for (i = 0; i < (nCell + nData); i++) {
    for (j = 0; j < dim; j++) { elements[i * dim + j] = 0; }
    pad[i] = 0;
  }
  /* set cell info */
  i                     = DMBF_CELLMEMIDX_INFO;
  elements[i * dim]     = sizeof(DM_BF_Cell);
  elements[i * dim + 1] = 1;
  pad[i]                = 0;
  /* set cell pointers */
  i                     = DMBF_CELLMEMIDX_POINTERS;
  e                     = nData;
  elements[i * dim]     = (0 < e ? sizeof(PetscScalar *) : 0);
  elements[i * dim + 1] = e;
  pad[i]                = _p_bytesAlignPadding(elements[i * dim] * elements[i * dim + 1] + elements[(i - 1) * dim] * elements[(i - 1) * dim + 1]);
  /* set cell data (read/read-write) TODO deprecated */
  i                     = DMBF_CELLMEMIDX_DATAREAD;
  e                     = (size_t)(bf->blockSize[0] * bf->blockSize[1] * bf->blockSize[2] * bf->valsPerElemReadTotal);
  elements[i * dim]     = (0 < e ? sizeof(PetscScalar) : 0);
  elements[i * dim + 1] = e;
  pad[i]                = _p_bytesAlignPadding(elements[i * dim] * elements[i * dim + 1]);
  i                     = DMBF_CELLMEMIDX_DATAREADWRITE;
  e                     = (size_t)(bf->blockSize[0] * bf->blockSize[1] * bf->blockSize[2] * bf->valsPerElemReadWriteTotal);
  elements[i * dim]     = (0 < e ? sizeof(PetscScalar) : 0);
  elements[i * dim + 1] = e;
  pad[i]                = _p_bytesAlignPadding(elements[i * dim] * elements[i * dim + 1]);
  /* set user/void cell data */
  i                     = DMBF_CELLMEMIDX_DATAV;
  elements[i * dim]     = bf->cellDataVSize;
  elements[i * dim + 1] = (0 < bf->cellDataVSize ? 1 : 0);
  pad[i]                = _p_bytesAlignPadding(elements[i * dim] * elements[i * dim + 1]);
  /* set entries pertaining to cell data */
  for (i = nCell; i < nCell + nData; i++) {
    s = e             = sizeof(PetscScalar);
    elements[i * dim] = e;
    for (j = 1; j < dim; j++) {
      e = bf->cellDataShape.list[i - nCell][j - 1];
      if (0 < e) s *= e;
      elements[i * dim + j] = e;
    }
    pad[i] = _p_bytesAlignPadding(s);
  }
  //{ //###DEV###
  //  PetscPrintf(PETSC_COMM_WORLD,"DMBF_CellMemoryShapeSetUp: nCell=%i nData=%i dim=%i\n",nCell,nData,dim);
  //  for (i=0; i<nCell+nData; i++) {
  //    PetscPrintf(PETSC_COMM_WORLD,"  i %i ; el ",i);
  //    for (j=0; j<dim; j++) {
  //      PetscPrintf(PETSC_COMM_WORLD,"%i ",elements[i*dim+j]);
  //    }
  //    PetscPrintf(PETSC_COMM_WORLD,"; pad %i\n",pad[i]);
  //  }
  //}
  /* set up memory shape */
  ierr = DMBFShapeSetUp(&bf->cellMemoryShape, nCell + nData, dim);
  CHKERRQ(ierr);
  ierr = DMBFShapeSet(&bf->cellMemoryShape, elements, pad);
  CHKERRQ(ierr);
  ierr = PetscFree(elements);
  CHKERRQ(ierr);
  ierr = PetscFree(pad);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline size_t _p_cellSize(DM_BF *bf)
{
  return bf->cellMemoryShape.size;
}

static inline DM_BF_Cell *_p_getCellPtrFromIndex(DM_BF *bf, PetscInt index)
{
  return (DM_BF_Cell *)(((char *)bf->cells) + bf->cellMemoryShape.size * ((size_t)index));
}

/***************************************
 * SETUP
 **************************************/

static PetscErrorCode DMBF_CellsCreate(DM dm)
{
  DM_BF         *bf;
  PetscInt       dim, n, ng;
  size_t         n_cells, ng_cells, cellSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  /* set cell memory shape */
  ierr = DMBF_CellMemoryShapeSetUp(dm);
  CHKERRQ(ierr);
  if (_p_nCells(dm)) {
    /* get number of cells and their size */
    ierr = DMBFGetInfo(dm, &dim, &n, PETSC_NULLPTR, &ng);
    CHKERRQ(ierr);
    n_cells  = (size_t)n;
    ng_cells = (size_t)ng;
    cellSize = _p_cellSize(bf);
    /* create DMBF cells */
    ierr = PetscMalloc1((n_cells + ng_cells) * cellSize, &bf->cells);
    CHKERRQ(ierr);
  } else {
    bf->cells = PETSC_NULLPTR;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBF_CellsDestroy(DM dm)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bf = _p_getBF(dm);
  /* destroy DMBF cells */
  if (bf->cells) {
    ierr = PetscFree(bf->cells);
    CHKERRQ(ierr);
  }
  bf->cells = PETSC_NULLPTR;
  /* destroy cell memory shape */
  ierr = DMBFShapeClear(&bf->cellMemoryShape);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBF_CellsSetUpOwned(DM dm)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells || !_p_nCells(dm), _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  /* set data of all owned cells */
  switch (_p_dim(dm)) {
  case 2:
    ierr = DMBF_2D_IterateSetUpCells(dm, bf->cells, &bf->cellMemoryShape);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateSetUpCells(dm, bf->cells, &bf->cellMemoryShape);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  bf->ownedCellsSetUpCalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBF_CellsSetUpGhost(DM dm)
{
  DM_BF         *bf;
  PetscInt       dim, offset_cells, ng_cells, i;
  DM_BF_Cell    *cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells || !_p_nCells(dm), _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  /* set data pointers of all ghost cells */
  ierr = DMBFGetInfo(dm, &dim, &offset_cells, PETSC_NULLPTR, &ng_cells);
  CHKERRQ(ierr);
  for (i = offset_cells; i < (offset_cells + ng_cells); i++) {
    cell = _p_getCellPtrFromIndex(bf, i);
    ierr = DMBFCellInitialize(cell, &bf->cellMemoryShape);
    CHKERRQ(ierr);
  }
  bf->ghostCellsSetUpCalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBF_LocalToGlobalScatterCreate(DM dm, VecScatter *ltog)
{
  DM_BF         *bf;
  PetscInt       dim, n, ng, locDof, bs, blockSize[3] = {1, 1, 1};
  PetscInt      *fromIdx, *toIdx;
  IS             fromIS, toIS;
  Vec            vloc, vglo;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  /* allocate local-to-global indices */
  bf   = _p_getBF(dm);
  ierr = DMBFGetInfo(dm, &dim, &n, PETSC_NULLPTR, &ng);
  CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  bs     = blockSize[0] * blockSize[1] * blockSize[2];
  locDof = (n + ng) * bs;
  if (locDof) {
    ierr = PetscMalloc1(locDof, &fromIdx);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(locDof, &toIdx);
    CHKERRQ(ierr);
  } else {
    fromIdx = PETSC_NULLPTR;
    toIdx   = PETSC_NULLPTR;
  }
  switch (dim) {
  case 2:
    ierr = DMBF_2D_GetLocalToGlobalIndices(dm, (DM_BF_2D_Cells *)bf->ftCells, fromIdx, toIdx);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_GetLocalToGlobalIndices(dm, (DM_BF_3D_Cells *)bf->ftCells, fromIdx, toIdx);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  /* create IS */
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);
  CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, locDof, fromIdx, PETSC_COPY_VALUES, &fromIS);
  CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, locDof, toIdx, PETSC_COPY_VALUES, &toIS);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm, 1, locDof, toIdx, PETSC_COPY_VALUES, &dm->ltogmap);
  CHKERRQ(ierr);
  ierr = PetscFree(fromIdx);
  CHKERRQ(ierr);
  ierr = PetscFree(toIdx);
  CHKERRQ(ierr);
  /* create vectors */
  ierr = DMCreateLocalVector(dm, &vloc);
  CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &vglo);
  CHKERRQ(ierr);
  /* create vec scatter object */
  ierr = VecScatterCreate(vloc, fromIS, vglo, toIS, ltog);
  CHKERRQ(ierr);
  /* destroy */
  ierr = ISDestroy(&fromIS);
  CHKERRQ(ierr);
  ierr = ISDestroy(&toIS);
  CHKERRQ(ierr);
  ierr = VecDestroy(&vloc);
  CHKERRQ(ierr);
  ierr = VecDestroy(&vglo);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBF_LocalToGlobalScatterDestroy(DM dm, VecScatter *ltog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  if (ltog) {
    ierr = VecScatterDestroy(ltog);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSetUp_BF(DM dm)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DMBF_VERBOSE_HI)
  PetscPrintf(PETSC_COMM_WORLD, "%s\n", __func__);
#endif
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf   = _p_getBF(dm);
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  PetscCheck(dim != PETSC_DETERMINE, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Topological dimension has to be set before setup");
  PetscCheck(2 <= dim && dim <= 3, _p_comm(dm), PETSC_ERR_SUP, "DM does not support %" PetscInt_FMT " dimensional domains", dim);
  PetscCheck(!bf->ftTopology, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Topology exists already");
  PetscCheck(!bf->ftCells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells exist already");
  PetscCheck(!bf->ftNodes, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Nodes exist already");
  /* create forest-of-tree topology */
  switch (dim) {
  case 2:
    ierr = DMBF_2D_TopologyCreate(dm, (DM_BF_2D_Topology **)&bf->ftTopology, bf->setUpUserFnAfterP4estTopology);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_TopologyCreate(dm, (DM_BF_3D_Topology **)&bf->ftTopology, bf->setUpUserFnAfterP4estTopology);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscCheck(bf->ftTopology, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Topology does not exist");
  /* create forest-of-tree cells */
  switch (dim) {
  case 2:
    ierr = DMBF_2D_CellsCreate(dm, (DM_BF_2D_Cells **)&bf->ftCells, bf->setUpUserFnAfterP4estCells);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_CellsCreate(dm, (DM_BF_3D_Cells **)&bf->ftCells, bf->setUpUserFnAfterP4estCells);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscCheck(bf->ftCells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  /* create forest-of-tree nodes */
  //TODO create nodes
  /* create and setup DMBF cells */
  ierr = DMBF_CellsCreate(dm);
  CHKERRQ(ierr);
  ierr = DMBF_CellsSetUpOwned(dm);
  CHKERRQ(ierr);
  ierr = DMBF_CellsSetUpGhost(dm);
  CHKERRQ(ierr);
  /* create local-to-global vector scattering info */
  ierr = DMBF_LocalToGlobalScatterCreate(dm, &bf->ltog);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFClear(DM dm)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DMBF_VERBOSE_HI)
  PetscPrintf(PETSC_COMM_WORLD, "%s\n", __func__);
#endif
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf   = _p_getBF(dm);
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  /* destroy forest-of-tree objects (in reverse order of creation) */
  switch (dim) {
  case 2:
    //TODO destroy nodes
    if (bf->ftCells) {
      ierr = DMBF_2D_CellsDestroy(dm, (DM_BF_2D_Cells *)bf->ftCells);
      CHKERRQ(ierr);
    }
    if (bf->ftTopology) {
      ierr = DMBF_2D_TopologyDestroy(dm, (DM_BF_2D_Topology *)bf->ftTopology);
      CHKERRQ(ierr);
    }
    break;
  case 3:
    //TODO destroy nodes
    if (bf->ftCells) {
      ierr = DMBF_3D_CellsDestroy(dm, (DM_BF_3D_Cells *)bf->ftCells);
      CHKERRQ(ierr);
    }
    if (bf->ftTopology) {
      ierr = DMBF_3D_TopologyDestroy(dm, (DM_BF_3D_Topology *)bf->ftTopology);
      CHKERRQ(ierr);
    }
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  bf->ftNodes    = PETSC_NULLPTR;
  bf->ftCells    = PETSC_NULLPTR;
  bf->ftTopology = PETSC_NULLPTR;
  /* destroy DMBF cells */
  ierr = DMBF_CellsDestroy(dm);
  CHKERRQ(ierr);
  /* destroy cell options */
  ierr = DMBFShapeClear(&bf->cellDataShape);
  CHKERRQ(ierr);
  if (bf->valsPerElemRead) {
    ierr = PetscFree(bf->valsPerElemRead);
    CHKERRQ(ierr);
  }
  if (bf->valsPerElemReadWrite) {
    ierr = PetscFree(bf->valsPerElemReadWrite);
    CHKERRQ(ierr);
  }
  bf->valsPerElemRead           = PETSC_NULLPTR;
  bf->valsPerElemReadWrite      = PETSC_NULLPTR;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;
  /* destroy local-to-global vector scattering info */
  ierr = DMBF_LocalToGlobalScatterDestroy(dm, &bf->ltog);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/***************************************
 * OPTIONS
 **************************************/

PetscErrorCode DMBFSetCellDataShape(DM dm, const PetscInt *shapeElements, PetscInt n, PetscInt dim)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(shapeElements, 2);
  PetscCheck(!dm->setupcalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cannot change cell data after setup");
  bf = _p_getBF(dm);
  /* set settings */
  if (0 < n && 0 < dim) {
    ierr = DMBFShapeSetUp(&bf->cellDataShape, (size_t)n, (size_t)dim);
    CHKERRQ(ierr);
    ierr = DMBFShapeSetFromInt(&bf->cellDataShape, shapeElements);
    CHKERRQ(ierr);
  } else {
    ierr = DMBFShapeClear(&bf->cellDataShape);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFGetCellDataShape(DM dm, PetscInt **shapeElements, PetscInt *n, PetscInt *dim)
{
  DM_BF         *bf;
  PetscBool      isSetUp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(shapeElements, 2);
  PetscAssertPointer(n, 3);
  PetscAssertPointer(dim, 4);
  bf = _p_getBF(dm);
  /* get settings */
  ierr = DMBFShapeIsSetUp(&bf->cellDataShape, &isSetUp);
  CHKERRQ(ierr);
  if (isSetUp) {
    ierr = DMBFShapeGetToInt(&bf->cellDataShape, shapeElements, n, dim);
    CHKERRQ(ierr);
  } else {
    *shapeElements = PETSC_NULLPTR;
    *n             = 0;
    *dim           = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFSetCellDataVSize(DM dm, size_t size)
{
  DM_BF *bf;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscCheck(!dm->setupcalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cannot change cell data after setup");
  bf = _p_getBF(dm);
  /* set setting */
  if (0 < size) {
    bf->cellDataVSize = size;
  } else {
    bf->cellDataVSize = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFGetCellDataVSize(DM dm, size_t *size)
{
  DM_BF *bf;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(size, 2);
  bf = _p_getBF(dm);
  /* get setting */
  *size = bf->cellDataVSize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMBFSetBlockSize - During the pre-setup phase, set the levels of uniform block refinement of each cell in each dimension.

  Logically collective on dm

  Input Parameters:
+ dm        - the DMBF object
- blockSize - levels of uniform block refinement of each cell in each dimension

  Level: intermediate

.seealso: `DMBFGetBlockSize()`, `DMGetDimension()`
@*/
PetscErrorCode DMBFSetBlockSize(DM dm, PetscInt *blockSize)
{
  DM_BF         *bf;
  PetscInt       dim, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(blockSize, 2);
  PetscCheck(!dm->setupcalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the block refinement after setup");
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  PetscCheck(dim != PETSC_DETERMINE, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cannot set block refinement before topological dimension");
  bf = _p_getBF(dm);
  for (i = 0; i < dim; i++) { bf->blockSize[i] = (1 <= blockSize[i] ? blockSize[i] : 1); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMBFGetBlockSize - Get the levels of uniform block refinement of each cell in each dimension.

  Logically collective on dm

  Input Parameters:
+ dm        - the DMBF object
- blockSize - levels of uniform block refinement of each cell in each dimension

  Level: intermediate

.seealso: `DMBFSetBlockSize()`, `DMGetDimension()`
@*/
PetscErrorCode DMBFGetBlockSize(DM dm, PetscInt *blockSize)
{
  DM_BF         *bf;
  PetscInt       dim, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(blockSize, 2);
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  PetscCheck(dim != PETSC_DETERMINE, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Topological dimension has to be set for block refinement");
  bf = _p_getBF(dm);
  for (i = 0; i < dim; i++) { blockSize[i] = bf->blockSize[i]; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFSetCellDataSize(DM dm, PetscInt *valsPerElemRead, PetscInt nValsPerElemRead, PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite)
{
  DM_BF         *bf;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(valsPerElemRead, 2);
  PetscAssertPointer(valsPerElemReadWrite, 4);
  PetscCheck(!dm->setupcalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cannot change cell data after setup");
  bf = _p_getBF(dm);
  /* reset exising settings */
  if (!bf->valsPerElemRead) {
    ierr = PetscFree(bf->valsPerElemRead);
    CHKERRQ(ierr);
  }
  if (!bf->valsPerElemReadWrite) {
    ierr = PetscFree(bf->valsPerElemReadWrite);
    CHKERRQ(ierr);
  }
  bf->valsPerElemRead           = PETSC_NULLPTR;
  bf->valsPerElemReadWrite      = PETSC_NULLPTR;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;
  /* set new settings */
  if (0 < nValsPerElemRead) {
    bf->nValsPerElemRead = nValsPerElemRead;
    ierr                 = PetscMalloc1(bf->nValsPerElemRead, &bf->valsPerElemRead);
    CHKERRQ(ierr);
    for (i = 0; i < bf->nValsPerElemRead; i++) {
      bf->valsPerElemRead[i] = valsPerElemRead[i];
      bf->valsPerElemReadTotal += valsPerElemRead[i];
    }
  }
  if (0 < nValsPerElemReadWrite) {
    bf->nValsPerElemReadWrite = nValsPerElemReadWrite;
    ierr                      = PetscMalloc1(bf->nValsPerElemReadWrite, &bf->valsPerElemReadWrite);
    CHKERRQ(ierr);
    for (i = 0; i < bf->nValsPerElemReadWrite; i++) {
      bf->valsPerElemReadWrite[i] = valsPerElemReadWrite[i];
      bf->valsPerElemReadWriteTotal += valsPerElemReadWrite[i];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFGetCellDataSize(DM dm, PetscInt **valsPerElemRead, PetscInt *nValsPerElemRead, PetscInt **valsPerElemReadWrite, PetscInt *nValsPerElemReadWrite)
{
  DM_BF         *bf;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  /* PetscAssertPointer(valsPerElemRead,3); */
  /* PetscAssertPointer(valsPerElemReadWrite,5); */
  bf = _p_getBF(dm);
  if (nValsPerElemRead) { *nValsPerElemRead = bf->nValsPerElemRead; }
  if (0 < bf->nValsPerElemRead && valsPerElemRead) {
    ierr = PetscMalloc1(bf->nValsPerElemRead, valsPerElemRead);
    CHKERRQ(ierr);
    for (i = 0; i < bf->nValsPerElemRead; i++) { (*valsPerElemRead)[i] = bf->valsPerElemRead[i]; }
  }
  if (nValsPerElemReadWrite) { *nValsPerElemReadWrite = bf->nValsPerElemReadWrite; }
  if (0 < bf->nValsPerElemReadWrite && valsPerElemReadWrite) {
    ierr = PetscMalloc1(bf->nValsPerElemReadWrite, valsPerElemReadWrite);
    CHKERRQ(ierr);
    for (i = 0; i < bf->nValsPerElemReadWrite; i++) { (*valsPerElemReadWrite)[i] = bf->valsPerElemReadWrite[i]; }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFSetDefaultOptions(DM dm)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);

  ierr = DMSetDimension(dm, 2);
  CHKERRQ(ierr);
  ierr = DMSetVecType(dm, VECSTANDARD);
  CHKERRQ(ierr);
  ierr = DMSetMatType(dm, MATSHELL);
  CHKERRQ(ierr);

  ierr = DMForestSetTopology(dm, "unit");
  CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(dm, 0);
  CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(dm, 0);
  CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(dm, 18);
  CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(dm, 2);
  CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm, 0);
  CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(dm, 0);
  CHKERRQ(ierr);

  bf = _p_getBF(dm);

  bf->ftTopology = PETSC_NULLPTR;
  bf->ftCells    = PETSC_NULLPTR;
  bf->ftNodes    = PETSC_NULLPTR;

  bf->cells                 = PETSC_NULLPTR;
  bf->ownedCellsSetUpCalled = PETSC_FALSE;
  bf->ghostCellsSetUpCalled = PETSC_FALSE;

  ierr = DMBFShapeClear(&bf->cellMemoryShape);
  CHKERRQ(ierr);
  ierr = DMBFShapeClear(&bf->cellDataShape);
  CHKERRQ(ierr);

  bf->cellDataVSize = 0;

  bf->blockSize[0] = 1;
  bf->blockSize[1] = 1;
  bf->blockSize[2] = 1;

  bf->valsPerElemRead           = PETSC_NULLPTR;
  bf->valsPerElemReadWrite      = PETSC_NULLPTR;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;

  bf->setUpUserFnAfterP4estTopology = PETSC_NULLPTR;
  bf->setUpUserFnAfterP4estCells    = PETSC_NULLPTR;
  bf->setUpUserFnAfterP4estNodes    = PETSC_NULLPTR;

  bf->amrOps = PETSC_NULLPTR;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFCopyOptions(DM srcdm, DM trgdm)
{
  DM_BF         *srcbf, *trgbf;
  PetscInt       dim;
  VecType        vecType;
  MatType        matType;
  PetscBool      isSetUp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(srcdm, DM_CLASSID, 1, DMBF);
  PetscValidHeaderSpecificType(trgdm, DM_CLASSID, 2, DMBF);

  ierr = DMGetDimension(srcdm, &dim);
  CHKERRQ(ierr);
  ierr = DMSetDimension(trgdm, dim);
  CHKERRQ(ierr);
  ierr = DMGetVecType(srcdm, &vecType);
  CHKERRQ(ierr);
  ierr = DMSetVecType(trgdm, vecType);
  CHKERRQ(ierr);
  ierr = DMGetMatType(srcdm, &matType);
  CHKERRQ(ierr);
  ierr = DMSetMatType(trgdm, matType);
  CHKERRQ(ierr);

  /* Note: Assume options via `DMForest[GS]et<Name>()` were copied already. */

  srcbf = _p_getBF(srcdm);
  trgbf = _p_getBF(trgdm);

  trgbf->ftTopology = PETSC_NULLPTR;
  trgbf->ftCells    = PETSC_NULLPTR;
  trgbf->ftNodes    = PETSC_NULLPTR;

  trgbf->cells                 = PETSC_NULLPTR;
  trgbf->ownedCellsSetUpCalled = PETSC_FALSE;
  trgbf->ghostCellsSetUpCalled = PETSC_FALSE;

  ierr = DMBFShapeIsSetUp(&srcbf->cellMemoryShape, &isSetUp);
  CHKERRQ(ierr);
  if (isSetUp) {
    ierr = DMBFShapeSetUp(&trgbf->cellMemoryShape, srcbf->cellMemoryShape.n, srcbf->cellMemoryShape.dim);
    CHKERRQ(ierr);
    ierr = DMBFShapeCopy(&trgbf->cellMemoryShape, &srcbf->cellMemoryShape);
    CHKERRQ(ierr);
  } else {
    ierr = DMBFShapeClear(&trgbf->cellMemoryShape);
    CHKERRQ(ierr);
  }

  ierr = DMBFShapeIsSetUp(&srcbf->cellDataShape, &isSetUp);
  CHKERRQ(ierr);
  if (isSetUp) {
    ierr = DMBFShapeSetUp(&trgbf->cellDataShape, srcbf->cellDataShape.n, srcbf->cellDataShape.dim);
    CHKERRQ(ierr);
    ierr = DMBFShapeCopy(&trgbf->cellDataShape, &srcbf->cellDataShape);
    CHKERRQ(ierr);
  } else {
    ierr = DMBFShapeClear(&trgbf->cellDataShape);
    CHKERRQ(ierr);
  }

  trgbf->cellDataVSize = srcbf->cellDataVSize;

  trgbf->blockSize[0] = srcbf->blockSize[0];
  trgbf->blockSize[1] = srcbf->blockSize[1];
  trgbf->blockSize[2] = srcbf->blockSize[2];

  if (0 < srcbf->nValsPerElemRead) {
    ierr = PetscMalloc1(srcbf->nValsPerElemRead, &trgbf->valsPerElemRead);
    CHKERRQ(ierr);
    ierr = PetscArraycpy(trgbf->valsPerElemRead, srcbf->valsPerElemRead, srcbf->nValsPerElemRead);
    CHKERRQ(ierr);
  }
  if (0 < srcbf->nValsPerElemReadWrite) {
    ierr = PetscMalloc1(srcbf->nValsPerElemReadWrite, &trgbf->valsPerElemReadWrite);
    CHKERRQ(ierr);
    ierr = PetscArraycpy(trgbf->valsPerElemReadWrite, srcbf->valsPerElemReadWrite, srcbf->nValsPerElemReadWrite);
    CHKERRQ(ierr);
  }
  trgbf->nValsPerElemRead          = srcbf->nValsPerElemRead;
  trgbf->nValsPerElemReadWrite     = srcbf->nValsPerElemReadWrite;
  trgbf->valsPerElemReadTotal      = srcbf->valsPerElemReadTotal;
  trgbf->valsPerElemReadWriteTotal = srcbf->valsPerElemReadWriteTotal;

  trgbf->setUpUserFnAfterP4estTopology = srcbf->setUpUserFnAfterP4estTopology;
  trgbf->setUpUserFnAfterP4estCells    = srcbf->setUpUserFnAfterP4estCells;
  trgbf->setUpUserFnAfterP4estNodes    = srcbf->setUpUserFnAfterP4estNodes;

  trgbf->amrOps = srcbf->amrOps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSetFromOptions_BF(DM dm, PetscOptionItems *PetscOptionsObject)
{
  PetscInt       blockSize[3], blockDim = 3;
  PetscBool      set;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  ierr = DMSetFromOptions_Forest(dm, PetscOptionsObject);
  CHKERRQ(ierr);
  /* block_size */
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-dm_bf_block_size", "set uniform refinement inside each cell in each dimension x,y,z", "DMBFSetBlockSize", blockSize, &blockDim, &set);
  CHKERRQ(ierr);
  if (set) {
    //TODO if (blockDim != dim)
    ierr = DMBFSetBlockSize(dm, blockSize);
    CHKERRQ(ierr);
  }
  //TODO
  //char              stringBuffer[256];
  //ierr = PetscOptionsHead(PetscOptionsObject,"DM" P4EST_STRING " options");CHKERRQ(ierr);
  //ierr = PetscOptionsBool("-dm_p4est_partition_for_coarsening","partition forest to allow for coarsening","DMP4estSetPartitionForCoarsening",pforest->partition_for_coarsening,&(pforest->partition_for_coarsening),NULL);CHKERRQ(ierr);
  //ierr = PetscOptionsString("-dm_p4est_ghost_label_name","the name of the ghost label when converting from a DMPlex",NULL,NULL,stringBuffer,sizeof(stringBuffer),&flg);CHKERRQ(ierr);
  //ierr = PetscOptionsTail();CHKERRQ(ierr);
  //if (flg) {
  //  ierr = PetscFree(pforest->ghostName);CHKERRQ(ierr);
  //  ierr = PetscStrallocpy(stringBuffer,&pforest->ghostName);CHKERRQ(ierr);
  //}
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFSetUpUserFnAfterP4estTopology(DM dm, PetscErrorCode (*fn)(DM, void *))
{
  DM_BF *bf;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf                                = _p_getBF(dm);
  bf->setUpUserFnAfterP4estTopology = fn;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFSetUpUserFnAfterP4estCells(DM dm, PetscErrorCode (*fn)(DM, void *))
{
  DM_BF *bf;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf                             = _p_getBF(dm);
  bf->setUpUserFnAfterP4estCells = fn;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFSetUpUserFnAfterP4estNodes(DM dm, PetscErrorCode (*fn)(DM, void *))
{
  DM_BF *bf;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf                             = _p_getBF(dm);
  bf->setUpUserFnAfterP4estNodes = fn;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/***************************************
 * CREATE/DESTROY
 **************************************/

static PetscErrorCode DMBFSetOps(DM dm)
{
  PetscFunctionBegin;
  dm->ops->setup          = DMSetUp_BF;
  dm->ops->setfromoptions = DMSetFromOptions_BF;
  dm->ops->clone          = DMClone_BF;
  dm->ops->view           = DMView_BF;

  dm->ops->createlocalvector  = DMCreateLocalVector_BF;
  dm->ops->createglobalvector = DMCreateGlobalVector_BF;
  dm->ops->creatematrix       = DMCreateMatrix_BF;

  dm->ops->coarsen = DMCoarsen_BF;
  dm->ops->refine  = DMRefine_BF;

  dm->ops->globaltolocalbegin = DMGlobalToLocalBegin_BF;
  dm->ops->globaltolocalend   = DMGlobalToLocalEnd_BF;
  dm->ops->localtoglobalbegin = DMLocalToGlobalBegin_BF;
  dm->ops->localtoglobalend   = DMLocalToGlobalEnd_BF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreate_BF(DM dm)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DMBF_VERBOSE_HI)
  PetscPrintf(PETSC_COMM_WORLD, "%s\n", __func__);
#endif
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  /* create Forest object */
  ierr = PetscP4estInitialize();
  CHKERRQ(ierr);
  ierr = DMCreate_Forest(dm);
  CHKERRQ(ierr);
  /* create BF object */
  ierr = PetscNew(&bf);
  CHKERRQ(ierr);
  /* set data and functions of Forest object */
  {
    DM_Forest *forest = (DM_Forest *)dm->data;

    forest->data    = bf;
    forest->destroy = DMForestDestroy_BF;
  }
  /* set operators */
  ierr = DMBFSetOps(dm);
  CHKERRQ(ierr);
  /* set default options */
  ierr = DMBFSetDefaultOptions(dm);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMForestDestroy_BF(DM dm)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DMBF_VERBOSE_HI)
  PetscPrintf(PETSC_COMM_WORLD, "%s\n", __func__);
#endif
  /* destroy contents of BF */
  ierr = DMBFClear(dm);
  CHKERRQ(ierr);
  /* destroy BF object */
  bf   = _p_getBF(dm);
  ierr = PetscFree(bf);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFCloneInit(DM dm, DM *newdm)
{
  DM_BF         *newbf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* clone Forest object (will implicitly call DMCreate_BF) */
  ierr = DMForestTemplate(dm, _p_comm(dm), newdm);
  CHKERRQ(ierr);
  /* check BF object */
  newbf = _p_getBF(*newdm);
  PetscCheck(newbf, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "BF object does not exist");
  /* set operators */
  ierr = DMBFSetOps(*newdm);
  CHKERRQ(ierr);
  /* copy options */
  ierr = DMBFCopyOptions(dm, *newdm);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFCloneForestOfTrees(DM dm, DM newdm)
{
  DM_BF         *bf, *newbf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bf    = _p_getBF(dm);
  newbf = _p_getBF(newdm);
  ierr  = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_TopologyClone((DM_BF_2D_Topology *)bf->ftTopology, (DM_BF_2D_Topology **)&newbf->ftTopology, newdm);
    CHKERRQ(ierr);
    ierr = DMBF_2D_CellsClone((DM_BF_2D_Cells *)bf->ftCells, (DM_BF_2D_Cells **)&newbf->ftCells, newdm);
    CHKERRQ(ierr);
    //TODO clone nodes
    break;
  case 3:
    ierr = DMBF_3D_TopologyClone((DM_BF_3D_Topology *)bf->ftTopology, (DM_BF_3D_Topology **)&newbf->ftTopology, newdm);
    CHKERRQ(ierr);
    ierr = DMBF_3D_CellsClone((DM_BF_3D_Cells *)bf->ftCells, (DM_BF_3D_Cells **)&newbf->ftCells, newdm);
    CHKERRQ(ierr);
    //TODO clone nodes
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFCloneFinalize(DM newdm)
{
  DM_BF         *newbf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* create and setup DMBF cells */
  ierr = DMBF_CellsCreate(newdm);
  CHKERRQ(ierr); //TODO create a clone fnc instead?
  ierr = DMBF_CellsSetUpOwned(newdm);
  CHKERRQ(ierr);
  ierr = DMBF_CellsSetUpGhost(newdm);
  CHKERRQ(ierr);
  /* create local-to-global vector scattering info */
  newbf = _p_getBF(newdm);
  ierr  = DMBF_LocalToGlobalScatterCreate(newdm, &newbf->ltog);
  CHKERRQ(ierr); //TODO create a clone fnc instead?
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMClone_BF(DM dm, DM *newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DMBF_VERBOSE_HI)
  PetscPrintf(PETSC_COMM_WORLD, "%s\n", __func__);
#endif
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  ierr = DMBFCloneInit(dm, newdm);
  CHKERRQ(ierr);
  ierr = DMBFCloneForestOfTrees(dm, *newdm);
  CHKERRQ(ierr);
  ierr = DMBFCloneFinalize(*newdm);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO need to use DMGetVecType and; add DMSetVecType -> STANDARD to be default for the DM
static PetscErrorCode DMCreateLocalVector_BF(DM dm, Vec *vec)
{
  PetscInt       blockSize[3] = {1, 1, 1};
  PetscInt       dim, n, ng, i;
  PetscInt       locDof, cellDof = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(vec, 2);
  /* get number of entries */
  ierr = DMBFGetInfo(dm, &dim, &n, PETSC_NULLPTR, &ng);
  CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  for (i = 0; i < dim; i++) { cellDof *= blockSize[i]; }
  locDof = cellDof * (n + ng);
  /* create vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF, locDof, vec);
  CHKERRQ(ierr);
  ierr = VecSetDM(*vec, dm);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO need to use DMGetVecType and; add DMSetVecType -> STANDARD to be default for the DM
static PetscErrorCode DMCreateGlobalVector_BF(DM dm, Vec *vec)
{
  PetscInt       blockSize[3] = {1, 1, 1};
  PetscInt       dim, n, N, i;
  PetscInt       locDof, gloDof, cellDof = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(vec, 2);
  /* get number of entries */
  ierr = DMBFGetInfo(dm, &dim, &n, &N, PETSC_NULLPTR);
  CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  for (i = 0; i < dim; i++) { cellDof *= blockSize[i]; }
  locDof = cellDof * n;
  gloDof = cellDof * N;
  /* create vector */
  ierr = VecCreateMPI(_p_comm(dm), locDof, gloDof, vec);
  CHKERRQ(ierr);
  ierr = VecSetDM(*vec, dm);
  CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_VIEW, (void (*)(void))VecView_BF);
  CHKERRQ(ierr);
  //TODO
  //ierr = VecSetOperation(*g,VECOP_VIEW,(void (*)(void))VecView_MPI_DA);CHKERRQ(ierr);
  //ierr = VecSetOperation(*vec, VECOP_VIEWNATIVE, (void (*)(void))VecView_pforest_Native);CHKERRQ(ierr);
  //ierr = VecSetOperation(*g,VECOP_LOAD,(void (*)(void))VecLoad_Default_DA);CHKERRQ(ierr);
  //ierr = VecSetOperation(*vec, VECOP_LOADNATIVE, (void (*)(void))VecLoad_pforest_Native);CHKERRQ(ierr);
  //ierr = VecSetOperation(*g,VECOP_DUPLICATE,(void (*)(void))VecDuplicate_MPI_DA);CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateMatrix_BF(DM dm, Mat *mat)
{
  PetscInt       blockSize[3] = {1, 1, 1};
  PetscInt       dim, n, N, i;
  PetscInt       locDof, gloDof, cellDof = 1;
  MatType        mattype;
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(mat, 2);
  /* get number of rows/cols */
  ierr = DMBFGetInfo(dm, &dim, &n, &N, PETSC_NULLPTR);
  CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  for (i = 0; i < dim; i++) { cellDof *= blockSize[i]; }
  locDof = cellDof * n;
  gloDof = cellDof * N;
  /* create matrix */
  ierr = MatCreate(_p_comm(dm), mat);
  CHKERRQ(ierr);
  ierr = MatSetSizes(*mat, locDof, locDof, gloDof, gloDof);
  CHKERRQ(ierr);
  ierr = MatSetBlockSize(*mat, 1 /*blocksize*/);
  CHKERRQ(ierr);
  /* set type */
  ierr = DMGetMatType(dm, &mattype);
  CHKERRQ(ierr);
  ierr = MatSetType(*mat, mattype);
  CHKERRQ(ierr);
  /* set mapping */
  ierr = MatSetLocalToGlobalMapping(*mat, dm->ltogmap, dm->ltogmap);
  CHKERRQ(ierr);
  /* set context */
  ierr = MatSetDM(*mat, dm);
  CHKERRQ(ierr);
  ierr = PetscStrcmp(mattype, MATSHELL, &match);
  CHKERRQ(ierr);
  if (match) {
    void *appctx;
    ierr = DMGetApplicationContext(dm, &appctx);
    CHKERRQ(ierr);
    ierr = MatShellSetContext(*mat, appctx);
    CHKERRQ(ierr);
  }
  ierr = MatSetUp(*mat);
  CHKERRQ(ierr);
  //TODO set null space?
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* take global vector and return local version */
static PetscErrorCode DMGlobalToLocalBegin_BF(DM dm, Vec glo, InsertMode mode, Vec loc)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidHeaderSpecific(glo, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(loc, VEC_CLASSID, 4);

  bf   = _p_getBF(dm);
  ierr = VecScatterBegin(bf->ltog, glo, loc, mode, SCATTER_REVERSE);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGlobalToLocalEnd_BF(DM dm, Vec glo, InsertMode mode, Vec loc)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidHeaderSpecific(glo, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(loc, VEC_CLASSID, 4);

  bf   = _p_getBF(dm);
  ierr = VecScatterEnd(bf->ltog, glo, loc, mode, SCATTER_REVERSE);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLocalToGlobalBegin_BF(DM dm, Vec loc, InsertMode mode, Vec glo)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidHeaderSpecific(loc, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(glo, VEC_CLASSID, 4);

  bf   = _p_getBF(dm);
  ierr = VecScatterBegin(bf->ltog, loc, glo, mode, SCATTER_FORWARD);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLocalToGlobalEnd_BF(DM dm, Vec loc, InsertMode mode, Vec glo)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidHeaderSpecific(loc, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(glo, VEC_CLASSID, 4);

  bf   = _p_getBF(dm);
  ierr = VecScatterEnd(bf->ltog, loc, glo, mode, SCATTER_FORWARD);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/***************************************
 * MESH
 **************************************/

/*@
  DMBFGetInfo - Gets information about the DM.

  Not Collective

  Input Parameter:
. dm - the DMBF object

  Output Parameters:
+ dim     - spatial dimension (2 or 3)
. nLocal  - number of local cells
. nGlobal - number of global cells
- nGhost  - number of ghost cells

  Level: beginner

.seealso: `DMGetDimension()`, `DMBFGetLocalSize()`, `DMBFGetGlobalSize()`, `DMBFGetGhostSize()`
@*/
PetscErrorCode DMBFGetInfo(DM dm, PetscInt *dim, PetscInt *nLocal, PetscInt *nGlobal, PetscInt *nGhost)
{
  DM_BF         *bf;
  PetscInt       n, ng, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscAssertPointer(dim, 2);
  bf = _p_getBF(dm);
  PetscCheck(bf->ftCells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Forest-of-tree cells do not exist");
  ierr = DMGetDimension(dm, dim);
  CHKERRQ(ierr);
  switch (*dim) {
  case 2:
    ierr = DMBF_2D_GetSizes(dm, (DM_BF_2D_Cells *)bf->ftCells, &n, &N, &ng);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_GetSizes(dm, (DM_BF_3D_Cells *)bf->ftCells, &n, &N, &ng);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  if (nLocal) *nLocal = n;
  if (nGlobal) *nGlobal = N;
  if (nGhost) *nGhost = ng;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMBFGetLocalSize - Gets local number of quadrants in the forest.

  Not Collective

  Input Parameters:
. dm - the DMBF object

  Output Parameters:
. nLocal - number of local cells (does not count ghost cells)

  Level: beginner

.seealso: `DMBFGetInfo()`, `DMBFGetGlobalSize()`, `DMBFGetGhostSize()`
@*/
PetscErrorCode DMBFGetLocalSize(DM dm, PetscInt *nLocal)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm, &dim, nLocal, PETSC_NULLPTR, PETSC_NULLPTR);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMBFGetGlobalSize - Gets global number of quadrants in the forest.

  Logically collective on DM

  Input Parameters:
. dm - the DMBF object

  Output Parameters:
. nGlobal - number of global cells

  Level: beginner

.seealso: `DMBFGetInfo()`, `DMBFGetLocalSize()`, `DMBFGetGhostSize()`
@*/
PetscErrorCode DMBFGetGlobalSize(DM dm, PetscInt *nGlobal)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm, &dim, PETSC_NULLPTR, nGlobal, PETSC_NULLPTR);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMBFGetGhostSize - Gets number of quadrants in the ghost layer.

  Not Collective

  Input Parameters:
. dm - the DMBF object

  Output Parameters:
. nGhost - number of ghost cells

  Level: beginner

.seealso: `DMBFGetInfo()`, `DMBFGetLocalSize()`, `DMBFGetGlobalSize()`
@*/
PetscErrorCode DMBFGetGhostSize(DM dm, PetscInt *nGhost)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm, &dim, PETSC_NULLPTR, PETSC_NULLPTR, nGhost);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/***************************************
 * P4EST
 **************************************/

PetscErrorCode DMBFGetConnectivity(DM dm, void *connectivity)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->ftTopology, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Topology does not exist");
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_TopologyGetConnectivity((DM_BF_2D_Topology *)bf->ftTopology, connectivity);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_TopologyGetConnectivity((DM_BF_3D_Topology *)bf->ftTopology, connectivity);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFGetP4est(DM dm, void *p4est)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->ftCells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_CellsGetP4est((DM_BF_2D_Cells *)bf->ftCells, p4est);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_CellsGetP4est((DM_BF_3D_Cells *)bf->ftCells, p4est);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFGetGhost(DM dm, void *ghost)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->ftCells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_CellsGetGhost((DM_BF_2D_Cells *)bf->ftCells, ghost);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_CellsGetGhost((DM_BF_3D_Cells *)bf->ftCells, ghost);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/***************************************
 * AMR
 **************************************/

static PetscErrorCode DMCoarsen_BF(DM dm, MPI_Comm comm, DM *coarseDm)
{
  DM_BF         *bf, *coarsebf;
  PetscInt       minLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DMBF_VERBOSE_HI)
  PetscPrintf(PETSC_COMM_WORLD, "%s\n", __func__);
#endif
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  CHKERRQ(DMBFCheck(dm));
  {
    PetscMPIInt mpiComparison;
    MPI_Comm    dmcomm = _p_comm(dm);

    ierr = MPI_Comm_compare(comm, dmcomm, &mpiComparison);
    CHKERRQ(ierr);
    PetscCheck(mpiComparison == MPI_IDENT || mpiComparison == MPI_CONGRUENT, dmcomm, PETSC_ERR_SUP, "No support for different communicators");
  }
  ierr = DMBFCloneInit(dm, coarseDm);
  CHKERRQ(ierr);
  ierr = DMForestGetMinimumRefinement(*coarseDm, &minLevel);
  CHKERRQ(ierr);
  bf       = _p_getBF(dm);
  coarsebf = _p_getBF(*coarseDm);
  switch (_p_dim(dm)) {
  case 2:
    ierr = DMBF_2D_TopologyClone((DM_BF_2D_Topology *)bf->ftTopology, (DM_BF_2D_Topology **)&coarsebf->ftTopology, *coarseDm);
    CHKERRQ(ierr);
    ierr = DMBF_2D_CellsCoarsen((DM_BF_2D_Cells *)bf->ftCells, (DM_BF_2D_Cells **)&coarsebf->ftCells, *coarseDm, minLevel);
    CHKERRQ(ierr);
    //TODO clone nodes
    break;
  case 3:
    ierr = DMBF_3D_TopologyClone((DM_BF_3D_Topology *)bf->ftTopology, (DM_BF_3D_Topology **)&coarsebf->ftTopology, *coarseDm);
    CHKERRQ(ierr);
    ierr = DMBF_3D_CellsCoarsen((DM_BF_3D_Cells *)bf->ftCells, (DM_BF_3D_Cells **)&coarsebf->ftCells, *coarseDm, minLevel);
    CHKERRQ(ierr);
    //TODO clone nodes
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  ierr = DMBFCloneFinalize(*coarseDm);
  CHKERRQ(ierr);
  CHKERRQ(DMBFCheck(*coarseDm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMRefine_BF(DM dm, MPI_Comm comm, DM *fineDm)
{
  DM_BF         *bf, *finebf;
  PetscInt       maxLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DMBF_VERBOSE_HI)
  PetscPrintf(PETSC_COMM_WORLD, "%s\n", __func__);
#endif
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  CHKERRQ(DMBFCheck(dm));
  {
    PetscMPIInt mpiComparison;
    MPI_Comm    dmcomm = _p_comm(dm);

    ierr = MPI_Comm_compare(comm, dmcomm, &mpiComparison);
    CHKERRQ(ierr);
    PetscCheck(mpiComparison == MPI_IDENT || mpiComparison == MPI_CONGRUENT, dmcomm, PETSC_ERR_SUP, "No support for different communicators");
  }
  ierr = DMBFCloneInit(dm, fineDm);
  CHKERRQ(ierr);
  ierr = DMForestGetMaximumRefinement(*fineDm, &maxLevel);
  CHKERRQ(ierr);
  bf     = _p_getBF(dm);
  finebf = _p_getBF(*fineDm);
  switch (_p_dim(dm)) {
  case 2:
    ierr = DMBF_2D_TopologyClone((DM_BF_2D_Topology *)bf->ftTopology, (DM_BF_2D_Topology **)&finebf->ftTopology, *fineDm);
    CHKERRQ(ierr);
    ierr = DMBF_2D_CellsRefine((DM_BF_2D_Cells *)bf->ftCells, (DM_BF_2D_Cells **)&finebf->ftCells, *fineDm, maxLevel);
    CHKERRQ(ierr);
    //TODO clone nodes
    break;
  case 3:
    ierr = DMBF_3D_TopologyClone((DM_BF_3D_Topology *)bf->ftTopology, (DM_BF_3D_Topology **)&finebf->ftTopology, *fineDm);
    CHKERRQ(ierr);
    ierr = DMBF_3D_CellsRefine((DM_BF_3D_Cells *)bf->ftCells, (DM_BF_3D_Cells **)&finebf->ftCells, *fineDm, maxLevel);
    CHKERRQ(ierr);
    //TODO clone nodes
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  ierr = DMBFCloneFinalize(*fineDm);
  CHKERRQ(ierr);
  CHKERRQ(DMBFCheck(*fineDm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFAMRSetOperators(DM dm, DM_BF_AmrOps *amrOps)
{
  DM_BF *bf;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf         = _p_getBF(dm);
  bf->amrOps = amrOps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFAMRFlag(DM dm)
{
  DM_BF         *bf;
  PetscInt       dim, n_cells, i;
  DM_BF_Cell    *cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  CHKERRQ(DMBFCheck(dm));
  bf = _p_getBF(dm);
  PetscCheck(bf->amrOps, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "AMR operators do not exist");
  /* set data pointers of all ghost cells */
  ierr = DMBFGetInfo(dm, &dim, &n_cells, PETSC_NULLPTR, PETSC_NULLPTR);
  CHKERRQ(ierr);
  for (i = 0; i < n_cells; i++) {
    cell = _p_getCellPtrFromIndex(bf, i);
    ierr = bf->amrOps->setAmrFlag(dm, cell, bf->amrOps->setAmrFlagCtx);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFAMRAdapt(DM dm, DM *adaptedDm)
{
  DM_BF         *bf, *adaptedbf;
  const PetscInt dim = _p_dim(dm);
  PetscInt       minLevel, maxLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_DMBF_VERBOSE_HI)
  PetscPrintf(PETSC_COMM_WORLD, "%s\n", __func__);
#endif
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  CHKERRQ(DMBFCheck(dm));
  ierr = DMBFCloneInit(dm, adaptedDm);
  CHKERRQ(ierr);
  ierr = DMForestGetMinimumRefinement(*adaptedDm, &minLevel);
  CHKERRQ(ierr);
  ierr = DMForestGetMaximumRefinement(*adaptedDm, &maxLevel);
  CHKERRQ(ierr);
  bf        = _p_getBF(dm);
  adaptedbf = _p_getBF(*adaptedDm);
  /* adapt and partition forest-of-tree cells */
  PetscCheck(bf->amrOps, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "AMR operators do not exist");
  switch (dim) {
  case 2:
    ierr = DMBF_2D_TopologyClone((DM_BF_2D_Topology *)bf->ftTopology, (DM_BF_2D_Topology **)&adaptedbf->ftTopology, *adaptedDm);
    CHKERRQ(ierr);
    ierr = DMBF_2D_CellsAmrAdapt((DM_BF_2D_Cells *)bf->ftCells, (DM_BF_2D_Cells **)&adaptedbf->ftCells, *adaptedDm, bf->amrOps, minLevel, maxLevel, &bf->cellMemoryShape);
    CHKERRQ(ierr);
    ierr = DMBF_2D_CellsAmrPartition((DM_BF_2D_Cells *)adaptedbf->ftCells);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_TopologyClone((DM_BF_3D_Topology *)bf->ftTopology, (DM_BF_3D_Topology **)&adaptedbf->ftTopology, *adaptedDm);
    CHKERRQ(ierr);
    ierr = DMBF_3D_CellsAmrAdapt((DM_BF_3D_Cells *)bf->ftCells, (DM_BF_3D_Cells **)&adaptedbf->ftCells, *adaptedDm, bf->amrOps, minLevel, maxLevel, &bf->cellMemoryShape);
    CHKERRQ(ierr);
    ierr = DMBF_3D_CellsAmrPartition((DM_BF_3D_Cells *)adaptedbf->ftCells);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  /* create DMBF cells */
  ierr = DMBF_CellsCreate(*adaptedDm);
  CHKERRQ(ierr);
  /* copy data of DMBF cells from p4est */
  switch (dim) {
  case 2:
    ierr = DMBF_2D_CellsAmrFinalize(*adaptedDm, (DM_BF_2D_Cells *)adaptedbf->ftCells, adaptedbf->cells, &adaptedbf->cellMemoryShape);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_CellsAmrFinalize(*adaptedDm, (DM_BF_3D_Cells *)adaptedbf->ftCells, adaptedbf->cells, &adaptedbf->cellMemoryShape);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  /* setup DMBF cells */
  ierr = DMBF_CellsSetUpOwned(*adaptedDm);
  CHKERRQ(ierr);
  ierr = DMBF_CellsSetUpGhost(*adaptedDm);
  CHKERRQ(ierr);
  ierr = DMBF_LocalToGlobalScatterCreate(*adaptedDm, &adaptedbf->ltog);
  CHKERRQ(ierr);
  /* create forest-of-tree nodes */
  //TODO create nodes
  /* check resulting DM */
  CHKERRQ(DMBFCheck(*adaptedDm));
  /* mark that adapted DM is set up */
  (*adaptedDm)->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/***************************************
 * ITERATORS
 **************************************/

PetscErrorCode DMBFIterateOverCellsVectors(DM dm, PetscErrorCode (*iterCell)(DM, DM_BF_Cell *, void *), void *userIterCtx, Vec *vecRead, PetscInt nVecsRead, Vec *vecReadWrite, PetscInt nVecsReadWrite)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidFunction(iterCell, 2);
  if (nVecsRead) PetscAssertPointer(vecRead, 4);
  if (nVecsReadWrite) PetscAssertPointer(vecReadWrite, 6);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells || !_p_nCells(dm), _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  PetscCheck(bf->ownedCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Owned cells not set up");
  /* iterate */
  switch (_p_dim(dm)) {
  case 2:
    ierr = DMBF_2D_IterateOverCellsVectors(dm, bf->cells, _p_cellSize(bf), iterCell, userIterCtx, vecRead, nVecsRead, vecReadWrite, nVecsReadWrite);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateOverCellsVectors(dm, bf->cells, _p_cellSize(bf), iterCell, userIterCtx, vecRead, nVecsRead, vecReadWrite, nVecsReadWrite);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFIterateOverCells(DM dm, PetscErrorCode (*iterCell)(DM, DM_BF_Cell *, void *), void *userIterCtx)
{
  PetscFunctionBegin;
  CHKERRQ(DMBFIterateOverCellsVectors(dm, iterCell, userIterCtx, PETSC_NULLPTR, 0, PETSC_NULLPTR, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFIterateOverFaces(DM dm, PetscErrorCode (*iterFace)(DM, DM_BF_Face *, void *), void *userIterCtx)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidFunction(iterFace, 2);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells || !_p_nCells(dm), _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  PetscCheck(bf->ownedCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Owned cells not set up");
  PetscCheck(bf->ghostCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Ghost cells not set up");
  /* iterate */
  switch (_p_dim(dm)) {
  case 2:
    ierr = DMBF_2D_IterateOverFaces(dm, bf->cells, _p_cellSize(bf), iterFace, userIterCtx);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateOverFaces(dm, bf->cells, _p_cellSize(bf), iterFace, userIterCtx);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO deprecated
#define _p_cellOffsetDataRead(bf)      _p_cellMemoryOffset(&(bf)->cellMemoryShape, (size_t)DMBF_CELLMEMIDX_DATAREAD)
#define _p_cellOffsetDataReadWrite(bf) _p_cellMemoryOffset(&(bf)->cellMemoryShape, (size_t)DMBF_CELLMEMIDX_DATAREADWRITE)

PetscErrorCode DMBFSetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  PetscCheck(bf->ownedCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Owned cells not set up");
  if (vecRead && bf->nValsPerElemRead) PetscAssertPointer(vecRead, 2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscAssertPointer(vecReadWrite, 3);
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_IterateSetCellData(dm, bf->cells, _p_cellSize(bf), _p_cellOffsetDataRead(bf), _p_cellOffsetDataReadWrite(bf), bf->valsPerElemRead, bf->nValsPerElemRead, bf->valsPerElemReadWrite, bf->nValsPerElemReadWrite, vecRead, vecReadWrite);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateSetCellData(dm, bf->cells, _p_cellSize(bf), _p_cellOffsetDataRead(bf), _p_cellOffsetDataReadWrite(bf), bf->valsPerElemRead, bf->nValsPerElemRead, bf->valsPerElemReadWrite, bf->nValsPerElemReadWrite, vecRead, vecReadWrite);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFSetCellFields(DM dm, Vec *vecRead, Vec *vecReadWrite, PetscInt nFieldsRead, PetscInt *fieldsRead, PetscInt nFieldsReadWrite, PetscInt *fieldsReadWrite)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  PetscCheck(bf->ownedCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Owned cells not set up");
  if (vecRead && bf->nValsPerElemRead) PetscAssertPointer(vecRead, 2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscAssertPointer(vecReadWrite, 3);
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_IterateSetCellFields(dm, bf->cells, _p_cellSize(bf), _p_cellOffsetDataRead(bf), _p_cellOffsetDataReadWrite(bf), bf->valsPerElemRead, bf->nValsPerElemRead, bf->valsPerElemReadWrite, bf->nValsPerElemReadWrite, vecRead, vecReadWrite, nFieldsRead, fieldsRead, nFieldsReadWrite, fieldsReadWrite);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateSetCellFields(dm, bf->cells, _p_cellSize(bf), _p_cellOffsetDataRead(bf), _p_cellOffsetDataReadWrite(bf), bf->valsPerElemRead, bf->nValsPerElemRead, bf->valsPerElemReadWrite, bf->nValsPerElemReadWrite, vecRead, vecReadWrite, nFieldsRead, fieldsRead, nFieldsReadWrite, fieldsReadWrite);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFGetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  PetscCheck(bf->ownedCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Owned cells not set up");
  if (vecRead && bf->nValsPerElemRead) PetscAssertPointer(vecRead, 2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscAssertPointer(vecReadWrite, 3);
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_IterateGetCellData(dm, bf->cells, _p_cellSize(bf), _p_cellOffsetDataRead(bf), _p_cellOffsetDataReadWrite(bf), bf->valsPerElemRead, bf->nValsPerElemRead, bf->valsPerElemReadWrite, bf->nValsPerElemReadWrite, vecRead, vecReadWrite);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateGetCellData(dm, bf->cells, _p_cellSize(bf), _p_cellOffsetDataRead(bf), _p_cellOffsetDataReadWrite(bf), bf->valsPerElemRead, bf->nValsPerElemRead, bf->valsPerElemReadWrite, bf->nValsPerElemReadWrite, vecRead, vecReadWrite);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFGetCellFields(DM dm, Vec *vecRead, Vec *vecReadWrite, PetscInt nFieldsRead, PetscInt *fieldsRead, PetscInt nFieldsReadWrite, PetscInt *fieldsReadWrite)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  PetscCheck(bf->ownedCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Owned cells not set up");
  if (vecRead && bf->nValsPerElemRead) PetscAssertPointer(vecRead, 2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscAssertPointer(vecReadWrite, 3);
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_IterateGetCellFields(dm, bf->cells, _p_cellSize(bf), _p_cellOffsetDataRead(bf), _p_cellOffsetDataReadWrite(bf), bf->valsPerElemRead, bf->nValsPerElemRead, bf->valsPerElemReadWrite, bf->nValsPerElemReadWrite, vecRead, vecReadWrite, nFieldsRead, fieldsRead, nFieldsReadWrite, fieldsReadWrite);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateGetCellFields(dm, bf->cells, _p_cellSize(bf), _p_cellOffsetDataRead(bf), _p_cellOffsetDataReadWrite(bf), bf->valsPerElemRead, bf->nValsPerElemRead, bf->valsPerElemReadWrite, bf->nValsPerElemReadWrite, vecRead, vecReadWrite, nFieldsRead, fieldsRead, nFieldsReadWrite, fieldsReadWrite);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFCommunicateGhostCells(DM dm)
{
  DM_BF         *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  bf = _p_getBF(dm);
  PetscCheck(bf->cells || !_p_nCells(dm), _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Cells do not exist");
  PetscCheck(bf->ownedCellsSetUpCalled, _p_comm(dm), PETSC_ERR_ARG_WRONGSTATE, "Owned cells not set up");
  /* run ghost exchange */
  ierr = DMGetDimension(dm, &dim);
  CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMBF_2D_IterateGhostExchange(dm, bf->cells, _p_cellSize(bf));
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateGhostExchange(dm, bf->cells, _p_cellSize(bf));
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  /* setup ghost cells */
  bf->ghostCellsSetUpCalled = PETSC_FALSE;
  ierr                      = DMBF_CellsSetUpGhost(dm);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFFVMatAssemble(DM dm, Mat mat, PetscErrorCode (*iterFace)(DM, DM_BF_Face *, PetscReal *, void *), void *userIterCtx)
{
  DM_BF         *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  PetscValidFunction(iterFace, 2);
  CHKERRQ(DMBFCheck(dm));
  bf   = _p_getBF(dm);
  ierr = MatZeroEntries(mat);
  CHKERRQ(ierr);
  switch (_p_dim(dm)) {
  case 2:
    ierr = DMBF_2D_IterateFVMatAssembly(dm, bf->cells, _p_cellSize(bf), mat, iterFace, userIterCtx);
    CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMBF_3D_IterateFVMatAssembly(dm, bf->cells, _p_cellSize(bf), mat, iterFace, userIterCtx);
    CHKERRQ(ierr);
    break;
  default:
    _p_SETERRQ_UNREACHABLE(dm);
  }
  ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/***************
 * VIEWER
 ***************/

static PetscErrorCode DMView_BF(DM dm, PetscViewer viewer)
{
  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMBF);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERVTK, &isvtk);
  CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5);
  CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw);
  CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERGLVIS, &isglvis);
  CHKERRQ(ierr);
  if (isvtk) {
    switch (_p_dim(dm)) {
    case 2:
      ierr = DMBF_2D_VTKWriteAll((PetscObject)dm, viewer);
      CHKERRQ(ierr);
      break;
    case 3:
      ierr = DMBF_3D_VTKWriteAll((PetscObject)dm, viewer);
      CHKERRQ(ierr);
      break;
    default:
      _p_SETERRQ_UNREACHABLE(dm);
    }
  } else if (ishdf5 || isdraw || isglvis) {
    SETERRQ(_p_comm(dm), PETSC_ERR_SUP, "non-VTK viewer currently not supported by BF");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecView_BF(Vec v, PetscViewer viewer)
{
  DM                      dm;
  PetscBool               isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode          ierr;
  PetscInt                vsize, locDof, ng;
  PetscInt                bs, blockSize[3] = {1, 1, 1};
  PetscViewerVTKFieldType ft;
  Vec                     locv;
  const char             *name;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);
  CHKERRQ(ierr);
  PetscCheck(dm, _p_comm(v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERVTK, &isvtk);
  CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5);
  CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw);
  CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERGLVIS, &isglvis);
  CHKERRQ(ierr);
  if (isvtk) {
    /* create a copy and store it in the viewer */
    ierr = DMCreateLocalVector(dm, &locv);
    CHKERRQ(ierr); /* we store local vectors in the viewer. done't know why, since we don't need ghost values */
    ierr = DMGlobalToLocal(dm, v, INSERT_VALUES, locv);
    CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)v, &name);
    CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)locv, name);
    CHKERRQ(ierr);
    ierr = DMBFGetLocalSize(dm, &locDof);
    CHKERRQ(ierr);
    ierr = DMBFGetGhostSize(dm, &ng);
    CHKERRQ(ierr);
    ierr = VecGetSize(locv, &vsize);
    CHKERRQ(ierr);
    ierr = DMBFGetBlockSize(dm, blockSize);
    CHKERRQ(ierr);
    bs     = blockSize[0] * blockSize[1] * blockSize[2];
    locDof = (locDof + ng) * bs;
    //if(vsize == P4EST_DIM*size)                { ft = PETSC_VTK_CELL_VECTOR_FIELD; } /* right now this is not actually supported (dm local to global is only for cell fields) */
    //else if(vsize == size)                     { ft = PETSC_VTK_CELL_FIELD;        }
    if (vsize == locDof) {
      ft = PETSC_VTK_CELL_FIELD;
    } /* if it's a local vector field, there will be an error before this in the dmlocaltoglobal */
    else {
      SETERRQ(_p_comm(locv), PETSC_ERR_SUP, "Only scalar cell fields currently supported");
    }
    switch (_p_dim(dm)) {
    case 2:
      ierr = PetscViewerVTKAddField(viewer, (PetscObject)dm, DMBF_2D_VTKWriteAll, PETSC_DEFAULT, ft, PETSC_TRUE, (PetscObject)locv);
      CHKERRQ(ierr);
      break;
    case 3:
      ierr = PetscViewerVTKAddField(viewer, (PetscObject)dm, DMBF_3D_VTKWriteAll, PETSC_DEFAULT, ft, PETSC_TRUE, (PetscObject)locv);
      CHKERRQ(ierr);
      break;
    default:
      _p_SETERRQ_UNREACHABLE(dm);
    }
  } else if (ishdf5 || isdraw || isglvis) {
    SETERRQ(_p_comm(dm), PETSC_ERR_SUP, "non-VTK viewer currently not supported by BF");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
