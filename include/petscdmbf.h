#pragma once

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMBFSetCellDataShape(DM, const PetscInt *, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMBFGetCellDataShape(DM, PetscInt **, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMBFSetCellDataVSize(DM, size_t);
PETSC_EXTERN PetscErrorCode DMBFGetCellDataVSize(DM, size_t *);
PETSC_EXTERN PetscErrorCode DMBFSetBlockSize(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMBFGetBlockSize(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMBFSetCellDataSize(DM, PetscInt *, PetscInt, PetscInt *, PetscInt);
PETSC_EXTERN PetscErrorCode DMBFGetCellDataSize(DM, PetscInt **, PetscInt *, PetscInt **, PetscInt *);

PETSC_EXTERN PetscErrorCode DMBFGetInfo(DM, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMBFGetLocalSize(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMBFGetGlobalSize(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMBFGetGhostSize(DM, PetscInt *);

typedef struct _p_DM_BF_Shape {
  size_t   n, dim, size, __padding__;
  size_t **list; /* size=(n x dim) */
  size_t  *pad;  /* size=(n)       */
} DM_BF_Shape;

typedef struct _p_DM_BF_Cell {
  /* corner coordinates, etc. */
  PetscReal corner[8 * 3], volume, sidelength[3], __padding1__[4];
  /* cell indices local to this processor rank and global across all ranks */
  PetscInt indexLocal, indexGlobal;
  /* cell refinement level */
  PetscInt level, __padding2__;
  /* flag for AMR */
  DMAdaptFlag adaptFlag;
  /* memory layout of this cell (not owned) */
  const DM_BF_Shape *memory;
  /* view of vector entries corresponding to this cell (not owned) */
  const PetscScalar **vecViewRead;
  PetscScalar       **vecViewReadWrite;
  /* data corresponding to this cell (owned) */
  const PetscScalar *dataRead;      //TODO deprecated
  PetscScalar       *dataReadWrite; //TODO deprecated
  void              *dataV;
  PetscScalar      **data;
} DM_BF_Cell;

//#define DMBFCellIsGhost(cell) (-1 == (cell)->indexGlobal) //TODO implement version that checks global indices

PETSC_EXTERN PetscErrorCode DMBFIterateOverCellsVectors(DM, PetscErrorCode (*)(DM, DM_BF_Cell *, void *), void *, Vec *, PetscInt, Vec *, PetscInt);
PETSC_EXTERN PetscErrorCode DMBFIterateOverCells(DM, PetscErrorCode (*)(DM, DM_BF_Cell *, void *), void *);

typedef enum {
  DM_BF_FACEDIR_XNEG = 0, // face x-
  DM_BF_FACEDIR_XPOS = 1, // face x+
  DM_BF_FACEDIR_YNEG = 2, // face y-
  DM_BF_FACEDIR_YPOS = 3, // face y+
  DM_BF_FACEDIR_ZNEG = 4, // face z-
  DM_BF_FACEDIR_ZPOS = 5  // face z+
} DM_BF_FaceDir;

typedef enum {
  DM_BF_FACEBOUNDARY_NONE = -1,
  DM_BF_FACEBOUNDARY_XNEG = DM_BF_FACEDIR_XNEG,
  DM_BF_FACEBOUNDARY_XPOS = DM_BF_FACEDIR_XPOS,
  DM_BF_FACEBOUNDARY_YNEG = DM_BF_FACEDIR_YNEG,
  DM_BF_FACEBOUNDARY_YPOS = DM_BF_FACEDIR_YPOS,
  DM_BF_FACEBOUNDARY_ZNEG = DM_BF_FACEDIR_ZNEG,
  DM_BF_FACEBOUNDARY_ZPOS = DM_BF_FACEDIR_ZPOS
} DM_BF_FaceBoundary;

typedef struct _p_DM_BF_Face {
  /* domain boundary */
  DM_BF_FaceBoundary boundary;
  /* face direction/orientation */
  DM_BF_FaceDir dir;
  /* cells on each side of the face */
  PetscInt    nCellsL, nCellsR;
  DM_BF_Cell *cellL[4], *cellR[4];
} DM_BF_Face;

PETSC_EXTERN PetscErrorCode DMBFIterateOverFacesVectors(DM, PetscErrorCode (*)(DM, DM_BF_Face *, void *), void *, Vec *, PetscInt, Vec *, PetscInt);
PETSC_EXTERN PetscErrorCode DMBFIterateOverFaces(DM, PetscErrorCode (*)(DM, DM_BF_Face *, void *), void *);

PETSC_EXTERN PetscErrorCode DMBFSetCellData(DM, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode DMBFSetCellFields(DM, Vec *, Vec *, PetscInt, PetscInt *, PetscInt, PetscInt *);

PETSC_EXTERN PetscErrorCode DMBFFVMatAssemble(DM, Mat, PetscErrorCode (*)(DM, DM_BF_Face *, PetscReal *, void *), void *);

PETSC_EXTERN PetscErrorCode DMBFGetCellData(DM, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode DMBFGetCellFields(DM, Vec *, Vec *, PetscInt, PetscInt *, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMBFCommunicateGhostCells(DM);

typedef struct _p_DM_BF_AmrOps {
  PetscErrorCode (*setAmrFlag)(DM, DM_BF_Cell *, void *);
  PetscErrorCode (*projectToCoarse)(DM, DM_BF_Cell **, PetscInt, DM_BF_Cell **, PetscInt, void *);
  PetscErrorCode (*projectToFine)(DM, DM_BF_Cell **, PetscInt, DM_BF_Cell **, PetscInt, void *);
  void *setAmrFlagCtx, *projectToCoarseCtx, *projectToFineCtx;
} DM_BF_AmrOps;

PETSC_EXTERN PetscErrorCode DMBFAMRSetOperators(DM, DM_BF_AmrOps *);
PETSC_EXTERN PetscErrorCode DMBFAMRFlag(DM);
PETSC_EXTERN PetscErrorCode DMBFAMRAdapt(DM, DM *);

PETSC_EXTERN PetscErrorCode DMBFVTKWriteAll(PetscObject, PetscViewer);
