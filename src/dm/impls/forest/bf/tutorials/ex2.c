#include <petsc.h>
#include <petscdmbf.h>
#include <petscdmforest.h>

/* 3-D Poisson problem with Dirichlet boundary conditions on the square [-1,1]^3.
   We add a slit on the strip \Gamma = {(x,y,z) : y,z = 0, -1 <= x <= 0} and treat \Gamma as part of the boundary.
   The exact solution is r^{1/2}*Sin(\theta/2) in polar coordinates, and has a singularity at the origin which we resolve using AMR.

  Three steps to applying the matrix-free operator:
    1. Transcribe unknowns from Vec into cell data.
    2. Set guard layer by iterating over faces, interpolating where necessary.
    3. Apply operator by looping over cells, using guard layer for fluxes.
 */

static char help[] = "3-D Poisson problem on the unit cube with AMR\n";

/*
stencils of convergence order O(h^p), we refer to p as the order of the stencil
*/

#define ORDER_2_STENCIL \
  { \
    -1.0, 2.0, -1.0 \
  }
#define ORDER_4_STENCIL \
  { \
    1.0 / 12.0, -4.0 / 3.0, 5.0 / 2.0, -4.0 / 3.0, 1.0 / 12.0 \
  }
#define ORDER_6_STENCIL \
  { \
    -1.0 / 90.0, 3.0 / 20.0, -3.0 / 2.0, 49.0 / 18.0, -3.0 / 2.0, 3.0 / 20.0, -1.0 / 90.0 \
  }

#define STENCIL_LEN 7
typedef enum {
  GUARD_XL,
  GUARD_XR,
  GUARD_YL,
  GUARD_YR,
  GUARD_ZL,
  GUARD_ZR,
  CENTER
} unkLoc;
#define N_FACE_COORDS_X 6
#define N_FACE_COORDS_Y 6
#define N_FACE_COORDS_Z 6
typedef enum {
  FACE_XL,
  FACE_XR,
  FACE_YL,
  FACE_YR,
  FACE_ZL,
  FACE_ZR
} faceLoc;
#define N_CENTER_COORDS_X 1
#define N_CENTER_COORDS_Y 1
#define N_CENTER_COORDS_Z 1
typedef enum {
  STENCIL,
  CENTER_COORD_X,
  CENTER_COORD_Y,
  CENTER_COORD_Z,
  FACE_COORD_X,
  FACE_COORD_Y,
  FACE_COORD_Z,
  CELLDATA_N_FIELDS
} cellDat;

#define maxCellDataFieldDim 1
#define maxCellDataFieldLen 1
static const PetscInt CELLDATA_SHAPE[CELLDATA_N_FIELDS * 2] = {
  /* STENCIL        */ STENCIL_LEN,       1,
  /* CENTER_COORD_X */ N_CENTER_COORDS_X, 1,
  /* CENTER_COORD_Y */ N_CENTER_COORDS_Y, 1,
  /* CENTER_COORD_Y */ N_CENTER_COORDS_Z, 1,
  /* FACE_COORD_X   */ N_FACE_COORDS_X,   1,
  /* FACE_COORD_Y   */ N_FACE_COORDS_Y,   1,
  /* FACE_COORD_Z   */ N_FACE_COORDS_Z,   1};

typedef PetscScalar (*SpatialFn_3D)(PetscScalar, PetscScalar, PetscScalar);

typedef struct {
  SpatialFn_3D bc;
  SpatialFn_3D src;
} AppCtx;

typedef struct {
  //nothing for now
} cellData_t;
// PetscErrorCode get(Boundary)FaceMidpoint()

PetscErrorCode init_cell_data(DM dm, DM_BF_Cell *cell, void *ctx)
{
  PetscFunctionBeginUser;

  PetscCall(PetscArrayzero(cell->data[STENCIL], STENCIL_LEN));

  cell->data[CENTER_COORD_X][0] = cell->corner[0] + .5 * cell->sidelength[0];
  cell->data[CENTER_COORD_Y][0] = cell->corner[1] + .5 * cell->sidelength[1];
  cell->data[CENTER_COORD_Z][0] = cell->corner[2] + .5 * cell->sidelength[2];

  cell->data[FACE_COORD_X][FACE_XL] = cell->corner[0];
  cell->data[FACE_COORD_X][FACE_XR] = cell->corner[0] + cell->sidelength[0];
  cell->data[FACE_COORD_X][FACE_YL] = cell->data[CENTER_COORD_X][0];
  cell->data[FACE_COORD_X][FACE_YR] = cell->data[CENTER_COORD_X][0];
  cell->data[FACE_COORD_X][FACE_ZL] = cell->data[CENTER_COORD_X][0];
  cell->data[FACE_COORD_X][FACE_ZR] = cell->data[CENTER_COORD_X][0];

  cell->data[FACE_COORD_Y][FACE_XL] = cell->data[CENTER_COORD_Y][0];
  cell->data[FACE_COORD_Y][FACE_XR] = cell->data[CENTER_COORD_Y][0];
  cell->data[FACE_COORD_Y][FACE_YL] = cell->corner[1];
  cell->data[FACE_COORD_Y][FACE_YR] = cell->corner[1] + cell->sidelength[1];
  cell->data[FACE_COORD_Y][FACE_ZL] = cell->data[CENTER_COORD_Y][0];
  cell->data[FACE_COORD_Y][FACE_ZR] = cell->data[CENTER_COORD_Y][0];

  cell->data[FACE_COORD_Z][FACE_XL] = cell->data[CENTER_COORD_Z][0];
  cell->data[FACE_COORD_Z][FACE_XR] = cell->data[CENTER_COORD_Z][0];
  cell->data[FACE_COORD_Z][FACE_YL] = cell->data[CENTER_COORD_Z][0];
  cell->data[FACE_COORD_Z][FACE_YR] = cell->data[CENTER_COORD_Z][0];
  cell->data[FACE_COORD_Z][FACE_ZL] = cell->corner[2];
  cell->data[FACE_COORD_Z][FACE_ZR] = cell->corner[2] + cell->sidelength[2];

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode print_cell_data(DM dm, DM_BF_Cell *cell, void *ctx)
{
  size_t i, j, k;

  PetscFunctionBeginUser;

  PetscPrintf(PETSC_COMM_SELF, "%s: cell global index %i\n", __func__, cell->indexGlobal);
  PetscPrintf(PETSC_COMM_SELF, "  adaptFlag %i\n", cell->adaptFlag);
  PetscPrintf(PETSC_COMM_SELF, "  corner (%g,%g,%g), side (%g,%g,%g)\n", cell->corner[0], cell->corner[1], cell->corner[2], cell->sidelength[0], cell->sidelength[1], cell->sidelength[2]);

  PetscPrintf(PETSC_COMM_SELF, "  STENCIL        ");
  for (i = 0; i < STENCIL_LEN; i++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[STENCIL][i]); }

  PetscPrintf(PETSC_COMM_SELF, "\n  CENTER_COORD_X ");
  for (i = 0; i < N_CENTER_COORDS_X; i++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[CENTER_COORD_X][i]); }
  PetscPrintf(PETSC_COMM_SELF, "\n  CENTER_COORD_Y ");
  for (j = 0; j < N_CENTER_COORDS_Y; j++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[CENTER_COORD_Y][j]); }
  PetscPrintf(PETSC_COMM_SELF, "\n  CENTER_COORD_Z ");
  for (k = 0; k < N_CENTER_COORDS_Z; k++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[CENTER_COORD_Z][j]); }

  PetscPrintf(PETSC_COMM_SELF, "\n  FACE_COORD_X   ");
  for (i = 0; i < N_FACE_COORDS_X; i++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[FACE_COORD_X][i]); }
  PetscPrintf(PETSC_COMM_SELF, "\n  FACE_COORD_Y   ");
  for (j = 0; j < N_FACE_COORDS_Y; j++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[FACE_COORD_Y][j]); }
  PetscPrintf(PETSC_COMM_SELF, "\n  FACE_COORD_Z   ");
  for (k = 0; k < N_FACE_COORDS_Z; k++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[FACE_COORD_Z][k]); }
  PetscPrintf(PETSC_COMM_SELF, "\n");

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode set_up_boundary_condition(DM dm, DM_BF_Face *face, void *ctx)
{
  AppCtx           *user       = ctx;
  DM_BF_Cell       *cell       = (face->nCellsL ? face->cellL[0] : face->cellR[0]);
  const PetscScalar hx         = cell->data[FACE_COORD_X][FACE_XR] - cell->data[FACE_COORD_X][FACE_XL];
  const PetscScalar hy         = cell->data[FACE_COORD_Y][FACE_YR] - cell->data[FACE_COORD_Y][FACE_YL];
  const PetscScalar hz         = cell->data[FACE_COORD_Z][FACE_ZR] - cell->data[FACE_COORD_Z][FACE_ZL];
  PetscBool         isBoundary = (DM_BF_FACEBOUNDARY_NONE != face->boundary);

  PetscFunctionBeginUser;

  if (isBoundary) {
    PetscScalar h, bndryFaceMidpoint_x, bndryFaceMidpoint_y, bndryFaceMidpoint_z;
    bndryFaceMidpoint_x = cell->data[FACE_COORD_X][face->dir];
    bndryFaceMidpoint_y = cell->data[FACE_COORD_Y][face->dir];
    bndryFaceMidpoint_z = cell->data[FACE_COORD_Z][face->dir];

    switch (face->dir) {
    case DM_BF_FACEDIR_XNEG:
    case DM_BF_FACEDIR_XPOS:
      h = hx;
      break;
    case DM_BF_FACEDIR_YNEG:
    case DM_BF_FACEDIR_YPOS:
      h = hy;
      break;
    case DM_BF_FACEDIR_ZNEG:
    case DM_BF_FACEDIR_ZPOS:
      h = hz;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Unknown face direction %i", face->dir);
    }
    cell->data[STENCIL][CENTER] += -2 * user->bc(bndryFaceMidpoint_x, bndryFaceMidpoint_y, bndryFaceMidpoint_z) / (h * h);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode interpolate_source(DM dm, DM_BF_Cell *cell, void *ctx)
{
  AppCtx      *user         = ctx;
  PetscScalar *readWriteVal = cell->vecViewReadWrite[0];
  PetscScalar  x            = cell->data[CENTER_COORD_X][0];
  PetscScalar  y            = cell->data[CENTER_COORD_Y][0];
  PetscScalar  z            = cell->data[CENTER_COORD_Z][0];

  PetscFunctionBeginUser;

  *readWriteVal = user->src(x, y, z) + cell->data[STENCIL][CENTER];

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode interpolate_exact(DM dm, DM_BF_Cell *cell, void *ctx)
{
  AppCtx      *user         = ctx;
  PetscScalar *readWriteVal = cell->vecViewReadWrite[0];
  PetscScalar  x            = cell->data[CENTER_COORD_X][0];
  PetscScalar  y            = cell->data[CENTER_COORD_Y][0];
  PetscScalar  z            = cell->data[CENTER_COORD_Z][0];

  PetscFunctionBeginUser;

  *readWriteVal = user->bc(x, y, z);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode _p_dmbf_poisson_set_unk_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  const PetscScalar *vecReadVal = cell->vecViewRead[0];

  PetscFunctionBeginUser;

  cell->data[STENCIL][CENTER] = *vecReadVal;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*

  In addition to the usual domain boundaries, the strip {(x,y) : y = 0, -1 <= x <= 0} is part of the boundary.
   The square domain naturally conforms to this boundary so that the interior of each edge either lies completely
   inside the strip or completely outside. This function checks if an interior face lies inside the strip or not
   by checking the coordinates of a point on the face.

 */
static inline PetscBool InteriorFaceIsBndry(DM_BF_Face *face)
{
  return (face->cellR[0]->data[FACE_COORD_X][FACE_YL] < 0.0 && PetscAbs(face->cellR[0]->data[FACE_COORD_Y][FACE_YL]) < PETSC_SMALL);
}

static PetscErrorCode _p_dmbf_poisson_set_guards_facefn(DM dm, DM_BF_Face *face, void *ctx)
{
  const PetscInt  nCellsL    = face->nCellsL;
  const PetscInt  nCellsR    = face->nCellsR;
  const PetscBool isBoundary = (DM_BF_FACEBOUNDARY_NONE != face->boundary);
  const PetscBool isHangingL = (1 < nCellsL);
  const PetscBool isHangingR = (1 < nCellsR);
  const PetscBool X_DIR      = (face->dir == DM_BF_FACEDIR_XNEG || face->dir == DM_BF_FACEDIR_XPOS);
  const PetscBool Z_DIR      = (face->dir == DM_BF_FACEDIR_ZNEG || face->dir == DM_BF_FACEDIR_ZPOS);

  PetscFunctionBeginUser;

  if (isBoundary) {
    DM_BF_Cell *cell               = nCellsL ? face->cellL[0] : face->cellR[0];
    cell->data[STENCIL][face->dir] = -cell->data[STENCIL][CENTER];
  } else if (!X_DIR && !Z_DIR && InteriorFaceIsBndry(face)) {
    DM_BF_Cell *cell;
    for (PetscInt i = 0; i < nCellsL; i++) {
      cell                          = face->cellL[i];
      cell->data[STENCIL][GUARD_YR] = -cell->data[STENCIL][CENTER];
    }
    for (PetscInt i = 0; i < nCellsR; i++) {
      cell                          = face->cellR[i];
      cell->data[STENCIL][GUARD_YL] = -cell->data[STENCIL][CENTER];
    }
  } else {
    if (isHangingL) {
      DM_BF_Cell **cellL                                                        = face->cellL;
      DM_BF_Cell  *cellR                                                        = face->cellR[0];
      cellL[0]->data[STENCIL][X_DIR ? GUARD_XR : (Z_DIR ? GUARD_ZR : GUARD_YR)] = (2. / 3.) * cellR->data[STENCIL][CENTER] +
                                                                                  (2. / 3.) * cellL[0]->data[STENCIL][CENTER]
                                                                                //+ (1./3.)*cellL[2]->data[STENCIL][CENTER] + (1./3.)*cellL[3]->data[STENCIL][CENTER]
                                                                                - (1. / 3.) * cellL[2]->data[STENCIL][CENTER];
      cellL[1]->data[STENCIL][X_DIR ? GUARD_XR : (Z_DIR ? GUARD_ZR : GUARD_YR)] = (2. / 3.) * cellR->data[STENCIL][CENTER] +
                                                                                  (2. / 3.) * cellL[1]->data[STENCIL][CENTER]
                                                                                //+ (1./3.)*cellL[2]->data[STENCIL][CENTER] + (1./3.)*cellL[3]->data[STENCIL][CENTER]
                                                                                - (1. / 3.) * cellL[3]->data[STENCIL][CENTER];
      cellL[2]->data[STENCIL][X_DIR ? GUARD_XR : (Z_DIR ? GUARD_ZR : GUARD_YR)] = (2. / 3.) * cellR->data[STENCIL][CENTER] +
                                                                                  (2. / 3.) * cellL[2]->data[STENCIL][CENTER]
                                                                                //+ (1./3.)*cellL[1]->data[STENCIL][CENTER] + (1./3.)*cellL[3]->data[STENCIL][CENTER]
                                                                                - (1. / 3.) * cellL[0]->data[STENCIL][CENTER];
      cellL[3]->data[STENCIL][X_DIR ? GUARD_XR : (Z_DIR ? GUARD_ZR : GUARD_YR)] = (2. / 3.) * cellR->data[STENCIL][CENTER] +
                                                                                  (2. / 3.) * cellL[3]->data[STENCIL][CENTER]
                                                                                //+ (1./3.)*cellL[1]->data[STENCIL][CENTER] + (1./3.)*cellL[2]->data[STENCIL][CENTER]
                                                                                - (1. / 3.) * cellL[1]->data[STENCIL][CENTER];
      cellR->data[STENCIL][X_DIR ? GUARD_XL : (Z_DIR ? GUARD_ZL : GUARD_YL)] = (1. / 3.) * cellL[0]->data[STENCIL][CENTER] + (1. / 3.) * cellL[1]->data[STENCIL][CENTER] + (1. / 3.) * cellL[2]->data[STENCIL][CENTER] + (1. / 3.) * cellL[3]->data[STENCIL][CENTER] -
                                                                               (1. / 3.) * cellR->data[STENCIL][CENTER];
    } else if (isHangingR) {
      DM_BF_Cell **cellR                                                        = face->cellR;
      DM_BF_Cell  *cellL                                                        = face->cellL[0];
      cellR[0]->data[STENCIL][X_DIR ? GUARD_XL : (Z_DIR ? GUARD_ZL : GUARD_YL)] = (2. / 3.) * cellL->data[STENCIL][CENTER] +
                                                                                  (2. / 3.) * cellR[0]->data[STENCIL][CENTER]
                                                                                //+ (1./3.)*cellR[2]->data[STENCIL][CENTER] + (1./3.)*cellR[3]->data[STENCIL][CENTER]
                                                                                - (1. / 3.) * cellR[2]->data[STENCIL][CENTER];
      cellR[1]->data[STENCIL][X_DIR ? GUARD_XL : (Z_DIR ? GUARD_ZL : GUARD_YL)] = (2. / 3.) * cellL->data[STENCIL][CENTER] +
                                                                                  (2. / 3.) * cellR[1]->data[STENCIL][CENTER]
                                                                                //+ (1./3.)*cellR[2]->data[STENCIL][CENTER] + (1./3.)*cellR[3]->data[STENCIL][CENTER]
                                                                                - (1. / 3.) * cellR[3]->data[STENCIL][CENTER];
      cellR[2]->data[STENCIL][X_DIR ? GUARD_XL : (Z_DIR ? GUARD_ZL : GUARD_YL)] = (2. / 3.) * cellL->data[STENCIL][CENTER] +
                                                                                  (2. / 3.) * cellR[2]->data[STENCIL][CENTER]
                                                                                //+ (1./3.)*cellR[1]->data[STENCIL][CENTER] + (1./3.)*cellR[3]->data[STENCIL][CENTER]
                                                                                - (1. / 3.) * cellR[0]->data[STENCIL][CENTER];
      cellR[3]->data[STENCIL][X_DIR ? GUARD_XL : (Z_DIR ? GUARD_ZL : GUARD_YL)] = (2. / 3.) * cellL->data[STENCIL][CENTER] +
                                                                                  (2. / 3.) * cellR[3]->data[STENCIL][CENTER]
                                                                                //+ (1./3.)*cellR[1]->data[STENCIL][CENTER] + (1./3.)*cellR[2]->data[STENCIL][CENTER]
                                                                                - (1. / 3.) * cellR[1]->data[STENCIL][CENTER];
      cellL->data[STENCIL][X_DIR ? GUARD_XR : (Z_DIR ? GUARD_ZR : GUARD_YR)] = (1. / 3.) * cellR[0]->data[STENCIL][CENTER] + (1. / 3.) * cellR[1]->data[STENCIL][CENTER] + (1. / 3.) * cellR[2]->data[STENCIL][CENTER] + (1. / 3.) * cellR[3]->data[STENCIL][CENTER] -
                                                                               (1. / 3.) * cellL->data[STENCIL][CENTER];
    } else {
      DM_BF_Cell *cellL                                                      = face->cellL[0];
      DM_BF_Cell *cellR                                                      = face->cellR[0];
      cellL->data[STENCIL][X_DIR ? GUARD_XR : (Z_DIR ? GUARD_ZR : GUARD_YR)] = cellR->data[STENCIL][CENTER];
      cellR->data[STENCIL][X_DIR ? GUARD_XL : (Z_DIR ? GUARD_ZL : GUARD_YL)] = cellL->data[STENCIL][CENTER];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode apply_mass_matrix(DM dm, DM_BF_Cell *cell, void *ctx)
{
  const PetscScalar hx           = cell->data[FACE_COORD_X][FACE_XR] - cell->data[FACE_COORD_X][FACE_XL];
  const PetscScalar hy           = cell->data[FACE_COORD_Y][FACE_YR] - cell->data[FACE_COORD_Y][FACE_YL];
  const PetscScalar hz           = cell->data[FACE_COORD_Z][FACE_ZR] - cell->data[FACE_COORD_Z][FACE_ZL];
  const PetscScalar mass         = hx * hy * hz;
  PetscScalar      *valReadWrite = cell->vecViewReadWrite[0];

  PetscFunctionBeginUser;
  *valReadWrite *= mass;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode _p_dmbf_poisson_apply_operator_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  const PetscScalar hx           = cell->data[FACE_COORD_X][FACE_XR] - cell->data[FACE_COORD_X][FACE_XL];
  const PetscScalar hy           = cell->data[FACE_COORD_Y][FACE_YR] - cell->data[FACE_COORD_Y][FACE_YL];
  const PetscScalar hz           = cell->data[FACE_COORD_Z][FACE_ZR] - cell->data[FACE_COORD_Z][FACE_ZL];
  PetscScalar      *valReadWrite = cell->vecViewReadWrite[0];

  PetscFunctionBeginUser;
  *valReadWrite = (cell->data[STENCIL][GUARD_XL] - 2 * cell->data[STENCIL][CENTER] + cell->data[STENCIL][GUARD_XR]) / (hx * hx) + (cell->data[STENCIL][GUARD_YL] - 2 * cell->data[STENCIL][CENTER] + cell->data[STENCIL][GUARD_YR]) / (hy * hy) +
                  (cell->data[STENCIL][GUARD_ZL] - 2 * cell->data[STENCIL][CENTER] + cell->data[STENCIL][GUARD_ZR]) / (hz * hz);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode apply_operator_dm(DM dm, Vec in, Vec out)
{
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(DMBFIterateOverCellsVectors(dm, _p_dmbf_poisson_set_unk_cellfn, user, &in, 1, PETSC_NULLPTR, 0));
  PetscCall(DMBFCommunicateGhostCells(dm));
  PetscCall(DMBFIterateOverFaces(dm, _p_dmbf_poisson_set_guards_facefn, user));
  PetscCall(DMBFIterateOverCellsVectors(dm, _p_dmbf_poisson_apply_operator_cellfn, user, PETSC_NULLPTR, 0, &out, 1));
  PetscCall(DMBFIterateOverCellsVectors(dm, apply_mass_matrix, user, PETSC_NULLPTR, 0, &out, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode apply_operator_mf(Mat K, Vec in, Vec out)
{
  DM dm;

  PetscFunctionBeginUser;
  PetscCall(MatGetDM(K, &dm));
  PetscCall(apply_operator_dm(dm, in, out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscScalar g(PetscScalar x, PetscScalar y, PetscScalar z)
{
  PetscScalar theta = PetscAtan2Real(y, x) + PETSC_PI;
  PetscScalar rsqr  = PetscSqr(x) + PetscSqr(y);

  return PetscPowReal(rsqr, .25) * PetscSinReal(.5 * theta) * PetscSinReal(2 * PETSC_PI * z);
}

static inline PetscScalar f(PetscScalar x, PetscScalar y, PetscScalar z)
{
  return -4 * PetscSqr(PETSC_PI) * g(x, y, z);
}

PetscErrorCode amr_refine_center(DM dm, DM_BF_Cell *cell, void *ctx)
{
  PetscScalar x = PetscMin(PetscAbs(cell->corner[0]), PetscAbs(cell->corner[0] + cell->sidelength[0]));
  PetscScalar y = PetscMin(PetscAbs(cell->corner[1]), PetscAbs(cell->corner[1] + cell->sidelength[1]));
  PetscScalar r = PetscSqrtReal(PetscSqr(x) + PetscSqr(y));

  PetscFunctionBeginUser;
  if (r < 5 * 1e-2) {
    cell->adaptFlag = DM_ADAPT_REFINE;
  } else {
    cell->adaptFlag = DM_ADAPT_KEEP;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode proj_no_op(DM dm, DM_BF_Cell **cellsFine, PetscInt i, DM_BF_Cell **cellsCoarse, PetscInt j, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  const char     funcname[] = "DMBF-Poisson-3D";
  DM             dm;
  Vec            sol, rhs, exact, error;
  Mat            A;
  KSP            ksp;
  PetscReal      inf_norm, l2_norm;
  AppCtx         ctx;
  PetscViewer    viewer;
  PetscErrorCode init_ierr;

  // initialize Petsc
  PetscFunctionBeginUser;
  init_ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (PETSC_SUCCESS != init_ierr) { return init_ierr; }

  // begin main
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Begin\n", funcname);

  // create DM
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Create DM\n", funcname);
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, "bf"));

  // set DM options
  PetscCall(DMSetDimension(dm, 3));
  PetscCall(DMSetFromOptions(dm));
  // set cell data shapes
  PetscCall(DMBFSetCellDataShape(dm, CELLDATA_SHAPE, CELLDATA_N_FIELDS, 2));
  //PetscCall(DMBFSetCellDataVSize(dm,sizeof(cellData_t))); //TODO unused at the moment

  // set application-specific data
  PetscCall(DMSetApplicationContext(dm, &ctx));

  // setup DM
  PetscCall(DMSetUp(dm));

  // run initial AMR
  {
    DM_BF_AmrOps amrOps;
    PetscInt     maxRefinement, initRefinement, l;
    DM           adapt;

    amrOps.setAmrFlag         = amr_refine_center;
    amrOps.setAmrFlagCtx      = &ctx;
    amrOps.projectToCoarse    = proj_no_op;
    amrOps.projectToFine      = proj_no_op;
    amrOps.projectToFineCtx   = PETSC_NULLPTR;
    amrOps.projectToCoarseCtx = PETSC_NULLPTR;

    PetscCall(DMBFAMRSetOperators(dm, &amrOps));

    PetscCall(DMForestGetInitialRefinement(dm, &initRefinement));
    PetscCall(DMForestGetMaximumRefinement(dm, &maxRefinement));

    for (l = 0; l < maxRefinement - initRefinement; l++) {
      PetscPrintf(PETSC_COMM_WORLD, "[%s] Run initial AMR (step %i of max %i)\n", funcname, l + 1, maxRefinement - initRefinement);
      PetscCall(DMBFAMRFlag(dm));
      PetscCall(DMBFAMRAdapt(dm, &adapt));
      PetscCall(DMDestroy(&dm));
      dm = adapt;
    }
    if (l) { PetscPrintf(PETSC_COMM_WORLD, "[%s] Finished initial AMR (%i steps)\n", funcname, l); }
  }

  // initialize cell data
  PetscCall(DMBFIterateOverCells(dm, init_cell_data, PETSC_NULLPTR));

  ctx.src = f;
  ctx.bc  = g;

  // create vectors
  PetscCall(DMCreateGlobalVector(dm, &sol));
  PetscCall(VecDuplicate(sol, &rhs));
  PetscCall(VecDuplicate(sol, &exact));

  PetscCall(PetscObjectSetName((PetscObject)sol, "sol"));
  PetscCall(PetscObjectSetName((PetscObject)rhs, "rhs"));
  PetscCall(PetscObjectSetName((PetscObject)exact, "exact"));

  PetscCall(DMBFIterateOverFaces(dm, set_up_boundary_condition, &ctx));
  PetscCall(DMBFIterateOverCellsVectors(dm, interpolate_source, &ctx, PETSC_NULLPTR, 0, &rhs, 1));

  PetscCall(DMSetMatType(dm, MATSHELL));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(MatSetOperation(A, MATOP_MULT, (void (*)(void))apply_operator_mf));
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERVTK));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
  PetscCall(PetscViewerFileSetName(viewer, "ex2_solution.vtu"));
  PetscCall(VecView(rhs, viewer));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(DMBFIterateOverCellsVectors(dm, apply_mass_matrix, &ctx, PETSC_NULLPTR, 0, &rhs, 1));
  PetscCall(KSPSolve(ksp, rhs, sol));

  PetscCall(VecView(sol, viewer));
  PetscCall(DMBFIterateOverCellsVectors(dm, interpolate_exact, &ctx, PETSC_NULLPTR, 0, &exact, 1));
  PetscCall(VecView(exact, viewer));
  PetscCall(VecAXPY(exact, -1, sol));
  error = exact;
  PetscCall(PetscObjectSetName((PetscObject)error, "error"));
  PetscCall(VecAbs(error));
  PetscCall(VecView(error, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecNorm(error, NORM_INFINITY, &inf_norm));
  PetscCall(VecCopy(error, rhs));
  PetscCall(DMBFIterateOverCellsVectors(dm, apply_mass_matrix, &ctx, PETSC_NULLPTR, 0, &error, 1));
  PetscCall(VecDot(error, rhs, &l2_norm));
  l2_norm = PetscSqrtReal(l2_norm);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] Linf error: %1.15f\n", funcname, inf_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] L2 error:   %1.15f\n", funcname, l2_norm));

  // destroy Petsc objects
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&exact));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(KSPDestroy(&ksp));

  // end main
  PetscPrintf(PETSC_COMM_WORLD, "[%s] End\n", funcname);

  // finalize Petsc
  PetscCall(PetscFinalize());
  PetscFunctionReturn(0);
}

/*
./ex2 -dm_forest_topology brick \
      -dm_p4est_brick_size 2,2,2  \
      -dm_p4est_brick_bounds -1.0,1.0,-1.0,1.0,-1.0,1.0 \
      -dm_p4est_brick_periodicity 0,0,0 \
      -ksp_type cg -ksp_max_it 10000 -ksp_atol 1e-10 -ksp_rtol 1e-11 \
      -ksp_monitor -ksp_view -ksp_converged_reason \
      -dm_forest_initial_refinement 2 \
      -dm_forest_maximum_refinement 4

TODO check if all objects are destroyed
*/
