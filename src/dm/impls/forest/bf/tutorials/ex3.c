#include <petsc.h>
#include <petscdmbf.h>
#include <petscdmforest.h>

/* 3-D advection-diffusion problem with Neuman boundary conditions, and with dynamic-in-time AMR.
   We use an explicit (dimension-by-dimension) Lax-Wendrof scheme.  A matrix-free apply function for the PDE operator is used.
 */

static char help[] = "3-D advection-diffusion problem with dynamic AMR\n";

/*** Advection-Diffusion Parameters ***/

#define WINDX 2.0
#define WINDY 1.0
#define WINDZ 0.0

/*** Initial Condition Parameters ***/

#define U0_CENTER_X -0.5
#define U0_CENTER_Y -0.5
#define U0_CENTER_Z 0.25
#define U0_WIDTH    0.25

/*** Block Sizes, Defined Uniformly for All Cells ***/

#define NX 4
#define NY 4
#define NZ 4

// ensure block sizes are multiples of 2
#if (NX % 2) || (NY % 2) || (NZ % 2)
  #error "Block sizes must be multiples of 2."
#endif

#define NGUARD_X 1
#define NGUARD_Y 1
#define NGUARD_Z 1

// ensure guard layer has size=1
#if (NGUARD_X != 1) || (NGUARD_Y != 1) || (NGUARD_Z != 1)
  #error "Number of nodes in guard layer has to be one."
#endif

/*** Derived Sizes ***/

#define NX_GUARD (2 * NGUARD_X + NX)
#define NY_GUARD (2 * NGUARD_Y + NY)
#define NZ_GUARD (2 * NGUARD_Z + NZ)

#define NJG (NY_GUARD * NX_GUARD)
#define NIG (NX_GUARD)

#define NJ0 (NY * NX)
#define NI0 (NX)

/*** Derived Indices for Cells ***/

#define ILO_GUARD 0
#define JLO_GUARD 0
#define KLO_GUARD 0
#define ILO       NGUARD_X
#define JLO       NGUARD_Y
#define KLO       NGUARD_Z
#define IHI       (NGUARD_X + NX)
#define JHI       (NGUARD_Y + NY)
#define KHI       (NGUARD_Z + NZ)
#define IHI_GUARD (2 * NGUARD_X + NX)
#define JHI_GUARD (2 * NGUARD_Y + NY)
#define KHI_GUARD (2 * NGUARD_Z + NZ)

/*** Cell Data for DMBF ***/

// define variables for cell data as enumerator items
typedef enum {
  CELLDATA_XC,  // x-coordiantes in centers
  CELLDATA_YC,  // y-coordiantes in centers
  CELLDATA_ZC,  // z-coordiantes in centers
  CELLDATA_UNK, // unknowns/solution/degrees of freedom
  CELLDATA_N_   // number of variables (required)
} cellData_t;

// define the maximum number of entries for the shape of each variable
#define CELLDATA_D_ 4

// define shapes of variables as a matrix of size=(CELLDATA_N_ x CELLDATA_D_)
static const PetscInt CELLDATA_SHAPE[CELLDATA_N_ * CELLDATA_D_] = {
  /* CELLDATA_XC  */ NX_GUARD, 1,        0,        0,
  /* CELLDATA_YC  */ NY_GUARD, 1,        0,        0,
  /* CELLDATA_ZC  */ NZ_GUARD, 1,        0,        0,
  /* CELLDATA_UNK */ NX_GUARD, NY_GUARD, NZ_GUARD, 1};

/*** DM Context ***/

typedef PetscScalar (*SpaceTimeFn_3D)(PetscScalar, PetscScalar, PetscScalar, PetscReal);

typedef struct {
  SpaceTimeFn_3D bc, src, exact;
  PetscReal      time, dt;
  PetscInt       dynamic_amr_freq;
  PetscScalar   *buffer1, *buffer2;
} AppCtx;

/***************************************
 * COORDINATES
 **************************************/

static PetscErrorCode set_NAN_coordinates_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  size_t i, j, k; // indices for centers

  PetscFunctionBegin;
  for (i = ILO_GUARD; i < IHI_GUARD; i++) cell->data[CELLDATA_XC][i] = NAN;
  for (j = JLO_GUARD; j < JHI_GUARD; j++) cell->data[CELLDATA_YC][j] = NAN;
  for (k = KLO_GUARD; k < KHI_GUARD; k++) cell->data[CELLDATA_ZC][k] = NAN;
  PetscFunctionReturn(0);
}

PetscErrorCode print_coordinates_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  size_t i, j, k; // indices for centers

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_SELF, "%s: cell global index %i\n", __func__, cell->indexGlobal);
  PetscPrintf(PETSC_COMM_SELF, "  xc  i=[%i..%i] ", ILO_GUARD, IHI_GUARD);
  for (i = ILO_GUARD; i < IHI_GUARD; i++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[CELLDATA_XC][i]); }
  PetscPrintf(PETSC_COMM_SELF, "\n  yc  j=[%i..%i] ", JLO_GUARD, JHI_GUARD);
  for (j = JLO_GUARD; j < JHI_GUARD; j++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[CELLDATA_YC][j]); }
  PetscPrintf(PETSC_COMM_SELF, "\n  zc  k=[%i..%i] ", KLO_GUARD, KHI_GUARD);
  for (k = KLO_GUARD; k < KHI_GUARD; k++) { PetscPrintf(PETSC_COMM_SELF, "%g ", cell->data[CELLDATA_ZC][k]); }
  PetscPrintf(PETSC_COMM_SELF, "\n");
  PetscFunctionReturn(0);
}

static PetscErrorCode setup_coordinates_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  // define lengths and corners
  const PetscReal x_length = cell->sidelength[0];
  const PetscReal y_length = cell->sidelength[1];
  const PetscReal z_length = cell->sidelength[2];
  const PetscReal x_corner = cell->corner[0];
  const PetscReal y_corner = cell->corner[1];
  const PetscReal z_corner = cell->corner[2];
  const PetscReal hx       = 1.0 / ((PetscReal)(IHI - ILO));
  const PetscReal hy       = 1.0 / ((PetscReal)(JHI - JLO));
  const PetscReal hz       = 1.0 / ((PetscReal)(KHI - KLO));
  // indices
  size_t i, j, k;    // indices for centers
  size_t i0, j0, k0; // indices beginning at zero

  PetscFunctionBegin;
  CHKERRQ(set_NAN_coordinates_cellfn(dm, cell, ctx));
  // calculate centered coordinates
  for (i = ILO; i < IHI; i++) {
    i0                         = i - ILO;
    cell->data[CELLDATA_XC][i] = (PetscScalar)(((PetscReal)i0 + 0.5) * hx * x_length + x_corner);
  }
  for (j = JLO; j < JHI; j++) {
    j0                         = j - JLO;
    cell->data[CELLDATA_YC][j] = (PetscScalar)(((PetscReal)j0 + 0.5) * hy * y_length + y_corner);
  }
  for (k = KLO; k < KHI; k++) {
    k0                         = k - KLO;
    cell->data[CELLDATA_ZC][k] = (PetscScalar)(((PetscReal)k0 + 0.5) * hz * z_length + z_corner);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode setup_coordinates_guard_layer_facefn(DM dm, DM_BF_Face *face, void *ctx)
{
  // cell info
  const PetscInt  nCellsL    = face->nCellsL;
  const PetscInt  nCellsR    = face->nCellsR;
  const PetscBool isHangingL = (1 < nCellsL);
  const PetscBool isHangingR = (1 < nCellsR);
  // data
  PetscScalar *xc, *yc, *zc, scaleL, scaleR;
  // indices
  size_t cellId;

  PetscFunctionBegin;
  if (isHangingL) {
    scaleL = 2.0;
    scaleR = 0.5;
  } else if (isHangingR) {
    scaleL = 0.5;
    scaleR = 2.0;
  } else { // otherwise cells are not hanging
    scaleL = 1.0;
    scaleR = 1.0;
  }
  switch (face->dir) {
  case DM_BF_FACEDIR_XNEG:
  case DM_BF_FACEDIR_XPOS:
    // fill (low) guard of right cells
    for (cellId = 0; cellId < nCellsR; cellId++) {
      xc            = face->cellR[cellId]->data[CELLDATA_XC];
      xc[ILO_GUARD] = xc[ILO] - scaleR * (xc[ILO + 1] - xc[ILO]);
    }
    // fill (high) guard of left cells
    for (cellId = 0; cellId < nCellsL; cellId++) {
      xc                = face->cellL[cellId]->data[CELLDATA_XC];
      xc[IHI_GUARD - 1] = xc[IHI - 1] + scaleL * (xc[IHI - 1] - xc[IHI - 2]);
    }
    break;
  case DM_BF_FACEDIR_YNEG:
  case DM_BF_FACEDIR_YPOS:
    // fill (low) guard of y-top cells
    for (cellId = 0; cellId < nCellsR; cellId++) {
      yc            = face->cellR[cellId]->data[CELLDATA_YC];
      yc[JLO_GUARD] = yc[JLO] - scaleR * (yc[JLO + 1] - yc[JLO]);
    }
    // fill (high) guard of y-bottom cells
    for (cellId = 0; cellId < nCellsL; cellId++) {
      yc                = face->cellL[cellId]->data[CELLDATA_YC];
      yc[JHI_GUARD - 1] = yc[JHI - 1] + scaleL * (yc[JHI - 1] - yc[JHI - 2]);
    }
    break;
  case DM_BF_FACEDIR_ZNEG:
  case DM_BF_FACEDIR_ZPOS:
    // fill (low) guard of z-top cells
    for (cellId = 0; cellId < nCellsR; cellId++) {
      zc            = face->cellR[cellId]->data[CELLDATA_ZC];
      zc[KLO_GUARD] = zc[KLO] - scaleR * (zc[KLO + 1] - zc[KLO]);
    }
    // fill (high) guard of z-bottom cells
    for (cellId = 0; cellId < nCellsL; cellId++) {
      zc                = face->cellL[cellId]->data[CELLDATA_ZC];
      zc[KHI_GUARD - 1] = zc[KHI - 1] + scaleL * (zc[KHI - 1] - zc[KHI - 2]);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Unknown face direction %i", face->dir);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode setup_coordinates(DM dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMBFIterateOverCells(dm, setup_coordinates_cellfn, PETSC_NULLPTR));
  CHKERRQ(DMBFIterateOverFaces(dm, setup_coordinates_guard_layer_facefn, PETSC_NULLPTR));
  PetscFunctionReturn(0);
}

/***************************************
 * UNKNOWNS
 **************************************/

static PetscErrorCode set_NAN_unknowns_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  // data
  PetscScalar *unk = cell->data[CELLDATA_UNK];
  // indices
  size_t idx;

  PetscFunctionBegin;
  for (idx = 0; idx < NZ_GUARD * NY_GUARD * NX_GUARD; idx++) unk[idx] = NAN;
  PetscFunctionReturn(0);
}

PetscErrorCode set_NAN_unknowns(DM dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMBFIterateOverCells(dm, set_NAN_unknowns_cellfn, PETSC_NULLPTR));
  PetscFunctionReturn(0);
}

PetscErrorCode print_unknowns_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  // data
  const PetscScalar *unk = cell->data[CELLDATA_UNK];
  // indices
  size_t k, i;    // indices for centers
  int    jSigned; // indices for centers

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_SELF, "%s: cell global index %i\n", __func__, cell->indexGlobal);
  for (k = KLO_GUARD; k < KHI_GUARD; k++) {
    // Note: loop backwards over signed integer to avoid wrapping behavior
    for (jSigned = JHI_GUARD - 1; JLO_GUARD <= jSigned; jSigned--) {
      PetscPrintf(PETSC_COMM_SELF, "  ");
      for (i = ILO_GUARD; i < IHI_GUARD; i++) { PetscPrintf(PETSC_COMM_SELF, "%1.8f ", unk[k * NJG + jSigned * NIG + i]); }
      PetscPrintf(PETSC_COMM_SELF, "\n");
    }
    PetscPrintf(PETSC_COMM_SELF, "\n");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode set_unknowns_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  // data
  const PetscScalar *unkIn = cell->vecViewRead[0];
  PetscScalar       *unk   = cell->data[CELLDATA_UNK];
  // indices
  size_t i, j, k;    // indices for centers
  size_t i0, j0, k0; // indices beginning at zero

  PetscFunctionBegin;
  for (k = KLO; k < KHI; k++) {
    k0 = k - KLO;
    for (j = JLO; j < JHI; j++) {
      j0 = j - JLO;
      for (i = ILO; i < IHI; i++) {
        i0                         = i - ILO;
        unk[k * NJG + j * NIG + i] = unkIn[k0 * NJ0 + j0 * NI0 + i0];
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode get_unknowns_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  // data
  PetscScalar       *unkOut = cell->vecViewReadWrite[0];
  const PetscScalar *unk    = cell->data[CELLDATA_UNK];
  // indices
  size_t i, j, k;    // indices for centers
  size_t i0, j0, k0; // indices beginning at zero

  PetscFunctionBegin;
  for (k = KLO; k < KHI; k++) {
    k0 = k - KLO;
    for (j = JLO; j < JHI; j++) {
      j0 = j - JLO;
      for (i = ILO; i < IHI; i++) {
        i0                               = i - ILO;
        unkOut[k0 * NJ0 + j0 * NI0 + i0] = unk[k * NJG + j * NIG + i];
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode set_unknowns(DM dm, Vec unk)
{
  PetscFunctionBegin;
  CHKERRQ(DMBFIterateOverCellsVectors(dm, set_unknowns_cellfn, PETSC_NULLPTR, &unk, 1, PETSC_NULLPTR, 0));
  PetscFunctionReturn(0);
}

PetscErrorCode get_unknowns(DM dm, Vec unk)
{
  PetscFunctionBegin;
  CHKERRQ(DMBFIterateOverCellsVectors(dm, get_unknowns_cellfn, PETSC_NULLPTR, PETSC_NULLPTR, 0, &unk, 1));
  PetscFunctionReturn(0);
}

typedef struct _p_set_unknowns_from_function_ctx {
  SpaceTimeFn_3D fn;
  PetscReal      time;
} set_unknowns_from_function_ctx_t;

static PetscErrorCode set_unknowns_from_function_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  set_unknowns_from_function_ctx_t *c = ctx;
  // data
  const PetscScalar *xc  = cell->data[CELLDATA_XC];
  const PetscScalar *yc  = cell->data[CELLDATA_YC];
  const PetscScalar *zc  = cell->data[CELLDATA_ZC];
  PetscScalar       *unk = cell->data[CELLDATA_UNK];
  // indices
  size_t i, j, k; // indices for centers

  PetscFunctionBegin;
  for (k = KLO; k < KHI; k++) {
    for (j = JLO; j < JHI; j++) {
      for (i = ILO; i < IHI; i++) { unk[k * NJG + j * NIG + i] = c->fn(xc[i], yc[j], zc[k], c->time); }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode set_unknowns_from_function(DM dm, SpaceTimeFn_3D fn, PetscReal time)
{
  set_unknowns_from_function_ctx_t callback_ctx;

  PetscFunctionBegin;
  callback_ctx.fn   = fn;
  callback_ctx.time = time;
  CHKERRQ(DMBFIterateOverCells(dm, set_unknowns_from_function_cellfn, &callback_ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode fill_unknowns_guard_layer_facefn(DM dm, DM_BF_Face *face, void *ctx)
{
  AppCtx *c = ctx;
  // cell info
  const PetscInt  nCellsL    = face->nCellsL;
  const PetscInt  nCellsR    = face->nCellsR;
  const PetscBool isHangingL = (1 < nCellsL);
  const PetscBool isHangingR = (1 < nCellsR);
  const PetscBool isBoundary = (DM_BF_FACEBOUNDARY_NONE != face->boundary);
  // indices
  size_t i, j, k; // indices for centers

  PetscFunctionBegin;
  if (isBoundary) {
    PetscScalar *xc, *yc, *zc, *unk;

    switch (face->dir) {
    case DM_BF_FACEDIR_XNEG:
    case DM_BF_FACEDIR_YNEG:
    case DM_BF_FACEDIR_ZNEG:
      xc  = face->cellR[0]->data[CELLDATA_XC];
      yc  = face->cellR[0]->data[CELLDATA_YC];
      zc  = face->cellR[0]->data[CELLDATA_ZC];
      unk = face->cellR[0]->data[CELLDATA_UNK];
      break;
    case DM_BF_FACEDIR_XPOS:
    case DM_BF_FACEDIR_YPOS:
    case DM_BF_FACEDIR_ZPOS:
      xc  = face->cellL[0]->data[CELLDATA_XC];
      yc  = face->cellL[0]->data[CELLDATA_YC];
      zc  = face->cellL[0]->data[CELLDATA_ZC];
      unk = face->cellL[0]->data[CELLDATA_UNK];
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Unknown face direction %i", face->dir);
    }
    switch (face->dir) {
    case DM_BF_FACEDIR_XNEG:
      i = ILO_GUARD;
      break; // fill (low) guard of right cell
    case DM_BF_FACEDIR_XPOS:
      i = IHI_GUARD - 1;
      break; // fill (high) guard of left cell
    case DM_BF_FACEDIR_YNEG:
      j = JLO_GUARD;
      break; // fill (low) guard of y-top cell
    case DM_BF_FACEDIR_YPOS:
      j = JHI_GUARD - 1;
      break; // fill (high) guard of y-bottom cell
    case DM_BF_FACEDIR_ZNEG:
      k = KLO_GUARD;
      break; // fill (low) guard of z-top cell
    case DM_BF_FACEDIR_ZPOS:
      k = KHI_GUARD - 1;
      break; // fill (high) guard of z-bottom cell
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Unknown face direction %i", face->dir);
    }
    switch (face->dir) {
    case DM_BF_FACEDIR_XNEG:
    case DM_BF_FACEDIR_XPOS:
      for (k = KLO; k < KHI; k++) {
        for (j = JLO; j < JHI; j++) { unk[k * NJG + j * NIG + i] = c->bc(xc[i], yc[j], zc[k], c->time); }
      }
      break;
    case DM_BF_FACEDIR_YNEG:
    case DM_BF_FACEDIR_YPOS:
      for (k = KLO; k < KHI; k++) {
        for (i = ILO; i < IHI; i++) { unk[k * NJG + j * NIG + i] = c->bc(xc[i], yc[j], zc[k], c->time); }
      }
      break;
    case DM_BF_FACEDIR_ZNEG:
    case DM_BF_FACEDIR_ZPOS:
      for (j = JLO; j < JHI; j++) {
        for (i = ILO; i < IHI; i++) { unk[k * NJG + j * NIG + i] = c->bc(xc[i], yc[j], zc[k], c->time); }
      }
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Unknown face direction %i", face->dir);
    }

  } else { // if !isBoundary

    PetscScalar *xcL, *ycL, *zcL, *unkL;
    PetscScalar *xcR, *ycR, *zcR, *unkR;
#if 0
    size_t            ilo, ihi, jlo, jhi, klo, khi;
    size_t            iL, iR, jL, jR, kL, kR;
#endif
    size_t cellId;

    switch (face->dir) {
    case DM_BF_FACEDIR_XNEG:
    case DM_BF_FACEDIR_XPOS:
      if (isHangingL) {
        // project to coarse and fill (low) guard of 1 right cell
        xcR  = face->cellR[0]->data[CELLDATA_XC];
        ycR  = face->cellR[0]->data[CELLDATA_YC];
        zcR  = face->cellR[0]->data[CELLDATA_ZC];
        unkR = face->cellR[0]->data[CELLDATA_UNK];
#if 0 // Note: Implementing interpolation functions takes some effort, and to keep \
      //       this example short, we cheat and use the exact solution. \
      //       One simple way of interpolating is shown in the commented code.
          for (cellId=0; cellId<nCellsL; cellId++) {
            xcL  = face->cellL[cellId]->data[CELLDATA_XC];
            ycL  = face->cellL[cellId]->data[CELLDATA_YC];
            zcL  = face->cellL[cellId]->data[CELLDATA_ZC];
            unkL = face->cellL[cellId]->data[CELLDATA_UNK];
            jlo  = JLO + (cellId % 2)*(NY/2);
            jhi  = JLO + (cellId % 2)*(NY/2) + NY/2;
            klo  = KLO + (cellId / 2)*(NZ/2);
            khi  = KLO + (cellId / 2)*(NZ/2) + NZ/2;
            for (k=klo; k<khi; k++) {
              // find index `kL` such that `zcL[kL]<=zcR[k]<zcL[kL+1]`
              for (kL=KLO; kL<KHI-1; kL++) { if (zcL[kL] <= zcR[k] && zcR[k] < zcL[kL+1]) break; }
              for (j=jlo; j<jhi; j++) {
                // find index `jL` such that `ycL[jL]<=ycR[j]<ycL[jL+1]`
                for (jL=JLO; jL<JHI-1; jL++) { if (ycL[jL] <= ycR[j] && ycR[j] < ycL[jL+1]) break; }
                // average
                unkR[k*NJG+j*NIG+(ILO_GUARD)] = 0.25 * (unkL[kL    *NJG+jL    *NIG+(IHI-1)] +
                                                        unkL[kL    *NJG+(jL+1)*NIG+(IHI-1)] +
                                                        unkL[(kL+1)*NJG+jL    *NIG+(IHI-1)] +
                                                        unkL[(kL+1)*NJG+(jL+1)*NIG+(IHI-1)]);
              }
            }
          }
#else
        for (k = KLO; k < KHI; k++) {
          for (j = JLO; j < JHI; j++) { unkR[k * NJG + j * NIG + (ILO_GUARD)] = c->exact(xcR[ILO_GUARD], ycR[j], zcR[k], c->time); }
        }
#endif
        // project to fine and fill (high) guard of 4 left cells
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        for (cellId = 0; cellId < nCellsL; cellId++) {
          xcL  = face->cellL[cellId]->data[CELLDATA_XC];
          ycL  = face->cellL[cellId]->data[CELLDATA_YC];
          zcL  = face->cellL[cellId]->data[CELLDATA_ZC];
          unkL = face->cellL[cellId]->data[CELLDATA_UNK];
          for (k = KLO; k < KHI; k++) {
            for (j = JLO; j < JHI; j++) { unkL[k * NJG + j * NIG + (IHI_GUARD - 1)] = c->exact(xcL[IHI_GUARD - 1], ycL[j], zcL[k], c->time); }
          }
        }
      } else if (isHangingR) {
        // project to fine and fill (low) guard of 4 right cells
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        for (cellId = 0; cellId < nCellsR; cellId++) {
          xcR  = face->cellR[cellId]->data[CELLDATA_XC];
          ycR  = face->cellR[cellId]->data[CELLDATA_YC];
          zcR  = face->cellR[cellId]->data[CELLDATA_ZC];
          unkR = face->cellR[cellId]->data[CELLDATA_UNK];
          for (k = KLO; k < KHI; k++) {
            for (j = JLO; j < JHI; j++) { unkR[k * NJG + j * NIG + (ILO_GUARD)] = c->exact(xcR[ILO_GUARD], ycR[j], zcR[k], c->time); }
          }
        }
        // project to coarsse and fill (high) guard of 1 left cell
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        xcL  = face->cellL[0]->data[CELLDATA_XC];
        ycL  = face->cellL[0]->data[CELLDATA_YC];
        zcL  = face->cellL[0]->data[CELLDATA_ZC];
        unkL = face->cellL[0]->data[CELLDATA_UNK];
        for (k = KLO; k < KHI; k++) {
          for (j = JLO; j < JHI; j++) { unkL[k * NJG + j * NIG + (IHI_GUARD - 1)] = c->exact(xcL[IHI_GUARD - 1], ycL[j], zcL[k], c->time); }
        }
      } else { /* otherwise cells are not hanging */
        unkL = face->cellL[0]->data[CELLDATA_UNK];
        unkR = face->cellR[0]->data[CELLDATA_UNK];
        for (k = KLO; k < KHI; k++) {
          for (j = JLO; j < JHI; j++) {
            unkR[k * NJG + j * NIG + (ILO_GUARD)]     = unkL[k * NJG + j * NIG + (IHI - 1)]; // fill (low) guard of right cell
            unkL[k * NJG + j * NIG + (IHI_GUARD - 1)] = unkR[k * NJG + j * NIG + (ILO)];     // fill (high) guard of left cell
          }
        }
      }
      break;
    case DM_BF_FACEDIR_YNEG:
    case DM_BF_FACEDIR_YPOS:
      if (isHangingL) {
        // project to coarse and fill (low) guard of 1 y-top cell
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        xcR  = face->cellR[0]->data[CELLDATA_XC];
        ycR  = face->cellR[0]->data[CELLDATA_YC];
        zcR  = face->cellR[0]->data[CELLDATA_ZC];
        unkR = face->cellR[0]->data[CELLDATA_UNK];
        for (k = KLO; k < KHI; k++) {
          for (i = ILO; i < IHI; i++) { unkR[k * NJG + (JLO_GUARD)*NIG + i] = c->exact(xcR[i], ycR[JLO_GUARD], zcR[k], c->time); }
        }
        // project to fine and fill (high) guard of 4 y-bottom cells
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        for (cellId = 0; cellId < nCellsL; cellId++) {
          xcL  = face->cellL[cellId]->data[CELLDATA_XC];
          ycL  = face->cellL[cellId]->data[CELLDATA_YC];
          zcL  = face->cellL[cellId]->data[CELLDATA_ZC];
          unkL = face->cellL[cellId]->data[CELLDATA_UNK];
          for (k = KLO; k < KHI; k++) {
            for (i = ILO; i < IHI; i++) { unkL[k * NJG + (JHI_GUARD - 1) * NIG + i] = c->exact(xcL[i], ycL[JHI_GUARD - 1], zcL[k], c->time); }
          }
        }
      } else if (isHangingR) {
        // project to coarse and fill (low) guard of 4 y-top cells
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        for (cellId = 0; cellId < nCellsR; cellId++) {
          xcR  = face->cellR[cellId]->data[CELLDATA_XC];
          ycR  = face->cellR[cellId]->data[CELLDATA_YC];
          zcR  = face->cellR[cellId]->data[CELLDATA_ZC];
          unkR = face->cellR[cellId]->data[CELLDATA_UNK];
          for (k = KLO; k < KHI; k++) {
            for (i = ILO; i < IHI; i++) { unkR[k * NJG + (JLO_GUARD)*NIG + i] = c->exact(xcR[i], ycR[JLO_GUARD], zcR[k], c->time); }
          }
        }
        // project to fine and fill (high) guard of 1 y-bottom cells
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        xcL  = face->cellL[0]->data[CELLDATA_XC];
        ycL  = face->cellL[0]->data[CELLDATA_YC];
        zcL  = face->cellL[0]->data[CELLDATA_ZC];
        unkL = face->cellL[0]->data[CELLDATA_UNK];
        for (k = KLO; k < KHI; k++) {
          for (i = ILO; i < IHI; i++) { unkL[k * NJG + (JHI_GUARD - 1) * NIG + i] = c->exact(xcL[i], ycL[JHI_GUARD - 1], zcL[k], c->time); }
        }
      } else { /* otherwise cells are not hanging */
        unkL = face->cellL[0]->data[CELLDATA_UNK];
        unkR = face->cellR[0]->data[CELLDATA_UNK];
        for (k = KLO; k < KHI; k++) {
          for (i = ILO; i < IHI; i++) {
            unkR[k * NJG + (JLO_GUARD)*NIG + i]       = unkL[k * NJG + (JHI - 1) * NIG + i]; // fill (low) guard of y-top cell
            unkL[k * NJG + (JHI_GUARD - 1) * NIG + i] = unkR[k * NJG + (JLO)*NIG + i];       // fill (high) guard of y-bottom cell
          }
        }
      }
      break;
    case DM_BF_FACEDIR_ZNEG:
    case DM_BF_FACEDIR_ZPOS:
      if (isHangingL) {
        // project to coarse and fill (low) guard of 1 z-top cell
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        xcR  = face->cellR[0]->data[CELLDATA_XC];
        ycR  = face->cellR[0]->data[CELLDATA_YC];
        zcR  = face->cellR[0]->data[CELLDATA_ZC];
        unkR = face->cellR[0]->data[CELLDATA_UNK];
        for (j = JLO; j < JHI; j++) {
          for (i = ILO; i < IHI; i++) { unkR[(KLO_GUARD)*NJG + j * NIG + i] = c->exact(xcR[i], ycR[j], zcR[KLO_GUARD], c->time); }
        }
        // project to fine and fill (high) guard of 4 z-bottom cells
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        for (cellId = 0; cellId < nCellsL; cellId++) {
          xcL  = face->cellL[cellId]->data[CELLDATA_XC];
          ycL  = face->cellL[cellId]->data[CELLDATA_YC];
          zcL  = face->cellL[cellId]->data[CELLDATA_ZC];
          unkL = face->cellL[cellId]->data[CELLDATA_UNK];
          for (j = JLO; j < JHI; j++) {
            for (i = ILO; i < IHI; i++) { unkL[(KHI_GUARD - 1) * NJG + j * NIG + i] = c->exact(xcL[i], ycL[j], zcL[KHI_GUARD - 1], c->time); }
          }
        }
      } else if (isHangingR) {
        // project to coarse and fill (low) guard of 4 z-top cells
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        for (cellId = 0; cellId < nCellsR; cellId++) {
          xcR  = face->cellR[cellId]->data[CELLDATA_XC];
          ycR  = face->cellR[cellId]->data[CELLDATA_YC];
          zcR  = face->cellR[cellId]->data[CELLDATA_ZC];
          unkR = face->cellR[cellId]->data[CELLDATA_UNK];
          for (j = JLO; j < JHI; j++) {
            for (i = ILO; i < IHI; i++) { unkR[(KLO_GUARD)*NJG + j * NIG + i] = c->exact(xcR[i], ycR[j], zcR[KLO_GUARD], c->time); }
          }
        }
        // project to fine and fill (high) guard of 1 z-bottom cells
        // Note: Implementing interpolation functions takes some effort, and to keep
        //       this example short, we cheat and use the exact solution.
        xcL  = face->cellL[0]->data[CELLDATA_XC];
        ycL  = face->cellL[0]->data[CELLDATA_YC];
        zcL  = face->cellL[0]->data[CELLDATA_ZC];
        unkL = face->cellL[0]->data[CELLDATA_UNK];
        for (j = JLO; j < JHI; j++) {
          for (i = ILO; i < IHI; i++) { unkL[(KHI_GUARD - 1) * NJG + j * NIG + i] = c->exact(xcL[i], ycL[j], zcL[KHI_GUARD - 1], c->time); }
        }
      } else { /* otherwise cells are not hanging */
        unkL = face->cellL[0]->data[CELLDATA_UNK];
        unkR = face->cellR[0]->data[CELLDATA_UNK];
        for (j = JLO; j < JHI; j++) {
          for (i = ILO; i < IHI; i++) {
            unkR[(KLO_GUARD)*NJG + j * NIG + i]       = unkL[(KHI - 1) * NJG + j * NIG + i]; // fill (low) guard of z-top cell
            unkL[(KHI_GUARD - 1) * NJG + j * NIG + i] = unkR[(KLO)*NJG + j * NIG + i];       // fill (high) guard of z-bottom cell
          }
        }
      }
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Unknown face direction %i", face->dir);
    }
  }
  PetscFunctionReturn(0);
}

/***************************************
 * OPERATORS
 **************************************/

static inline PetscScalar lax_wendroff_1d(const PetscScalar u[3], const PetscScalar x[3], PetscReal dt, PetscScalar coeff)
{
  PetscScalar cm, cp, cc;

  cm = coeff * dt / (x[1] - x[0]);
  cp = coeff * dt / (x[2] - x[1]);
  cc = coeff * dt / (x[2] - x[0]);

  return cc * (cm + 1.0) * u[0] + (1.0 - cc * (cm + cp)) * u[1] + cc * (cp - 1.0) * u[2];
}

/**
 * Applies an approximation of a finite difference scheme, called Lax-Wendroff
 * (LW).  Several shortcuts/approximations are taken to make the implementation
 * simple and short in three dimensions.  What suffers is the accuracy.  That
 * is why for simulations, where accuracy matters, this particular
 * approximation of LW is not recommended.
 */
static PetscErrorCode apply_operator_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  AppCtx *c = ctx;
  // data
  const PetscReal    dt   = c->dt;
  const PetscScalar *xc   = cell->data[CELLDATA_XC];
  const PetscScalar *yc   = cell->data[CELLDATA_YC];
  const PetscScalar *zc   = cell->data[CELLDATA_ZC];
  PetscScalar       *unk  = cell->data[CELLDATA_UNK];
  PetscScalar       *buf1 = c->buffer1;
  PetscScalar       *buf2 = c->buffer2;
  PetscScalar        u[3];
  // indices
  size_t i, j, k; // indices for centers

  PetscFunctionBegin;
  // compute along x-direction
  for (k = KLO; k < KHI; k++) {
    for (j = JLO; j < JHI; j++) {
      for (i = ILO; i < IHI; i++) {
        u[0]                        = unk[k * NJG + j * NIG + (i - 1)];
        u[1]                        = unk[k * NJG + j * NIG + (i)];
        u[2]                        = unk[k * NJG + j * NIG + (i + 1)];
        buf1[k * NJG + j * NIG + i] = lax_wendroff_1d(u, &xc[i - 1], dt, WINDX);
      }
    }
  }

  // fill buffer
  j = JLO_GUARD;
  for (k = KLO; k < KHI; k++) {
    for (i = ILO; i < IHI; i++) { buf1[k * NJG + j * NIG + i] = unk[k * NJG + j * NIG + i]; }
  }
  j = JHI_GUARD - 1;
  for (k = KLO; k < KHI; k++) {
    for (i = ILO; i < IHI; i++) { buf1[k * NJG + j * NIG + i] = unk[k * NJG + j * NIG + i]; }
  }
  // compute along y-direction
  for (k = KLO; k < KHI; k++) {
    for (j = JLO; j < JHI; j++) {
      for (i = ILO; i < IHI; i++) {
        u[0]                        = buf1[k * NJG + (j - 1) * NIG + i];
        u[1]                        = buf1[k * NJG + (j)*NIG + i];
        u[2]                        = buf1[k * NJG + (j + 1) * NIG + i];
        buf2[k * NJG + j * NIG + i] = lax_wendroff_1d(u, &yc[j - 1], dt, WINDY);
      }
    }
  }

  // fill buffer
  k = KLO_GUARD;
  for (j = JLO; j < JHI; j++) {
    for (i = ILO; i < IHI; i++) { buf2[k * NJG + j * NIG + i] = unk[k * NJG + j * NIG + i]; }
  }
  k = KHI_GUARD - 1;
  for (j = JLO; j < JHI; j++) {
    for (i = ILO; i < IHI; i++) { buf2[k * NJG + j * NIG + i] = unk[k * NJG + j * NIG + i]; }
  }
  // compute along z-direction
  for (k = KLO; k < KHI; k++) {
    for (j = JLO; j < JHI; j++) {
      for (i = ILO; i < IHI; i++) {
        u[0]                       = buf2[(k - 1) * NJG + j * NIG + i];
        u[1]                       = buf2[(k)*NJG + j * NIG + i];
        u[2]                       = buf2[(k + 1) * NJG + j * NIG + i];
        unk[k * NJG + j * NIG + i] = lax_wendroff_1d(u, &zc[k - 1], dt, WINDZ) + c->src(xc[i], yc[j], zc[k], c->time);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode apply_operator(DM dm)
{
  AppCtx        *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetApplicationContext(dm, &ctx);
  CHKERRQ(ierr);
  ierr = DMBFCommunicateGhostCells(dm);
  CHKERRQ(ierr);
  ierr = DMBFIterateOverFaces(dm, fill_unknowns_guard_layer_facefn, ctx);
  CHKERRQ(ierr);
  ierr = DMBFIterateOverCells(dm, apply_operator_cellfn, ctx);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************************************
 * AMR
 **************************************/

static PetscErrorCode amr_refine_at_initial_condition(DM dm, DM_BF_Cell *cell, void *ctx)
{
  // data
  const PetscScalar *xc = cell->data[CELLDATA_XC];
  const PetscScalar *yc = cell->data[CELLDATA_YC];
  const PetscScalar *zc = cell->data[CELLDATA_ZC];
  DMAdaptFlag        flagx, flagy, flagz;
  // indices
  size_t i, j, k; // indices for centers

  PetscFunctionBegin;
  // check if any x-coordinate is inside sphere of refinement
  flagx = DM_ADAPT_KEEP;
  for (i = ILO; i < IHI; i++) {
    if (PetscAbsScalar(xc[i] - U0_CENTER_X) < 2.0 * U0_WIDTH) flagx = DM_ADAPT_REFINE;
  }
  // check if any y-coordinate is inside sphere of refinement
  flagy = DM_ADAPT_KEEP;
  for (j = JLO; j < JHI; j++) {
    if (PetscAbsScalar(yc[j] - U0_CENTER_Y) < 2.0 * U0_WIDTH) flagy = DM_ADAPT_REFINE;
  }
  // check if any z-coordinate is inside sphere of refinement
  flagz = DM_ADAPT_KEEP;
  for (k = KLO; k < KHI; k++) {
    if (PetscAbsScalar(zc[k] - U0_CENTER_Z) < 2.0 * U0_WIDTH) flagz = DM_ADAPT_REFINE;
  }
  // refine if all flags for x, y, z-coordinates are set for refinement
  if (DM_ADAPT_REFINE == flagx && DM_ADAPT_REFINE == flagy && DM_ADAPT_REFINE == flagz) {
    cell->adaptFlag = DM_ADAPT_REFINE;
  } else {
    cell->adaptFlag = DM_ADAPT_KEEP;
  }
  PetscFunctionReturn(0);
}

/**
 * Projects from coarse to fine or from fine to coarse cells.
 * Note: Implementing interpolation functions takes some effort, and to keep
 *       this example short, we cheat and use the exact solution.
 */
static PetscErrorCode amr_project(DM dm, DM_BF_Cell **origCells, PetscInt nOrigCells, DM_BF_Cell **projCells, PetscInt nProjCells, void *ctx)
{
  AppCtx *c = ctx;
  // data
  PetscScalar *xc, *yc, *zc, *unk;
  // indices
  size_t i, j, k; // indices for centers
  size_t cellId;

  PetscFunctionBegin;
  for (cellId = 0; cellId < nProjCells; cellId++) {
    CHKERRQ(setup_coordinates_cellfn(dm, projCells[cellId], ctx));
    xc  = projCells[cellId]->data[CELLDATA_XC];
    yc  = projCells[cellId]->data[CELLDATA_YC];
    zc  = projCells[cellId]->data[CELLDATA_ZC];
    unk = projCells[cellId]->data[CELLDATA_UNK];
    for (k = KLO; k < KHI; k++) {
      for (j = JLO; j < JHI; j++) {
        for (i = ILO; i < IHI; i++) { unk[k * NJG + j * NIG + i] = c->exact(xc[i], yc[j], zc[k], c->time); }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode amr_project_no_op(DM dm, DM_BF_Cell **origCells, PetscInt nOrigCells, DM_BF_Cell **projCells, PetscInt nProjCells, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode amr_flag_logDR(DM dm, DM_BF_Cell *cell, void *ctx)
{
  // data
  const PetscScalar *unk       = cell->data[CELLDATA_UNK];
  const PetscReal    tol_lo    = 0.5;
  const PetscReal    tol_hi    = 5.0;
  PetscReal          threshold = 1.0e-4;
  PetscReal          magn_min, magn_max;
  PetscReal          indicator;
  // indices
  size_t i, j, k; // indices for centers

  PetscFunctionBegin;
  // compute min and max magnitudes inside the cell
  magn_min = magn_max = PetscAbsScalar(unk[KLO * NJG + JLO * NIG + ILO]);
  for (k = KLO; k < KHI; k++) {
    for (j = JLO; j < JHI; j++) {
      for (i = ILO; i < IHI; i++) {
        magn_min = (PetscReal)PetscMin(magn_min, PetscAbsScalar(unk[k * NJG + j * NIG + i]));
        magn_max = (PetscReal)PetscMax(magn_max, PetscAbsScalar(unk[k * NJG + j * NIG + i]));
      }
    }
  }
  // enforce lower magnitude to be positive
  magn_min = 1.0e-30 + PetscMax(1.0e-100, magn_min);
  // enforce upper magnitude to be >=magn_min
  magn_max = PetscMax(magn_min, magn_max);
  // compute indicator for this cell
  if (threshold < magn_max) {
    indicator = PetscLogReal(magn_max / magn_min);
  } else {
    indicator = 0.0;
  }
  // choose flag
  if (indicator < tol_lo) cell->adaptFlag = DM_ADAPT_COARSEN;
  else if (tol_hi < indicator) cell->adaptFlag = DM_ADAPT_REFINE;
  else cell->adaptFlag = DM_ADAPT_KEEP;
  PetscFunctionReturn(0);
}

/***************************************
 * TIME EVOLUTION
 **************************************/

static PetscErrorCode visualize_timestep(MPI_Comm comm, Vec sol, PetscInt timestep)
{
  PetscViewer    viewer;
  char           filepath[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(filepath, sizeof(filepath), "ex3_solution_ts%04i.vtu", timestep);
  CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &viewer);
  CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);
  CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filepath);
  CHKERRQ(ierr);
  ierr = VecView(sol, viewer);
  CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct _p_compute_error_ctx {
  SpaceTimeFn_3D fn;
  PetscReal      time;
  PetscReal      error_sum_loc;
  PetscReal      exact_sum_loc;
} compute_error_ctx_t;

static PetscErrorCode compute_error_at_timestep_cellfn(DM dm, DM_BF_Cell *cell, void *ctx)
{
  compute_error_ctx_t *c = ctx;
  // data
  const PetscScalar *xc  = cell->data[CELLDATA_XC];
  const PetscScalar *yc  = cell->data[CELLDATA_YC];
  const PetscScalar *zc  = cell->data[CELLDATA_ZC];
  const PetscScalar *unk = cell->data[CELLDATA_UNK];
  const PetscScalar  hx  = (xc[IHI - 1] - xc[ILO]) / ((PetscScalar)(NX - 1));
  const PetscScalar  hy  = (yc[JHI - 1] - yc[JLO]) / ((PetscScalar)(NY - 1));
  const PetscScalar  hz  = (zc[KHI - 1] - zc[KLO]) / ((PetscScalar)(NZ - 1));
  PetscScalar        exact, error;
  // indices
  size_t i, j, k; // indices for centers

  PetscFunctionBegin;
  for (k = KLO; k < KHI; k++) {
    for (j = JLO; j < JHI; j++) {
      for (i = ILO; i < IHI; i++) {
        exact = c->fn(xc[i], yc[j], zc[k], c->time);
        error = unk[k * NJG + j * NIG + i] - exact;
        c->exact_sum_loc += (PetscReal)(exact * (hx * hy * hz));
        c->error_sum_loc += (PetscReal)(error * (hx * hy * hz) * error);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode compute_error_at_timestep(DM dm, SpaceTimeFn_3D exact, PetscReal time, PetscReal *error_abs, PetscReal *error_rel)
{
  MPI_Comm            dmComm;
  PetscReal           error_glo, exact_glo;
  PetscErrorCode      ierr;
  compute_error_ctx_t callback_ctx;

  PetscFunctionBegin;
  callback_ctx.fn            = exact;
  callback_ctx.time          = time;
  callback_ctx.error_sum_loc = 0.0;
  callback_ctx.exact_sum_loc = 0.0;
  ierr                       = DMBFIterateOverCells(dm, compute_error_at_timestep_cellfn, &callback_ctx);
  CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &dmComm);
  CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&callback_ctx.error_sum_loc, &error_glo, 1, MPIU_REAL, MPI_SUM, dmComm);
  CHKERRMPI(ierr);
  ierr = MPIU_Allreduce(&callback_ctx.exact_sum_loc, &exact_glo, 1, MPIU_REAL, MPI_SUM, dmComm);
  CHKERRMPI(ierr);
  if (error_abs) { *error_abs = PetscSqrtReal(error_glo); }
  if (error_rel) { *error_rel = PetscSqrtReal(error_glo) / exact_glo; }
  PetscFunctionReturn(0);
}

PetscErrorCode evolve(DM *dm, PetscInt n_timesteps, Vec *sol, PetscBool useInit)
{
  AppCtx         *ctx;
  MPI_Comm        dmComm;
  DM_BF_AmrOps    amrOps;
  DM              adapt;
  const PetscBool write_vis = PETSC_TRUE;
  PetscInt        timestep;
  PetscReal       error_abs, error_rel;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)*dm, &dmComm);
  CHKERRQ(ierr);
  ierr = DMGetApplicationContext(*dm, &ctx);
  CHKERRQ(ierr);
  // setup dynamic AMR
  amrOps.setAmrFlag         = amr_flag_logDR;
  amrOps.setAmrFlagCtx      = PETSC_NULLPTR;
  amrOps.projectToCoarse    = amr_project;
  amrOps.projectToFine      = amr_project;
  amrOps.projectToFineCtx   = ctx;
  amrOps.projectToCoarseCtx = ctx;
  ierr                      = DMBFAMRSetOperators(*dm, &amrOps);
  CHKERRQ(ierr);
  // set initial condition
  if (useInit) {
    ierr = set_unknowns(*dm, *sol);
    CHKERRQ(ierr);
  }
  // run time steps
  for (timestep = 0; timestep < n_timesteps; timestep++) {
    ierr = compute_error_at_timestep(*dm, ctx->exact, ctx->time, &error_abs, &error_rel);
    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] %4i time step, t=%.8f, dt=%g, L2-norm error abs=%.4e rel=%.4e\n", __func__, timestep, ctx->time, ctx->dt, error_abs, error_rel);
    if (!(timestep % ctx->dynamic_amr_freq)) { // if perform AMR
      ierr = VecDestroy(sol);
      CHKERRQ(ierr);

      PetscPrintf(PETSC_COMM_WORLD, "[%s] Run dynamic AMR\n", __func__);
      ierr = DMBFAMRFlag(*dm);
      CHKERRQ(ierr);
      ierr = DMBFAMRAdapt(*dm, &adapt);
      CHKERRQ(ierr);
      ierr = DMDestroy(dm);
      CHKERRQ(ierr);
      *dm = adapt;

      // re-set DM context
      ierr = DMSetApplicationContext(*dm, ctx);
      CHKERRQ(ierr);
      // re-initialize coordinates
      ierr = setup_coordinates(*dm);
      // re-create solution vector
      ierr = DMCreateGlobalVector(*dm, sol);
      CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)*sol, "solution");
      CHKERRQ(ierr);
    }
    if (write_vis) { // if write vis files
      ierr = get_unknowns(*dm, *sol);
      CHKERRQ(ierr);
      ierr = visualize_timestep(dmComm, *sol, timestep);
      CHKERRQ(ierr);
    }
    ierr = apply_operator(*dm);
    CHKERRQ(ierr);
    ctx->time += ctx->dt;
  }
  ierr = compute_error_at_timestep(*dm, ctx->exact, ctx->time, &error_abs, &error_rel);
  CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "[%s] %4i time step, t=%.8f, dt=%g, L2-norm error abs=%.4e rel=%.4e\n", __func__, timestep, ctx->time, ctx->dt, error_abs, error_rel);
  // store solution
  ierr = get_unknowns(*dm, *sol);
  CHKERRQ(ierr);
  if (write_vis) { // if write vis files
    ierr = visualize_timestep(dmComm, *sol, timestep);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/***************************************
 * MAIN
 **************************************/

static inline PetscScalar u0(PetscScalar x, PetscScalar y, PetscScalar z)
{
  const PetscScalar dx = x - U0_CENTER_X;
  const PetscScalar dy = y - U0_CENTER_Y;
  const PetscScalar dz = z - U0_CENTER_Z;

  return 1.0 / (U0_WIDTH * PetscSqrtScalar(2.0 * PETSC_PI)) * PetscExpScalar(-(dx * dx + dy * dy + dz * dz) / (2.0 * U0_WIDTH * U0_WIDTH));
}

static inline PetscScalar u(PetscScalar x, PetscScalar y, PetscScalar z, PetscReal t)
{
  return u0(x - WINDX * t, y - WINDY * t, z - WINDZ * t);
}

static inline PetscScalar g(PetscScalar x, PetscScalar y, PetscScalar z, PetscReal t)
{
  return 0.0; // Neumann boundary
}

static inline PetscScalar f(PetscScalar x, PetscScalar y, PetscScalar z, PetscReal t)
{
  return 0.0; // no sources
}

int main(int argc, char **argv)
{
  const char     funcname[] = "DMBF-Advection-3D";
  DM             dm;
  PetscInt       blockSize[3] = {NX, NY, NZ};
  PetscInt       n_timesteps;
  PetscReal      max_time = 1.5;
  Vec            sol;
  AppCtx         ctx;
  PetscErrorCode ierr;

  // initialize Petsc
  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (ierr) return ierr;

  PetscPrintf(PETSC_COMM_WORLD, "[%s] Begin\n", funcname);

  // create DM
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Create DM\n", funcname);
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);
  CHKERRQ(ierr);
  ierr = DMSetType(dm, "bf");
  CHKERRQ(ierr);

  // set DM options
  ierr = DMSetDimension(dm, 3);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);
  CHKERRQ(ierr);
  // set cell data shapes
  ierr = DMBFSetCellDataShape(dm, CELLDATA_SHAPE, CELLDATA_N_, CELLDATA_D_);
  CHKERRQ(ierr);
  //ierr = DMBFSetCellDataVSize(dm,sizeof(cellData_t));CHKERRQ(ierr); //TODO unused at the moment
  ierr = DMBFSetBlockSize(dm, blockSize);
  CHKERRQ(ierr);

  // set application-specific data
  ctx.bc               = g;
  ctx.src              = f;
  ctx.exact            = u;
  ctx.time             = 0.0;
  ctx.dynamic_amr_freq = 5;
  ierr                 = PetscMalloc2(NX_GUARD * NY_GUARD * NZ_GUARD, &ctx.buffer1, NX_GUARD * NY_GUARD * NZ_GUARD, &ctx.buffer2);
  CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);
  CHKERRQ(ierr);

  // setup DM
  ierr = DMSetUp(dm);
  CHKERRQ(ierr);

  // initialize coordinates
  ierr = setup_coordinates(dm);
  // initialize unknowns
  ierr = set_NAN_unknowns(dm);
  CHKERRQ(ierr);
  ierr = set_unknowns_from_function(dm, u, 0.0 /*time*/);
  CHKERRQ(ierr);

  // run initial AMR
  {
    PetscInt     amr_step, n_amr_steps = 1;
    DM_BF_AmrOps amrOps;
    DM           adapt;

    amrOps.setAmrFlag         = amr_refine_at_initial_condition;
    amrOps.setAmrFlagCtx      = PETSC_NULLPTR;
    amrOps.projectToCoarse    = amr_project_no_op;
    amrOps.projectToFine      = amr_project_no_op;
    amrOps.projectToFineCtx   = PETSC_NULLPTR;
    amrOps.projectToCoarseCtx = PETSC_NULLPTR;
    ierr                      = DMBFAMRSetOperators(dm, &amrOps);
    CHKERRQ(ierr);

    for (amr_step = 0; amr_step < n_amr_steps; amr_step++) {
      PetscPrintf(PETSC_COMM_WORLD, "[%s] Run initial AMR step %i of %i\n", funcname, amr_step, n_amr_steps);
      ierr = DMBFAMRFlag(dm);
      CHKERRQ(ierr);
      ierr = DMBFAMRAdapt(dm, &adapt);
      CHKERRQ(ierr);
      ierr = DMDestroy(&dm);
      CHKERRQ(ierr);
      dm = adapt;

      // re-set DM context
      ierr = DMSetApplicationContext(dm, &ctx);
      CHKERRQ(ierr);
      // re-initialize coordinates
      ierr = setup_coordinates(dm);
      // re-initialize unknowns
      ierr = set_NAN_unknowns(dm);
      CHKERRQ(ierr);
      ierr = set_unknowns_from_function(dm, u, 0.0 /*time*/);
      CHKERRQ(ierr);
    }
    PetscPrintf(PETSC_COMM_WORLD, "[%s] Finished initial AMR (%i steps)\n", funcname, amr_step);
  }

  // create solution vector
  ierr = DMCreateGlobalVector(dm, &sol);
  CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sol, "solution");
  CHKERRQ(ierr);

  // set time step and number of steps
  {
    PetscReal B[6] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, hx, hy, hz;
    PetscInt  maxlevel, n = 6;
    PetscBool isset;

    ierr = PetscOptionsGetRealArray(PETSC_NULLPTR, PETSC_NULLPTR, "-dm_p4est_brick_bounds", B, &n, &isset);
    CHKERRQ(ierr);
    ierr = DMForestGetMaximumRefinement(dm, &maxlevel);
    CHKERRQ(ierr);
    hx = (B[1] - B[0]) / (PetscPowReal(2.0, maxlevel) * NX);
    hy = (B[3] - B[2]) / (PetscPowReal(2.0, maxlevel) * NY);
    hz = (B[5] - B[4]) / (PetscPowReal(2.0, maxlevel) * NZ);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] Min mesh refinement h: (%g, %g, %g)\n", funcname, hx, hy, hz);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] Wind: (%g, %g, %g)\n", funcname, WINDX, WINDY, WINDZ);
    // set dt according to CFL=1
    ctx.dt = __DBL_MAX__;
    ctx.dt = PetscMin(ctx.dt, (0 < WINDX ? hx / WINDX : __DBL_MAX__));
    ctx.dt = PetscMin(ctx.dt, (0 < WINDY ? hy / WINDY : __DBL_MAX__));
    ctx.dt = PetscMin(ctx.dt, (0 < WINDZ ? hz / WINDZ : __DBL_MAX__));
    ctx.dt *= 0.5; // reduce dt such that CFL<1
    PetscPrintf(PETSC_COMM_WORLD, "[%s] Compute dt=%g\n", funcname, ctx.dt);

    ierr = PetscOptionsGetReal(PETSC_NULLPTR, PETSC_NULLPTR, "-max_time", &max_time, &isset);
    CHKERRQ(ierr);
    n_timesteps = (PetscInt)PetscCeilReal(max_time / ctx.dt);
    PetscPrintf(PETSC_COMM_WORLD, "[%s] Max time=%g, number of time steps=%i\n", funcname, max_time, n_timesteps);
  }

  // run time steps
  PetscPrintf(PETSC_COMM_WORLD, "[%s] Evolve in time\n", funcname);
  ierr = evolve(&dm, n_timesteps, &sol, 0 /*don't use sol as initial condition*/);
  CHKERRQ(ierr);

  // destroy Petsc objects
  ierr = VecDestroy(&sol);
  CHKERRQ(ierr);
  ierr = DMDestroy(&dm);
  CHKERRQ(ierr);
  ierr = PetscFree2(ctx.buffer1, ctx.buffer2);
  CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD, "[%s] End\n", funcname);

  // finalize Petsc
  ierr = PetscFinalize();
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
./ex3 -dm_forest_topology brick \
      -dm_p4est_brick_size 2,1,1  \
      -dm_p4est_brick_bounds -1.0,3.0,-1.0,1.0,-1.0,1.0 \
      -dm_p4est_brick_periodicity 0,0,0 \
      -dm_forest_initial_refinement 2 \
      -dm_forest_maximum_refinement 3

TODO check if all objects are destroyed
*/
