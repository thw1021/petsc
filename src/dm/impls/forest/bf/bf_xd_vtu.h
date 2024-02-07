#include <petsc/private/dmbfimpl.h>
#include "bf_xd.h"
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>

#if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL___FP16)
/* output in float if single or half precision in memory */
static const char precision[] = "Float32";
typedef float     PetscVTUReal;
  #define MPIU_VTUREAL MPI_FLOAT
#elif defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL___FLOAT128)
/* output in double if double or quad precision in memory */
static const char precision[] = "Float64";
typedef double    PetscVTUReal;
  #define MPIU_VTUREAL MPI_DOUBLE
#else
static const char precision[] = "UnknownPrecision";
typedef PetscReal PetscVTUReal;
  #define MPIU_VTUREAL MPIU_REAL
#endif

static PetscErrorCode DMBFGetVTKVertexCoordinates(DM dm, PetscVTUReal *point_data, PetscInt nPoints)
{
  p4est_t *p4est;

  PetscErrorCode ierr;

  PetscVTUReal hx, hy, eta_x, eta_y, eta_z = 0.0;

  PetscVTUReal xyz[3]; /* 3 not P4EST_DIM */

  p4est_locidx_t        xi, yi, i, j, k, l, m;
  sc_array_t           *quadrants; /* use p4est data types here */
  sc_array_t           *trees;
  p4est_tree_t         *tree;
  p4est_quadrant_t     *quad;
  p4est_topidx_t        first_local_tree, last_local_tree, jt, vt[P4EST_CHILDREN];
  p4est_locidx_t        quad_count;
  size_t                num_quads, zz;
  p4est_qcoord_t        x, y;
  const p4est_topidx_t *tree_to_vertex;
  const PetscVTUReal   *v;
  const PetscVTUReal    intsize = 1.0 / P4EST_ROOT_LEN;
  PetscVTUReal          scale   = .999999;
  PetscInt              bs0, bs1, blockSize[3] = {1, 1, 1};

#ifdef P4_TO_P8
  p4est_qcoord_t z;
  p4est_locidx_t zi;
  PetscVTUReal   hz;
  PetscInt       bs2;
#endif

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);

  bs0 = blockSize[0];
  bs1 = blockSize[1];
#ifdef P4_TO_P8
  bs2 = blockSize[2];
#endif

  first_local_tree = p4est->first_local_tree;
  last_local_tree  = p4est->last_local_tree;
  trees            = p4est->trees;
  v                = p4est->connectivity->vertices;
  tree_to_vertex   = p4est->connectivity->tree_to_vertex;

  for (jt = first_local_tree, quad_count = 0; jt <= last_local_tree; ++jt) {
    tree      = p4est_tree_array_index(trees, jt);
    quadrants = &(tree->quadrants);
    num_quads = quadrants->elem_count;

    /* retrieve corners of the tree */
    for (k = 0; k < P4EST_CHILDREN; ++k) { vt[k] = tree_to_vertex[jt * P4EST_CHILDREN + k]; }

    /* loop over the elements in tree and calculate vertex coordinates */
    for (zz = 0; zz < num_quads; ++zz) {
      quad = p4est_quadrant_array_index(quadrants, zz);
      hx   = .5 * P4EST_QUADRANT_LEN(quad->level) / bs0;
      hy   = .5 * P4EST_QUADRANT_LEN(quad->level) / bs1;
#ifdef P4_TO_P8
      hz = .5 * P4EST_QUADRANT_LEN(quad->level) / bs2;
      for (k = 0; k < bs2; k++) {
        z = quad->z + 2. * k * hz;
#endif
        for (j = 0; j < bs1; j++) {
          y = quad->y + 2. * j * hy;
          for (i = 0; i < bs0; i++, quad_count++) {
            x = quad->x + 2. * i * hx;
            l = 0;
#ifdef P4_TO_P8
            for (zi = 0; zi < 2; ++zi) {
              eta_z = intsize * z + intsize * hz * (1. + (zi * 2 - 1) * scale);
#endif
              for (yi = 0; yi < 2; ++yi) {
                eta_y = intsize * y + intsize * hy * (1. + (yi * 2 - 1) * scale);
                for (xi = 0; xi < 2; ++xi) {
#if defined(PETSC_USE_DEBUG)
                  PetscCheck(0 <= l && l < P4EST_CHILDREN, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Index out of bounds: %i, bounds=[0,%i)", (int)l, P4EST_CHILDREN);
#endif
                  eta_x = intsize * x + intsize * hx * (1. + (xi * 2 - 1) * scale);
                  for (m = 0; m < 3 /* 3 not P4EST_DIM */; ++m) {
                    xyz[m] = (1. - eta_z) * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[0] + m] + eta_x * v[3 * vt[1] + m]) + eta_y * ((1. - eta_x) * v[3 * vt[2] + m] + eta_x * v[3 * vt[3] + m]))
#ifdef P4_TO_P8
                           + eta_z * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[4] + m] + eta_x * v[3 * vt[5] + m]) + eta_y * ((1. - eta_x) * v[3 * vt[6] + m] + eta_x * v[3 * vt[7] + m]))
#endif
                      ;
                    point_data[3 * (P4EST_CHILDREN * quad_count + l) + m] = (PetscVTUReal)xyz[m];
                  }
                  l++;
                } /* end for `xi` */
              }   /* end for `yi` */
#ifdef P4_TO_P8
            } /* end for `zi` */
#endif
          } /* end for `i` */
        }   /* end for `j` */
#ifdef P4_TO_P8
      } /* end for `k` */
#endif
    }
  }
  PetscCheck((P4EST_CHILDREN * quad_count) == nPoints, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Counts mismatch: %i != %i (nPoints)", (int)(P4EST_CHILDREN * quad_count), (int)nPoints);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFGetVTKConnectivity(DM dm, PetscVTKInt *conn_data, PetscInt nPoints)
{
  PetscInt il;

  PetscFunctionBegin;
  for (il = 0; il < nPoints; il++) { conn_data[il] = (PetscVTKInt)il; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFGetVTKCellOffsets(DM dm, PetscVTKInt *offset_data, PetscInt nCells)
{
  PetscInt il;

  PetscFunctionBegin;

  for (il = 1; il <= nCells; ++il) { offset_data[il - 1] = (PetscVTKInt)P4EST_CHILDREN * il; /* offsets */ }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFGetVTKCellTypes(DM dm, PetscVTKType *type_data, PetscInt nCells)
{
  PetscInt il;

  PetscFunctionBegin;

  for (il = 0; il < nCells; ++il) {
#ifdef P4_TO_P8
    type_data[il] = 11; /* VTK_VOXEL */
#else
    type_data[il] = 8; /* VTK_PIXEL */
#endif
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFGetVTKTreeIDs(DM dm, PetscVTKInt *treeids, PetscInt nCells)
{
  PetscErrorCode ierr;
  PetscInt       il, num_quads, zz;
  p4est_t       *p4est;
  p4est_topidx_t jt, first_local_tree, last_local_tree;
  p4est_tree_t  *tree;
  sc_array_t    *trees;
  PetscInt       bs, bs0, bs1, bs2, blockSize[3] = {1, 1, 1};

  PetscFunctionBegin;

  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  bs0 = blockSize[0];
  bs1 = blockSize[1];
  bs2 = blockSize[2];
  bs  = bs0 * bs1 * bs2;

  first_local_tree = p4est->first_local_tree;
  last_local_tree  = p4est->last_local_tree;
  trees            = p4est->trees;

  first_local_tree = p4est->first_local_tree;
  last_local_tree  = p4est->last_local_tree;

  for (il = 0, jt = first_local_tree; jt <= last_local_tree; ++jt) {
    tree      = p4est_tree_array_index(trees, jt);
    num_quads = (PetscInt)tree->quadrants.elem_count;
    for (zz = 0; zz < num_quads * bs; ++zz, ++il) { treeids[il] = (PetscVTKInt)jt; }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFGetVTKMPIRank(DM dm, PetscVTKInt *mpirank, PetscInt nCells)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       il;

  PetscFunctionBegin;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);
  for (il = 0; il < nCells; il++) { mpirank[il] = (PetscVTKInt)rank; }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMBFGetVTKQuadRefinementLevel(DM dm, PetscVTKInt *quadlevel, PetscInt nCells)
{
  PetscErrorCode ierr;
  PetscInt       i, k, Q, q;

  p4est_topidx_t    tt, first_local_tree, last_local_tree;
  sc_array_t       *trees, *tquadrants;
  p4est_tree_t     *tree;
  p4est_quadrant_t *quad;
  p4est_t          *p4est;
  PetscInt          bs, bs0, bs1, bs2, blockSize[3] = {1, 1, 1};

  PetscFunctionBegin;

  ierr = DMBFGetP4est(dm, &p4est);
  CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  bs0 = blockSize[0];
  bs1 = blockSize[1];
  bs2 = blockSize[2];
  bs  = bs0 * bs1 * bs2;

  first_local_tree = p4est->first_local_tree;
  last_local_tree  = p4est->last_local_tree;
  trees            = p4est->trees;

  for (tt = first_local_tree, k = 0; tt <= last_local_tree; ++tt) {
    tree       = p4est_tree_array_index(trees, tt);
    tquadrants = &tree->quadrants;
    Q          = (PetscInt)tquadrants->elem_count;
    for (q = 0; q < Q; ++q) {
      quad = p4est_quadrant_array_index(tquadrants, q);
      for (i = 0; i < bs; i++, k++) { quadlevel[k] = (PetscVTKInt)quad->level; }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Write all fields that have been provided to the viewer
  Multi-block XML format with binary appended data.
*/
static PetscErrorCode DMBFVTKWritePiece_VTU(DM dm, PetscViewer viewer)
{
  PetscViewer_VTK         *vtk = (PetscViewer_VTK *)viewer->data;
  PetscViewerVTKObjectLink link;
  FILE                    *f;
  PetscErrorCode           ierr;
  const char              *byte_order = PetscBinaryBigEndian() ? "BigEndian" : "LittleEndian";
  PetscInt                 locSize, nPoints, nCells;
  PetscInt                 bs, bs0, bs1, bs2, blockSize[3] = {1, 1, 1};
  PetscInt                 offset = 0;
  PetscVTKInt             *int_data;
  PetscVTUReal            *float_data;
  PetscVTKType            *type_data;
  char                     lfname[PETSC_MAX_PATH_LEN];
  char                     noext[PETSC_MAX_PATH_LEN];
  PetscMPIInt              rank;
  int                      n;
  PetscVTKInt              bytes = 0;
  size_t                   write_ret;

  PetscFunctionBegin;

  for (n = 0; n < PETSC_MAX_PATH_LEN; n++) { /* remove filename extension */
    if (vtk->filename[n] == '.') break;
  }

  ierr = PetscStrncpy(noext, vtk->filename, n + 1);
  CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);
  ierr = PetscSNPrintf(lfname, sizeof(lfname), "%s_%04d.vtu", noext, rank);
  CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF, lfname, "wb", &f);
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "<?xml version=\"1.0\"?>\n");
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n", byte_order);
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "  <UnstructuredGrid>\n");
  CHKERRQ(ierr);

  /* Get number of cells and number of points.
   * A cell corner is redundantly included on each of its supporting cells, giving
   * P4EST_CHILDREN*locSize local corners.
   */

  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  ierr = DMBFGetLocalSize(dm, &locSize);
  CHKERRQ(ierr);

  bs0 = blockSize[0];
  bs1 = blockSize[1];
  bs2 = blockSize[2];
  bs  = bs0 * bs1 * bs2;

  nCells  = locSize * bs;
  nPoints = P4EST_CHILDREN * nCells;

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "    <Piece NumberOfPoints=\"%" PetscInt_FMT "\" NumberOfCells=\"%" PetscInt_FMT "\">\n", nPoints, nCells);
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      <Points>\n");
  CHKERRQ(ierr);

  /* For each dimension 1,2,3, one coordinate */

  ierr = PetscFPrintf(PETSC_COMM_SELF, f,
                      "        <DataArray type=\"%s\" Name=\"Position\""
                      " NumberOfComponents=\"3\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n",
                      precision, offset);
  CHKERRQ(ierr);

  offset += 4; /* sizeof(int) in bytes */
  offset += 3 * sizeof(PetscVTUReal) * nPoints;

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      </Points>\n");
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      <Cells>\n");
  CHKERRQ(ierr);

  /* P4EST_CHILDREN indices for each cell. */

  ierr = PetscFPrintf(PETSC_COMM_SELF, f,
                      "        <DataArray type=\"%s\" Name=\"connectivity\""
                      " format=\"%s\" offset=\"%" PetscInt_FMT "\" />\n",
                      "Int32", "appended", offset);
  CHKERRQ(ierr);

  offset += 4;
  offset += sizeof(PetscVTKInt) * nPoints;

  /*
   * Data offsets for the cells.
   */

  fprintf(f,
          "        <DataArray type=\"%s\" Name=\"offsets\""
          " format=\"%s\"  offset=\"%" PetscInt_FMT "\" />\n",
          "Int32", "appended", offset);

  offset += 4;
  offset += sizeof(PetscVTKInt) * nCells;

  /* Cell types. Right now VTK_PIXEL (orthogonal quad, x, y aligned).*/

  ierr = PetscFPrintf(PETSC_COMM_SELF, f,
                      "        <DataArray type=\"UInt8\" Name=\"types\""
                      " format=\"%s\" offset=\"%" PetscInt_FMT "\" />\n",
                      "appended", offset); // might need to change
  CHKERRQ(ierr);

  offset += 4;
  offset += sizeof(PetscVTKType) * nCells;

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      </Cells>\n");
  CHKERRQ(ierr);

  /* Start writing cell data headers */

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      <CellData>\n");
  CHKERRQ(ierr);

  /* Cell MPIrank */

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "        <DataArray type=\"Int32\" Name=\"Rank\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n", offset);
  CHKERRQ(ierr);

  offset += 4;
  offset += sizeof(PetscVTKInt) * nCells;

  /* Cell tree ID */

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "        <DataArray type=\"Int32\" Name=\"TreeID\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n", offset);
  CHKERRQ(ierr);

  offset += 4;
  offset += sizeof(PetscVTKInt) * nCells;

  /* Cell refinement level */

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "        <DataArray type=\"Int32\" Name=\"Level\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n", offset);
  CHKERRQ(ierr);

  offset += 4;
  offset += sizeof(PetscVTKInt) * nCells;

  /* Cell data headers (right now, only cell data is supported) */

  for (link = vtk->link; link; link = link->next) {
    const char *vecname = "";
    Vec         v       = (Vec)link->vec;

    if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
    if (((PetscObject)v)->name || link != vtk->link) { /* If the object is already named, use it. If it is past the first link, name it to disambiguate. */
      ierr = PetscObjectGetName((PetscObject)v, &vecname);
      CHKERRQ(ierr);
    }

    if (link->ft == PETSC_VTK_CELL_FIELD) {
      /* TODO? does not handle complex case: see plexvtu.c */
      ierr = PetscFPrintf(PETSC_COMM_SELF, f, "        <DataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n", precision, vecname, offset);
      CHKERRQ(ierr);

      offset += 4;
      offset += sizeof(PetscVTUReal) * nCells;

    } else if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
      ierr = PetscFPrintf(PETSC_COMM_SELF, f, "        <DataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n", precision, vecname, offset);
      CHKERRQ(ierr);

      offset += 4;
      offset += 3 * sizeof(PetscVTUReal) * nCells;
    }
  }

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      </CellData>\n");
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "    </Piece>\n");
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "  </UnstructuredGrid>\n");
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "  <AppendedData encoding=\"raw\">\n");
  CHKERRQ(ierr);

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "_");
  CHKERRQ(ierr);

  // allocate workspace
  ierr = PetscMalloc1(3 * nPoints * sizeof(PetscVTUReal), &float_data);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(nPoints * sizeof(PetscVTKInt), &int_data);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(nCells * sizeof(PetscVTKType), &type_data);
  CHKERRQ(ierr);

  ierr = DMBFGetVTKVertexCoordinates(dm, float_data, nPoints);
  CHKERRQ(ierr);
  bytes = PetscVTKIntCast(3 * sizeof(PetscVTUReal) * nPoints);
  fwrite(&bytes, sizeof(PetscVTKInt), 1, f);
  fwrite(float_data, sizeof(PetscVTUReal), 3 * nPoints, f);

  ierr = DMBFGetVTKConnectivity(dm, int_data, nPoints);
  CHKERRQ(ierr);
  bytes = PetscVTKIntCast(sizeof(PetscVTKInt) * nPoints);
  fwrite(&bytes, sizeof(PetscVTKInt), 1, f);
  fwrite(int_data, sizeof(PetscVTKInt), nPoints, f);

  ierr = DMBFGetVTKCellOffsets(dm, int_data, nCells);
  CHKERRQ(ierr);
  fwrite(&bytes, sizeof(PetscVTKInt), 1, f);
  fwrite(int_data, sizeof(PetscVTKInt), nCells, f);

  ierr = DMBFGetVTKCellTypes(dm, type_data, nCells);
  CHKERRQ(ierr);
  bytes = PetscVTKIntCast(sizeof(PetscVTKType) * nCells);
  fwrite(&bytes, sizeof(PetscVTKInt), 1, f);
  fwrite(type_data, sizeof(PetscVTKType), nCells, f);

  ierr = DMBFGetVTKMPIRank(dm, int_data, nCells);
  CHKERRQ(ierr);
  bytes = PetscVTKIntCast(sizeof(PetscVTKInt) * nCells);
  fwrite(&bytes, sizeof(PetscVTKInt), 1, f);
  fwrite(int_data, sizeof(PetscVTKInt), nCells, f);

  ierr = DMBFGetVTKTreeIDs(dm, int_data, nCells);
  CHKERRQ(ierr);
  fwrite(&bytes, sizeof(PetscVTKInt), 1, f);
  fwrite(int_data, sizeof(PetscVTKInt), nCells, f);

  ierr = DMBFGetVTKQuadRefinementLevel(dm, int_data, nCells);
  CHKERRQ(ierr);
  fwrite(&bytes, sizeof(PetscVTKInt), 1, f);
  fwrite(int_data, sizeof(PetscVTKInt), nCells, f);

  for (link = vtk->link; link; link = link->next) {
    const char         *vecname = "";
    Vec                 v       = (Vec)link->vec;
    const PetscVTUReal *vec_data;

    if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
    if (((PetscObject)v)->name || link != vtk->link) { /* If the object is already named, use it. If it is past the first link, name it to disambiguate. */
      ierr = PetscObjectGetName((PetscObject)v, &vecname);
      CHKERRQ(ierr);
    }

    if (link->ft == PETSC_VTK_CELL_FIELD) {
      /* TODO: does not handle complex case: see plexvtu.c */
      /* TODO: PetscVTUReal or PetscReal? */
      /* ierr = VecGetArray(v,&sdata);CHKERRQ(ierr);
      for(PetscInt i = 0; i < nCells; i++) {
        float_data[i] = (PetscVTUReal) sdata[i];
      }       ierr = VecRestoreArrayRead(v,&sdata);CHKERRQ(ierr); */

      bytes     = PetscVTKIntCast(sizeof(PetscVTUReal) * nCells);
      write_ret = fwrite(&bytes, sizeof(PetscVTKInt), 1, f);
      PetscCheck(write_ret == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "VTK write failed");

      ierr = VecGetArrayRead(v, &vec_data);
      CHKERRQ(ierr);
      write_ret = fwrite(vec_data, sizeof(PetscVTUReal), nCells, f);
      PetscCheck(write_ret == nCells, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Vec write to VTU failed");

      ierr = VecRestoreArrayRead(v, &vec_data);
      CHKERRQ(ierr);

    } else if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
      bytes     = PetscVTKIntCast(3 * sizeof(PetscVTUReal) * nCells);
      write_ret = fwrite(&bytes, sizeof(PetscVTKInt), 1, f);

      PetscCheck(write_ret == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "VTK write failed");

      ierr = VecGetArrayRead(v, &vec_data);
      CHKERRQ(ierr);
      if (P4EST_DIM == 2) {
        for (PetscInt i = 0; i < nCells; i++) {
          float_data[3 * i + 0] = vec_data[2 * i + 0];
          float_data[3 * i + 1] = vec_data[2 * i + 1];
          float_data[3 * i + 2] = 0.0;
        }
        ierr = VecRestoreArrayRead(v, &vec_data);
        CHKERRQ(ierr);
        write_ret = fwrite(float_data, sizeof(PetscVTUReal), 3 * nCells, f);
      } else {
        write_ret = fwrite(vec_data, sizeof(PetscVTUReal), 3 * nCells, f);
        ierr      = VecRestoreArrayRead(v, &vec_data);
        CHKERRQ(ierr);
      }

      PetscCheck(write_ret == 3 * nCells, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Vec write to VTU failed");
    }
  }

  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "\n  </AppendedData>\n");
  CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF, f, "</VTKFile>\n");
  CHKERRQ(ierr);

  // destroy workspace
  ierr = PetscFree(float_data);
  ierr = PetscFree(int_data);
  ierr = PetscFree(type_data);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !defined(DMBF_XD_VTKWriteAll)
static
#endif
  PetscErrorCode
  DMBF_XD_VTKWriteAll(PetscObject odm, PetscViewer viewer)
{
  DM                       dm  = (DM)odm;
  PetscViewer_VTK         *vtk = (PetscViewer_VTK *)viewer->data;
  PetscViewerVTKObjectLink link;
  FILE                    *f;
  PetscErrorCode           ierr;
  const char              *byte_order = PetscBinaryBigEndian() ? "BigEndian" : "LittleEndian";
  char                     gfname[PETSC_MAX_PATH_LEN];
  char                     noext[PETSC_MAX_PATH_LEN];
  PetscMPIInt              rank, size;
  int                      n;

  PetscFunctionBegin;

  ierr = DMBFVTKWritePiece_VTU(dm, viewer);
  CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
  CHKERRQ(ierr);

  if (!rank) {
    for (n = 0; n < PETSC_MAX_PATH_LEN; n++) { /* remove filename extension */
      if (vtk->filename[n] == '.') break;
    }

    ierr = PetscStrncpy(noext, vtk->filename, n + 1);
    CHKERRQ(ierr);
    ierr = PetscSNPrintf(gfname, sizeof(gfname), "%s.pvtu", noext);
    CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF, gfname, "wb", &f);
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "<?xml version=\"1.0\"?>\n");
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n", byte_order);
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "    <PPoints>\n");
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f,
                        "      <PDataArray type=\"%s\" Name=\"Position\""
                        " NumberOfComponents=\"3\" format=\"appended\"  />\n",
                        precision);
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "    </PPoints>\n");
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "    <PCellData>\n");
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      <PDataArray type=\"Int32\" Name=\"Rank\" NumberOfComponents=\"1\" format=\"appended\" />\n");
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      <PDataArray type=\"Int32\" Name=\"TreeID\" NumberOfComponents=\"1\" format=\"appended\" />\n");
    CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      <PDataArray type=\"Int32\" Name=\"Level\" NumberOfComponents=\"1\" format=\"appended\" />\n");
    CHKERRQ(ierr);

    for (link = vtk->link; link; link = link->next) {
      const char *vecname = "";
      Vec         v       = (Vec)link->vec;

      if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
      if (((PetscObject)v)->name || link != vtk->link) { /* If the object is already named, use it. If it is past the first link, name it to disambiguate. */
        ierr = PetscObjectGetName((PetscObject)v, &vecname);
        CHKERRQ(ierr);
      }

      if (link->ft == PETSC_VTK_CELL_FIELD) {
        /* TODO? does not handle complex case: see plexvtu.c */
        ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      <PDataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"1\" format=\"appended\" />\n", precision, vecname);
        CHKERRQ(ierr);
      } else if (link->ft == PETSC_VTK_CELL_VECTOR_FIELD) {
        ierr = PetscFPrintf(PETSC_COMM_SELF, f, "      <PDataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"3\" format=\"appended\" />\n", precision, vecname);
        CHKERRQ(ierr);
      }
    }

    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "    </PCellData>\n");
    for (PetscVTKInt r = 0; r < size; r++) { ierr = PetscFPrintf(PETSC_COMM_SELF, f, "    <Piece Source=\"%s_%04d.vtu\"/>\n", noext, r); }
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "  </PUnstructuredGrid>\n");
    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "</VTKFile>");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
