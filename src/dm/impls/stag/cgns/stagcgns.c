/*
   Routines to view DMStag in CGNS format.
*/

#include <petsc/private/dmstagimpl.h>
#include <petsc/private/viewercgnsimpl.h>

#include <pcgnslib.h>
#include <cgns_io.h>
#include <petscdmproduct.h>

#if !defined(CGNS_ENUMT)
  #define CGNS_ENUMT(a) a
#endif
#if !defined(CGNS_ENUMV)
  #define CGNS_ENUMV(a) a
#endif

static PetscErrorCode PetscCGNSDataType_Private(PetscDataType pd, CGNS_ENUMT(DataType_t) * cd)
{
  PetscFunctionBegin;
  switch (pd) {
  case PETSC_FLOAT:
    *cd = CGNS_ENUMV(RealSingle);
    break;
  case PETSC_DOUBLE:
    *cd = CGNS_ENUMV(RealDouble);
    break;
  case PETSC_COMPLEX:
    *cd = CGNS_ENUMV(ComplexDouble);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Data type %s", PetscDataTypes[pd]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagGetLocalNodeCoordinate1d_Private(DM dm, PetscInt nStart[], PetscInt nEnd[], PetscScalar *x)
{
  DM            cdm;
  Vec           coord;
  PetscScalar **arr;
  PetscInt      ileft, i, cnt = 0;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coord));
  PetscCall(DMStagVecGetArrayRead(cdm, coord, &arr));
  PetscCall(DMStagGetLocationSlot(cdm, DMSTAG_LEFT, 0, &ileft));
  for (i = nStart[0]; i < nEnd[0]; ++i) x[cnt++] = arr[i][ileft];
  PetscCall(DMStagVecRestoreArrayRead(cdm, coord, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagGetLocalNodeCoordinate2d_Private(DM dm, PetscInt nStart[], PetscInt nEnd[], PetscInt d, PetscScalar *x)
{
  DM             cdm;
  Vec            coord;
  PetscScalar ***arr;
  PetscInt       idownleft, i, j, cnt = 0;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coord));
  PetscCall(DMStagVecGetArrayRead(cdm, coord, &arr));
  PetscCall(DMStagGetLocationSlot(cdm, DMSTAG_DOWN_LEFT, 0, &idownleft));
  for (j = nStart[1]; j < nEnd[1]; ++j)
    for (i = nStart[0]; i < nEnd[0]; ++i) x[cnt++] = arr[j][i][idownleft + d];
  PetscCall(DMStagVecRestoreArrayRead(cdm, coord, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagGetLocalNodeCoordinate3d_Private(DM dm, PetscInt nStart[], PetscInt nEnd[], PetscInt d, PetscScalar *x)
{
  DM              cdm;
  Vec             coord;
  PetscScalar ****arr;
  PetscInt        ibackdownleft, i, j, k, cnt = 0;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coord));
  PetscCall(DMStagVecGetArrayRead(cdm, coord, &arr));
  PetscCall(DMStagGetLocationSlot(cdm, DMSTAG_BACK_DOWN_LEFT, 0, &ibackdownleft));
  for (k = nStart[2]; k < nEnd[2]; ++k)
    for (j = nStart[1]; j < nEnd[1]; ++j)
      for (i = nStart[0]; i < nEnd[0]; ++i) x[cnt++] = arr[k][j][i][ibackdownleft + d];
  PetscCall(DMStagVecRestoreArrayRead(cdm, coord, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMView_Stag_CGNS(DM dm, PetscViewer viewer)
{
  DM_Stag *const    stag = (DM_Stag *)dm->data;
  PetscViewer_CGNS *cgv  = (PetscViewer_CGNS *)viewer->data;
  PetscInt          topo_dim, coord_dim;
  PetscInt          num_local_nodes, num_local_elems, num_global_elems;
  PetscInt          nStart[3], nEnd[3], eStart[3], eEnd[3];
  const char       *dm_name;
  int               base, zone;
  DM                cdm;
  PetscBool         is_last_rank[3];
  PetscBool         isstag, isproduct;
  cgsize_t          isize[9];
  CGNS_ENUMT(DataType_t) datatype;
  int          coord_ids[3];
  cgsize_t     start[3], end[3];
  PetscScalar *x;
  PetscInt     d;

  PetscFunctionBegin;
  // When `-dm_view` option exists, the DM is viewed in DMSetUp() but the coordinate is not set up at that point.
  // In order not to struggle with such error, simply return success here and view nothing if the DM is not set up.
  if (!dm->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (cgv->base) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheck(stag->coordinateDMType, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Must call DMStagSetCoordinateDMType() before calling DMView()");

  if (!cgv->file_num) {
    PetscInt time_step;
    PetscCall(DMGetOutputSequenceNumber(dm, &time_step, NULL));
    PetscCall(PetscViewerCGNSFileOpen_Internal(viewer, time_step));
  }
  PetscCall(DMGetDimension(dm, &topo_dim));
  PetscCall(DMGetCoordinateDim(dm, &coord_dim));
  PetscCall(PetscObjectGetName((PetscObject)dm, &dm_name));
  PetscCallCGNS(cg_base_write(cgv->file_num, dm_name, topo_dim, coord_dim, &base));
  PetscCallCGNS(cg_goto(cgv->file_num, base, NULL));
  PetscCallCGNS(cg_dataclass_write(CGNS_ENUMV(NormalizedByDimensional)));

  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(PetscStrcmp(stag->coordinateDMType, DMSTAG, &isstag));
  PetscCall(PetscStrcmp(stag->coordinateDMType, DMPRODUCT, &isproduct));
  if (isstag) {
    PetscInt N[3], width[3];
    PetscCall(DMStagGetGlobalSizes(cdm, &N[0], &N[1], &N[2]));
    num_global_elems = 1;
    for (d = 0; d < coord_dim; ++d) {
      isize[d]             = N[d] + 1; // number of vertices
      isize[d + coord_dim] = N[d];     // number of elements
      num_global_elems *= N[d];
    }
    PetscCall(DMStagGetCorners(cdm, &nStart[0], &nStart[1], &nStart[2], &width[0], &width[1], &width[2], NULL, NULL, NULL));
    PetscCall(DMStagGetIsLastRank(cdm, &is_last_rank[0], &is_last_rank[1], &is_last_rank[2]));
    num_local_nodes = 1;
    num_local_elems = 1;
    for (d = 0; d < coord_dim; ++d) {
      nEnd[d]   = nStart[d] + width[d] + (PetscInt)is_last_rank[d];
      eStart[d] = nStart[d];
      eEnd[d]   = eStart[d] + width[d];
      num_local_nodes *= nEnd[d] - nStart[d];
      num_local_elems *= eEnd[d] - eStart[d];
    }
  } else if (isproduct) {
    DM       subdm;
    PetscInt N, width;
    num_local_elems  = 1;
    num_global_elems = 1;
    num_local_nodes  = 1;
    for (d = 0; d < coord_dim; ++d) {
      PetscCall(DMProductGetDM(cdm, d, &subdm));
      PetscCheck(subdm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Coordinate DM is missing sub DM %" PetscInt_FMT, d);
      PetscCall(DMStagGetGlobalSizes(subdm, &N, NULL, NULL));
      isize[d]             = N + 1; // number of vertices
      isize[d + coord_dim] = N;     // number of elements
      num_global_elems *= N;
      PetscCall(DMStagGetCorners(subdm, &nStart[d], NULL, NULL, &width, NULL, NULL, NULL, NULL, NULL));
      PetscCall(DMStagGetIsLastRank(subdm, &is_last_rank[d], NULL, NULL));
      nEnd[d]   = nStart[d] + width + (PetscInt)is_last_rank[d];
      eStart[d] = nStart[d];
      eEnd[d]   = eStart[d] + width;
      num_local_nodes *= nEnd[d] - nStart[d];
      num_local_elems *= eEnd[d] - eStart[d];
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported coordinate DM type %s", stag->coordinateDMType);
  PetscCallCGNS(cg_zone_write(cgv->file_num, base, "Zone", isize, CGNS_ENUMV(Structured), &zone));

  PetscCall(PetscCGNSDataType_Private(PETSC_SCALAR, &datatype));
  for (d = 0; d < coord_dim; ++d) {
    const double exponents[] = {0, 1, 0, 0, 0};
    char         coord_name[64];
    PetscCall(PetscSNPrintf(coord_name, sizeof coord_name, "Coordinate%c", 'X' + (int)d));
    PetscCallCGNS(cgp_coord_write(cgv->file_num, base, zone, datatype, coord_name, &coord_ids[d]));
    PetscCallCGNS(cg_goto(cgv->file_num, base, "Zone_t", zone, "GridCoordinates", 0, coord_name, 0, NULL));
    PetscCallCGNS(cg_exponents_write(CGNS_ENUMV(RealDouble), exponents));
  }

  // CGNS nodes use 1-based indexing
  for (d = 0; d < coord_dim; ++d) {
    start[d] = nStart[d] + 1;
    end[d]   = nEnd[d];
  }

  PetscCall(PetscMalloc1(num_local_nodes, &x));
  if (isstag) {
    PetscInt node_dof;
    PetscCall(DMStagGetDOF(cdm, &node_dof, NULL, NULL, NULL));
    PetscCheck(node_dof == coord_dim, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Node dof of %" PetscInt_FMT "D coordinate DM is %" PetscInt_FMT, coord_dim, node_dof);
    for (d = 0; d < coord_dim; ++d) {
      switch (coord_dim) {
      case 1:
        PetscCall(DMStagGetLocalNodeCoordinate1d_Private(dm, nStart, nEnd, x));
        break;
      case 2:
        PetscCall(DMStagGetLocalNodeCoordinate2d_Private(dm, nStart, nEnd, d, x));
        break;
      case 3:
        PetscCall(DMStagGetLocalNodeCoordinate3d_Private(dm, nStart, nEnd, d, x));
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, coord_dim);
      }
      PetscCallCGNS(cgp_coord_write_data(cgv->file_num, base, zone, coord_ids[d], start, end, x));
    }
  } else if (isproduct) {
    const PetscScalar **arr[3];
    PetscInt            ileft, cnt, ind[3];
    PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr[0], &arr[1], &arr[2]));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &ileft));
    for (d = 0; d < coord_dim; ++d) {
      cnt = 0;
      switch (coord_dim) {
      case 1:
        for (ind[0] = nStart[0]; ind[0] < nEnd[0]; ++ind[0]) x[cnt++] = arr[d][ind[d]][ileft];
        break;
      case 2:
        for (ind[1] = nStart[1]; ind[1] < nEnd[1]; ++ind[1])
          for (ind[0] = nStart[0]; ind[0] < nEnd[0]; ++ind[0]) x[cnt++] = arr[d][ind[d]][ileft];
        break;
      case 3:
        for (ind[2] = nStart[2]; ind[2] < nEnd[2]; ++ind[2])
          for (ind[1] = nStart[1]; ind[1] < nEnd[1]; ++ind[1])
            for (ind[0] = nStart[0]; ind[0] < nEnd[0]; ++ind[0]) x[cnt++] = arr[d][ind[d]][ileft];
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, coord_dim);
      }
      PetscCallCGNS(cgp_coord_write_data(cgv->file_num, base, zone, coord_ids[d], start, end, x));
    }
    PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr[0], &arr[1], &arr[2]));
  }
  PetscCall(PetscFree(x));

  cgv->base            = base;
  cgv->zone            = zone;
  cgv->num_local_nodes = num_local_nodes;
  for (d = 0; d < coord_dim; ++d) {
    cgv->nStart[d] = nStart[d];
    cgv->nEnd[d]   = nEnd[d];
    cgv->eStart[d] = eStart[d];
    cgv->eEnd[d]   = eEnd[d];
  }

  {
    int         sol, field;
    PetscMPIInt rank;
    int        *x;
    PetscInt    i;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
    PetscCallCGNS(cg_sol_write(cgv->file_num, base, zone, "CellInfo", CGNS_ENUMV(CellCenter), &sol));

    PetscCall(PetscMalloc1(num_local_elems, &x));
    // CGNS nodes use 1-based indexing
    for (d = 0; d < coord_dim; ++d) {
      start[d] = eStart[d] + 1;
      end[d]   = eEnd[d];
    }

    for (i = 0; i < num_local_elems; ++i) x[i] = rank;
    PetscCallCGNS(cgp_field_write(cgv->file_num, base, zone, sol, CGNS_ENUMV(Integer), "Rank", &field));
    PetscCallCGNS(cgp_field_write_data(cgv->file_num, base, zone, sol, field, start, end, x));

    for (d = 0; d < coord_dim; ++d) {
      char field_name[64];
      for (i = 0; i < num_local_elems; ++i) x[i] = stag->rank[d];
      PetscCall(PetscSNPrintf(field_name, sizeof field_name, "Rank%c", 'I' + (int)d));
      PetscCallCGNS(cgp_field_write(cgv->file_num, base, zone, sol, CGNS_ENUMV(Integer), field_name, &field));
      PetscCallCGNS(cgp_field_write_data(cgv->file_num, base, zone, sol, field, start, end, x));
    }

    PetscCall(PetscFree(x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
