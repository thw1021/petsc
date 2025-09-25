static char help[] = "Tests for point location\n\n";

#include <petscsf.h>
#include <petscdmplex.h>

/* To inspect the location process, use

    -dm_plex_print_locate 5 -info :dm
*/

typedef struct {
  PetscBool centroids;
  PetscBool custom;
  PetscBool nearest;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->centroids = PETSC_TRUE;
  options->custom    = PETSC_FALSE;
  options->nearest   = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Point Location Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-centroids", "Locate cell centroids", "ex17.c", options->centroids, &options->centroids, NULL));
  PetscCall(PetscOptionsBool("-custom", "Locate user-defined points", "ex17.c", options->custom, &options->custom, NULL));
  PetscCall(PetscOptionsBool("-nearest", "Locate points outside the domain using nearest point location mode", "ex17.c", options->nearest, &options->nearest, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCentroidLocation(DM dm, AppCtx *user)
{
  Vec                points;
  PetscSF            cellSF = NULL;
  const PetscSFNode *cells;
  PetscScalar       *a;
  PetscInt           cdim, n;
  PetscInt           cStart, cEnd, c;

  PetscFunctionBeginUser;
  if (!user->centroids) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* Locate all centroids */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, (cEnd - cStart) * cdim, &points));
  PetscCall(VecSetBlockSize(points, cdim));
  PetscCall(VecGetArray(points, &a));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3];
    PetscInt  off = (c - cStart) * cdim, d;

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    for (d = 0; d < cdim; ++d) a[off + d] = centroid[d];
  }
  PetscCall(VecRestoreArray(points, &a));
  PetscCall(DMLocatePoints(dm, points, DM_POINTLOCATION_NONE, &cellSF));
  PetscCall(VecDestroy(&points));
  PetscCall(PetscSFGetGraph(cellSF, NULL, &n, NULL, &cells));
  if (n != (cEnd - cStart)) {
    for (c = 0; c < n; ++c) {
      if (cells[c].index != c + cStart) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Could not locate centroid of cell %" PetscInt_FMT ", error %" PetscInt_FMT "\n", c + cStart, cells[c].index));
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Located %" PetscInt_FMT " points instead of %" PetscInt_FMT, n, cEnd - cStart);
  }
  for (c = cStart; c < cEnd; ++c) PetscCheck(cells[c - cStart].index == c, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not locate centroid of cell %" PetscInt_FMT ", instead found %" PetscInt_FMT, c, cells[c - cStart].index);
  PetscCall(PetscSFDestroy(&cellSF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCustomLocation(DM dm, AppCtx *user)
{
  PetscSF            cellSF = NULL;
  const PetscSFNode *cells;
  const PetscInt    *found;
  Vec                points;
  PetscScalar        coords[2] = {0.5, 0.5};
  PetscInt           cdim, Np = 1, Nfd;
  PetscMPIInt        rank;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  if (!user->custom) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetCoordinateDim(dm, &cdim));

  // Locate serially on each process
  PetscCall(VecCreate(PETSC_COMM_SELF, &points));
  PetscCall(VecSetBlockSize(points, cdim));
  PetscCall(VecSetSizes(points, Np * cdim, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(points));
  for (PetscInt p = 0; p < Np; ++p) {
    const PetscInt idx[2] = {p * cdim, p * cdim + 1};
    PetscCall(VecSetValues(points, cdim, idx, coords, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(points));
  PetscCall(VecAssemblyEnd(points));

  PetscCall(DMLocatePoints(dm, points, DM_POINTLOCATION_NONE, &cellSF));

  PetscCall(PetscSFGetGraph(cellSF, NULL, &Nfd, &found, &cells));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSynchronizedPrintf(comm, "[%d] Found %" PetscInt_FMT " particles\n", rank, Nfd));
  for (PetscInt p = 0; p < Nfd; ++p) {
    const PetscInt     point = found ? found[p] : p;
    const PetscScalar *array;
    PetscScalar       *ccoords = NULL;
    PetscInt           numCoords;
    PetscBool          isDG;

    // Since the v comm is SELF, rank is always 0
    PetscCall(PetscSynchronizedPrintf(comm, "  point %" PetscInt_FMT " cell %" PetscInt_FMT "\n", point, cells[p].index));
    PetscCall(DMPlexGetCellCoordinates(dm, cells[p].index, &isDG, &numCoords, &array, &ccoords));
    for (PetscInt c = 0; c < numCoords / cdim; ++c) {
      PetscCall(PetscSynchronizedPrintf(comm, "  "));
      for (PetscInt d = 0; d < cdim; ++d) PetscCall(PetscSynchronizedPrintf(comm, " %g", (double)PetscRealPart(ccoords[c * cdim + d])));
      PetscCall(PetscSynchronizedPrintf(comm, "\n"));
    }
    PetscCall(DMPlexRestoreCellCoordinates(dm, cells[p].index, &isDG, &numCoords, &array, &ccoords));
  }
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));

  PetscCall(PetscSFDestroy(&cellSF));
  PetscCall(VecDestroy(&points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestNearestLocation2D(DM dm, AppCtx *user)
{
  PetscSF            cellSF = NULL;
  const PetscSFNode *cells;
  const PetscInt    *found;
  Vec                points;
  PetscScalar        coords[10][2] = {
    {-0.1,     -0.2 },
    {-0.1,     0.7  },
    {-0.1,     1.1  },
    {-0.99999, 1.1  },
    {0.3,      1.1  },
    {1.25,     1.1  },
    {1.25,     0.6  },
    {1.1,      0.1  },
    {1.25,     -0.2 },
    {0.6,      -0.05},
  };
  const PetscScalar expectedCoords[10][2] = {
    {0.0, 0.0},
    {0.0, 0.7},
    {0.0, 1.0},
    {0.0, 1.0},
    {0.3, 1.0},
    {1.0, 1.0},
    {1.0, 0.6},
    {1.0, 0.1},
    {1.0, 0.0},
    {0.6, 0.0},
  };
  PetscInt    cdim, Np = 10, Nfd;
  PetscMPIInt rank;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  if (!user->nearest || cdim != 2) PetscFunctionReturn(PETSC_SUCCESS);

  // Locate serially on each process
  PetscCall(VecCreate(PETSC_COMM_SELF, &points));
  PetscCall(VecSetBlockSize(points, cdim));
  PetscCall(VecSetSizes(points, Np * cdim, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(points));
  for (PetscInt p = 0; p < Np; ++p) {
    const PetscInt idx[2] = {p * cdim, p * cdim + 1};
    PetscCall(VecSetValues(points, cdim, idx, coords[p], INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(points));
  PetscCall(VecAssemblyEnd(points));

  PetscCall(DMLocatePoints(dm, points, DM_POINTLOCATION_NEAREST, &cellSF));

  PetscCall(PetscSFGetGraph(cellSF, NULL, &Nfd, &found, &cells));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSynchronizedPrintf(comm, "[%d] Found %" PetscInt_FMT " particles\n", rank, Nfd));
  for (PetscInt p = 0; p < Nfd; ++p) {
    const PetscInt     point = found ? found[p] : p;
    const PetscScalar *array, *pcoords;
    PetscScalar       *ccoords = NULL;
    PetscReal          error   = 0;
    PetscInt           numCoords;
    PetscBool          isDG;

    // Since the v comm is SELF, rank is always 0
    PetscCall(PetscSynchronizedPrintf(comm, "  point %" PetscInt_FMT " cell %" PetscInt_FMT "\n", point, cells[p].index));
    PetscCall(DMPlexGetCellCoordinates(dm, cells[p].index, &isDG, &numCoords, &array, &ccoords));
    for (PetscInt c = 0; c < numCoords / cdim; ++c) {
      PetscCall(PetscSynchronizedPrintf(comm, "  "));
      for (PetscInt d = 0; d < cdim; ++d) PetscCall(PetscSynchronizedPrintf(comm, " %g", (double)PetscRealPart(ccoords[c * cdim + d])));
      PetscCall(PetscSynchronizedPrintf(comm, "\n"));
    }
    PetscCall(DMPlexRestoreCellCoordinates(dm, cells[p].index, &isDG, &numCoords, &array, &ccoords));

    PetscCall(VecGetArrayRead(points, &pcoords));
    PetscCall(PetscSynchronizedPrintf(comm, "  point %" PetscInt_FMT " coordinates\n", point));
    PetscCall(PetscSynchronizedPrintf(comm, "  "));
    for (PetscInt d = 0; d < cdim; ++d) {
      PetscCall(PetscSynchronizedPrintf(comm, " %g", (double)PetscRealPart(pcoords[p * cdim + d])));
      error += PetscSqr(PetscAbsScalar(pcoords[p * cdim + d] - expectedCoords[p][d]));
    }
    if (error > PETSC_SMALL) PetscCall(PetscSynchronizedPrintf(comm, " (error: %g)", (double)error));
    PetscCall(PetscSynchronizedPrintf(comm, "\n"));
  }
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));

  PetscCall(PetscSFDestroy(&cellSF));
  PetscCall(VecDestroy(&points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(TestCentroidLocation(dm, &user));
  PetscCall(TestCustomLocation(dm, &user));
  PetscCall(TestNearestLocation2D(dm, &user));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_dim 1 -dm_plex_box_faces 10
    output_file: output/empty.out

    test:
      suffix: seg

    test:
      suffix: seg_hash
      args: -dm_refine 2 -dm_plex_point_location_algorithm hash

    test:
      suffix: seg_kdtree
      args: -dm_refine 2 -dm_plex_point_location_algorithm kdtree

  testset:
    args: -dm_plex_box_faces 5,5
    output_file: output/empty.out

    test:
      suffix: tri
      requires: triangle

    test:
      suffix: tri_hash
      requires: triangle
      args: -dm_refine 2 -dm_plex_point_location_algorithm hash

    test:
      suffix: tri_kdtree
      requires: triangle
      args: -dm_refine 2 -dm_plex_point_location_algorithm kdtree

    test:
      suffix: tri_order_2
      requires: triangle
      args: -dm_coord_petscspace_degree 2 -dm_plex_point_location_algorithm {{brute_force hash kdtree}}

    test:
      suffix: tri_order_3
      requires: triangle
      args: -dm_coord_petscspace_degree 3 -dm_plex_point_location_algorithm {{brute_force hash kdtree}}

    test:
      suffix: quad
      args: -dm_plex_simplex 0

    test:
      suffix: quad_order_2
      args: -dm_plex_simplex 0 -dm_coord_petscspace_degree 2 -dm_plex_point_location_algorithm {{brute_force hash kdtree}}

    test:
      suffix: quad_hash
      args: -dm_plex_simplex 0 -dm_refine 2 -dm_plex_point_location_algorithm hash

    test:
      suffix: quad_kdtree
      args: -dm_plex_simplex 0 -dm_refine 2 -dm_plex_point_location_algorithm kdtree

  testset:
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3
    output_file: output/empty.out

    test:
      suffix: tet
      requires: ctetgen

    test:
      suffix: tet_hash
      requires: ctetgen
      args: -dm_refine 1 -dm_plex_point_location_algorithm hash

    test:
      suffix: tet_kdtree
      requires: ctetgen
      args: -dm_refine 1 -dm_plex_point_location_algorithm kdtree

    test:
      suffix: tet_order_2
      requires: ctetgen
      args: -dm_refine 0 -dm_coord_petscspace_degree 2

    test:
      suffix: tet_order_2_kdtree
      requires: ctetgen
      args: -dm_refine 1 -dm_coord_petscspace_degree 2 -dm_plex_point_location_algorithm kdtree

    test:
      suffix: tet_order_2_hash
      requires: ctetgen
      args: -dm_refine 1 -dm_coord_petscspace_degree 2 -dm_plex_point_location_algorithm hash

    test:
      suffix: hex
      args: -dm_plex_simplex 0

    test:
      suffix: hex_hash
      args: -dm_plex_simplex 0 -dm_refine 1 -dm_plex_point_location_algorithm hash

    test:
      suffix: hex_kdtree
      args: -dm_plex_simplex 0 -dm_refine 1 -dm_plex_point_location_algorithm kdtree

    test:
      suffix: hex_order_2
      args: -dm_plex_simplex 0 -dm_refine 1 -dm_coord_petscspace_degree 2
      nsize: 2

    test:
        suffix: hex_order_2_hash
        args: -dm_plex_simplex 0 -dm_refine 2 -dm_coord_petscspace_degree 2 -dm_plex_point_location_algorithm hash
        nsize: 2

    test:
      suffix: hex_order_2_kdtree
      args: -dm_plex_simplex 0 -dm_refine 2 -dm_coord_petscspace_degree 2 -dm_plex_point_location_algorithm kdtree
      nsize: 2

    test:
      suffix: hex_order_3
      args: -dm_plex_simplex 0 -dm_coord_petscspace_degree 3

    test:
      suffix: hex_order_3_hash
      args: -dm_plex_simplex 0 -dm_refine 1 -dm_coord_petscspace_degree 3 -dm_plex_point_location_algorithm hash

    test:
      suffix: hex_order_3_kdtree
      args: -dm_plex_simplex 0 -dm_refine 1 -dm_coord_petscspace_degree 3 -dm_plex_point_location_algorithm kdtree

  testset:
    args: -centroids 0 -custom \
          -dm_plex_simplex 0 -dm_plex_box_faces 21,21 -dm_distribute_overlap 4 -petscpartitioner_type simple
    nsize: 2

    test:
      suffix: quad_overlap
      args: -dm_plex_point_location_algorithm {{brute_force hash kdtree}}

  testset:
    args: -centroids 0 -nearest -dm_plex_simplex 1 -dm_plex_box_faces 11,11
    requires: triangle

    test:
      suffix: tri_nearest_kdtree
      args: -dm_plex_point_location_algorithm kdtree

    test:
      TODO: Broken, hash cannot locate points outside of the axis-aligned bounding box of domain
      suffix: tri_nearest_hash
      args: -dm_plex_point_location_algorithm hash

  # Test location on a Monge Manifold
  testset:
    args: -dm_refine 3 -dm_coord_space 0 \
            -dm_plex_option_phases proj_ -cdm_proj_dm_plex_coordinate_dim 3 -proj_dm_coord_space \
            -proj_dm_coord_remap -proj_dm_coord_map sinusoid -proj_dm_coord_map_params 0.1,1.,1.
    output_file: output/empty.out

    test:
      requires: triangle
      suffix: tri_monge

    test:
      suffix: quad_monge
      args: -dm_plex_simplex 0

TEST*/
