static char help[] = "Test space-filling-curve reorder of a distributed cell list\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

// Verify that migrationSF maps the input cells onto the output cells one-to-one. Each input cell
// carries a globally unique tag. After migration every tag must appear exactly once.
static PetscErrorCode CheckMigrationIsPermutation(MPI_Comm comm, PetscSF migrationSF, PetscInt numCells, PetscInt newNumCells, PetscInt NCells)
{
  PetscInt *tags, *newtags, *hist;
  PetscInt  off = 0;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Exscan(&numCells, &off, 1, MPIU_INT, MPI_SUM, comm));
  PetscCall(PetscMalloc2(numCells, &tags, newNumCells, &newtags));
  PetscCall(PetscCalloc1(NCells, &hist));
  for (PetscInt c = 0; c < numCells; ++c) tags[c] = off + c;
  PetscCall(PetscSFBcastBegin(migrationSF, MPIU_INT, tags, newtags, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(migrationSF, MPIU_INT, tags, newtags, MPI_REPLACE));
  for (PetscInt c = 0; c < newNumCells; ++c) {
    PetscCheck(newtags[c] >= 0 && newtags[c] < NCells, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Migrated tag %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", newtags[c], NCells);
    ++hist[newtags[c]];
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, hist, NCells, MPIU_INT, MPI_SUM, comm));
  for (PetscInt c = 0; c < NCells; ++c) PetscCheck(hist[c] == 1, comm, PETSC_ERR_PLIB, "Cell tag %" PetscInt_FMT " appears %" PetscInt_FMT " times, not once", c, hist[c]);
  PetscCall(PetscFree(hist));
  PetscCall(PetscFree2(tags, newtags));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Independent Morton encoder, so the test does not reuse the implementation it checks. It must
// mirror the contract of DMPLEXCURVEMORTON: 21 bits per axis over the global bounding box, with
// axis 0 in the highest of each interleaved triple.
static PetscInt64 TestZEncode1(PetscInt t)
{
  PetscInt64 z = (PetscInt64)t & 0x1fffff;

  z = (z | (z << 32)) & 0x1f00000000ffffLL;
  z = (z | (z << 16)) & 0x1f0000ff0000ffLL;
  z = (z | (z << 8)) & 0x100f00f00f00f00fLL;
  z = (z | (z << 4)) & 0x10c30c30c30c30c3LL;
  z = (z | (z << 2)) & 0x1249249249249249LL;
  return z;
}

// Verify that the reordered cells form one ascending run of the curve: non-decreasing within each
// rank, and every rank's codes at or below the next non-empty rank's codes.
static PetscErrorCode CheckGloballyCurveSorted(MPI_Comm comm, PetscInt spaceDim, PetscInt n, const PetscReal centroids[])
{
  PetscReal      lo[3], hi[3], span[3];
  PetscInt64    *codes, prevmax;
  PetscInt64    *bounds;
  PetscMPIInt    size, rank;
  const PetscInt maxidx = (1 << 21) - 1;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  for (PetscInt d = 0; d < 3; ++d) {
    lo[d] = PETSC_MAX_REAL;
    hi[d] = PETSC_MIN_REAL;
  }
  for (PetscInt c = 0; c < n; ++c) {
    for (PetscInt d = 0; d < spaceDim; ++d) {
      lo[d] = PetscMin(lo[d], centroids[c * spaceDim + d]);
      hi[d] = PetscMax(hi[d], centroids[c * spaceDim + d]);
    }
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, lo, 3, MPIU_REAL, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, hi, 3, MPIU_REAL, MPI_MAX, comm));
  for (PetscInt d = 0; d < 3; ++d) span[d] = hi[d] > lo[d] ? hi[d] - lo[d] : 1.;

  PetscCall(PetscMalloc1(PetscMax(1, n), &codes));
  for (PetscInt c = 0; c < n; ++c) {
    PetscInt q[3] = {0, 0, 0};

    for (PetscInt d = 0; d < spaceDim; ++d) {
      const PetscReal t = (centroids[c * spaceDim + d] - lo[d]) / span[d];

      q[d] = PetscMax(0, PetscMin(maxidx, (PetscInt)(t * (PetscReal)maxidx)));
    }
    codes[c] = (TestZEncode1(q[0]) << 2) | (TestZEncode1(q[1]) << 1) | TestZEncode1(q[2]);
  }
  for (PetscInt c = 1; c < n; ++c) PetscCheck(codes[c - 1] <= codes[c], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local curve codes not ascending at %" PetscInt_FMT, c);

  // Exchange each rank's code range. Empty ranks report a range that constrains nothing.
  PetscCall(PetscMalloc1(2 * size, &bounds));
  {
    PetscInt64 mine[2];

    mine[0] = n ? codes[0] : PETSC_INT64_MAX;
    mine[1] = n ? codes[n - 1] : PETSC_INT64_MIN;
    PetscCallMPI(MPI_Allgather(mine, 2, MPIU_INT64, bounds, 2, MPIU_INT64, comm));
  }
  prevmax = PETSC_INT64_MIN;
  for (PetscMPIInt r = 0; r < size; ++r) {
    if (bounds[2 * r] == PETSC_INT64_MAX) continue; // rank r owns no cells
    PetscCheck(prevmax <= bounds[2 * r], comm, PETSC_ERR_PLIB, "Curve order broken across ranks before rank %d", r);
    prevmax = bounds[2 * r + 1];
  }
  PetscCall(PetscFree(bounds));
  PetscCall(PetscFree(codes));
  (void)rank;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part A: reorder a lattice of synthetic 3-D centroids. The initial distribution is strided, so
// every rank starts with cells spread over the whole domain.
static PetscErrorCode TestFromCentroids(MPI_Comm comm, PetscInt N)
{
  PetscSF     sf;
  PetscReal  *centroids, *newcentroids;
  PetscInt   *idx, *newidx;
  PetscInt    numCells = 0, newNumCells, NCells = N * N * N, gnew;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  for (PetscInt g = 0; g < NCells; ++g)
    if (g % size == rank) ++numCells;
  PetscCall(PetscMalloc2(numCells * 3, &centroids, numCells, &idx));
  {
    PetscInt c = 0;

    for (PetscInt g = 0; g < NCells; ++g) {
      if (g % size != rank) continue;
      centroids[c * 3 + 0] = (PetscReal)(g % N) + 0.5;
      centroids[c * 3 + 1] = (PetscReal)((g / N) % N) + 0.5;
      centroids[c * 3 + 2] = (PetscReal)(g / (N * N)) + 0.5;
      idx[c]               = g;
      ++c;
    }
  }
  PetscCall(DMPlexReorderCellListByCurveFromCentroids(comm, DMPLEXCURVEMORTON, 3, numCells, centroids, &sf, &newNumCells));
  gnew = newNumCells;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gnew, 1, MPIU_INT, MPI_SUM, comm));
  PetscCheck(gnew == NCells, comm, PETSC_ERR_PLIB, "Global cell count changed from %" PetscInt_FMT " to %" PetscInt_FMT, NCells, gnew);
  PetscCall(CheckMigrationIsPermutation(comm, sf, numCells, newNumCells, NCells));

  // Migrate the centroids themselves so the locality of the new distribution can be measured.
  PetscCall(PetscMalloc2(newNumCells * 3, &newcentroids, newNumCells, &newidx));
  {
    MPI_Datatype ctype;

    PetscCallMPI(MPI_Type_contiguous(3, MPIU_REAL, &ctype));
    PetscCallMPI(MPI_Type_commit(&ctype));
    PetscCall(PetscSFBcastBegin(sf, ctype, centroids, newcentroids, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, ctype, centroids, newcentroids, MPI_REPLACE));
    PetscCallMPI(MPI_Type_free(&ctype));
  }
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, idx, newidx, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, idx, newidx, MPI_REPLACE));
  PetscCall(CheckGloballyCurveSorted(comm, 3, newNumCells, newcentroids));
  PetscCall(PetscPrintf(comm, "FromCentroids: N=%" PetscInt_FMT " cells=%" PetscInt_FMT " permutation ok, globally curve sorted\n", N, NCells));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree2(newcentroids, newidx));
  PetscCall(PetscFree2(centroids, idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part B: reorder the connectivity of an N x N quadrilateral grid, then build a DMPlex from the
// reordered list. The vertex distribution stays in natural order, as the reorder requires.
static PetscErrorCode TestCellList(MPI_Comm comm, PetscInt N)
{
  DM          dm;
  PetscSF     sf;
  PetscInt   *cells, *newcells, *cellsSaved = NULL;
  PetscReal  *coords;
  PetscLayout vlayout;
  PetscInt    numCells = 0, newNumCells, NCells = N * N, NVertices = (N + 1) * (N + 1);
  PetscInt    numVertices, vStart, vEnd, cStart, cEnd, gcells;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  for (PetscInt g = 0; g < NCells; ++g)
    if (g % size == rank) ++numCells;
  PetscCall(PetscMalloc1(numCells * 4, &cells));
  {
    PetscInt c = 0;

    for (PetscInt g = 0; g < NCells; ++g) {
      const PetscInt i = g % N, j = g / N;

      if (g % size != rank) continue;
      cells[c * 4 + 0] = j * (N + 1) + i;
      cells[c * 4 + 1] = j * (N + 1) + i + 1;
      cells[c * 4 + 2] = (j + 1) * (N + 1) + i + 1;
      cells[c * 4 + 3] = (j + 1) * (N + 1) + i;
      ++c;
    }
  }
  // Own a contiguous slice of the vertices, matching what DMPlexCreateFromCellListParallel expects.
  PetscCall(PetscLayoutCreate(comm, &vlayout));
  PetscCall(PetscLayoutSetSize(vlayout, NVertices));
  PetscCall(PetscLayoutSetBlockSize(vlayout, 1));
  PetscCall(PetscLayoutSetUp(vlayout));
  PetscCall(PetscLayoutGetRange(vlayout, &vStart, &vEnd));
  PetscCall(PetscLayoutDestroy(&vlayout));
  numVertices = vEnd - vStart;
  PetscCall(PetscMalloc1(numVertices * 2, &coords));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    coords[(v - vStart) * 2 + 0] = (PetscReal)(v % (N + 1));
    coords[(v - vStart) * 2 + 1] = (PetscReal)(v / (N + 1));
  }

  PetscCall(DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, numCells, 4, cells, 2, numVertices, NVertices, coords, &sf, &newNumCells, &newcells));
  PetscCall(CheckMigrationIsPermutation(comm, sf, numCells, newNumCells, NCells));
  for (PetscInt i = 0; i < newNumCells * 4; ++i) PetscCheck(newcells[i] >= 0 && newcells[i] < NVertices, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reordered connectivity entry %" PetscInt_FMT " out of range", newcells[i]);

  // The reordered list must build a valid, interpolated, distributed plex.
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, 2, newNumCells, numVertices, NVertices, 4, PETSC_TRUE, newcells, 2, coords, NULL, &cellsSaved, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  gcells = cEnd - cStart;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gcells, 1, MPIU_INT, MPI_SUM, comm));
  PetscCheck(gcells == NCells, comm, PETSC_ERR_PLIB, "Built plex has %" PetscInt_FMT " cells, expected %" PetscInt_FMT, gcells, NCells);
  PetscCall(DMPlexCheck(dm));
  PetscCall(PetscPrintf(comm, "CellList: N=%" PetscInt_FMT " cells=%" PetscInt_FMT " plex built and checked\n", N, NCells));
  PetscCall(PetscFree(cellsSaved));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(newcells));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part C: the reorder exists to make parallel interpolation cheap. After interpolation the number
// of shared points measures how much of the mesh sits on rank boundaries. Reordering must not
// increase it.
static PetscErrorCode BuildAndCountSharedPoints(MPI_Comm comm, PetscInt numCells, const PetscInt cells[], PetscInt numVertices, PetscInt NVertices, const PetscReal coords[], PetscInt *nshared)
{
  DM        dm;
  PetscSF   pointSF;
  PetscInt *cellsSaved = NULL;
  PetscInt  nleaves;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, 2, numCells, numVertices, NVertices, 4, PETSC_TRUE, cells, 2, coords, NULL, &cellsSaved, &dm));
  PetscCall(DMGetPointSF(dm, &pointSF));
  PetscCall(PetscSFGetGraph(pointSF, NULL, &nleaves, NULL, NULL));
  *nshared = nleaves;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, nshared, 1, MPIU_INT, MPI_SUM, comm));
  PetscCall(PetscFree(cellsSaved));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestLocalityImproves(MPI_Comm comm, PetscInt N)
{
  PetscSF     sf;
  PetscInt   *cells, *newcells;
  PetscReal  *coords;
  PetscLayout vlayout;
  PetscInt    numCells = 0, newNumCells, NCells = N * N, NVertices = (N + 1) * (N + 1);
  PetscInt    numVertices, vStart, vEnd, sharedBefore, sharedAfter;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  for (PetscInt g = 0; g < NCells; ++g)
    if (g % size == rank) ++numCells;
  PetscCall(PetscMalloc1(numCells * 4, &cells));
  {
    PetscInt c = 0;

    for (PetscInt g = 0; g < NCells; ++g) {
      const PetscInt i = g % N, j = g / N;

      if (g % size != rank) continue;
      cells[c * 4 + 0] = j * (N + 1) + i;
      cells[c * 4 + 1] = j * (N + 1) + i + 1;
      cells[c * 4 + 2] = (j + 1) * (N + 1) + i + 1;
      cells[c * 4 + 3] = (j + 1) * (N + 1) + i;
      ++c;
    }
  }
  PetscCall(PetscLayoutCreate(comm, &vlayout));
  PetscCall(PetscLayoutSetSize(vlayout, NVertices));
  PetscCall(PetscLayoutSetBlockSize(vlayout, 1));
  PetscCall(PetscLayoutSetUp(vlayout));
  PetscCall(PetscLayoutGetRange(vlayout, &vStart, &vEnd));
  PetscCall(PetscLayoutDestroy(&vlayout));
  numVertices = vEnd - vStart;
  PetscCall(PetscMalloc1(numVertices * 2, &coords));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    coords[(v - vStart) * 2 + 0] = (PetscReal)(v % (N + 1));
    coords[(v - vStart) * 2 + 1] = (PetscReal)(v / (N + 1));
  }

  PetscCall(BuildAndCountSharedPoints(comm, numCells, cells, numVertices, NVertices, coords, &sharedBefore));
  PetscCall(DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, numCells, 4, cells, 2, numVertices, NVertices, coords, &sf, &newNumCells, &newcells));
  PetscCall(BuildAndCountSharedPoints(comm, newNumCells, newcells, numVertices, NVertices, coords, &sharedAfter));
  PetscCheck(sharedAfter <= sharedBefore, comm, PETSC_ERR_PLIB, "Shared points grew from %" PetscInt_FMT " to %" PetscInt_FMT, sharedBefore, sharedAfter);
  // A strided input distribution shares almost every point. The reorder must cut that sharply, not
  // merely avoid making it worse, so require at least a factor of two on more than one process.
  // Require the factor of two only when each process receives a block of several cells. A mesh with
  // very few cells per process has no locality to gain; every cell touches the same few vertices.
  if (size > 1 && NCells >= 4 * size) PetscCheck(2 * sharedAfter <= sharedBefore, comm, PETSC_ERR_PLIB, "Shared points only fell from %" PetscInt_FMT " to %" PetscInt_FMT ", less than a factor of two", sharedBefore, sharedAfter);
  // The counts depend on the number of processes, so keep them out of the reference output.
  PetscCall(PetscPrintf(comm, "Locality: reorder cuts shared points\n"));
  PetscCall(PetscInfo(NULL, "shared points %" PetscInt_FMT " -> %" PetscInt_FMT "\n", sharedBefore, sharedAfter));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(newcells));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm comm;
  PetscInt N = 4;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, "", "SFC cell-list reorder test options", "DMPLEX");
  PetscCall(PetscOptionsInt("-n", "Cells per side", "ex105.c", N, &N, NULL));
  PetscOptionsEnd();
  PetscCheck(N > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-n must be positive");
  PetscCall(TestFromCentroids(comm, N));
  PetscCall(TestCellList(comm, N));
  PetscCall(TestLocalityImproves(comm, N));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2 3 4}}
    args: -n 8

  test:
    suffix: 1
    nsize: {{2 5 7}}
    args: -n 16

  # Fewer cells than processes, so some ranks own nothing before and after the reorder.
  test:
    suffix: empty_ranks
    nsize: 8
    args: -n 2

TEST*/
