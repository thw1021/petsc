const char help[] = "Test DMPlexDistributeOverlap() after DMPlexSetCellType() calls on a distributed mesh.\n\
The cell type cache must not serve uninitialized values for points whose type was never set,\n\
and the cell type label must not keep answering a stale value after a type change.\n";

#include <petscdmplex.h>

/* Write cell types onto all cells and faces of an already distributed mesh, as a user of
   shape-agnostic polytopes does. Mode "same" rewrites the types the points already carry
   (value-wise a no-op), "typed" uses DM_POLYTOPE_UNKNOWN_CELL/DM_POLYTOPE_UNKNOWN_FACE,
   and "unknown" uses DM_POLYTOPE_UNKNOWN for both. The old value is read from the label,
   not from DMPlexGetCellType(), so that this function does not build the cell type cache
   itself. */
static PetscErrorCode RestampCellTypes(DM dm, PetscInt mode)
{
  DMLabel  label;
  PetscInt h, pStart, pEnd, p;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetCellTypeLabel(dm, &label));
  for (h = 0; h < 2; ++h) {
    PetscCall(DMPlexGetHeightStratum(dm, h, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt ct;

      if (mode == 0) PetscCall(DMLabelGetValue(label, p, &ct));
      else if (mode == 1) ct = h ? DM_POLYTOPE_UNKNOWN_FACE : DM_POLYTOPE_UNKNOWN_CELL;
      else ct = DM_POLYTOPE_UNKNOWN;
      PetscCall(DMPlexSetCellType(dm, p, (DMPolytopeType)ct));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckCellTypes(DM dm, PetscInt mode)
{
  DMLabel  label;
  PetscInt pStart, pEnd, p, h;

  PetscFunctionBeginUser;
  /* The cache and the label must agree on every point of the chart */
  PetscCall(DMPlexGetCellTypeLabel(dm, &label));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType ct;
    PetscInt       lct;

    PetscCall(DMPlexGetCellType(dm, p, &ct));
    PetscCall(DMLabelGetValue(label, p, &lct));
    PetscCheck((PetscInt)ct == lct, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT ": cell type %d does not match label value %" PetscInt_FMT, p, (int)ct, lct);
  }
  /* Restamped strata must carry the type that was set */
  if (mode) {
    for (h = 0; h < 2; ++h) {
      const DMPolytopeType want = mode == 1 ? (h ? DM_POLYTOPE_UNKNOWN_FACE : DM_POLYTOPE_UNKNOWN_CELL) : DM_POLYTOPE_UNKNOWN;

      PetscCall(DMPlexGetHeightStratum(dm, h, &pStart, &pEnd));
      for (p = pStart; p < pEnd; ++p) {
        DMPolytopeType ct;

        PetscCall(DMPlexGetCellType(dm, p, &ct));
        PetscCheck(ct == want, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " at height %" PetscInt_FMT ": cell type %d, expected %d", p, h, (int)ct, (int)want);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm, dmOv = NULL;
  const char *modes[3] = {"same", "typed", "unknown"};
  PetscInt    mode     = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetEList(NULL, NULL, "-stamp", modes, 3, &mode, NULL));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  {
    /* Distribute with an explicit call: -dm_distribute leaves the cell type cache of the
       distributed DM built, and a built cache masks the uninitialized-cache defect */
    DM dmDist = NULL;

    PetscCall(DMPlexDistribute(dm, 0, NULL, &dmDist));
    if (dmDist) {
      PetscCall(DMDestroy(&dm));
      dm = dmDist;
    }
  }
  PetscCall(PetscObjectSetName((PetscObject)dm, "Mesh"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(RestampCellTypes(dm, mode));
  PetscCall(DMPlexDistributeOverlap(dm, 1, NULL, &dmOv));
  PetscCall(CheckCellTypes(dm, mode));
  if (dmOv) {
    PetscCall(PetscObjectSetName((PetscObject)dmOv, "Overlap Mesh"));
    PetscCall(DMViewFromOptions(dmOv, NULL, "-dm_view"));
    PetscCall(DMPlexCheckSymmetry(dmOv));
    PetscCall(CheckCellTypes(dmOv, mode));
    PetscCall(DMDestroy(&dmOv));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Overlap distribution after cell type changes succeeded\n"));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 3,3,3 -petscpartitioner_type simple -dm_plex_adj_cone -dm_plex_adj_closure 0

    test:
      suffix: same
      nsize: 2
      args: -stamp same

    test:
      suffix: typed
      nsize: 2
      args: -stamp typed

    test:
      suffix: unknown
      nsize: 2
      args: -stamp unknown

    test:
      suffix: unknown_3
      nsize: 3
      args: -stamp unknown

TEST*/
