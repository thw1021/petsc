static char help[] = "Test DMPlexCreatePointNumbering_Plex() and DMPlexGetGhostPointMask()\n\n";
#define EX "ex47"

#include <petsc/private/dmpleximpl.h>

typedef struct {
  MPI_Comm          comm;
  PetscMPIInt       rank;
  DM                dm;
  PetscViewer       verbose;
} AppCtx;

static PetscErrorCode DMPlexCreatePointNumbering_Old(DM dm, IS *globalPointNumbers, PetscInt *globalSize)
{
  IS             nums[4];
  PetscInt       depths[4], gdepths[4], starts[4];
  PetscInt       depth, d, shift = 0;
  PetscSF        sf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetPointSF(dm, &sf));
  /* For unstratified meshes use dim instead of depth */
  if (depth < 0) PetscCall(DMGetDimension(dm, &depth));
  for (d = 0; d <= depth; ++d) {
    PetscInt end;

    depths[d] = depth-d;
    PetscCall(DMPlexGetDepthStratum(dm, depths[d], &starts[d], &end));
    if (!(starts[d]-end)) { starts[d] = depths[d] = -1; }
  }
  PetscCall(PetscSortIntWithArray(depth+1, starts, depths));
  PetscCall(MPIU_Allreduce(depths, gdepths, depth+1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm)));
  for (d = 0; d <= depth; ++d) {
    PetscCheck(starts[d] < 0 || depths[d] == gdepths[d],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected depth %" PetscInt_FMT ", found %" PetscInt_FMT,depths[d],gdepths[d]);
  }
  for (d = 0; d <= depth; ++d) {
    PetscInt pStart, pEnd, gsize;

    PetscCall(DMPlexGetDepthStratum(dm, gdepths[d], &pStart, &pEnd));
    PetscCall(DMPlexCreateNumbering_Plex(dm, pStart, pEnd, shift, &gsize, sf, &nums[d]));
    shift += gsize;
  }
  PetscCall(ISConcatenate(PetscObjectComm((PetscObject) dm), depth+1, nums, globalPointNumbers));
  for (d = 0; d <= depth; ++d) PetscCall(ISDestroy(&nums[d]));
  if (globalSize) *globalSize = shift;
  PetscFunctionReturn(0);
}

/* This is the same as deprecated DMPlexCreatePointNumbering() */
static PetscErrorCode DMPlexCreatePointNumbering_Deprecated(DM dm, IS *globalPointNumbers)
{
  const PetscBool   *mask;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMPlexGetPointNumbering(dm, globalPointNumbers, &mask, NULL, NULL));
  PetscCall(ISMakeGhostsNegative_Internal(*globalPointNumbers, mask, globalPointNumbers));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(AppCtx *, DM *);

static PetscErrorCode Initialize(MPI_Comm comm, AppCtx *ctx)
{
  PetscViewerFormat format;

  PetscFunctionBeginUser;
  ctx->comm  = comm;
  PetscCallMPI(MPI_Comm_rank(comm, &ctx->rank));
  PetscCall(CreateMesh(ctx, &ctx->dm));
  PetscOptionsBegin(comm, NULL, EX " options", "DMPLEX");
  PetscCall(PetscOptionsViewer("-verbose_view", "Verbose view", EX, &ctx->verbose, &format, NULL));
  if (ctx->verbose) PetscCall(PetscViewerPushFormat(ctx->verbose, format));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode Finalize(AppCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(DMDestroy(&ctx->dm));
  if (ctx->verbose) PetscCall(PetscViewerPopFormat(ctx->verbose));
  PetscCall(PetscViewerDestroy(&ctx->verbose));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(AppCtx *ctx, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(ctx->comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, ctx));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode VerboseView(AppCtx *ctx, IS is, PetscBool nonnegative, const PetscBool mask[])
{
  PetscViewer   v = ctx->verbose;
  PetscInt      i, n;

  PetscFunctionBeginUser;
  if (!v) PetscFunctionReturn(0);
  PetscCall(ISView(is, v));
  PetscCall(ISGetLocalSize(is, &n));
  if (nonnegative) {
    PetscInt     *maskInt;

    PetscCall(PetscMalloc1(n, &maskInt));
    for (i=0; i<n; i++) maskInt[i] = (PetscInt) mask[i];
    PetscCall(PetscIntView(n, maskInt, v));
    PetscCall(PetscFree(maskInt));
  }
  PetscFunctionReturn(0);
}

#define BoolToStr(i, arr) (arr ? PetscBools[arr[i]] : "NONE")

static PetscErrorCode ISGetIndicesForComparison(AppCtx *ctx, IS is, PetscBool nonnegative, const PetscBool ghostMask[], const PetscInt *idx[])
{
  PetscInt        n;

  PetscFunctionBeginUser;
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetIndices(is, idx));
  if (nonnegative) {
    IS is_cmp;

    PetscCheck(n == 0 || ghostMask, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ghostMask must be passed if nonnegative is true");
    PetscCall(ISMakeGhostsNegative_Internal(is, ghostMask, &is_cmp));
    PetscCall(PetscObjectCompose((PetscObject)is, "IS_with_negative_ghosts", (PetscObject) is_cmp));
    PetscCall(ISGetIndices(is_cmp, idx));
    PetscCall(ISDestroy(&is_cmp));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISRestoreIndicesForComparison(AppCtx *ctx, IS is, PetscBool nonnegative, const PetscBool ghostMask[], const PetscInt *idx[])
{
  IS              is_cmp;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)is, "IS_with_negative_ghosts", (PetscObject*) &is_cmp));
  PetscCheck(!nonnegative || is_cmp, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Assertion failed: !nonnegative || is_cmp");
  if (is_cmp) is = is_cmp;
  PetscCall(ISRestoreIndices(is, idx));
  PetscCall(PetscObjectCompose((PetscObject)is, "IS_with_negative_ghosts", NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode Compare(AppCtx *ctx, IS numbering0, PetscBool nonnegative0, const PetscBool ghostMask0[], PetscInt nGhosts0, IS numbering1, PetscBool nonnegative1, const PetscBool ghostMask1[], PetscInt nGhosts1)
{
  const PetscInt     *idx0, *idx1;
  PetscInt            i, n0, n1, nGhosts=0;

  PetscFunctionBeginUser;
  PetscCall(VerboseView(ctx, numbering0, nonnegative0, ghostMask0));
  PetscCall(VerboseView(ctx, numbering1, nonnegative1, ghostMask1));
  PetscCall(ISGetLocalSize(numbering0, &n0));
  PetscCall(ISGetLocalSize(numbering1, &n1));
  PetscCheck(n0 == n1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "n0 = %" PetscInt_FMT " != %" PetscInt_FMT " = n1", n0, n1);
  PetscCall(ISGetIndicesForComparison(ctx, numbering0, nonnegative0, ghostMask0, &idx0));
  PetscCall(ISGetIndicesForComparison(ctx, numbering1, nonnegative1, ghostMask1, &idx1));
  for (i=0; i<n0; i++) {
    PetscCheck(idx0[i] == idx1[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "i = %" PetscInt_FMT ":  idx0[i] = %" PetscInt_FMT " ghostMask0[i] = %s  idx1[i] = %" PetscInt_FMT " ghostMask1[i] = %s", i, idx0[i], BoolToStr(i, ghostMask0), idx1[i], BoolToStr(i, ghostMask1));
    if (idx0[i] < 0) nGhosts++;
  }
  if (nGhosts0 >= 0) PetscCheck(nGhosts0 == nGhosts, PETSC_COMM_SELF, PETSC_ERR_PLIB, "counted number of ghosts %" PetscInt_FMT " is not equal to nGhosts0 = %" PetscInt_FMT, nGhosts, nGhosts0);
  if (nGhosts1 >= 0) PetscCheck(nGhosts1 == nGhosts, PETSC_COMM_SELF, PETSC_ERR_PLIB, "counted number of ghosts %" PetscInt_FMT " is not equal to nGhosts1 = %" PetscInt_FMT, nGhosts, nGhosts1);
  PetscCall(ISRestoreIndicesForComparison(ctx, numbering0, nonnegative0, ghostMask0, &idx0));
  PetscCall(ISRestoreIndicesForComparison(ctx, numbering1, nonnegative1, ghostMask1, &idx1));
  PetscFunctionReturn(0);
}

typedef enum {
  ALL,
  DEPTH,
  HEIGHT,
  N_MODES
} Test1Mode;

const char * const Test1Modes[] = {"all", "depth", "height", NULL};

static PetscErrorCode Test1_Private(AppCtx *ctx, DM dm, Test1Mode mode, PetscInt stratum)
{
  IS                  numbering0 = NULL, numbering1 = NULL;
  PetscInt            globalSize0 = -1, globalSize1 = -1;
  PetscLayout         ownedLayout1 = NULL, ghostLayout1 = NULL;
  const PetscBool    *ghostMask1;

  PetscFunctionBeginUser;
  if (ctx->verbose) PetscCall(PetscViewerASCIIPrintf(ctx->verbose, "# Test1: mode %s stratum %" PetscInt_FMT "\n", Test1Modes[mode], stratum));
  {
    PetscSF   pointSF;
    PetscInt  pStart = -1, pEnd = -1;

    PetscCall(DMGetPointSF(dm, &pointSF));
    switch (mode) {
      case ALL:     break;
      case DEPTH:   PetscCall(DMPlexGetDepthStratum(dm, stratum, &pStart, &pEnd)); break;
      case HEIGHT:  PetscCall(DMPlexGetHeightStratum(dm, stratum, &pStart, &pEnd)); break;
      default: SETERRQ(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid mode %d", mode);
    }
    switch (mode) {
      case DEPTH:
      case HEIGHT:
        PetscCall(DMPlexCreateNumbering_Plex(dm, pStart, pEnd, 0, &globalSize0, pointSF, &numbering0)); break;
      default:
        PetscCall(DMPlexCreatePointNumbering_Old(dm, &numbering0, &globalSize0));
    }
  }
  switch (mode) {
    case DEPTH:     PetscCall(DMPlexGetDepthStratumNumbering(dm, stratum, &numbering1, &ghostMask1, &ownedLayout1, &ghostLayout1)); break;
    case HEIGHT:    PetscCall(DMPlexGetHeightStratumNumbering(dm, stratum, &numbering1, &ghostMask1, &ownedLayout1, &ghostLayout1)); break;
    default:        PetscCall(DMPlexGetPointNumbering(dm, &numbering1, &ghostMask1, &ownedLayout1, &ghostLayout1)); break;
  }
  globalSize1 = ownedLayout1->N;
  PetscCheck(globalSize0 == globalSize1, ctx->comm, PETSC_ERR_PLIB, "globalSize0 = %" PetscInt_FMT " != globalSize1 = %" PetscInt_FMT, globalSize0, globalSize1);
  PetscCall(Compare(ctx, numbering0, PETSC_FALSE, NULL, -1, numbering1, PETSC_TRUE, ghostMask1, ghostLayout1->n));
  PetscCall(ISDestroy(&numbering0));
  PetscFunctionReturn(0);
}

static PetscErrorCode Test1(AppCtx *ctx)
{
  DM                  dm = ctx->dm;
  PetscInt            d, depth;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(Test1_Private(ctx, dm, ALL, -1));
  for (d=0; d<=depth; d++) {
    PetscCall(Test1_Private(ctx, dm, DEPTH, d));
  }
  for (d=0; d<=depth; d++) {
    PetscCall(Test1_Private(ctx, dm, HEIGHT, d));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode Test2(AppCtx *ctx)
{
  DM                  dm = ctx->dm;
  IS                  is0;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreatePointNumbering_Old(dm, &is0, NULL));
  {
    IS is1;

    if (ctx->verbose) PetscCall(PetscViewerASCIIPrintf(ctx->verbose, "# Test2 A\n"));
    PetscCall(DMPlexCreatePointNumbering_Deprecated(dm, &is1));
    PetscCall(Compare(ctx, is0, PETSC_FALSE, NULL, -1, is1, PETSC_FALSE, NULL, -1));
    PetscCall(ISDestroy(&is1));
  }
  {
    IS                is1;
    const PetscBool  *ghostMask1;
    PetscLayout       ghostLayout1;

    if (ctx->verbose) PetscCall(PetscViewerASCIIPrintf(ctx->verbose, "# Test2 B\n"));
    PetscCall(DMPlexGetPointNumbering(dm, &is1, &ghostMask1, NULL, &ghostLayout1));
    PetscCall(Compare(ctx, is0, PETSC_FALSE, NULL, -1, is1, PETSC_TRUE, ghostMask1, ghostLayout1->n));
  }
  PetscCall(ISDestroy(&is0));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx              ctx;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(Initialize(PETSC_COMM_WORLD, &ctx));
  PetscCall(Test1(&ctx));
  PetscCall(Test2(&ctx));
  PetscCall(Finalize(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: ctetgen
    nsize: 2
    args: -dm_plex_dim 2 -dm_plex_box_faces 2 -dm_distribute -petscpartitioner_type simple -dm_view ::ascii_info_detail -verbose_view

  test:
    suffix: 1
    requires: ctetgen
    nsize: {{1 3 7}}
    args: -dm_plex_dim {{1 2 3}} -dm_plex_box_faces 30 -dm_distribute {{0 1}} -dm_plex_interpolate {{0 1}} -petscpartitioner_type simple

TEST*/
