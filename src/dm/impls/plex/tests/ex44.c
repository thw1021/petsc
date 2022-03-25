static const char help[] = "Tests for mesh extrusion";

#include <petscdmplex.h>

typedef struct {
  char     bdLabel[PETSC_MAX_PATH_LEN]; /* The boundary label name */
  PetscInt Nbd;                         /* The number of boundary markers to extrude, 0 for all */
  PetscInt bd[64];                      /* The boundary markers to be extruded */
} AppCtx;

PETSC_EXTERN PetscErrorCode pyramidNormal(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);

/* The pyramid apex is at (0.5, 0.5, -1) */
PetscErrorCode pyramidNormal(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt r, PetscScalar u[], void *ctx)
{
  PetscReal apex[3] = {0.5, 0.5, -1.0};
  PetscInt  d;

  for (d = 0;   d < dim; ++d) u[d] = x[d] - apex[d];
  for (d = dim; d < 3;   ++d) u[d] = 0.0  - apex[d];
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 64;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcpy(options->bdLabel, "marker");CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Parallel Mesh Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-label", "The boundary label name", "ex44.c", options->bdLabel, options->bdLabel, sizeof(options->bdLabel), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-bd", "The boundaries to be extruded", "ex44.c", options->bd, &n, &flg);CHKERRQ(ierr);
  options->Nbd = flg ? n : 0;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAdaptLabel(DM dm, AppCtx *ctx, DMLabel *adaptLabel)
{
  DMLabel        label;
  PetscInt       b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ctx->Nbd) {*adaptLabel = NULL; PetscFunctionReturn(0);}
  ierr = DMGetLabel(dm, ctx->bdLabel, &label);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel);CHKERRQ(ierr);
  for (b = 0; b < ctx->Nbd; ++b) {
    IS              bdIS;
    const PetscInt *points;
    PetscInt        n, i;

    ierr = DMLabelGetStratumIS(label, ctx->bd[b], &bdIS);CHKERRQ(ierr);
    if (!bdIS) continue;
    ierr = ISGetLocalSize(bdIS, &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bdIS, &points);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {ierr = DMLabelSetValue(*adaptLabel, points[i], DM_ADAPT_REFINE);CHKERRQ(ierr);}
    ierr = ISRestoreIndices(bdIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dma;
  DMLabel        adaptLabel;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = CreateAdaptLabel(dm, &ctx, &adaptLabel);CHKERRQ(ierr);
  if (adaptLabel) {ierr = DMAdaptLabel(dm, adaptLabel, &dma);CHKERRQ(ierr);}
  else            {ierr = DMExtrude(dm, 3, &dma);CHKERRQ(ierr);}
  ierr = PetscObjectSetName((PetscObject) dma, "Adapted Mesh");CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dma, NULL, "-adapt_dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dma);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: tri_tensor_0
    requires: triangle
    args: -dm_plex_transform_extrude_use_tensor {{0 1}separate output} \
          -dm_view -adapt_dm_view -dm_plex_check_all

  test:
    suffix: quad_tensor_0
    args: -dm_plex_simplex 0 -dm_plex_transform_extrude_use_tensor {{0 1}separate output} \
          -dm_view -adapt_dm_view -dm_plex_check_all

  test:
    suffix: quad_normal_0
    args: -dm_plex_simplex 0 -dm_plex_transform_extrude_normal 0,1,1 \
          -dm_view -adapt_dm_view -dm_plex_check_all

  test:
    suffix: quad_normal_1
    args: -dm_plex_simplex 0 -dm_plex_transform_extrude_normal_function pyramidNormal \
          -dm_view -adapt_dm_view -dm_plex_check_all

  test:
    suffix: quad_symmetric_0
    args: -dm_plex_simplex 0 -dm_plex_transform_extrude_symmetric \
          -dm_view -adapt_dm_view -dm_plex_check_all

  testset:
    args: -dm_adaptor cellrefiner -dm_plex_transform_type extrude \
          -dm_view -adapt_dm_view

    test:
      suffix: quad_adapt_0
      args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_plex_separate_marker -bd 1,3

TEST*/
