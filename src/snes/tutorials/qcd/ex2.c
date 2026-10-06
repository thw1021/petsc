static char help[] = "Set Gauge fields with GRID and compare GRID and PETSc operators.\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscgrid.h>

const PetscReal M = -0.92;
//const PetscReal M = 0.1;

typedef struct {
  char           gridFile[PETSC_MAX_PATH_LEN];
  GRID_LOAD_TYPE gauge_type;
  PetscReal      mass;
  PetscInt       Ls;
  PetscBool      fixGauge;
  PetscBool      checkDdwf;
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;

  options->mass      = 0.001;//lower = more poorly conditioned
  options->Ls        = 0;
  options->fixGauge  = PETSC_FALSE;
  options->checkDdwf = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-grid_file", options->gridFile, sizeof(options->gridFile), NULL));
  PetscCall(PetscOptionsReal("-mass", "Fermion mass parameter", "ex2.c", options->mass, &options->mass, NULL));
  PetscCall(PetscOptionsInt("-Ls", "Number of d5 slices", "ex2.c", options->Ls, &options->Ls, NULL));
  PetscCall(PetscOptionsBool("-petsc_grid_fix_gauge", "Run Grid's steepest-descent gauge fixing", "ex2.c", options->fixGauge, &options->fixGauge, NULL));
  PetscCall(PetscOptionsBool("-domain_wall", "Run Ddwf check", "ex2.c", options->checkDdwf, &options->checkDdwf, NULL));
  //PetscCall(PetscOptionsEnum("-grid_load_type", "How to initialize data from grid", NULL, GRID_LOAD_TYPE, (PetscEnum)options->gauge_type, (PetscEnum *)&options->gauge_type, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  PetscSection s;
  PetscInt     vStart, vEnd, v;

  PetscFunctionBegin;
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscSectionSetChart(s, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionSetDof(s, v, 12));
    /* TODO Divide the values into fields/components */
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(dm, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupAuxDiscretization(DM dm, AppCtx *user)
{
  DM           dmAux, coordDM;
  PetscSection s;
  Vec          gauge;
  PetscInt     eStart, eEnd, e;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMSetCoordinateDM(dmAux, coordDM));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  PetscCall(PetscSectionSetChart(s, eStart, eEnd));
  for (e = eStart; e < eEnd; ++e) {
    /* TODO Should we store the whole SU(3) matrix, or the symmetric part? */
    PetscCall(PetscSectionSetDof(s, e, 9));
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(dmAux, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscCall(DMCreateLocalVector(dmAux, &gauge));
  PetscCall(DMDestroy(&dmAux));
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, gauge));
  PetscCall(VecDestroy(&gauge));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscRandom r;
  DM          dm;
  Vec         u, f;
  Mat         M;
  AppCtx      user;
  MPI_Comm    comm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscInitializeGrid(argc, argv));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &user, &dm));
  PetscCall(DMSetApplicationContext(dm, &user));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(SetupAuxDiscretization(dm, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &f));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  PetscCall(VecSetRandom(u, r));
  PetscCall(PetscRandomDestroy(&r));
  if (user.checkDdwf) {
    PetscCall(PetscSetGauge_Grid(dm, GRID_LATTICE_FILE, PETSC_FALSE, user.Ls, user.fixGauge, NULL, user.gridFile));
    PetscCall(PetscGridSetUpDdwf(dm, PETSC_FALSE, &M));
    PetscCall(PetscCheckDwfWithGrid(dm, M, u, f, user.gridFile));
  } else {
    PetscCall(PetscSetGauge_Grid(dm, GRID_LATTICE_FILE, PETSC_FALSE, user.Ls, user.fixGauge, NULL, user.gridFile));
    PetscCall(PetscGridSetUpWilson(dm, PETSC_FALSE, &M));
    PetscCall(PetscCheckWilsonWithGrid(dm, M, u, f, user.gridFile));
  }

  PetscCall(MatDestroy(&M));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&f));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: complex grid
  testset:
    suffix: petsc_grid_8888
    args: -grid_file ${GRID_LATTICE_DIR}/ckpoint_EODWF_lat.125
    test:
      suffix: wilson
      args: -dm_plex_box_faces 8,8,8,8 --grid 8.8.8.8 \
            -dm_plex_dim 4 -dm_plex_shape hypercubic
    test:
      suffix: dwf
      args: -dm_plex_box_faces 8,8,8,8,8 --grid 8.8.8.8 \
            -dm_plex_dim 5 -dm_plex_shape hypercubic \
            -Ls 8 -domain_wall
  testset:
    suffix: petsc_grid_16161632
    args: -grid_file ${GRID_LATTICE_DIR}/ckpoint_lat.4000
    test:
      suffix: wilson
      args: -dm_plex_box_faces 16,16,16,32 --grid 16.16.16.32 \
            -dm_plex_dim 4 -dm_plex_shape hypercubic
    test:
      suffix: dwf
      args: -dm_plex_box_faces 16,16,16,16,32 --grid 16.16.16.32 \
            -dm_plex_dim 5 -dm_plex_shape hypercubic \
            -Ls 16 -domain_wall
TEST*/
