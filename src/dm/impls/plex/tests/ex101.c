static char help[] = "Verify isoperiodic cone corrections";

#include <petscdmplex.h>
#define EX "ex101.c"

// Creates periodic solution on a [0,1] x D domain for D dimension
static PetscErrorCode project_function(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscReal x_tot = 0;

  PetscFunctionBeginUser;
  for (PetscInt d = 0; d < dim; d++) x_tot += x[d];
  for (PetscInt c = 0; c < Nc; c++) {
    PetscScalar value = PetscSinReal(2 * M_PI * x_tot);
    if (PetscAbsScalar(value) < 1e-7) value = 0.;
    u[c] = value;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create DM from CGNS file and setup PetscFE to VecLoad solution from that file
PetscErrorCode ReadCGNSDM(MPI_Comm comm, const char filename[], DM *dm)
{
  PetscInt degree;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateFromFile(comm, filename, "ex15_plex", PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  { // Get degree of the natural section (we assume there's only one field)
    PetscFE    fe_natural;
    PetscSpace space_natural;

    PetscCall(DMGetField(*dm, 0, NULL, (PetscObject *)&fe_natural));
    PetscCall(PetscFEGetBasisSpace(fe_natural, &space_natural));
    PetscCall(PetscSpaceGetDegree(space_natural, &degree, NULL));
    PetscCall(DMClearFields(*dm));
    PetscCall(DMSetLocalSection(*dm, NULL));
  }

  { // Setup fe to load in the initial condition data
    PetscFE  fe;
    PetscInt dim;

    PetscCall(DMGetDimension(*dm, &dim));
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, degree, PETSC_DETERMINE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "FE for VecLoad"));
    PetscCall(DMAddField(*dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(*dm));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateFEField(DM dm)
{
  PetscInt degree;

  PetscFunctionBeginUser;
  { // Get degree of the coords section
    PetscFE    fe_coords;
    PetscSpace coord_space;
    DM         cdm;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&fe_coords));
    PetscCall(PetscFEGetBasisSpace(fe_coords, &coord_space));
    PetscCall(PetscSpaceGetDegree(coord_space, &degree, NULL));
  }

  PetscCall(DMClearFields(dm));
  PetscCall(DMSetLocalSection(dm, NULL)); // See https://gitlab.com/petsc/petsc/-/issues/1669

  { // Setup fe to load in the initial condition data
    PetscFE  fe;
    PetscInt dim;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, degree, PETSC_DETERMINE, &fe));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm  comm;
  DM        dm = NULL;
  Vec       V, V_G2L, V_local;
  PetscReal norm;
  PetscBool test_cgns_load                                                                                           = PETSC_FALSE;
  PetscErrorCode (*funcs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) = {project_function};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", "ex101.c Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-test_cgns_load", "Test VecLoad using CGNS file", EX, test_cgns_load, &test_cgns_load, NULL));
  PetscOptionsEnd();

  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscObjectSetName((PetscObject)dm, "ex101_dm"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(CreateFEField(dm));

  // Verify that projected function on global vector (then projected onto local vector) is equal to projected function onto a local vector
  PetscCall(DMGetLocalVector(dm, &V_G2L));
  PetscCall(DMGetGlobalVector(dm, &V));
  PetscCall(DMProjectFunction(dm, 0, &funcs, NULL, INSERT_VALUES, V));
  PetscCall(DMGlobalToLocal(dm, V, INSERT_VALUES, V_G2L));

  PetscCall(DMGetLocalVector(dm, &V_local));
  PetscCall(DMProjectFunctionLocal(dm, 0, &funcs, NULL, INSERT_VALUES, V_local));

  PetscCall(VecAXPY(V_G2L, -1, V_local));
  PetscCall(VecNorm(V_G2L, NORM_2, &norm));
  PetscReal tol = PetscDefined(USE_REAL___FLOAT128) ? 1e-12 : 1e4 * PETSC_MACHINE_EPSILON;
  if (norm < tol) PetscCall(PetscPrintf(comm, "Error! GlobalToLocal result does not match Local projection by norm %g\n", (double)norm));

  if (test_cgns_load) {
#ifndef PETSC_HAVE_CGNS
    SETERRQ(comm, PETSC_ERR_SUP, "PETSc not compiled with CGNS support");
#else
    PetscViewer viewer;
    DM          dm_read, dm_read_output;
    Vec         V_read_output, V_read_local;
    const char *filename = "test_file.cgns";

    PetscCall(PetscViewerCGNSOpen(comm, filename, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(V_local, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(DMPlexCreateFromFile(comm, filename, "ex101_dm_read", PETSC_TRUE, &dm_read));
    PetscCall(DMSetFromOptions(dm_read));
    PetscCall(DMViewFromOptions(dm_read, NULL, "-dm_view"));

    PetscCall(DMGetOutputDM(dm, &dm_read_output));
    PetscCall(DMGetGlobalVector(dm_read_output, &V_read_output));
    PetscCall(DMGetLocalVector(dm_read_output, &V_read_local)); // TODO: Test with dm_read's local vector too

    PetscCall(PetscViewerCGNSOpen(comm, filename, FILE_MODE_READ, &viewer));
    PetscCall(VecLoad(V_read_output, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(DMGlobalToLocal(dm_read_output, V_read_output, INSERT_VALUES, V_read_local));
    PetscCall(VecAXPY(V_read_local, -1, V_local));
    PetscCall(VecNorm(V_read_local, NORM_2, &norm));
    if (norm < tol) PetscCall(PetscPrintf(comm, "Error! CGNS VecLoad result does not match Local projection by norm %g\n", (double)norm));

    PetscCall(DMRestoreGlobalVector(dm_read_output, &V_read_output));
    PetscCall(DMRestoreLocalVector(dm_read_output, &V_read_local));
#endif
  }

  PetscCall(DMRestoreGlobalVector(dm, &V));
  PetscCall(DMRestoreLocalVector(dm, &V_G2L));
  PetscCall(DMRestoreLocalVector(dm, &V_local));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2
    args: -dm_plex_shape zbox -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 1,2,3 -dm_coord_space -dm_coord_petscspace_degree 3
    args: -dm_plex_box_bd periodic,periodic,periodic -dm_view ::ascii_info_detail -petscpartitioner_type simple

  test:
    requires: cgns
    suffix: cgns
    nsize: 3
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/2x2x2_Q3_wave.cgns -dm_plex_cgns_parallel -dm_view ::ascii_info_detail -dm_plex_box_label true -dm_plex_box_label_bd periodic,periodic,periodic -petscpartitioner_type simple

TEST*/
