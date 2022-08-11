static char help[] = "Tests mesh-to-mesh interpolation.\n\n";

#include <petscdmplex.h>

/*
  In order to see the distribution in 2D, use

    -source_dm_view draw -target_dm_view draw -draw_pause 3

  Some useful commands for writing the field in VTU format before
  interpolation, after interpolating onto the target space and then
  after interpolating back to the base space:

    -source_vec_view vtk:output/ex61_source.vtu
    -target_vec_view vtk:output/ex61_target.vtu
    -final_vec_view vtk:output/ex61_final.vtu

  If multiple mesh-to-mesh interpolations are performed then the
  l2 errors at each step can be outputted using:

    -error_vec_view ascii:output/ex61_error.txt
*/

static PetscErrorCode CreateMesh(MPI_Comm comm, const char prefix[], DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *dm, prefix));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode sinusoid(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscFunctionBeginUser;
  u[0] = PetscSinReal(PETSC_PI*x[0]);
  if (dim > 1) u[0] *= PetscSinReal(PETSC_PI*x[1]);
  if (dim > 2) u[0] *= PetscSinReal(PETSC_PI*x[2]);
  PetscFunctionReturn(0);
}

/*
  Sensor field options:
    0: constant unity;
    1: sinusoid.
*/
static PetscErrorCode SetSensor(DM dm, PetscInt sensor, Vec v)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*) = {sinusoid};

  PetscFunctionBeginUser;
  switch (sensor) {
    case 0:
      PetscCall(VecSet(v, 1.0));
      break;
    case 1:
      PetscCall(DMProjectFunction(dm, 0., funcs, NULL, INSERT_VALUES, v));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "sensor = 0, 1, cannot be %" PetscInt_FMT "", sensor);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  DM             dmS, dmT;
  Mat            A;
  MPI_Comm       comm;
  PetscBool      simplexS, simplexT;
  PetscErrorCode ierr;
  PetscFE        fe;
  PetscInt       dim, n = 1, i, sensor = 0;
  PetscScalar   *e, minS, minT, maxS, maxT, tol = 1.0e-04;
  Vec            original, source, target, error, errors, scaling;

  /* Parse user input */
  ierr = PetscInitialize(&argc, &argv, NULL, help);PetscCall(ierr);
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, "", "Mesh-to-mesh interpolation options", "DMPlex");
  PetscCall(PetscOptionsBoundedInt("-n", "The number of mesh-to-mesh interpolation steps", "ex61.c", n, &n, NULL, 0));
  PetscCall(PetscOptionsRangeInt("-sensor", "Code for the sensor field", "ex61.c", sensor, &sensor, NULL, 0, 1));
  PetscOptionsEnd();

  /* Create two arbitrary meshes of the unit box/cube */
  PetscCall(CreateMesh(comm, "source_", &dmS));
  PetscCall(CreateMesh(comm, "target_", &dmT));
  PetscCall(DMPlexIsSimplex(dmS, &simplexS));
  PetscCall(DMPlexIsSimplex(dmT, &simplexT));
  PetscCheck((simplexS && simplexT) || (!simplexS && !simplexT), comm, PETSC_ERR_ARG_WRONG, "The two meshes should have consistent cell types");

  /* Create fields on each mesh */
  PetscCall(DMGetDimension(dmS, &dim));
  PetscCall(PetscFECreateDefault(comm, dim, 1, simplexS, "source_", -1, &fe));
  PetscCall(DMSetField(dmS, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dmS));
  PetscCall(PetscFECreateDefault(comm, dim, 1, simplexT, "target_", -1, &fe));
  PetscCall(DMSetField(dmT, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dmT));

  /* Set source field */
  PetscCall(DMCreateGlobalVector(dmS, &original));
  PetscCall(DMCreateGlobalVector(dmS, &source));
  PetscCall(DMCreateGlobalVector(dmT, &target));
  PetscCall(SetSensor(dmS, sensor, original));
  PetscCall(VecCopy(original, source));
  PetscCall(VecMin(source, NULL, &minS));
  PetscCall(VecMax(source, NULL, &maxS));
  PetscCall(VecViewFromOptions(source, NULL, "-source_vec_view"));

  /* Create the interpolation matrix */
  PetscCall(DMCreateInterpolation(dmS, dmT, &A, &scaling));

  /* Create vectors to compute the errors */
  PetscCall(VecDuplicate(source, &error));
  PetscCall(VecCreate(comm, &errors));
  PetscCall(VecSetType(errors, VECMPI));
  PetscCall(VecSetSizes(errors, PETSC_DECIDE, n+1));
  PetscCall(VecGetArray(errors, &e));

  /* Interpolate back and forth between source mesh and target mesh */
  for (i = 0; i < n; i++) {

    /* Check whether new extrema are introduced by refinement */
    PetscCall(MatInterpolate(A, source, target));
    PetscCall(VecMin(target, NULL, &minT));
    PetscCall(VecMax(target, NULL, &maxT));
    if (PetscRealPart(minT) < PetscRealPart(minS - tol)) PetscPrintf(comm, "refine: new minima introduced\n");
    else PetscPrintf(comm, "refine: no new minima introduced\n");
    if (PetscRealPart(maxT) > PetscRealPart(maxS + tol)) PetscPrintf(comm, "refine: new maxima introduced\n");
    else PetscPrintf(comm, "refine: no new maxima introduced\n");

    /* Compute l2 error */
    PetscCall(VecCopy(source, error));
    PetscCall(VecAXPY(error, -1.0, original));
    PetscCall(VecNorm(error, NORM_2, &e[i]));

    /* Check whether new extrema are introduced by coarsening */
    PetscCall(MatInterpolate(A, target, source));
    PetscCall(VecPointwiseMult(source, source, scaling));
    PetscCall(VecMin(source, NULL, &minS));
    PetscCall(VecMax(source, NULL, &maxS));
    if (PetscRealPart(minS) < PetscRealPart(minT - tol)) PetscPrintf(comm, "coarsen: new minima introduced\n");
    else PetscPrintf(comm, "coarsen: no new minima introduced\n");
    if (PetscRealPart(maxS) > PetscRealPart(maxT + tol)) PetscPrintf(comm, "coarsen: new maxima introduced\n");
    else PetscPrintf(comm, "coarsen: no new maxima introduced\n");
  }
  PetscCall(VecViewFromOptions(target, NULL, "-target_vec_view"));
  PetscCall(VecViewFromOptions(source, NULL, "-final_vec_view"));

  /* Compute final l2 error */
  PetscCall(VecCopy(source, error));
  PetscCall(VecAXPY(error, -1.0, original));
  PetscCall(VecNorm(error, NORM_2, &e[n]));
  PetscCall(VecRestoreArray(errors, &e));
  PetscCall(VecViewFromOptions(errors, NULL, "-error_vec_view"));

  /* Clean up */
  PetscCall(VecDestroy(&errors));
  PetscCall(VecDestroy(&error));
  PetscCall(VecDestroy(&scaling));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&target));
  PetscCall(VecDestroy(&source));
  PetscCall(VecDestroy(&original));
  PetscCall(DMDestroy(&dmT));
  PetscCall(DMDestroy(&dmS));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Nested triangular meshes, matching distribution
  testset:
    args: -source_dm_distribute -source_dm_plex_box_faces 4,4 -source_dm_plex_hash_location \
          -target_dm_distribute -target_dm_plex_box_faces 4,4 -target_dm_plex_hash_location -target_dm_refine 1
    test:
      suffix: nested_ref_tri_deg0_unity_serial
    test:
      suffix: nested_ref_tri_deg0_sinusoid_serial
      args: -sensor 1
    test:
      suffix: nested_ref_tri_deg1_unity_serial
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nested_ref_tri_deg1_sinusoid_serial
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nested_ref_tri_deg0_unity_parallel
      nsize: 2
    test:
      suffix: nested_ref_tri_deg0_sinusoid_parallel
      args: -sensor 1
      nsize: 2
    test:
      suffix: nested_ref_tri_deg1_unity_parallel
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2
    test:
      suffix: nested_ref_tri_deg1_sinusoid_parallel
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2

  # Nested triangular meshes, non-matching distribution
  testset:
    args: -source_dm_distribute -source_dm_plex_box_faces 4,4 -source_dm_plex_hash_location \
          -target_dm_distribute -target_dm_plex_box_faces 8,8 -target_dm_plex_hash_location
    test:
      suffix: nested_tri_deg0_unity_serial
    test:
      suffix: nested_tri_deg0_sinusoid_serial
      args: -sensor 1
    test:
      suffix: nested_tri_deg1_unity_serial
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nested_tri_deg1_sinusoid_serial
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nested_tri_deg0_unity_parallel
      nsize: 2
    test:
      suffix: nested_tri_deg0_sinusoid_parallel
      args: -sensor 1
      nsize: 2
    test:
      suffix: nested_tri_deg1_unity_parallel
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2
    test:
      suffix: nested_tri_deg1_sinusoid_parallel
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2

  # Non-nested triangular meshes, non-matching distribution
  testset:
    args: -source_dm_distribute -source_dm_plex_box_faces 4,4 -source_dm_plex_hash_location \
          -target_dm_distribute -target_dm_plex_box_faces 6,6 -target_dm_plex_hash_location
    test:
      suffix: nonnested_tri_deg0_unity_serial
    test:
      suffix: nonnested_tri_deg0_sinusoid_serial
      args: -sensor 1
    test:
      suffix: nonnested_tri_deg1_unity_serial
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nonnested_tri_deg1_sinusoid_serial
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nonnested_tri_deg0_unity_parallel
      nsize: 2
    test:
      suffix: nonnested_tri_deg0_sinusoid_parallel
      args: -sensor 1
      nsize: 2
    test:
      suffix: nonnested_tri_deg1_unity_parallel
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2
    test:
      suffix: nonnested_tri_deg1_sinusoid_parallel
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2

  # Nested tetrahedral meshes, matching distribution
  testset:
    args: -source_dm_plex_dim 3 -source_dm_distribute -source_dm_plex_box_faces 4,4,4 -source_dm_plex_hash_location \
          -target_dm_plex_dim 3 -target_dm_distribute -target_dm_plex_box_faces 4,4,4 -target_dm_plex_hash_location -target_dm_refine 1
    test:
      suffix: nested_ref_tet_deg0_unity_serial
    test:
      suffix: nested_ref_tet_deg0_sinusoid_serial
      args: -sensor 1
    test:
      suffix: nested_ref_tet_deg1_unity_serial
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nested_ref_tet_deg1_sinusoid_serial
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nested_ref_tet_deg0_unity_parallel
      nsize: 2
    test:
      suffix: nested_ref_tet_deg0_sinusoid_parallel
      args: -sensor 1
      nsize: 2
    test:
      suffix: nested_ref_tet_deg1_unity_parallel
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2
    test:
      suffix: nested_ref_tet_deg1_sinusoid_parallel
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2

  # Nested tetrahedral meshes, non-matching distribution
  testset:
    args: -source_dm_plex_dim 3 -source_dm_distribute -source_dm_plex_box_faces 4,4,4 -source_dm_plex_hash_location \
          -target_dm_plex_dim 3 -target_dm_distribute -target_dm_plex_box_faces 8,8,8 -target_dm_plex_hash_location
    test:
      suffix: nested_tet_deg0_unity_serial
    test:
      suffix: nested_tet_deg0_sinusoid_serial
      args: -sensor 1
    test:
      suffix: nested_tet_deg1_unity_serial
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nested_tet_deg1_sinusoid_serial
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nested_tet_deg0_unity_parallel
      nsize: 2
    test:
      suffix: nested_tet_deg0_sinusoid_parallel
      args: -sensor 1
      nsize: 2
    test:
      suffix: nested_tet_deg1_unity_parallel
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2
    test:
      suffix: nested_tet_deg1_sinusoid_parallel
      args: -sensor -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2

  # Non-nested tetrahedral meshes, non-matching distribution
  testset:
    args: -source_dm_plex_dim 3 -source_dm_distribute -source_dm_plex_box_faces 4,4,4 -source_dm_plex_hash_location \
          -target_dm_plex_dim 3 -target_dm_distribute -target_dm_plex_box_faces 6,6,6 -target_dm_plex_hash_location
    test:
      suffix: nonnested_tet_deg0_unity_serial
    test:
      suffix: nonnested_tet_deg0_sinusoid_serial
      args: -sensor 1
    test:
      suffix: nonnested_tet_deg1_unity_serial
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nonnested_tet_deg1_sinusoid_serial
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
    test:
      suffix: nonnested_tet_deg0_unity_parallel
      nsize: 2
    test:
      suffix: nonnested_tet_deg0_sinusoid_parallel
      args: -sensor
      nsize: 2
    test:
      suffix: nonnested_tet_deg1_unity_parallel
      args: -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2
    test:
      suffix: nonnested_tet_deg1_sinusoid_parallel
      args: -sensor 1 -source_petscspace_degree 1 -target_petscspace_degree 1
      nsize: 2

  # Nested quadrilateral meshes, matching distribution
  testset:
    args: -source_dm_plex_simplex 0 -source_dm_distribute -source_dm_plex_box_faces 4,4 -source_dm_plex_hash_location \
          -target_dm_plex_simplex 0 -target_dm_distribute -target_dm_plex_box_faces 4,4 -target_dm_plex_hash_location -target_dm_refine 1
    test:
      suffix: nested_ref_quad_deg0_unity_serial
    test:
      suffix: nested_ref_quad_deg0_unity_parallel
      nsize: 2

  # Nested quadrilateral meshes, non-matching distribution
  testset:
    args: -source_dm_plex_simplex 0 -source_dm_distribute -source_dm_plex_box_faces 4,4 -source_dm_plex_hash_location \
          -target_dm_plex_simplex 0 -target_dm_distribute -target_dm_plex_box_faces 8,8 -target_dm_plex_hash_location
    test:
      suffix: nested_quad_deg0_unity_serial
    test:
      suffix: nested_quad_deg0_unity_parallel
      nsize: 2

  # Non-nested quadrilateral meshes, non-matching distribution
  testset:
    args: -source_dm_plex_simplex 0 -source_dm_distribute -source_dm_plex_box_faces 4,4 -source_dm_plex_hash_location \
          -target_dm_plex_simplex 0 -target_dm_distribute -target_dm_plex_box_faces 6,6 -target_dm_plex_hash_location
    test:
      suffix: nonnested_quad_deg0_unity_serial
    test:
      suffix: nonnested_quad_deg0_unity_parallel
      nsize: 2

  # Nested hexahedral meshes, matching distribution
  testset:
    args: -source_dm_plex_simplex 0 -source_dm_plex_dim 3 -source_dm_distribute -source_dm_plex_box_faces 4,4,4 -source_dm_plex_hash_location \
          -target_dm_plex_simplex 0 -target_dm_plex_dim 3 -target_dm_distribute -target_dm_plex_box_faces 4,4,4 -target_dm_plex_hash_location -target_dm_refine 1
    test:
      suffix: nested_ref_hex_deg0_unity_serial
    test:
      suffix: nested_ref_hex_deg0_unity_parallel
      nsize: 2

  # Nested hexahedral meshes, non-matching distribution
  testset:
    args: -source_dm_plex_simplex 0 -source_dm_plex_dim 3 -source_dm_distribute -source_dm_plex_box_faces 4,4,4 -source_dm_plex_hash_location \
          -target_dm_plex_simplex 0 -target_dm_plex_dim 3 -target_dm_distribute -target_dm_plex_box_faces 8,8,8 -target_dm_plex_hash_location
    test:
      suffix: nested_hex_deg0_unity_serial
    test:
      suffix: nested_hex_deg0_unity_parallel
      nsize: 2

  # Non-nested hexahedral meshes, non-matching distribution
  testset:
    args: -source_dm_plex_simplex 0 -source_dm_plex_dim 3 -source_dm_distribute -source_dm_plex_box_faces 4,4,4 -source_dm_plex_hash_location \
          -target_dm_plex_simplex 0 -target_dm_plex_dim 3 -target_dm_distribute -target_dm_plex_box_faces 6,6,6 -target_dm_plex_hash_location
    test:
      suffix: nonnested_hex_deg0_unity_serial
    test:
      suffix: nonnested_hex_deg0_unity_parallel
      nsize: 2

TEST*/
