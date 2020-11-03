static char help[] = "Test that MatPartitioning and PetscPartitioner interfaces are equivalent when using PETSCPARTITIONERMATPARTITIONING\n\n";
static char FILENAME[] = "ex24.c";

#include <petscmatpartitioning.h>
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_PTSCOTCH)
EXTERN_C_BEGIN
#include <ptscotch.h>
EXTERN_C_END
#endif

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscInt  faces[3];                     /* Number of faces per dimension */
  PetscBool simplex;                      /* Use simplices or hexes */
  PetscBool interpolate;                  /* Interpolate mesh */
  PetscBool compare_is;                   /* Compare ISs and PetscSections */
  PetscBool compare_dm;                   /* Compare DM */
  PetscBool tpw;                          /* Use target partition weights */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  char      partitioning[64];
  char      repartitioning[64];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->compare_is   = PETSC_FALSE;
  options->compare_dm   = PETSC_FALSE;
  options->dim          = 3;
  options->simplex      = PETSC_TRUE;
  options->interpolate  = PETSC_FALSE;
  options->filename[0]  = '\0';
  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare_is", "Compare ISs and PetscSections?", FILENAME, options->compare_is, &options->compare_is, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare_dm", "Compare DMs?", FILENAME, options->compare_dm, &options->compare_dm, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", FILENAME, options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise hexes", FILENAME, options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", FILENAME, options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", FILENAME, options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  options->faces[0] = 1; options->faces[1] = 1; options->faces[2] = 1;
  dim = options->dim;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", FILENAME, options->faces, &dim, NULL);CHKERRQ(ierr);
  if (dim) options->dim = dim;
  ierr = PetscStrncpy(options->partitioning,MATPARTITIONINGPARMETIS,sizeof(options->partitioning));CHKERRQ(ierr);
  ierr = PetscOptionsString("-partitioning","The mat partitioning type to test","None",options->partitioning, options->partitioning,sizeof(options->partitioning),NULL);CHKERRQ(ierr);
  options->repartitioning[0] = '\0';
  ierr = PetscOptionsString("-repartitioning","The mat partitioning type to test (second partitioning)","None", options->repartitioning, options->repartitioning,sizeof(options->repartitioning),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tpweight", "Use target partition weights", FILENAME, options->tpw, &options->tpw, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode ScotchResetRandomSeed()
{
#if defined(PETSC_HAVE_PTSCOTCH)
  SCOTCH_randomReset();
#endif
  PetscFunctionReturn(0);
}


static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim          = user->dim;
  PetscInt      *faces        = user->faces;
  PetscBool      simplex      = user->simplex;
  PetscBool      interpolate  = user->interpolate;
  const char    *filename     = user->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm1, dm2, dmdist1, dmdist2;
  MatPartitioning mp;
  PetscPartitioner part1, part2;
  AppCtx         user;
  IS             is1=NULL, is2=NULL;
  IS             is1g, is2g;
  PetscSection   s1=NULL, s2=NULL, tpws = NULL;
  PetscInt       i;
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm1);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm2);CHKERRQ(ierr);

  {
    DMPlexInterpolatedFlag interpolated;
    ierr = DMPlexIsDistributed(dm1, &flg);CHKERRQ(ierr);
    ierr = DMPlexIsInterpolated(dm1, &interpolated);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "initial dm1\n  -distributed? %s\n  -interpolated? %s\n", PetscBools[flg], DMPlexInterpolatedFlags[interpolated]);CHKERRQ(ierr);
    ierr = DMPlexIsDistributed(dm2, &flg);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "initial dm2\n  -distributed? %s\n  -interpolated? %s\n", PetscBools[flg], DMPlexInterpolatedFlags[interpolated]);CHKERRQ(ierr);
  }

  if (user.tpw) {
    ierr = PetscSectionCreate(comm, &tpws);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(tpws, 0, size);CHKERRQ(ierr);
    for (i=0;i<size;i++) {
      PetscInt tdof = i%2 ? 2*i -1 : i+2;
      ierr = PetscSectionSetDof(tpws, i, tdof);CHKERRQ(ierr);
    }
    if (size > 1) { /* test zero tpw entry */
      ierr = PetscSectionSetDof(tpws, 0, 0);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(tpws);CHKERRQ(ierr);
  }

  /* partition dm1 using PETSCPARTITIONERPARMETIS */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm1, &part1);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part1,"p1_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part1, user.partitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part1);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s1);CHKERRQ(ierr);
  ierr = PetscPartitionerDMPlexPartition(part1, dm1, tpws, s1, &is1);CHKERRQ(ierr);

  /* partition dm2 using PETSCPARTITIONERMATPARTITIONING with MATPARTITIONINGPARMETIS */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm2, &part2);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part2,"p2_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING);CHKERRQ(ierr);
  ierr = PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(mp, user.partitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part2);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s2);CHKERRQ(ierr);
  ierr = PetscPartitionerDMPlexPartition(part2, dm2, tpws, s2, &is2);CHKERRQ(ierr);

  ierr = ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g);CHKERRQ(ierr);
  ierr = ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g);CHKERRQ(ierr);
  ierr = ISViewFromOptions(is1g, NULL, "-seq_is1_view");CHKERRQ(ierr);
  ierr = ISViewFromOptions(is2g, NULL, "-seq_is2_view");CHKERRQ(ierr);
  /* compare the two ISs */
  if (user.compare_is) {
    ierr = ISEqualUnsorted(is1g, is2g, &flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscPrintf(comm, "ISs are not equal\n");CHKERRQ(ierr);}
    else      {ierr = PetscPrintf(comm, "ISs are equal\n");CHKERRQ(ierr);}
  }
  ierr = ISDestroy(&is1g);CHKERRQ(ierr);
  ierr = ISDestroy(&is2g);CHKERRQ(ierr);

  /* compare the two PetscSections */
  ierr = PetscSectionViewFromOptions(s1, NULL, "-seq_s1_view");CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(s2, NULL, "-seq_s2_view");CHKERRQ(ierr);
  if (user.compare_is) {
    ierr = PetscSectionCompare(s1, s2, &flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscPrintf(comm, "PetscSections are not equal\n");CHKERRQ(ierr);}
    else      {ierr = PetscPrintf(comm, "PetscSections are equal\n");CHKERRQ(ierr);}
  }

  /* distribute both DMs */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm1, 0, NULL, &dmdist1);CHKERRQ(ierr);
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm2, 0, NULL, &dmdist2);CHKERRQ(ierr);

  /* cleanup */
  ierr = PetscSectionDestroy(&tpws);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s1);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s2);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = DMDestroy(&dm1);CHKERRQ(ierr);
  ierr = DMDestroy(&dm2);CHKERRQ(ierr);

  /* if distributed DMs are NULL (sequential case), then assume equality and quit */
  if (!dmdist1 || !dmdist2) {
    ierr = DMDestroy(&dmdist1);CHKERRQ(ierr);
    ierr = DMDestroy(&dmdist2);CHKERRQ(ierr);
    if (user.compare_dm) {
      ierr = PetscPrintf(comm, "Distributed DMs are equal\n");CHKERRQ(ierr);
    }
    if (user.repartitioning[0]) {
      if (user.compare_is) {
        ierr = PetscPrintf(comm, "Distributed ISs are equal.\n");CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "Distributed PetscSections are equal\n");CHKERRQ(ierr);
      }
      if (user.compare_dm) {
        ierr = PetscPrintf(comm, "Redistributed DMs are equal\n");CHKERRQ(ierr);
      }
    }
    goto cleanup;
  }

  ierr = DMViewFromOptions(dmdist1, NULL, "-dm_dist1_view");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmdist2, NULL, "-dm_dist2_view");CHKERRQ(ierr);

  /* compare the two distributed DMs */
  if (user.compare_dm) {
    ierr = DMPlexEqual(dmdist1, dmdist2, &flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscPrintf(comm, "Distributed DMs are not equal\n");CHKERRQ(ierr);}
    else      {ierr = PetscPrintf(comm, "Distributed DMs are equal\n");CHKERRQ(ierr);}
  }

  /* if repartitioning is disabled, then quit */
  if (!user.repartitioning[0]) goto cleanup;

  if (user.tpw) {
    ierr = PetscSectionCreate(comm, &tpws);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(tpws, 0, size);CHKERRQ(ierr);
    for (i=0;i<size;i++) {
      PetscInt tdof = i%2 ? i+1 : size - i;
      ierr = PetscSectionSetDof(tpws, i, tdof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(tpws);CHKERRQ(ierr);
  }

  /* repartition distributed DM dmdist1 */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dmdist1, &part1);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part1,"dp1_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part1, user.repartitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part1);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s1);CHKERRQ(ierr);
  ierr = PetscPartitionerDMPlexPartition(part1, dmdist1, tpws, s1, &is1);CHKERRQ(ierr);

  /* repartition distributed DM dmdist2 */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dmdist2, &part2);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part2,"dp2_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part2, PETSCPARTITIONERMATPARTITIONING);CHKERRQ(ierr);
  ierr = PetscPartitionerMatPartitioningGetMatPartitioning(part2, &mp);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(mp, user.repartitioning);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part2);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s2);CHKERRQ(ierr);
  ierr = PetscPartitionerDMPlexPartition(part2, dmdist2, tpws, s2, &is2);CHKERRQ(ierr);

  /* compare the two ISs */
  ierr = ISOnComm(is1, comm, PETSC_USE_POINTER, &is1g);CHKERRQ(ierr);
  ierr = ISOnComm(is2, comm, PETSC_USE_POINTER, &is2g);CHKERRQ(ierr);
  ierr = ISViewFromOptions(is1g, NULL, "-dist_is1_view");CHKERRQ(ierr);
  ierr = ISViewFromOptions(is2g, NULL, "-dist_is2_view");CHKERRQ(ierr);
  if (user.compare_is) {
    ierr = ISEqualUnsorted(is1g, is2g, &flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscPrintf(comm, "Distributed ISs are not equal.\n");CHKERRQ(ierr);}
    else      {ierr = PetscPrintf(comm, "Distributed ISs are equal.\n");CHKERRQ(ierr);}
  }
  ierr = ISDestroy(&is1g);CHKERRQ(ierr);
  ierr = ISDestroy(&is2g);CHKERRQ(ierr);

  /* compare the two PetscSections */
  ierr = PetscSectionViewFromOptions(s1, NULL, "-dist_s1_view");CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(s2, NULL, "-dist_s2_view");CHKERRQ(ierr);
  if (user.compare_is) {
    ierr = PetscSectionCompare(s1, s2, &flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscPrintf(comm, "Distributed PetscSections are not equal\n");CHKERRQ(ierr);}
    else      {ierr = PetscPrintf(comm, "Distributed PetscSections are equal\n");CHKERRQ(ierr);}
  }

  /* redistribute both distributed DMs */
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexDistribute(dmdist1, 0, NULL, &dm1);CHKERRQ(ierr);
  ierr = ScotchResetRandomSeed();CHKERRQ(ierr);
  ierr = DMPlexDistribute(dmdist2, 0, NULL, &dm2);CHKERRQ(ierr);

  /* compare the two distributed DMs */
  if (user.compare_dm) {
    ierr = DMPlexEqual(dm1, dm2, &flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscPrintf(comm, "Redistributed DMs are not equal\n");CHKERRQ(ierr);}
    else      {ierr = PetscPrintf(comm, "Redistributed DMs are equal\n");CHKERRQ(ierr);}
  }

cleanup:
  /* cleanup */
  ierr = PetscSectionDestroy(&tpws);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s1);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s2);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = DMDestroy(&dm1);CHKERRQ(ierr);
  ierr = DMDestroy(&dm2);CHKERRQ(ierr);
  ierr = DMDestroy(&dmdist1);CHKERRQ(ierr);
  ierr = DMDestroy(&dmdist2);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    # partition sequential mesh loaded from Exodus file
    nsize: {{1 2 3 4 8}}
    requires: exodusii
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    test:
      suffix: 0_chaco
      requires: chaco
      args: -interpolate {{0 1}separate output}
      args: -partitioning chaco -tpweight 0 -compare_is -compare_dm
    test:
      suffix: 0_parmetis
      requires: parmetis
      args: -interpolate {{0 1}separate output}
      args: -partitioning parmetis -tpweight {{0 1}} -compare_is -compare_dm
    test:
      suffix: 0_ptscotch
      requires: ptscotch
      args: -interpolate {{0 1}separate output}
      args: -partitioning ptscotch -tpweight {{0 1}}
  testset:
    # partition sequential mesh loaded from Exodus file using simple partitioner and repartition
    nsize: {{1 2 3 4 8}}
    requires: exodusii
    args: -filename ${PETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -interpolate 1
    args: -p1_petscpartitioner_type simple
    args: -p2_petscpartitioner_type simple
    test:
      suffix: 1_parmetis
      requires: parmetis
      args: -repartitioning parmetis -tpweight {{0 1}} -compare_is -compare_dm
    test:
      suffix: 1_ptscotch
      requires: ptscotch
      args: -repartitioning ptscotch -tpweight {{0 1}}
  test:
    # partition mesh generated by ctetgen using scotch, then repartition with scotch, diff view
    suffix: 3
    nsize: 4
    requires: ptscotch ctetgen
    args: -faces 2,3,2 -partitioning ptscotch -repartitioning ptscotch -interpolate
    args: -p1_petscpartitioner_view -p2_petscpartitioner_view -dp1_petscpartitioner_view -dp2_petscpartitioner_view -tpweight {{0 1}}
  testset:
    # partition mesh generated by ctetgen using partitioners supported both by MatPartitioning and PetscPartitioner
    #nsize: {{1 2 3 4 8}}
    nsize: 2
    requires: ctetgen
    args: -simplex {{0 1}}
    args: -faces {{2,3,4  7,11,5}}
    test:
      suffix: 4_chaco
      requires: chaco
      args: -interpolate {{0 1}separate output}
      args: -partitioning chaco -tpweight 0 -compare_is -compare_dm
    test:
      suffix: 4_parmetis
      requires: parmetis
      args: -interpolate {{0 1}separate output}
      args: -partitioning parmetis -tpweight {{0 1}} -compare_is -compare_dm
    test:
      suffix: 4_ptscotch
      requires: ptscotch
      args: -interpolate {{0 1}separate output}
      args: -partitioning ptscotch -tpweight {{0 1}}

TEST*/

