static char help[] = "Load and save the mesh to the native HDF5 format\n\n";

#define EX "ex56.c"

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include <petscsf.h>

typedef struct {
  MPI_Comm  comm;
  PetscBool compare;                      /* Compare the meshes using DMPlexEqual() and DMCompareLabels() */
  PetscBool compare_labels;               /* Compare labels in the meshes using DMCompareLabels() */
  PetscBool distribute;                   /* Distribute the mesh */
  PetscBool interpolate;                  /* Generate intermediate mesh elements */
  char      infile[PETSC_MAX_PATH_LEN];   /* Input file */
  char      outfile[PETSC_MAX_PATH_LEN];  /* Output file */
  char      meshname[PETSC_MAX_PATH_LEN]; /* Mesh name */
  PetscBool use_low_level_functions;      /* Use low level functions for viewing and loading */
  //TODO This is meant as temporary option; can be removed once we have full parallel loading in place
  PetscBool distribute_after_topo_load;   /* Distribute topology right after DMPlexTopologyLoad(), if use_low_level_functions=true */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->comm                       = comm;
  options->compare                    = PETSC_FALSE;
  options->compare_labels             = PETSC_FALSE;
  options->distribute                 = PETSC_TRUE;
  options->interpolate                = PETSC_FALSE;
  options->infile[0]                  = '\0';
  options->outfile[0]                 = '\0';
  options->use_low_level_functions    = PETSC_FALSE;
  options->distribute_after_topo_load = PETSC_FALSE;
  ierr = PetscStrcpy(options->meshname, "mesh");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare", "Compare the meshes using DMPlexEqual() and DMCompareLabels()", EX, options->compare, &options->compare, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare_labels", "Compare labels in the meshes using DMCompareLabels()", "ex55.c", options->compare_labels, &options->compare_labels, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-infile", "Input mesh file", EX, options->infile, options->infile, sizeof(options->infile), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-outfile", "Output mesh file", EX, options->outfile, options->outfile, sizeof(options->outfile), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-meshname", "Name of the mesh", EX, options->meshname, options->meshname, sizeof(options->meshname), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_low_level_functions", "Use low level functions for viewing and loading", EX, options->use_low_level_functions, &options->use_low_level_functions, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute_after_topo_load", "Distribute topology right after DMPlexTopologyLoad(), if use_low_level_functions=true", EX, options->distribute_after_topo_load, &options->distribute_after_topo_load, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

static PetscErrorCode CreateMesh(AppCtx *options, DM *newdm)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateFromFile(options->comm, options->infile, options->meshname, options->interpolate, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  *newdm = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode SaveMesh(AppCtx *options, DM dm)
{
  PetscViewer    v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerHDF5Open(PetscObjectComm((PetscObject) dm), options->outfile, FILE_MODE_WRITE, &v);CHKERRQ(ierr);
  if (options->use_low_level_functions) {
    ierr = DMPlexTopologyView(dm, v);CHKERRQ(ierr);
    ierr = DMPlexCoordinatesView(dm, v);CHKERRQ(ierr);
    ierr = DMPlexLabelsView(dm, v);CHKERRQ(ierr);
  } else {
    ierr = DMView(dm, v);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LoadMesh(AppCtx *options, DM *dmnew)
{
  DM             dm;
  PetscViewer    v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(options->comm, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, options->meshname);CHKERRQ(ierr);

  ierr = PetscViewerHDF5Open(PetscObjectComm((PetscObject) dm), options->outfile, FILE_MODE_READ, &v);CHKERRQ(ierr);
  if (options->use_low_level_functions) {
    PetscSF sfXC;

    ierr = DMPlexTopologyLoad(dm, v, &sfXC);CHKERRQ(ierr);
    if (options->distribute_after_topo_load) {
      DM      dmdist;
      PetscSF sfXB = sfXC, sfBC;

      ierr = DMPlexDistribute(dm, 0, &sfBC, &dmdist);CHKERRQ(ierr);
      if (dmdist) {
        ierr = PetscObjectSetName((PetscObject) dmdist, options->meshname);CHKERRQ(ierr);
        ierr = PetscSFCompose(sfXB, sfBC, &sfXC);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&sfXB);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&sfBC);CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = dmdist;
      }
    }
    ierr = DMPlexCoordinatesLoad(dm, v, sfXC);CHKERRQ(ierr);
    ierr = DMPlexLabelsLoad(dm, v);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfXC);CHKERRQ(ierr);
  } else {
    ierr = DMLoad(dm, v);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);

  ierr = DMSetOptionsPrefix(dm, "load_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  *dmnew = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode CompareMeshes(AppCtx *options, DM dm0, DM dm1)
{
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  if (options->compare) {
    ierr = DMPlexEqual(dm0, dm1, &flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(options->comm, PETSC_ERR_ARG_INCOMP, "DMs are not equal");
    ierr = PetscPrintf(options->comm,"DMs equal\n");CHKERRQ(ierr);
  }
  if (options->compare_labels) {
    ierr = DMCompareLabels(dm0, dm1, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(options->comm,"DMLabels equal\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dmnew;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(&user, &dm);CHKERRQ(ierr);
  ierr = SaveMesh(&user, dm);CHKERRQ(ierr);
  ierr = LoadMesh(&user, &dmnew);CHKERRQ(ierr);
  ierr = CompareMeshes(&user, dm, dmnew);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmnew);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

//TODO we can -compare once the new parallel topology format is in place
/*TEST
  build:
    requires: hdf5

  # load old format, save in new format, reload
  testset:
    suffix: 1
    requires: hdf5 !complex datafilespath
    args: -dm_plex_check_all -dm_plex_view_hdf5_storage_version 2.0.0
    args: -dm_distribute -dm_plex_interpolate
    args: -load_dm_plex_check_all
    args: -use_low_level_functions {{0 1}} -distribute_after_topo_load 1
    args: -outfile ex56_1.h5
    nsize: {{1 3}}
    test:
      suffix: a
      args: -infile ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/annulus-20.h5
    test:
      suffix: b
      TODO: broken
      args: -infile ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/barycentricallyrefinedcube.h5
    test:
      suffix: c
      args: -infile ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/blockcylinder-50.h5
    test:
      suffix: d
      args: -infile ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/cube-hexahedra-refined.h5
    test:
      suffix: e
      args: -infile ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/hybrid_hexwedge.h5
    test:
      suffix: f
      args: -infile ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/square.h5

TEST*/
