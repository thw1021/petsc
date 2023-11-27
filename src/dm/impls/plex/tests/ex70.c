static char help[] = "Tests for DMPlexMarkBoundaryFaces()\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

typedef struct {
  PetscInt overlap; /* The overlap size used when partitioning */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->overlap = 0;

  PetscOptionsBegin(comm, "", "Options for DMPlexMarkBoundaryFaces() problem", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-overlap", "The overlap size used when partitioning", "ex79.c", options->overlap, &options->overlap, NULL, 0));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  DMLabel     extLabel;
  MPI_Comm    comm;
  PetscMPIInt size;
  AppCtx      user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size != 2) {
    PetscCall(PetscPrintf(comm, "This example is specifically designed for comm size == 2.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCall(ProcessOptions(comm, &user));
  {
    DM               pdm;
    const PetscInt   faces[2] = {2, 2};
    PetscPartitioner part;

    PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm));
    PetscCall(DMPlexGetPartitioner(dm, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMSetAdjacency(dm, -1, PETSC_FALSE, PETSC_TRUE));
    PetscCall(DMPlexDistribute(dm, user.overlap, NULL, &pdm));
    if (pdm) {
      PetscCall(DMDestroy(&dm));
      dm = pdm;
    }
  }
  PetscCall(DMCreateLabel(dm, "exterior_facets"));
  PetscCall(DMGetLabel(dm, "exterior_facets", &extLabel));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, extLabel));
  PetscCall(PetscObjectSetName((PetscObject)dm, "Example_DM"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2
    requires: chaco
    args: -overlap {{0 1}separate output} -petscpartitioner_type chaco -dm_view ascii::ascii_info_detail

TEST*/
