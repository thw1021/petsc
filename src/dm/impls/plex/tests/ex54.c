static char help[] = "Test DMPlex distribute/overlap global point numbers.\n\n";

#include <petscdmplex.h>

/* A six-element mesh

Serial mesh:

Create and cache globalPointNumbers with DMPlexGetPointNumbering(dm, &iset):

            13--26--14--27--15--28--16--29--17--30--18--31--19
             |       |       |       |       |       |       |
            32   0  33   1  34   2  35   3  36   4  37   5  38
             |       |       |       |       |       |       |
             6--20---7--21---8--22---9--23--10--24--11--25--12

Parallel mesh:

Check that globalPointNumbers are preserved in dist + overlap:

            13--26--14--27--15--28-(16)(29)(17)
             |       |       |       |       |
  rank 0:   32   0  33   1  34   2 (35) (3)(36)
             |       |       |       |       |
             6--20---7--21---8--22--(9)(23)(10)

                           (15)(28)-16--29--17--30--18--31--19
                             |       |       |       |       |
  rank 1:                  (34) (2) 35   3  36   4  37   5  38
                             |       |       |       |       |
                            (8)(22)--9--23--10--24--11--25--12

Check that globalPointNumbers are reset with DMPlexSetPointNumbering(dm, NULL):

             9--23--10--24--11--25-(16)(32)(17)
             |       |       |       |       |
  rank 0:   26   0  27   1  28   2 (35) (3)(36)
             |       |       |       |       |
             6--20---7--21---8--22-(12)(29)(13)

                           (11)(25)-16--32--17--33--18--34--19
                             |       |       |       |       |
  rank 1:                  (28) (2) 35   3  36   4  37   5  38
                             |       |       |       |       |
                            (8)(22)-12--29--13--30--14--31--15

*/

int main(int argc, char **argv)
{
  DM               dm, pdm, odm;
  const PetscInt   faces[2] = {6, 1};
  PetscPartitioner part;
  PetscInt         overlap = 1;
  IS               iset, iset_parallel;
  PetscMPIInt      size, rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example only works with size == 2");
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_TRUE, &dm));
  PetscCall(DMPlexGetPartitioner(dm, &part));
  PetscCall(PetscPartitionerSetFromOptions(part));
  /* Create and cache global point numbers. */
  PetscCall(DMPlexGetPointNumbering(dm, &iset));
  PetscCall(PetscObjectSetName((PetscObject)dm, "DMSerial"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(ISOnComm(iset, PETSC_COMM_WORLD, PETSC_USE_POINTER, &iset_parallel));
  PetscCall(PetscObjectSetName((PetscObject)iset_parallel, "ISSerial"));
  PetscCall(ISView(iset_parallel, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&iset_parallel));
  /* Distribute global point numbers. */
  PetscCall(DMPlexDistribute(dm, 0, NULL, &pdm));
  if (pdm) {
    PetscCall(DMDestroy(&dm));
    dm = pdm;
  }
  /* Distribute global point numbers. */
  PetscCall(DMPlexDistributeOverlap(dm, overlap, NULL, &odm));
  if (odm) {
    PetscCall(DMDestroy(&dm));
    dm = odm;
  }
  PetscCall(DMPlexGetPointNumbering(dm, &iset));
  PetscCall(PetscObjectSetName((PetscObject)dm, "DMParallel"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(ISOnComm(iset, PETSC_COMM_WORLD, PETSC_USE_POINTER, &iset_parallel));
  PetscCall(PetscObjectSetName((PetscObject)iset_parallel, "ISParallel"));
  PetscCall(ISView(iset_parallel, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&iset_parallel));
  /* Reset global point numbers. */
  PetscCall(DMPlexSetPointNumbering(dm, NULL));
  PetscCall(DMPlexGetPointNumbering(dm, &iset));
  PetscCall(ISOnComm(iset, PETSC_COMM_WORLD, PETSC_USE_POINTER, &iset_parallel));
  PetscCall(PetscObjectSetName((PetscObject)iset_parallel, "ISParallelReset"));
  PetscCall(ISView(iset_parallel, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&iset_parallel));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: 2
    args: -petscpartitioner_type simple -dm_view ascii::ascii_info_detail

TEST*/
