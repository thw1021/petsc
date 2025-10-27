static char help[] = "Tests N-M-L checkpointing.\n\n";

#include <petscdmshell.h>
#include <petscdmplex.h>
#include <petscsection.h>
#include <petscsf.h>
#include <petsclayouthdf5.h>

/* A six-element mesh

Test N-M-L checkpointing with (N, M, L) = (1, 2, 3):

On 1 process:
  create and save plexA, secA, and vecA.

On 2 processes:
  load plexA and secA,
  create vecA1 and save with DMPlexGlobalVectorView(), and
  create vecA2 and save with DMPlexLocalVectorView().

On 3 processes:
  load plexA, secA, vecA, vecA1, and vecA2.

Global point numbers are created on 1-process plex,
and preserved in the save-load cycles.

            13--26--14--27--15--28--16--29--17--30--18--31--19
             |       |       |       |       |       |       |
            32   0  33   1  34   2  35   3  36   4  37   5  38
             |       |       |       |       |       |       |
             6--20---7--21---8--22---9--23--10--24--11--25--12

For sections:
  use includesConstraints = TRUE for local section (default),
  use includesConstraints = FALSE for global section (default).

=========
1 process
=========

plexA:

            13--26--14--27--15--28--16--29--17--30--18--31--19
             |       |       |       |       |       |       |
  rank 0:   32   0  33   1  34   2  35   3  36   4  37   5  38
             |       |       |       |       |       |       |
             6--20---7--21---8--22---9--23--10--24--11--25--12

secA Dofs:                                                     constrained
                                                              /
             1---1---1---1---1---1---1---1---1---1---1---1---1
             |       |       |       |       |       |       |
             0   0   0   0   0   0   0   0   0   0   0   0   0
             |       |       |       |       |       |       |
             0---0---0---0---0---0---0---0---0---0---0---0---0

secA Offsets:

             0---7---1---8---2---9---3--10---4--11---5--12---6
             |       |       |       |       |       |       |
            13   0  13   0  13   0  13   0  13   0  13   0  13
             |       |       |       |       |       |       |
             0---7---0---7---0---7---0---7---0---7---0---7---0
                                                               constrained
vecA:                                                         /
             0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.
             *---*---*---*---*---*---*---*---*---*---*---*---*
             |       |       |       |       |       |       |
             |       |       |       |       |       |       |
             |       |       |       |       |       |       |
             +-------+-------+-------+-------+-------+-------+

vecA (global):

  rank 0: [0., 2., 4., 6., 8., 10., 1., 3., 5., 7., 9., 11.]

===========
2 processes
===========

plexA:

             7---17--8---18--9--19--(12)(24)(13)
             |       |       |       |       |
  rank 0:   20   0  21   1  22   2  (25) (3)(26)
             |       |       |       |       |
             4---14--5---15--6--16--(10)(23)(11)

                           (13)(25)--8--17---9--18--10--19--11
                             |       |       |       |       |
  rank 1:                  (26) (3) 20   0   21  1  22   2  23
                             |       |       |       |       |
                           (12)(24)--4--14---5--15---6--16---7

vecA1:                                                         constrained
                                                              /
    100. +   0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.
             *---*---*---*---*---*---*---*---*---*---*---*---*
             |       |       |       |       |       |       |
             |       |       |       |       |       |       |
             |       |       |       |       |       |       |
             +-------+-------+-------+-------+-------+-------+

vecA1 (global):

  rank 0: [100., 102., 104., 101., 103., 105.]
  rank 1: [106., 108., 110., 107., 109., 111.]

vecA2:                                                         constrained
                                                              /
    200. +   0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.
             *---*---*---*---*---*---*---*---*---*---*---*---*
             |       |       |       |       |       |       |
             |       |       |       |       |       |       |
             |       |       |       |       |       |       |
             +-------+-------+-------+-------+-------+-------+

vecA2 (local):

  rank 0: [200., 202., 204., 206., 208., 201., 203., 205., 207.]
  rank 1: [206., 208., 210., 212., 204., 207., 209., 211., 205.]

===========
3 Processes
===========

plexA:

             5--13---6--14--(9)(18)(10)
             |       |       |       |
  rank 0:   15   0  16   1 (19) (2)(20)
             |       |       |       |
             3--11---4--12--(7)(17)-(8)

                    (9)(21)--6--16---7--17-(12)(24)(13)
                     |       |       |       |       |
  rank 1:          (22) (2) 18   0  19   1 (25) (3)(26)
                     |       |       |       |       |
                    (8)(20)--4--14---5--15-(10)(23)(11)

                                   (10)(19)--6--13---7--14---8
                                     |       |       |       |
  rank 2:                          (20) (2) 15   0  16   1  17
                                     |       |       |       |
                                    (9)(18)--3--11---4--12---5

vecA (global):

  rank 0: [0., 2., 1., 3.]
  rank 1: [4., 6., 5., 7.]
  rank 2: [8., 10., 9. 11.]

vecA1 (global):

  vecA + 100.

vecA2 (global):

  vecA + 200.

*/

typedef struct {
  char      fname[PETSC_MAX_PATH_LEN]; /* Output mesh filename */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool flg;

  PetscFunctionBegin;
  options->fname[0] = '\0';
  PetscOptionsBegin(comm, "", "N-M-L checkpointing Test Options", "DMPLEX");
  PetscCall(PetscOptionsString("-fname", "The output mesh file", "ex58.c", options->fname, options->fname, sizeof(options->fname), &flg));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LoadPlexA(PetscViewer viewer, DM *plexA)
{
  PetscViewerFormat format = PETSC_VIEWER_HDF5_PETSC;

  PetscFunctionBegin;
  PetscCall(DMCreate(PetscObjectComm((PetscObject)viewer), plexA));
  PetscCall(DMSetType(*plexA, DMPLEX));
  PetscCall(PetscObjectSetName((PetscObject)(*plexA), "plexA"));
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(DMPlexTopologyLoad(*plexA, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  {
    DM               pdm;
    PetscInt         overlap = 1;
    PetscPartitioner part;

    PetscCall(DMPlexGetPartitioner(*plexA, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(*plexA, overlap, NULL, &pdm));
    if (pdm) {
      PetscCall(DMDestroy(plexA));
      *plexA = pdm;
    }
    PetscCall(PetscObjectSetName((PetscObject)(*plexA), "plexA"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LoadSecA(PetscViewer viewer, DM plexA, DM *dmA, PetscSF* gsf, PetscSF *lsf)
{
  PetscSection secA;

  PetscFunctionBegin;
  PetscCall(DMClone(plexA, dmA));
  PetscCall(PetscObjectSetName((PetscObject)(*dmA), "dmA"));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)(*dmA)), &secA));
  PetscCall(DMSetLocalSection(*dmA, secA));
  PetscCall(PetscSectionDestroy(&secA));
  PetscCall(DMPlexSectionLoad(plexA, viewer, *dmA, gsf, lsf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm          comm;
  PetscMPIInt       size, rank, mycolor;
  PetscViewerFormat format = PETSC_VIEWER_HDF5_PETSC;
  AppCtx            user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size >= 3, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example only works with three or more processes");
  /* nprocs = 1 */
  mycolor = (PetscMPIInt)(rank >= 1);
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    DM          plexA;
    PetscViewer viewer;

    PetscCall(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_WRITE, &viewer));
    /* Create and save plexA */
    {
      const PetscInt faces[2] = {6, 1};

      PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_TRUE, &plexA));
      PetscCall(PetscObjectSetName((PetscObject)plexA, "plexA"));
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(DMPlexTopologyView(plexA, viewer));
      PetscCall(PetscViewerPopFormat(viewer));
    }
    /* Create and save secA and vecA */
    {
      DM           dmA;
      PetscSection secA, gsecA;
      PetscBool    includesConstraints = PETSC_FALSE;
      Vec          vecA;
      PetscScalar *array = NULL;
      PetscInt     pStart = -1, pEnd = -1, p, i;

      /* Create and save secA */
      PetscCall(PetscSectionCreate(comm, &secA));
      PetscCall(PetscObjectSetName((PetscObject)secA, "secA"));
      PetscCall(DMPlexGetChart(plexA, &pStart, &pEnd));
      PetscCall(PetscSectionSetChart(secA, pStart, pEnd));
      for (p = 13; p < 20; ++p) PetscCall(PetscSectionSetDof(secA, p, 1));
      for (p = 26; p < 32; ++p) PetscCall(PetscSectionSetDof(secA, p, 1));
      PetscCall(PetscSectionSetConstraintDof(secA, 19, 1));
      PetscCall(PetscSectionSetUp(secA));
      {
        const PetscInt indices[]  = {0};

        PetscCall(PetscSectionSetConstraintIndices(secA, 19, indices));
      }
      PetscCall(DMClone(plexA, &dmA));
      PetscCall(PetscObjectSetName((PetscObject)dmA, "dmA"));
      PetscCall(DMSetLocalSection(dmA, secA));
      PetscCall(PetscSectionDestroy(&secA));
      PetscCall(DMPlexSectionView(plexA, viewer, dmA));
      /* Create and save vecA */
      PetscCall(DMGetGlobalSection(dmA, &gsecA));
      PetscCall(PetscSectionGetIncludesConstraints(gsecA, &includesConstraints));
      PetscCheck(!includesConstraints, PETSC_COMM_SELF, PETSC_ERR_LIB, "Not expected");
      PetscCall(DMGetGlobalVector(dmA, &vecA));
      PetscCall(PetscObjectSetName((PetscObject)vecA, "vecA"));
      PetscCall(VecGetArrayWrite(vecA, &array));
      {
        const PetscScalar array_[]  = {0., 2., 4., 6., 8., 10., 1., 3., 5., 7., 9., 11.};

        for (i = 0; i < 12; ++i) array[i] = array_[i];
      }
      PetscCall(VecRestoreArrayWrite(vecA, &array));
      PetscCall(DMPlexGlobalVectorView(plexA, viewer, dmA, NULL, vecA));
      PetscCall(DMRestoreGlobalVector(dmA, &vecA));
      PetscCall(DMDestroy(&dmA));
    }
    PetscCall(DMDestroy(&plexA));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCallMPI(MPI_Comm_free(&comm));
  /* nprocs = 2 */
  mycolor = (PetscMPIInt)(rank >= 2);
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    DM          plexA, dmA;
    PetscSF     gsf, lsf;
    PetscViewer viewer;

    PetscCall(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_READ, &viewer));
    PetscCall(LoadPlexA(viewer, &plexA));
    PetscCall(LoadSecA(viewer, plexA, &dmA, &gsf, &lsf));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_APPEND, &viewer));
    /* Create and save vecA1 */
    {
      PetscSection gsecA;
      PetscBool    includesConstraints = PETSC_FALSE;
      Vec          vecA1;
      PetscScalar *array = NULL;
      PetscInt     i;

      PetscCall(DMGetGlobalSection(dmA, &gsecA));
      PetscCall(PetscSectionGetIncludesConstraints(gsecA, &includesConstraints));
      PetscCheck(!includesConstraints, PETSC_COMM_SELF, PETSC_ERR_LIB, "Not expected");
      PetscCall(DMGetGlobalVector(dmA, &vecA1));
      PetscCall(PetscObjectSetName((PetscObject)vecA1, "vecA1"));
      PetscCall(VecGetArrayWrite(vecA1, &array));
      if (rank == 0) {
        const PetscScalar array_[]  = {100., 102., 104., 101., 103., 105.};

        for (i = 0; i < 6; ++i) array[i] = array_[i];
      } else /* rank == 1 */ {
        const PetscScalar array_[]  = {106., 108., 110., 107., 109., 111.};

        for (i = 0; i < 6; ++i) array[i] = array_[i];
      }
      PetscCall(VecRestoreArrayWrite(vecA1, &array));
      PetscCall(DMPlexGlobalVectorView(plexA, viewer, dmA, gsf, vecA1));
      PetscCall(DMRestoreGlobalVector(dmA, &vecA1));
    }
    PetscCall(PetscSFDestroy(&gsf));
    /* Create and save vecA2 */
    {
      PetscSection lsecA;
      PetscBool    includesConstraints = PETSC_FALSE;
      Vec          vecA2;
      PetscScalar *array = NULL;
      PetscInt     i;

      PetscCall(DMGetLocalSection(dmA, &lsecA));
      PetscCall(PetscSectionGetIncludesConstraints(lsecA, &includesConstraints));
      PetscCheck(includesConstraints, PETSC_COMM_SELF, PETSC_ERR_LIB, "Not expected");
      PetscCall(DMGetLocalVector(dmA, &vecA2));
      PetscCall(PetscObjectSetName((PetscObject)vecA2, "vecA2"));
      PetscCall(VecGetArrayWrite(vecA2, &array));
      if (rank == 0) {
        const PetscScalar array_[]  = {200., 202., 204., 206., 208., 201., 203., 205., 207.};

        for (i = 0; i < 9; ++i) array[i] = array_[i];
      } else /* rank == 1 */ {
        const PetscScalar array_[]  = {206., 208., 210., 212., 204., 207., 209., 211., 205.};

        for (i = 0; i < 9; ++i) array[i] = array_[i];
      }
      PetscCall(VecRestoreArrayWrite(vecA2, &array));
      PetscCall(DMPlexLocalVectorView(plexA, viewer, dmA, lsf, vecA2));
      PetscCall(DMRestoreLocalVector(dmA, &vecA2));
    }
    PetscCall(PetscSFDestroy(&lsf));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(DMDestroy(&dmA));
    PetscCall(DMDestroy(&plexA));
  }
  PetscCallMPI(MPI_Comm_free(&comm));
  /* nprocs = 3 */
  mycolor = (PetscMPIInt)(rank >= 3);
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, rank, &comm));
  if (mycolor == 0) {
    DM           plexA, dmA;
    PetscSF      gsf, lsf;
    PetscSection gsecA;
    Vec          vecA, vecA1, vecA2;
    PetscViewer  viewer;

    PetscCall(PetscViewerHDF5Open(comm, user.fname, FILE_MODE_APPEND, &viewer));
    /* Load plexA, secA, vecA, vecA1 */
    PetscCall(LoadPlexA(viewer, &plexA));
    PetscCall(DMViewFromOptions(plexA, NULL, "-dm_view"));
    PetscCall(LoadSecA(viewer, plexA, &dmA, &gsf, &lsf));
    PetscCall(DMGetGlobalSection(dmA, &gsecA));
    PetscCall(PetscSectionView(gsecA, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(DMGetGlobalVector(dmA, &vecA));
    PetscCall(PetscObjectSetName((PetscObject)vecA, "vecA"));
    PetscCall(DMPlexGlobalVectorLoad(plexA, viewer, dmA, gsf, vecA));
    PetscCall(VecView(vecA, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(DMRestoreGlobalVector(dmA, &vecA));
    PetscCall(DMGetGlobalVector(dmA, &vecA1));
    PetscCall(PetscObjectSetName((PetscObject)vecA1, "vecA1"));
    PetscCall(DMPlexGlobalVectorLoad(plexA, viewer, dmA, gsf, vecA1));
    PetscCall(VecView(vecA1, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(DMRestoreGlobalVector(dmA, &vecA1));
    PetscCall(DMGetGlobalVector(dmA, &vecA2));
    PetscCall(PetscObjectSetName((PetscObject)vecA2, "vecA2"));
    PetscCall(DMPlexGlobalVectorLoad(plexA, viewer, dmA, gsf, vecA2));
    PetscCall(VecView(vecA2, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(DMRestoreGlobalVector(dmA, &vecA2));
    PetscCall(PetscSFDestroy(&gsf));
    PetscCall(PetscSFDestroy(&lsf));
    PetscCall(DMDestroy(&dmA));
    PetscCall(DMDestroy(&plexA));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCallMPI(MPI_Comm_free(&comm));
  /* Finalize */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hdf5
  test:
    suffix: 0
    requires: !complex
    nsize: 3
    args: -fname ex58_dump.h5 -dm_plex_view_hdf5_storage_version 3.1.0 -petscpartitioner_type simple -dm_view ascii::ascii_info_detail

TEST*/
