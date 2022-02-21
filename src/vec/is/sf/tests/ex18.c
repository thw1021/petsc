
static char help[]= "Test PetscSFFCompose when the ilocal array is not the identity\n\n";

#include <petscsf.h>

static PetscErrorCode PetscSFCheckEqual_Private(PetscSF sf0, PetscSF sf1)
{
  PetscInt          nRoot, nLeave;
  Vec               vecRoot0, vecLeave0, vecRoot1, vecLeave1;
  MPI_Comm          comm;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)sf0, &comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf0, &nRoot, &nLeave, NULL, NULL);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm, nRoot, PETSC_DECIDE, &vecRoot0);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm, nLeave, PETSC_DECIDE, &vecLeave0);CHKERRQ(ierr);
  ierr = VecDuplicate(vecRoot0, &vecRoot1);CHKERRQ(ierr);
  ierr = VecDuplicate(vecLeave0, &vecLeave1);CHKERRQ(ierr);
  {
    PetscRandom       rand;

    ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
    ierr = VecSetRandom(vecRoot0, rand);CHKERRQ(ierr);
    ierr = VecSetRandom(vecLeave0, rand);CHKERRQ(ierr);
    ierr = VecCopy(vecRoot0, vecRoot1);CHKERRQ(ierr);
    ierr = VecCopy(vecLeave0, vecLeave1);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  }

  ierr = VecScatterBegin(sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(  sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(  sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecEqual(vecLeave0, vecLeave1, &flg);CHKERRQ(ierr);
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "leave vectors differ");

  ierr = VecScatterBegin(sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(  sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(  sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecEqual(vecRoot0, vecRoot1, &flg);CHKERRQ(ierr);
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "root vectors differ");

  ierr = VecDestroy(&vecRoot0);CHKERRQ(ierr);
  ierr = VecDestroy(&vecRoot1);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLeave0);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLeave1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt          i, nsfs = 3, nLeavesPerRank = 4;
  PetscBool         shareRoots = PETSC_FALSE;
  //TODO test non-null
  PetscInt         *leafOffsets = NULL;
  PetscSF          *sfs;
  PetscSF           sf, sfRef;
  MPI_Comm          comm;
  PetscMPIInt       rank, size;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);

  ierr = PetscOptionsGetInt(NULL, NULL, "-nsfs", &nsfs, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-n_leaves_per_rank", &nLeavesPerRank, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-share_roots", &shareRoots, NULL);CHKERRQ(ierr);

  ierr = PetscMalloc1(nsfs, &sfs);CHKERRQ(ierr);
  for (i=0; i<nsfs; i++) {
    PetscInt nroots  = shareRoots ? nLeavesPerRank * nsfs : nLeavesPerRank;
    PetscInt nleaves = nLeavesPerRank * size;
    PetscInt j, k;
    PetscMPIInt r;
    PetscSFNode *iremote;

    ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
    for (r=0, j=0; r<size; r++) {
      for (k=0; k<nLeavesPerRank; k++, j++) {
        iremote[j].rank = r;
        iremote[j].index = shareRoots ? k + i * nLeavesPerRank : k;
      }
    }

    ierr = PetscSFCreate(comm, &sfs[i]);CHKERRQ(ierr);
    //TODO test non-NULL ilocal
    ierr = PetscSFSetGraph(sfs[i], nroots, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }

  ierr = PetscSFConcatenate(comm, nsfs, sfs, shareRoots, leafOffsets, &sf);CHKERRQ(ierr);

  {
    PetscInt      j, k, r;
    PetscSFNode  *iremote;
    PetscInt      nleaves = nsfs * nLeavesPerRank * size;
    PetscInt      nroots  = nLeavesPerRank * nsfs;

    ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
    ierr = PetscSFCreate(comm, &sfRef);CHKERRQ(ierr);
    for (i=0, j=0; i<nsfs; i++) {
      for (r=0; r<size; r++) {
        for (k=0; k<nLeavesPerRank; k++, j++) {
          iremote[j].rank = r;
          iremote[j].index = k + i * nLeavesPerRank;
        }
      }
    }
    ierr = PetscSFSetGraph(sfRef, nroots, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }

  ierr = PetscSFCheckEqual_Private(sf, sfRef);CHKERRQ(ierr);

  for (i=0; i<nsfs; i++) {
    ierr = PetscSFDestroy(&sfs[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(sfs);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfRef);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
