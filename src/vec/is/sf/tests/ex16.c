static char help[]= "Test PetscSFBuildWithGlobalIndices\n\n";

#include <petsc.h>
#include <petscsf.h>

/* Test PetscSFBuildWithGlobalIndices.

testnum 0:

  rank             : 0            1            2
  numGlobalIndices : 4            4            4
  bufferSize       : PETSC_DECIDE PETSC_DECIDE PETSC_DECIDE 
  numRootIndices   : 3            1            1
  rootIndices      : [1 0 2]      [3]          [3]
  rootOffset       : 100          200          300
  numLeafIndices   : 1            1            2
  leafIndices      : [0]          [2]          [0 3]
  leafOffset       : 400          500          600
  flag             : PETSC_FALSE  PETSC_FALSE  PETSC_FALSE

would build the following SF:

  [0] 400 <- (0,101)
  [1] 500 <- (0,102)
  [2] 600 <- (0,101)
  [2] 601 <- (2,300)

testnum 1:

  rank             : 0            1            2
  numGlobalIndices : 4            4            4
  bufferSize       : PETSC_DECIDE PETSC_DECIDE PETSC_DECIDE 
  numRootIndices   : 3            1            1
  rootIndices      : [1 0 2]      [3]          [3]
  rootOffset       : 100          200          300
  numLeafIndices   : PETSC_DECIDE PETSC_DECIDE PETSC_DECIDE
  leafIndices      : NULL         NULL         NULL
  leafOffset       : PETSC_DECIDE PETSC_DECIDE PETSC_DECIDE
  flag             : PETSC_TRUE   PETSC_TRUE   PETSC_TRUE

would build the following SF:

  [1] 200 <- (2,300)

testnum 2:

  rank             : 0            1            2
  numGlobalIndices : PETSC_DECIDE PETSC_DECIDE PETSC_DECIDE
  bufferSize       : PETSC_DECIDE PETSC_DECIDE PETSC_DECIDE 
  numRootIndices   : 1            2            1
  rootIndices      : NULL         NULL         NULL
  rootOffset       : 100          200          300
  numLeafIndices   : 1            1            2
  leafIndices      : [0]          [2]          [0 3]
  leafOffset       : 400          500          600
  flag             : PETSC_FALSE  PETSC_FALSE  PETSC_FALSE

would build the following SF:

  [0] 400 <- (0,100)
  [1] 500 <- (1,201)
  [2] 600 <- (0,100)
  [2] 601 <- (2,300)

*/

int main(int argc, char **argv)
{
  PetscSF         sf;
  PetscInt        N, n;
  PetscInt        nA, *A, offsetA;
  PetscInt        nB, *B, offsetB;
  PetscBool       flag;
  PetscMPIInt     size, rank;
  PetscInt        testnum;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-testnum", &testnum, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscSFCreate(PETSC_COMM_WORLD, &sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  switch (testnum) {
  case 0:
    N = 4;
    n = PETSC_DECIDE;
    flag = PETSC_FALSE;
    switch (rank) {
    case 0: nA = 3; offsetA = 100; nB = 1; offsetB = 400; break;
    case 1: nA = 1; offsetA = 200; nB = 1; offsetB = 500; break;
    case 2: nA = 1; offsetA = 300; nB = 2; offsetB = 600; break;
    }
    ierr = PetscMalloc2(nA, &A, nB, &B);CHKERRQ(ierr);
    switch (rank) {
    case 0:
      A[0] = 1; A[1] = 0; A[2] = 2;
      B[0] = 0;
      break;
    case 1:
      A[0] = 3;
      B[0] = 2;
      break;
    case 2:
      A[0] = 3;
      B[0] = 0; B[1] = 3;
      break;
    }
    ierr = PetscSFBuildWithGlobalIndices(sf, N, n, nA, A, NULL, offsetA, nB, B, NULL, offsetB, flag, NULL);CHKERRQ(ierr);
    ierr = PetscFree2(A, B);CHKERRQ(ierr);
    break;
  case 1:
    N = 4;
    n = PETSC_DECIDE;
    flag = PETSC_TRUE;
    switch (rank) {
    case 0: nA = 3; offsetA = 100; break;
    case 1: nA = 1; offsetA = 200; break;
    case 2: nA = 1; offsetA = 300; break;
    }
    ierr = PetscMalloc1(nA, &A);CHKERRQ(ierr);
    switch (rank) {
    case 0:
      A[0] = 1; A[1] = 0; A[2] = 2;
      break;
    case 1:
      A[0] = 3;
      break;
    case 2:
      A[0] = 3;
      break;
    }
    nB = PETSC_DECIDE;
    offsetB = PETSC_DECIDE;
    B = NULL;
    ierr = PetscSFBuildWithGlobalIndices(sf, N, n, nA, A, NULL, offsetA, nB, B, NULL, offsetB, flag, NULL);CHKERRQ(ierr);
    ierr = PetscFree(A);CHKERRQ(ierr);
    break;
  case 2:
    N = PETSC_DECIDE; /* N = 4 */
    n = PETSC_DECIDE;
    flag = PETSC_FALSE;
    switch (rank) {
    case 0: nA = 1; offsetA = 100; nB = 1; offsetB = 400; break;
    case 1: nA = 2; offsetA = 200; nB = 1; offsetB = 500; break;
    case 2: nA = 1; offsetA = 300; nB = 2; offsetB = 600; break;
    }
    A = NULL;
    ierr = PetscMalloc1(nB, &B);CHKERRQ(ierr);
    switch (rank) {
    case 0:
      B[0] = 0;
      break;
    case 1:
      B[0] = 2;
      break;
    case 2:
      B[0] = 0; B[1] = 3;
      break;
    }
    ierr = PetscSFBuildWithGlobalIndices(sf, N, n, nA, A, NULL, offsetA, nB, B, NULL, offsetB, flag, NULL);CHKERRQ(ierr);
    ierr = PetscFree(B);CHKERRQ(ierr);
    break;
  }
  ierr = PetscObjectSetName((PetscObject)sf, "sf");CHKERRQ(ierr);
  ierr = PetscSFView(sf, NULL);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    nsize: 3
    args: -testnum 0

  test:
    suffix: 1
    nsize: 3
    args: -testnum 1

  test:
    suffix: 2
    nsize: 3
    args: -testnum 2

TEST*/
