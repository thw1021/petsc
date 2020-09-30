static char help[] = "Test Index Map\n";

#include <petscim.h>

int main(int argc, char **argv)
{
  IM             m, m2;
  MPI_Comm       comm;
  PetscInt       i, istart = 0, n = 10, keyStart = 0, keyEnd = 10;
  PetscInt       *dkeys;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (rank) {n = 5, istart = 5;}

  ierr = IMCreate(comm, &m);CHKERRQ(ierr);
  ierr = IMSetType(m, IMBASIC);CHKERRQ(ierr);
  ierr = IMSetKeysContiguous(m, keyStart, keyEnd);CHKERRQ(ierr);
  ierr = IMSetUp(m);CHKERRQ(ierr);
  ierr = IMView(m, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = IMDestroy(&m);CHKERRQ(ierr);

  ierr = IMCreate(comm, &m2);CHKERRQ(ierr);
  ierr = IMSetType(m2, IMBASIC);CHKERRQ(ierr);
  ierr = PetscCalloc1(n, &dkeys);CHKERRQ(ierr);
  for (i = istart; i < istart+n; ++i) dkeys[i-istart] = i;
  ierr = IMSetKeysDiscontiguous(m2, n, dkeys, PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = IMSetUp(m2);CHKERRQ(ierr);
  ierr = IMView(m2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFree(dkeys);CHKERRQ(ierr);
  ierr = IMDestroy(&m2);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
