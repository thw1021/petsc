static char help[] = "Test Index Map\n";

#include <petscim.h>

int main(int argc, char **argv)
{
  IM             m, m2;
  PetscInt       *arr;
  PetscInt       i, n = 10;
  PetscMPIInt    rank, size;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (rank) n = rank*n;

  ierr = PetscMalloc1(n, &arr);CHKERRQ(ierr);
  for (i = 0; i <  n; ++i) arr[i] = rank;
  ierr = IMCreate(comm, &m);CHKERRQ(ierr);
  ierr = IMSetType(m, IMBASIC);CHKERRQ(ierr);
  ierr = IMSetKeyArray(m, n, arr, PETSC_TRUE, PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = IMSetUp(m);CHKERRQ(ierr);
  ierr = IMView(m, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = IMDestroy(&m);CHKERRQ(ierr);

  ierr = IMBasicCreateFromSizes(comm, IM_CONTIGUOUS, n, PETSC_DECIDE, &m2);CHKERRQ(ierr);
  ierr = IMView(m2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = IMDestroy(&m2);CHKERRQ(ierr);

  ierr = IMBasicCreateFromSizes(comm, IM_ARRAY, n, PETSC_DECIDE, &m2);CHKERRQ(ierr);
  ierr = IMView(m2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = IMDestroy(&m2);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
