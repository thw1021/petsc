#include "petscdmplex.h"
#include "petscsys.h"
static char help[] = "Test PetscViewer_ExodusII\n\n";

#include <petsc.h>
#include <exodusII.h>

int main(int argc, char **argv)
{
  int         exoid;
  PetscViewer viewer;
  int         CPU_word_size, IO_word_size;
  float       EXO_version;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCallExternal(ex_opts,EX_VERBOSE + EX_DEBUG);

  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);
  exoid    = ex_open_par("test2.exo", EX_READ, &CPU_word_size, &IO_word_size, &EXO_version, PETSC_COMM_WORLD, MPI_INFO_NULL);
  int ierr = ex_close(exoid);
  printf("ierr: %d",ierr);

//   PetscCheck(exo->exoid >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open_par failed for %s", exo->filename);




//   PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, "test2.exo", FILE_MODE_READ, &viewer));
//   PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
//   PetscPrintf(PETSC_COMM_WORLD,"exoid: %d\n",exoid);
//   PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));
//   PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscFinalize());
  return 0;
}