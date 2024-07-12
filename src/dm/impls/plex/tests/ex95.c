static char help[] = "Test PetscViewer_ExodusII\n\n";

#include <petsc.h>
#include <exodusII.h>

int main(int argc, char **argv)
{
  DM          dm;
  char        ifilename[PETSC_MAX_PATH_LEN], ofilename[PETSC_MAX_PATH_LEN];
  int         numZVars, numNVars;
  int         nNodalVar = 4;
  int         nZonalVar = 3;
  int         order     = 1;
  PetscViewer viewer;
  int         exoid           = -1;
  int         index           = -1;
  const char *nodalVarName[4] = {"U_x", "U_y", "Alpha", "Beta"};
  const char *zonalVarName[3] = {"Sigma_11", "Sigma_12", "Sigma_22"};
  const char *testNames[3]    = {"U", "Sigma", "Gamma"};
  char       *name=NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PetscViewer_ExodusII test", "ex96");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex96", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscCall(PetscOptionsString("-o", "Filename to write", "ex96", ofilename, ofilename, sizeof(ofilename), NULL));
  PetscOptionsEnd();

  PetscCallExternal(ex_opts, EX_VERBOSE + EX_DEBUG);

  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);
  exoid         = ex_open_par("test2.exo", EX_READ, &CPU_word_size, &IO_word_size, &EXO_version, PETSC_COMM_WORLD, MPI_INFO_NULL);
  int ierr      = ex_close(exoid);
  printf("ierr: %d", ierr);

  //   PetscCheck(exo->exoid >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open_par failed for %s", exo->filename);

  //   PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, "test2.exo", FILE_MODE_READ, &viewer));
  //   PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
  //   PetscPrintf(PETSC_COMM_WORLD,"exoid: %d\n",exoid);
  //   PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));
  //   PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscFinalize());
  return 0;
}
