#include "exodusII.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Info mpi_info = MPI_INFO_NULL;

  float  version;

  int CPU_word_size = 0; /* sizeof(float) */
  int IO_word_size  = 0; /* use what is stored in file */
  int exoid,error;

  ex_opts(EX_VERBOSE | EX_ABORT);

  /* Initialize MPI. */
  MPI_Init(&argc, &argv);



  exoid = ex_open("test.exo",                /* filename path */
                          EX_READ,           /* access mode = READ */
                          &CPU_word_size, /* CPU word size */
                          &IO_word_size,    /* IO word size */
                          &version);      /* ExodusII library version */

  error = ex_close(exoid);
  printf("\nafter ex_close, error = %3d\n", error);

  /* open EXODUS II files */
  exoid = ex_open_par("test.exo",        /* filename path */
                          EX_READ,           /* access mode = READ */
                          &CPU_word_size, /* CPU word size */
                          &IO_word_size,    /* IO word size */
                          &version,       /* ExodusII library version */
                          mpi_comm, mpi_info);
  error = ex_close(exoid);
  printf("\nafter ex_close, error = %3d\n", error);

  MPI_Finalize();
  return 0;
}
