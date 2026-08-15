/* Timed GPU-aware MPI ping-pong on cudaMalloc buffers, sweeping message size.
   Run with 2 ranks; use --map-by ppr:1:node to place them on different nodes. */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int     rank, size, ndev, lr = 0;
  const char *e;
  double *buf;
  size_t  nmax = 1 << 22; /* 4M doubles = 32 MB */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  e = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (e) lr = atoi(e);
  cudaGetDeviceCount(&ndev);
  cudaSetDevice(lr % ndev);
  if (cudaMalloc((void **)&buf, nmax * sizeof(double)) != cudaSuccess) {
    printf("cudaMalloc failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  cudaMemset(buf, 0, nmax * sizeof(double));
  cudaDeviceSynchronize();
  if (rank == 0) {
    char host[64];
    gethostname(host, sizeof(host));
    printf("rank 0 on %s, %d ranks\n", host, size);
  }
  for (size_t n = 1; n <= nmax; n *= 8) {
    int    reps = n * sizeof(double) > (1 << 20) ? 50 : 200, warm = 10;
    double t0 = 0, t1 = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warm + reps; i++) {
      if (i == warm) t0 = MPI_Wtime();
      if (rank == 0) {
        MPI_Send(buf, (int)n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buf, (int)n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else if (rank == 1) {
        MPI_Recv(buf, (int)n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buf, (int)n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }
    }
    t1 = MPI_Wtime();
    if (rank == 0) {
      double us = (t1 - t0) / reps * 1e6; /* round trip */
      double bytes = n * sizeof(double);
      printf("%10zu B  rtt %9.2f us  one-way %8.2f us  bw %8.2f GB/s\n", (size_t)bytes, us, us / 2, bytes / (us / 2 * 1e-6) / 1e9);
    }
  }
  cudaFree(buf);
  MPI_Finalize();
  return 0;
}
