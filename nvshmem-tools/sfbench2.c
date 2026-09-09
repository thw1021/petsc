static const char help[] = "PetscSF: K independent exchanges in flight, with optional overlapped compute.\n\
\n\
Models the difference between one tightly-coupled global solve (K=1, no compute) and a\n\
decoupled scheme that has several independent exchanges outstanding and real work to hide\n\
them behind. MPI must host-synchronize once per BcastBegin; NVSHMEM never blocks the host,\n\
so if that matters it should show up as K grows or as compute is added.\n\
\n\
  -n <count>     message size per exchange, in PetscScalars (default 4096)\n\
  -nexch <K>     number of independent concurrent exchanges (default 1)\n\
  -naxpy <m>     VecAXPYs on a work vector between Begin and End, as overlappable compute\n\
  -iters <it>    timed iterations (default 200)\n\
  -w <len>       work vector length in PetscScalars (default 1048576; use >= 16M for GPU-bound compute)\n\n";

#include <petscsf.h>
#include <petscvec.h>
#include <petscdevice_cuda.h>

#define MAXK 32

int main(int argc, char **argv)
{
  PetscSF        sf;
  PetscSFNode   *iremote;
  Vec            rootvec[MAXK], leafvec[MAXK], wa, wb;
  PetscScalar   *rootdata[MAXK], *leafdata[MAXK];
  PetscMemType   rmt, lmt;
  PetscMPIInt    rank, size, src;
  PetscInt       n = 4096, K = 1, naxpy = 0, iters = 200, warmup = 20, W = 1 << 20;
  PetscLogDouble t0, t1, dt, dtmax, tc0, tc1, tcmax = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nexch", &K, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-naxpy", &naxpy, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iters", &iters, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-w", &W, NULL));
  PetscCheck(size > 1, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Need >= 2 ranks");
  PetscCheck(K >= 1 && K <= MAXK, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "-nexch must be in [1,%d]", MAXK);

  src = (PetscMPIInt)((rank - 1 + size) % size);
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
  PetscCall(PetscSFSetType(sf, PETSCSFBASIC));
  PetscCall(PetscMalloc1(n, &iremote));
  for (PetscInt i = 0; i < n; i++) {
    iremote[i].rank  = src;
    iremote[i].index = i;
  }
  PetscCall(PetscSFSetGraph(sf, n, n, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sf));

  /* K independent buffer pairs => K independent SF links live at once */
  for (PetscInt k = 0; k < K; k++) {
    PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, n, &rootvec[k]));
    PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, n, &leafvec[k]));
    PetscCall(VecSet(rootvec[k], (PetscScalar)(rank + k)));
    PetscCall(VecSet(leafvec[k], 0.0));
    PetscCall(VecGetArrayAndMemType(rootvec[k], &rootdata[k], &rmt));
    PetscCall(VecGetArrayAndMemType(leafvec[k], &leafdata[k], &lmt));
    PetscCheck(PetscMemTypeDevice(rmt) && PetscMemTypeDevice(lmt), PETSC_COMM_SELF, PETSC_ERR_PLIB, "expected device memory");
  }
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, W, &wa));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, W, &wb));
  PetscCall(VecSet(wa, 1.0));
  PetscCall(VecSet(wb, 2.0));

  for (PetscInt i = 0; i < warmup + iters; i++) {
    if (i == warmup) {
      PetscCallCUDA(cudaDeviceSynchronize());
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&t0));
    }
    for (PetscInt k = 0; k < K; k++) PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR, rmt, rootdata[k], lmt, leafdata[k], MPI_REPLACE));
    for (PetscInt j = 0; j < naxpy; j++) PetscCall(VecAXPY(wb, 1.0e-8, wa));
    for (PetscInt k = 0; k < K; k++) PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, rootdata[k], leafdata[k], MPI_REPLACE));
  }
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscCall(PetscTime(&t1));

  /* compute-only cost, for reference */
  if (naxpy) {
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCall(PetscTime(&tc0));
    for (PetscInt i = 0; i < iters; i++) {
      for (PetscInt j = 0; j < naxpy; j++) PetscCall(VecAXPY(wb, 1.0e-8, wa));
    }
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCall(PetscTime(&tc1));
    dt = (tc1 - tc0) / iters;
    PetscCallMPI(MPI_Reduce(&dt, &tcmax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
  }

  dt = (t1 - t0) / iters;
  PetscCallMPI(MPI_Reduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
  if (rank == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  n=%-7" PetscInt_FMT " K=%-3" PetscInt_FMT " naxpy=%-5" PetscInt_FMT " total=%9.2f us  per_exchange=%8.2f us  compute_alone=%8.2f us\n", n, K, naxpy, dtmax * 1e6, dtmax * 1e6 / K, tcmax * 1e6));

  for (PetscInt k = 0; k < K; k++) {
    PetscCall(VecRestoreArrayAndMemType(rootvec[k], &rootdata[k]));
    PetscCall(VecRestoreArrayAndMemType(leafvec[k], &leafdata[k]));
    PetscCall(VecDestroy(&rootvec[k]));
    PetscCall(VecDestroy(&leafvec[k]));
  }
  PetscCall(VecDestroy(&wa));
  PetscCall(VecDestroy(&wb));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFinalize());
  return 0;
}
