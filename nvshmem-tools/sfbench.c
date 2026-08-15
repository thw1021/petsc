static const char help[] = "PetscSF ring halo-exchange benchmark on GPU memory.\n\
\n\
Each rank owns n roots; its n leaves reference the roots owned by the previous rank, so\n\
every byte crosses a GPU boundary. Sweeps the message size and times PetscSFBcast() (or\n\
PetscSFReduce() with -reduce), reporting microseconds per call and effective bandwidth.\n\
\n\
Intended use is comparing PETSc's NVSHMEM PetscSF backend against the ordinary GPU-aware\n\
MPI path:\n\
  mpirun -n 4 --bind-to none ./sfbench -use_nvshmem 0                 # GPU-aware MPI\n\
  mpirun -n 4 --bind-to none ./sfbench -use_nvshmem 1                 # NVSHMEM, put protocol\n\
  mpirun -n 4 --bind-to none ./sfbench -use_nvshmem 1 -use_nvshmem_get 1  # NVSHMEM, get protocol\n\
\n\
Options:\n\
  -nmin  <int>   first message size, in PetscScalars (default 8)\n\
  -nmax  <int>   last message size, in PetscScalars (default 4194304)\n\
  -iters <int>   upper bound on timed iterations per size (default 500)\n\
  -reduce        time PetscSFReduce() instead of PetscSFBcast()\n\n";

/*
  Two hard-won cautions, both of which silently produce garbage numbers:

  1. ALWAYS give each rank more than one core, e.g. `mpirun --bind-to none` or
     `--map-by ppr:1:numa`. NVSHMEM runs a proxy thread that spins; under Open MPI's
     default one-core-per-rank binding it competes with the rank itself and NVSHMEM
     appears ~10x slower than it is. NVSHMEM warns about this at startup:
       "Proxy thread shares a core with the main PE, performance may be impacted"
     The damage shows up in the MPI *reductions* (VecMDot()/VecNorm()), not in the
     halo exchange, which makes it easy to misattribute.

  2. Feed PetscSF *PETSc-managed* device memory (as here, via VecCreateSeqCUDA() and
     VecGetArrayAndMemType()). Handing raw cudaMalloc() pointers to
     PetscSFBcastWithMemTypeBegin() segfaults inside UCX on the GPU-aware MPI path with
       UCX ERROR cuMemGetAddressRange(0x...) error: named symbol not found
     even though bare MPI accepts the same pointers.

  For the NVSHMEM path to engage at all, PetscSFLinkNvshmemCheck() requires the SF to be
  PETSCSFBASIC on a communicator congruent with PETSC_COMM_WORLD, with both root and leaf
  data in device memory. All of that holds below. Confirm engagement by running with
  NVSHMEM_VERSION=1, which prints a banner only when NVSHMEM actually initializes.
*/

#include <petscsf.h>
#include <petscvec.h>
#include <petscdevice_cuda.h>

int main(int argc, char **argv)
{
  PetscSF        sf;
  PetscSFNode   *iremote;
  Vec            rootvec, leafvec;
  PetscScalar   *rootdata, *leafdata;
  PetscMemType   rootmtype, leafmtype;
  PetscMPIInt    rank, size, src;
  PetscInt       n, nmin = 8, nmax = 4194304, iters = 500, warmup = 20, it;
  PetscLogDouble t0, t1, dt, dtmax;
  PetscBool      doreduce = PETSC_FALSE;
  size_t         bytes;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nmin", &nmin, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nmax", &nmax, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iters", &iters, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reduce", &doreduce, NULL));
  PetscCheck(size > 1, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Need at least 2 MPI ranks");

  src = (PetscMPIInt)((rank - 1 + size) % size);
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "# PetscSF%s, %d ranks, ring topology, device memory\n", doreduce ? "Reduce" : "Bcast", size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "# %12s %12s %14s %14s\n", "count", "bytes", "usec/iter", "GB/s"));
  }

  for (n = nmin; n <= nmax; n *= 2) {
    bytes = (size_t)n * sizeof(PetscScalar);
    /* Shrink the iteration count as messages grow so each size costs about the same wall time. */
    it = PetscMax(20, PetscMin(iters, (PetscInt)(4194304 / n)));

    PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
    PetscCall(PetscSFSetType(sf, PETSCSFBASIC));
    PetscCall(PetscMalloc1(n, &iremote));
    for (PetscInt i = 0; i < n; i++) {
      iremote[i].rank  = src;
      iremote[i].index = i;
    }
    /* ilocal = NULL makes the leaves contiguous 0..n-1, which keeps leafcontig true and
       lets PetscSF use its direct (unpacked) path where the backend supports it. */
    PetscCall(PetscSFSetGraph(sf, n, n, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    PetscCall(PetscSFSetUp(sf));

    PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, n, &rootvec));
    PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, n, &leafvec));
    PetscCall(VecSet(rootvec, (PetscScalar)rank));
    PetscCall(VecSet(leafvec, 0.0));
    PetscCall(VecGetArrayAndMemType(rootvec, &rootdata, &rootmtype));
    PetscCall(VecGetArrayAndMemType(leafvec, &leafdata, &leafmtype));
    PetscCheck(PetscMemTypeDevice(rootmtype) && PetscMemTypeDevice(leafmtype), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected device memory; NVSHMEM would silently fall back to MPI");

    for (PetscInt i = 0; i < warmup + it; i++) {
      if (i == warmup) { // stop warming up: drain the device, line the ranks up, start the clock
        PetscCallCUDA(cudaDeviceSynchronize());
        PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
        PetscCall(PetscTime(&t0));
      }
      if (doreduce) {
        PetscCall(PetscSFReduceWithMemTypeBegin(sf, MPIU_SCALAR, leafmtype, leafdata, rootmtype, rootdata, MPI_REPLACE));
        PetscCall(PetscSFReduceEnd(sf, MPIU_SCALAR, leafdata, rootdata, MPI_REPLACE));
      } else {
        PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR, rootmtype, rootdata, leafmtype, leafdata, MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, rootdata, leafdata, MPI_REPLACE));
      }
    }
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCall(PetscTime(&t1));

    /* Report the slowest rank: the exchange is only as fast as its laggard. */
    dt = (t1 - t0) / it;
    PetscCallMPI(MPI_Reduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
    if (rank == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %12" PetscInt_FMT " %12zu %14.3f %14.3f\n", n, bytes, dtmax * 1e6, bytes / dtmax / 1e9));

    PetscCall(VecRestoreArrayAndMemType(rootvec, &rootdata));
    PetscCall(VecRestoreArrayAndMemType(leafvec, &leafdata));
    PetscCall(VecDestroy(&rootvec));
    PetscCall(VecDestroy(&leafvec));
    PetscCall(PetscSFDestroy(&sf));
  }
  PetscCall(PetscFinalize());
  return 0;
}
