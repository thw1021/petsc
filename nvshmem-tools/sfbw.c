static const char help[] = "Large-message bandwidth of PetscSF vs the bare transports, on device memory, under log stages.\n\
\n\
Every rank sends n PetscScalars to the next rank of a ring (or to its partner with -pair) and\n\
receives as many, all in device memory. The same buffers are exchanged by one of three arms:\n\
  -arm sf     PetscSFBcast() (PETSCSFBASIC; -use_nccl 0/1 selects MPI or NCCL underneath)\n\
  -arm mpi    bare MPI_Irecv()/MPI_Isend() on the device pointers (GPU-aware MPI, no PetscSF)\n\
  -arm nccl   bare ncclGroupStart()/ncclSend()/ncclRecv()/ncclGroupEnd() on PETSc's stream\n\
Each message size runs in its own PetscLogStage so that -log_view reports it without the\n\
setup (NCCL communicator creation, first-touch connections, buffer allocation), which sits in\n\
the Setup stage. The wall clock per stage is also printed (max over ranks).\n\
\n\
  -nmin, -nmax <scalars>   size sweep (default 131072 .. 8388608 = 1 MB .. 64 MB)\n\
  -iters <it>              timed iterations per size (default 50)\n\
  -pair                    exchange with rank^1 instead of the ring\n\
  -uni                     unidirectional: even ranks only send, odd ranks only receive\n\
  -nccl_side               nccl arm: fork/join to a high-priority side stream as PetscSF does\n\n";

#include <petscsf.h>
#include <petscvec.h>
#include <petscdevice_cuda.h>
#include <nccl.h>

#define CallNCCL(...) \
  do { \
    ncclResult_t r_ = __VA_ARGS__; \
    PetscCheck(r_ == ncclSuccess, PETSC_COMM_SELF, PETSC_ERR_LIB, "NCCL error: %s", ncclGetErrorString(r_)); \
  } while (0)

typedef enum {
  ARM_SF,
  ARM_MPI,
  ARM_NCCL
} Arm;

int main(int argc, char **argv)
{
  Vec                rootvec, leafvec;
  PetscScalar       *rootdata, *leafdata;
  PetscMPIInt        rank, size, src, dst;
  PetscInt           nmin = 131072, nmax = 8388608, iters = 50, warmup = 5;
  PetscBool          pair = PETSC_FALSE, uni = PETSC_FALSE, side = PETSC_FALSE, dosend, dorecv;
  char               armname[16] = "sf";
  Arm                arm = ARM_SF;
  PetscLogStage      stage;
  PetscLogDouble     t0, t1, dt, dtmax;
  PetscDeviceContext dctx;
  cudaStream_t      *stream, sstream = NULL;
  cudaEvent_t        e0 = NULL, e1 = NULL;
  ncclComm_t         ncomm = NULL;
  char               stagename[64];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nmin", &nmin, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nmax", &nmax, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iters", &iters, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-pair", &pair, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-uni", &uni, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nccl_side", &side, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-arm", armname, sizeof(armname), NULL));
  if (armname[0] == 'm') arm = ARM_MPI;
  else if (armname[0] == 'n') arm = ARM_NCCL;
  PetscCheck(size > 1 && (!pair || size % 2 == 0), PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Need >= 2 ranks (an even number with -pair)");

  if (pair) src = dst = (PetscMPIInt)(rank ^ 1);
  else {
    src = (PetscMPIInt)((rank - 1 + size) % size);
    dst = (PetscMPIInt)((rank + 1) % size);
  }
  dosend = (!uni || rank % 2 == 0) ? PETSC_TRUE : PETSC_FALSE;
  dorecv = (!uni || rank % 2 == 1) ? PETSC_TRUE : PETSC_FALSE;
  PetscCheck(!uni || pair, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "-uni needs -pair");

  PetscCall(PetscLogStageRegister("Setup", &stage));
  PetscCall(PetscLogStagePush(stage));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, nmax, &rootvec));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, nmax, &leafvec));
  PetscCall(VecSet(rootvec, (PetscScalar)(rank + 1)));
  PetscCall(VecSet(leafvec, 0.0));
  PetscCall(VecGetArrayAndMemType(rootvec, &rootdata, NULL));
  PetscCall(VecGetArrayAndMemType(leafvec, &leafdata, NULL));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, (void **)&stream));
  if (arm == ARM_NCCL) {
    ncclUniqueId id;
    int          prio;

    if (rank == 0) CallNCCL(ncclGetUniqueId(&id));
    PetscCallMPI(MPI_Bcast(&id, (PetscMPIInt)sizeof(id), MPI_BYTE, 0, PETSC_COMM_WORLD));
    CallNCCL(ncclCommInitRank(&ncomm, size, id, rank));
    PetscCallCUDA(cudaDeviceGetStreamPriorityRange(NULL, &prio));
    PetscCallCUDA(cudaStreamCreateWithPriority(&sstream, cudaStreamNonBlocking, prio));
    PetscCallCUDA(cudaEventCreateWithFlags(&e0, cudaEventDisableTiming));
    PetscCallCUDA(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
  }
  PetscCall(PetscLogStagePop());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "# arm=%s %s%s %d ranks, %s; usec per exchange (max over ranks), GB/s = bytes sent per rank / time\n", armname, pair ? "pair" : "ring", uni ? " unidirectional" : "", size, arm == ARM_SF ? "PetscSF" : arm == ARM_MPI ? "bare MPI" : side ? "bare NCCL, side stream" : "bare NCCL, PETSc stream"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "# %12s %12s %12s %10s\n", "count", "bytes", "usec", "GB/s"));

  for (PetscInt n = nmin; n <= nmax; n *= 2) {
    PetscSF      sf = NULL;
    size_t       bytes = (size_t)n * sizeof(PetscScalar);
    cudaStream_t s     = (arm == ARM_NCCL && side) ? sstream : *stream;
    PetscReal    norm;

    PetscCall(PetscSNPrintf(stagename, sizeof(stagename), "n=%" PetscInt_FMT, n));
    PetscCall(PetscLogStageRegister(stagename, &stage));
    if (arm == ARM_SF) { /* the SF is built outside the timed stage, as the NCCL communicator was */
      PetscSFNode *iremote;

      PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
      PetscCall(PetscSFSetType(sf, PETSCSFBASIC));
      PetscCall(PetscMalloc1(dorecv ? n : 0, &iremote));
      for (PetscInt i = 0; dorecv && i < n; i++) {
        iremote[i].rank  = src;
        iremote[i].index = i;
      }
      PetscCall(PetscSFSetGraph(sf, dosend ? n : 0, dorecv ? n : 0, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
      PetscCall(PetscSFSetUp(sf));
    }
    for (PetscInt i = 0; i < warmup + iters; i++) {
      if (i == warmup) { /* the warm-up (first-touch connections, link creation) is not timed */
        PetscCallCUDA(cudaDeviceSynchronize());
        PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
        PetscCall(PetscLogStagePush(stage));
        PetscCall(PetscTime(&t0));
      }
      switch (arm) {
      case ARM_SF:
        PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR, PETSC_MEMTYPE_CUDA, rootdata, PETSC_MEMTYPE_CUDA, leafdata, MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, rootdata, leafdata, MPI_REPLACE));
        break;
      case ARM_MPI: {
        MPI_Request req[2];
        PetscMPIInt nreq = 0;

        PetscCallCUDA(cudaStreamSynchronize(*stream)); /* as PetscSF does before handing device buffers to MPI */
        if (dorecv) PetscCallMPI(MPI_Irecv(leafdata, (PetscMPIInt)n, MPIU_SCALAR, src, 0, PETSC_COMM_WORLD, &req[nreq++]));
        if (dosend) PetscCallMPI(MPI_Isend(rootdata, (PetscMPIInt)n, MPIU_SCALAR, dst, 0, PETSC_COMM_WORLD, &req[nreq++]));
        PetscCallMPI(MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE));
      } break;
      case ARM_NCCL:
        if (side) {
          PetscCallCUDA(cudaEventRecord(e0, *stream));
          PetscCallCUDA(cudaStreamWaitEvent(sstream, e0, 0));
        }
        CallNCCL(ncclGroupStart());
        if (dorecv) CallNCCL(ncclRecv(leafdata, bytes, ncclInt8, src, ncomm, s));
        if (dosend) CallNCCL(ncclSend(rootdata, bytes, ncclInt8, dst, ncomm, s));
        CallNCCL(ncclGroupEnd());
        if (side) {
          PetscCallCUDA(cudaEventRecord(e1, sstream));
          PetscCallCUDA(cudaStreamWaitEvent(*stream, e1, 0));
        }
        break;
      }
    }
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCall(PetscTime(&t1));
    PetscCall(PetscLogStagePop());
    dt = (t1 - t0) / iters;
    PetscCallMPI(MPI_Reduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %12" PetscInt_FMT " %12zu %12.2f %10.1f\n", n, bytes, dtmax * 1e6, dosend ? bytes / dtmax / 1e9 : 0.0));
    if (dorecv) { /* check the received data: leaves must hold src+1 */
      PetscCall(VecRestoreArrayAndMemType(leafvec, &leafdata));
      PetscCall(VecNorm(leafvec, NORM_INFINITY, &norm));
      PetscCheck(norm == (PetscReal)(src + 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "rank %d: wrong leaf data (max %g, expected %d)", rank, (double)norm, src + 1);
      PetscCall(VecGetArrayAndMemType(leafvec, &leafdata, NULL));
    }
    PetscCall(PetscSFDestroy(&sf));
  }

  if (ncomm) {
    CallNCCL(ncclCommDestroy(ncomm));
    PetscCallCUDA(cudaStreamDestroy(sstream));
    PetscCallCUDA(cudaEventDestroy(e0));
    PetscCallCUDA(cudaEventDestroy(e1));
  }
  PetscCall(VecRestoreArrayAndMemType(rootvec, &rootdata));
  PetscCall(VecRestoreArrayAndMemType(leafvec, &leafdata));
  PetscCall(VecDestroy(&rootvec));
  PetscCall(VecDestroy(&leafvec));
  PetscCall(PetscFinalize());
  return 0;
}
