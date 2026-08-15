static const char help[] = "ACGN splitting iteration, NCCL edges, executed via CUDA-graph replay (route 2a).\n\
\n\
Same DAG and emulated compute as acgnbench-nccl.c, but after an eager warmup the whole\n\
iteration body (VecAXPYs + fused NCCL groups) is stream-captured into one CUDA graph and\n\
the timed loop is graph replays: one cudaGraphLaunch per k iterations.\n\
\n\
  -d <len>        state vector length in PetscScalars (default 65536 = 512 KB)\n\
  -gaxpy <G>      AXPYs per shard gradient evaluation (default 8)\n\
  -paxpy <P>      AXPYs per TV prox evaluation (default 4)\n\
  -allreduce      use a global gradient ncclAllReduce instead of branch routing\n\
  -graph_iters k  iterations captured per graph (default 1)\n\
  -iters <it>     timed iterations, rounded down to a multiple of k (default 300)\n\n";

#include <petscsf.h>
#include <petscvec.h>
#include <petscdevice_cuda.h>
#include <nccl.h>

#if defined(PETSC_USE_COMPLEX) || defined(PETSC_USE_REAL_SINGLE)
  #error "This benchmark assumes PetscScalar == double (maps to ncclDouble)"
#endif

#define PetscCallNCCL(...) \
  do { \
    ncclResult_t nccl_res_ = __VA_ARGS__; \
    PetscCheck(nccl_res_ == ncclSuccess, PETSC_COMM_SELF, PETSC_ERR_LIB, "NCCL error %d: %s", (int)nccl_res_, ncclGetErrorString(nccl_res_)); \
  } while (0)

typedef struct {
  Vec          wa, wb;
  PetscScalar *x1d, *x1ld, *gd, *gl2d, *gl3d, *x2d, *x2ld, *x3d, *x3ld;
  PetscInt     d, G, P;
  PetscMPIInt  rank;
  PetscBool    allreduce;
  ncclComm_t   comm;
  cudaStream_t stream;
} BenchCtx;

static PetscErrorCode AxpyN(Vec y, Vec x, PetscInt k)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < k; i++) PetscCall(VecAXPY(y, 1.0e-8, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One ACGN iteration: identical dataflow to acgnbench-nccl.c, single stream */
static PetscErrorCode IterBody(BenchCtx *c)
{
  const PetscInt    d = c->d;
  const PetscMPIInt rank = c->rank;

  PetscFunctionBeginUser;
  if (rank == 1) PetscCall(AxpyN(c->wb, c->wa, 1));
  PetscCallNCCL(ncclGroupStart());
  if (rank == 1) {
    PetscCallNCCL(ncclSend(c->x1d, d, ncclDouble, 0, c->comm, c->stream));
    PetscCallNCCL(ncclSend(c->x1d, d, ncclDouble, 2, c->comm, c->stream));
    PetscCallNCCL(ncclSend(c->x1d, d, ncclDouble, 3, c->comm, c->stream));
  } else PetscCallNCCL(ncclRecv(c->x1ld, d, ncclDouble, 1, c->comm, c->stream));
  PetscCallNCCL(ncclGroupEnd());
  if (rank <= 2) PetscCall(AxpyN(c->wb, c->wa, c->G));
  if (c->allreduce) {
    PetscCallNCCL(ncclAllReduce(c->gd, c->gd, d, ncclDouble, ncclSum, c->comm, c->stream));
  } else {
    PetscCallNCCL(ncclGroupStart());
    if (rank == 0 || rank == 1) {
      PetscCallNCCL(ncclSend(c->gd, d, ncclDouble, 2, c->comm, c->stream));
      PetscCallNCCL(ncclSend(c->gd, d, ncclDouble, 3, c->comm, c->stream));
    }
    if (rank == 2) {
      PetscCallNCCL(ncclRecv(c->gl2d, d, ncclDouble, 0, c->comm, c->stream));
      PetscCallNCCL(ncclRecv(c->gl2d + d, d, ncclDouble, 1, c->comm, c->stream));
      PetscCallNCCL(ncclSend(c->gd, d, ncclDouble, 3, c->comm, c->stream));
    }
    if (rank == 3) {
      PetscCallNCCL(ncclRecv(c->gl3d, d, ncclDouble, 0, c->comm, c->stream));
      PetscCallNCCL(ncclRecv(c->gl3d + d, d, ncclDouble, 1, c->comm, c->stream));
      PetscCallNCCL(ncclRecv(c->gl3d + 2 * d, d, ncclDouble, 2, c->comm, c->stream));
    }
    PetscCallNCCL(ncclGroupEnd());
  }
  if (rank == 2) PetscCall(AxpyN(c->wb, c->wa, 2 + c->P));
  PetscCallNCCL(ncclGroupStart());
  if (rank == 2) {
    PetscCallNCCL(ncclSend(c->x2d, d, ncclDouble, 1, c->comm, c->stream));
    PetscCallNCCL(ncclSend(c->x2d, d, ncclDouble, 3, c->comm, c->stream));
  } else if (rank == 1 || rank == 3) PetscCallNCCL(ncclRecv(c->x2ld, d, ncclDouble, 2, c->comm, c->stream));
  PetscCallNCCL(ncclGroupEnd());
  if (rank == 3) PetscCall(AxpyN(c->wb, c->wa, c->G));
  if (rank == 3) PetscCall(AxpyN(c->wb, c->wa, 3 + c->P));
  PetscCallNCCL(ncclGroupStart());
  if (rank == 3) {
    PetscCallNCCL(ncclSend(c->x3d, d, ncclDouble, 1, c->comm, c->stream));
    PetscCallNCCL(ncclSend(c->x3d, d, ncclDouble, 2, c->comm, c->stream));
  } else if (rank == 1 || rank == 2) PetscCallNCCL(ncclRecv(c->x3ld, d, ncclDouble, 3, c->comm, c->stream));
  PetscCallNCCL(ncclGroupEnd());
  if (rank >= 1) PetscCall(AxpyN(c->wb, c->wa, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSum(Vec v, PetscReal expect, PetscInt d, const char name[], PetscInt *fails)
{
  PetscScalar s;

  PetscFunctionBeginUser;
  PetscCall(VecSum(v, &s));
  if (PetscAbsScalar(s - expect * (PetscReal)d) > 1e-6 * (1.0 + PetscAbsReal(expect * (PetscReal)d))) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "VALIDATION FAIL: %s sum=%g expected=%g\n", name, (double)PetscRealPart(s), (double)(expect * (PetscReal)d)));
    (*fails)++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec                x1v, x1lv, gv, gl2v, gl3v, x2v, x2lv, x3v, x3lv, wa, wb;
  PetscMemType       mt;
  PetscMPIInt        size;
  PetscInt           iters = 300, warmup = 30, k = 1, nlaunch, fails = 0, allfails = 0;
  PetscLogDouble     t0, t1, dt, dtmax;
  PetscDeviceContext dctx;
  void              *sh;
  ncclUniqueId       uid;
  int                ncclver = 0;
  cudaGraph_t        graph;
  cudaGraphExec_t    gexec;
  BenchCtx           c = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 65536, 8, 4, 0, PETSC_FALSE, NULL, NULL};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &c.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 4, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "This benchmark models the 4-node poster problem; run with exactly 4 ranks");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-d", &c.d, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-gaxpy", &c.G, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-paxpy", &c.P, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iters", &iters, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-graph_iters", &k, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-allreduce", &c.allreduce, NULL));
  PetscCheck(k >= 1 && k <= iters, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "-graph_iters must be in [1, iters]");

  /* PETSc's default stream type is PETSC_STREAM_DEFAULT = the legacy NULL stream, which
     cudaStreamBeginCapture() rejects. Retype the global context to a real nonblocking
     stream and re-set it as current, which also re-syncs PetscDefaultCudaStream. */
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextSetStreamType(dctx, PETSC_STREAM_NONBLOCKING));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscCall(PetscDeviceContextSetCurrentContext(dctx));

  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &x1v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &x1lv));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &gv));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, 2 * c.d, &gl2v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, 3 * c.d, &gl3v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &x2v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &x2lv));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &x3v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &x3lv));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &wa));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, c.d, &wb));
  c.wa = wa;
  c.wb = wb;
  PetscCall(VecSet(x1v, 1.0));
  PetscCall(VecSet(x1lv, 0.0));
  PetscCall(VecSet(gv, (PetscScalar)c.rank));
  PetscCall(VecSet(gl2v, 0.0));
  PetscCall(VecSet(gl3v, 0.0));
  PetscCall(VecSet(x2v, 1.0));
  PetscCall(VecSet(x2lv, 0.0));
  PetscCall(VecSet(x3v, 1.0));
  PetscCall(VecSet(x3lv, 0.0));
  PetscCall(VecSet(wa, 1.0));
  PetscCall(VecSet(wb, 2.0));
  PetscCall(VecGetArrayAndMemType(x1v, &c.x1d, &mt));
  PetscCall(VecGetArrayAndMemType(x1lv, &c.x1ld, NULL));
  PetscCall(VecGetArrayAndMemType(gv, &c.gd, NULL));
  PetscCall(VecGetArrayAndMemType(gl2v, &c.gl2d, NULL));
  PetscCall(VecGetArrayAndMemType(gl3v, &c.gl3d, NULL));
  PetscCall(VecGetArrayAndMemType(x2v, &c.x2d, NULL));
  PetscCall(VecGetArrayAndMemType(x2lv, &c.x2ld, NULL));
  PetscCall(VecGetArrayAndMemType(x3v, &c.x3d, NULL));
  PetscCall(VecGetArrayAndMemType(x3lv, &c.x3ld, NULL));
  PetscCheck(PetscMemTypeDevice(mt), PETSC_COMM_SELF, PETSC_ERR_PLIB, "expected device memory");

  if (c.rank == 0) PetscCallNCCL(ncclGetUniqueId(&uid));
  PetscCallMPI(MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, PETSC_COMM_WORLD));
  PetscCallNCCL(ncclCommInitRank(&c.comm, size, uid, c.rank));
  PetscCallNCCL(ncclGetVersion(&ncclver));

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &sh));
  c.stream = *(cudaStream_t *)sh;

  /* eager warmup: connects all NCCL channels and settles PETSc lazy init, so the
     capture below records a steady-state iteration */
  for (PetscInt it = 0; it < warmup; it++) PetscCall(IterBody(&c));
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  /* capture k iterations into one graph */
  PetscCallCUDA(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeRelaxed));
  for (PetscInt j = 0; j < k; j++) PetscCall(IterBody(&c));
  PetscCallCUDA(cudaStreamEndCapture(c.stream, &graph));
  PetscCallCUDA(cudaGraphInstantiate(&gexec, graph, 0));

  nlaunch = iters / k;
  PetscCallCUDA(cudaGraphLaunch(gexec, c.stream)); /* first replay outside timing: NCCL may finalize plans */
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&t0));
  for (PetscInt it = 0; it < nlaunch; it++) PetscCallCUDA(cudaGraphLaunch(gexec, c.stream));
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscCall(PetscTime(&t1));

  dt = (t1 - t0) / (nlaunch * k);
  PetscCallMPI(MPI_Reduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
  if (c.rank == 0)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mode=%-10s nccl=%d.%d.%d graph_k=%-3" PetscInt_FMT " d=%-8" PetscInt_FMT " msg=%7.0f KB  G=%-3" PetscInt_FMT " P=%-3" PetscInt_FMT " per_iter=%9.2f us\n", c.allreduce ? "allred-gr" : "branch-gr", ncclver / 10000, ncclver / 100 % 100, ncclver % 100, k, c.d, c.d * sizeof(PetscScalar) / 1024.0, c.G, c.P, dtmax * 1e6));

  PetscCall(VecRestoreArrayAndMemType(x1v, &c.x1d));
  PetscCall(VecRestoreArrayAndMemType(x1lv, &c.x1ld));
  PetscCall(VecRestoreArrayAndMemType(gv, &c.gd));
  PetscCall(VecRestoreArrayAndMemType(gl2v, &c.gl2d));
  PetscCall(VecRestoreArrayAndMemType(gl3v, &c.gl3d));
  PetscCall(VecRestoreArrayAndMemType(x2v, &c.x2d));
  PetscCall(VecRestoreArrayAndMemType(x2lv, &c.x2ld));
  PetscCall(VecRestoreArrayAndMemType(x3v, &c.x3d));
  PetscCall(VecRestoreArrayAndMemType(x3lv, &c.x3ld));

  if (c.rank != 1) PetscCall(CheckSum(x1lv, 1.0, c.d, "x1", &fails));
  if (!c.allreduce && c.rank == 2) PetscCall(CheckSum(gl2v, 0.0 + 1.0, c.d, "g->slot2", &fails));
  if (!c.allreduce && c.rank == 3) PetscCall(CheckSum(gl3v, 0.0 + 1.0 + 2.0, c.d, "g->slot3", &fails));
  if (c.rank == 1 || c.rank == 3) PetscCall(CheckSum(x2lv, 1.0, c.d, "x2", &fails));
  if (c.rank == 1 || c.rank == 2) PetscCall(CheckSum(x3lv, 1.0, c.d, "x3", &fails));
  PetscCallMPI(MPI_Reduce(&fails, &allfails, 1, MPIU_INT, MPI_SUM, 0, PETSC_COMM_WORLD));
  if (c.rank == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "validation: %s\n", allfails ? "FAIL" : "PASS"));

  PetscCallCUDA(cudaGraphExecDestroy(gexec));
  PetscCallCUDA(cudaGraphDestroy(graph));
  PetscCallNCCL(ncclCommDestroy(c.comm));
  PetscCall(VecDestroy(&x1v));
  PetscCall(VecDestroy(&x1lv));
  PetscCall(VecDestroy(&gv));
  PetscCall(VecDestroy(&gl2v));
  PetscCall(VecDestroy(&gl3v));
  PetscCall(VecDestroy(&x2v));
  PetscCall(VecDestroy(&x2lv));
  PetscCall(VecDestroy(&x3v));
  PetscCall(VecDestroy(&x3lv));
  PetscCall(VecDestroy(&wa));
  PetscCall(VecDestroy(&wb));
  PetscCall(PetscFinalize());
  return 0;
}
