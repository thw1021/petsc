static const char help[] = "PetscSF ring exchange captured into a CUDA graph (needs a stream-ordered transport).\n\
\n\
Times PetscSFBcast() on a ring three ways: eagerly issued (as sfbench.c), captured once into a\n\
CUDA graph holding -per_graph exchanges and replayed, and the same with -naxpy VecAXPYs of\n\
overlappable compute between Begin and End. The GPU-aware MPI path host-synchronizes inside\n\
Begin and therefore cannot be captured (cudaStreamBeginCapture() rejects it); the NCCL path\n\
(-use_nccl 1) is stream-ordered and captures. Run with\n\
  -root_device_context_stream_type nonblocking\n\
because the legacy null stream cannot be captured either.\n\
\n\
  -n <count>       message size per exchange in PetscScalars (default 4096 = 32 KB)\n\
  -per_graph <m>   exchanges captured into one graph (default 1)\n\
  -naxpy <k>       VecAXPYs on a 1M-entry work vector between Begin and End (default 0)\n\
  -iters <it>      timed exchanges (default 1000)\n\
  -eager_only      skip the graph arms\n\n";

#include <petscsf.h>
#include <petscvec.h>
#include <petscdevice_cuda.h>

static PetscErrorCode Exchange(PetscSF sf, PetscScalar *rootdata, PetscScalar *leafdata, Vec wa, Vec wb, PetscInt naxpy)
{
  PetscFunctionBeginUser;
  PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR, PETSC_MEMTYPE_CUDA, rootdata, PETSC_MEMTYPE_CUDA, leafdata, MPI_REPLACE));
  for (PetscInt j = 0; j < naxpy; j++) PetscCall(VecAXPY(wb, 1.0e-8, wa));
  PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, rootdata, leafdata, MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Report(const char *label, PetscInt n, PetscInt m, PetscInt naxpy, PetscLogDouble dt, PetscInt cnt)
{
  PetscLogDouble dtmax;

  PetscFunctionBeginUser;
  dt /= cnt;
  PetscCallMPI(MPI_Reduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %-8s n=%-7" PetscInt_FMT " per_graph=%-3" PetscInt_FMT " naxpy=%-3" PetscInt_FMT " %9.2f us/exchange\n", label, n, m, naxpy, dtmax * 1e6));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscSF            sf;
  PetscSFNode       *iremote;
  Vec                rootvec, leafvec, wa, wb;
  PetscScalar       *rootdata, *leafdata;
  PetscMPIInt        rank, size, src;
  PetscInt           n = 4096, m = 1, naxpy = 0, iters = 1000, warmup = 20, W = 1 << 20;
  PetscLogDouble     t0, t1;
  PetscBool          eager_only = PETSC_FALSE;
  PetscDeviceContext dctx;
  cudaStream_t      *stream;
  cudaGraph_t        graph;
  cudaGraphExec_t    gexec;
  PetscReal          norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-per_graph", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-naxpy", &naxpy, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iters", &iters, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-eager_only", &eager_only, NULL));
  PetscCheck(size > 1, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Need >= 2 ranks");
  iters = (iters / m) * m;

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

  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, n, &rootvec));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, n, &leafvec));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, W, &wa));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, W, &wb));
  PetscCall(VecSet(rootvec, (PetscScalar)(rank + 1)));
  PetscCall(VecSet(leafvec, 0.0));
  PetscCall(VecSet(wa, 1.0));
  PetscCall(VecSet(wb, 2.0));
  PetscCall(VecGetArrayAndMemType(rootvec, &rootdata, NULL));
  PetscCall(VecGetArrayAndMemType(leafvec, &leafdata, NULL));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, (void **)&stream));

  /* eager arm: also serves as the warm-up that creates the SF link, the NCCL communicator and its connections */
  for (PetscInt i = 0; i < warmup + iters; i++) {
    if (i == warmup) {
      PetscCallCUDA(cudaDeviceSynchronize());
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&t0));
    }
    PetscCall(Exchange(sf, rootdata, leafdata, wa, wb, naxpy));
  }
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscCall(PetscTime(&t1));
  PetscCall(Report("eager", n, 1, naxpy, t1 - t0, iters));
  PetscCall(VecRestoreArrayAndMemType(leafvec, &leafdata)); /* leafdata was written through the raw pointer; the restore drops the norm VecSet() cached */
  PetscCall(VecNorm(leafvec, NORM_1, &norm));
  PetscCall(VecGetArrayAndMemType(leafvec, &leafdata, NULL));
  PetscCheck(PetscAbsReal(norm - (PetscReal)n * (src + 1)) < 1e-6 * n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "eager: wrong leaf data on rank %d: norm %g", rank, (double)norm);

  if (!eager_only) {
    PetscCall(VecSet(leafvec, 0.0));
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCallCUDA(cudaStreamBeginCapture(*stream, cudaStreamCaptureModeThreadLocal));
    for (PetscInt k = 0; k < m; k++) PetscCall(Exchange(sf, rootdata, leafdata, wa, wb, naxpy));
    PetscCallCUDA(cudaStreamEndCapture(*stream, &graph));
    PetscCallCUDA(cudaGraphInstantiate(&gexec, graph, 0));
    for (PetscInt i = 0; i < warmup + iters / m; i++) {
      if (i == warmup) {
        PetscCallCUDA(cudaDeviceSynchronize());
        PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
        PetscCall(PetscTime(&t0));
      }
      PetscCallCUDA(cudaGraphLaunch(gexec, *stream));
    }
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCall(PetscTime(&t1));
    PetscCall(Report("graph", n, m, naxpy, t1 - t0, iters));
    PetscCall(VecRestoreArrayAndMemType(leafvec, &leafdata)); /* leafdata was written through the raw pointer; the restore drops the norm VecSet() cached */
  PetscCall(VecNorm(leafvec, NORM_1, &norm));
  PetscCall(VecGetArrayAndMemType(leafvec, &leafdata, NULL));
    PetscCheck(PetscAbsReal(norm - (PetscReal)n * (src + 1)) < 1e-6 * n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "graph: wrong leaf data on rank %d: norm %g", rank, (double)norm);
    PetscCallCUDA(cudaGraphExecDestroy(gexec));
    PetscCallCUDA(cudaGraphDestroy(graph));
  }

  PetscCall(VecRestoreArrayAndMemType(rootvec, &rootdata));
  PetscCall(VecRestoreArrayAndMemType(leafvec, &leafdata));
  PetscCall(VecDestroy(&rootvec));
  PetscCall(VecDestroy(&leafvec));
  PetscCall(VecDestroy(&wa));
  PetscCall(VecDestroy(&wb));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFinalize());
  return 0;
}
