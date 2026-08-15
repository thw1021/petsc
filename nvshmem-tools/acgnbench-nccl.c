static const char help[] = "ACGN splitting iteration with NCCL edge transport (SC26 poster problem) on 4 GPUs.\n\
\n\
Same DAG and emulated compute as acgnbench.c, but every DAG stage's edges are one fused\n\
ncclGroupStart/End of ncclSend/ncclRecv, issued on PETSc's compute stream. Per-rank\n\
programs contain only their own edges; the host never blocks inside the iteration.\n\
\n\
  -d <len>       state vector length in PetscScalars (default 65536 = 512 KB)\n\
  -gaxpy <G>     AXPYs per shard gradient evaluation (default 8)\n\
  -paxpy <P>     AXPYs per TV prox evaluation (default 4)\n\
  -allreduce     use a global gradient ncclAllReduce instead of branch routing\n\
  -skip_comm     skip all communication (pure-compute reference)\n\
  -iters <it>    timed iterations (default 300)\n\n";

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

static PetscErrorCode AxpyN(Vec y, Vec x, PetscInt k)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < k; i++) PetscCall(VecAXPY(y, 1.0e-8, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Validate one received lane: sum(v) must equal expect * d */
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
  PetscScalar       *x1d, *x1ld, *gd, *gl2d, *gl3d, *x2d, *x2ld, *x3d, *x3ld;
  PetscMemType       mt;
  PetscMPIInt        rank, size;
  PetscInt           d = 65536, G = 8, P = 4, iters = 300, warmup = 30, fails = 0, allfails = 0;
  PetscBool          allreduce = PETSC_FALSE, skipcomm = PETSC_FALSE;
  PetscLogDouble     t0, t1, dt, dtmax;
  PetscDeviceContext dctx;
  cudaStream_t       stream;
  void              *sh;
  ncclComm_t         comm;
  ncclUniqueId       uid;
  int                ncclver = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 4, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "This benchmark models the 4-node poster problem; run with exactly 4 ranks");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-d", &d, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-gaxpy", &G, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-paxpy", &P, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iters", &iters, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-allreduce", &allreduce, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-skip_comm", &skipcomm, NULL));

  /* All buffers on all ranks, as in acgnbench.c; creating them initializes the device */
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &x1v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &x1lv));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &gv));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, 2 * d, &gl2v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, 3 * d, &gl3v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &x2v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &x2lv));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &x3v));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &x3lv));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &wa));
  PetscCall(VecCreateSeqCUDA(PETSC_COMM_SELF, d, &wb));
  PetscCall(VecSet(x1v, 1.0));
  PetscCall(VecSet(x1lv, 0.0));
  PetscCall(VecSet(gv, (PetscScalar)rank));
  PetscCall(VecSet(gl2v, 0.0));
  PetscCall(VecSet(gl3v, 0.0));
  PetscCall(VecSet(x2v, 1.0));
  PetscCall(VecSet(x2lv, 0.0));
  PetscCall(VecSet(x3v, 1.0));
  PetscCall(VecSet(x3lv, 0.0));
  PetscCall(VecSet(wa, 1.0));
  PetscCall(VecSet(wb, 2.0));
  PetscCall(VecGetArrayAndMemType(x1v, &x1d, &mt));
  PetscCall(VecGetArrayAndMemType(x1lv, &x1ld, NULL));
  PetscCall(VecGetArrayAndMemType(gv, &gd, NULL));
  PetscCall(VecGetArrayAndMemType(gl2v, &gl2d, NULL));
  PetscCall(VecGetArrayAndMemType(gl3v, &gl3d, NULL));
  PetscCall(VecGetArrayAndMemType(x2v, &x2d, NULL));
  PetscCall(VecGetArrayAndMemType(x2lv, &x2ld, NULL));
  PetscCall(VecGetArrayAndMemType(x3v, &x3d, NULL));
  PetscCall(VecGetArrayAndMemType(x3lv, &x3ld, NULL));
  PetscCheck(PetscMemTypeDevice(mt), PETSC_COMM_SELF, PETSC_ERR_PLIB, "expected device memory");

  /* NCCL communicator over the 4 ranks; PETSc has already set the device */
  if (rank == 0) PetscCallNCCL(ncclGetUniqueId(&uid));
  PetscCallMPI(MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, PETSC_COMM_WORLD));
  PetscCallNCCL(ncclCommInitRank(&comm, size, uid, rank));
  PetscCallNCCL(ncclGetVersion(&ncclver));

  /* PETSc's compute stream: NCCL ops issued here are ordered with the VecAXPYs */
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &sh));
  stream = *(cudaStream_t *)sh;

  for (PetscInt it = 0; it < warmup + iters; it++) {
    if (it == warmup) {
      PetscCallCUDA(cudaDeviceSynchronize());
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&t0));
    }
    /* slot 1: box prox on rank 1, then x1 -> ranks 0,2,3 (one fused group) */
    if (rank == 1) PetscCall(AxpyN(wb, wa, 1));
    if (!skipcomm) {
      PetscCallNCCL(ncclGroupStart());
      if (rank == 1) {
        PetscCallNCCL(ncclSend(x1d, d, ncclDouble, 0, comm, stream));
        PetscCallNCCL(ncclSend(x1d, d, ncclDouble, 2, comm, stream));
        PetscCallNCCL(ncclSend(x1d, d, ncclDouble, 3, comm, stream));
      } else PetscCallNCCL(ncclRecv(x1ld, d, ncclDouble, 1, comm, stream));
      PetscCallNCCL(ncclGroupEnd());
    }
    /* shard gradients g0,g1,g2 at x1, in parallel on ranks 0,1,2 */
    if (rank <= 2) PetscCall(AxpyN(wb, wa, G));
    if (!skipcomm) {
      if (allreduce) {
        /* global gradient reduction, stream-ordered: no host synchronization */
        PetscCallNCCL(ncclAllReduce(gd, gd, d, ncclDouble, ncclSum, comm, stream));
      } else {
        /* branch routing: one fused group of 5 sends/recvs; only endpoints post ops */
        PetscCallNCCL(ncclGroupStart());
        if (rank == 0) {
          PetscCallNCCL(ncclSend(gd, d, ncclDouble, 2, comm, stream));
          PetscCallNCCL(ncclSend(gd, d, ncclDouble, 3, comm, stream));
        }
        if (rank == 1) {
          PetscCallNCCL(ncclSend(gd, d, ncclDouble, 2, comm, stream));
          PetscCallNCCL(ncclSend(gd, d, ncclDouble, 3, comm, stream));
        }
        if (rank == 2) {
          PetscCallNCCL(ncclRecv(gl2d, d, ncclDouble, 0, comm, stream));
          PetscCallNCCL(ncclRecv(gl2d + d, d, ncclDouble, 1, comm, stream));
          PetscCallNCCL(ncclSend(gd, d, ncclDouble, 3, comm, stream));
        }
        if (rank == 3) {
          PetscCallNCCL(ncclRecv(gl3d, d, ncclDouble, 0, comm, stream));
          PetscCallNCCL(ncclRecv(gl3d + d, d, ncclDouble, 1, comm, stream));
          PetscCallNCCL(ncclRecv(gl3d + 2 * d, d, ncclDouble, 2, comm, stream));
        }
        PetscCallNCCL(ncclGroupEnd());
      }
    }
    /* slot 2: accumulate g0,g1 + rowTV prox on rank 2, then x2 -> ranks 1,3 */
    if (rank == 2) PetscCall(AxpyN(wb, wa, 2 + P));
    if (!skipcomm) {
      PetscCallNCCL(ncclGroupStart());
      if (rank == 2) {
        PetscCallNCCL(ncclSend(x2d, d, ncclDouble, 1, comm, stream));
        PetscCallNCCL(ncclSend(x2d, d, ncclDouble, 3, comm, stream));
      } else if (rank == 1 || rank == 3) PetscCallNCCL(ncclRecv(x2ld, d, ncclDouble, 2, comm, stream));
      PetscCallNCCL(ncclGroupEnd());
    }
    /* late gradient g3 at x2 on rank 3 (g0,g1,g2 already landed, stream-ordered) */
    if (rank == 3) PetscCall(AxpyN(wb, wa, G));
    /* slot 3: accumulate + colTV prox on rank 3, then x3 -> ranks 1,2 */
    if (rank == 3) PetscCall(AxpyN(wb, wa, 3 + P));
    if (!skipcomm) {
      PetscCallNCCL(ncclGroupStart());
      if (rank == 3) {
        PetscCallNCCL(ncclSend(x3d, d, ncclDouble, 1, comm, stream));
        PetscCallNCCL(ncclSend(x3d, d, ncclDouble, 2, comm, stream));
      } else if (rank == 1 || rank == 2) PetscCallNCCL(ncclRecv(x3ld, d, ncclDouble, 3, comm, stream));
      PetscCallNCCL(ncclGroupEnd());
    }
    /* state mixing w -= theta*L*x on ranks 1,2,3 */
    if (rank >= 1) PetscCall(AxpyN(wb, wa, 2));
  }
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscCall(PetscTime(&t1));

  dt = (t1 - t0) / iters;
  PetscCallMPI(MPI_Reduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
  if (rank == 0)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mode=%-10s nccl=%d.%d.%d d=%-8" PetscInt_FMT " msg=%7.0f KB  G=%-3" PetscInt_FMT " P=%-3" PetscInt_FMT " per_iter=%9.2f us\n", skipcomm ? "compute" : allreduce ? "allreduce" : "branch", ncclver / 10000, ncclver / 100 % 100, ncclver % 100, d, d * sizeof(PetscScalar) / 1024.0, G, P, dtmax * 1e6));

  PetscCall(VecRestoreArrayAndMemType(x1v, &x1d));
  PetscCall(VecRestoreArrayAndMemType(x1lv, &x1ld));
  PetscCall(VecRestoreArrayAndMemType(gv, &gd));
  PetscCall(VecRestoreArrayAndMemType(gl2v, &gl2d));
  PetscCall(VecRestoreArrayAndMemType(gl3v, &gl3d));
  PetscCall(VecRestoreArrayAndMemType(x2v, &x2d));
  PetscCall(VecRestoreArrayAndMemType(x2lv, &x2ld));
  PetscCall(VecRestoreArrayAndMemType(x3v, &x3d));
  PetscCall(VecRestoreArrayAndMemType(x3lv, &x3ld));

  /* Correctness: every received lane holds known constants (skip g lanes under allreduce) */
  if (!skipcomm) {
    if (rank != 1) PetscCall(CheckSum(x1lv, 1.0, d, "x1", &fails));
    if (!allreduce && rank == 2) PetscCall(CheckSum(gl2v, 0.0 + 1.0, d, "g->slot2", &fails));
    if (!allreduce && rank == 3) PetscCall(CheckSum(gl3v, 0.0 + 1.0 + 2.0, d, "g->slot3", &fails));
    if (rank == 1 || rank == 3) PetscCall(CheckSum(x2lv, 1.0, d, "x2", &fails));
    if (rank == 1 || rank == 2) PetscCall(CheckSum(x3lv, 1.0, d, "x3", &fails));
  }
  PetscCallMPI(MPI_Reduce(&fails, &allfails, 1, MPIU_INT, MPI_SUM, 0, PETSC_COMM_WORLD));
  if (rank == 0 && !skipcomm) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "validation: %s\n", allfails ? "FAIL" : "PASS"));

  PetscCallNCCL(ncclCommDestroy(comm));
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
