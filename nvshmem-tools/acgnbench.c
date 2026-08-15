static const char help[] = "ACGN splitting iteration skeleton (SC26 poster problem) on 4 GPUs.\n\
\n\
Reproduces the communication DAG of the four-node tomography splitting:\n\
  slot1 box prox (rank 1) -> x1 broadcast to ranks 0,2,3\n\
  shard gradients g0,g1,g2 at x1 (ranks 0,1,2, parallel)\n\
  branch routing: g0,g1 -> slot2 (rank 2); g0,g1,g2 -> slot3 (rank 3, consumed LATE)\n\
  slot2 rowTV prox (rank 2) -> x2 -> ranks 1,3\n\
  late gradient g3 at x2 (rank 3), then slot3 colTV prox -> x3 -> ranks 1,2\n\
  state mixing w -= theta*L*x on ranks 1,2,3\n\
Compute is emulated with VecAXPYs of the state size (same methodology as sfbench2.c).\n\
\n\
  -d <len>       state vector length in PetscScalars (default 65536 = 512 KB)\n\
  -gaxpy <G>     AXPYs per shard gradient evaluation (default 8)\n\
  -paxpy <P>     AXPYs per TV prox evaluation (default 4)\n\
  -allreduce     use a global gradient MPI_Allreduce instead of branch routing\n\
  -shards <m>    split each gradient into m slices, sending slice k while computing k+1\n\
                 (sender-side pipelining; m must divide -d and -gaxpy; branch mode only)\n\
  -monolithic    dumb single-core-style baseline: rank 0 does ALL compute serially, no comm\n\
  -skip_comm     skip all communication (pure-compute reference)\n\
  -iters <it>    timed iterations (default 300)\n\n";

#include <petscsf.h>
#include <petscvec.h>
#include <petscdevice_cuda.h>

static PetscErrorCode AxpyN(Vec y, Vec x, PetscInt k)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < k; i++) PetscCall(VecAXPY(y, 1.0e-8, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build an SF whose leaves on this rank reference nsrc remote roots, d entries each */
static PetscErrorCode MakeSF(MPI_Comm comm, PetscInt d, PetscBool isroot, PetscInt nsrc, const PetscMPIInt srcs[], PetscSF *sf)
{
  PetscSFNode *iremote = NULL;
  PetscInt     nleaves = nsrc * d;

  PetscFunctionBeginUser;
  PetscCall(PetscSFCreate(comm, sf));
  PetscCall(PetscSFSetType(*sf, PETSCSFBASIC));
  if (nleaves) {
    PetscCall(PetscMalloc1(nleaves, &iremote));
    for (PetscInt s = 0; s < nsrc; s++)
      for (PetscInt i = 0; i < d; i++) {
        iremote[s * d + i].rank  = srcs[s];
        iremote[s * d + i].index = i;
      }
  }
  PetscCall(PetscSFSetGraph(*sf, isroot ? d : 0, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(*sf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Shard k (of m slices, each ss = d/m entries) of the same edge set: roots are the slice
   [k*ss, (k+1)*ss) of the sender's d-length buffer; the receiver's leaves land at the
   same slice within each source's lane of its (nsrc * d)-length buffer, so an explicit
   ilocal is required (leaves are strided, one slice per lane) */
static PetscErrorCode MakeSFShard(MPI_Comm comm, PetscInt d, PetscInt ss, PetscInt k, PetscBool isroot, PetscInt nsrc, const PetscMPIInt srcs[], PetscSF *sf)
{
  PetscSFNode *iremote = NULL;
  PetscInt    *ilocal  = NULL;
  PetscInt     nleaves = nsrc * ss;

  PetscFunctionBeginUser;
  PetscCall(PetscSFCreate(comm, sf));
  PetscCall(PetscSFSetType(*sf, PETSCSFBASIC));
  if (nleaves) {
    PetscCall(PetscMalloc1(nleaves, &iremote));
    PetscCall(PetscMalloc1(nleaves, &ilocal));
    for (PetscInt s = 0; s < nsrc; s++)
      for (PetscInt i = 0; i < ss; i++) {
        ilocal[s * ss + i]        = s * d + k * ss + i;
        iremote[s * ss + i].rank  = srcs[s];
        iremote[s * ss + i].index = k * ss + i;
      }
  }
  PetscCall(PetscSFSetGraph(*sf, isroot ? d : 0, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(*sf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscSF        sfx1, sfx2, sfx3, *sfg2s, *sfg3s;
  Vec            x1v, x1lv, gv, gl2v, gl3v, x2v, x2lv, x3v, x3lv, wa, wb;
  PetscScalar   *x1d, *x1ld, *gd, *gl2d, *gl3d, *x2d, *x2ld, *x3d, *x3ld;
  PetscMemType   mt;
  PetscMPIInt    rank, size;
  PetscInt       d = 65536, G = 8, P = 4, iters = 300, warmup = 30, m = 1;
  PetscBool      allreduce = PETSC_FALSE, skipcomm = PETSC_FALSE, nvshmem = PETSC_FALSE, monolithic = PETSC_FALSE;
  PetscLogDouble t0, t1, dt, dtmax;
  const PetscMPIInt srcs_x1[] = {1}, srcs_g2[] = {0, 1}, srcs_g3[] = {0, 1, 2}, srcs_x2[] = {2}, srcs_x3[] = {3};

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
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-monolithic", &monolithic, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-skip_comm", &skipcomm, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_nvshmem", &nvshmem, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-shards", &m, NULL));
  PetscCheck(m >= 1 && d % m == 0 && G % m == 0, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "-shards m must divide both -d and -gaxpy");

  /* One SF per DAG edge group; every rank participates in every SF. The gradient edges
     are split into m shard slices so each slice's sends can be posted as soon as that
     slice's compute finishes (sender-side pipelining; m=1 reproduces the original). */
  PetscCall(MakeSF(PETSC_COMM_WORLD, d, (PetscBool)(rank == 1), rank != 1 ? 1 : 0, srcs_x1, &sfx1));
  PetscCall(PetscMalloc1(m, &sfg2s));
  PetscCall(PetscMalloc1(m, &sfg3s));
  if (m == 1) {
    PetscCall(MakeSF(PETSC_COMM_WORLD, d, (PetscBool)(rank <= 1), rank == 2 ? 2 : 0, srcs_g2, &sfg2s[0]));
    PetscCall(MakeSF(PETSC_COMM_WORLD, d, (PetscBool)(rank <= 2), rank == 3 ? 3 : 0, srcs_g3, &sfg3s[0]));
  } else {
    for (PetscInt k = 0; k < m; k++) {
      PetscCall(MakeSFShard(PETSC_COMM_WORLD, d, d / m, k, (PetscBool)(rank <= 1), rank == 2 ? 2 : 0, srcs_g2, &sfg2s[k]));
      PetscCall(MakeSFShard(PETSC_COMM_WORLD, d, d / m, k, (PetscBool)(rank <= 2), rank == 3 ? 3 : 0, srcs_g3, &sfg3s[k]));
    }
  }
  PetscCall(MakeSF(PETSC_COMM_WORLD, d, (PetscBool)(rank == 2), (rank == 1 || rank == 3) ? 1 : 0, srcs_x2, &sfx2));
  PetscCall(MakeSF(PETSC_COMM_WORLD, d, (PetscBool)(rank == 3), (rank == 1 || rank == 2) ? 1 : 0, srcs_x3, &sfx3));

  /* All buffers allocated on all ranks so every rank passes non-NULL device pointers */
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

  for (PetscInt it = 0; it < warmup + iters; it++) {
    if (it == warmup) {
      PetscCallCUDA(cudaDeviceSynchronize());
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
      PetscCall(PetscTime(&t0));
    }
    /* monolithic baseline: the whole iteration's compute serially on rank 0, no comm
       (box 1 + gradients 4G + rowTV accumulate/prox 2+P + colTV accumulate/prox 3+P + mixing 6) */
    if (monolithic) {
      if (rank == 0) PetscCall(AxpyN(wb, wa, 4 * G + 2 * P + 12));
      continue;
    }
    /* slot 1: box prox on rank 1, then x1 -> ranks 0,2,3 */
    if (rank == 1) PetscCall(AxpyN(wb, wa, 1));
    if (!skipcomm) {
      PetscCall(PetscSFBcastWithMemTypeBegin(sfx1, MPIU_SCALAR, PETSC_MEMTYPE_CUDA, x1d, PETSC_MEMTYPE_CUDA, x1ld, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sfx1, MPIU_SCALAR, x1d, x1ld, MPI_REPLACE));
    }
    /* shard gradients g0,g1,g2 at x1, in parallel on ranks 0,1,2; with -shards m each
       rank computes slice k (G/m AXPYs), immediately posts slice k's sends, and computes
       slice k+1 while they fly */
    if (skipcomm || allreduce) {
      if (rank <= 2) PetscCall(AxpyN(wb, wa, G));
      if (!skipcomm) {
        /* global gradient reduction: every rank synchronizes and receives the full sum */
        PetscCallCUDA(cudaDeviceSynchronize());
        PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, gd, (PetscMPIInt)d, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD));
      }
    } else {
      /* branch-specific routing; slot 3's copies are consumed only after g3 below */
      for (PetscInt k = 0; k < m; k++) {
        if (rank <= 2) PetscCall(AxpyN(wb, wa, G / m));
        PetscCall(PetscSFBcastWithMemTypeBegin(sfg2s[k], MPIU_SCALAR, PETSC_MEMTYPE_CUDA, gd, PETSC_MEMTYPE_CUDA, gl2d, MPI_REPLACE));
        PetscCall(PetscSFBcastWithMemTypeBegin(sfg3s[k], MPIU_SCALAR, PETSC_MEMTYPE_CUDA, gd, PETSC_MEMTYPE_CUDA, gl3d, MPI_REPLACE));
      }
      for (PetscInt k = 0; k < m; k++) PetscCall(PetscSFBcastEnd(sfg2s[k], MPIU_SCALAR, gd, gl2d, MPI_REPLACE));
    }
    /* slot 2: accumulate g0,g1 + rowTV prox on rank 2, then x2 -> ranks 1,3 */
    if (rank == 2) PetscCall(AxpyN(wb, wa, 2 + P));
    if (!skipcomm) {
      PetscCall(PetscSFBcastWithMemTypeBegin(sfx2, MPIU_SCALAR, PETSC_MEMTYPE_CUDA, x2d, PETSC_MEMTYPE_CUDA, x2ld, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sfx2, MPIU_SCALAR, x2d, x2ld, MPI_REPLACE));
    }
    /* late gradient g3 at x2 on rank 3, then collect g0,g1,g2 (overlapped since Begin) */
    if (rank == 3) PetscCall(AxpyN(wb, wa, G));
    if (!skipcomm && !allreduce)
      for (PetscInt k = 0; k < m; k++) PetscCall(PetscSFBcastEnd(sfg3s[k], MPIU_SCALAR, gd, gl3d, MPI_REPLACE));
    /* slot 3: accumulate + colTV prox on rank 3, then x3 -> ranks 1,2 */
    if (rank == 3) PetscCall(AxpyN(wb, wa, 3 + P));
    if (!skipcomm) {
      PetscCall(PetscSFBcastWithMemTypeBegin(sfx3, MPIU_SCALAR, PETSC_MEMTYPE_CUDA, x3d, PETSC_MEMTYPE_CUDA, x3ld, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sfx3, MPIU_SCALAR, x3d, x3ld, MPI_REPLACE));
    }
    /* state mixing w -= theta*L*x on ranks 1,2,3 */
    if (rank >= 1) PetscCall(AxpyN(wb, wa, 2));
  }
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscCall(PetscTime(&t1));

  dt = (t1 - t0) / iters;
  PetscCallMPI(MPI_Reduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
  if (rank == 0)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mode=%-10s nvshmem=%d d=%-8" PetscInt_FMT " msg=%7.0f KB  G=%-3" PetscInt_FMT " P=%-3" PetscInt_FMT " m=%-3" PetscInt_FMT " per_iter=%9.2f us\n", skipcomm ? "compute" : monolithic ? "monolithic" : allreduce ? "allreduce" : "branch", (int)nvshmem, d, d * sizeof(PetscScalar) / 1024.0, G, P, m, dtmax * 1e6));

  PetscCall(VecRestoreArrayAndMemType(x1v, &x1d));
  PetscCall(VecRestoreArrayAndMemType(x1lv, &x1ld));
  PetscCall(VecRestoreArrayAndMemType(gv, &gd));
  PetscCall(VecRestoreArrayAndMemType(gl2v, &gl2d));
  PetscCall(VecRestoreArrayAndMemType(gl3v, &gl3d));
  PetscCall(VecRestoreArrayAndMemType(x2v, &x2d));
  PetscCall(VecRestoreArrayAndMemType(x2lv, &x2ld));
  PetscCall(VecRestoreArrayAndMemType(x3v, &x3d));
  PetscCall(VecRestoreArrayAndMemType(x3lv, &x3ld));
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
  PetscCall(PetscSFDestroy(&sfx1));
  for (PetscInt k = 0; k < m; k++) {
    PetscCall(PetscSFDestroy(&sfg2s[k]));
    PetscCall(PetscSFDestroy(&sfg3s[k]));
  }
  PetscCall(PetscFree(sfg2s));
  PetscCall(PetscFree(sfg3s));
  PetscCall(PetscSFDestroy(&sfx2));
  PetscCall(PetscSFDestroy(&sfx3));
  PetscCall(PetscFinalize());
  return 0;
}
