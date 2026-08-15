/* Route 2b: ACGN splitting iteration compiled to device-side NVSHMEM (SC26 poster).
 *
 * Each iteration is ONE fused cooperative kernel per rank (or one persistent kernel for
 * the whole run with -persistent). Cross-GPU dependencies are nvshmemx_signal_op()
 * (producer) / nvshmem_signal_wait_until() (consumer) on symmetric-heap flags, issued
 * from inside the running kernel. Data moves by consumer-side pull: the producer writes
 * its own symmetric buffer, fences, signals; the consumer reads it directly through
 * nvshmem_ptr() over NVLink. No PetscSF, no NCCL, no per-op launches.
 *
 * Iteration reuse: signal values are monotonic (epoch v = it+1, NVSHMEM_CMP_GE), buffers
 * are double-buffered (slot = it & 1), and every data edge has a reverse ack signal so a
 * producer never overwrites a slot until its consumers acked iteration it-2.
 *
 * Correctness is race-detecting by construction: every published element is stamped with
 * (role_base + v); consumers validate every element every iteration, so a stale slot or
 * missing fence shows up as a nonzero error count, not silence.
 *
 * Options: -d <len> -gaxpy <G> -paxpy <P> -iters <n> -persistent -allreduce
 *   -allreduce replaces the branch routing by a direct all-pull allreduce of the
 *   gradient over all 4 PEs (publish + 3-way pull; latency-optimal at np=4).
 *   -converge [-tol t -rho r -check_every k] (persistent only): device-side stopping
 *   test -- a synthetic residual vector decays by rho per iteration; every k iterations
 *   each PE grid-reduces ||rv||^2, all-pulls the four partials (rank-ordered sum, so all
 *   PEs compute the bit-identical global norm and the same stop decision), and all four
 *   resident kernels break their loops on the same iteration. The host verifies the stop
 *   iteration against the analytic prediction.
 * Run with exactly 4 ranks, 1 per GPU, single NVLink node (nvshmem_ptr must be non-NULL).
 *
 * Build (notes section 5 recipe):
 *   nvcc -ccbin g++ -arch=sm_90 -rdc=true -O3 -I$MPI_INC -I$NVSHMEM_DIR/include -c acgnbench-nvdev.cu -o nvdev.o
 *   nvcc -ccbin g++ -arch=sm_90 -dlink nvdev.o -L$NVSHMEM_DIR/lib -lnvshmem_device -o nvdev_dlink.o
 *   mpicxx nvdev.o nvdev_dlink.o -L$NVSHMEM_DIR/lib -lnvshmem_host -lnvshmem_device \
 *          -L$CUDA_HOME/lib64 -lcudart -L$CUDA_HOME/lib64/stubs -lcuda -o acgnbench-nvdev
 */
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace cg = cooperative_groups;

#define CUDA_CHK(c) \
  do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
  } while (0)

/* data-signal indices (in each consumer's sig[]); SIG_N<p> carries PE p's partial norm
   epoch for the device-side convergence test */
enum { SIG_X1 = 0, SIG_G0, SIG_G1, SIG_G2, SIG_G3, SIG_X2, SIG_X3, SIG_N0, SIG_N1, SIG_N2, SIG_N3, NDATASIG };
/* ack indices (in each producer's ack[]; meaning depends on the producer rank)
   pe0: 0 = g0 ack from 2, 1 = g0 ack from 3            (branch mode)
   pe1: 0 = x1 from 0, 1 = x1 from 2, 2 = x1 from 3, 3 = g1 from 2, 4 = g1 from 3
   pe2: 0 = g2 from 3, 1 = x2 from 1, 2 = x2 from 3
   pe3: 0 = x3 from 1, 1 = x3 from 2
   allreduce mode g-exchange (uniform on every producer): 5 + consumer_rank */
#define ACK_G(r) (5 + (r))
#define NACK 9

/* stamp bases so a wrong-edge read is also detectable, not just a stale one */
#define X1S 10.0
#define X2S 20.0
#define X3S 30.0
#define GS  100.0 /* + producer rank */

typedef struct {
  double             *pub, *gpub; /* my symmetric buffers, 2*d each (slot-major)     */
  const double       *x1p, *g0p, *g1p, *g2p, *g3p, *x2p, *x3p; /* nvshmem_ptr peer views */
  unsigned long long *sig, *ack;  /* symmetric signal words                          */
  double             *wa, *wb;    /* local compute-emulation state                   */
  unsigned long long *err;        /* local validation-failure counter                */
  unsigned int       *cnt;        /* local block-arrival counters, per site x slot   */
  /* device-side convergence test (-converge, persistent mode only) */
  double             *rv;         /* local "residual" vector, decays by rho per iter */
  double             *npub;       /* symmetric double[2]: my partial norm, per cslot */
  const double       *npp[4];     /* nvshmem_ptr views of every PE's npub            */
  double             *normacc;    /* local double[2]: grid-summed partial, per cslot */
  int                *dec;        /* local int[2]: stop decision, per cslot          */
  unsigned long long *decepoch;   /* local ull[2]: epoch releasing dec[], per cslot  */
  long long          *stopiter;   /* local: iteration we stopped at (-1 = ran out)   */
  double              tol2, rho;
  long long           d;
  int                 G, P, pe, ar, converge, check_every;
} Ctx;

/* counter index: publish sites 0-1 and ack sites 0-2, each double-buffered by slot;
   NPCI is the norm-check arrival counter, double-buffered by CHECK index (not iteration:
   with -check_every k even, consecutive checks would land on one iteration parity) */
#define PCI(site, it) ((site) * 2 + (int)((it) & 1))
#define ACI(site, it) (4 + (site) * 2 + (int)((it) & 1))
#define NPCI(cslot)   (10 + (cslot))
#define NCNT 12

/* ---- device helpers; every helper leaves the grid converged ---- */

__device__ static inline void axpyN(const Ctx &c, int k, const cg::grid_group &g)
{
  const long long n = c.d, tid = (long long)g.thread_rank(), stride = (long long)g.num_threads();
  for (int j = 0; j < k; j++)
    for (long long i = tid; i < n; i += stride) c.wb[i] += 1e-8 * c.wa[i];
}

/* producer: wait my consumers' acks for this slot (iteration it-2). Every thread polls
   its local ack words; no barrier -- each thread gates itself. */
__device__ static inline void ackwait(const Ctx &c, long long it, const int *idx, int n)
{
  if (it < 2) return;
  const unsigned long long need = (unsigned long long)(it - 1); /* consumer acked it-2 with value it-1 */
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; i++) {
      const volatile unsigned long long *a = &c.ack[idx[i]];
      while (*a < need) {}
    }
    __threadfence();
  }
  __syncthreads();
}

/* producer: fill my slot with the stamp; the LAST block to arrive fires the signals.
   No grid barrier: blocks flow on immediately and self-gate at their next wait. Reset
   before signal is safe because no block can re-enter this site+slot (iteration it+2)
   until a consumer acks iteration it, which requires the very signal sent below. */
__device__ static inline void publish(const Ctx &c, double *buf, double stamp, unsigned long long v, int sigidx, const int *pes, int npes, int ci, const cg::grid_group &g)
{
  const long long n = c.d, tid = (long long)g.thread_rank(), stride = (long long)g.num_threads();
  for (long long i = tid; i < n; i += stride) buf[i] = stamp;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&c.cnt[ci], 1u) == gridDim.x - 1) {
    atomicExch(&c.cnt[ci], 0u);
    __threadfence_system();
    for (int i = 0; i < npes; i++) nvshmemx_signal_op((uint64_t *)&c.sig[sigidx], v, NVSHMEM_SIGNAL_SET, pes[i]);
  }
}

/* consumer: wait for the epochs. Every thread polls its LOCAL copy of the signal word
   (remote signal_op wrote it), so no grid barrier is needed on the wait side -- each
   thread gates itself, and the fence gives the poll acquire-like ordering before the
   peer-data reads that follow. */
__device__ static inline void waitsigs(const Ctx &c, unsigned long long v, const int *sigidx, int n, cg::grid_group &g)
{
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; i++) {
      const volatile unsigned long long *s = &c.sig[sigidx[i]];
      while (*s < v) {}
    }
    __threadfence();
  }
  __syncthreads();
}

__device__ static inline void consume(const Ctx &c, const double *src, long long off, double stamp, const cg::grid_group &g)
{
  const long long n = c.d, tid = (long long)g.thread_rank(), stride = (long long)g.num_threads();
  const volatile double *p = (const volatile double *)(src + off); /* bypass L1: peer data */
  for (long long i = tid; i < n; i += stride)
    if (p[i] != stamp) atomicAdd(c.err, 1ull);
}

/* consumer: this block's reads are done -> count arrival; the LAST block sends the
   ack(s) releasing the producer's slot. Same reset-safety argument as publish(). */
__device__ static inline void ackblock(const Ctx &c, unsigned long long v, const int *ackidx, const int *pes, int n, int ci)
{
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&c.cnt[ci], 1u) == gridDim.x - 1) {
    atomicExch(&c.cnt[ci], 0u);
    __threadfence();
    for (int i = 0; i < n; i++) nvshmemx_signal_op((uint64_t *)&c.ack[ackidx[i]], v, NVSHMEM_SIGNAL_SET, pes[i]);
  }
}

/* ---- one ACGN iteration, per-rank program; branches follow the frozen DAG ---- */

__device__ static inline const double *gpeer(const Ctx &c, int p)
{
  return p == 0 ? c.g0p : p == 1 ? c.g1p : p == 2 ? c.g2p : c.g3p;
}

/* Direct all-pull allreduce of the gradient over all 4 PEs: publish my stamped
   contribution, signal the three peers, pull and validate their three lanes. Latency-
   optimal at np=4 (one signal round, no ring steps); every PE runs this same code. */
__device__ static void ar_gphase(const Ctx &c, long long it, unsigned long long v, long long off, int gsite, cg::grid_group &g)
{
  int myg[3], dstp[3], ai[3], nb = 0;

  for (int p = 0; p < 4; p++)
    if (p != c.pe) {
      myg[nb]  = ACK_G(p);
      dstp[nb] = p;
      ai[nb]   = ACK_G(c.pe);
      nb++;
    }
  ackwait(c, it, myg, 3);
  publish(c, c.gpub + off, GS + c.pe + v, v, SIG_G0 + c.pe, dstp, 3, PCI(1, it), g);
  for (int p = 0; p < 4; p++) {
    int s1[1] = {SIG_G0 + p};
    if (p == c.pe) continue;
    waitsigs(c, v, s1, 1, g);
    consume(c, gpeer(c, p), off, GS + p + v, g);
  }
  ackblock(c, v, ai, dstp, 3, ACI(gsite, it));
}

/* allreduce-mode per-rank programs: x1/x2/x3 distribution as in branch mode, but the
   branch routing is replaced by ar_gphase() and rank 3 has no late-consume (it already
   holds the full reduction). Publish/ack counter sites are assigned per mode. */
__device__ static void iter_body_ar(const Ctx &c, long long it, cg::grid_group &g)
{
  const unsigned long long v = (unsigned long long)it + 1;
  const long long          off = (it & 1) * c.d;

  if (c.pe == 0) {
    static const int sx1[] = {SIG_X1}, ax1i[] = {0}, ax1p[] = {1};
    waitsigs(c, v, sx1, 1, g);
    consume(c, c.x1p, off, X1S + v, g);
    ackblock(c, v, ax1i, ax1p, 1, ACI(0, it));
    axpyN(c, c.G, g);                          /* shard gradient g0 */
    ar_gphase(c, it, v, off, 1, g);
  } else if (c.pe == 1) {
    static const int ax1[] = {0, 1, 2}, dx1[] = {0, 2, 3}, sx2[] = {SIG_X2}, sx3[] = {SIG_X3};
    static const int ax2i[] = {1}, ax2p[] = {2}, ax3i[] = {0}, ax3p[] = {3};
    axpyN(c, 1, g);                            /* box prox */
    ackwait(c, it, ax1, 3);
    publish(c, c.pub + off, X1S + v, v, SIG_X1, dx1, 3, PCI(0, it), g);
    axpyN(c, c.G, g);                          /* shard gradient g1 */
    ar_gphase(c, it, v, off, 0, g);
    waitsigs(c, v, sx2, 1, g);
    consume(c, c.x2p, off, X2S + v, g);
    ackblock(c, v, ax2i, ax2p, 1, ACI(1, it));
    waitsigs(c, v, sx3, 1, g);
    consume(c, c.x3p, off, X3S + v, g);
    ackblock(c, v, ax3i, ax3p, 1, ACI(2, it));
    axpyN(c, 2, g);                            /* state mixing */
  } else if (c.pe == 2) {
    static const int sx1[] = {SIG_X1}, ax1i[] = {1}, ax1p[] = {1};
    static const int ax2[] = {1, 2}, dx2[] = {1, 3}, sx3[] = {SIG_X3}, ax3i[] = {1}, ax3p[] = {3};
    waitsigs(c, v, sx1, 1, g);
    consume(c, c.x1p, off, X1S + v, g);
    ackblock(c, v, ax1i, ax1p, 1, ACI(0, it));
    axpyN(c, c.G, g);                          /* shard gradient g2 */
    ar_gphase(c, it, v, off, 1, g);
    axpyN(c, 2 + c.P, g);                      /* accumulate + rowTV prox */
    ackwait(c, it, ax2, 2);
    publish(c, c.pub + off, X2S + v, v, SIG_X2, dx2, 2, PCI(0, it), g);
    waitsigs(c, v, sx3, 1, g);
    consume(c, c.x3p, off, X3S + v, g);
    ackblock(c, v, ax3i, ax3p, 1, ACI(2, it));
    axpyN(c, 2, g);                            /* state mixing */
  } else { /* pe == 3 */
    static const int sx1[] = {SIG_X1}, sx2[] = {SIG_X2}, ax3[] = {0, 1}, dx3[] = {1, 2};
    static const int ax1i[] = {2}, ax1p[] = {1}, ax2i[] = {2}, ax2p[] = {2};
    waitsigs(c, v, sx1, 1, g);
    consume(c, c.x1p, off, X1S + v, g);
    ackblock(c, v, ax1i, ax1p, 1, ACI(0, it));
    ar_gphase(c, it, v, off, 2, g);            /* contributes its g, receives the sum */
    waitsigs(c, v, sx2, 1, g);
    consume(c, c.x2p, off, X2S + v, g);
    ackblock(c, v, ax2i, ax2p, 1, ACI(1, it));
    axpyN(c, c.G, g);                          /* late gradient g3 at x2 */
    axpyN(c, 3 + c.P, g);                      /* accumulate + colTV prox */
    ackwait(c, it, ax3, 2);
    publish(c, c.pub + off, X3S + v, v, SIG_X3, dx3, 2, PCI(0, it), g);
    axpyN(c, 2, g);                            /* state mixing */
  }
}

__device__ static void iter_body(const Ctx &c, long long it, cg::grid_group &g)
{
  const unsigned long long v = (unsigned long long)it + 1;
  const long long          off = (it & 1) * c.d;

  if (c.ar) {
    iter_body_ar(c, it, g);
    return;
  }
  if (c.pe == 0) {
    static const int sx1[] = {SIG_X1}, dst[] = {2, 3}, a[] = {0, 1}, ax1i[] = {0}, ax1p[] = {1};
    waitsigs(c, v, sx1, 1, g);
    consume(c, c.x1p, off, X1S + v, g);
    ackblock(c, v, ax1i, ax1p, 1, ACI(0, it)); /* pe1.ack[0] = x1 ack from 0 */
    axpyN(c, c.G, g);                          /* shard gradient g0 */
    ackwait(c, it, a, 2);                      /* my g0 slot free? (acks in my ack[0..1]) */
    publish(c, c.gpub + off, GS + 0 + v, v, SIG_G0, dst, 2, PCI(0, it), g);
  } else if (c.pe == 1) {
    static const int ax1[] = {0, 1, 2}, dx1[] = {0, 2, 3}, ag1[] = {3, 4}, dg1[] = {2, 3};
    static const int sx2[] = {SIG_X2}, sx3[] = {SIG_X3};
    static const int ax2i[] = {1}, ax2p[] = {2}, ax3i[] = {0}, ax3p[] = {3};
    axpyN(c, 1, g);                            /* box prox */
    ackwait(c, it, ax1, 3);
    publish(c, c.pub + off, X1S + v, v, SIG_X1, dx1, 3, PCI(0, it), g);
    axpyN(c, c.G, g);                          /* shard gradient g1 (x1 is local) */
    ackwait(c, it, ag1, 2);
    publish(c, c.gpub + off, GS + 1 + v, v, SIG_G1, dg1, 2, PCI(1, it), g);
    waitsigs(c, v, sx2, 1, g);
    consume(c, c.x2p, off, X2S + v, g);
    ackblock(c, v, ax2i, ax2p, 1, ACI(0, it)); /* pe2.ack[1] = x2 ack from 1 */
    waitsigs(c, v, sx3, 1, g);
    consume(c, c.x3p, off, X3S + v, g);
    ackblock(c, v, ax3i, ax3p, 1, ACI(1, it)); /* pe3.ack[0] = x3 ack from 1 */
    axpyN(c, 2, g);                            /* state mixing */
  } else if (c.pe == 2) {
    static const int sx1[] = {SIG_X1}, ag2[] = {0}, dg2[] = {3}, sg[] = {SIG_G0, SIG_G1};
    static const int ax2[] = {1, 2}, dx2[] = {1, 3}, sx3[] = {SIG_X3};
    static const int ax1i[] = {1}, ax1p[] = {1}, agi[] = {0, 3}, agp[] = {0, 1}, ax3i[] = {1}, ax3p[] = {3};
    waitsigs(c, v, sx1, 1, g);
    consume(c, c.x1p, off, X1S + v, g);
    ackblock(c, v, ax1i, ax1p, 1, ACI(0, it)); /* pe1.ack[1] = x1 ack from 2 */
    axpyN(c, c.G, g);                          /* shard gradient g2 */
    ackwait(c, it, ag2, 1);
    publish(c, c.gpub + off, GS + 2 + v, v, SIG_G2, dg2, 1, PCI(0, it), g);
    waitsigs(c, v, sg, 2, g);                  /* g0, g1 arrive */
    consume(c, c.g0p, off, GS + 0 + v, g);
    consume(c, c.g1p, off, GS + 1 + v, g);
    ackblock(c, v, agi, agp, 2, ACI(1, it));   /* pe0.ack[0], pe1.ack[3] */
    axpyN(c, 2 + c.P, g);                      /* accumulate + rowTV prox */
    ackwait(c, it, ax2, 2);
    publish(c, c.pub + off, X2S + v, v, SIG_X2, dx2, 2, PCI(1, it), g);
    waitsigs(c, v, sx3, 1, g);
    consume(c, c.x3p, off, X3S + v, g);
    ackblock(c, v, ax3i, ax3p, 1, ACI(2, it)); /* pe3.ack[1] = x3 ack from 2 */
    axpyN(c, 2, g);                            /* state mixing */
  } else { /* pe == 3 */
    static const int sx1[] = {SIG_X1}, sx2[] = {SIG_X2}, sg[] = {SIG_G0, SIG_G1, SIG_G2};
    static const int ax3[] = {0, 1}, dx3[] = {1, 2};
    static const int ax1i[] = {2}, ax1p[] = {1}, ax2i[] = {2}, ax2p[] = {2}, agi[] = {1, 4, 0}, agp[] = {0, 1, 2};
    waitsigs(c, v, sx1, 1, g);
    consume(c, c.x1p, off, X1S + v, g);
    ackblock(c, v, ax1i, ax1p, 1, ACI(0, it)); /* pe1.ack[2] = x1 ack from 3 */
    waitsigs(c, v, sx2, 1, g);
    consume(c, c.x2p, off, X2S + v, g);
    ackblock(c, v, ax2i, ax2p, 1, ACI(1, it)); /* pe2.ack[2] = x2 ack from 3 */
    axpyN(c, c.G, g);                          /* late gradient g3 at x2 */
    waitsigs(c, v, sg, 3, g);                  /* g0,g1,g2 consumed LATE, after g3 */
    consume(c, c.g0p, off, GS + 0 + v, g);
    consume(c, c.g1p, off, GS + 1 + v, g);
    consume(c, c.g2p, off, GS + 2 + v, g);
    ackblock(c, v, agi, agp, 3, ACI(2, it));   /* pe0.ack[1], pe1.ack[4], pe2.ack[0] */
    axpyN(c, 3 + c.P, g);                      /* accumulate + colTV prox */
    ackwait(c, it, ax3, 2);
    publish(c, c.pub + off, X3S + v, v, SIG_X3, dx3, 2, PCI(0, it), g);
    axpyN(c, 2, g);                            /* state mixing */
  }
}

/* Device-side convergence test: after decaying rv (done in the main loop), grid-reduce
   ||rv||^2 locally, all-pull the four partial norms (rank-ordered sum => bit-identical
   global value on every PE => identical stop decision, no extra agreement round), and
   release the decision through an epoch-stamped word every block polls. Flow control
   needs no acks: a PE reaches check c+1's publish only after passing check c's decision,
   which required reading everyone's check-c partials -- and the arrival counter cannot
   fire for check c+2 until every local block passed check c's decision poll, so slot and
   counter reuse are serialized behind the slowest block by construction. */
__device__ static bool check_phase(const Ctx &c, long long it, cg::grid_group &g)
{
  __shared__ double           bpart[256];
  const unsigned long long    v = (unsigned long long)it + 1;
  const int                   cslot = (int)(((it + 1) / c.check_every) & 1);
  const long long             n = c.d, tid = (long long)g.thread_rank(), stride = (long long)g.num_threads();
  double                      acc = 0;
  volatile const int         *decp = c.dec;

  for (long long i = tid; i < n; i += stride) acc += c.rv[i] * c.rv[i];
  bpart[threadIdx.x] = acc;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) bpart[threadIdx.x] += bpart[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&c.normacc[cslot], bpart[0]);
    __threadfence();
    if (atomicAdd(&c.cnt[NPCI(cslot)], 1u) == gridDim.x - 1) { /* last block: decide */
      double gsum = 0;
      atomicExch(&c.cnt[NPCI(cslot)], 0u);
      c.npub[cslot] = c.normacc[cslot];
      __threadfence_system();
      for (int p = 0; p < 4; p++)
        if (p != c.pe) nvshmemx_signal_op((uint64_t *)&c.sig[SIG_N0 + c.pe], v, NVSHMEM_SIGNAL_SET, p);
      for (int p = 0; p < 4; p++) { /* fixed rank order: bit-identical sum on all PEs */
        if (p != c.pe) {
          const volatile unsigned long long *s = &c.sig[SIG_N0 + p];
          while (*s < v) {}
          __threadfence();
        }
        gsum += ((volatile const double *)c.npp[p])[cslot];
      }
      c.normacc[cslot] = 0.0; /* reset for the check after next; gated by the decision poll */
      c.dec[cslot]     = (gsum <= c.tol2) ? 1 : 0;
      __threadfence();
      c.decepoch[cslot] = v; /* release */
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) { /* every block: leader polls the decision epoch */
    const volatile unsigned long long *e = &c.decepoch[cslot];
    while (*e < v) {}
    __threadfence();
  }
  __syncthreads();
  if (decp[cslot]) {
    if (g.thread_rank() == 0) *c.stopiter = it;
    return true;
  }
  return false;
}

__global__ void acgn_kernel(Ctx c, long long it0, long long it1)
{
  cg::grid_group g = cg::this_grid();
  for (long long it = it0; it < it1; it++) {
    iter_body(c, it, g);
    if (c.converge) {
      const long long n = c.d, tid = (long long)g.thread_rank(), stride = (long long)g.num_threads();
      for (long long i = tid; i < n; i += stride) c.rv[i] *= c.rho;
      if ((it + 1) % c.check_every == 0 && check_phase(c, it, g)) break;
    }
  }
}

int main(int argc, char **argv)
{
  int                 rank, size, ndev, local_rank = 0, persistent = 0, nsm = 0, maxblk = 0, blkcap = 32;
  double              tol = 24.0;
  long long           stopped = -1, nexec, npred = 0;
  long long           d = 65536, iters = 300, warmup = 30;
  int                 G = 8, P = 4;
  const char         *lr;
  MPI_Comm            comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  Ctx                 c = {};
  double             *hostinit, t0, t1, dt, dtmax;
  unsigned long long  errs = 0, allerrs = 0;
  dim3                grid, block(256);
  void               *kargs[3];
  long long           a0, a1;
  cudaDeviceProp      prop;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (size != 4) {
    if (!rank) fprintf(stderr, "run with exactly 4 ranks\n");
    MPI_Abort(comm, 1);
  }
  c.rho         = 0.99;
  c.check_every = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-d") && i + 1 < argc) d = atoll(argv[++i]);
    else if (!strcmp(argv[i], "-gaxpy") && i + 1 < argc) G = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-paxpy") && i + 1 < argc) P = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-iters") && i + 1 < argc) iters = atoll(argv[++i]);
    else if (!strcmp(argv[i], "-persistent")) persistent = 1;
    else if (!strcmp(argv[i], "-allreduce")) c.ar = 1;
    else if (!strcmp(argv[i], "-blocks") && i + 1 < argc) blkcap = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-converge")) c.converge = 1;
    else if (!strcmp(argv[i], "-tol") && i + 1 < argc) tol = atof(argv[++i]);
    else if (!strcmp(argv[i], "-rho") && i + 1 < argc) c.rho = atof(argv[++i]);
    else if (!strcmp(argv[i], "-check_every") && i + 1 < argc) c.check_every = atoi(argv[++i]);
  }
  if (c.converge && !persistent) {
    if (!rank) fprintf(stderr, "-converge requires -persistent (the stopping test lives inside the one resident kernel)\n");
    MPI_Abort(comm, 1);
  }
  c.tol2 = tol * tol;

  lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (lr) local_rank = atoi(lr);
  CUDA_CHK(cudaGetDeviceCount(&ndev));
  CUDA_CHK(cudaSetDevice(local_rank % ndev)); /* device BEFORE nvshmem init (notes sec 7) */
  attr.mpi_comm = &comm;
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr)) {
    fprintf(stderr, "nvshmem init failed\n");
    MPI_Abort(comm, 1);
  }
  c.pe = nvshmem_my_pe();
  c.d  = d;
  c.G  = G;
  c.P  = P;
  c.ar = c.ar ? 1 : 0;

  c.pub  = (double *)nvshmem_malloc(2 * d * sizeof(double));
  c.gpub = (double *)nvshmem_malloc(2 * d * sizeof(double));
  c.sig  = (unsigned long long *)nvshmem_calloc(NDATASIG, sizeof(unsigned long long));
  c.ack  = (unsigned long long *)nvshmem_calloc(NACK, sizeof(unsigned long long));
  if (!c.pub || !c.gpub || !c.sig || !c.ack) {
    fprintf(stderr, "nvshmem_malloc failed\n");
    MPI_Abort(comm, 1);
  }
  c.x1p = (const double *)nvshmem_ptr(c.pub, 1);
  c.x2p = (const double *)nvshmem_ptr(c.pub, 2);
  c.x3p = (const double *)nvshmem_ptr(c.pub, 3);
  c.g0p = (const double *)nvshmem_ptr(c.gpub, 0);
  c.g1p = (const double *)nvshmem_ptr(c.gpub, 1);
  c.g2p = (const double *)nvshmem_ptr(c.gpub, 2);
  c.g3p = (const double *)nvshmem_ptr(c.gpub, 3);
  if (!c.x1p || !c.x2p || !c.x3p || !c.g0p || !c.g1p || !c.g2p || !c.g3p) {
    fprintf(stderr, "rank %d: a peer is not nvshmem_ptr-accessible; this benchmark is single-NVLink-node only\n", rank);
    MPI_Abort(comm, 1);
  }

  CUDA_CHK(cudaMalloc(&c.wa, d * sizeof(double)));
  CUDA_CHK(cudaMalloc(&c.wb, d * sizeof(double)));
  hostinit = (double *)malloc(d * sizeof(double));
  for (long long i = 0; i < d; i++) hostinit[i] = 1.0;
  CUDA_CHK(cudaMemcpy(c.wa, hostinit, d * sizeof(double), cudaMemcpyHostToDevice));
  for (long long i = 0; i < d; i++) hostinit[i] = 2.0;
  CUDA_CHK(cudaMemcpy(c.wb, hostinit, d * sizeof(double), cudaMemcpyHostToDevice));
  free(hostinit);
  CUDA_CHK(cudaMalloc(&c.err, sizeof(unsigned long long)));
  CUDA_CHK(cudaMemset(c.err, 0, sizeof(unsigned long long)));
  CUDA_CHK(cudaMalloc(&c.cnt, NCNT * sizeof(unsigned int)));
  CUDA_CHK(cudaMemset(c.cnt, 0, NCNT * sizeof(unsigned int)));

  /* convergence-test state (allocated regardless; used only with -converge) */
  c.npub = (double *)nvshmem_malloc(2 * sizeof(double));
  if (!c.npub) {
    fprintf(stderr, "nvshmem_malloc(npub) failed\n");
    MPI_Abort(comm, 1);
  }
  for (int p = 0; p < 4; p++) c.npp[p] = (const double *)nvshmem_ptr(c.npub, p);
  CUDA_CHK(cudaMalloc(&c.rv, d * sizeof(double)));
  hostinit = (double *)malloc(d * sizeof(double));
  for (long long i = 0; i < d; i++) hostinit[i] = 1.0;
  CUDA_CHK(cudaMemcpy(c.rv, hostinit, d * sizeof(double), cudaMemcpyHostToDevice));
  free(hostinit);
  CUDA_CHK(cudaMalloc(&c.normacc, 2 * sizeof(double)));
  CUDA_CHK(cudaMemset(c.normacc, 0, 2 * sizeof(double)));
  CUDA_CHK(cudaMalloc(&c.dec, 2 * sizeof(int)));
  CUDA_CHK(cudaMemset(c.dec, 0, 2 * sizeof(int)));
  CUDA_CHK(cudaMalloc(&c.decepoch, 2 * sizeof(unsigned long long)));
  CUDA_CHK(cudaMemset(c.decepoch, 0, 2 * sizeof(unsigned long long)));
  CUDA_CHK(cudaMalloc(&c.stopiter, sizeof(long long)));
  CUDA_CHK(cudaMemcpy(c.stopiter, &stopped, sizeof(long long), cudaMemcpyHostToDevice)); /* -1 */

  /* cooperative launch: all blocks must be co-resident */
  CUDA_CHK(cudaGetDeviceProperties(&prop, local_rank % ndev));
  nsm = prop.multiProcessorCount;
  CUDA_CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, acgn_kernel, block.x, 0));
  {
    /* grid-stride loops handle any d, so cap the grid: grid.sync() cost grows with block
       count and dominated the first version at large d (283 us/iter at 1 MB with 264
       blocks). -blocks overrides the default cap of 32. */
    long long want = (d + block.x - 1) / block.x, cap = (long long)nsm * maxblk;
    if (cap > blkcap) cap = blkcap;
    grid = dim3((unsigned)(want < cap ? (want > 0 ? want : 1) : cap));
  }

  kargs[0] = &c;
  kargs[1] = &a0;
  kargs[2] = &a1;

  /* warmup (also settles first-touch of peer mappings), then timed region. Warmup runs
     with the convergence test OFF so rv starts undecayed when the clock starts. */
  {
    Ctx cw      = c;
    cw.converge = 0;
    kargs[0]    = &cw;
    if (persistent) {
      a0 = 0; a1 = warmup;
      CUDA_CHK(cudaLaunchCooperativeKernel((void *)acgn_kernel, grid, block, kargs));
    } else {
      for (long long it = 0; it < warmup; it++) {
        a0 = it; a1 = it + 1;
        CUDA_CHK(cudaLaunchCooperativeKernel((void *)acgn_kernel, grid, block, kargs));
      }
    }
    CUDA_CHK(cudaDeviceSynchronize());
    kargs[0] = &c;
  }
  CUDA_CHK(cudaDeviceSynchronize());
  MPI_Barrier(comm);
  t0 = MPI_Wtime();
  if (persistent) {
    a0 = warmup; a1 = warmup + iters;
    CUDA_CHK(cudaLaunchCooperativeKernel((void *)acgn_kernel, grid, block, kargs));
  } else {
    for (long long it = warmup; it < warmup + iters; it++) {
      a0 = it; a1 = it + 1;
      CUDA_CHK(cudaLaunchCooperativeKernel((void *)acgn_kernel, grid, block, kargs));
    }
  }
  CUDA_CHK(cudaDeviceSynchronize());
  t1 = MPI_Wtime();

  CUDA_CHK(cudaMemcpy(&stopped, c.stopiter, sizeof(stopped), cudaMemcpyDeviceToHost));
  nexec = (c.converge && stopped >= 0) ? stopped - warmup + 1 : iters;
  dt    = (t1 - t0) / (double)nexec;
  MPI_Reduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  CUDA_CHK(cudaMemcpy(&errs, c.err, sizeof(errs), cudaMemcpyDeviceToHost));
  MPI_Reduce(&errs, &allerrs, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
  if (rank == 0) {
    printf("mode=%-10s nvshmem-dev d=%-8lld msg=%7.0f KB  G=%-3d P=%-3d grid=%u per_iter=%9.2f us\n", c.ar ? (persistent ? "allred-pk" : "allred-dev") : (persistent ? "branch-pk" : "branch-dev"), d, d * sizeof(double) / 1024.0, G, P, grid.x, dtmax * 1e6);
    printf("validation: %s (%llu element mismatches over %lld iterations, all ranks)\n", allerrs ? "FAIL" : "PASS", allerrs, warmup + nexec);
    if (c.converge) {
      /* prediction: after n decays the global norm^2 is 4*d*rho^(2n); the kernel stops at
         the first CHECK (multiple of check_every) with that <= tol^2 */
      double nstar = ceil(log(c.tol2 / (4.0 * (double)d)) / (2.0 * log(c.rho)));
      if (nstar < 1) nstar = 1;
      npred = ((long long)nstar + c.check_every - 1) / c.check_every * c.check_every;
      printf("convergence: stopped after %lld iterations, predicted %lld: %s (tol=%g rho=%g check_every=%d)\n", nexec, npred, (stopped >= 0 && nexec == npred) ? "MATCH" : "MISMATCH", sqrt(c.tol2), c.rho, c.check_every);
    }
  }

  CUDA_CHK(cudaFree(c.err));
  CUDA_CHK(cudaFree(c.cnt));
  CUDA_CHK(cudaFree(c.rv));
  CUDA_CHK(cudaFree(c.normacc));
  CUDA_CHK(cudaFree(c.dec));
  CUDA_CHK(cudaFree(c.decepoch));
  CUDA_CHK(cudaFree(c.stopiter));
  nvshmem_free(c.npub);
  CUDA_CHK(cudaFree(c.wa));
  CUDA_CHK(cudaFree(c.wb));
  nvshmem_free(c.pub);
  nvshmem_free(c.gpub);
  nvshmem_free((void *)c.sig);
  nvshmem_free((void *)c.ack);
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
