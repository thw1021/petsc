# TaoTerm wishlist, driven by the ACGN/NVSHMEM measurements

Written 2026-08-14 alongside section 17 of `NVSHMEM-PERF-NOTES.md` (the `acgnbench.c`
measurement of the SC26 poster's four-node tomography splitting). TaoTerm development
happens on a different machine; this file records what the communication measurements say
the TaoTerm API needs so that the poster's decentralized schedules are expressible at all.
Each item cites the measurement that motivates it.

## 1. Split-phase evaluation APIs: `TaoTermGradientBegin()`/`TaoTermGradientEnd()`

The winning ACGN schedule sends g0,g1,g2 toward slot 3 *early* (right after the shard
gradients finish) but consumes them *late* (after the late gradient g3 at x2). That
early-send/late-consume edge is the DAG's only real overlap window, and it is
inexpressible in a single-call `TaoTermGradient()`: the call would have to either block
until the result is usable or hide the routing entirely. PetscSF got this right with
`Begin`/`End` pairs; TaoTerm needs the same split for gradient evaluation, and the same
applies to proximal outputs (`TaoTermProximalMap()` results are also routed: x2 -> ranks
1,3 in the worked example).

## 2. Gradient sinks/routing as a first-class concept (the H matrix)

Measured: branch-specific gradient routing beats a global gradient Allreduce by 22-30%
under plain GPU-aware MPI (perf notes section 17 table). Today the natural TaoTerm-sum
implementation implies global assembly of the summed gradient — exactly the barrier the
poster's splitting avoids. A term's partial gradient needs a way to declare *where it
goes* (which rank/buffer, with which weight), i.e. expose the ACGN H-matrix routing as a
sink specification instead of an implied reduction over the term sum. Dually, the K matrix
means "evaluate this term's gradient at a point received from another rank" — evaluation
points must be routable inputs, not assumed-local vectors.

## 3. Per-term device-context / stream association

The section 16.2/16.3 discriminator: a stream-ordered transport only pays off when there
is same-rank independent compute concurrently enqueued with the exchange. Terms that share
one stream serialize and destroy that overlap. A `TaoTermSetDeviceContext()` (or
equivalent) letting independent terms enqueue on distinct streams is what *creates* the
overlap windows; without it, the transport question is moot (section 17: NVSHMEM lost
1.38-1.56x precisely because the schedule had no overlap to exploit).

## 4. A capture-safety capability flag

The route with the largest headroom on one node (section 17, third lever; section 14.4)
is compiling the frozen iteration into a CUDA graph / persistent kernel with device-side
signaling. That requires every callback in the iteration to be capture-safe: no host
synchronization, no MPI calls, no allocation. TaoTerm should let an implementation declare
this (`TAOTERM_CAPTURE_SAFE` or a queryable capability alongside the existing
available-operations mechanism), so a schedule compiler can check eligibility instead of
discovering a host sync at capture time.

## 5. Sub-communicator / single-rank term placement with defined semantics

The poster's placement puts each prox on one rank (box on rank 1, rowTV on rank 2, colTV
on rank 3) while shard terms live on their data-owning ranks. TaoTerm needs term-per-GPU
placement to be well-defined: which calls are collective on which communicator, what the
non-hosting ranks do during a term's evaluation, and how a term hosted on a
subcommunicator participates in the sum. The motivation block on the poster ("varying GPU
requirements" per term) is this item.

## 6. What "PETSc really supporting NVSHMEM" ideally looks like: the device API, not a transport swap

Added 2026-08-15 after the route-2b measurements (perf notes section 23;
`nvshmem-tools/acgnbench-nvdev.cu`). PETSc's existing NVSHMEM support uses the *host*
API as one more CPU-driven transport under PetscSF -- and on that playing field it loses
to GPU-aware MPI (1.4-1.6x, section 17), because it competes on CPU-enqueue efficiency,
which is MPI's strength. The measured winning form is the *device* API: the frozen
iteration is a fused/persistent kernel, cross-GPU dependencies are
`nvshmemx_signal_op()`/`nvshmem_signal_wait_until()` issued inside the kernel, and data
moves as plain load/store through `nvshmem_ptr()` mappings -- 41 us/iter vs 162 for
branch-MPI at 64 KB, validated, with literally one kernel launch per solve in persistent
mode. Communication becomes memory access; no library call exists in the data path.

What TaoTerm/PETSc would need for this to be a supported execution mode rather than a
hand-written prototype (each already foreshadowed by items 1-5):

- **Symmetric-heap-resident vectors**: exchanged Vec/buffer allocations must come from
  `nvshmem_malloc()` (collective, same size on all PEs) so peers can map them --
  a Vec allocation mode, not a copy path.
- **Device-callable term kernels**: term gradient/prox bodies as `__device__` functions
  (or graph-node equivalents) so the schedule compiler can fuse them into the iteration
  kernel with signaling between them (item 4's capture-safety, one level deeper).
- **The frozen schedule as input**: signal/ack topology, double-buffer slots, and epoch
  values are all derivable mechanically from the DAG the co-design already produces --
  this is the "construct and schedule once -> freeze" doctrine compiled to the GPU.
- **Scope honesty**: on one NVLink node this is load/store PGAS (`nvshmem_ptr` non-NULL);
  across nodes the same device calls need NIC transports, which on Janus are blocked on
  `nvidia_peermem` (build notes 13.3). The execution model is portable; the fabric
  requirement is not waivable.

The protocol lessons from building it (arrival counters instead of grid barriers;
leader-per-block polling because 8k spinning threads self-jam the L2; reset-safety via
the ack chain) are recorded in perf notes section 23 and are directly reusable by any
future PETSc implementation.
