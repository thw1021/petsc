# Next steps for the ACGN/NVSHMEM poster experiments

**STATUS UPDATE (2026-08-15, 2-node job):** Routes 1 and 2a are DONE and measured
(perf notes sections 19-22); route 3's sanity ladder is DONE (build notes 13.3, perf notes
18, 21). Headlines: graph-replayed NCCL hits **32 us/iter at 64 KB** (5x vs branch-MPI,
beats the route-2a prediction); the route-1 threat test **fired** -- allreduce-NCCL matches
or beats branch-NCCL, so the splitting advantage is an MPI-cost-model statement, not a
volume statement; multi-node NVSHMEM is hard-blocked on `nvidia_peermem`/dmabuf (precise
ALCF ticket evidence recorded); 2-node NCCL works (`NCCL_IB_HCA=mlx5_bond_0` mandatory)
but its GDR-less staging is ~3 GB/s, so MPI stays the inter-node transport of record.
Route 2b is now ALSO built, measured, and optimized (`acgnbench-nvdev.cu`, perf notes
section 23): the device-resident execution model works and validates (zero mismatches;
persistent mode = 1 kernel launch per 300-iteration run). After barrier elimination
(per-slot block-arrival counters + leader-per-block polling; the all-thread-polling
middle step REGRESSED from L2 contention -- measured lesson): **41 us at 64 KB**, 4x vs
branch-MPI, 2.6x vs eager NCCL, 1.28x behind graph-NCCL (32). Zero grid-wide syncs per
iteration; realistic floor for this DAG ~high-20s us (serial depth x hop latency).

Written 2026-08-15, following the section 17 measurements in `NVSHMEM-PERF-NOTES.md`
(branch routing beats Allreduce by 22-30% under MPI; PetscSF-NVSHMEM loses 1.4-1.6x
everywhere in the ACGN DAG). Three routes, in increasing ambition. Numbers quoted from
section 17 unless noted; all predictions are labeled as predictions.

---

## 1. NCCL variant of the branch DAG (`acgnbench-nccl.c`)

### Layering (to prevent a misreading)

NCCL is NOT an orchestration/master layer and does not dispatch PETSc functions. It is a
byte-mover at exactly the level PetscSF's transport occupies today:

    master level:  the frozen ACGN schedule (solver loop; eventually TaoTerm) -- unchanged
    compute:       PETSc Vec/Mat CUDA ops -- unchanged
    edge transport: today PetscSF (-> MPI or NVSHMEM); route 1 = direct ncclSend/Recv
                    groups enqueued by the schedule, a THIRD arm replacing PetscSF for
                    the DAG edges only

It bypasses PetscSF (rather than becoming an SF backend) because (a) NCCL's win is fusing a
whole stage's edges into one group -- PetscSF's per-exchange Begin/End unit would forfeit
that; (b) SF Begin/End are collectively called, which forbids per-rank schedules (the
sfg2/sfg3 split compromise in acgnbench.c); (c) timeline. Longer term, a NCCL backend for
PetscSF is a plausible PETSc MR: SF knows the graph, Begin posts one fused group per
exchange, End degenerates to a stream/event wait (no host block). It would fuse within an
exchange but not across exchanges; the hand-rolled benchmark's numbers will show how much
of the win such a backend could preserve.

### The idea in pseudocode (one iteration, rank r's program)

    /* s = my PETSc compute stream, s_aux = side stream; host never blocks */
    if (r==1) box_prox(x1);                          /* PETSc, on s */
    ncclGroup { r==1 ? Send(x1,{0,2,3},s) : Recv(x1,1,s) }        /* 1 fused kernel */
    if (r<=2) g = grad_shard(x1);                    /* PETSc, on s */
    ncclGroupStart();                                /* branch routing */
      if (r<=1) { Send(g,2,s); Send(g,3,s); }
      if (r==2) { Recv(g0,0,s); Recv(g1,1,s); Send(g,3,s); }
      if (r==3) { Recv(g0,g1,g2 on s_aux); }         /* LATE lane */
    ncclGroupEnd();
    if (r==2) x2 = rowTV_prox(w2 + c*x1 - avg(g0,g1,g));  /* ordered by s */
    ncclGroup { r==2 ? Send(x2,{1,3},s) : (r==1||r==3) Recv(x2,2,s) }
    if (r==3) { g3 = grad_shard(x2);                 /* overlaps s_aux recvs */
                StreamWaitEvent(s, done(s_aux));     /* late consume, no host wait */
                x3 = colTV_prox(w3 + c*x1 - avg(g0,g1,g2) - g3); }
    ncclGroup { r==3 ? Send(x3,{1,2},s) : (r==1||r==2) Recv(x3,3,s) }
    if (r>=1) w -= theta * Lrow(r) . (x1,x2,x3);     /* PETSc, on s */
    /* graph-capturable as-is -> route 2a: replay = 1 launch/iter */

What PetscSF cannot express, visible above: per-rank programs containing only their own
edges; receive-early/consume-late as a side stream + one event wait; a host that never
blocks, making the body capture-eligible.

### Why it targets the measured bottleneck

The measured constraint (sections 14.4, 17): every stream-ordered operation costs ~2 us and
PetscSF's NVSHMEM protocol needs ~5 per exchange; MPI needs fewer ops but blocks the host
~8-11.5 us per exchange. You cannot make an operation cheaper -- only issue fewer. NCCL is
the only off-the-shelf transport that *fuses*: all `ncclSend`/`ncclRecv` calls inside one
`ncclGroupStart()`/`ncclGroupEnd()` are aggregated into (typically) a single kernel launch
per rank. A whole DAG stage becomes one stream op, and nothing blocks the host.

### Mapping the ACGN iteration

| DAG stage | edges | NCCL realization |
| --- | --- | --- |
| x1 distribution | 1->0, 1->2, 1->3 | one group (or `ncclBroadcast(root=1)`) |
| gradient routing | 0->2, 1->2, 0->3, 1->3, 2->3 | one group of 5 send/recv |
| x2 distribution | 2->1, 2->3 | one group |
| x3 distribution | 3->1, 3->2 | one group |

~4 fused launches per rank per iteration, zero host blocks.

Two structural bonuses over the PetscSF version:

- **P2P ops are not collective.** Only the two endpoints post a send/recv; other ranks'
  programs simply omit the edge. This lifts the artificial single-program-order constraint
  that forced `acgnbench.c` to split sfg2/sfg3 and place collective `End`s at compromise
  points. Each rank gets a true per-rank schedule -- the poster's "branch-specific" idea
  expressed literally.
- **Late consumption is free.** Enqueue rank 3's recvs on a side stream and make the
  accumulate kernel wait on an event; consumption timing becomes stream dataflow, not a
  host `End` call.

Also do the global-reduction arm with `ncclAllReduce` (stream-ordered, no
`cudaDeviceSynchronize`). This is a *threat test* for the poster story: if
branch-NCCL ~= allreduce-NCCL, the splitting advantage was mostly MPI's host-block, not
volume/barrier. Prediction: branch still wins, because the allreduce moves ~4x the needed
data, is a full barrier over all 4 ranks, and puts ranks 0/1 on the critical path; but it
must be measured, and reviewers will ask.

### Predictions (to falsify)

At 64 KB state: MPI exposes ~124 us/iter of comm+orchestration above the 39 us compute
floor. NCCL: 4 stages x (one ~2 us fused launch + small-message NVLink transfer ~2-5 us,
partially overlapped) -> predicted **~70-100 us/iter vs 163 (MPI)**, i.e. 1.6-2.3x. If it
lands below ~110 us it beats every measured cell.

### Practical notes

- `ncclSend/Recv` need NCCL >= 2.7; check `/soft/libraries/nccl` version. `aws-ofi-nccl`
  is only relevant for multi-node over libfabric (Slingshot); irrelevant intra-node.
- Send/recv pairs inside a group must match or the group deadlocks; the DAG is static, so
  audit once.
- Keep the compute side byte-identical to `acgnbench.c` (PETSc VecAXPY). NCCL must run on
  the same stream PETSc computes on (get it via `PetscDeviceContextGetStreamHandle()`), or
  bridge with events; otherwise the DAG ordering silently breaks.
- NCCL communicator init is a one-time cost (same amortization doctrine as NVSHMEM init).
- Validate correctness once per configuration by checksumming received buffers.
- Effort: ~1 day including validation.

---

## 2. Frozen-schedule device-side iteration

The co-design doctrine (companion note: obtain bounds once -> construct and schedule once
-> freeze) means the iteration is a *static program*. Two tiers of compiling it to the GPU:

### Tier (a): CUDA-graph capture of the stream-ordered iteration

Capture one iteration's kernels + NCCL groups into a graph; replay per iteration as one
graph launch (~5-10 us host cost, then GPU-side execution). NCCL supports graph capture
(>= 2.9). **NVSHMEM's host/on-stream APIs are documented as not capture-safe** (proxy
involvement) -- so tier (a) pairs with NCCL, not NVSHMEM. Expected: removes the per-op
launch tax from both compute (the 39 us floor at 64 KB is itself launch-bound: ~13-15
AXPY launches) and comm. Prediction: **~40-70 us/iter at 64 KB**, 2-4x vs branch-MPI.
Effort: ~1-2 days on top of route 1.

### Tier (b): persistent/fused kernels with NVSHMEM device API -- the real NVSHMEM story

Mechanism, exploiting that on one NVLink node every peer is load/store-accessible
(`nvshmem_ptr()` non-NULL, section 13.1):

- Fuse each rank's per-iteration work into one (or few) kernels. Cross-GPU dependencies
  become `nvshmemx_signal_op()` (producer) and `nvshmem_uint64_wait_until()` (consumer)
  on symmetric-heap flags -- issued *from inside* the running kernel, no launch cost.
- Data movement: consumer-side pull -- after the signal test passes, the consumer kernel
  reads the producer's buffer directly through `nvshmem_ptr()` with vectorized loads over
  NVLink (or producer push via `nvshmem_putmem_nbi`). No pack/unpack kernels, no
  quiet/fence stream ops.
- Iteration reuse: monotonic signal values (`wait_until(sig, NVSHMEM_CMP_GE, k)` for
  iteration k) avoid signal resets; double-buffer the exchanged vectors so iteration k+1's
  writes never race k's readers.
- Deadlock safety is provable from the frozen schedule: signals follow the DAG's
  topological order per iteration and the w-mixing edges close the loop across iterations;
  each GPU's kernels are enqueued in program order, so every wait's producer is already
  enqueued. (Persistent-kernel variant: one kernel loops over iterations with grid sync;
  simpler timeline, but occupancy-limits the compute -- start with the per-iteration
  fused-kernel version.)
- Build: device-side NVSHMEM calls need `-rdc=true` + `-dlink` against
  `libnvshmem_device.a` -- already solved in this branch (build notes section 5); for a
  standalone .cu prototype, invoke nvcc directly, no PETSc build involvement.

Prediction: per-iteration cost collapses toward actual DRAM/NVLink traffic plus a couple
of launches: **~10-30 us/iter at 64 KB vs 163 (MPI)**, i.e. 5-15x. These are the numbers
that make the poster's NVSHMEM panel work *and* they are honest: MPI structurally cannot
follow (host calls cannot be captured or fused into kernels), which is the claim
"GPU-resident signaling" was supposed to support all along.

Risks: correctness harness is mandatory (checksum state every N iterations in a debug
mode); spin-waits burn SMs if a producer is slow (fine here -- the DAG is tight); the
skeleton proves the execution model, then the real TV prox/gradient kernels replace AXPYs.
Effort: ~2-4 days for the skeleton prototype.

---

## 3. Multi-node, if ALCF ever loads `nvidia_peermem`: worth a try?

**Yes -- it is the regime NVSHMEM was designed for, and the only place the *existing*
PetscSF NVSHMEM path can plausibly beat MPI without new engineering.** But sequence it.

### Why the balance flips structurally (section 15's trade, at network latency)

    intra-node:  MPI host-block ~8 us   vs  NVSHMEM 5 stream ops ~10 us   -> MPI wins
    inter-node:  MPI host-block = real network round trip (~10-40 us for
                 small GPU messages, unhideable, per exchange)
                 vs  NVSHMEM device/stream ops: still ~10 us, overlapped   -> flips

    MPI, inter-node          HOST |pack|## BLOCKED on NIC round trip ##|unpack|
                             GPU  |pack|         (idle)               |unpack|

    NVSHMEM, inter-node      HOST |enqueue, returns|
                             GPU  |pack|put|next iteration's compute overlaps|

The ACGN DAG's weakness on one node -- exchanges on the critical path -- does not go away
multi-node; MPI just gets *worse* per hop while NVSHMEM's cost stays flat. Falsifiable
prediction: branch-NVSHMEM beats branch-MPI across nodes even in the current PetscSF form,
because the section 16 model's MPI term grows with network latency and the NVSHMEM term
does not.

### Cautions, in order

1. **Verify plain multi-node MPI first** (build notes section 13.2): this fabric already
   shows `ibv_create_ah` timeouts intra-node. If 2-node `MPI_Allreduce` on host buffers
   does not work, nothing downstream can be measured, including baselines.
2. **Fair-baseline coupling:** UCX's GPUDirect RDMA *also* needs `nvidia_peermem` (or
   dmabuf). Today's inter-node GPU-aware MPI would stage through host memory -- a strawman.
   When peermem loads, both MPI-GDR and NVSHMEM-IBRC light up; only then is the comparison
   fair.
3. **Transports:** IBRC (default with peermem) runs a host proxy thread -- the CPU-binding
   trap (section 10.1) gets *more* dangerous; give ranks dedicated cores. IBGDA
   (`NVSHMEM_IB_ENABLE_IBGDA=1`) moves NIC control to the GPU -- the structurally
   interesting arm for tier-(b) device-side iteration across nodes. Note: IBGDA moves the
   *control* path; the *data* path (NIC DMA into GPU memory) still needs a registration
   route (peermem or dmabuf -- kernel here is 5.14/el9, dmabuf plausible; verify, don't
   assume the section 13.2 "no peermem needed" reading).
4. **Problem mapping:** with 4 GPUs x N nodes, the natural instance is the poster's
   commented-out 4x4 figure: each ACGN role (shard group / prox) owns one node's 4 GPUs;
   intra-role traffic on NVLink, inter-role edges cross the fabric. That makes the
   placement variables (u_iv) real instead of fixed -- a genuinely richer co-design
   instance, not just a bigger run. Alternative: m >> 4 shards for sender-side overlap.

Bottom line: keep the ALCF ticket open; when it lands, budget half a day for the sanity
ladder (host MPI -> GPU MPI -> perftest inter-node -> acgnbench 2-node) before believing
any number.
