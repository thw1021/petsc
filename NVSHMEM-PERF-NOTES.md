# PetscSF: NVSHMEM vs GPU-aware MPI on Janus (H100, single node)

Measured 2026-08-14 on `x2003c0s9b0n0` (4x H100, NVLink `NV6` all-pairs, node vetted clean),
`PETSC_ARCH=arch-janus-nvshmem`, `--with-debugging=0`. Vehicle: `src/snes/tutorials/ex19`
(lid-driven cavity, DMDA halo exchange), `-dm_vec_type cuda -dm_mat_type aijcusparse
-pc_type jacobi -lidvelocity 10 -grashof 100`, 1 MPI rank per GPU.

Baseline arm is `-use_nvshmem 0`, which is **GPU-aware MPI** (`use_gpu_aware_mpi` defaults to
`PETSC_TRUE`, `src/sys/objects/init.c:37`) over HPC-X Open MPI 4.1.7a1. Both arms use the same
binary, same `UCX_TLS`, best-of-N runs.

## Headline

**GPU-aware MPI is faster than NVSHMEM for this workload at these sizes.** NVSHMEM has a lower
marginal cost per exchange but pays a large fixed startup cost that this workload never amortizes.

| quantity | GPU-aware MPI | NVSHMEM | note |
| --- | --- | --- | --- |
| marginal cost per halo exchange | 25.85 us | **19.79 us** | NVSHMEM 23% cheaper |
| fixed one-time overhead | ~0 | **~0.385 s** | NVSHMEM init |
| break-even | | **~64,000 exchanges** | 0.385 s / 6.06 us |

This workload performs 2.5k-17k exchanges, i.e. well below break-even, so MPI wins end to end:

| run (np=4) | exchanges | scatter time MPI | scatter time NVSHMEM | SNESSolve MPI | SNESSolve NVSHMEM |
| --- | --- | --- | --- | --- | --- |
| `-da_refine 5`                 |  2,480 | 0.384 s | 0.754 s | 0.963 s | 1.351 s (+40%) |
| `-da_refine 6`                 |  5,852 | 0.466 s | 0.815 s | 1.829 s | 2.207 s (+21%) |
| `-da_refine 6 -ksp_rtol 1e-10` | 16,650 | 0.745 s | 1.029 s | 4.561 s | 4.871 s (+6.8%) |

The relative penalty shrinks monotonically as work grows (+40% -> +21% -> +6.8%), exactly as a
fixed cost plus a marginal advantage predicts. Extrapolating, NVSHMEM overtakes MPI somewhere
around 64k exchanges — beyond anything measured here, so treat that number as an extrapolation
from two clean points, not an observation.

## Two traps that produce badly wrong answers

### 1. CPU binding — worth 10-30x, in the wrong direction

NVSHMEM runs a **spin-polling proxy thread**. Open MPI's default binding on this node gives each
rank exactly **one** core:

    $ mpirun -n 4 bash -c 'echo $OMPI_COMM_WORLD_RANK: $(taskset -cp $$)'
    rank 0 cores: 0     rank 1 cores: 32     rank 2 cores: 1     rank 3 cores: 33

so the proxy thread contends with the main thread for a single core. NVSHMEM warns about it:

    NVSHMEM WARN Proxy thread shares a core with the main PE, performance may be impacted

The damage is severe, and it lands on **compute** events, not communication — which makes it easy
to misdiagnose as an NVSHMEM communication problem when it is CPU starvation:

| event (np=4, `-da_refine 5`) | MPI | NVSHMEM, 1 core/rank | NVSHMEM, `--bind-to none` |
| --- | --- | --- | --- |
| `VecMDot`        | 0.210 s | 6.395 s (30x) | 0.145 s |
| `VecNorm`        | 0.164 s | 4.928 s (30x) | — |
| `KSPGMRESOrthog` | 0.274 s | 6.523 s (24x) | — |
| `MatMult`        | 0.245 s | 3.424 s (14x) | — |
| `SNESSolve`      | 1.069 s | 11.556 s (11x) | 1.351 s |

**Always run NVSHMEM with more than one core per rank** (`--bind-to none`, or `--map-by
slot:pe=N`). All numbers in this note use `--bind-to none` for *both* arms.

### 2. `VecScatterEnd` alone is a misleading metric

The two implementations put their work in different halves of the begin/end pair. NVSHMEM issues
the put and synchronizes in `Begin`; MPI defers the wait to `End`. Reading only `VecScatterEnd`
suggests NVSHMEM is 7.7x faster; summing both shows it is 1.75x slower:

| np=4, `-da_refine 6` | VecScatterBegin | VecScatterEnd | **Begin+End** |
| --- | --- | --- | --- |
| MPI     | 0.076 s | 0.390 s | **0.466 s** |
| NVSHMEM | 0.765 s | 0.051 s | **0.815 s** |

Always sum `VecScatterBegin + VecScatterEnd` (or `SFBcastBegin + SFBcastEnd`).

## Where the fixed cost lands

It appears inside `SNESFunctionEval` (0.305 s -> 0.702 s at `-da_refine 6`), because
`PetscNvshmemInitializeCheck()` is lazy — NVSHMEM is initialized on the first SF operation that
passes `PetscSFLinkNvshmemCheck()`, which is the first residual evaluation. Consistency check
across the three runs gives a stable fixed cost:

    gap = fixed - 6.06us * calls
    refine6 loose : 0.3496 = fixed - 6.06us*5852   -> fixed = 0.385 s
    refine6 tight : 0.2842 = fixed - 6.06us*16650  -> fixed = 0.385 s

Both solve for the same value, which is why the fixed/marginal split is trustworthy.

## Caveats

- **Single node only.** NVSHMEM's inter-node transport is unavailable here (IBRC skipped: no
  `nvidia_peermem`), so this measures NVLink/IPC intra-node only. NVSHMEM's design advantages are
  strongest for fine-grained inter-node traffic, which is precisely what cannot be tested here.
  **Do not generalize this result to multi-node.**
- One application (`ex19`), one preconditioner, one message-size regime (AvgLen ~3.1e3 B).
- The marginal-cost figures come from two points at fixed message size; the break-even estimate
  extrapolates well beyond the measured range.
- `-use_nvshmem_get 1` (get-based protocol) was verified correct but not separately profiled.
- The put-based protocol was used throughout (PETSc's default).

## 17. The SC26 poster's ACGN splitting DAG, measured (`acgnbench.c`)

Question: the poster's decentralized operator splitting (four-node tomography, branch-specific
gradient aggregation) avoids global reductions -- does that layout benefit from NVSHMEM?
`nvshmem-tools/acgnbench.c` reproduces the exact communication DAG of the companion note's
worked example on 4 ranks / 4 H100s: box prox on rank 1, x1 broadcast, parallel shard
gradients, branch routing (g0,g1 -> slot 2; g0,g1,g2 -> slot 3 consumed *after* the late g3),
rowTV/colTV prox chain, state mixing. Compute emulated by VecAXPYs (G per gradient, P per
prox). Aggregation is switchable: `-allreduce` (global reduction) vs default branch routing.
Measured 2026-08-14 on `x2003c0s9b0n0`, 300 iters, best of 3, `--bind-to none`, NVSHMEM
engagement verified by the banner discriminator.

Per-iteration time (us), G=8 P=4:

| msg | branch MPI | branch NV | allred MPI | allred NV | compute ref |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | **163** | 225 | 210 | 227 | 39 |
| 256 KB | **168** | 234 | 214 | 239 | 39 |
| 1 MB   | **219** | 320 | 264 | 292 | 44 |
| 4 MB   | **367** | 510 | 521 | 585 | 65 |

Heavier compute (G=32 P=16) does not change the picture; the NVSHMEM gap *widens*
(319 -> 460 us at 8 KB, 417 -> 651 us at 1 MB).

Two findings:

1. **The algorithmic claim is real and measurable today: branch-specific routing beats the
   global Allreduce by 22-30% under plain GPU-aware MPI.** This is the poster's splitting
   story, and it needs no NVSHMEM.
2. **The decentralized layout does NOT rescue NVSHMEM: 1.38-1.56x slower in every cell.**
   The reason is the section 16.3 discriminator: the ACGN schedule is a *dependency chain* --
   at the moment each exchange runs, the receiving rank is idle-waiting on it, so there is no
   same-rank independent compute to hide NVSHMEM's ~5 stream ops behind. The one overlap
   window the DAG has (g -> slot 3 sent early, consumed after g3) is exploited equally well
   by MPI's Isend/Irecv. "Decentralized" != "overlapped": the exchanges sit *on* the critical
   path, which is exactly the regime where MPI's lower per-exchange floor (~11.5 vs ~17 us)
   wins.

Routes that could still give NVSHMEM (or a stream-ordered transport) a win on this problem,
in increasing order of ambition:

- **Many shards per GPU** (m >> 4): each GPU computes several shard gradients and sends each
  while computing the next -- manufactures the section 16.2 overlap regime on the sender
  side. Expected gain is only the ~30-40% of *exposed* per-send cost, a few us per send.
- **NCCL for the edge set**: `ncclGroupStart/End` batches all sends/recvs of a DAG stage
  into one fused kernel -- attacking the same per-operation count that NVSHMEM's protocol
  pays 5 ops for, and it is stream-ordered (no host block) *and* CUDA-graph-capturable.
  Plausibly beats both current arms; needs a hand-rolled NCCL variant of acgnbench.
- **Frozen-schedule device-side execution**: the co-design already freezes coefficients and
  schedule before the solve; that is a license to compile the whole iteration into a CUDA
  graph (or persistent kernel) whose cross-GPU dependencies are NVSHMEM device-side
  puts/signal-waits -- no host involvement per iteration at all. This attacks the ~2 us
  per-stream-op floor itself (section 14.4 named graphs the only route past MPI on one
  node). MPI structurally cannot be captured. At 64 KB the ceiling is large: 163 us/iter
  (MPI) vs a ~39 us pipelined compute floor.

### Implications for the SC26 poster

- **Fill the 2x2 result matrix with the honest interaction, and make the NVSHMEM axis
  "GPU-orchestrated execution", not "PetscSF transport swap".** As designed, the NVSHMEM
  panels would show slowdowns if filled with the current PetscSF path. The honest story is
  *stronger* for the co-design thesis: a transport cannot fix a schedule -- splitting wins on
  its own (measured, 22-30%), NVSHMEM-as-transport-swap does not (measured, 1.4x slower),
  and only schedule+transport designed together (NCCL grouped stages, or the frozen-schedule
  device-side iteration) can beat MPI on this node. That interaction *is* the poster's
  claim.
- **Measurement protocol for the plots:** report per-iteration time or time-to-tolerance
  *excluding* the one-time ~0.385 s NVSHMEM init -- same doctrine as the companion note's
  "norm estimation is setup cost, not iteration-time work" (setup once, freeze, amortize).
  Always `--bind-to none` (section 10.1), always sum `Begin+End` (section 10.2), and verify
  NVSHMEM engagement with the `NVSHMEM_VERSION=1` banner discriminator (section 8).
- **Multi-node panels are out of reach on Janus** until `nvidia_peermem` is loaded (section
  13.2); worth an ALCF ticket, with IBGDA/IBDEVX as fallbacks. Within-node scaling means
  more shards per GPU, not more GPUs.
- The TaoTerm API changes these schedules require (split-phase gradient/prox evaluation,
  gradient sinks for H/K routing, per-term streams, capture-safety, per-rank placement) are
  recorded in `TAOTERM-WISHLIST.md` in this worktree, each tied to the measurement that
  motivates it.

## 18. Two nodes, measured (2026-08-15): the ACGN DAG across the fabric, MPI arms

First 2-node allocation (`x2002c0s9b0n0` + `x2003c0s1b0n0`, 4x H100 each, both vetted
clean). Multi-node NVSHMEM remains impossible (all transports blocked on peermem/dmabuf;
build notes section 13.3), so these are the GPU-aware-MPI arms only, host-staged over the
NIC (`UCX_TLS=sm,self,cuda_copy,cuda_ipc,rc`), ranks 0,1 on node 1 and ranks 2,3 on node 2
(`--map-by ppr:2:node`). Same binary and protocol as section 17; best of 3, 300 iters.

Same-day single-node control first: on today's node the section 17 numbers reproduce
within ~1% (161.8 vs 163, 209.1 vs 210, 216.3 vs 219, 366.8 vs 367, 523.6 vs 521 us) --
the measurement is stable across nodes and days.

Per-iteration time (us), G=8 P=4, MPI arms:

| msg | branch 1-node | branch 2-node | allred 1-node | allred 2-node | branch adv. 2-node |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | 162 | 206 | 209 | 259 | 21% |
| 512 KB | 189 | 297 | 234 | 332 | 10% |
| 1 MB   | 216 | 378 | 260 | 416 |  9% |
| 4 MB   | 367 | 886 | 524 | 942 |  6% |

Two observations:

1. **The splitting advantage survives the fabric but shrinks with size** (22-30% intra-node
   at all sizes -> 21% at 64 KB down to 6% at 4 MB inter-node). At small sizes the
   inter-node penalty is the ~19 us host-blocked NIC latency per crossing edge; branch
   routing crosses fewer edges than the allreduce and keeps its win. At large sizes both
   arms are bound by the same NIC bandwidth (~39 GB/s host-staged vs ~159 GB/s NVLink), and
   volume differences wash out in the shared bottleneck.
2. **The 2+2 placement puts 8 of the DAG's 12 edges across the fabric** (x1: 1->2, 1->3;
   g: 0->2, 1->2, 0->3, 1->3; x2: 2->1; x3: 3->1). A placement-aware
   mapping (the poster's u_iv variables) would hide more; this is the co-design lever the
   4xN instance makes real.

## 19. Route 1 executed: NCCL edge transport (`acgnbench-nccl.c`), single node

Measured 2026-08-15 on `x2002c0s9b0n0`, NCCL 2.28.3 (cuda12.9 build from `/soft`), same
DAG/compute/protocol as section 17, every DAG stage one fused `ncclGroupStart/End` of
`ncclSend/ncclRecv` on PETSc's compute stream (`PetscDeviceContextGetStreamHandle()`).
Correctness: checksums on every received lane, all PASS. Also new: `-monolithic` arm in
`acgnbench.c` -- the "dumb single-core" baseline, rank 0 does the whole iteration's compute
serially (4G+2P+12 AXPYs), zero communication.

Per-iteration (us), G=8 P=4, best of 3 (spread < 2%):

| msg | monolithic | branch MPI | allred MPI | branch NCCL | allred NCCL | compute ref |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | 118 | 162 | 209 | **107** | 109 | 39 |
| 512 KB | 128 | 189 | 234 | 149 | **144** | 42 |
| 1 MB   | 133 | 216 | 260 | 177 | **169** | 43 |
| 4 MB   | 195 | 367 | 524 | 363 | **348** | 65 |

Heavy compute (G=32 P=16): monolithic 369/423/450/642; branch NCCL 273/329/364/636;
allred NCCL 274/324/358/624; compute ref 122/131/134/203.

Three findings:

1. **NCCL beats MPI, but less than predicted eagerly** (107 vs the predicted 70-100 at
   64 KB; -34% vs branch MPI, converging to a tie at 4 MB where both are NVLink-bound).
2. **The threat test fired: under NCCL, branch and allreduce essentially tie.** Branch wins
   by ~2% at 64 KB; allreduce wins by 3-8% at every larger size, light or heavy compute,
   eager or graphed. The section 17 "branch beats allreduce by 22-30%" result is therefore
   a statement about *MPI's* cost model -- every exchange costs a host block, and the
   allreduce is the most exchange- and barrier-heavy pattern -- not about data volume.
   NCCL removes the host block entirely, and `ncclAllReduce` is one bandwidth-optimal ring
   driving all NVLink directions concurrently, while the branch DAG is four
   *dependency-serialized* fused groups; at np=4 the ring's extra volume (~1.5x buffer per
   GPU) costs less than the branch's stage serialization. Qualifiers before declaring
   branching dead: (a) this is np=4 on flat all-to-all NVLink -- at scale, on
   hierarchical/limited topologies, and inter-node under MPI (section 18: branch wins
   21% -> 6%), the volume and locality arguments return; (b) the allreduce arm is a
   performance *proxy*, not the algorithm -- ACGN's slots want different aggregations, and
   a global sum only emulates the cost of the naive design, so branch routing keeps its
   semantic and placement (co-design) value even where its raw transport advantage
   vanishes. Poster phrasing: splitting wins *given* the transport the ecosystem actually
   uses (MPI); co-designed transports flatten the routing difference and beat both.
3. **The monolithic baseline is only honest with the compute-weight axis attached.** At
   G=8 P=4 the serial no-comm version beats every parallel MPI arm at >= 512 KB (its 52
   launch-bound AXPYs cost less than the parallel arms' communication). At G=32 P=16 it
   loses ~1.4x to the parallel arms at small/medium sizes. The crossover -- "splitting pays
   once per-term compute exceeds the cost of routing" -- is itself the co-design statement.

## 20. Route 2a executed: CUDA-graph replay (`acgnbench-nccl-graph.c`) -- the headline

Same iteration body, but after an eager 30-iter warmup the whole body (VecAXPYs + NCCL
groups) is stream-captured once and the timed loop is `cudaGraphLaunch` replays
(`-graph_iters k` iterations per graph; k=1 and k=10 within ~4% of each other, so the
per-op launch tax, not the graph-launch count, was the enemy). Validation PASS everywhere.

What "eager" vs "graph replay" means -- same kernels, different host behavior:

    EAGER (acgnbench-nccl.c): the host re-issues every operation, every iteration

     HOST  |axpy|axpy|..|grp1|axpy|..|grp2|axpy|..|grp3|..|grp4|axpy|axpy|   ~18 enqueues/iter
           ~2us CPU cost each, every iteration; never blocks, never stops enqueueing
     GPU      |axpy|axpy|..|NCCL k1|axpy|..|NCCL k2| ...                     executes behind

    A "fused group" = one ncclGroupStart/End wrapping a DAG stage's sends/recvs; NCCL
    aggregates them into ONE kernel per rank (4 comm launches/iter instead of 12 edge ops).

    GRAPH REPLAY (acgnbench-nccl-graph.c): record the body once, then

     HOST  |launch|                            |launch|                      1 enqueue / k iters
     GPU   |axpy axpy .. NCCL1 axpy .. NCCL2 .. NCCL4 axpy axpy| (replay)    same kernels, no
            per-op gap: the GPU walks the pre-built dependency graph itself

The warmup runs eagerly first (connects all NCCL channels, settles PETSc lazy init), then
`cudaStreamBeginCapture` .. one iteration's calls .. `EndCapture` records the whole
stream-ordered sequence -- AXPY kernels *and* NCCL kernels -- into a `cudaGraph_t`,
instantiated once. Nothing about the kernels changes; what dies is the ~2 us-per-operation
CPU enqueue cost x ~18 ops. That is why 107 -> 32 us, and why the 39 us "compute floor"
also fell: that floor was itself mostly enqueue tax.

**Prerequisite discovered: PETSc's default device-context stream is the legacy NULL stream
(`PETSC_STREAM_DEFAULT` destroys the `CUPMStream` and hands out `NULL`), which
`cudaStreamBeginCapture()` rejects with `cudaErrorStreamCaptureUnsupported`.** Two
equivalent fixes (both verified): the command line `-root_device_context_stream_type
nonblocking` (the root context parses the `device_context_stream_type` key under the
`root_` prefix, `device.cxx:697`), or programmatically retype the global context to
`PETSC_STREAM_NONBLOCKING` right after `PetscInitialize()` and call
`PetscDeviceContextSetCurrentContext()` again so `PetscDefaultCudaStream` re-syncs (what
`acgnbench-nccl-graph.c` does, so it cannot be run wrong). This is the first concrete
capture-safety item for the TaoTerm wishlist: any future capture-based execution path in
PETSc needs a non-NULL stream type as a precondition.

Per-iteration (us), G=8 P=4, k=10, best of 2:

| msg | branch graph | allred graph | best eager NCCL | branch MPI | compute ref (eager) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | **32** | 34 | 107 | 162 | 39 |
| 512 KB | 68 | **63** | 144 | 189 | 42 |
| 1 MB   | 91 | **83** | 169 | 216 | 43 |
| 4 MB   | 227 | **211** | 348 | 367 | 65 |

- **5.0x vs branch-MPI at 64 KB, and below the eager compute floor** (32 < 39): the
  39 us "compute reference" was itself ~14 launch-bound AXPY launches; replay erases the
  launch tax for compute and comm alike, leaving ~20 us of AXPY execution + ~12 us of NCCL
  kernels.
- Heavy compute, 64 KB: **32 us** vs 273 eager branch-NCCL, 369 monolithic -- when
  everything is launch-bound, the graph win compounds: 8-11x against the eager arms.
- The route 2 prediction (~40-70 us at 64 KB) is beaten. The remaining gap to pure
  transfer time is AXPY execution serialization -- attacking that is route 2b (fused
  kernels), which is now about the last ~20 us, not the first 130.
- Allreduce-vs-branch under graphs: same story as eager NCCL (finding 2 above); branch
  wins only at 64 KB, by 2 us.

### Does graph replay require knowing the iteration count? (Reviewer question; no.)

The graph is the **fixed-point map, not the solve**: it records one application of T in
x_{k+1} = T(x_k), and the host decides at runtime how many times to replay it. What
capture requires is that T itself is data-independent -- fixed steps, fixed prox
parameters, fixed schedule, no line search or adaptive restart. That is exactly the
co-design doctrine ("obtain bounds once -> construct and schedule once -> freeze"): the
freeze is the *mathematical license* to compile the iteration, not just a performance
trick. The only data-dependent piece is the stopping test; the standard resolution is
chunked replay -- launch k iterations per graph, compute the residual inside the graph
(a fixed computation), read-and-decide on the host once per chunk. Cost: one sync per k
iterations plus overshoot of at most k-1 iterations, which is harmless for a
nonexpansive map (extra applications of T do not corrupt the iterate, they waste at most
k-1 iterations of work). Removable entirely with CUDA >= 12.4 conditional graph nodes
(device-evaluated while-loops), or in the device-NVSHMEM persistent kernel by computing
the norm in-kernel, allreducing it with the section 23 machinery, and breaking all PEs'
loops on the same signaled epoch -- the stopping test becomes one more edge of the
frozen protocol. `-graph_iters k` in `acgnbench-nccl-graph.c` is precisely this chunked
structure (k=1 and k=10 measured within ~4%).

## 21. Two-node NCCL: works, one mandatory knob, honest numbers are poor without GDR

`NCCL_IB_HCA=mlx5_bond_0` is **mandatory** on this fabric: by default NCCL enumerates the
raw bond-slave devices (`mlx5_0`, `mlx5_1`) alongside the bond and lands on dead routes --
**78,000+ us/iter of retransmit timeouts** instead of ~300. (Same family of fabric quirk as
the UCX `ud` failure in build notes section 13.3.) With the knob, 2+2 across nodes:

| msg | branch MPI | allred MPI | branch NCCL | allred NCCL |
| ---: | ---: | ---: | ---: | ---: |
| 64 KB  | **206** | 259 | 296 | 290 |
| 512 KB | **297** | 332 | 1542 | 1183 |
| 1 MB   | **378** | 416 | 3005 | 2268 |
| 4 MB   | **886** | 942 | 11709 | 8802 |

The right reading is three separate layers, not "NCCL is broken":

1. **A config trap (fixed):** default HCA enumeration includes the raw bond slaves and
   lands on dead routes -> the 78 ms pathology. `NCCL_IB_HCA=mlx5_bond_0` cures it
   completely. Same family of fabric quirk as the UCX `ud` failure (build notes 13.3).
2. **Small messages are fine:** 290-296 us vs MPI's 206 at 64 KB -- only 1.4x worse, with
   the launch-tax share removable by graph replay (280 us).
3. **Bulk bandwidth is the GDR-less fallback path:** without `nvidia_peermem` the NIC
   cannot DMA GPU memory, so NCCL's proxy stages through fixed host bounce buffers,
   reaching ~2.7 GB/s aggregate (32 MB of crossing edges / 11.7 ms at 4 MB), vs UCX's
   pipelined host staging at ~39 GB/s over the same wire.

Two side observations: graph replay works across nodes too (280 us vs 296 eager at 64 KB)
but only shaves the launch tax -- the NIC path dominates; and allreduce-NCCL *beats*
branch-NCCL inter-node at bandwidth sizes (8.8 vs 11.7 ms at 4 MB) because the ring
pipelines through the bond in both directions while the branch DAG serializes bulk edges.

### Can layer 3 be fixed without root? No -- measured, not assumed

The wire is provably capable (UCX does 39 GB/s host-staged), so the hypothesis "NCCL's
staging pipeline is just under-tuned" was worth testing. Every plausible user-space lever
was tried at the 4 MB size; the result is *completely flat*:

| attempt | per_iter |
| --- | ---: |
| baseline | 11705 us |
| `NCCL_IB_QPS_PER_CONNECTION=4` + `NCCL_IB_SPLIT_DATA_ON_QPS=1` | 11737 |
| `NCCL_BUFFSIZE=16M` | 11720 |
| `NCCL_MIN/MAX_NCHANNELS=16` | 11707 |
| `NCCL_P2P_NET_CHUNKSIZE=4M` | 11801 |
| all of the above combined | 11869 |
| HPC-X `nccl_rdma_sharp_plugin` (loads after extracting `libibumad.so.3` from `/soft/repos/hpe-doca-ofed-rhel9.4/` into `LD_LIBRARY_PATH`) | unchanged |

Knob-invariance at this level means the limiter is inside the proxy's bounce-buffer
protocol itself, not its parallelism parameters. The remaining routes all need root:
`modprobe nvidia_peermem` (the direct fix), or switching the nodes to the open-source
NVIDIA kernel modules for dmabuf (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0 on the
current stack). A newer NCCL cannot help -- 2.28.3 is already the newest installed, and no
NCCL version can register GPU memory with the NIC without one of those kernel-side pieces.

### Did the admin "miss" peermem? Almost certainly, and the ticket writes itself

The module **ships with the exact driver installed on these nodes** (`modinfo
nvidia_peermem` -> version 610.57.04, build notes 13.2); it is the standard GPUDirect RDMA
component for any IB/RoCE GPU cluster; it is simply not loaded (`lsmod` shows only
`nvidia`, `nvidia_uvm`) and not persisted in any `modules-load.d`. On H100 nodes with
bonded RoCE NICs that is an oversight or a deferral, not a design decision. One
`modprobe nvidia_peermem` (plus a one-line persist) unlocks three things at once:

    1. NVSHMEM multi-node -- ALL of its transports are gated on it (build notes 13.3)
    2. NCCL's inter-node fast path (this section's layer 3)
    3. GPUDirect RDMA for UCX/MPI itself -- today's 39 GB/s is host-staged; GDR removes
       the staging copies and the ~19 us host-blocked small-message floor drops too

**Verdict: inter-node, GPU-aware MPI remains the transport of record on this machine until
`nvidia_peermem` is loaded; every GPU-orchestrated transport (NVSHMEM entirely, NCCL's
data path) is gated on the same kernel module. The ALCF ticket is the single unlock.**

## 22. The poster's 2x2 matrix, with today's numbers (single node, 64 KB, G=8 P=4)

    monolithic serial (dumb baseline)      118 us
    term-per-node + Allreduce (MPI)        209 us   <- parallelism alone LOSES at light compute
    branch routing (MPI)                   162 us   <- splitting recovers it, still above serial
    branch routing (NCCL eager)            107 us   <- co-designed transport starts winning
    branch/allred + CUDA-graph replay       32 us   <- schedule+transport+execution co-design: 3.7x vs serial,
                                                       5x vs branch-MPI, at the compute floor

    At heavy compute (G=32 P=16): serial 369, branch-MPI ~320 (sec 17), graph 32 -- the
    ordering is strict and the co-design win is 10x.

## 23. Route 2b executed: device-side NVSHMEM skeleton (`acgnbench-nvdev.cu`)

Measured 2026-08-15, same node/protocol. Standalone MPI+NVSHMEM `.cu` (no PETSc, no
NCCL): each rank's whole ACGN iteration is ONE fused cooperative kernel (or one persistent
kernel for the entire run, `-persistent`). Cross-GPU dependencies are
`nvshmemx_signal_op()` / local signal-word polling inside the running kernel; data moves
by consumer-side pull through `nvshmem_ptr()` NVLink loads. Monotonic epoch signals
(`v = it+1`, `CMP_GE`), double-buffered slots (`it & 1`), reverse ack signals gate slot
reuse at depth 2. Race detection is built in: every published element is stamped
`role_base + v` and consumers validate every element, every iteration -- a missing fence
or stale slot is a counted error, not silence. **Zero mismatches across the whole matrix,
~100M validated element-reads per run.**

Per-iteration (us), G=8 P=4, best of 3, best block cap (32 at 64 KB, 64 above), final
(v4) protocol. The `-allreduce` arm is a **direct all-pull allreduce**: every PE
publishes its stamped contribution and pulls the other three lanes (one signal round, no
ring steps -- latency-optimal at np=4):

| msg | branch-dev (1 launch/iter) | branch-pk (1 launch/RUN) | allred-dev | allred-pk | graph NCCL | eager NCCL | branch MPI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 KB  | 48 | **41** | 44 | 42 | 32 | 107 | 162 |
| 512 KB | 92 | 88 | 86 | **76** | 63-68 | 144 | 189 |
| 1 MB   | 133 | 122 | 118 | **108** | 83-91 | 169 | 216 |
| 4 MB   | 387 | 382 | 326 | **316** | 211-227 | 348 | 363 |

The section 19 finding replicates on a third transport: allreduce ties branch at 64 KB
and *wins* above it (11-17%), despite moving ~2.4x the lane volume. The mechanism is
visible in the DAG: the all-pull g-phase lets rank 3 fetch its gradient lanes early and
concurrently with rank 2's pulls, taking the branch schedule's late-consume reads OFF the
critical path, while branch routing keeps them on it. Three transports now agree: the
splitting advantage is specific to host-blocking (MPI-style) cost models.

Heavy compute (G=32 P=16, 64 KB): 48 us -- compute inside the fused kernel is nearly free
(comm-only G=0 matches the full run), exactly the tier-b thesis.

### The intra-kernel synchronization protocol took four revisions; each step was measured

These are in-place revisions of the same file during one session; only v4 survives on
disk (v2's numbers are the untagged rows in `results-20260815-nvdev.txt`, v4's are tagged).

| protocol version | 64 KB persistent | lesson |
| --- | ---: | --- |
| v1: thread-0 waits + full `grid.sync()` after every phase (~10/iter), grid sized to d | 68-71 | grid barriers cost ~1-3 us each and scale with block count (264 blocks -> 283 us at 1 MB) |
| v2: all-thread polling for waits + grid capped at 32-64 blocks | 54 | wait-side barriers gone; publish/ack grid syncs remain |
| v3: barrier-free arrival counters, but ALL threads spin-poll signal and ack words | 74-75 | **worse than v2** -- 8192 threads hammering one L2 word starve the very L2 that serves incoming signals and peers' NVLink pulls |
| v4: arrival counters + one leader poller per block + `__syncthreads()` release | **41** | barrier-free only wins with contention-free polling: 32 pollers, not 8192 |

How each revision worked, and what its measurement forced next:

**v1 (conservative-correct).** Every dependency point was the textbook pattern: thread 0
of the grid performs `nvshmem_signal_wait_until()`, then `grid.sync()` broadcasts the
result; every publish was writes -> `__threadfence_system()` -> `grid.sync()` -> thread 0
signals. ~10 grid barriers per iteration, grid sized to the vector. It validated
immediately (68 us @ 64 KB) but two profile facts condemned it: comm-only (G=0) cost the
same as the full iteration -- the protocol, not compute or data, was the entire bill --
and 1 MB ran at 283 us because 264-block `grid.sync()`s are expensive. Persistent mode
was no faster than per-iteration launches, proving launch overhead was not the term that
mattered.

**v2 (wait-side barriers removed, grid capped).** Waits became all-thread polling of the
local signal word (each thread gates itself; the remote `signal_op` write lands in local
memory, so polling needs no NVSHMEM call), and the grid was capped at 32-64 blocks with
grid-stride loops covering any d. 54 us @ 64 KB, 112 @ 512 KB. Publish/ack still carried
~5 grid barriers. This was the version first reported as "route 2b done".

**v3 (publish/ack barriers removed -- the regression).** Grid syncs were replaced by
per-site-per-slot block-arrival counters (mechanism below), and ack-waits also became
all-thread polls: zero grid-wide synchronizations. It stayed correct (validation PASS)
but got SLOWER than v2 (74-75 us): with ~8192 threads spin-loading the same 8-byte
signal words, the consumer GPU's L2 is saturated by its own polling -- the same L2 that
must serve the incoming NVLink signal write and the peers' remote data pulls. The
barrier was gone but the detection path was self-jammed.

**v4 (contention-free polling).** Same barrier-free structure, but every poll (signal
waits and ack waits) is done by thread 0 of each block, releasing its block through
`__syncthreads()` -- 32 pollers instead of 8192, cross-block decoupling preserved.
41 us @ 64 KB, 88 @ 512 KB, 122 @ 1 MB.

v4's publish/ack protocol: each block fences (`__threadfence_system`) and bumps a local
per-site-per-slot atomic; the LAST block to arrive resets the counter and fires the
`nvshmemx_signal_op`s; all other blocks flow on immediately and self-gate at their next
signal wait. The reset-before-signal is provably safe: no block can re-enter a site+slot
(iteration it+2) before a consumer acks iteration it, and that ack requires the very
signal that follows the reset. There are ZERO grid-wide synchronizations per iteration --
blocks of one PE may run skewed across iterations, coupled only by the dataflow. The
epoch-stamped per-element validation ran through every revision, which is what made this
loop safe to iterate quickly: a fencing or reset mistake would have been a counted error,
not a silently wrong benchmark.

Findings:

1. **The execution model is proven.** A frozen schedule really can run as a device-side
   program: in persistent mode the host launches ONE kernel for 300 iterations -- zero
   host involvement per iteration, which no MPI-based path can structurally match. All
   dataflow, ordering, flow control, and validation live on the GPUs.
2. **It beats every eager arm** (4x vs branch-MPI, 2.6x vs eager NCCL at 64 KB) **and
   lands 1.28x behind graph-NCCL** (41 vs 32 us). The 10-30 us prediction was just
   missed: 41 us sits at its edge, and the remaining cost decomposes as the DAG's serial
   depth (~7 signal-hop segments on rank 3's critical path) x ~3-5 us per hop (NVLink
   signal write + poll detect + remote pull + arrival accounting) plus ~7 us compute --
   the realistic floor for THIS DAG on this fabric is high-20s us, i.e. the prediction's
   upper edge was about right and its lower edge was optimistic about hop latency.
3. **Remaining (unbuilt) headroom:** merge consume-validation into the following compute
   loop; per-warp specialization so the leader poller overlaps the consume reads;
   deferring off-critical-path acks (e.g. rank 3's x1 ack). Each is a ~2-4 us-class item;
   none changes the structural picture.
4. **What this arm uniquely demonstrates for the poster:** the co-design ladder's last
   rung is *feasible and correct* (device-resident iteration, validated), it is the only
   arm with literally zero per-iteration host cost, and it is the only route available
   when kernels must react to data mid-flight (graphs replay a fixed recording; NVSHMEM
   host APIs are not capture-safe -- device APIs go *inside* the kernel instead). As pure
   single-node throughput, graph-NCCL wins today.

Build is the notes-section-5 three-step recipe (`-rdc=true`, `-dlink` against
`libnvshmem_device.a`, host link); raw numbers in
`nvshmem-tools/results-20260815-nvdev.txt`.

## 24. The full measured configuration space, in one table

The DOF structure of everything measured on the ACGN DAG:

- **comm layer** (4): GPU-aware MPI; PetscSF-NVSHMEM (NVSHMEM host/stream API under
  PetscSF); NCCL fused groups; NVSHMEM device API inside fused kernels. (There is no
  "NCCL+NVSHMEM" arm -- the fourth layer replaces NCCL, it does not combine with it.)
- **algorithm** (2): global allreduce vs branch routing. Plus two baselines outside the
  axis: monolithic-serial (no comm) and compute-only floor.
- **execution model** (2 where it exists): eager (host re-enqueues every operation each
  iteration) vs replay/fused (graph replay for NCCL; per-iteration-launch vs persistent
  single-launch for the device arm). For MPI this axis is STRUCTURALLY EMPTY (blocking
  host calls cannot be captured); for PetscSF-NVSHMEM likewise (host API is not
  capture-safe, and its `<<<1,1>>>` spin kernels would deadlock a single fused kernel).
  Terminology: graph replay is ONE LAUNCH of a recorded multi-kernel sequence (kernels
  stay distinct); true single-kernel fusion is what the device arm does.
- **environment**: message size (64 KB-4 MB), compute weight (G8P4 / G32P16), placement
  (1 node; 2+2 across nodes for MPI and NCCL).

Master table, 64 KB, G=8 P=4, single node, us/iter (best-of-N; sources: sections 17-23):

| comm layer \ algo            | allreduce | branch | execution |
| --- | ---: | ---: | --- |
| GPU-aware MPI                | 209 | 162 | eager only (capture impossible) |
| PetscSF-NVSHMEM              | 227 | 225 | eager only (not capture-safe); 08-14 numbers |
| NCCL, eager                  | 109 | 107 | |
| NCCL, graph replay (k=10)    | 34  | **32** | |
| NVSHMEM device, 1 launch/iter| 44 | 48 | direct all-pull allreduce |
| NVSHMEM device, persistent   | 42 | **41** | |
| monolithic serial (baseline) | \-- 118 -- | | no comm at all |
| compute-only floor           | \-- 39 -- | | eager launches; the graph arms beat it |

Cell status is part of the result: the two "eager only" rows are *structural* absences
(and exactly the poster's point -- host-blocking transports cannot enter the replay/fused
regime). The device-side allreduce cells were filled after this section was first
written (direct all-pull; section 23); as the NCCL rows predicted, allreduce ties branch
at 64 KB -- and above 64 KB it wins 11-17%, because all-pull takes the late-consume reads
off the critical path. The matrix is now complete: every buildable cell is measured, and
every empty cell is structurally empty.

### How much of each arm is PETSc -- the deliverable ladder

The rows are NOT equally "PETSc results", and honesty about that is part of the story:

| rows | compute | transport | status |
| --- | --- | --- | --- |
| MPI, PetscSF-NVSHMEM | PETSc `VecAXPY` | **PetscSF** | in-tree today; no in-tree config beats 162 us |
| NCCL eager + graph | PETSc `VecAXPY` | hand-rolled NCCL, **bypasses PetscSF** | PETSc compute, foreign transport |
| NVSHMEM device | raw CUDA loops | NVSHMEM device API | PETSc-less showcase / upper bound |

The NCCL bypass is deliberate: SF's collectively-called per-exchange `Begin`/`End` cannot
express stage-fused groups or per-rank edge programs (the layering note in
`NVSHMEM-NEXT-STEPS.md`); an SF-NCCL backend MR could fuse within an exchange only, so it
would preserve part of the 107 us eager number and none of the cross-op structure. The
graph rows carry one genuinely PETSc-positive fact: **capture over real PETSc operations
works** -- the 32 us graph replays genuine `VecAXPY`s -- given the nonblocking-stream
precondition (section 20) and no host syncs in the captured region; making that a
supported pattern rather than folklore is wishlist item 4. The device arm's path from
showcase to deliverable is wishlist item 6. Deliverable ladder, in order of effort:
(1) this branch's build fixes upstream; (2) `sfnvshmem.cu` op fusion (section 13.1);
(3) SF-NCCL backend with partial fusion; (4) capture-safety support; (5) the device-API
execution mode.

Same table at other sizes/weights: section 19 (NCCL eager, monolithic), 20 (graph), 23
(device), 17 (MPI, PetscSF-NVSHMEM); 2-node variants in sections 18 and 21.

## 25. The sfnvshmem.cu op fusion, implemented and measured (2026-08-15, second job)

TODO item 7 executed: the section 14.4 fusion lever is now real code, behind
`-use_nvshmem_putsig` (default TRUE; 0 restores the legacy protocol for A/B).

**The change** (three edits, all local to the put protocol):

1. `PetscSFLinkPutDataBegin_NVSHMEM()`: for locally accessible PEs,
   `nvshmemx_putmem_nbi_on_stream()` becomes `nvshmemx_putmem_signal_nbi_on_stream()` --
   the receiver's arrival signal (`RecvSig` at its `sigdisp` offset) is delivered *with
   the data*, ordered after it by API contract.
2. The `nvshmemx_quiet_on_stream()` workaround (line ~590, "fence does not fence
   on_stream puts") is skipped -- its only job was ordering the puts before the
   End-time signals, which no longer exist for local PEs. One stream op removed.
3. `PutDataEnd<<<1,1>>>` signals only remotely-accessible dst ranks (an `nvshmem_ptr()`
   test in the kernel); its wait half is unchanged. The remote path's IB-ordering
   argument is untouched.

**Correctness:** `sf/tests/ex22` FetchAndOp at np=2,4 passes both protocols;
`snes/tutorials/ex19` (-da_refine 4, 3 Newton steps) is byte-identical to the
`-use_nvshmem 0` MPI baseline under both fused and legacy. (The first ex19 comparison
was vacuously "identical" -- the binary had failed to build for a missing `-lm` and all
outputs were empty. Check your file sizes before trusting a diff.)

**Measured, np=4, `--bind-to none`:**

| instrument | legacy | fused | delta |
| --- | ---: | ---: | --- |
| `sfbench` ring floor, 64 B-128 KB | 18.5-20.1 us | 17.4-19.0 us | ~-1.1-1.6 us (~7%), never worse |
| `acgnbench` branch DAG, 64 KB | 227-231 us/iter | **205-207** | **-9.5%** |
| `acgnbench` branch DAG, 1 MB | 320-325 us/iter | **292-296** | **-9%** |

The DAG gains ~4 us per exchange -- MORE than the ~1.4 us the ring microbenchmark shows,
and the difference is the second, subtler effect: the arrival signal now lands at
*Begin* time (with the data) instead of at the sender's *End* time, so a receiver
blocked on the critical path unblocks as soon as bytes arrive. A back-to-back
microbenchmark cannot see this latency term; a dependency chain feels it on every edge.

Follow-up cells (same protocol A/B on the other arms): allreduce-NVSHMEM 229 -> 219 at
64 KB, 292 -> 264 at 1 MB (-4.5%/-9.5%); the Reduce (LEAF2ROOT) direction runs correct
at parity within noise; the get protocol is unaffected by the new default. An in-tree
harness test now exists (`sf/tests/ex22` suffix `cuda_nvshmem`, both protocols x np
{1,4}, all pass through `make test`) -- PETSc's first NVSHMEM test.

Net position: branch-NVSHMEM's intra-node gap vs branch-MPI narrows from 1.39x
(225/162) to **1.27x** (206/162). Still not a win on one node -- as predicted, fusion
was never going to close the remaining ~3 ops x 2 us -- but it is free, strictly
nonnegative in every measurement, deletes a documented workaround, and its per-exchange
saving is fabric-independent (it survives to multi-node when peermem lands). This plus
the build fixes is the MR package; an in-tree NVSHMEM test should ride along (none
exist, build notes section 8).

## 26. Shards (m >> 1): the last algorithmic axis, measured and closed

TODO item 11 executed (`acgnbench.c -shards m`): each rank's gradient is split into m
slices; slice k's sends (its own pair of sharded SFs, explicit strided `ilocal`) are
posted immediately after slice k's compute, so slice k+1's compute overlaps them --
sender-side pipelining. Total compute (G AXPYs) and total bytes (d) are held constant;
m changes only the granularity (requires m | G and m | d). Sweep: d in {64 KB, 1 MB,
4 MB} x G in {8, 32} x m in {1,2,4,8} x {MPI, NVSHMEM-legacy, NVSHMEM-fused}, best of 2
(`results-20260815-shards.txt`).

**Result: m=1 is optimal in 23 of 24 cells.**

- **MPI degrades monotonically with m everywhere** (64 KB G=8: 165 -> 186 -> 237 -> 312).
  Every shard adds a host-blocked exchange; the per-exchange floor multiplies faster
  than any overlap can pay for.
- **The one predicted exception is real but small**: at 1 MB + heavy compute (G=32) the
  NVSHMEM arms' optimum shifts to m=4 -- fused 622 -> 569 (-8.6%), legacy 647 -> 596
  (-7.8%). This is exactly the section 16.2 regime: per-shard compute (~8 AXPYs) is
  finally large enough to hide a per-shard send, and the 256 KB shard is still
  latency-ish. At 4 MB the effect vanishes (bandwidth-bound; section 16.5's rule).
- **Even the best sharded NVSHMEM cell loses to unsharded MPI** (569 vs 410); sharding
  never reorders any transport ranking.
- The put+signal fusion (section 25) holds under sharding: fused < legacy in all 12
  NVSHMEM cells, by 5-9%.

One-line verdict for the poster: **sender-side sharding cannot rescue a host-API
transport on this DAG -- it recovers at most ~9% in one narrow window and changes no
conclusion.** With this, the section 17 route list is fully explored: every proposed
route has now been built and measured.

## 27. The device-side convergence test: the stopping decision as one more frozen edge

TODO item 14 executed (`acgnbench-nvdev.cu -converge`, persistent mode). The section 20
reviewer answer ("the graph/kernel freezes the map, not the iteration count") is now a
measured artifact: a synthetic residual vector decays by rho per iteration; every
`-check_every` iterations each PE grid-reduces its ||rv||^2, publishes the partial to a
symmetric slot, all-pulls the other three, and sums **in fixed rank order so every PE
computes the bit-identical global norm** -- hence the same stop decision with no extra
agreement round. An epoch-stamped decision word releases all blocks; all four resident
kernels break their loops on the same iteration. Flow control needs no ack signals: the
decision poll itself gates every block, so slot/counter reuse is provably serialized
behind the slowest block (the same ack-chain style argument as section 23).

Measured (64 KB, persistent, tol=24, rho=0.99 -> analytic stop at n*=202):

| configuration | stopped after | predicted | per-iter |
| --- | ---: | ---: | ---: |
| no convergence test (reference) | -- | -- | 43.1 us |
| `-converge -check_every 1` | 202 | 202 **MATCH** | 54.0 us |
| `-converge -check_every 10` | 210 | 210 **MATCH** (overshoot 8 <= k-1) | **42.2 us** |

Reading: checking *every* iteration costs ~11 us/iter (a grid reduction + a 4-double
all-pull + a decision gate); checking every 10th amortizes to noise while overshooting
at most k-1 iterations -- the exact chunked-replay tradeoff section 20 described for
graphs, here executed with zero host involvement: **one kernel launch runs the whole
solve, including deciding when it is over.** This is the capability that distinguishes
the device arm from graph replay (a graph cannot decide; a resident kernel can).

## Open issue found while building the benchmark

A hand-written PetscSF benchmark (`sfbench.c`: ring topology, raw `cudaMalloc` root/leaf buffers,
`PetscSFBcastWithMemTypeBegin` with `PETSC_MEMTYPE_CUDA`) **segfaults under GPU-aware MPI**:

    UCX ERROR cuMemGetAddressRange(0x7fabd7e00000) error: named symbol not found
    [1]PETSC ERROR: Caught signal number 11 SEGV

Established about it:
- Works with `-use_gpu_aware_mpi 0`, and works with `-use_nvshmem 1`.
- Not a UCX transport-selection issue: reproduced with `UCX_TLS` = `sm,self,cuda_copy,cuda_ipc`,
  `sm,self,cuda_ipc`, `...,gdr_copy`, and `all`.
- Not caused by leaf contiguity: reproduced with both contiguous and strided `ilocal`.
- Plain MPI+CUDA `MPI_Send`/`MPI_Recv` on raw `cudaMalloc` pointers works fine (no PETSc).
- PETSc's own `ex19`/`ex22` with GPU-aware MPI work fine.

So it is specific to this SF construction rather than to GPU-aware MPI generally. Unresolved;
the ex19-based measurements above avoid it entirely. Worth reducing further before reporting
upstream — note the CUDA driver here is 13.3 while the toolkit and HPC-X UCX are 12.x, which is a
plausible contributor.
