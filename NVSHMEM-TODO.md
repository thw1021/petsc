# Remaining TODO items, ACGN/NVSHMEM poster project

**Status 2026-09-02:** multi-node is unblocked and measured (D), the research harness has a
measured cost model, a validated schedule simulator, a closed-loop executor and measured
operator costs (see `~/prox-latency/SESSION_SUMMARY_2026-09-02.md` and
`~/prox-latency/experiments/RESULTS.md` Result 7). Section G below is the consolidated
open list as of that date; A-F keep their history.

Written 2026-08-15 at the end of the 2-node measurement campaign. The measurement matrix
is COMPLETE (perf notes section 24: every buildable cell measured, every empty cell
structurally empty). What remains is turning measurements into deliverables. Organized by
where the work happens; evidence pointers cite `NVSHMEM-PERF-NOTES.md` (PN),
`NVSHMEM-BUILD-NOTES.md` (BN), `TAOTERM-WISHLIST.md` (WL).

## A. The poster (desk work; the actual deliverable)

1. **Fill the five "TO APPEAR" panels** in `~/sc26-poster/main.tex` with measured
   (T_iter, TTS) pairs via the `\ConvBody` macros. Data: PN 24 master matrix.
2. **Relabel the result matrix's "NVSHMEM" axis to "GPU-orchestrated execution".**
   No longer optional: the threat test fired (PN 19 finding 2, replicated on three
   transports PN 23) -- the current panel titles would misrepresent the findings.
3. **Rewrite the "Measured findings" bullets** to the honest story: splitting wins
   22-30% under MPI and ONLY under MPI; the co-design ladder 162 -> 107 -> 41 -> 32 us;
   caption honesty about the deliverable ladder (PETSc-today / PETSc-compute+prototype
   transport / showcase; PN 24 deliverable-ladder table).
4. **Compute-weight calibration**: G/P are arbitrary AXPY counts and the
   monolithic-vs-splitting crossover depends on them (PN 19 finding 3). Either pin G/P
   to real tomography operator costs, or present the crossover itself as the result
   (recommended).
5. **Decide the monolithic panel treatment** -- the dumb-serial arm is measured
   (118 us @ 64 KB light; 369 heavy).

## B. Upstream PETSc work (desk work, this branch)

6. **MOSTLY DONE (2026-08-15, second job)**: the branch now carries the MR-shaped
   series -- `676ea87` build fixes (NVSHMEM.py liblist + gmakefile dlink), `53baaf9`
   put+signal fusion + the first in-tree NVSHMEM test (ex22 cuda_nvshmem, passes the
   harness), and the campaign-preservation commit at the branch tip (excluded from
   upstream). **`make check` PASSED on the committed state** (1/2 ranks + CUDA).
   Remaining before opening the MR: reword the WIP
   commit message, run `make clangformat` on a machine that has clang-format (absent
   on compute nodes), decide whether `-use_nvshmem_putsig` stays a toggle or fused
   becomes unconditional, and split the preservation commit out of the MR branch.
7. **DONE (2026-08-15, second job; PN 25): `sfnvshmem.cu` op fusion implemented** behind
   `-use_nvshmem_putsig` (default on). Correct (ex22, ex19 byte-identical both ways);
   sfbench floor -7%, branch DAG **-9%** (227->206 us at 64 KB; the extra gain is the
   arrival signal landing at Begin-time -- a critical-path latency effect invisible to
   ring microbenchmarks). Gap to branch-MPI narrows 1.39x -> 1.27x. Remaining decision
   for the MR: keep the toggle or make fused unconditional. Original analysis follows
   for reference. (PN 13.1, 14.4, 15.) The put
   protocol issues ~5 stream ops x ~2 us each; substitution was measured dead (PN 14),
   so the lever is issuing FEWER: (a) `nvshmemx_putmem_signal_nbi_on_stream` (present in
   3.4.5, `nvshmemx_api.h:152`) replaces put + quiet + signal and deletes the ordering
   workaround at `sfnvshmem.cu:590` and the TODO at `:597`; (b) the arrival wait moves
   into the unpack kernel's prologue (leader-per-block poll + `__syncthreads`, the
   pattern validated by route 2b's v4 protocol -- NOT all-thread polling, which PN 23
   showed self-jams the L2); (c) the PostUnpack release signal moves into unpack's
   epilogue. 7 ops -> 3, expected 17 -> ~13-14 us; a further unverified fold (flow
   control into pack) approaches ~12. Does NOT flip the single-node verdict vs MPI's
   11.5; the payoff is multi-node, where the per-exchange op count survives unchanged.
   MR should preserve the get-protocol and proxy branches and add an in-tree NVSHMEM
   test (none exist, BN 8).
8. **Consider an SF-NCCL backend MR** -- fuses within an exchange only (SF's collective
   per-exchange Begin/End cannot express stage fusion or per-rank programs), so it
   preserves part of the 107 us eager number (NEXT-STEPS layering note, PN 24).
9. **Capture-safety upstreaming**: document/support the
   `-root_device_context_stream_type nonblocking` precondition for graph capture over
   PETSc ops (PN 20; WL item 4).
10. **Reduce + report the sfbench segfault**: raw-`cudaMalloc` SF buffers crash UCX
    under GPU-aware MPI (PN "Open issue"; driver 13.3 vs toolkit 12.x suspected).

## C. Needs a GPU job (single node suffices)

11. **DONE (2026-08-15, second job; PN 26): shards arm measured.** m=1 optimal in 23/24
    cells; the one exception (1 MB + heavy compute, NVSHMEM m=4, -8.6%) confirms the
    16.2 overlap regime but reorders nothing. Axis closed; every PN 17 route is now
    built and measured.
12. **Device-arm micro-optimizations** toward the ~high-20s floor (PN 23 headroom:
    consume fused into compute, warp-specialized polling, deferred off-critical-path
    acks) -- only worthwhile if the device arm should beat graph-NCCL (32 us) on the
    poster.
13. **Matrix gap fill**: heavy-compute (G=32) measured only at 64 KB for graph/device
    arms; more reps if the poster wants error bars.
14. **DONE (2026-08-15, second job; PN 27): device-side convergence test built and
    measured.** Stops at exactly the analytically predicted iteration (202 at k=1;
    210 = predicted with overshoot <= k-1 at k=10); costs ~11 us/iter checked every
    iteration, amortizes to noise at k=10. One launch runs the whole solve including
    the stopping decision -- the capability graphs cannot express.

## D. Multi-node (UNBLOCKED 2026-09-01: peermem live via DOCA-OFED 26.04)

15. ~~File the ALCF ticket~~ DONE; resolved. `ALCF-TICKET-PEERMEM.md` now carries the
    resolution plus five follow-up items (dead 400G port on x2000c0s5b0n0, NCCL RoCE v2
    stall, no `memory_peers` sysfs for UCX 1.17, no GDRCopy, IBGDA driver params).
16. **Peermem landed -- ladder rerun 2026-09-01** (BN 13.4, PN 28; raw logs
    `nvshmem-tools/results-20260901-*`): NVSHMEM 8 PEs over 2 nodes initializes and
    validates; PETSc suite PASS at np=8 over 2 nodes (put, get, ex19); sfbench and
    acgnbench inter-node numbers measured for MPI vs NVSHMEM put/get; NCCL fixed on the
    400G port (RoCE v1); MPI-GDR baseline via UCX 1.19 preload; IBGDA measured in its
    CPU-doorbell hybrid mode. Still open from this item:
    - **4xN placement instance** (PN 18 observation 2): `acgnbench` hard-codes 4 ranks;
      a placement-aware 4x2 variant is new benchmark design, not a rerun. **First
      placement result is in (PN 28.9, no new code): the role-0-alone 3+1 cut is worth
      7-16% at 64 KB and 20-44% at 4 MB for MPI/NCCL/graph; NVSHMEM refuses unbalanced
      placements ("same number of PEs on all nodes"), so the 4xN design must model
      chain-hop latency + per-port volume and keep placements balanced.**
    - **Device-side (`acgnbench-nvdev`) across nodes**: the consumer pull uses
      `nvshmem_ptr()`, NULL for remote PEs; needs an `nvshmem_getmem`/put-based edge
      variant to run 2+2. Now possible (IBGDA hybrid) -- the poster's "GPU reacts
      mid-flight across the fabric" arm.
    - True IBGDA (GPU rings doorbells) waits on the driver params (ticket follow-up 5);
      the CPU-doorbell hybrid is measured (PN 28.6) and is slower than IBRC -- do not use.
    - Graph-NCCL 2+2 = 122 us at 64 KB (PN 28.7), the best inter-node arm; poster panel
      material together with the PN 28.8 matrix.
    - **DONE 2026-09-02: measured cost catalog + schedule simulator + re-screen of the
      3072 skeletons** live in `~/prox-latency/experiments/` (RESULTS.md Result 7; PN
      28.10). Validated to x1.23 over 80 measured cells. Next in that line: finals-stage
      iteration counts for the top-20, a generic executor (acgnbench taking a skeleton +
      placement), and GPU operators (shard gradient, TV/box proxes) to measure tau_tq.

## E. TaoTerm machine

17. **Carry `TAOTERM-WISHLIST.md` (items 1-6)** into TaoTerm API development -- notably
    item 6, the measured case that "PETSc really supporting NVSHMEM" means the device
    API (symmetric-heap Vecs, device-callable term kernels, frozen schedule as input),
    not a transport swap.

## F. Repo hygiene (next session or end of this one)

18. **Commit the campaign to the branch**: notes (`NVSHMEM-*.md`, `TAOTERM-WISHLIST.md`,
    `ALCF-TICKET-PEERMEM.md`), benchmarks (`nvshmem-tools/`), and results files
    (`results-20260815-*.txt`), so the work survives independently of any node. Keep
    the eventual upstream MR (item 6) free of the benchmark/notes files.

## G. Consolidated open list (2026-09-02)

Poster (desk work):
19. Fill the five "TO APPEAR" panels with (T_iter, TTS): single-node ladder (PN 24) plus the
    2-node matrix (PN 28.8) and the placement result (PN 28.9); relabel the NVSHMEM axis
    "GPU-orchestrated execution"; rewrite "Measured findings" (items A1-A3, A5 still open).
20. Replace the compute-weight calibration question (A4) with the measured operator table
    (PN 28.11 / opbench): pin G/P to real gradient/prox costs or present the crossover.
21. Add the harness result to the co-design panel: reported skeleton right to within a few
    percent under measured costs, allreduce 1.3-2x, placement/transport the residual lever;
    state the balanced-placement constraint NVSHMEM imposes on the step-2 family.
22. Companion note: replace Step 5's synthetic durations by the measured catalog; record the
    finals-only ranking lesson and the need for an untouched test set.

Measurements still worth taking (2 nodes suffice unless noted):
23. DONE 2026-09-02 (PN 28.12): `acgnbench-nvdev2.cu -push` -- 104 us at 64 KB 2+2 (the
    fastest inter-node arm), 36.5 us on one node (20% better than pull), stopping test
    exact across the fabric, zero mismatches. Open: true IBGDA to approach ~70 us.
24. DONE 2026-09-02: `acgnrun-nccl.c`; closed loop on 140 cells. Open residual: the model
    ranks NCCL/graph skeletons only ~2/3 right (per-connection pipelining at 4 MB; the
    late-gradient rank placement effect the winner's permutation exposes).
25. Work-efficient TV prox kernel and a real tomography projector (opbench uses a naive
    one-thread-per-row Condat and a synthetic ray matrix) -- these two numbers set the
    measured regime and the poster's TTS panels.
26. PARTLY DONE 2026-09-02: `gangbench.c` measured tau_grad(g=1,2,4) incl. the NVLink
    reduce (256²: 80/56/41 us; 1024²: 1016/604/338 us); modeled (`Costs(gang=g)`); the
    finals re-score says gang 2 = -4..13% TTS, gang 4 = -8..26%. Still open: the 8-rank
    gang executor (leaders route) to close this loop on hardware.
27. 4-node runs (the poster's 4x4 instance) when a 4-node allocation exists; the model and
    executor already take arbitrary host maps.
28. MOSTLY ANSWERED 2026-09-02 (PN 28.12): NCCL fan-out is serial (modeled); SF put at
    16 MB is proxy-throughput bound (~25 GB/s per on-stream put); UCX_RNDV_THRESH=16k
    closes the 32 KB hole (now default under JANUS_UCX119); NVSHMEM get_bw collapse at
    16 MB is real (report to NVIDIA; avoid get above 8 MB); the MPI 4 MB drift is
    run-to-run variance. Still open: the model's NCCL/graph *ranking* of skeletons
    (~half the pairs) -- the late-gradient rank placement effect.
29. True IBGDA once the driver parameter lands (ticket follow-up 5); do not use the hybrid.

Admin:
30. Send `ALCF-TICKET-FOLLOWUPS.md` (dead 400G port, NCCL RoCE v2, UCX 1.17 detection,
    GDRCopy, IBGDA parameters). Items 3 and 5 change our numbers.

Upstream PETSc (B6-B10 still open):
31. MR prep: clang-format on a machine that has it, commit-message rewrite, decide the
    `-use_nvshmem_putsig` toggle, split the campaign commit out; consider adding the
    multi-node run of the ex22 cuda_nvshmem test to the MR notes (passes at np=8).
32. TaoTerm: no TV/box proximal maps exist (types: callbacks, shell, sum, halfl2squared,
    l1, quadratic); the wishlist items 1-6 remain the API consequence of the campaign.

Repo hygiene:
33. Commit the campaign in the worktree (50 files) and the harness work in prox-latency
    (16 files); keep the upstream MR free of them.
34. The GPUDirect-MPI arms depend on `~/opt/ucx-1.19-doca` (shared home); node-local
    scratch must never hold anything remote ranks need.

## H. SF-NCCL backend (2026-09-04, written without a node; see `SESSION-SUMMARY-2026-09-04.md`)

35. **Build and test the SF-NCCL backend**: `--with-nccl-dir=/soft/libraries/nccl`, `make all`,
    `make clangformat`, `make test search=vec_is_sf_tests-ex22_cuda_nccl`, then ex22/ex19 at
    np=8 over 2 nodes with `-use_nccl 1 -info :sf` (the info line is the discriminator).
    Files: `NCCL.py`, `src/vec/is/sf/impls/basic/nccl/{sfnccl.cu,makefile}`, `sfimpl.h`,
    `sf.c`, `sfpack.{h,c}`, `sfmpi.c`, `sfbasic.c`, `sf/tests/ex22.c`, `doc/changes/dev.md`.
36. **Measure it**: sfbench ring (1 node, 8-rank ring), sfbench2 `-naxpy` overlap regime,
    acgnbench/acgnrun branch+allreduce at 64 KB and 4 MB, 1 node and 2+2, against MPI, MPI-GDR
    and NVSHMEM. Predictions in the summary section 5.
37. **Measure the host-scalar `MPI_Allreduce` floor** (1 double, 4 ranks, 1 node and 2+2) with a
    host-buffer arm in `collbench.c`; it is the missing number in the "NCCL for VecNorm" argument.
38. **Upstream**: the MPIX-selection bug fix in `sfmpi.c` (SetCommunicationOps ran before the
    link's memory types were set, so `-sf_use_stream_aware_mpi` never took effect) goes with the
    NCCL MR or as its own small MR first. Keep it separate from the NVSHMEM MR.
39. Open design question for the Jeff conversation: device-resident reductions + chunked
    convergence checks + iteration capture in KSP (the shape where NCCL wins 5x), i.e. wishlist
    item 6 seen from the KSP side.
