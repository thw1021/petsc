# Remaining TODO items, ACGN/NVSHMEM poster project

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

## D. Multi-node (externally blocked)

15. **File the ALCF ticket** -- draft ready in `ALCF-TICKET-PEERMEM.md` (peermem not
    loaded, dmabuf=0; one modprobe unlocks NVSHMEM multi-node, NCCL's fast path, and
    MPI GDR). User action.
16. **When peermem lands**: rerun the sanity ladder (BN 13.3), re-measure NVSHMEM/NCCL
    inter-node, fair MPI-GDR baseline, IBGDA arm, and the 4xN placement instance
    (PN 18 observation 2).

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
