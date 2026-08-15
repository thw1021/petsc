# Chapter 9 — The full matrix, the deliverable ladder, and the lessons

## 9.1 Everything, in one table

Single node, 64 KB messages, light compute (G=8, P=4), µs per iteration, best-of-N,
all arms validated:

| comm layer \ algorithm | allreduce | branch | execution axis |
| --- | ---: | ---: | --- |
| GPU-aware MPI (via PetscSF) | 209 | 162 | eager only — capture structurally impossible |
| NVSHMEM host API (via PetscSF) | 227 | 225 (206 with the ch. 4.4 put+signal fusion) | eager only — not capture-safe |
| NCCL fused groups, eager | 109 | 107 | |
| NCCL fused groups, graph replay | 34 | **32** | |
| NVSHMEM device API, 1 launch/iter | 44 | 48 | |
| NVSHMEM device API, persistent | 42 | **41** | one launch per 300-iteration run |
| monolithic serial (no comm) | — 118 — | | baseline |
| compute-only floor, eager | — 39 — | | the graph arms beat it (ch. 6.4) |

The same matrix exists at 512 KB / 1 MB / 4 MB, at heavy compute (G=32, P=16), and — for
the MPI and NCCL rows — across two nodes. Empty cells are *structurally* empty, not
unmeasured; that distinction is itself a result.

The ladder, drawn:

```
   us/iter at 64 KB (branch column)
   162 |####################| MPI            <- the CPU blocks per exchange
   107 |#############       | NCCL eager     <- stop blocking; fuse each stage
    41 |#####               | NVSHMEM device <- no CPU in the loop at all
    32 |####                | NCCL + graph   <- stop re-enqueueing entirely
    39 |=====               | "compute floor" (itself launch-bound!)
```

Each rung is a change in *who orchestrates*, not a tuning of the previous rung — and no
rung was reachable by tuning the one above it. That is the poster's thesis, measured:
schedule, transport, and execution model must be designed together.

## 9.2 Three findings that outrank the speedups

**1. The splitting advantage is a statement about the transport's cost model.** Branch
routing beats allreduce by 21–30% under MPI (both intra- and inter-node) — and ties or
loses under every fused/non-blocking transport we built (NCCL eager, NCCL graph, NVSHMEM
device: three independent implementations agreeing). The claim "our decentralized
routing avoids the global reduction" survives only with the qualifier *"...and this
matters under host-blocking transports."* We only know this because we ran the
experiment designed to kill our own claim (the threat test, ch. 5.3). At np=4 on
all-to-all NVLink the ring allreduce's extra volume is cheaper than the branch DAG's
stage serialization; at scale, on weaker topologies, or under MPI, the volume argument
returns.

**2. The dumbest baseline is a diagnostic instrument.** Serial-on-one-GPU beat every
parallel MPI arm at light compute weight. Parallelizing across GPUs is not free — it
buys compute scaling at the price of communication, and below a measurable compute
threshold it is a net loss. Any parallel-performance story that cannot state its
crossover point against the serial baseline is incomplete.

**3. Not all measured arms are equally "real."** The deliverable ladder:

| rung | what it is | status |
| --- | --- | --- |
| PetscSF over MPI / NVSHMEM host | the library as shipped | in-tree today; nothing in-tree beats 162 µs — and the one in-tree improvement our measurements supported (the ch. 4.4 fusion, +9% for the NVSHMEM path) is now implemented and MR-ready |
| NCCL arms | PETSc compute, hand-rolled transport *bypassing* the library's abstraction | prototype; defines what an API should become |
| graph replay | works over real PETSc kernels, given capture-safety discipline | possible today with care; needs API support to be safe |
| device NVSHMEM | PETSc-free showcase | upper bound + API requirements list |

Publishing the ladder honestly — labeling which number is product and which is
possibility — is worth more than the flashiest single number, because it maps each gap
to a concrete piece of missing API.

## 9.3 Method lessons (the transferable part)

1. **Find the invariant cost early.** One microbenchmark (~2 µs per stream op, any op)
   explained a library's 10 µs overhead, predicted two optimizations' failure before
   implementation, and pointed at the three designs that eventually won. Measure the
   floor before optimizing anything above it.
2. **Substitution vs. count.** When every operation costs the same, swapping operation
   *types* is motion without progress. Only reducing the *number* of operations (fusing,
   recording, moving orchestration into the kernel) moves the needle.
3. **Write predictions down before measuring.** Ours: 70–100 µs (measured 107), 40–70
   (measured 32), 10–30 (measured 41), "branch still wins under NCCL" (falsified).
   Recorded predictions make both the hits and the misses information.
4. **Run the experiment that could kill your story.** The threat test produced the
   campaign's most valuable finding — by firing.
5. **Build validation into the benchmark.** Epoch-stamped data + per-element checks
   turned "did my synchronization protocol race?" from an anxiety into a counter that
   reads zero. It is the only reason four protocol revisions in one afternoon were safe.
6. **Beware silent fallbacks.** Transports quietly degrade (NVSHMEM -> MPI; NCCL ->
   bounce buffers; UCX -> host staging). Establish a positive discriminator — a banner,
   a debug line, a counter — for every "the fast path is on" claim.
7. **A regression can be the most informative result.** v3's barrier-free protocol was
   *slower* — and the diagnosis (polling contention self-jamming the L2) taught the
   principle (contention-free polling) that made v4 the right design.
8. **Config traps dominate first contact with any fabric.** Core binding (30x), bond
   member ports (250x), transport lists per-regime, capture-hostile default streams.
   None of these are in the manual's happy path; all were found by refusing to accept a
   weird number.
9. **Knob-invariance is a diagnosis.** When six independent tuning parameters all land
   within 1%, stop tuning: the limiter is structural, and the fix lives elsewhere
   (in our case, in a kernel module only root can load).
10. **Write the ticket like a proof.** Failure signatures from every path, control
    experiments proving the healthy parts healthy, and the one-line fix. Evidence
    assembled at discovery time costs minutes; reconstructed later, days.
11. **Microbenchmarks measure cost; dependency chains measure when costs land.** The
    put+signal fusion (ch. 4.4) saved 1.4 µs in a ring test but 4 µs per exchange in the
    real DAG, because it also moved the arrival signal to data-arrival time — a latency
    effect that only shows when someone is actually blocked waiting. Validate every
    microbenchmark conclusion once at application shape before believing its magnitude.

## 9.4 Where the trail continues

The working documents carry the live state: `NVSHMEM-TODO.md` (all remaining items),
`NVSHMEM-PERF-NOTES.md` §17–24 (every measurement behind these chapters),
`NVSHMEM-BUILD-NOTES.md` (build recipes, environment traps, the multi-node evidence),
`TAOTERM-WISHLIST.md` (what the API must become — item 6 is chapter 7's requirements
list), and `../nvshmem-tools/` (every benchmark source, each small enough to read in one
sitting — `acgnbench.c` -> `acgnbench-nccl.c` -> `acgnbench-nccl-graph.c` ->
`acgnbench-nvdev.cu` is the ladder in code form).
