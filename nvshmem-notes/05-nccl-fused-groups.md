# Chapter 5 — NCCL fused groups, and the threat test that fired

Chapter 4 ended with a diagnosis: too many stream operations per iteration. NCCL's group
mechanism (chapter 3.3) attacks exactly that: all sends/receives of a DAG stage, fused
into one kernel per rank. This chapter builds that benchmark and reports the two things
it taught us — one expected, one that overturned the poster's framing.

## 5.1 Bypassing the library on purpose

An important design decision: the NCCL benchmark (`acgnbench-nccl.c`) keeps PETSc for
all *compute* (the AXPY emulation runs through PETSc's vector class, on PETSc's stream)
but calls NCCL **directly for the edges, bypassing PetscSF**. Not laziness — PetscSF's
interface structurally cannot express what makes NCCL fast:

- PetscSF's `Begin/End` must be called *collectively by all ranks* per exchange. Our DAG
  wants each rank to post only its own edges (rank 0's program never mentions x2).
- PetscSF's unit is one exchange; the win comes from fusing *five* edges of a stage into
  one group. A hypothetical SF-NCCL backend could fuse within an exchange only.

When a library's abstraction and the winning execution shape disagree, a hand-rolled
benchmark that bypasses the abstraction tells you what an API *should* become — that is
its deliverable, and it is honest to say so (chapter 9's "deliverable ladder").

Mechanically, the integration has one critical detail: NCCL operations must be enqueued
on **the same stream PETSc computes on**, or the DAG ordering silently breaks. PETSc
exposes it (`PetscDeviceContextGetStreamHandle()`), and then stream order does all the
sequencing: prox kernel, then group, then gradient kernels, then group...

```
 one iteration, rank 2's stream:
 [x1 recv (group1)][G axpys][g-routing (group2: recv g0,recv g1,send g2)]
                                [2+P axpys][x2 send (group3)]...[mixing]
    ^ each [group] = ONE kernel; the host enqueued ~8 things total and never blocked
```

Per iteration and per rank: 4 fused communication launches + ~14 compute launches,
no CPU blocks. Validation: every received lane checksummed after the run (chapter 2.4).

## 5.2 Results, round one

64 KB messages, light compute (best of 3; branch-MPI baseline 162 µs, compute floor 39):

| arm | µs/iter |
| --- | ---: |
| branch over MPI | 162 |
| branch over PetscSF-NVSHMEM | 225 |
| **branch over NCCL groups** | **107** |

A 34% win over MPI from fusing stages — the 2 µs law cashing out. (Our written
prediction had been 70–100 µs; 107 just misses the band. Predictions are recorded before
measuring precisely so that misses stay visible.)

## 5.3 The threat test

Good experimental practice: before celebrating, run the measurement that could destroy
your story. Ours: implement the **allreduce** arm over NCCL too (`ncclAllReduce`, one
stream-ordered op, no device sync). If branch-NCCL ≈ allreduce-NCCL, then the poster's
"branch routing beats global reduction" claim was never about data volume — it was an
artifact of MPI's cost model. Prediction on record: branch still wins, because allreduce
moves ~4x the data and is a barrier.

| 64 KB, light compute | branch | allreduce |
| --- | ---: | ---: |
| MPI | **162** | 209 |
| NCCL | 107 | **109** ... and at every larger size allreduce *wins* (512 KB: 144 vs 149; 4 MB: 348 vs 363) |

**The threat test fired.** Under a fused, non-blocking transport the splitting advantage
is gone — a tie at 64 KB, allreduce ahead by 3–8% above.

Why? Under MPI, every exchange costs a *host block*, and allreduce is the most
block-and-barrier-heavy pattern, so branch's fewer, targeted edges win. NCCL removes the
blocks entirely, and `ncclAllReduce` is a **ring**: rank r sends a chunk to r+1 while
receiving from r-1, all links busy simultaneously, cost ~2(p-1)/p x buffer per rank —
at p = 4 only ~1.5x the buffer, pipelined perfectly:

```
 ring allreduce, all 4 links concurrently:      branch, stage-serialized:
   r0 -> r1 -> r2 -> r3                            stage 2 cannot start
    ^                 |     x (few rounds)          before stage 1's data
    +-----------------+                             lands: dependency CHAIN
```

The branch DAG's stages are *dependency-serialized*; the ring's extra volume is cheaper
than that serialization. (Chapter 7 replicates this on a third transport, at which point
it graduates from observation to conclusion: **the splitting advantage is specific to
host-blocking cost models.** The claim survives, but only with that qualifier — which
the poster must now state.)

Two honest qualifiers recorded with it: this is 4 ranks on all-to-all NVLink (at larger
scale or weaker topologies volume arguments return — and inter-node under MPI, branch
still wins, chapter 8); and the allreduce arm is a cost *proxy* — the real algorithm
wants different aggregations per branch, so branch routing retains semantic value even
where its raw speed advantage does not.

## 5.4 The dumbest baseline embarrasses everyone

The monolithic arm (whole iteration serially on rank 0, zero communication), same table:

| 64 KB | monolithic | branch-MPI | branch-NCCL |
| --- | ---: | ---: | ---: |
| light compute (G=8) | **118** | 162 | 107 |
| heavy compute (G=32) | 369 | ~320 | **273** |

At light compute, *not communicating at all* beats every parallel MPI arm — four GPUs
lose to one because the parallel arms pay more in communication than they save in
compute. Only when per-term compute grows does splitting pay. This is not an
embarrassment to hide; the crossover — "splitting pays once per-term compute exceeds
routing cost" — is the co-design thesis in one sentence, now with a measured location.

Next: [Chapter 6 — stop enqueueing: record the iteration once and replay it](06-cuda-graphs.md).
