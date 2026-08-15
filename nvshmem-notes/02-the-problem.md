# Chapter 2 — The problem: a splitting algorithm and its communication graph

## 2.1 The application, in one paragraph

The SC26 poster concerns *composite optimization*: minimizing a sum of terms
F(x) = Σ α_i f_i(A_i x), where the terms are qualitatively different — a data-misfit
term, several regularizers (total-variation "TV" penalties), a box constraint. The
poster's thesis is *co-design*: instead of treating the objective as one monolithic
callback evaluated data-parallel across all GPUs, assign each term to the GPU that owns
its data, freeze a communication schedule derived from the term structure, and let each
GPU run only its own part. The concrete instance is a four-node tomography problem
("ACGN" splitting): gradient shards on ranks 0–2, a box-prox on rank 1, a row-TV prox on
rank 2, a column-TV prox on rank 3.

You do not need the optimization theory. What matters here is the *shape of the
communication* the algorithm needs each iteration.

## 2.2 The DAG

One iteration is a directed acyclic graph (DAG) of compute steps and data movements over
4 ranks (r0..r3). Arrows are messages; boxes are compute:

```
 rank 0            rank 1              rank 2              rank 3
 ------            ------              ------              ------
                  [box prox]
                   produce x1
      <------------ x1 ---+------- x1 ------>+------- x1 ------>
 [grad g0]        [grad g1]           [grad g2]
     |                |                   |
     +---- g0 --------+---- g1 --------->[recv g0,g1]           (also:)
     +---- g0 ------------- g1 ---------- g2 ------------------>[recv g0,g1,g2]
                                      [rowTV prox]                (kept for LATER)
                                       produce x2
                    <------- x2 ----------+-------- x2 -------->
                                                            [grad g3 at x2]
                                                            [NOW consume g0,g1,g2]
                                                            [colTV prox]
                                                             produce x3
                    <------- x3 --------- x3 ---------------+
                  [mixing]            [mixing]              [mixing]
```

Count the edges: 12 point-to-point messages per iteration, organized in 4 "stages"
(x1 distribution, gradient routing, x2 distribution, x3 distribution). Three features
matter enormously later:

1. **It is a dependency chain.** Rank 2 cannot start its prox before g0 and g1 arrive;
   rank 3 cannot finish before x2 arrives. Most messages sit *on the critical path* —
   when they are in flight, the receiver is idle. (Chapter 4 shows why this specific
   property decides which transport wins.)
2. **One edge is send-early/consume-late.** g0,g1,g2 are sent toward rank 3 as soon as
   they exist, but rank 3 only *uses* them after computing g3. That is the DAG's one
   genuine overlap window.
3. **Each rank's program is different.** Rank 0 only computes a gradient and sends it;
   rank 3 receives five things. "One program, branch by rank" — remember this when a
   library insists that all ranks must call the same operations (chapter 5).

## 2.3 The two competing communication strategies

The routing above ("**branch**") delivers each gradient only where it is needed. The
conventional alternative ("**allreduce**") has every rank contribute its gradient to a
global sum that everyone receives:

```
 branch:    g0 -> {r2, r3}     5 messages, each rank gets exactly what it needs,
            g1 -> {r2, r3}     no global synchronization point
            g2 -> {r3}

 allreduce: SUM(g0..g3) -> everyone     one collective operation, but:
            - moves ~4x the useful data
            - is a barrier: ALL ranks meet at it
            - puts ranks 0,1 (who don't need the sum) on the critical path
```

The poster's headline claim was: branch beats allreduce. Our job was to measure by how
much — and under which transports the claim survives. (Spoiler, chapter 5: the claim is
true under MPI and *only* under MPI, which is a more interesting result than the one we
were hoping for.)

We also added a third arm mid-campaign, at the request of "what's the dumbest baseline?":
**monolithic** — the entire iteration's compute executed serially on rank 0, zero
communication, as if a student wrote it for one GPU. It turned out to be embarrassingly
competitive at light compute weight, which forced an honest refinement of the story
(chapter 9).

## 2.4 How to benchmark this honestly

The benchmark (`acgnbench.c` and its descendants) reproduces the DAG exactly but
**emulates the compute**: each "gradient evaluation" is G repetitions of a vector AXPY
(y += a*x) on a length-d vector, each "prox" is P repetitions. Why emulate?

- The real operators would fix the compute/communication ratio at whatever the test
  problem happens to have. With knobs G and P we can *sweep* that ratio and find where
  conclusions flip (they do — see the monolithic arm).
- AXPY is memory-bound and utterly predictable, so the compute baseline is clean.
- The message size is the vector size d: sweeping d from 64 KB to 4 MB moves us from the
  latency-bound to the bandwidth-bound regime.

The measurement protocol, every rule of which exists because we got burned without it:

| rule | reason |
| --- | --- |
| 30 warmup iterations before the clock starts | first iterations pay one-time costs: memory pools, lazy library init, connection setup |
| `cudaDeviceSynchronize()` + `MPI_Barrier()` immediately before and after the timed region | otherwise you time the *enqueueing* of work, not the work (see 1.3) |
| report per-iteration time = total / 300, take max across ranks, best of 3 runs | the slowest rank is the honest number; best-of-N rejects node noise |
| `--bind-to none` always | chapter 1.5's 30x footgun |
| sum `Begin+End` when reading library profiles | different transports put their cost in different halves of a Begin/End pair; reading one half flattered NVSHMEM by 7.7x once |
| positively verify which transport actually ran | libraries **silently fall back**. PETSc's NVSHMEM path quietly uses MPI if any of five eligibility conditions fails. We verify via a banner that only prints when NVSHMEM truly initializes: zero banners in all control runs, exactly one in all test runs |
| exclude one-time setup from per-iteration numbers | NVSHMEM init costs ~0.4 s once; amortized over a real solve it is irrelevant, but it dominates a short benchmark if you let it |

One more, which became load-bearing in chapter 7: **build correctness checking into the
benchmark itself.** Our later benchmarks stamp every transmitted value with the iteration
number and validate every received element every iteration. A synchronization bug then
shows up as a counted error, not as a suspiciously fast wrong answer.

Next: [Chapter 3 — the three ways GPUs can talk](03-how-gpus-talk.md).
