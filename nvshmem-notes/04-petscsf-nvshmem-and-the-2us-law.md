# Chapter 4 — PetscSF's NVSHMEM path, the `<<<1,1>>>` kernel, and the 2 µs law

PETSc (the numerical library the poster builds on) routes all its neighbor communication
through an abstraction called **PetscSF** ("star forest"): you describe once which remote
entries each rank needs, then repeatedly call `PetscSFBcastBegin()` / `PetscSFBcastEnd()`
to move them. PetscSF has an MPI backend and an NVSHMEM backend (host API). This chapter
is the autopsy of why the NVSHMEM backend — on one node — *loses* to the MPI backend,
and why the reasons are structural rather than bugs.

Headline numbers first (ring microbenchmark, 4 ranks, device buffers):

| bytes | MPI backend | NVSHMEM backend |
| ---: | ---: | ---: |
| 1 KB – 64 KB | ~11.5–12 µs | ~16–18 µs |
| raw NVSHMEM put+signal round trip (no PETSc) | | ~6.4 µs |

The transport itself costs ~6 µs. PETSc's protocol wraps ~10 µs of orchestration around
it. Where does 10 µs go?

## 4.1 The put protocol, operation by operation

One `Bcast(Begin+End)` on the NVSHMEM path issues this sequence (simplified from the
real source, `sfnvshmem.cu`):

```
      HOST (CPU) enqueues:          GPU comm stream executes:            moves data?
      -------------------           -------------------------            -----------
  1   pack kernel                   [gather my entries into send buf]       YES
  2   WaitSignals <<<1,1>>>         [SPIN until receiver's flag says         no
                                     "my buffer is free", then claim it]
  3   putmem_nbi (per neighbor)     [copy send buf -> receiver's heap]      YES
  4   quiet                         [barrier: "all my puts are done"]        no
  5   PutDataEnd <<<1,1>>>          [set arrival flag on each receiver;      no
                                     SPIN until MY arrival flag is set]
  6   unpack kernel                 [scatter received entries into place]   YES
  7   PostUnpack signal kernel      [reset senders' flags: "buffer free"]    no
```

Seven stream operations; **four are pure protocol** — flow control (2, 7), ordering (4),
and completion notification (5). By the 2 µs law that is ~10 µs of enqueue tax
regardless of message size, which is precisely the measured gap. (The remaining ops also
include two event pairs bridging the compute stream and a dedicated communication
stream; those cost only 0.53 µs total — a red herring we chased anyway, see 4.3.)

## 4.2 Why on earth `<<<1,1>>>`? — the spin-wait kernel explained

Operations 2 and 5 are kernels launched with **one block of one thread**, whose entire
body is "loop until this memory word changes". To a newcomer this looks absurd — the
machine has 132 SMs and we occupy one thread — so let's justify it from requirements:

The protocol needs, at a precise point *in the middle of a stream's work queue*, the
behavior "do not proceed past here until a flag (written by a remote GPU) becomes 1."

- Option A: have the **CPU** poll the flag, then enqueue the rest. But the whole point
  of this backend is that the CPU never blocks (that is MPI's weakness we're trying to
  avoid). Rejected by design.
- Option B: some "wait-on-memory" primitive in the stream. CUDA has one
  (`cuStreamWaitValue32`), but it is not portable across setups, and NVSHMEM's own
  stream-wait API — we measured it (`sigcost.cu`, chapter 1.4) — costs the same 2.24 µs
  as a kernel launch because it *is* one internally.
- Option C: enqueue a kernel that spins. A kernel occupies its place in the stream's
  FIFO, so everything enqueued after it simply cannot start until it returns — which is
  exactly "the stream waits here". Polling a flag is one thread reading one word in a
  loop; more threads would add nothing. Hence `<<<1,1>>>`: **the minimal possible kernel
  used as a programmable traffic light inside a work queue.**

So the `<<<1,1>>>` kernel is not naivety — it is the honest cost of expressing "wait" in
a stream-ordered world. The problem is not that it is small; it is that it is *a whole
stream operation* (2 µs) whose only product is one bit of information.

One subtlety we learned the hard way: these spin kernels **block their stream by
design** — so they must live on a *separate* stream from compute, or they would block
the compute behind them. That dedicated communication stream, plus events to link it to
the compute stream, is load-bearing (next section).

## 4.3 Two failed optimizations, and what they proved

We attempted two "obvious" improvements. Both were killed by measurement, and the
process is more instructive than success would have been.

**Attempt 1 — "the second stream is overhead; use one stream."** The event pairs
(0.53 µs) and the cross-stream hops looked wasteful. We added a flag to issue everything
on the compute stream. Result: *34–41% slower*. The spin-wait kernels, now on the
compute stream, blocked unrelated compute behind them; on their own high-priority stream
they had overlapped with it. The dedicated stream was not overhead but the thing making
the spin-waits affordable. **Lesson: before deleting structure, understand what it
buys.**

**Attempt 2 — "replace spin kernels with the host stream-wait/signal API."** Sounded
free: an API call instead of a kernel launch. We measured the API's cost *first*
(`sigcost.cu`): 2.24–2.55 µs — identical to a launch, because it enqueues an internal
kernel. Substitution gains zero. **Not implemented — the 30-minute measurement saved a
day of pointless refactoring.**

Those two failures pin down the law from chapter 1:

> The protocol's cost is (number of stream ops) x (~2 µs), and no substitution changes
> either factor. The only ways forward are to FUSE ops (do two protocol jobs in one op)
> or to stop enqueueing per-iteration entirely (record & replay — chapter 6; or move
> the whole protocol inside a kernel — chapter 7).

(The fusion route was left on the TODO list at this point in the campaign; one job
later it was implemented and measured — see 4.4.)

## 4.4 Epilogue: the fusion, implemented

A second job gave us the time to build the fusion, and it is a nice capstone for this
chapter because it exercises everything above. The change, three edits inside the
protocol:

1. For locally accessible peers, the put becomes NVSHMEM's **fused put+signal**
   (`nvshmemx_putmem_signal_nbi_on_stream`): the receiver's arrival flag is delivered
   *with the data*, ordered after it by API contract.
2. The `quiet` (op 4) is deleted — its only job was ordering the puts before the
   End-time signals, and those signals no longer exist for local peers.
3. The `PutDataEnd` `<<<1,1>>>` kernel now signals only *remotely* accessible peers
   (an `nvshmem_ptr()` test inside the kernel); its wait half is unchanged.

Correctness discipline before any timing: the FetchAndOp unit test at 2 and 4 ranks, and
the full nonlinear solve (`ex19`) byte-compared against the MPI baseline — under both the
new and old protocol, toggleable at runtime so A/B runs share one binary. (First
comparison came back "identical" suspiciously fast: the test binary had failed to build
and we were diffing empty files. Check sizes before trusting a diff.)

The results teach one more lesson each:

| instrument | legacy | fused | note |
| --- | ---: | ---: | --- |
| ring microbenchmark floor | ~19.8 µs | ~18.2 µs | −1.4 µs: exactly the one deleted op — the 2 µs law, confirmed in reverse |
| the ACGN DAG, 64 KB | 227 µs/iter | **206** | −9%, ~4 µs per exchange — **2.5x more than the microbenchmark predicts** |

Where does the extra saving come from? The second edit's side effect: the arrival signal
now lands at *Begin* time (with the data) instead of at the sender's *End* time. In a
back-to-back microbenchmark nobody is waiting, so only the op count shows. In the DAG,
a receiver **blocked on the critical path** unblocks the moment bytes arrive rather than
when the sender gets around to its `End` call — a latency term that multiplies across
every dependency edge. *Microbenchmarks measure cost; dependency chains measure when
costs land.* Both numbers are true; only the second one is what the application feels.

Net: the NVSHMEM backend's intra-node gap versus MPI narrows from 1.39x to 1.27x. Still
not a win — as the op-count arithmetic always said it wouldn't be — but strictly
nonnegative everywhere, it deletes a documented workaround, and the saving is
per-exchange, so it survives unchanged to the multi-node regime the backend was actually
built for (chapter 8).

## 4.5 The one regime where the host API wins anyway

Before dismissing the NVSHMEM backend: we swept for a regime where its one structural
advantage — *the CPU never blocks* — pays. If there is **independent GPU compute to
overlap** with the exchange, NVSHMEM's exposed cost (total minus compute) drops to
~5.6 µs while MPI's floors at ~8.1 µs, because MPI's CPU blocks cannot be hidden by any
amount of GPU work:

```
 MPI:      HOST |pack|##BLOCKED##|send|enqueue axpys|  Waitall  |unpack|
                       ^^^^ this stall happens BEFORE the compute exists
                            -> nothing can hide it

 NVSHMEM:  HOST |enqueue everything, returns immediately|
           GPU  [axpys.........................]   <- compute stream
           GPU  [spin|put|quiet|flags]             <- comm stream, CONCURRENT
```

But only when the exchange is latency-bound (< ~1 MB); at 4 MB+ both saturate the same
NVLink and the advantage vanishes. And our DAG (chapter 2) is a dependency chain with
almost no such independent work — which is exactly why, on the full benchmark, the
NVSHMEM backend lost in every cell (1.4–1.6x). **A transport cannot fix a schedule.**

Could the *schedule* manufacture the missing overlap? We tested the one remaining idea:
split each gradient into m shards, sending shard k while computing shard k+1
(sender-side pipelining — total compute and bytes held constant, only the granularity
changes). Verdict from a 24-cell sweep: **m=1 is optimal in 23 cells.** Every shard adds
a whole exchange's protocol cost, and that multiplies faster than overlap can pay for
it; MPI degrades monotonically with m, and NVSHMEM recovers at most 8.6% in exactly one
cell (1 MB messages + heavy compute — the one place per-shard compute is big enough to
hide a send that is still latency-bound), never enough to change any ranking. The
overlap regime this section describes is real, but you cannot *shard your way into it*
against a per-exchange cost floor.

Measured verdict for the DAG at 64 KB: branch-MPI 162 µs/iter, branch-NVSHMEM 225 µs/iter
(206 after the 4.4 fusion), against a 39 µs compute floor. The next three chapters are
about closing that gap by changing who is in charge.

Next: [Chapter 5 — NCCL, and the experiment that falsified our favorite claim](05-nccl-fused-groups.md).
