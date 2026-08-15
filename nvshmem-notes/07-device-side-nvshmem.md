# Chapter 7 — Device-side NVSHMEM: the GPUs run the whole show

Graphs (chapter 6) reduced the CPU to one launch per k iterations, replaying distinct
kernels. The logical endpoint goes further: fuse each rank's **entire iteration into one
kernel**, express the cross-GPU dependencies *inside* it with NVSHMEM's device API, and
— in the extreme — launch **one kernel for the whole 300-iteration run**. The CPU's role
shrinks to "start" and "collect the answer."

This chapter is the design of that benchmark (`acgnbench-nvdev.cu`, standalone
CUDA+MPI+NVSHMEM, deliberately PETSc-free — it is a *showcase* of what the hardware
permits, and its API lessons feed the wishlist rather than pretending to be a
deliverable), and the four protocol revisions it took to get it fast.

## 7.1 Communication as a load instruction

Recall chapter 3.2: on one NVLink node, `nvshmem_ptr(buf, pe)` returns a raw pointer
into a *peer GPU's* memory. So "rank 2 receives g0 from rank 0" is, literally:

```c
while (sig[SIG_G0] < v) { }     // poll a flag in MY OWN memory
                                //   (rank 0 set it, remotely, when data was ready)
x = g0_ptr[i];                  // an ordinary load; the address resolves to
                                //   rank 0's HBM, and the read crosses NVLink
```

No send, no receive, no library call, no CPU, no DMA engine — a message compiles down to
memory instructions. The producer's side is symmetrical: write your own buffer, make the
writes visible, then set the consumer's flag with `nvshmemx_signal_op(...)` (a remote
store with defined semantics). The price of this freedom: *you* now own ordering, flow
control, and reuse, at the GPU memory-model level. The rest of the chapter is exactly
those three problems.

## 7.2 The four correctness problems, and their standard solutions

**(a) Completion: how does the consumer know data is ready?** Signals: one 64-bit word
per edge in the consumer's symmetric heap. Producer writes data, executes
`__threadfence_system()` (a memory fence: "my writes are now visible system-wide,
including over NVLink"), *then* sets the flag. Consumer polls the flag, fences, then
reads. Fence placement is not optional; without it the flag can arrive before the data.

**(b) Reuse: iteration k+1 uses the same flags.** Never *reset* flags between
iterations — a reset is itself a write that races the next iteration's readers. Instead
make values **monotonic epochs**: at iteration `it`, the producer sets the flag to
`v = it+1` and the consumer waits for `flag >= v`. No resets, ever; a flag late by one
epoch is simply "not yet".

**(c) Buffer reuse: when may the producer overwrite its buffer?** Iteration k+2 writes
the same slot k used (we **double-buffer**: slot = it mod 2, so k+1 never touches k's
data). But k+2 must not start writing while some straggler is still *reading* k. Hence
**acknowledgment signals** flowing backward: after a consumer finishes reading iteration
it, it sets `ack = it+1` on the producer; the producer, before writing iteration it,
waits for all its consumers' acks to reach `it-1` (i.e., everyone is done with it-2, the
previous occupant of this slot):

```
 producer, iteration it:                     consumer, iteration it:
   wait: every ack >= it-1   <----------+      wait: flag >= it+1
   write slot (it mod 2)                |      fence; read slot (it mod 2)
   fence                                +----- set ack = it+1
   set consumers' flags = it+1  ------------>  (then use the data)
```

**(d) How do you know you got all of this right?** You will not, on the first try, and
the failure mode is silent wrong data. So the benchmark is **race-detecting by
construction**: every published element is stamped `role_base + epoch` (x1 carries
10+v, g0 carries 100+v, ...), and consumers validate *every element, every iteration*,
counting mismatches into a device counter. A missing fence, a premature overwrite, or a
stale read produces a nonzero count, not a plausible-looking benchmark. Final tally
across the whole campaign: **zero mismatches over ~10^8 validated element-reads** —
which is what made it safe to iterate aggressively on the protocol below.

## 7.3 One more ingredient: the cooperative kernel

Inside one fused kernel, "the whole GPU waits here for a flag" needs grid-wide
coordination — but thread blocks are normally independent (chapter 1.2) and cannot all
synchronize. CUDA's **cooperative launch** guarantees all blocks are resident
simultaneously and provides `grid.sync()`, a barrier across the entire kernel. Our first
protocol leaned on it heavily. Its cost became the story.

## 7.4 Four protocol revisions, each forced by a measurement

(64 KB, persistent mode, best of runs; graph-NCCL's 32 µs is the bar.)

**v1 — conservative-correct: 68–71 µs.** Every dependency point: thread 0 waits on the
flag, then `grid.sync()` broadcasts; every publish: write, fence, `grid.sync()`, thread
0 signals. ~10 grid barriers per iteration. It validated immediately — and profiling
condemned it: comm-only (zero compute) cost the same as the full iteration, so protocol
was the whole bill; and at 1 MB (264 blocks) it exploded to 283 µs because **grid.sync
cost scales with block count**. Also: persistent mode was no faster than per-iteration
launches — proof that launch overhead was NOT the problem, the inside of the kernel was.

**v2 — poll-don't-barrier on the wait side; cap the grid: 54 µs.** Waits became each
thread polling the flag word directly (it lives in local memory; the remote signal
updated it) — no barrier needed to *wait*. And since grid-stride loops let any block
count cover any vector length, we capped the grid at 32–64 blocks: 1 MB dropped 283 -> 150.

**v3 — kill the remaining publish/ack barriers: 74 µs. WORSE.** The elegant idea:
per-slot **arrival counters**. Each block, after fencing its share of a publish,
atomically increments a counter; the *last* block to arrive (increment returns N-1)
resets the counter and fires the signals; every other block streams ahead, self-gated
only by its next data dependency. Reset-before-signal is provably safe: nobody can
re-enter this slot (iteration it+2) before a consumer acks it, and that ack *requires
the very signal sent after the reset* — the ack chain is the lock. Zero grid-wide
barriers... and it got SLOWER than v2. Why: v3 also made *ack*-waits all-thread polls,
so ~8192 threads sat in tight loops loading the same 8-byte words — saturating the very
L2 cache that must serve the incoming NVLink signal writes and the peers' data pulls.
**The barrier was gone, but the detection path was self-jammed.**

**v4 — contention-free polling: 41 µs.** Same barrier-free structure; every poll is done
by **one leader thread per block**, which releases its block via `__syncthreads()`
(a block-local barrier — nanoseconds, not microseconds). 32 pollers instead of 8192.

```
 v1: |work|--GRID BARRIER--|work|--GRID BARRIER--|...        ~10 x ~1-3 us
 v4: block0: |work|poll^|work|...      ^ = leader polls flag, __syncthreads
     block1: |work|poll^|work|...        blocks decoupled; may even run
     block2: |work|-ahead-|poll^|...     SKEWED across iterations, coupled
                                         only by real dataflow
```

Final numbers (persistent = one launch per 300-iteration run):

| 64 KB | µs/iter |   | larger sizes (branch-pk) |
| --- | ---: | --- | --- |
| branch-MPI | 162 | | 512 KB: 88 1 MB: 122 4 MB: 382 |
| eager NCCL | 107 | | |
| graph NCCL | **32** | | |
| device NVSHMEM v4 | **41** | | heavy compute: 48 (compute is nearly free in-kernel) |

4x over MPI, 1.28x behind graph-NCCL. Our written prediction was 10–30 µs: just missed,
and the decomposition says why — the DAG's ~7 serialized signal hops x ~3–5 µs of
NVLink-signal+poll+pull each, plus ~7 µs compute; the realistic floor is high-20s. The
prediction's upper edge was right; its lower edge underestimated hop latency.

We then filled the last matrix cell: a device-side **allreduce** (each PE publishes its
contribution, then pulls and validates all three peers' lanes — "direct all-pull", one
signal round, latency-optimal at 4 ranks). It ties branch at 64 KB (42 vs 41) and *wins*
above (76 vs 88 at 512 KB) — the third independent transport confirming chapter 5's
conclusion, with a bonus insight: all-pull moves the DAG's late-consume reads off the
critical path.

## 7.5 What this arm is, and is not

It is: proof the execution model works (a frozen schedule really can run as a
device-resident program, validated); the only arm with literally zero per-iteration CPU
cost; and the only option when kernels must react to data mid-flight — a graph replays a
fixed recording, but a persistent kernel can *decide*.

That last claim was later made concrete (`-converge`): a synthetic residual decays by a
known factor each iteration, and every k iterations each PE grid-reduces its partial
norm, all-pulls the other three partials, and sums them **in fixed rank order — so all
four kernels compute the bit-identical global norm and therefore the identical stop
decision, with no extra agreement round.** An epoch-stamped decision word releases every
block, and all four resident kernels break their loops on the same iteration. Flow
control falls out for free: the decision poll gates every block, so buffer/counter reuse
is serialized without any ack signals. Measured: the kernel stops at *exactly* the
analytically predicted iteration (202 at k=1; 210 = the first check past 202 at k=10,
overshoot ≤ k−1 as chapter 6.3's arithmetic promised); checking every iteration costs
~11 µs, checking every 10th costs nothing measurable. **One kernel launch runs the whole
solve, including deciding when it is over** — the sentence that separates this execution
model from graph replay.

It is not: a deliverable. It is PETSc-less by construction, and the road from showcase
to supported feature is a named API wishlist item: symmetric-heap-resident vectors,
device-callable term kernels, the frozen schedule as compiler input.

One build note, because it bites everyone once: NVSHMEM's device functions live in a
static library (`libnvshmem_device.a`), and calling device functions across compilation
units requires **relocatable device code**: compile with `-rdc=true`, then run a
*device link* step (`nvcc -dlink`) to resolve device symbols, then the ordinary host
link. Forgetting the middle step yields the notorious
`undefined reference to __cudaRegisterLinkedBinary_...`.

Next: [Chapter 8 — leaving the node, and the kernel module that wasn't there](08-multinode-and-gpudirect.md).
