# Chapter 6 — CUDA graphs: record once, replay forever

After chapter 5 the iteration costs ~18 stream operations x ~2 µs of CPU enqueue tax —
every iteration, forever, for work whose *shape never changes*. CUDA graphs are the
mechanism for paying that tax once.

## 6.1 What stream capture actually does

`cudaStreamBeginCapture(stream)` flips the stream into a recording mode: subsequent
"launches" on it **do not execute** — they are appended, with their dependencies, to a
graph object. `cudaStreamEndCapture` hands you the graph; `cudaGraphInstantiate` bakes
it into an executable; `cudaGraphLaunch` replays the whole recording as **one** enqueue:

```
 EAGER (every iteration):                      GRAPH (once):
 CPU: |a|b|c|d|e|f|g|h|...|r|   ~18 x 2 us     capture:  [a..r] -> graph -> exec
                                               then per iteration:
                                               CPU: |launch|            1 x ~5-10 us
                                               GPU: [a b c d ... r]     same kernels,
                                                     back-to-back, no enqueue gaps
```

Crucial nuance (it came up repeatedly in discussion): **replay is one *launch* of a
recorded multi-kernel sequence — the kernels remain distinct on the GPU.** It is not
kernel fusion. What dies is the per-op CPU cost and the inter-op launch gaps; what
survives is each kernel's own execution. (True fusion — many operations inside ONE
kernel — is chapter 7.)

Both PETSc compute kernels and NCCL communication kernels are capturable (NCCL supports
capture since 2.9). MPI is **not** — its host-blocking calls are CPU code, not stream
work; there is nothing to record. This is structural: the eager-only transports of
chapter 3's scorecard can never enter this regime.

## 6.2 The trap: the stream that refuses to be recorded

First attempt: `cudaStreamBeginCapture` on PETSc's stream ->
`cudaErrorStreamCaptureUnsupported`, instantly.

Diagnosis: PETSc's default device context uses the **legacy default stream** (chapter
1.3) — internally it never creates a stream at all and hands out the NULL handle, and
the NULL stream's implicit-synchronization semantics are incompatible with capture, so
CUDA refuses. The fix is to make PETSc use a real, nonblocking stream; two equivalent
routes (both verified):

- command line: `-root_device_context_stream_type nonblocking`
- programmatic: retype the global device context to `PETSC_STREAM_NONBLOCKING`, set it
  up, and re-set it as current (which also refreshes PETSc's cached stream handle).

Worth generalizing: **libraries default to the legacy stream more often than you would
think, and "capture-safety" is a real property** a codebase must be audited for — no
host syncs, no allocation, no MPI, no NULL stream anywhere in the captured region. That
audit item went into the API wishlist as a first-class requirement.

Second practical rule: **warm up eagerly before capturing.** First-use costs (NCCL
channel connection, memory-pool growth, lazy init) must happen *before* recording, or
they get baked into the graph or break the capture. We run 30 eager iterations, then
capture iteration 31 as the template.

## 6.3 "But we don't know how many iterations until convergence!"

The sharpest question asked of this design, and it deserves a precise answer.

An iterative solver runs x_{k+1} = T(x_k) until a data-dependent stopping test passes.
Capture does NOT require knowing the iteration count, because **the graph records the
map T, not the solve**. Replay = apply T once; the host decides at runtime how many
times to launch. What capture *does* require is that T itself is data-independent:
fixed step sizes, fixed schedule, no line search, no adaptive anything. For our
algorithm that is guaranteed by the poster's own doctrine — coefficients and schedule
are frozen before the solve begins. The freeze is not just a performance trick; it is
the *mathematical license* to compile the iteration.

The stopping test is data-dependent control flow, which a recording cannot contain. The
standard resolution is chunking:

```
 while (not converged):
     cudaGraphLaunch( graph of k iterations )    # no syncs inside
     read residual norm from device; test        # ONE host sync per k iterations
```

The residual *computation* can live inside the graph (it is a fixed computation into a
device buffer); only the read-and-decide syncs. Costs: one sync per k iterations, and an
overshoot of at most k-1 iterations past the criterion — harmless for a nonexpansive
fixed-point map (extra applications of T do not corrupt the iterate; they waste at most
k-1 iterations of work). Even this is removable: CUDA >= 12.4 has conditional graph
nodes (device-evaluated while-loops), and chapter 7's persistent kernel can evaluate the
test on-device outright.

Our benchmark's `-graph_iters k` knob is exactly this chunk structure; k=1 and k=10
measured within ~4% of each other, so the tradeoff is nearly free to tune here.

## 6.4 Results — the campaign's fastest numbers

64 KB, light compute, best-of runs (validation PASS throughout):

| arm | µs/iter |
| --- | ---: |
| branch over MPI | 162 |
| branch over NCCL, eager | 107 |
| **branch over NCCL, graph replay** | **32** |
| allreduce over NCCL, graph replay | 34 |
| compute-only floor, eager | 39 |

Read that fourth row again: the graph arm is **faster than the eager compute-only
floor**. Not a paradox — the "floor" was itself ~14 launch-bound AXPY enqueues; replay
erases the tax for compute and communication alike, leaving ~20 µs of AXPY execution
plus ~12 µs of NCCL kernels. At heavy compute the effect compounds (graph 32 µs vs 273
eager-NCCL vs 369 monolithic): when everything is small and launch-bound, the recording
wins by an order of magnitude.

Two footnotes for honesty: at 4 MB the win shrinks (227 vs 363) because execution, not
enqueue, dominates there; and the allreduce-ties-branch finding of chapter 5 persists
under graphs — 32 vs 34 at 64 KB, allreduce ahead at every larger size.

Next: [Chapter 7 — remove the CPU entirely](07-device-side-nvshmem.md).
