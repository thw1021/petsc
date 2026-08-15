# Chapter 1 — The machine, and the ~2 µs law

Before any communication library makes sense, you need an accurate mental model of what
a GPU compute node *is*. Most surprises in this campaign trace back to one fact: **a GPU
node is not one computer. It is five computers in a box, connected by cables of very
different quality.**

## 1.1 Anatomy of a node

Each Janus compute node looks like this:

```
                            ONE COMPUTE NODE
   +---------------------------------------------------------------------+
   |                                                                     |
   |   +----------------+        PCIe         +--------+                 |
   |   |  CPU ("host")  |======================|  NIC   |===> to other   |
   |   |  64 cores      |                      | (2x    |     nodes      |
   |   |  754 GB RAM    |                      |  bonded)|    (ch. 8)    |
   |   +-------+--------+                      +--------+                 |
   |           | PCIe (control + data, ~50 GB/s)                          |
   |     +-----+------+------+------+                                    |
   |     |            |      |      |                                    |
   |  +--v---+   +----v-+  +-v----+ +v-----+                             |
   |  | GPU0 |   | GPU1 |  | GPU2 | | GPU3 |    each: H100,              |
   |  | HBM  |   | HBM  |  | HBM  | | HBM  |    ~80 GB HBM memory,       |
   |  +--+---+   +--+---+  +--+---+ +--+---+    132 "SMs" (see 1.3)      |
   |     |          |         |        |                                 |
   |     +==========+=========+========+   NVLink mesh: EVERY pair of    |
   |          all-to-all, ~159 GB/s        GPUs directly connected       |
   |          per direction per pair                                     |
   +---------------------------------------------------------------------+
```

Five computers: the CPU and the four GPUs. Each has its **own memory** (the CPU's RAM;
each GPU's HBM). No one can casually read anyone else's memory; every byte that moves
between them moves over one of those links, and the links differ enormously:

- **NVLink** (GPU <-> GPU, same node): ~159 GB/s each way per pair, sub-microsecond
  latency. The best cable in the building.
- **PCIe** (CPU <-> GPU): ~50 GB/s, used for control messages and staging.
- **The network** (node <-> node): a different world entirely — chapter 8.

On this machine we run **one MPI rank per GPU**: four CPU processes, each "owning" one
GPU. When we say "rank 2 sends to rank 3" we mean "some data in GPU2's HBM must end up
in GPU3's HBM."

## 1.2 The GPU does nothing on its own

A GPU is a passive coprocessor. It sits idle until the CPU sends it work. The unit of
work is a **kernel**: a C++ function marked `__global__` that the CPU *launches* onto
the GPU, where it is executed by thousands of threads in parallel.

```c
__global__ void axpy(double *y, const double *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // "which thread am I?"
  if (i < n) y[i] += 1e-8 * x[i];
}

// on the CPU:
axpy<<<32, 256>>>(y, x, n);   // launch: 32 blocks x 256 threads = 8192 threads
```

That triple-angle-bracket syntax `<<<blocks, threads>>>` is the **launch configuration**:
how many thread *blocks*, and how many threads per block. Threads within a block can
synchronize with each other cheaply (`__syncthreads()`) and share fast memory; blocks
are independent and are scattered across the GPU's 132 **SMs** (streaming
multiprocessors — think "cores that each run many threads").

Keep that syntax in mind: in chapter 4 you will meet the pathological-looking launch
`<<<1,1>>>` — one block containing one thread — and understand why a library does that
on purpose.

## 1.3 Streams: the work queue, and why "launch" returns immediately

The CPU does not wait for a kernel to finish. A launch just **enqueues** the kernel onto
a **stream** — a FIFO work queue that the GPU drains in order — and returns to the CPU
caller in about two microseconds. This is called **asynchronous** or **stream-ordered**
execution:

```
 CPU:   |launch A|launch B|launch C| ... goes off to do other things ...
             \        \        \
              v        v        v          (enqueue only; ~2 us each)
 stream: [ A ][ B ][ C ]  ---> GPU executes them one after another, in order
```

Two consequences that shape everything in these notes:

1. **Ordering within a stream is free.** If B needs A's output, just enqueue B after A
   on the same stream. No explicit synchronization needed.
2. **The CPU only learns that work FINISHED if it explicitly blocks**, e.g.
   `cudaStreamSynchronize(stream)` ("wait until this queue is empty") or
   `cudaDeviceSynchronize()` ("wait until this whole GPU is idle"). Blocking the CPU is
   sometimes necessary — and, as chapter 3 shows, it is the defining cost of MPI-style
   communication.

There is also a special stream you get if you never create one: the **legacy default
stream** (stream `0`, the "NULL stream"). It has weird extra synchronization semantics
with every other stream, and — a fact that cost us an afternoon in chapter 6 — some CUDA
features refuse to work with it at all.

## 1.4 The ~2 µs law

Here is the single most important measurement of the whole campaign. We wrote a
microbenchmark (`launchcost.cu`) that timed the CPU-side cost of enqueueing things on an
H100:

| operation (enqueue cost, CPU side) | time |
| --- | ---: |
| launching an empty kernel (pipelined) | **1.91 µs** |
| a pair of event operations (record + wait, x2) | 0.53 µs |
| a "5 kernel launches + 4 event ops" sequence | 10.22 µs |

And later (`sigcost.cu`), the same for NVSHMEM's own primitives:

| operation | time |
| --- | ---: |
| `<<<1,1>>>` kernel performing a signal write | 2.43 µs |
| NVSHMEM host API: enqueue a signal write on a stream | 2.55 µs |
| NVSHMEM host API: enqueue a signal wait on a stream | 2.24 µs |

Notice what this says: **every stream-ordered operation costs ~2 µs of CPU time to
enqueue, no matter what it is.** An empty kernel, a useful kernel, a fancy library call
that "avoids kernel launches" — all the same. The library call is not cheaper because,
under the hood, it enqueues an internal kernel anyway.

Now scale that against the work we care about. Adding two vectors of 8192 doubles
(64 KB) takes the GPU well under a microsecond of actual arithmetic. So at the message
sizes our algorithm uses, **the enqueue tax exceeds the work**. An iteration made of ~18
small operations pays ~36 µs of CPU enqueue cost before a single flop matters.

This is the law that explains the entire campaign:

> You cannot make a stream operation cheaper. You can only issue fewer of them.

Chapter 4 shows a library losing because it issues five ops where two would do; chapter
5 shows NCCL winning by batching many transfers into one op; chapter 6 shows CUDA graphs
winning bigger by replaying a *recording* of many ops for the price of one; chapter 7
shows the endgame — one kernel for an entire 300-iteration solve.

## 1.5 Getting onto the machine: jobs, ranks, and one vicious footgun

Practical HPC mechanics, briefly:

- **Login nodes vs compute nodes.** You ssh into a login node, which has no (or one)
  GPU. Real nodes are requested from the batch scheduler (PBS here):
  `qsub -I -l select=2:ncpus=64:ngpus=4 ...` gives an interactive session on 2 nodes.
- **`mpirun -np 4` starts 4 processes**; each picks a GPU via "my local rank mod number
  of GPUs" — rank 0 -> GPU0, rank 1 -> GPU1, etc.
- **Vet the node.** One of our earlier sessions silently ran all 4 ranks on ONE GPU
  because a previous user's crashed job had wedged three of them. `nvidia-smi` showing
  4 idle GPUs, and a probe calling `cudaGetDevice()` from each rank, are cheap insurance.

The footgun: **CPU core binding.** `mpirun` by default pins each rank to a single CPU
core. NVSHMEM runs a hidden helper thread (a "proxy") that spin-polls — burns CPU
continuously. Two threads fighting for one core made *compute* routines 30x slower in
our early tests, in a way that looked exactly like "NVSHMEM communication is slow":

```
   1 core per rank:      [main thread | proxy | main | proxy | ...]  <- 30x slowdown,
                                                                        lands on MPI
                                                                        reductions (!)
   --bind-to none:       main thread on core A, proxy on core B      <- normal speed
```

Every measurement in these notes uses `mpirun --bind-to none`. The general lesson: when
a library runs background threads, a benchmark under default binding measures CPU
starvation, not the library.

Next: [Chapter 2 — the algorithm we actually want to run](02-the-problem.md).
