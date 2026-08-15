# Chapter 3 — How GPUs talk: three transports, three philosophies

Recall the setup: data in GPU2's HBM must appear in GPU3's HBM, over NVLink. There are
three mainstream ways to make that happen, and they differ not in the wire — all use the
same NVLink — but in **who is in charge and who has to wait**.

## 3.1 GPU-aware MPI: the CPU is in charge, and it waits

Classic MPI is two-sided: the sender calls `MPI_Send`, the receiver calls `MPI_Recv`,
and the library matches them. "**GPU-aware**" MPI means you may pass *GPU memory
pointers* to those calls and the library (via a layer called UCX) figures out a good
path — on our node, a direct GPU-to-GPU copy over NVLink ("`cuda_ipc`").

GPU-aware does **not** mean GPU-driven. The CPU still executes every MPI call, and — the
crucial part — MPI has no concept of streams. If the data an `MPI_Send` wants to send is
still being produced by a kernel sitting in a stream, the CPU must first **block** until
that kernel finishes:

```
      HOST (CPU)                                GPU stream            NVLink
      ----------                                ----------            ------
  1   enqueue pack kernel  --------------->     [pack]
  2   cudaStreamSynchronize()  #### CPU BLOCKED, waiting for pack to FINISH ####
  3   MPI_Isend/Irecv (pointers now safe) --------------------------->  data moves
  4   MPI_Waitall()            #### CPU BLOCKED until data arrives ####
  5   enqueue unpack kernel  ------------->     [unpack]
```

Cost profile (measured, chapter 4's benchmark): about **11.5 µs** per exchange on this
node. Only 2 stream ops (pack, unpack — cheap by the 2 µs law), but two CPU blocks. On
one node the blocks are short, because NVLink is fast and the pack has usually already
finished. Keep in mind for chapter 8: *on two nodes, block #4 becomes a real network
round trip, and no amount of GPU power can hide a blocked CPU.*

## 3.2 NVSHMEM: a shared address space, and two very different APIs

NVSHMEM implements the **PGAS** (Partitioned Global Address Space) model, and the core
trick is worth understanding precisely.

At startup, every rank (NVSHMEM calls them **PEs**) allocates a **symmetric heap**: a
memory region of identical layout on every GPU. Allocation is collective — everyone
calls `nvshmem_malloc(same size)` together — so "the buffer `x` on PE 2" is a
well-defined remote address that any PE can name:

```
   PE0 heap        PE1 heap        PE2 heap        PE3 heap
   +-------+       +-------+       +-------+       +-------+
   |  x    |       |  x    |       |  x    |       |  x    |   same offsets
   |  sig  |       |  sig  |       |  sig  |       |  sig  |   everywhere
   +-------+       +-------+       +-------+       +-------+
       ^               ^               ^               ^
       +------- every PE can read/write every OTHER PE's copy -------+
```

On a single NVLink node the implementation is beautifully direct: NVSHMEM maps each
GPU's heap into every peer's address space (CUDA's inter-process memory sharing), and
`nvshmem_ptr(x, 2)` hands you **a raw pointer that dereferences straight into GPU2's
HBM over NVLink**. One-sided: PE 2 does not participate, does not call anything, is not
interrupted.

The two APIs sit at opposite ends of the machine:

- **Host API** (used by PETSc's built-in support, chapter 4): the *CPU* enqueues
  operations onto streams — `nvshmemx_putmem_nbi_on_stream(...)` "put this buffer into
  that PE's heap", `nvshmemx_signal_op_on_stream(...)` "then set a flag over there".
  Never blocks the CPU. But every enqueue pays the 2 µs law, and correctness needs
  several of them per exchange.
- **Device API** (chapter 7): the same operations as `__device__` functions called
  *from inside a running kernel* — plus the raw-pointer trick above, where communication
  is not a call at all but an ordinary load/store instruction executed by a GPU thread.
  Zero CPU involvement, zero enqueue cost. The catch: now *you* are the runtime —
  ordering, flow control, and completion are your problem, at GPU memory-model level.

One-sided communication needs one more ingredient: since the receiver never "receives",
how does it know data has arrived? **Signals**: one-word flags in the symmetric heap.
The producer writes data, then sets the consumer's flag; the consumer polls the flag
before reading. Getting this exactly right (fences, monotonic values, reuse across
iterations) is most of chapter 7.

## 3.3 NCCL: a specialist that batches

NCCL is NVIDIA's collectives library (it powers large-scale ML training). Its model:
build a **communicator** over the GPUs once, then enqueue communication operations that
execute *as kernels* on your stream. The CPU never blocks; ordering with your compute is
just stream ordering.

Its killer feature for our DAG is the **group**:

```c
ncclGroupStart();
if (rank == 0) { ncclSend(g, n, ..., /*to*/2, comm, s); ncclSend(g, n, ..., 3, comm, s); }
if (rank == 2) { ncclRecv(g0, n, ..., 0, comm, s);  ...  ncclSend(g, n, ..., 3, comm, s); }
if (rank == 3) { ncclRecv(g0, ...); ncclRecv(g1, ...); ncclRecv(g2, ...); }
ncclGroupEnd();       // <- everything above becomes ONE fused kernel on this rank
```

All sends/receives between Start and End are aggregated into (typically) a single kernel
launch per rank. A whole DAG *stage* — five messages — becomes **one** stream operation.
By the 2 µs law, that is the entire game. Note also what the `if (rank == ...)` shows:
NCCL point-to-point ops are not collective — each rank's program lists only its own
edges, which is exactly the "different program per rank" freedom our DAG wants.

## 3.4 The scorecard

| | GPU-aware MPI | NVSHMEM host API | NVSHMEM device API | NCCL |
| --- | --- | --- | --- | --- |
| who initiates | CPU | CPU (enqueues) | GPU threads | CPU (enqueues) |
| does the CPU block? | **yes, twice** | no | no (not even involved) | no |
| stream ops per exchange | 2 + 2 blocks | ~5, 4 after fusing (ch. 4) | **0** | ~1 per stage (fused) |
| sidedness | two-sided | one-sided | one-sided | two-sided, but per-rank programs |
| granularity | per message | per message | per load/store | per *group* |
| can it enter a CUDA graph? (ch. 6) | no | no | n/a (it *is* the kernel) | yes |

Every row of that table becomes a measured number in the chapters that follow.

Next: [Chapter 4 — what happens when a real library uses the host API](04-petscsf-nvshmem-and-the-2us-law.md).
