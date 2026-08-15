# Chapter 8 — Multi-node, GPUDirect, and diagnosing a fabric from userspace

Everything so far happened inside one node, over NVLink. Crossing to a second node
changes the physics: bytes now travel through a **NIC** (network interface card) onto a
switched fabric. This chapter is (a) the minimal theory of how a NIC and a GPU
cooperate, (b) the disciplined ladder we climbed when we finally got a 2-node job, and
(c) the trail of evidence showing every GPU-orchestrated transport blocked by one
missing kernel module.

## 8.1 Why a NIC cannot just read GPU memory

Modern fabrics use **RDMA** (remote direct memory access): the NIC itself copies memory
across the network without CPU involvement — but only memory it has been *taught about*.
Registering memory with a NIC means the OS kernel pins the pages and hands the NIC a
mapping it can DMA against. That machinery was built for CPU RAM. GPU HBM is not OS-page
memory; by default **a NIC has no way to address it at all**. Hence every GPU byte
crossing the network takes the scenic route:

```
 WITHOUT GPUDirect:   GPU HBM --PCIe--> host "bounce buffer" --NIC--> fabric --> ...
                                (extra copy, extra latency,     (reverse on
                                 CPU/DMA involvement)            the far side)

 WITH GPUDirect RDMA: GPU HBM ------------- NIC reads HBM directly ------> fabric
```

"GPUDirect RDMA" — the direct path — requires a kernel module that exposes GPU memory to
the NIC's registration machinery: **`nvidia_peermem`** (or, on newer stacks, the
`dmabuf` mechanism). No module, no direct path, for *every* library at once: NVSHMEM's
network transports refuse to start, NCCL falls back to bounce buffers, and even MPI/UCX
silently stages through the host.

## 8.2 The sanity ladder

Multi-node debugging has a strict order of operations, because each rung's failure
explains the next rung's weirdness. Ours, with results:

**Rung 1 — plain CPU MPI across nodes.** An 8-rank host-memory `MPI_Allreduce`. It
failed under default settings — and the failure was *already known from single-node
work*: UCX's `ud` transport dies on this fabric (`ibv_create_ah ... Connection timed
out`). Constraining UCX's transport list fixed it; `rc`, `dc`, and `tcp` all pass. One
subtlety worth stealing: transport lists are **per-regime** — adding `rc` for the
network made *intra-node* small messages slower (UCX then routes them through NIC
loopback), so single-node runs keep the shared-memory-only list.

**Rung 2 — GPU-aware MPI across nodes.** A timed GPU ping-pong: ~19 µs one-way small-
message latency, ~39 GB/s at 16 MB. Works — but that 39 GB/s *is the host-staged path*
(peermem is absent). Note it for later: the wire and the staging pipeline are provably
capable of 39 GB/s. Numbers like this are the control experiments that make later
diagnoses stick.

**Rung 3 — NVSHMEM across nodes.** Four remote transports tried, four distinct
failures, one root cause:

| transport | failure |
| --- | --- |
| `ibrc` (default) | "neither nv_peer_mem, or nvidia_peermem detected. Skipping transport." |
| `ibdevx` | same message, its own check |
| `ibgda` | needs an extra enable flag; then the same peermem check |
| `ucx` | initializes, then fails to register the symmetric heap with the NIC |

And the modern escape hatch is closed too: the driver reports
`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0`. Meanwhile `modinfo nvidia_peermem` shows
the module *ships with the installed driver* — it is simply not loaded. That
combination — module present, not loaded, no dmabuf, four independent failure
signatures, plus rung 2's proof the fabric itself is healthy — is a complete,
deflection-proof support ticket. Writing it as one (`ALCF-TICKET-PEERMEM.md`) was a
campaign deliverable in its own right.

## 8.3 NCCL across nodes: a config trap, then a hard floor

NCCL *can* cross nodes without peermem (bounce buffers), so we measured it. First
result: **78,000 µs per iteration** — 250x worse than expected. Not a typo; a trap:

The nodes' NICs are a *bonded pair* (two physical ports acting as one logical `bond0`).
NCCL enumerates all InfiniBand-verbs devices and happily uses the raw member ports
alongside the bond — and on this fabric the raw ports have no valid routes, so every
message waits out retransmission timeouts. One environment variable
(`NCCL_IB_HCA=mlx5_bond_0`: "use only the bond") turns 78 ms into 297 µs.
**Lesson: a transport that runs is not a transport that is configured; sane-looking
defaults can be catastrophically wrong on unusual fabrics.**

With that fixed, the honest numbers (64 KB / 4 MB per iteration): MPI 206 / 886 µs,
NCCL 296 / 11,709 µs. NCCL's bounce-buffer path manages only ~2.7 GB/s where UCX's
staging does 39 GB/s over the same wire. Is that tunable? We swept every plausible knob
— queue pairs, channel counts, chunk sizes, buffer sizes, even loading the vendor's
alternative NCCL network plugin (which required extracting a missing system library
from an RPM into userspace). **Every configuration landed within 1% of the same
number.** Knob-invariance is itself diagnostic: the limiter is inside the protocol, not
its parameters, and no userspace action can fix it. The fix is the kernel module —
root's, and the ticket's.

## 8.4 What multi-node did to the algorithm story

The 2+2 placement (ranks 0,1 on node A; 2,3 on node B) pushes 8 of the DAG's 12 edges
across the fabric. Under MPI: branch still beats allreduce inter-node — 21% at 64 KB,
shrinking to 6% at 4 MB as both saturate the same NIC. So chapter 5's conclusion holds
in refined form: *under a host-blocking transport, splitting keeps paying, and pays most
when messages are latency-bound.* The GPU-orchestrated transports could not be tested
fairly inter-node at all — which is precisely what the ticket, once acted on, unblocks:
NVSHMEM's entire multi-node existence, NCCL's fast path, and a fair GPUDirect MPI
baseline, all gated on the same `modprobe`.

Next: [Chapter 9 — the full matrix, and what we actually learned](09-results-and-lessons.md).
