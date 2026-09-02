# Chapter 8 — Multi-node, GPUDirect, and diagnosing a fabric from userspace

Everything so far happened inside one node, over NVLink. Crossing to a second node
changes the physics: bytes now travel through a **NIC** (network interface card) onto a
switched fabric. This chapter is (a) the minimal theory of how a NIC and a GPU
cooperate, (b) the disciplined ladder we climbed when we finally got a 2-node job, and
(c) the trail of evidence showing every GPU-orchestrated transport blocked by one
missing kernel module -- and, in 8.5, what happened when the module arrived, including
the part of our own diagnosis that turned out to be wrong.

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

*(Correction, 8.5: this paragraph's reading of the devices was wrong. `mlx5_bond_0` is the
25 GbE management bond; the "raw member ports" are the 400 Gb/s data ports; the "fix" below
moved NCCL onto the management network. The lesson stated at the end of the paragraph is,
ironically, still right.)*

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

## 8.5 Epilogue (2026-09-01): the module arrived, and so did a correction

Two weeks after the ticket, the admins installed a proper OFED stack and a working
`nvidia_peermem`. The ladder of 8.2 was climbed again on a fresh 2-node job. Three things
happened, in this order: NVSHMEM crossed nodes for the first time; NCCL got 12x faster
for a reason that had nothing to do with peermem; and we discovered that chapter 8.3 had
described the machine wrongly.

### 8.5.1 What we had assumed, and what the hardware says

```
   WHAT 8.3 ASSUMED                             WHAT lspci / link rate / ethtool SAY
   (from device NAMES alone)                    (2026-09-01, both nodes)

   +-----------------------------+             +------------------------------------------------+
   | node                        |             | node          NUMA 0          |    NUMA 1      |
   |  GPU0 GPU1 GPU2 GPU3        |             |  GPU0   GPU1                  |  GPU2   GPU3   |
   |    \   |    |   /           |             |   |      |                    |   |            |
   |   +----+----+----+          |             | mlx5_1  mlx5_0                | mlx5_bond_0    |
   |   |  mlx5_bond_0  |  "the   |             | 400G    400G RoCE             | 25 GbE, = bond0|
   |   |  = mlx5_0 +   |   NIC"  |             | (link   THE DATA PORT         | = mgmt0+mgmt1  |
   |   |    mlx5_1     |         |             |  down   BlueField-3 / CX-7    | ConnectX-5     |
   |   +-------+-------+         |             |  on one                       | MANAGEMENT     |
   |           |                 |             |  node)   |                    |     |          |
   +-----------|-----------------+             +----------|--------------------|-----|----------+
               v                                          v                          v
          "the fabric"                            400 Gb/s data fabric        25 Gb/s management
                                                  (UCX had been here          network (NCCL had
                                                   all along: 39 GB/s)         been here: 2.7 GB/s
                                                                               = 25 Gb/s line rate)
```

The name `mlx5_bond_0` and the existence of `bond0` had been read as "the two data ports are
bonded". `/proc/net/bonding/bond0` says the bond's members are `mgmt0` and `mgmt1`, 25 Gb/s
each, active-backup; `lspci` says that device is a ConnectX-5 and the other two are
BlueField-3 400G ports; `/sys/class/infiniband/*/ports/1/rate` says 400 vs 25. The
arithmetic we should have done in August: 2.7 GB/s is 21.6 Gb/s, i.e. a 25 Gb/s link running
near line rate. **A number that matches a link speed is a fingerprint, and we did not check
it against the wrong link.**

### 8.5.2 Why the module's presence is not enough: who can *see* it

```
   kernel                                 sysfs                                     user space
   nvidia_peermem.ko --registers with--> ib_uverbs (DOCA-OFED 26.04)
        |                                     |
        |                                     +-- /sys/kernel/mm/memory_peers/nv_mem/version   ABSENT here
        +-- /sys/module/nvidia_peermem/version                                              PRESENT
                                                    |
        NVSHMEM 3.4.5    probes both paths ---------+---> detected --> IBRC / IBGDA transports start
        NCCL 2.28        probes both paths ---------+---> detected --> "GDRDMA" channels
        UCX 1.19 (DOCA)  probes both paths ---------+---> detected --> cuda memory registrable
        UCX 1.17 (HPC-X) probes memory_peers ONLY --X---> "GPUDirect RDMA is disabled" --> MPI stays host-staged
```

Each library asks the kernel a slightly different question. The one our MPI ships with
asks the old question, so GPU-aware MPI through HPC-X did not change at all (19 us / 39 GB/s
before and after). Preloading the newer UCX from the vendor's RPM repository (as a plain
user: extract with `rpm2cpio`, `LD_PRELOAD` the four libraries, from a path visible on
*every* node) gave MPI the direct path: 10.8 us / 47 GB/s. Two traps on the way: setting
`LD_LIBRARY_PATH` did nothing (the MPI's own components carry a search path that wins), and
the first preload lived on one node's local `/tmp`, so rank 0 ran the new UCX and rank 1 the
old one -- and the run *worked*, and printed a number, and the number was meaningless.
Mixed versions that interoperate are worse than ones that crash.

### 8.5.3 What a byte's journey costs now, per mechanism (measured, one-way, 8 bytes)

```
   MPI, host-staged (HPC-X UCX 1.17)                                 19.4 us
   GPU src  |D2H copy|
   CPU src  |        |stage, post RDMA|
   wire     |                         |====== 400G ======>|
   CPU dst  |                                             |recv|H2D copy|
   GPU dst  |                                                           |data|

   MPI, GPUDirect (UCX 1.19 preloaded)                                10.8 us
   GPU src  |post (NIC reads HBM directly)|
   wire     |                             |====== 400G ======>|
   GPU dst  |                                                 |data lands in HBM|

   NVSHMEM IBRC: GPU asks a CPU "proxy" thread to post                10.1 us
   GPU src  |kernel: enqueue put|
   CPU proxy|                   |poll|post RDMA write|
   wire     |                                        |====== 400G ======>|
   GPU dst  |                                                            |data|signal|

   NVSHMEM IBGDA hybrid: GPU writes the NIC's queue, CPU rings the bell  16.1 us
   GPU src  |kernel: write WQE in HBM|
   CPU      |                        |ring doorbell|
   wire     |                                      |====== 400G ======>|
   GPU dst  |                                                          |data|signal|

   NVSHMEM IBGDA, true: GPU rings the bell itself             not available (driver parameter)
   GPU src  |kernel: write WQE, ring doorbell|
   wire     |                                |====== 400G ======>|
   GPU dst  |                                                    |data|signal|

   for scale: the same put over NVLink inside a node                    2.0 us
```

Two surprises. First, with a modern UCX, **MPI and NVSHMEM have the same transport floor**
(10-11 us); NVSHMEM's raw advantage over MPI exists only against the host-staged path. Second,
IBGDA -- the "GPU talks to the NIC directly" transport this whole campaign had been waiting
for -- initializes now, but the GPU is not allowed to map the NIC's doorbell page (a driver
module parameter the admins have not set), so NVSHMEM silently runs a hybrid where a CPU
thread rings the doorbell. That hybrid is *slower* than the proxy it was meant to replace.
A transport that "works" still needs its numbers checked against the one it replaces.

### 8.5.4 The smoke test that had been lying to us

```
   PE 3 (node 1)                                        PE 4 (node 2)
   src = cudaMalloc(...)  --nvshmem_int_put(dst, src)--> dst = nvshmem_malloc(...)

   inside a node : P2P -- the GPU load/stores through an NVLink mapping; ANY source pointer works  -> OK
   across nodes  : the proxy posts an RDMA write; the NIC needs a registration key for src;
                   an unregistered source is a "local protection error" (IBV_WC_LOC_PROT_ERR)  -> FAIL

   fix: src = nvshmem_malloc(...)   (or nvshmemx_buffer_register(src, bytes))
```

The first cross-node attempt failed with a protection error and, for an hour, looked like
a peermem problem. It was our test: NVSHMEM's *destination* must be symmetric memory, but
so must the *source* whenever an IB transport carries the put, and our test had used an
ordinary `cudaMalloc` source since July -- it had always passed because NVLink does not
care. PETSc's own path was never exposed (both ends of every exchange are symmetric), which
is exactly what the fixed test then proved: the library's fetch-and-op test and the
lid-cavity solve are bit-identical to MPI at 8 ranks across 2 nodes.

### 8.5.5 NCCL: the second "config trap", and it was ours

```
   NCCL default on the 400G port: RoCE v2 (UDP-encapsulated, GID index 3)  --> 78,000 us / iteration
       tried: NCCL_IB_TC, GID 1/3, QPs=1, no adaptive routing, GDR off, relaxed ordering  --> all ~78,000
   NCCL_IB_ROCE_VERSION_NUM=1 NCCL_IB_GID_INDEX=2  (RoCE v1, no IP header)   -->     205 us / iteration
   UCX (rc) and NVSHMEM (ibrc) choose their own GIDs and were never affected.
```

Chapter 8.3's "bond-slave dead routes" were NCCL's RoCE v2 packets stalling on the data
port. Pinning NCCL to the "bond" had made the stall go away by moving NCCL to a different,
25x slower, network -- a fix that worked for the wrong reason. With RoCE v1 on the real
port, the 4 MB iteration went from 11.7 ms to 1.0 ms and NCCL became the fastest eager
transport across nodes.

### 8.5.6 The 2+2 placement, and what crosses

```
   node 1: ranks 0, 1                          node 2: ranks 2, 3
   +-------------------+                       +-------------------+
   |   r0        r1    |                       |   r2        r3    |
   +-------------------+                       +-------------------+
   x1 :  1 -> 0 (local)              1 -> 2, 1 -> 3                     2 cross
   g  :  0 -> 2, 1 -> 2, 0 -> 3, 1 -> 3       (2 -> 3 local)            4 cross
   x2 :  2 -> 1                              (2 -> 3 local)            1 cross
   x3 :  3 -> 1                              (3 -> 2 local)            1 cross
                                                                  8 of 12 edges cross;
                                                                  node 1 sends 6 messages per
                                                                  iteration through ONE 400G port
```

### 8.5.7 The matrix across the fabric (64 KB per message, us per iteration)

```
   10 us per '#'          0         100        200        300
   graph-NCCL    1 node   |###  32
                 2 nodes  |############  122
   eager NCCL    1 node   |###########  107
                 2 nodes  |##################  177
   MPI, staged   1 node   |################  162
                 2 nodes  |#####################  206
   MPI, GPUDirect 2 nodes |###################  192
   PetscSF-NVSHMEM 1 node |#####################  206
                 2 nodes  |############################  280
   IBGDA hybrid  2 nodes  |################################  320
   device NVSHMEM 1 node  |####  41              (2 nodes: not built yet -- chapter 9.5)
```

The ordering of the arms does not change when the fabric is real; every arm pays 44-90 us
for its 8 crossing edges; and the PetscSF-NVSHMEM arm is still last at small sizes even
though its transport is now as fast as anyone's. Chapter 4's diagnosis -- the cost is the
library's per-exchange protocol, not the wire -- survived the one experiment that could
have overturned it.

Next: [Chapter 9 — the full matrix, and what we actually learned](09-results-and-lessons.md).
