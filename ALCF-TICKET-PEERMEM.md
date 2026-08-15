# ALCF ticket draft: please load `nvidia_peermem` on the Janus H100 nodes

Prepared 2026-08-15 from measurements on `x2002c0s9b0n0` + `x2003c0s1b0n0` (2-node PBS job)
and `x2003c0s9b0n0` (earlier single-node work). Everything below was measured directly;
supporting detail is in `NVSHMEM-BUILD-NOTES.md` sections 13.2-13.3 and
`NVSHMEM-PERF-NOTES.md` section 21 in this worktree.

---

## Ready-to-paste ticket body

> **Subject: GPUDirect RDMA unavailable on Janus H100 nodes -- `nvidia_peermem` module
> present but not loaded**
>
> On the Janus H100 compute nodes, the `nvidia_peermem` kernel module ships with the
> installed driver but is not loaded:
>
>     $ modinfo nvidia_peermem
>     filename: /lib/modules/5.14.0-687.22.1.el9_8.x86_64/kernel/drivers/video/nvidia-peermem.ko
>     version:  610.57.04
>     $ lsmod | grep -i peermem
>     (nothing; only nvidia and nvidia_uvm are loaded)
>
> Without it, no GPUDirect RDMA path exists on these nodes, which blocks three things:
>
> 1. **Multi-node NVSHMEM is impossible.** Every NVSHMEM 3.4.5 remote transport fails:
>    `ibrc`, `ibdevx`, and `ibgda` each print
>    `neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.`;
>    the `ucx` transport initializes but then fails to register the symmetric heap with
>    the NIC (`Failed to map memory in UCX transport`). 8-PE 2-node runs abort with
>    `building transport map failed`.
> 2. **NCCL inter-node runs at ~2.7 GB/s instead of wire speed.** Without GPUDirect,
>    NCCL falls back to host bounce-buffer staging. We verified this is not tunable from
>    user space (QPs, channels, chunk/buffer sizes, HPC-X `nccl_rdma_sharp_plugin` -- all
>    within 1% of the same 2.7 GB/s), while UCX host-staging reaches ~39 GB/s on the same
>    wire, so the NIC and fabric are healthy.
> 3. **GPU-aware MPI (UCX) itself is host-staged** -- functional, but GPUDirect would
>    remove the staging copies and lower small-message latency (~19 us one-way today).
>
> The dmabuf alternative is also closed on the current stack: the driver reports
> `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0` (proprietary kernel modules), so
> registration-by-dmabuf paths in NVSHMEM/NCCL/rdma-core cannot engage either.
>
> **Request:** load `nvidia_peermem` on the H100 compute nodes and persist it across
> reboots (e.g. an `/etc/modules-load.d/nvidia-peermem.conf` entry), or alternatively
> migrate the nodes to the NVIDIA open kernel modules so dmabuf-based registration works.
> The module ships with the driver already installed, so this should be a
> configuration-only change.
>
> **Verification after the change** (we are happy to run these):
>
>     lsmod | grep peermem                      # module present
>     # NVSHMEM: 8 PEs across 2 nodes no longer skip IBRC and initialize cleanly
>     # NCCL_DEBUG=INFO shows [GDRDMA] on inter-node channels
>     # ucx_info -d | grep -i gdr               # UCX gdr paths appear

---

## Supporting detail (keep out of the ticket unless asked)

### Exact failure signatures collected

| component | signature |
| --- | --- |
| NVSHMEM `ibrc` | `ibrc.cpp:nvshmemt_init:1978: neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.` |
| NVSHMEM `ibdevx` | `ibdevx.cpp:nvshmemt_init:2118:` same message |
| NVSHMEM `ibgda` (with `NVSHMEM_IB_ENABLE_IBGDA=1`) | `ibgda.cpp:nvshmemt_init:3706:` same message |
| NVSHMEM `ucx` | `ucx.cpp:331: non-zero status: 7 Failed to map memory in UCX transport` while registering the symmetric heap |
| NVSHMEM, any 2-node run | `topo.cpp:469: [GPU n] Peer GPU 0 is not accessible` -> `init.cu:1037: building transport map failed` |
| dmabuf probe | `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED)` returns 0 on every GPU |
| NCCL 2.28.3 inter-node | functional but ~2.7 GB/s aggregate at 4 MB messages; invariant under `NCCL_IB_QPS_PER_CONNECTION`, `NCCL_MIN/MAX_NCHANNELS`, `NCCL_P2P_NET_CHUNKSIZE`, `NCCL_BUFFSIZE`, and the HPC-X net plugin |

The IBGDA transport binary contains a dmabuf registration path
(`ibv_reg_dmabuf_mr`/`mlx5dv_reg_dmabuf_mr`, with the message "The system does not support
registering the NIC control buffers with DMABUF. Fallback to use either nv_peer_mem or
nvidia_peermem.") -- so on this stack it tries dmabuf, finds it unsupported, falls back to
the peermem check, and dies. Either root-level fix (peermem or open modules) unblocks it.

### Control experiments proving the fabric itself is fine

- 8-rank 2-node host-memory `MPI_Allreduce` passes over UCX `rc`, `dc`, and `tcp`.
- 2-node GPU-aware MPI ping-pong: ~19 us one-way small-message latency, ~39 GB/s at
  16 MB (host-staged) -- the wire and NICs deliver.
- Intra-node NVSHMEM (NVLink/IPC) is fully functional and validated at 2 and 4 PEs.

### Secondary fabric observations (separate issues; mention only if useful)

1. UCX's `ud` transport is broken even intra-node: `ibv_create_ah(...) failed: Connection
   timed out` on `mlx5_0` and `mlx5_bond_0`. Default-configured MPI jobs abort in
   `MPI_Init` at >= 4 ranks; we work around it by pinning
   `UCX_TLS=sm,self,cuda_copy,cuda_ipc[,rc]`. Possibly a RoCE ARP/GID configuration issue
   on the bond; an admin may want to know.
2. NCCL's default HCA enumeration picks the raw bond slaves (`mlx5_0`, `mlx5_1`) alongside
   `mlx5_bond_0` and lands on dead routes (~78 ms/iter retransmit stalls). Workaround:
   `NCCL_IB_HCA=mlx5_bond_0`. If ALCF publishes recommended NCCL settings for Janus, this
   belongs there.
3. `libibumad` is not installed on the nodes, so HPC-X's bundled NCCL net plugin
   (`nccl_rdma_sharp_plugin`) fails to load out of the box (the RPM exists under
   `/soft/repos/hpe-doca-ofed-rhel9.4/`). Cosmetic today, but trivial to fix while in there.
