# NVSHMEM investigation tools

Everything here was written while producing `../NVSHMEM-BUILD-NOTES.md`. Each file is the
instrument behind a specific claim in those notes; the section it supports is listed below.
All of them need `source ../janus-env-nvshmem.sh` first, and every multi-rank run needs
`mpirun --bind-to none` (see notes section 10.1 -- default one-core-per-rank binding makes
NVSHMEM look ~10x slower than it is).

## Node vetting -- run these BEFORE trusting any measurement

| file | what it answers | notes section |
| --- | --- | --- |
| `../../../vet-node.sh` | Is this node's GPU state clean? Read-only, creates no CUDA context, so it cannot wedge a suspect node. | leak report |
| `devorder.cu` | How many GPUs can CUDA actually open, and at which PCI bus? Catches the case where `nvidia-smi` lists 4 but `cudaGetDeviceCount()` returns 1. | 9, leak report |

## Environment / transport isolation

| file | what it answers | notes section |
| --- | --- | --- |
| `mpi_hello.c` | Does plain MPI work at N ranks? Proved the UCX InfiniBand timeout is a site issue, not NVSHMEM's fault, and that `UCX_TLS=sm,self,cuda_copy,cuda_ipc` fixes it. | 7 |
| `mpi_gpu_pp.c` | Does bare GPU-aware MPI work on plain `cudaMalloc` buffers? Used to prove the `sfbench` UCX segfault came from raw device pointers, not from GPU-aware MPI being broken. | 10.5 |
| `nvshmem_smoke.cu` | Does NVSHMEM itself work, using PETSc's exact init order (`cudaSetDevice` **before** `nvshmemx_init_attr`) and PETSc's `-rdc=true`/`-dlink` build recipe? Ring put + `nvshmemx_signal_op`, verified. | 5, 9 |

## PETSc behaviour

| file | what it answers | notes section |
| --- | --- | --- |
| `whichdev.c` | Which CUDA device does each PETSc rank land on? Proved PETSc's default `rank % ndev` gives 1 PE per GPU with no `CUDA_VISIBLE_DEVICES` wrapper. Trust this over `nvidia-smi --query-compute-apps`, which is unreadable during an NVSHMEM run because IPC peer mappings appear against every GPU. | 9 |
| `run-petsc-nvshmem-tests.sh` | Correctness suite: runs each case with `-use_nvshmem 0` and `1`, requires identical output **and** positive proof NVSHMEM engaged (via the `NVSHMEM_VERSION=1` banner, since a failed eligibility check falls back to MPI silently). | 9 |

## Performance

| file | what it answers | notes section |
| --- | --- | --- |
| `sfbench.c` | `PetscSFBcast` latency/bandwidth vs message size, ring topology, device memory. The honest instrument for comparing NVSHMEM against GPU-aware MPI. | 10.2 |
| `sfbench-run.sh` | Builds `sfbench.c` and prints the MPI / NVSHMEM-put / NVSHMEM-get comparison table. `NMIN`/`NMAX` env vars truncate the sweep for a quick check. | 10.2 |
| `sfbench2.c` | Two experiments: `-nexch K` (K independent exchanges in flight) and `-naxpy m` (overlappable compute between Begin and End). Established that NVSHMEM wins only when there is independent GPU work to hide a *latency-bound* exchange behind. | 16 |
| `launchcost.cu` | What does a CUDA kernel launch and an event pair actually cost here? Gave 1.91 us/launch and 0.53 us for 4 event ops -- which is what proved the overhead is launches, not stream plumbing. | 14.2 |
| `sigcost.cu` | Is `nvshmemx_signal_op_on_stream()` cheaper than a `<<<1,1>>>` kernel? No -- ~2.5 us either way. This killed the proposed optimization before it was written. | 14.3 |

## Build recipe reminder

`sfbench*.c` and `whichdev.c` are plain C against libpetsc:

    mpicc -O3 -o sfbench sfbench.c -I$PETSC_DIR/include -I$PETSC_DIR/$PETSC_ARCH/include \
      -I$CUDA_HOME/include -Wl,-rpath,$PETSC_DIR/$PETSC_ARCH/lib \
      -L$PETSC_DIR/$PETSC_ARCH/lib -lpetsc -L$CUDA_HOME/lib64 -lcudart

`nvshmem_smoke.cu` and `sigcost.cu` call NVSHMEM **device** functions, so they need the same
three-step relocatable-device-code build PETSc uses for `sfnvshmem.o` (notes section 5):

    nvcc -ccbin g++ -arch=sm_90 -rdc=true -O2 $MPI_INC -I$NVSHMEM_DIR/include -c x.cu -o x.o
    nvcc -ccbin g++ -arch=sm_90 -dlink x.o -L$NVSHMEM_DIR/lib -lnvshmem_device -o x_dlink.o
    mpicxx x.o x_dlink.o -o x -L$NVSHMEM_DIR/lib -lnvshmem_host -lnvshmem_device \
      -L$CUDA_HOME/lib64 -lcudart -L$CUDA_HOME/lib64/stubs -lcuda

## 2026-09-01 additions: multi-node with peermem live

Build notes 13.4 and perf notes 28 are the write-ups; `results-20260901-*.txt` the raw logs.

| file | what it answers | section |
| --- | --- | --- |
| `nvshmem_smoke.cu` (fixed) | Now also a multi-node smoke: reports whether each PE's target is `P2P-mapped` or `REMOTE (IB)`. The put *source* moved to the symmetric heap -- an unregistered `cudaMalloc` source passes over NVLink but dies with `IBV_WC_LOC_PROT_ERR` over IB. | BN 13.4 |
| `run-petsc-nvshmem-tests.sh` | `NPS="8" MPIRUN_EXTRA="--hostfile $PBS_NODEFILE --map-by ppr:4:node"` runs the suite across nodes; forwards the HCA/UCX pins with `-x`; the output filter drops IBGDA's warning lines. | BN 13.4 |
| `sfbench-run.sh` | Same `MPIRUN_EXTRA` hook; builds into the (shared) script directory so remote ranks can see the binary. | PN 28.2 |
| `ladder-20260901/` | The five driver scripts behind every 2026-09-01 result, in order. | PN 28 |
| `devorder` | Now built here (`nvcc -o devorder devorder.cu`); the suite header uses it. | -- |

Multi-node pins live in `../janus-env-nvshmem.sh` (`NVSHMEM_HCA_LIST`, `UCX_NET_DEVICES`,
the NCCL RoCE v1 trio, and the `JANUS_UCX119=1` GPUDirect-MPI opt-in). Never use
`mlx5_bond_0`: it is the 25 GbE management network.
| `collbench.c` | Exposed per-op cost of NCCL/MPI allreduce, reduce, bcast, p2p on GPU buffers, 8 B-16 MB, under any placement. Fills the collective-curve gap of the measured catalog in `~/prox-latency/experiments/harness/machine_janus.py`. | PN 28.10 |
| `acgnrun.c` | Generic executor for ANY skeleton of the poster's certified family (`-order p1,p2,p3 -grad o0,o1,o2,o3`, `-allreduce`), PetscSF transports, same AXPY emulation as `acgnbench.c` (matches it within 2%). Closes the loop on the re-screen in `~/prox-latency/experiments/` (`closeloop.py`). | RESULTS.md Result 7 (prox-latency) |
| `opbench.c` + `tvprox.cu` | Operator-cost probe on one H100: synthetic ray-like sparse gradient (MATAIJCUSPARSE MatMult + MatMultTranspose), exact row-TV prox (Condat, one thread per row), column-TV via transpose, box clamp, mixing AXPYs, for images 128^2..2048^2. Turns the schedule model's compute-regime sweep into measured durations. | results-20260902-opbench.txt |
| `acgnrun-nccl.c` | NCCL counterpart of `acgnrun.c`: any skeleton, one fused `ncclGroup` per DAG stage, single stream; `-graph_iters k` captures k iterations into a CUDA graph and times replays. Reproduces `acgnbench-nccl` (107 vs 110 us) and `acgnbench-nccl-graph` (31.9 vs 32.1 us); checksum-validated. | RESULTS.md Result 7 (prox-latency) |
| `gangbench.c` | Cooperative (gang) gradient of one shard over g GPUs of a node: each rank owns rays/g rows of the synthetic projector, computes a full-length partial, `ncclReduce` to the leader. Reports local / reduce / total per g -- the catalog entry `gang_grad_us` in `harness/machine_janus.py`. Lesson: initialize PETSc's device before `ncclCommInitRank` or every rank binds NCCL to GPU 0. | results-20260902-gangbench.txt |
| `acgnbench-nvdev2.cu` | `acgnbench-nvdev.cu` + `-push`: producers put block-contiguous chunks into symmetric receive lanes on every consumer (last block fences + signals); runs across nodes. 2+2: 104 us at 64 KB with the device-side stopping test exact; 1 node: 36.5 us (pull 45.6). | PN 28.12 |
| `sfbw.c` | Large-message bandwidth on device memory: the same buffers exchanged by PetscSF (`-arm sf`, with `-use_nccl 0/1`), bare `MPI_Isend/Irecv` (`-arm mpi`) or a bare NCCL group (`-arm nccl`), ring or `-pair`, `-uni` for one direction; every size in its own `PetscLogStage` so `-log_view` excludes the setup. Answers the 4 MB question: PetscSF adds nothing; NCCL's p2p kernel uses 4 blocks per 4 MB op and shares them between directions; `NCCL_P2P_NVL_CHUNKSIZE=65536` closes the gap on a node. | PN 29.9, results-20260909-sfnccl.txt |
| `sfgraph.c` | PetscSF ring exchange captured into a CUDA graph (`-per_graph m` exchanges, `-naxpy k` AXPYs between Begin/End) and replayed; needs a stream-ordered transport (`-use_nccl 1`) and `-root_device_context_stream_type nonblocking`. The GPU-aware MPI path aborts at capture (host sync inside Begin). 1 node, 32 KB + 8 AXPYs: 45 us eager vs 9 us replayed. | PN 29.5, results-20260909-sfnccl.txt |
| `sfbench2.c` (`-w`) | New `-w <len>` sets the work-vector length; the 1M default is host-launch-bound (5 us per VecAXPY on the host, ~2.5 us on the GPU), so use `-w 16777216` for a GPU-bound overlap regime. | PN 29.4 |
