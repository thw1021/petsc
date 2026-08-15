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
