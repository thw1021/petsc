# Building PETSc with NVSHMEM on ALCF Janus (H100, sm_90)

Recorded 2026-08-14 on compute node `x2000c0s1b0n0` (4x H100, PBS job 2076),
branch `hsuh/nvshmem-pg1`, worktree `/home/hsuh/petsc/.claude/worktrees/nvshmem-v1`,
`PETSC_ARCH=arch-janus-nvshmem`. Environment script: `janus-env-nvshmem.sh` (in this worktree).

Result: configure + `make all` + `make check` all pass; PetscSF NVSHMEM path verified
active at 2 and 4 ranks.

---

## 1. Why a new arch was needed

The pre-existing `arch-janus-cuda-nvshmem` under `/home/hsuh/petsc` is **not reusable**.
It lives in the `main` worktree, and `main`'s working tree contains none of the NVSHMEM
fixes — those exist only as a single WIP commit on `hsuh/nvshmem-pg1`. Rebuilding there
would reintroduce the original `sfnvshmem.cu` compile failures. Build from the worktree.

That old arch was also configured against the **monolithic** `libnvshmem.a`, which is the
wrong library layout for this branch (see §4).

## 2. Toolchain, and why each piece was chosen

| Component | Choice | Reason |
| --- | --- | --- |
| Host compiler | system **gcc 11.5** (`/usr/bin/gcc`) | NVHPC's `nvc`/`nvc++` are broken on this box — their `localrc` points at a nonexistent SUSE gcc path (`/usr/lib64/gcc/x86_64-suse-linux/7/`), so even `#include <stddef.h>` fails |
| MPI | **HPC-X Open MPI 4.1.7a1**, from `$NVHPC_ROOT/comm_libs/12.6/hpcx/hpcx-2.20` | CUDA-aware (`ompi_info` shows `MPI extensions: cuda`, `pml: ucx`). **Critical:** NVSHMEM's prebuilt `nvshmem_bootstrap_mpi.so` has `DT_NEEDED libmpi.so.40` = the Open MPI **4.x** ABI. `/soft/compilers/openmpi-gnu` is 5.x (`libmpi.so.80`) and its MPI bootstrap will not load |
| CUDA | **12.9.1** (`/soft/compilers/cudatoolkit/cuda-12.9.1`) | Must match NVSHMEM 3.4.5's device objects. `nvlink` refuses device code from a toolkit newer than nvcc: CUDA 12.6 gives `nvlink fatal: Input file 'libnvshmem.a:init.cu.o' newer than toolkit (129 vs 126)`. Note the node also has a stray `/usr/local/cuda-12.4` early in `PATH` — the env script prepends 12.9.1 to override it |
| NVSHMEM | **3.4.5**, `/soft/libraries/nvshmem/libnvshmem-linux-x86_64-3.4.5_cuda12-archive` | Newest available; `NVSHMEM_MPI_SUPPORT=ON` |
| BLAS/LAPACK | **AMD AOCL** `.so` (`/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_LP64`) | No internet on compute nodes, so `--download-f2cblaslapack` fails. `/soft` OpenBLAS needs `libgfortran.so.4` (only `.so.5` present); NVHPC lapack needs `libatomic.so.1` (absent). AOCL resolves within its own tree and needs no Fortran runtime, which suits `--with-fc=0`. Use the `.so` files — the `.a` route fails because `libaoclutils.a` is C++ and `libflame.a` wants OpenMP symbols |
| Kokkos | **omitted** | The PetscSF NVSHMEM path needs only VECCUDA. Dropping it roughly halves build time and, usefully, leaves `sf->backend = PETSCSF_BACKEND_CUDA` instead of `PETSCSF_BACKEND_KOKKOS` (see `src/vec/is/sf/interface/sf.c:75-81`), which is the simpler path for NVSHMEM. Add it back only if a Kokkos-specific test is wanted |

The MPI wrappers are forced to gcc with `OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran` —
`mpicc` is a wrapper script and these env vars override the compiler it invokes.

### Sourcing gotcha

`janus-env-nvshmem.sh` must be sourced **from bash and not through a pipe**:

    source ./janus-env-nvshmem.sh            # correct
    source ./janus-env-nvshmem.sh | tail -5  # WRONG - silently exports nothing

A pipe puts `source` in a subshell, so every `export` is discarded. This bites hard
because the script's own echo'd sanity check runs *inside* that subshell and prints
correct-looking values while the parent shell got nothing. `PETSC_CONFIGURE_OPTS` is
also a bash **array** (some values contain spaces, e.g. `-ccbin g++`), so it must be
expanded as `"${PETSC_CONFIGURE_OPTS[@]}"`.

## 3. Configure line

    ./configure \
      --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
      --with-cuda=1 --with-cuda-dir=$CUDA_HOME --with-cuda-arch=90 \
      --with-cudac=$CUDA_HOME/bin/nvcc --CUDAFLAGS="-ccbin g++" \
      --with-nvshmem=1 \
      --with-nvshmem-include=$NVSHMEM_DIR/include \
      --with-nvshmem-lib="$NVSHMEM_DIR/lib/libnvshmem_host.so \
                          $NVSHMEM_DIR/lib/libnvshmem_device.a \
                          $CUDA_HOME/lib64/stubs/libcuda.so" \
      --with-blaslapack-lib="$AOCL_DIR/libflame.so $AOCL_DIR/libblis.so $AOCL_DIR/libaoclutils.so" \
      --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3

`libcuda.so` comes from `$CUDA_HOME/lib64/stubs` because the login/admin nodes have no
driver library; the real `libcuda.so.1` is picked up at run time on compute nodes.

## 4. The NVSHMEM library split — the part that matters most

NVSHMEM 3.0 **split the old monolithic `libnvshmem.a` into a host shared library plus a
device static library**:

    libnvshmem_host.so    host API  (nvshmem_malloc, nvshmemx_hostlib_init_attr, ...)
    libnvshmem_device.a   device API (the __device__ code: nvshmemx_signal_op, ...)
    libnvshmem.a          legacy monolithic archive, still shipped

This branch's `config/BuildSystem/config/packages/NVSHMEM.py` was updated to prefer the
split pair:

    self.liblist = [['libnvshmem_host.so','libnvshmem_device.a','libcuda.a'],
                    ['libnvshmem.a','libcuda.a'],
                    ['nvshmem.lib','cuda.lib']]

A confusing detail worth knowing: `nm -D libnvshmem_host.so` shows **no** `nvshmem_init`,
`nvshmemx_init_attr`, or `nvshmem_finalize`. They are `static inline` wrappers in
`include/host/nvshmemx_api.h` that forward to exported symbols such as
`nvshmemx_hostlib_init_attr`. The library is fine; don't be misled into thinking it is
incomplete.

Configure must end up emitting a `-L` in `NVSHMEM_LIB`, because the device-link rule
greps for it (§5). Passing full `.so`/`.a` paths as above produces exactly that:

    NVSHMEM_LIB = -Wl,-rpath,<nvshmem>/lib -L<nvshmem>/lib \
                  -L<cuda>/lib64/stubs -lnvshmem_host -lnvshmem_device -lcuda

## 5. CUDA device linking — why `gmakefile` needed patching

`src/vec/is/sf/impls/basic/nvshmem/sfnvshmem.cu` calls NVSHMEM **device** functions
(`nvshmemx_signal_op()` and friends) whose implementations live in `libnvshmem_device.a`,
not in a header. Cross-translation-unit device calls require three steps, all mandatory:

1. **Compile with `-rdc=true`** (relocatable device code), deferring the device link.
   Without it: `ptxas fatal: Unresolved extern function '...'`.
2. **Device-link:** `nvcc -dlink <objs> <device libs> -o dlink.o`. `nvlink` resolves the
   device symbols and emits one *host* object holding the linked device image plus the
   CUDA-runtime registration code. Without it, the final host link fails with
   `undefined reference to __cudaRegisterLinkedBinary_*`.
3. **Host-link `dlink.o` alongside the original objects.**

PETSc's build system had no step 2. The patch in `gmakefile` adds a target-specific rule
that keeps the change local — it produces the ordinary `sfnvshmem.o` the rest of the build
expects, by merging the rdc object and the device-link object with `ld -r`:

    ifneq ($(NVSHMEM_LIB),)
    NVSHMEM_DLINK_LIBS := $(filter -L%,$(NVSHMEM_LIB)) -lnvshmem_device
    $(OBJDIR)/src/vec/is/sf/impls/basic/nvshmem/sfnvshmem.o : ...
        $(PETSC_COMPILE.cu) -rdc=true $(abspath $<) -o $(@:%.o=%.rdc.o)
        $(CUDAC) $(CUDAC_FLAGS) -Xcompiler -fPIC -dlink $(@:%.o=%.rdc.o) -o $(@:%.o=%.dlink.o) $(NVSHMEM_DLINK_LIBS)
        ld -r $(@:%.o=%.rdc.o) $(@:%.o=%.dlink.o) -o $@
    endif

Subtleties encoded there:

- `$(filter -L%,...)` strips `-Wl,-rpath,...` and `-l` entries: **`nvcc -dlink` rejects
  host linker flags**, and only the device library is relevant.
- **`-Xcompiler -fPIC` is required** because `dlink.o` ends up inside `libpetsc.so`.
  Without it: `relocation R_X86_64_32S against '__nv_module_id' can not be used when
  making a shared object`.
- Device-link against **`libnvshmem_device.a`**, never a mix of that and `libnvshmem.a` —
  the latter also contains the device code and `nvlink` then reports
  `Multiple definition of ...`.

Observed build output confirming all three steps:

    nvcc -ccbin g++ ... -rdc=true .../sfnvshmem.cu -o .../sfnvshmem.rdc.o
    nvcc -ccbin g++ ... -Xcompiler -fPIC -dlink .../sfnvshmem.rdc.o -o .../sfnvshmem.dlink.o \
         -L<nvshmem>/lib -L<cuda>/lib64/stubs -lnvshmem_device
    ld -r .../sfnvshmem.rdc.o .../sfnvshmem.dlink.o -o .../sfnvshmem.o

### Diagnosing device objects

`nm` on a `.cu.o` shows nothing useful — device symbols are not host ELF. Use
`cuobjdump --dump-elf-symbols file.o`, and `cuobjdump file.o | grep 'arch = sm_'` for
target archs. Toolkit version is encoded in the fatbin, which is how the nvlink
version check above triggers.

## 6. Build and verification

    source ./janus-env-nvshmem.sh
    ./configure "${PETSC_CONFIGURE_OPTS[@]}"
    make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH MAKE_NP=32 all
    make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check

Post-link checks that the NVSHMEM wiring is real:

    readelf -d arch-janus-nvshmem/lib/libpetsc.so | grep NEEDED   # -> libnvshmem_host.so.3
    nm -D  arch-janus-nvshmem/lib/libpetsc.so | grep -c nvshmem   # -> 29
    nm     arch-janus-nvshmem/lib/libpetsc.so | grep -c __cudaRegisterLinkedBinary  # -> 8 (device link OK)

`make check` prints a `*******Error detected during compile or link!*******` banner from a
`.PETSc` guard target whose errors are `(ignored)`. Judge success by the presence of
`run successfully` and the **absence** of a trailing `make: *** Error 2`.

## 7. Runtime environment (needed to RUN, not just build)

AOCL and NVSHMEM are shared libraries, so `janus-env-nvshmem.sh` must be sourced before
running PETSc binaries too (it sets `LD_LIBRARY_PATH`), unless `-Wl,-rpath` was baked in.

Two site-specific runtime settings are folded into the script:

    export UCX_TLS=sm,self,cuda_copy,cuda_ipc
    export NVSHMEM_SYMMETRIC_SIZE=256M

**`UCX_TLS` is not optional at >= 4 ranks.** With the default UCX configuration, a plain
4-rank `MPI_Init` + `MPI_Allreduce` **aborts** on this node:

    UCX ERROR ibv_create_ah(dlid=49152 ... on mlx5_bond_0 failed: Connection timed out
    ucp_ep_create(proc=1) failed: Endpoint timeout
    *** An error occurred in MPI_Init

UCX tries the InfiniBand UD path even for intra-node traffic. Restricting it to
shared-memory + CUDA transports fixes it and preserves CUDA-awareness. This was
reproduced with a bare MPI program and has nothing to do with NVSHMEM or PETSc.

`NVSHMEM_SYMMETRIC_SIZE` defaults to 1 GiB **per PE**; trimming it lets several PEs share
one GPU comfortably.

### NVSHMEM limits on this machine

- **Single node only.** The IBRC transport is skipped:
  `neither nv_peer_mem, or nvidia_peermem detected` — the GPUDirect RDMA peer-memory
  kernel module is not loaded, so inter-node NVSHMEM is unavailable. Intra-node
  P2P/IPC over NVLink works (`nvidia-smi topo -p2p r` shows `OK` between all 4 GPUs).
- **Do not use NVIDIA's bundled perftest binaries as a reference.** They call
  `cudaSetDevice()` *after* `nvshmem_init()`, so every PE lands on one GPU and NVSHMEM
  reports `More than 1 PE per GPU detected. This is an MPG run.` Restricting
  `CUDA_VISIBLE_DEVICES` to fix it makes them fail differently
  (`cuda failed with (100)` — they then request a device index that no longer exists).
  PETSc gets this right: `PetscNvshmemInitializeCheck()` calls
  `PetscDeviceInitialize(PETSC_DEVICE_CUDA)` **before** `nvshmemx_init_attr()`
  (`src/vec/is/sf/impls/basic/nvshmem/sfnvshmem.cu:8-21`).

## 8. Using it: how PETSc decides to take the NVSHMEM path

There are **no NVSHMEM tests in the PETSc test suite** — nothing under
`src/vec/is/sf/tests` or `tutorials` references it. It is purely a runtime flag,
read in `PetscSFCreate()` straight from the global options database
(`src/vec/is/sf/interface/sf.c:86`), so it needs no prefix and no `SetFromOptions`:

    -use_nvshmem 1        # enable (default 0)
    -use_nvshmem_get 1    # use the get-based protocol instead of the default put-based one

`PetscSFLinkNvshmemCheck()` (`sfnvshmem.cu:173`) then requires **all** of:

1. `sf->use_nvshmem` set;
2. SF type is `PETSCSFBASIC`;
3. the SF's communicator is `MPI_IDENT`/`MPI_CONGRUENT` with `PETSC_COMM_WORLD`;
4. no rank has both `rootdata` and `leafdata` NULL;
5. root and leaf memtypes are both CUDA on every rank.

If any fails it silently falls back to MPI, so **a clean run is not evidence the NVSHMEM
path ran.** To get positive proof, run with `NVSHMEM_VERSION=1`: NVSHMEM prints a
`NVSHMEM v3.4.5` banner when it initializes, and PETSc only initializes NVSHMEM once an SF
passes the check above. Verified discriminator — 0 banners in every `-use_nvshmem 0` run,
exactly 1 in every `-use_nvshmem 1` run.

---

## 9. Verification results

Two rounds were run. Only the second is a valid multi-GPU result.

### Round 1 — `x2000c0s1b0n0`, degraded node (MPG only)

That node had 3 of 4 GPUs wedged by a leaked container, so `cudaGetDeviceCount()` returned 1
and every rank shared one GPU ("MPG" — multiple PEs per GPU). All cases passed, but the result
only covers same-GPU operation. See `/home/hsuh/petsc/janus-node-gpu-leak-report.md`.

### Round 2 — `x2003c0s9b0n0`, clean node, **1 PE per GPU** (definitive)

Node vetted first: all 4 GPUs `Disabled` / 0 MiB / no `ERR!`, `cudaGetDeviceCount() -> 4`,
all-pairs P2P `OK`, and `nvidia-smi topo -m` reporting `NV6` between every GPU pair. (On the
degraded nodes `topo -m` failed outright — a useful health signal in itself.)

PETSc's own rank->device mapping confirmed one rank per GPU, via a probe calling
`cudaGetDevice()` after forcing a `VECCUDA` allocation:

    rank 0 -> cuda dev 0  bus 0x04       rank 2 -> cuda dev 2  bus 0x84
    rank 1 -> cuda dev 1  bus 0x64       rank 3 -> cuda dev 3  bus 0xe4

This is PETSc's default `rank % ndev` policy in
`src/sys/objects/device/impls/cupm/cupmdevice.cxx:293`, used whenever the device id is
`PETSC_DECIDE`. No `CUDA_VISIBLE_DEVICES` wrapper is needed for PETSc.
(Caution when reading `nvidia-smi --query-compute-apps` during an NVSHMEM run: every PE also
appears against its peers' GPUs because of the IPC symmetric-heap mappings, which makes it look
as though all ranks are on one GPU. Trust `cudaGetDevice()`, not the nvidia-smi attribution.)

| test | np | result |
| --- | --- | --- |
| `sf/tests/ex22` FetchAndOp, put protocol | 2, 4 | PASS |
| `sf/tests/ex22` FetchAndOp, `-use_nvshmem_get 1` | 2, 4 | PASS |
| `snes/tutorials/ex19` lid-driven cavity | 2, 4 | PASS |

`PASS = output byte-identical to the -use_nvshmem 0 baseline AND NVSHMEM confirmed engaged.`

Controls checked, not assumed:
- **0** NVSHMEM banners across all six baseline runs; exactly **1** across all six NVSHMEM runs.
- **0** `More than 1 PE per GPU` warnings anywhere -- these were genuine 1-PE-per-GPU runs.
- `ex19` is a real solve: 3 Newton steps, `CONVERGED_FNORM_RELATIVE`.
  (An earlier attempt used `-ksp_type cg` on this nonsymmetric Jacobian; KSP hit `DIVERGED_DTOL`
  and SNES stopped at 0 iterations, making the comparison vacuous. Use the default GMRES.)
- `ex22` operates on real `mpicuda` vectors with correct FetchAndOp values.

### Still not covered

- **Multi-node.** Impossible here: IBRC is skipped for want of `nvidia_peermem`.
- **Performance.** Everything above is correctness only; no timings were taken.
- Kokkos backend (`sf->backend = PETSCSF_BACKEND_KOKKOS`), since this arch omits Kokkos.

---

## 10. Performance: NVSHMEM vs GPU-aware MPI (single node, 4x H100 NVLink)

Measured on `x2003c0s9b0n0` (vetted clean), 1 PE per GPU, `arch-janus-nvshmem`
(`--with-debugging=0 -O3`). Baseline is PETSc's normal MPI path with GPU-aware MPI, which
is **on by default** (`use_gpu_aware_mpi = PETSC_TRUE`, `src/sys/objects/init.c:37`) over
HPC-X/UCX with `cuda_ipc`. So both arms move data GPU-to-GPU over NVLink.

### 10.1 CPU binding dominates everything else -- read this first

NVSHMEM starts a **proxy thread that spins**. Open MPI's default binding gives each rank a
single core, so that proxy thread fights the rank itself for one CPU. NVSHMEM warns about it:

    NVSHMEM WARN Proxy thread shares a core with the main PE, performance may be impacted

That warning understates it. `snes/tutorials/ex19 -da_refine 6`, 4 ranks:

| binding | MPI `SNESSolve` | NVSHMEM `SNESSolve` |
| --- | --- | --- |
| default (1 core/rank) | 1.32 s | **15.47 s** |
| `--bind-to none`      | 1.16 s | **1.53 s** |

**A 10x swing from CPU binding alone.** With default binding the slowdown lands not in the
halo exchange but in the MPI *reductions* -- `-log_view` shows `VecMDot` going 11% -> 45% of
runtime and `VecNorm` 9% -> 35%, while `VecScatterEnd` *drops* from 15% to 1%. The exchange
got faster; the starved core made every `MPI_Allreduce` crawl.

**Always give NVSHMEM ranks more than one core** (`--bind-to none`, or better
`--map-by ppr:1:numa` / `--map-by slot:PE=<n>`). Every number below uses `--bind-to none`.
Any NVSHMEM benchmark run under default binding is measuring core starvation, not NVSHMEM.

### 10.2 Microbenchmark: `PetscSFBcast` latency, ring topology, device memory

`sfbench.c` -- each rank owns n roots, its n leaves reference the previous rank's roots, so
all traffic is inter-GPU. Best of 3 reps, microseconds per `Bcast(Begin+End)`.

| bytes | MPI | NVSHMEM put | put/MPI (np=2) | put/MPI (np=4) |
| ---: | ---: | ---: | ---: | ---: |
| 64        | 19.3 | 16.6  | **0.86** | **0.88** |
| 1 Ki      | 11.3 | 16.2  | 1.43 | 1.49 |
| 64 Ki     | 11.9 | 17.9  | 1.50 | 1.55 |
| 1 Mi      | 19.7 | 27.9  | 1.42 | 1.40 |
| 32 Mi     | 266  | 329   | 1.24 | 1.24 |

NVSHMEM wins only at the very smallest message (64 B), then runs **1.2x-1.5x slower**
across the range. MPI's floor is ~11-12 us; NVSHMEM's is ~16-18 us. The get-based protocol
(`-use_nvshmem_get 1`) tracks the put protocol, consistently a few percent behind.

Caveat: this benchmark issues `Begin`/`End` back-to-back with no intervening compute, so it
measures pure latency and gives NVSHMEM's stream-ordered design no chance to overlap.

### 10.3 Application level: `ex19` lid-driven cavity, `-da_refine 6`

Mean of 4 reps, seconds.

| np | event | MPI | NVSHMEM | ratio |
| ---: | --- | ---: | ---: | ---: |
| 2 | `KSPSolve`  | 0.797 | 0.800 | 1.00 |
| 2 | `SNESSolve` | 0.964 | 1.199 | 1.24 |
| 4 | `KSPSolve`  | 0.821 | **0.795** | **0.97** |
| 4 | `SNESSolve` | 1.162 | 1.528 | 1.31 |

Two distinct effects, and they must not be conflated:

1. **Steady-state solve is a wash to a slight win.** `KSPSolve` is identical at np=2 and
   **~3% faster** under NVSHMEM at np=4 -- and markedly more reproducible (spread across 4
   reps: 0.002 s for NVSHMEM vs 0.021 s for MPI). Unlike the microbenchmark, the real solve
   overlaps communication with compute, which is what NVSHMEM's design rewards.
2. **Startup costs a fixed ~0.24 s (np=2) to ~0.37 s (np=4).** That is `nvshmem_init` plus
   symmetric-heap allocation and IPC handle exchange, and it grows with PE count. It is
   one-time, so it dominates `SNESSolve` in a 1-second run and would be negligible in a long
   one. **Do not read the `SNESSolve` ratio as a throughput result.**

### 10.4 Verdict for this machine

On a single NVLink-connected node, PETSc's NVSHMEM path is **not a win**: GPU-aware MPI over
`cuda_ipc` is already near-optimal here, and NVSHMEM adds a fixed startup cost for at best a
few percent in the solve. That is an unsurprising result -- the PetscSF NVSHMEM backend
targets *multi-node* runs where MPI overheads dominate, which is exactly the regime this
machine cannot test (IBRC is disabled for want of `nvidia_peermem`; see section 7).

Nothing here should be extrapolated to multi-node. What it does establish: the
implementation is correct, is not pathologically slow when configured properly, and its one
real configuration hazard is CPU binding.

### 10.5 Benchmark gotcha worth remembering

`sfbench.c` originally passed raw `cudaMalloc` pointers to `PetscSFBcastWithMemTypeBegin()`
and **segfaulted inside UCX** on the GPU-aware MPI path:

    UCX ERROR cuMemGetAddressRange(0x7f...) error: named symbol not found
    Caught signal number 11 SEGV

Bare MPI with the same `cudaMalloc` buffers works, and `ex19` (which uses `Vec` device
arrays) works. Switching the benchmark to `VecCreateSeqCUDA` + `VecGetArrayAndMemType`
fixed it. Use PETSc-managed device memory when feeding PetscSF -- it is also what real
PETSc code does.

---

## 11. Validating the NVSHMEM install against NVIDIA's own perftest

Everything in section 10 measures PETSc. None of it says whether *NVSHMEM itself* is
configured sanely, so the PETSc numbers were run against NVIDIA's bundled benchmarks in
`$NVSHMEM_DIR/bin/perftest/`. Run them with `NVSHMEM_BOOTSTRAP=MPI` and `--bind-to none`;
on a node with all GPUs healthy they place one PE per GPU correctly.

### Hardware ceiling

    $ nvidia-smi nvlink -s -i 0      ->  6 links x 26.562 GB/s
    $ nvidia-smi topo -m             ->  NV6 between every GPU pair

So ~159 GB/s per direction per GPU pair, theoretical.

### Measured, 2 PEs on 2 GPUs

| benchmark | result |
| --- | --- |
| `device/pt-to-pt/shmem_put_latency` (thread scope, one-way) | **2.39-2.56 us** at 4-64 B |
| `device/pt-to-pt/shmem_put_bw` (block scope) | **115.3 GB/s** at 4 MiB |
| `device/pt-to-pt/shmem_put_signal_ping_pong_latency` (round trip) | **6.40-7.09 us** at 4-64 B |

115.3 / 159.4 = **72% of theoretical peak**, which is a normal achieved-vs-peak ratio for
NVLink. Small-message put latency of ~2.5 us is likewise what an NVLink-local put should
cost. **The NVSHMEM configuration is sound** -- the section 10 numbers measure PETSc, not a
misconfigured transport.

### What this says about PETSc's SF layer

| quantity | value |
| --- | --- |
| raw NVSHMEM put+signal, round trip (its protocol) | ~6.4 us |
| raw NVSHMEM put, one way | ~2.5 us |
| PETSc SF **NVSHMEM** `Bcast` floor (section 10.2) | ~16-18 us |
| PETSc SF **MPI** `Bcast` floor (section 10.2) | ~11-12 us |

PETSc's SF adds roughly **10 us on top of raw NVSHMEM** at small messages -- pack/unpack
kernel launches, the `remoteCommStream` event dependencies in
`PetscSFLinkBuildDependenceBegin()`/`End()`, and link bookkeeping. That overhead, not the
transport, is why the NVSHMEM path trails GPU-aware MPI at small sizes on this machine.

At large messages the picture is healthy: PETSc reaches ~126 GB/s (MPI) and ~102 GB/s
(NVSHMEM) for 32 MiB, straddling perftest's 115 GB/s, i.e. PETSc is at the hardware ceiling
once messages are big enough to amortize the per-call overhead.

Caveat on comparing the two: perftest's ping-pong is a bare round trip between two PEs,
while `sfbench` times a full `PetscSFBcastBegin()`/`End()` pair including pack/unpack and
stream synchronization over a 2- or 4-rank ring. The comparison bounds the overhead; it is
not an apples-to-apples subtraction.

### Reproducing

    source janus-env-nvshmem.sh
    export NVSHMEM_BOOTSTRAP=MPI
    mpirun -n 2 --bind-to none $NVSHMEM_DIR/bin/perftest/device/pt-to-pt/shmem_put_latency
    mpirun -n 2 --bind-to none $NVSHMEM_DIR/bin/perftest/device/pt-to-pt/shmem_put_bw
    mpirun -n 2 --bind-to none $NVSHMEM_DIR/bin/perftest/device/pt-to-pt/shmem_put_signal_ping_pong_latency

(Section 7's warning about these binaries still applies on a *degraded* node: they call
`cudaSetDevice()` after `nvshmem_init()`, so if some GPUs are wedged they silently pile
every PE onto one GPU. Vet the node first.)

---

## 12. Is `ex19` even a good NVSHMEM benchmark? (No.) And what NVSHMEM actually covers

### 12.1 How much of `ex19` is the communication NVSHMEM affects?

`ex19`, np=4, `--bind-to none`, MPI path. "Scatter %T" = `VecScatterBegin` + `VecScatterEnd`,
i.e. the halo exchange, the *only* thing PetscSF NVSHMEM touches.

| `-da_refine` | avg msg (B) | Scatter %T | `VecMDot`+`VecNorm` %T | `KSPSolve` (s) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 400  | 16% | 4%  | 0.106 |
| 4 | 780  | 17% | 6%  | 0.189 |
| 5 | 1600 | 18% | 10% | 0.351 |
| 6 | 3100 | 18% | 18% | 0.829 |
| 7 | 6200 | 18% | 35% | 2.599 |

Two things follow, and they undercut using `ex19` to judge NVSHMEM:

1. **The exchange is only ~16-18% of runtime, and stubbornly flat.** By Amdahl, a *free*
   halo exchange would buy at most ~18%. Measuring a 1.2x transport difference inside an
   18% slice is measuring noise with extra steps.
2. **The messages are tiny -- 0.4 to 6.2 KB.** That is precisely the regime where section
   10.2 shows NVSHMEM losing (~17 us floor vs MPI's ~11.5 us). `ex19` samples NVSHMEM at
   its worst message size *and* gives it almost no share of the runtime.

The rest is local GPU work plus `VecMDot`/`VecNorm`, which are reductions -- local flops
followed by `MPI_Allreduce`. **PetscSF NVSHMEM does not touch collectives at all** (the
`MPIU_Allreduce` calls in `sfnvshmem.cu` are eligibility/setup checks, lines 114/196/207,
not the data path). So the growing right-hand column is entirely outside NVSHMEM's reach.

**Conclusion: `sfbench` (section 10.2) is the honest instrument here; `ex19` is a sanity
check that NVSHMEM does not break or slow a real solve, nothing more.** A benchmark that
would actually stress NVSHMEM needs a high communication fraction, many neighbors, and
ideally multi-node -- none of which is constructible on 4 NVLink-connected GPUs in one box.

### 12.2 Does PETSc's NVSHMEM support only `Vec`?

No -- and the framing is slightly off. NVSHMEM is plumbed in at the **PetscSF** layer
(`src/vec/is/sf/impls/basic/nvshmem/sfnvshmem.cu`), which is the transport under
`VecScatter`. Anything whose neighbor exchange goes through PetscSF gets it, including
matrix operations:

- **`MatMult_MPIAIJ` uses it.** It gathers off-process vector entries through `a->Mvctx`,
  which is a `VecScatter` and therefore a `PetscSF`
  (`src/mat/impls/aij/mpi/mpiaij.c:986-987`, `:1053`). The 4.6e+04 messages in the `ex19`
  profile above are overwhelmingly MatMult's gather. Switching `-use_nvshmem 1` moves them
  off MPI -- visible as MPI message count collapsing from `4.6e+04` to `3.2e+01`.
- **Vec ghost updates and DM halo exchange** likewise.

What is **not** covered:

- **Matrix assembly.** The off-process stash uses raw `MPI_Isend`/`MPI_Waitall`
  (`src/mat/utils/matstash.c:139`, `:973`), not PetscSF.
- **Algebraic products** (`MatPtAP`, `MatMatMult`) that ship actual matrix rows.
- **Collectives** -- `VecDot`, `VecNorm`, and all `MPI_Allreduce` traffic.

So the accurate statement is: *NVSHMEM accelerates PetscSF point-to-point neighbor
exchange, regardless of whether a Vec or a Mat operation initiated it. It does not carry
matrix data, and it does not carry collectives.*

### 12.3 Would extending NVSHMEM to matrix data make sense?

Largely no, and the reasons are structural rather than a matter of effort:

1. **The hot path is already covered.** In a solve, the repeated matrix communication *is*
   MatMult's vector gather, which is PetscSF and already NVSHMEM-capable. Matrix entries
   themselves do not move during a solve.
2. **Symmetric heap versus irregular volumes.** `nvshmem_malloc()` is collective and
   symmetric -- every PE allocates the same size. Assembly stash volumes and `MatPtAP` row
   counts are data-dependent and differ per rank and per call. You would have to allocate
   the max across PEs (wasteful) or reallocate collectively (expensive, and `nvshmem_malloc`
   is not cheap).
3. **Frequency is wrong.** In the `ex19` run there were **4** `MatAssemblyBegin/End` calls
   against **5785** `MatMult`s. Assembly is a one-off; per-call latency is irrelevant there.
4. **The profile is wrong.** Assembly and `MatPtAP` move bulk, bandwidth-bound data, and
   section 11 shows MPI already reaching the NVLink ceiling (~126 GB/s measured, against
   perftest's 115 GB/s). NVSHMEM's advantage is *latency on small, frequent, repeated*
   messages -- the exact opposite profile.

If the goal is to make NVSHMEM pay off in PETSc, the levers are (a) cutting PetscSF's ~10 us
per-exchange overhead above raw NVSHMEM (section 11), and (b) multi-node runs where MPI
overheads actually dominate -- not extending it to matrix data.

---

## 13. The two levers, concretely

### 13.1 Lever A -- PetscSF's ~10 us per-exchange overhead

Section 11 measured raw NVSHMEM put+signal at ~6.4 us round trip while PetscSF's NVSHMEM
`Bcast` floors at ~16-18 us. Reading the put path explains where the difference goes.

On a single node every peer is locally accessible (`nvshmem_ptr()` returns non-NULL over
NVLink), so `PetscSFLinkPutDataBegin_NVSHMEM()` takes the host-API branch. One
`PetscSFBcastBegin()`/`End()` pair then costs:

| step | cost |
| --- | --- |
| `PetscSFLinkBuildDependenceBegin()` | `cudaEventRecord` + `cudaStreamWaitEvent` |
| pack | kernel launch |
| `WaitSignalsFromLocallyAccessible<<<1,1,...>>>` | **one-thread kernel launch** |
| `nvshmemx_putmem_nbi_on_stream()` x n_neighbors | the actual transfer |
| `nvshmemx_quiet_on_stream()` | stream-ordered quiet |
| `PutDataEnd<<<1,1,...>>>` | **one-thread kernel launch** (signal + wait) |
| `PetscSFLinkBuildDependenceEnd()` | `cudaEventRecord` + `cudaStreamWaitEvent` |
| unpack | kernel launch |
| `PetscSFLinkSendSignalsToAllowPuttingData_NVSHMEM()` (PostUnpack) | kernel launch |

That is **~5 kernel launches and 4 event/stream operations per exchange**, two of the
launches being `<<<1,1>>>` kernels whose entire job is to spin on a signal word. At the
usual few-microseconds-per-launch this sums to right about the observed 10 us gap -- the
overhead is launch and synchronization bookkeeping, not the transport.

Candidate optimizations, cheapest first:

1. **Replace the `<<<1,1>>>` signal-wait kernels with host-side stream API.** NVSHMEM
   provides `nvshmemx_signal_wait_until_on_stream()`, which enqueues the same wait without
   a kernel launch. That removes `WaitSignalsFromLocallyAccessible` outright and possibly
   most of `PutDataEnd`.
2. **Fuse the PostUnpack signal into the unpack kernel.** It is a separate launch that
   writes a handful of signal words; the unpack kernel could do it in its last block.
3. **Drop the second stream on the intra-node path.** `remoteCommStream` plus two events
   exists so remote comm can progress independently, but with `*_on_stream` host puts the
   work could be issued directly on `link->stream`, eliminating 4 CUDA API calls and two
   cross-stream serialization points per exchange. Keep the extra stream only when there
   are genuinely remote (non-`nvshmem_ptr`) peers.
4. **CUDA Graph capture.** The `ex19` profile has 5785 MatMults issuing an identical
   sequence every time. Capturing the exchange once and replaying collapses all launch
   overhead. This is the big structural win and also helps the MPI path.

There is even an acknowledged loose end in the source: `PutDataEnd()` carries a
`/* TODO: Shall we finished the non-blocking remote puts? */`.

Note this overhead is *per exchange*, not per byte, which is exactly why the penalty fades
with message size in section 10.2 (1.5x at 64 KiB down to 1.24x at 32 MiB).

### 13.2 Lever B -- multi-node, and what currently blocks it here

NVSHMEM's design target is the regime where MPI's per-message overhead dominates, i.e.
many neighbors across many nodes. That is untestable on this machine today, and the reason
is specific and fixable:

    $ modinfo nvidia_peermem
    filename: /lib/modules/5.14.0-687.22.1.el9_8.x86_64/kernel/drivers/video/nvidia-peermem.ko
    version:  610.57.04
    $ lsmod | grep -i peermem
    (nothing -- only nvidia, nvidia_uvm are loaded)

**The module ships with the installed driver but is not loaded.** That is precisely why
every NVSHMEM run prints

    ibrc.cpp:nvshmemt_init: neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.

and why NVSHMEM here is single-node only. `modprobe nvidia_peermem` (needs root) would
enable the default `ibrc` remote transport. Worth asking ALCF for, alongside the GPU-leak
report.

Other routes that may not need `peermem`, all present in this build
(`NVSHMEM_IBGDA_SUPPORT=ON`, `NVSHMEM_IBDEVX_SUPPORT=ON`, `NVSHMEM_UCX_SUPPORT=ON`,
`NVSHMEM_LIBFABRIC_SUPPORT=ON`) and selectable via `NVSHMEM_REMOTE_TRANSPORT`
(`ibrc` | `ucx` | `libfabric` | `ibdevx` | `none`), plus IBGDA via `NVSHMEM_IB_ENABLE_IBGDA=1`:
IBGDA and IBDEVX drive the NIC through DevX rather than the peer-memory plug-in, so they
are the natural fallbacks if `peermem` cannot be loaded.

**A caution before planning multi-node work:** the InfiniBand fabric on this partition is
not obviously healthy. A bare 4-rank `MPI_Init` fails on the IB path with
`ibv_create_ah(...) failed: Connection timed out` on both `mlx5_0` and `mlx5_bond_0`
(section 7), which is why `UCX_TLS` must be pinned to shared-memory transports. That
failure is intra-node, but it suggests inter-node MPI may be broken too -- which would
block multi-node testing at a more basic level than NVSHMEM. **Verify plain multi-node MPI
works before investing in multi-node NVSHMEM.**

### 13.3 Two-node ladder, executed (2026-08-15, nodes `x2002c0s9b0n0` + `x2003c0s1b0n0`)

The section 13.2 caution is resolved; the fabric is healthy for MPI, and the NVSHMEM
blocker is now precisely characterized.

**Inter-node MPI works.** The `ibv_create_ah` timeout is specific to the UCX `ud`
transport; `rc`, `dc`, and `tcp` all pass an 8-rank 2-node host-memory `MPI_Allreduce`.
For multi-node runs extend the pin to

    export UCX_TLS=sm,self,cuda_copy,cuda_ipc,rc

**Caution: do not use that extended set for single-node measurements.** With `rc` present,
UCX routes small intra-node messages through NIC loopback: an intra-node GPU ping-pong that
floors at ~11.5 us under the sm-only set shows ~17-18 us one-way at small sizes with `rc`
listed. Keep the original `sm,self,cuda_copy,cuda_ipc` for all intra-node arms.

**Inter-node GPU-aware MPI works and is host-staged** (no GPUDirect RDMA): 2-node GPU
ping-pong measures ~19 us one-way at 8 B-4 KB, ~39 GB/s at 16 MB (a good fraction of the
NIC; the staging pipeline is efficient at bulk). The ~19 us unhideable host-blocked latency
per small exchange is exactly the regime the section 15 trade predicts NVSHMEM would win --
if it could run.

**Multi-node NVSHMEM is hard-blocked, all four transports, with evidence:**

| transport | failure |
| --- | --- |
| `ibrc` | `neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.` |
| `ibdevx` | same peermem check, same message |
| `ibgda` (with `NVSHMEM_IB_ENABLE_IBGDA=1`; without it: "IBGDA Disabled by the environment") | same peermem check in `ibgda.cpp:nvshmemt_init` |
| `ucx` | initializes, then `Failed to map memory in UCX transport` registering the symmetric heap with the NIC |
| `libfabric` | `nvshmem detect topo failed` (no matching provider on this fabric) |

The dmabuf escape hatch is also closed: the IBGDA transport contains a dmabuf registration
path (`ibv_reg_dmabuf_mr`), but `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` = **0** on this
driver stack, so it falls back to the peermem check and dies. **The single fix is root
loading `nvidia_peermem` (module ships with driver 610.57.04) or switching the nodes to the
open-source kernel module for dmabuf support -- ALCF ticket material, now with exact
evidence.** A ready-to-paste ticket draft with all signatures, control experiments, and
verification steps is in `ALCF-TICKET-PEERMEM.md` in this worktree.

**NCCL is the one GPU-orchestrated transport that crosses nodes today** (it falls back to
its own host-staged IB/socket path without GDR); see the NCCL sections in
`NVSHMEM-PERF-NOTES.md`.

---

## 14. Attempting lever A: two failed hypotheses and what the numbers actually say

Section 13.1 proposed cutting PetscSF's ~10 us per-exchange overhead. Two concrete changes
were implemented and measured. **Both failed**, and the measurements that killed them are
more useful than the changes would have been. The tree is back to its original state; only
the knowledge is kept.

### 14.1 Attempt 1 -- issue communication on one stream (FAILED, made it worse)

Hypothesis: `remoteCommStream` plus the `dataReady`/`endRemoteComm` event pairs in
`PetscSFLinkBuildDependenceBegin()`/`End()` cost 4 CUDA calls and 2 serialization points per
exchange; issuing everything on `link->stream` removes them, and a single stream is totally
ordered so correctness is preserved.

Implemented behind `-sf_nvshmem_single_stream`. All 6 correctness tests passed. Performance,
best of 3, `--bind-to none`:

| bytes | MPI | NV 2-stream (default) | NV 1-stream | change |
| ---: | ---: | ---: | ---: | ---: |
| 1 KiB  | 11.3 | **16.5** | 23.2 | **-41%** |
| 64 KiB | 12.0 | **17.8** | 23.9 | -34% |
| 32 MiB | 266  | **329**  | 334  | -1.5% |

**The dedicated stream is load-bearing, not overhead.** The `<<<1,1>>>` spin-wait kernels
block whatever stream they are on. On a separate high-priority stream they overlap with
other GPU work; forced onto `link->stream` they serialize everything behind them. Reverted.

### 14.2 The measurement that explains it

A standalone CUDA micro-benchmark (`launchcost.cu`) on this H100:

| operation | cost |
| --- | --- |
| empty `<<<1,1>>>` launch, pipelined | **1.91 us** |
| 2x (`cudaEventRecord` + `cudaStreamWaitEvent`) | **0.53 us** |
| PETSc-shaped: 5 launches + 4 event ops | **10.22 us** |

That 10.22 us matches the measured PETSc-vs-raw-NVSHMEM gap (16-18 us vs 6.4 us) almost
exactly, which validates the section 13.1 model -- but it also shows attempt 1 targeted the
wrong term. **The events are 0.53 us, about 5% of the overhead. The five kernel launches at
~1.9 us each are the other 95%.**

### 14.3 Attempt 2 -- replace the `<<<1,1>>>` signal kernels with host stream APIs (FAILED before implementation)

Of the ~5 launches per exchange, only pack and unpack do real work; three are pure
signalling (`WaitSignalsFromLocallyAccessible`, `PutDataEnd`, and the PostUnpack
`NvshmemSendSignals`). Removing those three would save ~5.7 us and put PETSc's NVSHMEM path
level with MPI. NVSHMEM 3.4.5 exposes host-side stream-ordered equivalents, so the swap
looked free.

Measured first (`sigcost.cu`, 2 PEs) rather than assumed:

| operation | cost |
| --- | --- |
| empty kernel launch (baseline) | 2.02 us |
| `<<<1,1>>>` kernel performing `nvshmemx_signal_op()` | 2.43 us |
| `nvshmemx_signal_op_on_stream()` (host API) | **2.55 us** |
| `nvshmemx_signal_wait_until_on_stream()` (host API) | **2.24 us** |

**The host-side stream APIs cost the same as a kernel launch** -- they evidently enqueue an
internal kernel. Substituting them gains nothing. Not implemented.

### 14.4 What this leaves

The real constraint is now clear and quantitative: **every stream-ordered operation on this
GPU costs ~2 us, regardless of whether it is a user kernel, an NVSHMEM device kernel, or an
NVSHMEM host stream call.** PETSc's put protocol needs about five of them, so its floor is
~10 us of orchestration on top of a ~6 us transport. You cannot make an operation cheaper;
you can only issue fewer.

That rules out substitution and leaves two real options:

1. **Fuse operations.**
   - `nvshmemx_putmem_signal_nbi_on_stream()` performs the put and sets the remote signal in
     one stream op, replacing put + `quiet` + a separate signal, and removing the ordering
     workaround flagged in the source (*"Calling nvshmem_fence/quiet() does not fence the
     above nvshmemx_putmem_nbi_on_stream!"*).
   - The PostUnpack signal could be written by the **unpack kernel itself** -- that kernel is
     already being launched, so the signal becomes free rather than a 6th operation.
   - Optimistic ceiling: 5 ops -> ~3, i.e. ~17 us -> ~14 us. Better, but still short of MPI's
     ~11.5 us.
2. **CUDA Graphs.** Capture the fixed exchange sequence once and replay it as a single
   launch. This is the only route that can plausibly beat MPI on one node, because it attacks
   the per-operation cost itself rather than the operation count. Risk: the protocol contains
   data-dependent spin-waits, which are awkward to capture, and NVSHMEM operations may not be
   capture-safe. Verify capturability before investing.

**Honest bottom line for this machine:** PETSc's NVSHMEM path is structurally ~2 us x 5
operations behind a ~6 us transport, and GPU-aware MPI over NVLink already does the same job
in ~11.5 us. Closing that on a single node requires fusion plus graph capture, not tuning.
The design pays off where MPI's own per-message cost is much larger -- i.e. multi-node
(section 13.2) -- which is exactly what this machine cannot currently test.

---

## 15. Control flow: MPI path vs NVSHMEM path

Both diagrams trace one `PetscSFBcastBegin()`/`PetscSFBcastEnd()` pair, root-to-leaf, with
device root/leaf data, on a single node.

### MPI path (GPU-aware MPI over UCX `cuda_ipc`)

      HOST (CPU)                        GPU: link->stream          WIRE
      ==========                        =================          ====

    PetscSFBcastBegin_Basic
      |
      +- PetscSFLinkCreate ---------> link from cache;
      |                               persistent MPI reqs already built
      |
      +- PackRootData --------------> +-------------+
      |     launch (1)  ~1.9us        | pack kernel |  rootdata -> rootbuf
      |                               +-------------+
      |
      +- StartCommunication_Persistent_Basic          (sfbasic.c:58)
      |    |
      |    +- SyncStreamBeforeCallMPI ===> cudaStreamSynchronize
      |    |    ### HOST BLOCKS ###        MPI is not stream-aware, so the
      |    |                               host must wait for pack to land
      |    +- MPI_Startall_irecv  -+  persistent requests: built once,
      |    +- MPI_Startall_isend  -+- restarted per call --->  UCX cuda_ipc
      |                                                        P2P / NVLink
      +- ScatterLocal   (self-to-self, overlaps the wire)
      |
    PetscSFBcastEnd_Basic
      |
      +- FinishCommunication_Default                  (sfmpi.c:6)
      |    +- MPI_Waitall   ### HOST BLOCKS ### <----------  data lands
      |
      +- UnpackLeafData ------------> +---------------+
            launch (2)  ~1.9us        | unpack kernel |  leafbuf -> leafdata
                                      +---------------+

      per exchange: 2 kernel launches + 1 blocking stream sync + 2 MPI calls
      measured floor: ~11.5 us

### NVSHMEM path (put protocol; all peers locally accessible via `nvshmem_ptr()`)

      HOST (CPU)              GPU: link->stream    GPU: remoteCommStream     WIRE
      ==========              =================    =====================     ====

    PetscSFBcastBegin_Basic
      |
      +- PetscSFLinkCreate_NVSHMEM -> symmetric bufs + 4 signal arrays in
      |                               nvshmem heap; 2nd stream + 2 events
      |
      +- PackRootData ------> +-----------+
      |    launch (1)         |pack kernel|
      |                       +-----------+
      +- PutDataBegin_NVSHMEM
      |   +- BuildDependenceBegin
      |   |    cudaEventRecord(dataReady, stream) --+  all 4 event ops
      |   |    cudaStreamWaitEvent(commStream) <----+  total only 0.53us
      |   |
      |   +- launch (2) ---------------------------> +----------------------+
      |   |   WaitSignalsFromLocallyAccessible       | <<<1,1>>> SPIN       |
      |   |   wait ssig==0, set 1  (flow control)    +----------------------+
      |   |
      |   +- nvshmemx_putmem_nbi_on_stream --------> +--------+ ==========> NVLink
      |   |      (one call per neighbour)            |  put   |             P2P/IPC
      |   |                                          +--------+
      |   +- nvshmemx_quiet_on_stream -------------> +--------+
      |        orders put before signal              | quiet  |
      |        (fence does NOT cover _on_stream)     +--------+
      +- ScatterLocal
      |
    PetscSFBcastEnd_Basic
      |
      +- PutDataEnd_NVSHMEM
      |   +- launch (3) ---------------------------> +----------------------+
      |   |   PutDataEnd<<<1,1>>>                    | signal each dst =====|==> rsig:=1
      |   |                                          | then SPIN on my rsig |<== peer's
      |   |                                          +----------------------+    signal
      |   +- BuildDependenceEnd
      |        cudaEventRecord(endRemoteComm, commStream) --+
      |        cudaStreamWaitEvent(stream) <----------------+
      |
      +- UnpackLeafData ----> +-------------+
      |    launch (4)         |unpack kernel|
      |                       +-------------+
      |
      +- PostUnpack: launch (5) ----------------> +--------------------+
            NvshmemSendSignals                    | ssig:=0 on senders |==> frees
                                                  +--------------------+    sender buf

      ### HOST NEVER BLOCKS ### - everything is stream-ordered
      per exchange: 5 stream ops x ~2.0us = ~10us orchestration + ~6us transport
      measured floor: ~16-18 us

### The trade, in one line

    MPI      : 2 launches + 1 HOST BLOCK   -> fewer ops, but CPU stalls on the GPU
    NVSHMEM  : 5 stream ops, no host block -> CPU free, but 5 x 2us of orchestration

MPI wins on one node because the host block is cheap there: the pack kernel has already
finished by the time `MPI_Startall` runs, and persistent requests make the MPI calls nearly
free. NVSHMEM's asynchrony buys nothing when there is no other host work to overlap with,
and it costs three extra stream operations.

Multi-node flips this: `MPI_Waitall` then blocks on a real network round trip, while
NVSHMEM's five operations stay ~10 us and the GPU keeps running.

Note that operations (2), (3) and (5) are **pure signalling** -- only (1) and (4) move data.
Those three are the ~5.7 us of pure protocol overhead identified in section 14, and the
reason fusion (folding (5) into (4), and put+signal into a single op) is the only remaining
lever short of CUDA Graph capture.

---

## 16. When does NVSHMEM actually win? Two experiments

Motivated by the question: the section 10/12 results all came from one tightly-coupled global
SNES on one node. Would a *decoupled* scheme -- several independent solves exchanging with
neighbours -- favour NVSHMEM? Benchmark: `sfbench2.c`, np=4, 32 KB messages, `--bind-to none`.

### 16.1 Experiment A -- K independent exchanges in flight (hypothesis REFUTED)

K independent buffer pairs on one SF, so K SF links are live at once: all `BcastBegin`s, then
all `BcastEnd`s. MPI must host-synchronize once per `Begin`; NVSHMEM never blocks the host, so
NVSHMEM "should" pull ahead as K grows.

| K | MPI us/exchange | NVSHMEM us/exchange |
| ---: | ---: | ---: |
| 1  | 17.04 | 18.15 |
| 2  | 11.02 | 16.13 |
| 4  | **8.29** | 16.05 |
| 8  | **8.57** | 16.42 |
| 16 | **8.13** | 16.61 |

**The opposite happens.** MPI's per-exchange cost halves (17 -> 8 us) while NVSHMEM stays flat
at ~16 us; the gap widens from 1.07x to 2.0x.

Why: MPI's host sync is only expensive when the stream is actually busy. By the second
`Begin`, the previous pack has long finished, so each subsequent sync is nearly free and the
persistent requests pipeline. NVSHMEM has nothing to amortize -- its five stream operations
are *per exchange* and stream-ordered, so K exchanges cost K x 5 operations no matter what.

**Simply having many outstanding exchanges does not help NVSHMEM.**

### 16.2 Experiment B -- independent compute between Begin and End (hypothesis CONFIRMED)

Same benchmark, K=1, with `-naxpy m` doing m `VecAXPY`s on an 8 MB device vector between
`BcastBegin` and `BcastEnd`. "Exposed" = total - compute-alone, i.e. the communication cost
that failed to hide.

| AXPYs | compute alone | MPI total | MPI exposed | NVSHMEM total | NVSHMEM exposed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0  | 17.14 | 17.14 | 18.27 | 18.27 |
| 1 | 5.3  | 20.42 | 15.16 | 20.24 | 14.83 |
| 2 | 10.5 | 22.61 | 12.10 | 20.71 | **9.94** |
| 4 | 21.0 | 29.88 |  8.91 | 27.10 | **5.59** |
| 8 | 41.9 | 50.00 |  8.11 | 48.63 | **5.70** |

**NVSHMEM hides ~70% of its cost (18.3 -> 5.6 us); MPI bottoms out at ~8.1 us.** With enough
overlappable work NVSHMEM's exposed cost is ~30% lower than MPI's -- the only configuration
measured in this whole exercise where NVSHMEM wins.

The mechanism is exactly the structural difference in section 15: NVSHMEM's work is
stream-ordered and proceeds while the compute kernel runs. MPI's `cudaStreamSynchronize` +
`MPI_Waitall` are host blocks that cannot be hidden behind GPU work no matter how much of it
there is; ~8 us of MPI's cost is irreducibly exposed.

### 16.3 What this means for algorithm choice

The discriminator is **not** "one global solve vs many decoupled solves", and **not** how many
exchanges are outstanding. It is:

> **Is there independent GPU work available to overlap with the exchange?**

- Tight `MatMult` -> `Allreduce` -> `MatMult` Krylov iteration: nothing to overlap, every
  exchange is on the critical path. MPI wins (sections 10, 12).
- Many exchanges issued back-to-back with no compute: MPI wins by a *wider* margin (16.1).
- Real computation overlapping the exchange: NVSHMEM wins by ~30% of exposed cost (16.2).

A decoupled per-GPU-subdomain scheme would benefit -- but because it creates long stretches of
independent local work to hide the exchange behind, not because the exchanges are independent
of each other. Note also that such a scheme removes the global Krylov reductions
(`VecMDot`/`VecNorm`), which PETSc's NVSHMEM path does not accelerate at all (section 12.2)
and which grew to 35% of runtime at `-da_refine 7`.

**Caveat, and it is not a small one:** replacing one global Newton solve with per-subdomain
solves plus neighbour exchange is a Schwarz-type domain decomposition. It generally needs more
outer iterations to reach the same accuracy, so a ~30% saving on exposed communication can be
erased by extra iterations. This is a numerical-methods tradeoff first and a communication
tradeoff second.

### 16.4 Experiment B as a timeline

    MPI PATH
                         time ------------------------------------------->

     HOST    |launch|####### BLOCKED #######|Start|enqueue|  Waitall  |launch |
     (CPU)   | pack | cudaStreamSynchronize | all | AXPYs | (cheap if |unpack |
             |      |  waits for pack to    |irecv|       |  transfer |       |
             |      |  actually EXECUTE     |isend|       |  finished)|       |
             +--+---+-----------------------+--+--+---+---+-----------+---+---+
                |                              |      |                   |
     GPU     +--v---+                          |   +--v--------------+ +--v---+
     stream  | pack |                          |   |   AXPY x m      | |unpack|
             +------+                          |   +-----------------+ +------+
                                               |
     WIRE                                      +->#################
     (UCX copy engine / IPC)                      #   transfer     #  overlaps OK
                                                  #################

             |---------- EXPOSED ~8.1 us ----------|  |-- hidden --|
             The stall is on the HOST and happens BEFORE any compute is
             enqueued -- there is nothing yet to hide it behind.


    NVSHMEM PATH
                         time ------------------------------------------->

     HOST    | enqueue pack, events, spin, put, quiet, AXPYs, PutDataEnd, unpack |
     (CPU)   |                   NEVER BLOCKS -- returns immediately             |
             +-+--+---+---+----+----------+-------------------+------------------+
               |  |   |   |    |          |                   |
     GPU     +-v--+|   |   |    |   +------v------------+   +--v---+
     link->  |pack||   |   |    |   |     AXPY x m      |   |unpack|
     stream  +----+|   |   |    |   +-------------------+   +------+
                   |   |   |    |            ^
     GPU        +--v---v---v----v----------+ |  runs CONCURRENTLY
     remoteComm |spin| put |quiet|PutDataEnd| |  on the second stream OK
     stream     +----+-----+-----+----------+ |
                                              |
     WIRE            #################        |
                     #   transfer   # --------+
                     #################

             |-- EXPOSED ~5.6 us --|
             Orchestration is GPU-side on its own stream, so it overlaps the
             AXPYs. Only pack + unpack stay on the critical path.

The two floors decompose as:

    MPI      exposed ~= pack launch + pack EXECUTION (synchronously waited)
                        + MPI_Startall + unpack launch          ~= 8.1 us
    NVSHMEM  exposed ~= pack launch + unpack launch
                        (everything else overlapped)            ~= 5.6 us

The entire difference reduces to one sentence: **MPI synchronously waits for the pack kernel
to finish executing; NVSHMEM never does.** That is structural, not tunable -- MPI is not
stream-aware, so the CPU must know the pack landed before handing the buffer to UCX, and no
amount of GPU work hides a blocked CPU.

It also explains experiment A: with no compute in the window there is nothing to overlap, so
NVSHMEM's second stream buys nothing while still paying five stream operations, and MPI's
sync gets progressively cheaper as the stream drains.

### 16.5 Experiment B across message sizes -- the overlap win is regime-dependent

Section 16.2 used a single message size (n=4096 scalars = 32 KB; the 8 MB figure there is the
AXPY *work vector*, not the message). Sweeping the message size shows the win is not universal.

Exposed cost (= total - compute_alone) with ample overlappable compute, naxpy=64, np=4:

                NVSHMEM exposed / MPI exposed        (< 1.0 = NVSHMEM better)

       4 KB   0.58  |###########:............|   NVSHMEM 42% cheaper
      32 KB   0.56  |###########:............|   NVSHMEM 44% cheaper
     512 KB   0.66  |#############:..........|   NVSHMEM 34% cheaper
       4 MB   1.09  |#####################:  |   break-even
      32 MB   1.25  |#########################|  NVSHMEM 25% WORSE
                    +--------------+-------------+
                     LATENCY-BOUND   BANDWIDTH-BOUND
                     overlap hides   both saturate the same
                     orchestration   NVLink; nothing to hide behind

Each message size was run at three levels of overlappable compute (`naxpy` = number of
`VecAXPY`s issued between `BcastBegin` and `BcastEnd`; each AXPY is ~5.3 us on an 8 MB device
vector). `exposed = total - compute_alone` is the part of the exchange that did **not** hide.
Read *down* a group to see the effect of adding overlap; read *across* groups (same naxpy) to
see the effect of message size.

    message | naxpy |  compute |     MPI      MPI |  NVSHMEM  NVSHMEM |  ratio
       size |       |    alone |   total  exposed |    total  exposed | NV/MPI
    ========================================================================
       4 KB |     0 |      0.0 |    17.4     17.4 |     18.2     18.2 |   1.04   tie
       4 KB |     8 |     42.0 |    50.2      8.2 |     48.6      5.6 |   0.68   NVSHMEM better
       4 KB |    64 |    336.1 |   343.7      7.6 |    349.0      4.4 |   0.58   NVSHMEM better
    ------------------------------------------------------------------------
      32 KB |     0 |      0.0 |    17.6     17.6 |     18.3     18.3 |   1.04   tie
      32 KB |     8 |     41.8 |    50.0      8.2 |     48.6      5.7 |   0.69   NVSHMEM better
      32 KB |    64 |    336.4 |   344.1      7.7 |    349.0      4.3 |   0.56   NVSHMEM better
    ------------------------------------------------------------------------
     512 KB |     0 |      0.0 |    21.3     21.3 |     22.6     22.6 |   1.06   MPI better
     512 KB |     8 |     41.8 |    50.6      8.8 |     49.5      6.6 |   0.75   NVSHMEM better
     512 KB |    64 |    334.7 |   344.2      9.5 |    349.7      6.3 |   0.66   NVSHMEM better
    ------------------------------------------------------------------------
       4 MB |     0 |      0.0 |    49.2     49.2 |     52.8     52.8 |   1.07   MPI better
       4 MB |     8 |     42.0 |    69.2     27.2 |     57.4     14.3 |   0.53   NVSHMEM better
       4 MB |    64 |    334.5 |   346.1     11.6 |    356.1     12.7 |   1.09   MPI better
    ------------------------------------------------------------------------
      32 MB |     0 |      0.0 |   272.1    272.1 |    327.9    327.9 |   1.21   MPI better
      32 MB |     8 |     42.0 |   289.8    247.8 |    338.9    295.5 |   1.19   MPI better
      32 MB |    64 |    334.5 |   406.7     72.2 |    433.3     90.1 |   1.25   MPI better
    ------------------------------------------------------------------------

Worked example, the `32 KB, naxpy=64` row: 336.4 us of compute by itself, 349.0 us when the
exchange runs too -- so the whole exchange added only **4.3 us** of wall time. MPI in the same
slot added 7.7 us. Ratio 0.56.

**Small and medium messages (<= 512 KB): NVSHMEM wins 35-44%.** The exchange is latency-bound,
dominated by the ~10 us of per-operation orchestration, and that orchestration is GPU-side so
compute hides it. MPI's host stall cannot be hidden by any amount of GPU work.

**Large messages (>= 4 MB): the advantage disappears.** The exchange becomes bandwidth-bound,
so the transfer and the overlapping compute contend for the same memory/NVLink bandwidth.
Overlap cannot help when both saturate the same resource, and NVSHMEM's raw ~1.24x bandwidth
deficit (section 10.2) becomes the whole story.

The 4 MB row shows the crossover directly: at naxpy=8 (compute < comm) NVSHMEM still wins at
0.53, but at naxpy=64 (compute >> comm, both fully hidden) it is 1.09 -- the residual is just
pack/unpack plus contention, and the two converge.

**Caveat:** `exposed = total - compute_alone` assumes the compute takes the same time with and
without communication. In the bandwidth-bound rows that is false -- they contend -- so the
4 MB and 32 MB ratios are a rougher decomposition than the small-message ones. The qualitative
flip is robust; the exact large-message ratios are not.

**Refined rule: NVSHMEM's overlap advantage requires the exchange to be latency-bound.** A
decoupled per-subdomain scheme lands in the winning regime only if its boundary exchanges stay
below roughly 1 MB.
