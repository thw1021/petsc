# Handoff: convert the device COO paths to the compacted local index map

This branch (`lindad/vec-coo-prealloc-local-size`, off `origin/main`) changed how a vector's
*local* COO contributions are recorded. The host implementations are converted, built and
tested. The CUDA/HIP and Kokkos implementations are **not**, and the tree will not build
correctly with either enabled until they are. That is the whole of the remaining work.

This file is scaffolding for that task and should be dropped before the branch is submitted
upstream.

## What changed, and why the device paths break

`VecSetPreallocationCOO_Seq` and `VecSetPreallocationCOO_MPI` used to build

    jmap1[m+1]    // m = vector's local size

so entry `i` of the vector owned `perm1[jmap1[i] .. jmap1[i+1])`. Cost and memory were
proportional to `m` no matter how few contributions were described.

They now build, mirroring the naming the *remote* half of the MPI path already used
(`nnz2`, `imap2[nnz2]`, `jmap2[nnz2+1]`):

    nnz1            // number of local entries receiving a contribution
    imap1[nnz1]     // the i-th such entry is imap1[i] in the vector
    jmap1[nnz1+1]   // that entry owns perm1[jmap1[i] .. jmap1[i+1])

`perm1[tot1]` and `tot1` are unchanged. Anything that still indexes `jmap1` by a vector
entry, or sizes it `m+1`, is now wrong: it will read past `nnz1+1` and produce garbage
rather than fail loudly.

## The two semantics to preserve

1. **`INSERT_VALUES` zeroes entries no contribution names.** This is the reason the old loop
   ran over all `m`. With a compacted map you must zero the whole array first, then add. The
   host code does `PetscArrayzero` then accumulates; `add_coo_values_impl` doing `xv[idx] = sum`
   is also correct after an explicit zero, since a named entry receives exactly `sum`.
2. **For MPI, zero before both accumulations.** Remote values are added on top of what the
   local pass leaves, so zeroing between them would discard local contributions. An entry named
   only remotely must still start from zero.

## Sites to convert

Grep `jmap1` under `src/vec/` and `include/petsc/private/veccupmimpl.h`; every hit outside the
four converted host files is work. As of this commit:

| File | Line | What to do |
|---|---|---|
| `include/petsc/private/veccupmimpl.h` | 150 | add `imap1_d`, fix the `jmap1_d` size comment |
| `include/petsc/private/veccupmimpl.h` | 1077 | add `std::ref(vcu->imap1_d)` to the free list |
| `include/petsc/private/veccupmimpl.h` | 1107 | `make_coo_pair` for `imap1_d`/`nnz1`; `jmap1_d` becomes `nnz1+1`, not `map->n+1` |
| `src/vec/vec/impls/seq/cupm/vecseqcupm.hpp` | 167 | `add_coo_values` takes `imap1` and passes `[=](i){return imap1[i];}` instead of the identity |
| `src/vec/vec/impls/seq/cupm/vecseqcupm_impl.hpp` | 2366 | launch over `nnz1`, pass `imap1_d`; zero the array first under `INSERT_VALUES` |
| `src/vec/vec/impls/mpi/cupm/vecmpicupm_impl.hpp` | 358 | the same for the local half; the remote half at 287 is already compacted and is the model to copy |
| `src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp` | 28, 110, 118 | add `imap1_d`; mirror it with `nnz1`, and `jmap1` with `nnz1+1` in both `SetUpCOO` overloads |
| `src/vec/vec/impls/seq/kokkos/veckok.kokkos.cxx` | 1649, 1672 | `RangePolicy(0, nnz1)`, write `xv(imap1(i))`; zero via `Kokkos::deep_copy` under `INSERT_VALUES` |
| `src/vec/vec/impls/mpi/kokkos/mpikok.kokkos.cxx` | 157, 190 | the same for the local half |

Two notes on the device zeroing, which is the only part that is not mechanical. The CUPM
idiom is `PetscCUPMMemsetAsync`, used in `veccupmimpl.h` around line 603 -- check its exact
signature there rather than assuming. Keep the existing `if (const auto n = x->map->n)` guard
structure: the memset is needed whenever `n > 0`, including when `nnz1 == 0`, and the empty-vector
branch must stay.

## Verifying

`src/vec/vec/tests/ex61.c` already carries the cases. It sweeps `nsize` 1/2/3 and
`-ignore_remote` 0/1, exercises both `ADD_VALUES` and `INSERT_VALUES`, and `-sparse` lengthens
the vector to 10000 while leaving the pattern alone so most entries receive nothing -- which is
the regime this change is about, and the one where an unconverted device path will disagree.
The `kokkos_sparse` and `cuda_sparse` variants are registered and waiting for a build that can
run them.

    make -f gmakefile test search='vec_vec_tests-ex61*'
    make -f gmakefile test search='vec_vec*'          # 316 pass, 0 fail on host

On the host, with no device support configured, both already pass at this commit. A device build
that passes `ex61_cuda_sparse` and `ex61_kokkos_sparse` at 1, 2 and 3 ranks is the bar for the
conversion being done.

## Measuring the point of the change

`VecSetPreallocationCOO` cost should follow the contribution count, not the vector length. A
short program that creates an `m`-entry vector, preallocates `nc` scattered contributions and
times the preallocation and repeated `VecSetValuesCOO` calls is enough to show it: sweep `nc`
over two decades at fixed `m` and confirm the time moves with `nc`. Before this commit it was
flat. On the host in a debug build, at `m` = 16777216, the preallocation went from 40.7 ms to
0.1 ms at `nc` = 1000, and `VecSetValuesCOO` from 32.0 ms to under 0.1 ms per call. The device
figures are unmeasured and are the other half of what a CUDA-capable machine can add.

## Where this came from

A MOOSE Kokkos solver reduced off-process contributions to a 16.8M-entry vector through this
interface. The contribution count is the number of shared degrees of freedom, far smaller than
the vector, and there is exactly one contribution per shared degree of freedom so the
preallocation has nothing to coalesce. It paid 128 MB and about 40 ms per work vector, times
the several dozen work vectors a Krylov and multigrid solve cycles among, because PETSc caches
the preallocation on the vector it was called for.
