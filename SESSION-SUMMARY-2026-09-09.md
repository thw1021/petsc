# Session Summary — 2026-09-09: SF-NCCL backend compiled, verified and measured (job 2779, 2 nodes x 4 H100)

Everything section 5 of `SESSION-SUMMARY-2026-09-04.md` asked for was done. Results:
`NVSHMEM-PERF-NOTES.md` section 29 (the write-up) and `nvshmem-tools/results-20260909-sfnccl.txt`
(raw). Nothing is committed; see "state" below.

## What changed in the tree today

- `config/BuildSystem/config/packages/NCCL.py`: static library name is `libnccl_static.a`. The
  version probe (`NCCL_MAJOR.NCCL_MINOR.NCCL_PATCH` under nvcc) worked as written: configure reports
  NCCL 2.28.3. Configure line: the usual `PETSC_CONFIGURE_OPTS` plus
  `--with-nccl-dir=/soft/libraries/nccl/nccl_2.28.3-1+cuda12.9_x86_64`.
- `src/vec/is/sf/impls/basic/nccl/sfnccl.cu`: compiled without warnings and unchanged in logic.
  Two edits: message accounting through `PetscAddLogDouble()` (raw `+=` on `petsc_isend_ct` never
  reached `-log_view`, whose event rows read the `_th` copies), and the communicator-creation
  `PetscInfo` now prints the `ncclCommInitRank` time (2.4-2.9 s here).
- `janus-env-nvshmem.sh`: `UCX_IB_GID_INDEX=2` default (section 29.7 trap).
- `nvshmem-tools/`: new `sfgraph.c` (CUDA-graph capture of PetscSF exchanges), `sfbench2.c -w`
  (work-vector length; the 1M default is host-launch-bound), README rows, results file.
- clang-format 22 (`pip install --user clang-format`, in `~/.local/bin`) and `make checkbadSource`
  are clean for the PETSc sources; the only checkbadSource hit is `nvshmem-tools/acgnbench.c`
  (campaign file, not for upstream).

## Verified

ex22 / snes ex19 / ksp ex45 byte-identical to MPI at np=4 and np=8 (2 nodes), harness test
`vec_is_sf_tests-ex22_cuda_nccl` ok at nsize 1 and 4, all three eligibility branches exercised
(single rank refuses, shared GPUs fall back with an info line, 8 GPUs over 2 nodes engage).

## Findings (numbers in PN 29)

1. Ring: NCCL 2x faster than GPU-aware MPI below 256 KB on a node, 1.3x across nodes up to 1 MB,
   1.2-1.5x slower above 2 MB. 2-17x faster than non-GPU-aware MPI.
2. PETSc's default (legacy null) stream costs NCCL ~35 us per exchange next to compute; use
   `-root_device_context_stream_type nonblocking`. With it NCCL wins the ACGN branch DAG
   (93 vs 148 us at 64 KB on a node, 165 vs 190 at 2+2).
3. CUDA-graph capture of Begin/End works (MPI cannot): 45 -> 9 us per exchange-plus-8-AXPYs.
4. Real solver (ksp ex45): NCCL setup costs ~2.9 s once (`ncclCommInitRank` 2.4 s + 128 lazy p2p
   connections in the first exchange); per iteration NCCL is 20% cheaper on a node and 2x more
   expensive across two nodes (unexplained, not chased).
5. Fabric trap: UCX on RoCE v2 next to NCCL on RoCE v1 stalls host MPI collectives 40-350x at
   2+2; fixed by `UCX_IB_GID_INDEX=2`.
6. Several independent exchanges in flight serialize under NCCL (one communicator orders them).

## State

Uncommitted, on top of the two `wip` commits: `NCCL.py` and `sfnccl.cu` are still untracked (the
rest of the backend was in the 2026-09-05 `wip`), plus the env/tool/doc changes above. For the
upstream MR: the PETSc sources plus `doc/changes/dev.md` and the ex22 test; keep the campaign files
out. Open items for the MR text: document the one-time init cost and the nonblocking-stream
recommendation in the `-use_nccl` option help; consider a per-link `ncclCommSplit` for concurrent
exchanges; the 2-node DMDA per-iteration loss deserves an nsys look before claiming a win there.

## Later the same day

- Backend committed as `1055b7feb89` (the four backend files; the rest of the edits sit in the
  earlier `wip` commit).
- The >2 MB loss taken apart with `nvshmem-tools/sfbw.c` (PN 29.9-29.11): PetscSF adds nothing;
  NCCL's p2p is an SM-driven staged copy with 4 blocks per 4 MB op shared between directions,
  MPI's is a copy-engine DMA. `NCCL_P2P_NVL_CHUNKSIZE=65536` and `NCCL_NCHANNELS_PER_NET_PEER=8`
  (now env-script defaults) cut the one-node bulk loss to 9-14% and lift 512 KB-1 MB across
  nodes by 15-23%; from 8 MB up a rank straddling NVLink and the network still loses 1.3-1.5x.
  Host-staged MPI at 2+2 above 8 MB stalls by milliseconds run to run on either RoCE GID.
