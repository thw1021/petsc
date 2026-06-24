# paper_data — Reproducibility Artifacts

Branch: `adams/seqbaijkokkos-gamg`
Machine: Perlmutter (NERSC), A100 GPUs, job allocation m1516_g
PETSc revision at time of runs: `v3.25.2-340-g83d2bd4a8b4` (2026-06-21)

## Contents

### logs/

All raw PETSc `-log_view` output captured during benchmarking.

| Directory | What it contains |
|-----------|-----------------|
| `ne63/`       | Main n=4 GPU weak-scaling point: scalar (`aijk`) and block (`baijk`) at ne=63. The `*_unprec.log` files are the definitive runs with `-ksp_norm_type unpreconditioned`; `*_logview.log` are full event-table dumps. |
| `ne127/`      | Capacity / OOM study: `aijk_ne127_oom.log` shows `mpiaijkokkos` (cuSPARSE) failing in `KokkosSparse_spgemm_symbolic` at ne=127 n=4; `blk127.log` shows `mpibaijkokkos` completing; `aij127.log` is a stub. |
| `ptap/`       | nsys-scoped hot PtAP comparison (TPL vs no-TPL): `ptap_runs2.log` is the canonical 8-run set (4 configs × 2 arches) yielding the cuSPARSE-vs-KK 7.7× gap. `ptap_runs.log` is an earlier set. |
| `notpl_bench/`| No-TPL Q1/Q2, n=1/8: built with `--with-kokkos-kernels-tpl=0` (arch `arch-perlmutter-opt-gcc-kokkos-cuda-notpl`). Paired files: `*_gputimer.log` (async GPU timing on) and `*_notimer.log` (baseline). |
| `tpl_sameday/`| TPL Q1/Q2, n=1/8: same-day runs with `arch-perlmutter-opt-gcc-kokkos-cuda` for apples-to-apples TPL comparison against `notpl_bench/`. |
| `q2_fe/`      | Q1/Q2 FE nnz/row study, n=1/8: `snes/tutorials/ex56` (DMPlex+PetscFE, Q2 tensor elements), comparing scalar vs block across element order. |

### figs/

Plotting scripts and generated figures.

| File | Description |
|------|-------------|
| `plot_speedup.py`     | Generates `speedup_vs_gpus.png`: weak-scaling speedup (block/scalar) for KSPSolve, SpMV, PtAP at n=1/8/27/64. |
| `plot_events.py`      | Generates event-timeline bar charts from log files (`events_ne63.png`, `events_all.png`, etc.). |
| `plot_events_oom.py`  | Generates `events_oom.png` / `events_ne127.png` from OOM and capacity runs. |
| `plot_events_n8.py`   | Generates `events_n8.png` detail for n=8 comparison. |
| `speedup_vs_gpus.png` | Fig 2 candidate: block/scalar speedup ratios across GPU counts. |
| `events_ne63.png`     | Event timelines at ne=63, n=4. |
| `events_ne31.png`     | Event timelines at ne=31, n=4. |
| `events_n8.png`       | Detailed event breakdown at n=8. |
| `events_all.png`      | All scales overlaid. |
| `events_q1.png`       | Q1 vs Q2 comparison (scalar). |
| `events_q2.png`       | Q2 event timelines. |
| `events_q1q2.png`     | Q1/Q2 side-by-side (block vs scalar). |
| `events_oom.png`      | OOM vs success event comparison. |
| `events_ne127.png`    | ne=127 capacity result. |

### scripts/

Run scripts and nohup driver logs showing the exact `srun` command lines used.

| File | Description |
|------|-------------|
| `run_parity.sh`           | Parity check: runs both `aijkokkos` and `mpibaijkokkos` at ne=7 n=1/8, confirms iteration count matches. |
| `run_mb.sh`               | Quick mpibaijkokkos sanity run at ne=7 n=8. |
| `notpl_bench_nohup.log`   | Console output capturing the exact `srun` invocations for the no-TPL benchmark suite (`notpl_bench/`). |
| `tpl_sameday_nohup.log`   | Console output for the same-day TPL control runs (`tpl_sameday/`). |
| `notpl_q2_nohup.log`      | Console output for the no-TPL Q2 FE runs. |
| `tpl_q2_nohup.log`        | Console output for the TPL Q2 FE runs. |

## Run Environment

- **Arch (TPL):** `arch-perlmutter-opt-gcc-kokkos-cuda` — cuSPARSE + cuBLAS + cuSOLVER TPLs ON (default)
- **Arch (no-TPL):** `arch-perlmutter-opt-gcc-kokkos-cuda-notpl` — built with `--with-kokkos-kernels-tpl=0`, all three TPL flags `#undef`-ed
- **Nodes:** 2× Perlmutter GPU nodes, 4× A100 40 GB each → 8 GPUs total at n=8
- **Binding:** `srun --gpus-per-node 4 --gpu-bind=none`
- **Key solver flags:**
  ```
  -ksp_type cg
  -pc_type gamg -pc_gamg_agg_nsmooths 1
  -pc_gamg_reuse_interpolation true
  -ksp_norm_type unpreconditioned
  -ksp_rtol 1e-8
  -use_mat_nearnullspace
  -mg_levels_ksp_type chebyshev
  -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05
  -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg
  ```
- **Profiling:** custom `logdefault.c` patch prints async GPU event times without `-log_view_gpu_time` barrier, so plain `-log_view` output reflects real GPU time.

## Key Results Summary

| Metric | n=1 | n=8 | n=27 | n=64 |
|--------|-----|-----|------|------|
| Iters (block==scalar) | 24 | 30 | 35 | 38 |
| KSPSolve block/scalar | 0.83× | **1.04×** | **1.24×** | **1.16×** |
| SpMV block/scalar | 0.90× | **1.12×** | **1.42×** | **1.30×** |
| PtAP block/scalar (nsys) | ≈1.00× | **1.45×** | **1.80×** | **2.27×** |

PtAP win grows monotonically with GPU count; SpMV/KSPSolve peak at n=27. Scalar n=1 advantages due to single-GPU cuSPARSE with no communication overhead.

Capacity: `mpiaijkokkos`+cuSPARSE OOMs at ne=127 n=4 (`cudaErrorMemoryAllocation` in SpGEMM symbolic phase). `mpibaijkokkos` solves the same problem.
