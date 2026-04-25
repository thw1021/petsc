#pragma once

#include <petscvec_kokkos.hpp>
#include <petsc/private/pcimpl.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/kspimpl.h>

#include "Kokkos_Core.hpp"

#if defined(PETSC_HAVE_CUDA)
  #if PETSC_PKG_CUDA_VERSION_GE(10, 0, 0)
    #include <nvtx3/nvToolsExt.h>
  #else
    #include <nvToolsExt.h>
  #endif
#endif

#define PCBJKOKKOS_SHARED_LEVEL 1 // 0 is shared, 1 is global
#define PCBJKOKKOS_VEC_SIZE     16
#define PCBJKOKKOS_TEAM_SIZE    16

#define PCBJKOKKOS_VERBOSE_LEVEL 1

typedef enum {
  BATCH_KSP_BICG_JAC_IDX,
  BATCH_KSP_TFQMR_JAC_IDX,
  BATCH_KSP_BICG_AMG_IDX,
  BATCH_KSP_TFQMR_AMG_IDX,
  BATCH_KSP_GMRES_AMG_IDX,   /* native GMRES (real scalars only) */
  BATCH_KSP_GMRES_JAC_IDX,   /* native GMRES (real scalars only) */
  BATCH_KSP_GMRESKK_JAC_IDX, /* KokkosKernels batched GMRES -- requires PETSC_HAVE_KOKKOS_KERNELS_BATCH; currently not assigned in PCSetUp */
  BATCH_KSP_GMRESKK_AMG_IDX, /* KokkosKernels batched GMRES+AMG -- requires PETSC_HAVE_KOKKOS_KERNELS_BATCH; currently not assigned in PCSetUp */
  BATCH_KSP_PREONLY_IDX,
  NUM_BATCH_TYPES
} KSPIndex;

typedef Kokkos::DefaultExecutionSpace exec_space;
using layout           = Kokkos::LayoutRight;
using IntView          = Kokkos::View<PetscInt **, layout, exec_space>;
using AMatrixValueView = const Kokkos::View<PetscScalar **, layout, exec_space>;
using XYType           = const Kokkos::View<PetscScalar **, layout, exec_space>;

// -----------------------------------------------------------------------
// AMG hierarchy data structures
// -----------------------------------------------------------------------
#define PCBJKOKKOS_MAX_AMG_LEVELS 8

typedef enum {
  BJKOKKOS_SMOOTH_JACOBI,   // standard diagonal: norms[i] = A[i,i]
  BJKOKKOS_SMOOTH_L1_JACOBI // L1 row norm: norms[i] = sum_j |A[i,j]|
} BJKokkosSmootherType;

// One inter-level transfer: fine grid (nrows_fine) -> coarse grid (nrows_coarse)
// P is stored as nrows_fine x nrows_coarse (standard AMG convention):
//   P_ai has nrows_fine+1 entries, P_aj column indices are in [0, nrows_coarse).
// R = P^T is stored as nrows_coarse x nrows_fine.
struct AMGLevel {
  PetscInt nrows_fine, nrows_coarse;
  // P (prolongation) in CSR (nrows_fine x nrows_coarse) -- shared across all blocks on this grid
  PetscInt    *P_ai, *P_aj;
  PetscScalar *P_aa;
  // R = P^T in CSR
  PetscInt    *R_ai, *R_aj;
  PetscScalar *R_aa;
  // A_c sparsity (shared); values are per-block and live on device
  PetscInt  Ac_nnz;
  PetscInt *Ac_ai, *Ac_aj;
  // Device views for structure (shared across blocks)
  Kokkos::View<PetscInt *>    *d_P_ai, *d_P_aj;
  Kokkos::View<PetscScalar *> *d_P_aa;
  Kokkos::View<PetscInt *>    *d_R_ai, *d_R_aj;
  Kokkos::View<PetscScalar *> *d_R_aa;
  Kokkos::View<PetscInt *>    *d_Ac_ai, *d_Ac_aj;
};

// Full AMG hierarchy for one unique grid
struct AMGHierarchy {
  PetscInt nlevels;
  PetscInt grid_size;                         // fine-grid DOF count this hierarchy was built for
  AMGLevel levels[PCBJKOKKOS_MAX_AMG_LEVELS]; // levels[0] = finest->next transition
  // coarsest-level matrix (stored separately; values per-block on device)
  PetscInt                  nrows_coarsest;
  PetscInt                  Ac_coarsest_nnz;
  PetscInt                 *Ac_coarsest_ai, *Ac_coarsest_aj;
  Kokkos::View<PetscInt *> *d_Ac_coarsest_ai, *d_Ac_coarsest_aj;
  // fine-grid local CSR sparsity (local 0-based column indices, shared across blocks)
  PetscInt                  fine_nnz;
  PetscInt                 *fine_ai, *fine_aj;     // host copies
  Kokkos::View<PetscInt *> *d_fine_ai, *d_fine_aj; // device views
  // smoother parameters
  PetscInt             pre_sweeps, post_sweeps, coarse_sweeps;
  PetscReal            strong_threshold;
  BJKokkosSmootherType smoother_type;
  PetscReal            smoother_omega; // damping factor (default 1.0)
};

// -----------------------------------------------------------------------
// AMG V-cycle data structures (shared between bjkokkos.kokkos.cxx and
// bjkokkoskernels.kokkos.cxx)
// -----------------------------------------------------------------------

struct AMGLevelInfo {
  // shared structure (same for all blocks on this grid)
  const PetscInt    *P_ai, *P_aj;
  const PetscScalar *P_aa;
  const PetscInt    *R_ai, *R_aj;
  const PetscScalar *R_aa;
  const PetscInt    *Ac_ai, *Ac_aj; // sparsity of coarse matrix at this level
  PetscInt           nrows_fine, nrows_coarse, Ac_nnz;
  // offsets into per-block work buffer for the COARSE level (lev+1 in V-cycle)
  PetscInt off_Ac_aa; // Ac_aa values (coarse matrix values)
  PetscInt off_l1;    // l1 norms for coarse matrix
  PetscInt off_x;     // x vector at coarse level
  PetscInt off_b;     // b vector at coarse level
  PetscInt off_r;     // residual scratch at coarse level
  PetscInt off_spa;   // sparse accumulator scratch (team_size * nrows_coarse)
  // smoother params
  PetscInt             pre_sweeps, post_sweeps;
  PetscReal            omega;         // damping factor
  BJKokkosSmootherType smoother_type; // smoother type for norm computation
};

// AMGFineInfo: work buffer offsets for the fine-grid level (level 0 in V-cycle)
struct AMGFineInfo {
  PetscInt             nrows;             // fine-grid rows (= levels[0].nrows_fine)
  PetscInt             fine_nnz;          // number of nonzeros in fine-grid local CSR
  const PetscInt      *fine_ai, *fine_aj; // fine-grid local CSR (0-based col indices, device ptrs)
  PetscInt             off_fine_aa;       // offset into per-block work buffer for fine-grid Aa values
  PetscInt             off_l1;            // l1 norms for fine-grid A
  PetscInt             off_x;             // x vector at fine level
  PetscInt             off_b;             // b vector at fine level
  PetscInt             off_r;             // residual scratch at fine level
  PetscInt             pre_sweeps, post_sweeps;
  PetscReal            omega;         // damping factor
  BJKokkosSmootherType smoother_type; // smoother type for norm computation
};

struct AMGCoarsestInfo {
  const PetscInt      *Ac_ai, *Ac_aj;
  PetscInt             nrows, Ac_nnz;
  PetscInt             off_Ac_aa; // shared with levels[nlevels-1].off_Ac_aa
  PetscInt             off_l1;
  PetscInt             off_x;
  PetscInt             off_b;
  PetscInt             off_r;
  PetscInt             coarse_sweeps;
  PetscReal            omega;         // damping factor
  BJKokkosSmootherType smoother_type; // smoother type for norm computation
};

typedef struct {
  Vec                                               vec_diag;
  PetscInt                                          nBlocks; /* total number of blocks */
  PetscInt                                          n;       // cache host version of d_bid_eqOffset_k[nBlocks]
  KSP                                               ksp;     // Used just for options. Should have one for each block
  Kokkos::View<PetscInt *, Kokkos::LayoutRight>    *d_bid_eqOffset_k;
  Kokkos::View<PetscScalar *, Kokkos::LayoutRight> *d_idiag_k;
  Kokkos::View<PetscInt *>                         *d_isrow_k;
  Kokkos::View<PetscInt *>                         *d_isicol_k;
  KSPIndex                                          ksp_type_idx;
  PetscInt                                          nwork;
  PetscInt                                          const_block_size; // used to decide to use shared memory for work vectors
  PetscInt                                         *dm_Nf;            // Number of fields in each DM
  PetscInt                                          num_dms;
  // diagnostics
  PetscBool reason;
  PetscBool monitor;
  PetscInt  batch_target;
  PetscInt  rank_target;
  PetscInt  nsolves_team;
  PetscInt  max_nits;
  // caches
  IntView          *rowOffsets;
  IntView          *colIndices;
  XYType           *batch_b;
  XYType           *batch_x;
  AMatrixValueView *batch_values;
  // AMG
  PetscInt             num_unique_grids; // number of distinct grid sizes
  AMGHierarchy        *amg_hierarchy;    // [num_unique_grids] -- one hierarchy per unique grid
  PetscInt            *block_to_grid;    // [nBlocks] -- maps each block to its grid index
  PetscInt             amg_max_levels;
  PetscInt             amg_min_coarse_size; // stop coarsening when cur_n <= this (default 10)
  PetscReal            amg_strong_threshold;
  PetscInt             amg_pre_sweeps, amg_post_sweeps, amg_coarse_sweeps;
  BJKokkosSmootherType amg_smoother_type;  // default: BJKOKKOS_SMOOTH_L1_JACOBI
  PetscReal            amg_smoother_omega; // default: 1.0
  // Flat device buffer for per-block AMG work
  // Layout per block: [Ac_aa_lev0 | l1_lev0 | x_lev0 | b_lev0 | r_lev0 | spa_lev0 |
  //                    Ac_aa_lev1 | l1_lev1 | x_lev1 | b_lev1 | r_lev1 | spa_lev1 | ...]
  // plus coarsest Ac_aa at the end
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> *d_amg_work;      // [nBlocks * amg_work_stride]
  PetscInt                                                    amg_work_stride; // words per block in d_amg_work
  // Precomputed per-block index array for fine-grid value extraction.
  // d_fine_aa_gidx[blkID * fine_nnz + lk] = position in glb_Aaa[] for local entry lk of block blkID.
  // Eliminates the O(global_nnz_per_row) linear search in PCApply RAP kernel.
  // Allocated once on DIFFERENT_NONZERO_PATTERN; freed in PCDestroy_BJKOKKOS.
  Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace> *d_fine_aa_gidx;      // [nBlocks * fine_nnz]
  PetscInt                                                 fine_aa_gidx_stride; // = fine_nnz (per-block stride)
  // Cached AMG device views -- built once in PCSetUp, reused across PCApply calls.
  // These encode the AMG hierarchy topology which is constant between PCSetUp calls.
  Kokkos::View<AMGFineInfo *, Kokkos::DefaultExecutionSpace>     *d_amg_fine_arr_k;
  Kokkos::View<AMGLevelInfo *, Kokkos::DefaultExecutionSpace>    *d_amg_levels_flat_k;
  Kokkos::View<AMGCoarsestInfo *, Kokkos::DefaultExecutionSpace> *d_amg_coarsest_arr_k;
  Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>        *d_amg_nlevels_k;
  Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>        *d_amg_level_offsets_k;
  Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>        *d_block_to_grid_k;
  // Pre-allocated PCApply work buffers (avoids per-call GPU malloc that fragments memory)
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> *d_work_vecs_k;   // [n * nwork] global Krylov work vectors
  Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> *d_gmres_hwork_k; // [nBlocks * gmres_hwork_per_blk] GMRES Hessenberg
} PC_PCBJKOKKOS;

typedef Kokkos::TeamPolicy<>::member_type team_member;

// -----------------------------------------------------------------------
// PetscLog event handles for PCApply_BJKOKKOS sub-phases
// (registered in PCSetUp_BJKOKKOS; defined in bjkokkos.kokkos.cxx)
// -----------------------------------------------------------------------
PETSC_INTERN PetscLogEvent BJKOKKOS_AMG_RAP;
PETSC_INTERN PetscLogEvent BJKOKKOS_Krylov_Solve;
PETSC_INTERN PetscLogEvent BJKOKKOS_Post_solve;

// -----------------------------------------------------------------------
// AMG sparse kernels (KOKKOS_INLINE_FUNCTION, shared across TUs)
// -----------------------------------------------------------------------

// y = A * x  (local CSR, 0-based column indices, no permutation)
KOKKOS_INLINE_FUNCTION void SpMV_Local(const team_member team, const PetscInt *ai, const PetscInt *aj, const PetscScalar *aa, PetscInt nrows, const PetscScalar *x, PetscScalar *y)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nrows), [=](const int row) {
    PetscScalar sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, ai[row], ai[row + 1]), [=](const int k, PetscScalar &lsum) { lsum += aa[k] * x[aj[k]]; }, sum);
    Kokkos::single(Kokkos::PerThread(team), [=]() { y[row] = sum; });
  });
  team.team_barrier();
}

// l1_norms[i] = sum_j |A[i,j]|  (recomputed when A values change)
KOKKOS_INLINE_FUNCTION void ComputeL1Norms(const team_member team, const PetscInt *ai, const PetscScalar *aa, PetscInt nrows, PetscScalar *l1_norms)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nrows), [=](const int row) {
    PetscReal s = 0.0;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, ai[row], ai[row + 1]), [=](const int k, PetscReal &lsum) { lsum += PetscAbsScalar(aa[k]); }, s);
    Kokkos::single(Kokkos::PerThread(team), [=]() { l1_norms[row] = (PetscScalar)s; });
  });
  team.team_barrier();
}

// Compute smoother norms: either diagonal or L1 row norm
KOKKOS_INLINE_FUNCTION void ComputeSmootherNorms(const team_member team, const PetscInt *ai, const PetscInt *aj, const PetscScalar *aa, PetscInt nrows, PetscScalar *norms, BJKokkosSmootherType type)
{
  if (type == BJKOKKOS_SMOOTH_JACOBI) {
    // norms[i] = A[i,i] (diagonal) -- scan row and break on first match
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nrows), [=](const int row) {
      PetscScalar diag = 0.0;
      for (PetscInt k = ai[row]; k < ai[row + 1]; k++)
        if (aj[k] == row) {
          diag = aa[k];
          break;
        }
      norms[row] = diag;
    });
  } else {
    // L1 row norm: norms[i] = sum_j |A[i,j]|
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nrows), [=](const int row) {
      PetscReal s = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, ai[row], ai[row + 1]), [=](const int k, PetscReal &lsum) { lsum += PetscAbsScalar(aa[k]); }, s);
      Kokkos::single(Kokkos::PerThread(team), [=]() { norms[row] = (PetscScalar)s; });
    });
  }
  team.team_barrier();
}

// nsweeps of damped Jacobi: x += omega * (1/norms[i]) * (b - A*x)
// residual[] is a scratch vector of length nrows
KOKKOS_INLINE_FUNCTION void L1JacobiSmooth(const team_member team, const PetscInt *ai, const PetscInt *aj, const PetscScalar *aa, const PetscScalar *l1_norms, PetscInt nrows, const PetscScalar *b, PetscScalar *x, PetscScalar *residual, PetscInt nsweeps, PetscReal omega)
{
  for (PetscInt sweep = 0; sweep < nsweeps; sweep++) {
    // residual = b - A*x
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nrows), [=](const int row) {
      PetscScalar sum = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, ai[row], ai[row + 1]), [=](const int k, PetscScalar &lsum) { lsum += aa[k] * x[aj[k]]; }, sum);
      Kokkos::single(Kokkos::PerThread(team), [=]() { residual[row] = b[row] - sum; });
    });
    team.team_barrier();
    // x += omega * (1/norms) * residual  (skip rows with zero norm to avoid NaN)
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nrows), [=](int i) {
      if (l1_norms[i] != 0.0) x[i] += omega * residual[i] / l1_norms[i];
    });
    team.team_barrier();
  }
}

// Fill Ac values given known sparsity: Ac = R * A * P  (Galerkin triple product)
// Uses a sparse accumulator: each thread gets a private dense array spa[0..R_nrows)
// indexed by coarse column.  For each coarse row I:
//   1. Zero spa[Ac_aj[jj]] for each jj in Ac[I,:]
//   2. Accumulate: for (i in R[I,:]) for (j in A[i,:]) for (J in P[j,:]):
//        spa[J] += R[I,i] * A[i,j] * P[j,J]
//   3. Scatter: Ac_aa[jj] = spa[Ac_aj[jj]]
// spa[] must point to team_size * R_nrows words of per-block scratch (off_spa).
KOKKOS_INLINE_FUNCTION void NumericRAP(const team_member team, const PetscInt *A_ai, const PetscInt *A_aj, const PetscScalar *A_aa, PetscInt A_nrows, const PetscInt *P_ai, const PetscInt *P_aj, const PetscScalar *P_aa, PetscInt P_nrows, const PetscInt *R_ai, const PetscInt *R_aj, const PetscScalar *R_aa, PetscInt R_nrows, const PetscInt *Ac_ai, const PetscInt *Ac_aj, PetscScalar *Ac_aa, PetscScalar *spa)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, R_nrows), [=](const int I) {
    PetscScalar *myspa = spa + team.team_rank() * R_nrows; // spa stride = R_nrows == ncoarse (indexed by coarse column)
    // 1. Zero spa entries for columns in Ac[I,:]
    for (PetscInt jj = Ac_ai[I]; jj < Ac_ai[I + 1]; jj++) myspa[Ac_aj[jj]] = 0.0;
    // 2. Accumulate R[I,i] * A[i,j] * P[j,J] into spa[J]
    for (PetscInt ri = R_ai[I]; ri < R_ai[I + 1]; ri++) {
      const PetscInt    i    = R_aj[ri];
      const PetscScalar Rval = R_aa[ri];
      for (PetscInt ai = A_ai[i]; ai < A_ai[i + 1]; ai++) {
        const PetscInt    j     = A_aj[ai];
        const PetscScalar RAval = Rval * A_aa[ai];
        for (PetscInt pi = P_ai[j]; pi < P_ai[j + 1]; pi++) myspa[P_aj[pi]] += RAval * P_aa[pi];
      }
    }
    // 3. Scatter spa into Ac_aa
    for (PetscInt jj = Ac_ai[I]; jj < Ac_ai[I + 1]; jj++) Ac_aa[jj] = myspa[Ac_aj[jj]];
  });
  team.team_barrier();
}

// V-cycle: operates on the fine-grid residual b_fine[0..nrows_fine) and produces correction x_fine.
KOKKOS_INLINE_FUNCTION void AMGVCycle(const team_member team, const AMGFineInfo &fine, const AMGLevelInfo *levels, PetscInt nlevels, const AMGCoarsestInfo &coarsest, PetscScalar *work, const PetscScalar *b_fine, PetscScalar *x_fine)
{
  PetscInt           nrows_fine = fine.nrows;
  const PetscInt    *f_ai       = fine.fine_ai;
  const PetscInt    *f_aj       = fine.fine_aj;
  const PetscScalar *f_aa       = work + fine.off_fine_aa;

  // ---- Down-leg ----
  // Level 0 (fine grid): smooth using fine-grid local CSR
  {
    PetscScalar *l1 = work + fine.off_l1;
    PetscScalar *xf = work + fine.off_x;
    PetscScalar *bf = work + fine.off_b;
    PetscScalar *rf = work + fine.off_r;

    // Copy b_fine into bf, zero xf
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nrows_fine), [=](int i) {
      bf[i] = b_fine[i];
      xf[i] = 0.0;
    });
    team.team_barrier();

    // Pre-smooth on fine grid using local CSR (0-based column indices)
    L1JacobiSmooth(team, f_ai, f_aj, f_aa, l1, nrows_fine, bf, xf, rf, fine.pre_sweeps, fine.omega);

    // Compute residual: rf = bf - A*xf
    SpMV_Local(team, f_ai, f_aj, f_aa, nrows_fine, xf, rf);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nrows_fine), [=](int i) { rf[i] = bf[i] - rf[i]; });
    team.team_barrier();

    // Restrict residual to level 1 (or coarsest if nlevels==0)
    if (nlevels > 0) {
      const AMGLevelInfo &L0 = levels[0];
      PetscScalar        *b1 = work + L0.off_b;
      PetscScalar        *x1 = work + L0.off_x;
      SpMV_Local(team, L0.R_ai, L0.R_aj, L0.R_aa, L0.nrows_coarse, rf, b1);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, L0.nrows_coarse), [=](int i) { x1[i] = 0.0; });
      team.team_barrier();
    } else {
      // Only one level: restrict to coarsest
      PetscScalar *bc = work + coarsest.off_b;
      PetscScalar *xc = work + coarsest.off_x;
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nrows_fine), [=](int i) {
        bc[i] = rf[i];
        xc[i] = 0.0;
      });
      team.team_barrier();
    }
  }

  // Levels 1..nlevels-1: smooth using Ac[lev-1], restrict to next level
  // Levels 0..nlevels-1: pre-smooth, compute residual, restrict to next level.
  // The last iteration (lev == nlevels-1) restricts to the coarsest level via
  // direct copy (coarsest.nrows == levels[nlevels-1].nrows_coarse).
  for (PetscInt lev = 0; lev < nlevels; lev++) {
    const AMGLevelInfo &L    = levels[lev];
    PetscScalar        *Ac   = work + L.off_Ac_aa;
    PetscScalar        *l1   = work + L.off_l1;
    PetscScalar        *xlev = work + L.off_x;
    PetscScalar        *blev = work + L.off_b;
    PetscScalar        *rlev = work + L.off_r;

    // Pre-smooth using Ac[lev] (coarse matrix at this level)
    L1JacobiSmooth(team, L.Ac_ai, L.Ac_aj, Ac, l1, L.nrows_coarse, blev, xlev, rlev, L.pre_sweeps, L.omega);

    // Compute residual: rlev = blev - Ac*xlev
    SpMV_Local(team, L.Ac_ai, L.Ac_aj, Ac, L.nrows_coarse, xlev, rlev);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, L.nrows_coarse), [=](int i) { rlev[i] = blev[i] - rlev[i]; });
    team.team_barrier();

    if (lev < nlevels - 1) {
      // Restrict residual to next inter-level
      const AMGLevelInfo &Lnext  = levels[lev + 1];
      PetscScalar        *b_next = work + Lnext.off_b;
      PetscScalar        *x_next = work + Lnext.off_x;
      SpMV_Local(team, Lnext.R_ai, Lnext.R_aj, Lnext.R_aa, Lnext.nrows_coarse, rlev, b_next);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, Lnext.nrows_coarse), [=](int i) { x_next[i] = 0.0; });
      team.team_barrier();
    } else {
      // Restrict residual to coarsest level (direct copy: coarsest.nrows == L.nrows_coarse)
      PetscScalar *bc = work + coarsest.off_b;
      PetscScalar *xc = work + coarsest.off_x;
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, coarsest.nrows), [=](int i) {
        bc[i] = rlev[i];
        xc[i] = 0.0;
      });
      team.team_barrier();
    }
  }

  // ---- Coarsest solve ----
  {
    PetscScalar *Ac = work + coarsest.off_Ac_aa;
    PetscScalar *l1 = work + coarsest.off_l1;
    PetscScalar *xc = work + coarsest.off_x;
    PetscScalar *bc = work + coarsest.off_b;
    PetscScalar *rc = work + coarsest.off_r;
    L1JacobiSmooth(team, coarsest.Ac_ai, coarsest.Ac_aj, Ac, l1, coarsest.nrows, bc, xc, rc, coarsest.coarse_sweeps, coarsest.omega);
  }

  // ---- Up-leg ----
  if (nlevels > 0) {
    const AMGLevelInfo &Llast = levels[nlevels - 1];
    PetscScalar        *Ac    = work + Llast.off_Ac_aa;
    PetscScalar        *l1    = work + Llast.off_l1;
    PetscScalar        *xlev  = work + Llast.off_x;
    PetscScalar        *blev  = work + Llast.off_b;
    PetscScalar        *rlev  = work + Llast.off_r;
    const PetscScalar  *xc    = work + coarsest.off_x;
    // Add coarsest correction to pre-smoothed approximation at levels[nlevels-1]
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, Llast.nrows_coarse), [=](int i) { xlev[i] += xc[i]; });
    team.team_barrier();
    // Post-smooth on levels[nlevels-1] (was missing before)
    L1JacobiSmooth(team, Llast.Ac_ai, Llast.Ac_aj, Ac, l1, Llast.nrows_coarse, blev, xlev, rlev, Llast.post_sweeps, Llast.omega);
  }

  // Levels nlevels-2..0: interpolate from next coarser, post-smooth
  for (PetscInt lev = nlevels - 2; lev >= 0; lev--) {
    const AMGLevelInfo &L      = levels[lev];
    const AMGLevelInfo &Lnext  = levels[lev + 1];
    PetscScalar        *Ac     = work + L.off_Ac_aa;
    PetscScalar        *l1     = work + L.off_l1;
    PetscScalar        *xlev   = work + L.off_x;
    PetscScalar        *blev   = work + L.off_b;
    PetscScalar        *rlev   = work + L.off_r;
    const PetscScalar  *x_next = work + Lnext.off_x;

    SpMV_Local(team, Lnext.P_ai, Lnext.P_aj, Lnext.P_aa, L.nrows_coarse, x_next, rlev);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, L.nrows_coarse), [=](int i) { xlev[i] += rlev[i]; });
    team.team_barrier();
    L1JacobiSmooth(team, L.Ac_ai, L.Ac_aj, Ac, l1, L.nrows_coarse, blev, xlev, rlev, L.post_sweeps, L.omega);
  }

  // Fine-grid up-leg: interpolate from level 0 coarse, post-smooth on fine grid
  {
    PetscScalar *l1 = work + fine.off_l1;
    PetscScalar *xf = work + fine.off_x;
    PetscScalar *bf = work + fine.off_b;
    PetscScalar *rf = work + fine.off_r;

    if (nlevels > 0) {
      const PetscScalar *x1 = work + levels[0].off_x;
      SpMV_Local(team, levels[0].P_ai, levels[0].P_aj, levels[0].P_aa, levels[0].nrows_fine, x1, rf);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nrows_fine), [=](int i) { xf[i] += rf[i]; });
      team.team_barrier();
    } else {
      const PetscScalar *xc = work + coarsest.off_x;
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nrows_fine), [=](int i) { xf[i] += xc[i]; });
      team.team_barrier();
    }
    L1JacobiSmooth(team, f_ai, f_aj, f_aa, l1, nrows_fine, bf, xf, rf, fine.post_sweeps, fine.omega);
  }

  // Copy fine-level x back to x_fine
  {
    const PetscScalar *xf = work + fine.off_x;
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nrows_fine), [=](int i) { x_fine[i] = xf[i]; });
    team.team_barrier();
  }
}

#if defined(PETSC_HAVE_KOKKOS_KERNELS_BATCH)
PETSC_INTERN PetscErrorCode PCApply_BJKOKKOSKERNELS(PC, const PetscScalar *, PetscScalar *, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt, MatInfo, const PetscInt, PCFailedReason *);
PETSC_INTERN PetscErrorCode PCApply_BJKOKKOSKERNELS_AMG(PC, const PetscScalar *, PetscScalar *, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt, MatInfo, const PetscInt, PCFailedReason *, const AMGFineInfo *, const AMGLevelInfo *, const PetscInt *, const PetscInt *, const AMGCoarsestInfo *, const PetscInt *, PetscScalar *, PetscInt);
#endif

// AMG hierarchy setup -- implemented in bjkokkos_amg.kokkos.cxx
PETSC_INTERN PetscErrorCode PCBJKOKKOSSetupAMG(PC, Mat);
PETSC_INTERN PetscErrorCode PCBJKOKKOSDestroyAMG(PC_PCBJKOKKOS *);
