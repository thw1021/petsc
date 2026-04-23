/*
  bjkokkos_amg.kokkos.cxx -- Host-side classical AMG hierarchy setup for PCBJKOKKOS.

  Builds one AMGHierarchy per unique grid (block size).  All blocks on the same
  grid share the same P, R, and A_c sparsity; only A_c values differ per block
  and are recomputed on the device each Newton step via NumericRAP.
*/

#include <petsc/private/pcbjkokkosimpl.h>
#include <petsc/private/pcimpl.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>
#include <petscdmcomposite.h>

#include <algorithm>
#include <vector>
#include <unordered_set>

/* -----------------------------------------------------------------------
   Internal helper: free one AMGLevel's host allocations
   ----------------------------------------------------------------------- */
static PetscErrorCode AMGLevelFreeHost(AMGLevel *lev)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(lev->P_ai));
  PetscCall(PetscFree(lev->P_aj));
  PetscCall(PetscFree(lev->P_aa));
  PetscCall(PetscFree(lev->R_ai));
  PetscCall(PetscFree(lev->R_aj));
  PetscCall(PetscFree(lev->R_aa));
  PetscCall(PetscFree(lev->Ac_ai));
  PetscCall(PetscFree(lev->Ac_aj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AMGLevelFreeDevice(AMGLevel *lev)
{
  PetscFunctionBegin;
  delete lev->d_P_ai;
  lev->d_P_ai = nullptr;
  delete lev->d_P_aj;
  lev->d_P_aj = nullptr;
  delete lev->d_P_aa;
  lev->d_P_aa = nullptr;
  delete lev->d_R_ai;
  lev->d_R_ai = nullptr;
  delete lev->d_R_aj;
  lev->d_R_aj = nullptr;
  delete lev->d_R_aa;
  lev->d_R_aa = nullptr;
  delete lev->d_Ac_ai;
  lev->d_Ac_ai = nullptr;
  delete lev->d_Ac_aj;
  lev->d_Ac_aj = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Build strength graph (absolute-value criterion)
   S[i] = set of j such that |A[i,j]| >= threshold * max_{k!=i}(|A[i,k]|)
   Uses absolute values so that both positive and negative off-diagonal
   connections are detected.  This is essential for non-M-matrix operators
   such as FEM stiffness matrices from the Landau collision operator.
   Returns: s_ai[n+1], s_aj[nnz_s]  (CSR, 0-based)
   ----------------------------------------------------------------------- */
static PetscErrorCode BuildStrengthGraph(const PetscInt *ai, const PetscInt *aj, const PetscScalar *aa, PetscInt n, PetscReal threshold, PetscInt **s_ai_out, PetscInt **s_aj_out)
{
  PetscInt *s_ai, *s_aj;
  PetscInt  nnz_s = 0;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(n + 1, &s_ai));

  /* first pass: count */
  s_ai[0] = 0;
  for (PetscInt i = 0; i < n; i++) {
    PetscReal max_abs = 0.0;
    for (PetscInt k = ai[i]; k < ai[i + 1]; k++) {
      if (aj[k] != i) {
        PetscReal v = PetscAbsScalar(aa[k]);
        if (v > max_abs) max_abs = v;
      }
    }
    PetscInt cnt = 0;
    if (max_abs > 0.0) {
      PetscReal thr = threshold * max_abs;
      for (PetscInt k = ai[i]; k < ai[i + 1]; k++) {
        if (aj[k] != i && PetscAbsScalar(aa[k]) >= thr) cnt++;
      }
    }
    s_ai[i + 1] = s_ai[i] + cnt;
    nnz_s += cnt;
  }

  PetscCall(PetscMalloc1(nnz_s + 1, &s_aj));

  /* second pass: fill */
  for (PetscInt i = 0; i < n; i++) {
    PetscReal max_abs = 0.0;
    for (PetscInt k = ai[i]; k < ai[i + 1]; k++) {
      if (aj[k] != i) {
        PetscReal v = PetscAbsScalar(aa[k]);
        if (v > max_abs) max_abs = v;
      }
    }
    PetscInt pos = s_ai[i];
    if (max_abs > 0.0) {
      PetscReal thr = threshold * max_abs;
      for (PetscInt k = ai[i]; k < ai[i + 1]; k++) {
        if (aj[k] != i && PetscAbsScalar(aa[k]) >= thr) s_aj[pos++] = aj[k];
      }
    }
  }

  *s_ai_out = s_ai;
  *s_aj_out = s_aj;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   C/F splitting -- greedy Ruge-Stuben
   CF_marker[i] = 1 (C-point) or -1 (F-point)

   Complexity: O(n^2) due to linear scan for max-lambda node on each
   iteration.  Acceptable for the small block sizes typical of batched
   Landau collision operators (~30-300 DOFs).
   ----------------------------------------------------------------------- */
enum {
  CF_UNSET  = 0,
  CF_CPOINT = 1,
  CF_FPOINT = -1
};

static PetscErrorCode CFSplitting(const PetscInt *s_ai, const PetscInt *s_aj, PetscInt n, PetscInt **CF_out, PetscInt *nC_out)
{
  PetscInt *CF, *lambda;
  PetscInt  nC = 0;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(n, &CF));
  PetscCall(PetscMalloc1(n, &lambda));

  /* Build transpose of strength graph: st_ai/st_aj
     st[j] = set of i such that i strongly depends on j (i.e., j strongly influences i) */
  PetscInt *st_ai, *st_aj;
  PetscCall(PetscCalloc1(n + 1, &st_ai));
  /* count in-degree */
  for (PetscInt i = 0; i < n; i++)
    for (PetscInt k = s_ai[i]; k < s_ai[i + 1]; k++) st_ai[s_aj[k] + 1]++;
  /* prefix sum */
  for (PetscInt i = 0; i < n; i++) st_ai[i + 1] += st_ai[i];
  PetscCall(PetscMalloc1(st_ai[n] + 1, &st_aj));
  /* fill */
  PetscInt *cnt;
  PetscCall(PetscCalloc1(n, &cnt));
  for (PetscInt i = 0; i < n; i++)
    for (PetscInt k = s_ai[i]; k < s_ai[i + 1]; k++) {
      PetscInt j               = s_aj[k];
      st_aj[st_ai[j] + cnt[j]] = i;
      cnt[j]++;
    }
  PetscCall(PetscFree(cnt));

  /* lambda[i] = |S^T_i| = number of undecided nodes that i strongly influences */
  for (PetscInt i = 0; i < n; i++) lambda[i] = st_ai[i + 1] - st_ai[i];

  /* greedy: pick node with max lambda, mark C, mark nodes that depend on it as F */
  PetscBool *decided;
  PetscCall(PetscCalloc1(n, &decided));
  for (;;) {
    /* find undecided node with max lambda */
    PetscInt best     = -1;
    PetscInt best_lam = -1;
    for (PetscInt i = 0; i < n; i++) {
      if (!decided[i] && lambda[i] > best_lam) {
        best_lam = lambda[i];
        best     = i;
      }
    }
    if (best < 0 || best_lam < 0) break; /* all decided or isolated */

    /* mark best as C */
    CF[best]      = CF_CPOINT;
    decided[best] = PETSC_TRUE;
    nC++;

    /* mark nodes that strongly depend on best as F (if not yet decided) */
    for (PetscInt k = st_ai[best]; k < st_ai[best + 1]; k++) {
      PetscInt j = st_aj[k]; /* j strongly depends on best */
      if (!decided[j]) {
        CF[j]      = CF_FPOINT;
        decided[j] = PETSC_TRUE;
        /* increment lambda for undecided nodes that j strongly influences */
        for (PetscInt m = st_ai[j]; m < st_ai[j + 1]; m++) {
          PetscInt nb = st_aj[m];
          if (!decided[nb]) lambda[nb]++;
        }
      }
    }
  }

  /* any remaining undecided nodes become C-points */
  for (PetscInt i = 0; i < n; i++) {
    if (!decided[i]) {
      CF[i] = CF_CPOINT;
      nC++;
    }
  }

  PetscCall(PetscFree(decided));
  PetscCall(PetscFree(st_ai));
  PetscCall(PetscFree(st_aj));
  PetscCall(PetscFree(lambda));
  *CF_out = CF;
  *nC_out = nC;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Classical direct interpolation -> P in CSR
   C-points: identity row (P[i, coarse_idx[i]] = 1)
   F-points: weighted average of strong C-neighbors
   ----------------------------------------------------------------------- */
static PetscErrorCode BuildProlongation(const PetscInt *ai, const PetscInt *aj, const PetscScalar *aa, PetscInt n, const PetscInt *s_ai, const PetscInt *s_aj, const PetscInt *CF, PetscInt nC, PetscInt **P_ai_out, PetscInt **P_aj_out, PetscScalar **P_aa_out)
{
  PetscInt    *coarse_idx; /* fine->coarse index for C-points, -1 for F */
  PetscInt    *P_ai, *P_aj;
  PetscScalar *P_aa;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(n, &coarse_idx));
  for (PetscInt i = 0, c = 0; i < n; i++) coarse_idx[i] = (CF[i] == CF_CPOINT) ? c++ : -1;

  /* count P nnz */
  PetscCall(PetscMalloc1(n + 1, &P_ai));
  P_ai[0] = 0;
  for (PetscInt i = 0; i < n; i++) {
    if (CF[i] == CF_CPOINT) {
      P_ai[i + 1] = P_ai[i] + 1;
    } else {
      /* count strong C-neighbors */
      PetscInt cnt = 0;
      for (PetscInt k = s_ai[i]; k < s_ai[i + 1]; k++) {
        if (CF[s_aj[k]] == CF_CPOINT) cnt++;
      }
      if (cnt == 0) cnt = 1; /* fallback: diagonal injection */
      P_ai[i + 1] = P_ai[i] + cnt;
    }
  }

  PetscInt nnz_P = P_ai[n];
  PetscCall(PetscMalloc1(nnz_P + 1, &P_aj));
  PetscCall(PetscMalloc1(nnz_P + 1, &P_aa));

  for (PetscInt i = 0; i < n; i++) {
    PetscInt pos = P_ai[i];
    if (CF[i] == CF_CPOINT) {
      P_aj[pos] = coarse_idx[i];
      P_aa[pos] = 1.0;
    } else {
      /* Standard direct interpolation (Ruge-Stuben / PETSc GAMG classical direct).
         Handles both positive and negative off-diagonals.
         alpha = -a_neg / g_neg  redistributes weak negative connections to strong C-point negatives
         beta  = -a_pos / g_pos  redistributes weak positive connections to strong C-point positives
         P[i,j] = a_ij * alpha / diag  (for a_ij < 0)
         P[i,j] = a_ij * beta  / diag  (for a_ij > 0)  */
      PetscScalar diag = 0.0, a_neg = 0.0, a_pos = 0.0, g_neg = 0.0, g_pos = 0.0;
      /* Accumulate diagonal and total off-diagonal sums */
      for (PetscInt k = ai[i]; k < ai[i + 1]; k++) {
        if (aj[k] == i) diag = aa[k];
        else if (PetscRealPart(aa[k]) < 0.0) a_neg += aa[k];
        else a_pos += aa[k];
      }
      /* Accumulate strong C-point sums */
      for (PetscInt k = s_ai[i]; k < s_ai[i + 1]; k++) {
        if (CF[s_aj[k]] == CF_CPOINT) {
          PetscScalar aij = 0.0;
          for (PetscInt m = ai[i]; m < ai[i + 1]; m++) {
            if (aj[m] == s_aj[k]) {
              aij = aa[m];
              break;
            }
          }
          if (PetscRealPart(aij) < 0.0) g_neg += aij;
          else g_pos += aij;
        }
      }
      PetscScalar alpha = (g_neg != 0.0) ? -a_neg / g_neg : 0.0;
      PetscScalar beta;
      if (g_pos == 0.0) {
        diag += a_pos; /* absorb positive off-diag into diagonal */
        beta = 0.0;
      } else beta = -a_pos / g_pos;
      PetscScalar invdiag = (diag != 0.0) ? 1.0 / diag : 0.0;
      /* Fill prolongation weights */
      PetscInt cnt = 0;
      for (PetscInt k = s_ai[i]; k < s_ai[i + 1]; k++) {
        PetscInt j = s_aj[k];
        if (CF[j] == CF_CPOINT) {
          PetscScalar aij = 0.0;
          for (PetscInt m = ai[i]; m < ai[i + 1]; m++) {
            if (aj[m] == j) {
              aij = aa[m];
              break;
            }
          }
          P_aj[pos + cnt] = coarse_idx[j];
          if (PetscRealPart(aij) < 0.0) P_aa[pos + cnt] = aij * alpha * invdiag;
          else P_aa[pos + cnt] = aij * beta * invdiag;
          cnt++;
        }
      }
      if (cnt == 0) {
        /* isolated F-point: inject to nearest C-neighbor in the matrix connectivity */
        PetscBool found_c = PETSC_FALSE;
        for (PetscInt k = ai[i]; k < ai[i + 1]; k++) {
          if (aj[k] != i && CF[aj[k]] == CF_CPOINT) {
            P_aj[pos] = coarse_idx[aj[k]];
            P_aa[pos] = 1.0;
            found_c   = PETSC_TRUE;
            break;
          }
        }
        if (!found_c) {
          /* no C-neighbor in matrix row: fall back to globally first C-point */
          for (PetscInt c = 0; c < n; c++) {
            if (CF[c] == CF_CPOINT) {
              P_aj[pos] = coarse_idx[c];
              P_aa[pos] = 1.0;
              found_c   = PETSC_TRUE;
              break;
            }
          }
        }
        PetscCheck(found_c, PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMG: isolated F-point row %" PetscInt_FMT " with no C-points anywhere (nC=%" PetscInt_FMT ")", i, nC);
        PetscCall(PetscInfo(NULL, "AMG: isolated F-point row %" PetscInt_FMT " with no strong C-neighbors; using fallback injection\n", i));
      }
    }
  }

  PetscCall(PetscFree(coarse_idx));
  *P_ai_out = P_ai;
  *P_aj_out = P_aj;
  *P_aa_out = P_aa;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   R = P^T in CSR
   P is (nfine x ncoarse), R is (ncoarse x nfine)
   ----------------------------------------------------------------------- */
static PetscErrorCode BuildRestriction(const PetscInt *P_ai, const PetscInt *P_aj, const PetscScalar *P_aa, PetscInt nfine, PetscInt ncoarse, PetscInt **R_ai_out, PetscInt **R_aj_out, PetscScalar **R_aa_out)
{
  PetscInt    *R_ai, *R_aj, *cnt_tmp;
  PetscScalar *R_aa;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(ncoarse + 1, &R_ai));
  PetscCall(PetscMalloc1(ncoarse, &cnt_tmp));

  /* count entries per coarse row */
  PetscInt nnz_P = P_ai[nfine];
  for (PetscInt k = 0; k < nnz_P; k++) R_ai[P_aj[k] + 1]++;
  /* prefix sum */
  for (PetscInt i = 0; i < ncoarse; i++) R_ai[i + 1] += R_ai[i];
  for (PetscInt i = 0; i < ncoarse; i++) cnt_tmp[i] = 0;

  PetscCall(PetscMalloc1(nnz_P + 1, &R_aj));
  PetscCall(PetscMalloc1(nnz_P + 1, &R_aa));

  for (PetscInt i = 0; i < nfine; i++) {
    for (PetscInt k = P_ai[i]; k < P_ai[i + 1]; k++) {
      PetscInt J   = P_aj[k];
      PetscInt pos = R_ai[J] + cnt_tmp[J];
      R_aj[pos]    = i;
      R_aa[pos]    = P_aa[k];
      cnt_tmp[J]++;
    }
  }

  PetscCall(PetscFree(cnt_tmp));
  *R_ai_out = R_ai;
  *R_aj_out = R_aj;
  *R_aa_out = R_aa;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Symbolic RAP -- compute A_c sparsity pattern
   A_c = R * A * P  (ncoarse x ncoarse)
   ----------------------------------------------------------------------- */
static PetscErrorCode SymbolicRAP(const PetscInt *A_ai, const PetscInt *A_aj, PetscInt A_nrows, const PetscInt *P_ai, const PetscInt *P_aj, PetscInt P_nrows, const PetscInt *R_ai, const PetscInt *R_aj, PetscInt R_nrows, PetscInt **Ac_ai_out, PetscInt **Ac_aj_out, PetscInt *Ac_nnz_out)
{
  PetscInt *Ac_ai, *Ac_aj;
  PetscInt  ncoarse = R_nrows;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(ncoarse + 1, &Ac_ai));
  Ac_ai[0] = 0;

  /* use a boolean marker array to avoid duplicates */
  std::vector<bool>     marker(ncoarse, false);
  std::vector<PetscInt> row_cols;
  std::vector<PetscInt> all_cols;

  for (PetscInt I = 0; I < ncoarse; I++) {
    row_cols.clear();
    for (PetscInt ri = R_ai[I]; ri < R_ai[I + 1]; ri++) {
      PetscInt i = R_aj[ri];
      for (PetscInt ai = A_ai[i]; ai < A_ai[i + 1]; ai++) {
        PetscInt j = A_aj[ai];
        for (PetscInt pi = P_ai[j]; pi < P_ai[j + 1]; pi++) {
          PetscInt J = P_aj[pi];
          if (!marker[J]) {
            marker[J] = true;
            row_cols.push_back(J);
          }
        }
      }
    }
    std::sort(row_cols.begin(), row_cols.end());
    for (PetscInt J : row_cols) {
      marker[J] = false;
      all_cols.push_back(J);
    }
    Ac_ai[I + 1] = (PetscInt)all_cols.size();
  }

  PetscInt nnz = (PetscInt)all_cols.size();
  PetscCall(PetscMalloc1(nnz + 1, &Ac_aj));
  for (PetscInt k = 0; k < nnz; k++) Ac_aj[k] = all_cols[k];

  *Ac_ai_out  = Ac_ai;
  *Ac_aj_out  = Ac_aj;
  *Ac_nnz_out = nnz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Numeric RAP -- host sequential version
   Fills Ac_aa given known sparsity.  Same algorithm as the device kernel
   NumericRAP() in pcbjkokkosimpl.h but runs sequentially on the host.
   ----------------------------------------------------------------------- */
static PetscErrorCode NumericRAP_Host(const PetscInt *A_ai, const PetscInt *A_aj, const PetscScalar *A_aa, PetscInt A_nrows, const PetscInt *P_ai, const PetscInt *P_aj, const PetscScalar *P_aa, const PetscInt *R_ai, const PetscInt *R_aj, const PetscScalar *R_aa, PetscInt R_nrows, const PetscInt *Ac_ai, const PetscInt *Ac_aj, PetscScalar **Ac_aa_out)
{
  PetscInt     ncoarse = R_nrows;
  PetscInt     nnz     = Ac_ai[ncoarse];
  PetscScalar *Ac_aa, *work;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(nnz + 1, &Ac_aa));
  /* work array indexed by coarse column */
  PetscCall(PetscCalloc1(ncoarse, &work));

  for (PetscInt I = 0; I < ncoarse; I++) {
    /* zero work for this row */
    for (PetscInt jj = Ac_ai[I]; jj < Ac_ai[I + 1]; jj++) work[Ac_aj[jj]] = 0.0;
    /* accumulate R[I,i] * A[i,j] * P[j,J] */
    for (PetscInt ri = R_ai[I]; ri < R_ai[I + 1]; ri++) {
      PetscInt    i    = R_aj[ri];
      PetscScalar Rval = R_aa[ri];
      for (PetscInt ai = A_ai[i]; ai < A_ai[i + 1]; ai++) {
        PetscInt    j     = A_aj[ai];
        PetscScalar RAval = Rval * A_aa[ai];
        for (PetscInt pi = P_ai[j]; pi < P_ai[j + 1]; pi++) work[P_aj[pi]] += RAval * P_aa[pi];
      }
    }
    /* scatter into Ac_aa */
    for (PetscInt jj = Ac_ai[I]; jj < Ac_ai[I + 1]; jj++) {
      Ac_aa[jj]       = work[Ac_aj[jj]];
      work[Ac_aj[jj]] = 0.0;
    }
  }

  PetscCall(PetscFree(work));
  *Ac_aa_out = Ac_aa;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Copy one AMGLevel's structure to device Kokkos Views
   ----------------------------------------------------------------------- */
static PetscErrorCode AMGLevelCopyToDevice(AMGLevel *lev)
{
  using IntView1D    = Kokkos::View<PetscInt *>;
  using ScalarView1D = Kokkos::View<PetscScalar *>;
  using HostIntView  = Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using HostScView   = Kokkos::View<PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  PetscFunctionBegin;
  PetscInt nfine   = lev->nrows_fine;
  PetscInt ncoarse = lev->nrows_coarse;
  PetscInt nnz_P   = lev->P_ai[nfine];
  PetscInt nnz_Ac  = lev->Ac_nnz;

  /* P */
  HostIntView h_P_ai(lev->P_ai, nfine + 1);
  HostIntView h_P_aj(lev->P_aj, nnz_P);
  HostScView  h_P_aa(lev->P_aa, nnz_P);
  lev->d_P_ai = new IntView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_P_ai));
  lev->d_P_aj = new IntView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_P_aj));
  lev->d_P_aa = new ScalarView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_P_aa));
  Kokkos::deep_copy(*lev->d_P_ai, h_P_ai);
  Kokkos::deep_copy(*lev->d_P_aj, h_P_aj);
  Kokkos::deep_copy(*lev->d_P_aa, h_P_aa);

  /* R */
  HostIntView h_R_ai(lev->R_ai, ncoarse + 1);
  HostIntView h_R_aj(lev->R_aj, nnz_P);
  HostScView  h_R_aa(lev->R_aa, nnz_P);
  lev->d_R_ai = new IntView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_R_ai));
  lev->d_R_aj = new IntView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_R_aj));
  lev->d_R_aa = new ScalarView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_R_aa));
  Kokkos::deep_copy(*lev->d_R_ai, h_R_ai);
  Kokkos::deep_copy(*lev->d_R_aj, h_R_aj);
  Kokkos::deep_copy(*lev->d_R_aa, h_R_aa);

  /* Ac sparsity */
  HostIntView h_Ac_ai(lev->Ac_ai, ncoarse + 1);
  HostIntView h_Ac_aj(lev->Ac_aj, nnz_Ac);
  lev->d_Ac_ai = new IntView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_Ac_ai));
  lev->d_Ac_aj = new IntView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_Ac_aj));
  Kokkos::deep_copy(*lev->d_Ac_ai, h_Ac_ai);
  Kokkos::deep_copy(*lev->d_Ac_aj, h_Ac_aj);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Steps 3.2-3.10: Build one complete AMG hierarchy for a matrix given
   in local CSR (ai, aj, aa, nrows).
   ----------------------------------------------------------------------- */
static PetscErrorCode BuildAMGHierarchy(const PetscInt *ai, const PetscInt *aj, const PetscScalar *aa, PetscInt nrows, PetscReal strong_threshold, PetscInt max_levels, PetscInt min_coarse_size, PetscInt pre_sweeps, PetscInt post_sweeps, PetscInt coarse_sweeps, BJKokkosSmootherType smoother_type, PetscScalar smoother_omega, AMGHierarchy *hier)
{
  /* working copies of the current-level matrix (host) */
  PetscInt    *cur_ai = (PetscInt *)ai, *cur_aj = (PetscInt *)aj;
  PetscScalar *cur_aa   = (PetscScalar *)aa;
  PetscInt     cur_n    = nrows;
  PetscBool    owns_cur = PETSC_FALSE; /* whether we own cur_ai/aj/aa */

  PetscFunctionBegin;
  hier->nlevels          = 0;
  hier->grid_size        = nrows;
  hier->pre_sweeps       = pre_sweeps;
  hier->post_sweeps      = post_sweeps;
  hier->coarse_sweeps    = coarse_sweeps;
  hier->strong_threshold = strong_threshold;
  hier->smoother_type    = smoother_type;
  hier->smoother_omega   = smoother_omega;

  for (PetscInt lev = 0; lev < max_levels - 1; lev++) {
    if (cur_n <= min_coarse_size) break; /* coarse enough */

    PetscInt    *s_ai = NULL, *s_aj = NULL;
    PetscInt    *CF = NULL;
    PetscInt     nC;
    PetscInt    *P_ai = NULL, *P_aj = NULL;
    PetscScalar *P_aa = NULL;
    PetscInt    *R_ai = NULL, *R_aj = NULL;
    PetscScalar *R_aa  = NULL;
    PetscInt    *Ac_ai = NULL, *Ac_aj = NULL, Ac_nnz = 0;
    PetscScalar *Ac_aa = NULL;

    /* 3.2 strength */
    PetscCall(BuildStrengthGraph(cur_ai, cur_aj, cur_aa, cur_n, strong_threshold, &s_ai, &s_aj));
    /* 3.3 C/F split */
    PetscCall(CFSplitting(s_ai, s_aj, cur_n, &CF, &nC));

    if (nC == 0 || nC == cur_n) {
      PetscCall(PetscFree(s_ai));
      PetscCall(PetscFree(s_aj));
      PetscCall(PetscFree(CF));
      PetscCall(PetscInfo(NULL, "AMG: coarsening stalled at level %" PetscInt_FMT " (nC=%" PetscInt_FMT ", cur_n=%" PetscInt_FMT ")\n", lev, nC, cur_n));
      break; /* coarsening stalled */
    }

    /* 3.4 prolongation -- reuse strength graph from 3.2 (avoid redundant rebuild) */
    PetscCall(BuildProlongation(cur_ai, cur_aj, cur_aa, cur_n, s_ai, s_aj, CF, nC, &P_ai, &P_aj, &P_aa));
    PetscCall(PetscFree(s_ai));
    PetscCall(PetscFree(s_aj));
    PetscCall(PetscFree(CF));

    /* 3.5 restriction */
    PetscCall(BuildRestriction(P_ai, P_aj, P_aa, cur_n, nC, &R_ai, &R_aj, &R_aa));

    /* 3.6 symbolic RAP */
    PetscCall(SymbolicRAP(cur_ai, cur_aj, cur_n, P_ai, P_aj, cur_n, R_ai, R_aj, nC, &Ac_ai, &Ac_aj, &Ac_nnz));

    /* 3.7 numeric RAP */
    PetscCall(NumericRAP_Host(cur_ai, cur_aj, cur_aa, cur_n, P_ai, P_aj, P_aa, R_ai, R_aj, R_aa, nC, Ac_ai, Ac_aj, &Ac_aa));

    /* 3.8 validate coarse-level diagonal: every row must have a structural diagonal entry.
       When using Jacobi smoother, the diagonal value must also be non-zero.
       When using L1-Jacobi smoother, the L1 row norm must also be non-zero. */
    for (PetscInt i = 0; i < nC; i++) {
      PetscBool has_diag = PETSC_FALSE;
      for (PetscInt k = Ac_ai[i]; k < Ac_ai[i + 1]; k++)
        if (Ac_aj[k] == i) {
          has_diag = PETSC_TRUE;
          PetscCheck(smoother_type != BJKOKKOS_SMOOTH_JACOBI || PetscAbsScalar(Ac_aa[k]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "AMG level %" PetscInt_FMT ": coarse row %" PetscInt_FMT " has zero diagonal; Jacobi smoother requires non-zero diagonal", lev, i);
          break;
        }
      PetscCheck(has_diag, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "AMG level %" PetscInt_FMT ": coarse row %" PetscInt_FMT " has no structural diagonal entry", lev, i);
      if (smoother_type == BJKOKKOS_SMOOTH_L1_JACOBI) {
        PetscReal l1 = 0.0;
        for (PetscInt k = Ac_ai[i]; k < Ac_ai[i + 1]; k++) l1 += PetscAbsScalar(Ac_aa[k]);
        PetscCheck(l1 > 0.0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "AMG level %" PetscInt_FMT ": coarse row %" PetscInt_FMT " has zero L1 row norm; L1-Jacobi smoother requires non-zero row norm", lev, i);
      }
    }

    /* store level */
    AMGLevel *L     = &hier->levels[hier->nlevels];
    L->nrows_fine   = cur_n;
    L->nrows_coarse = nC;
    L->P_ai         = P_ai;
    L->P_aj         = P_aj;
    L->P_aa         = P_aa;
    L->R_ai         = R_ai;
    L->R_aj         = R_aj;
    L->R_aa         = R_aa;
    L->Ac_ai        = Ac_ai;
    L->Ac_aj        = Ac_aj;
    L->Ac_nnz       = Ac_nnz;
    /* device views initialised to null; filled in step 3.10 */
    L->d_P_ai = L->d_P_aj = L->d_R_ai = L->d_R_aj = L->d_Ac_ai = L->d_Ac_aj = nullptr;
    L->d_P_aa = L->d_R_aa = nullptr;
    hier->nlevels++;
    PetscCall(PetscInfo(NULL, "AMG: level %" PetscInt_FMT ": %" PetscInt_FMT " -> %" PetscInt_FMT " DOFs (ratio %.2f), Ac_nnz=%" PetscInt_FMT ", P_nnz=%" PetscInt_FMT "\n", lev, cur_n, nC, (double)cur_n / (double)nC, Ac_nnz, P_ai[cur_n]));

    /* 3.10 copy this level to device */
    PetscCall(AMGLevelCopyToDevice(L));

    /* advance to coarse level.
       Do NOT free cur_ai/cur_aj here -- they are stored in levels[prev].Ac_ai/Ac_aj
       and will be freed by AMGLevelFreeHost.  Only free cur_aa (values are not stored
       in the level struct; they are recomputed per-block on the device). */
    if (owns_cur) PetscCall(PetscFree(cur_aa));
    cur_ai   = Ac_ai;
    cur_aj   = Ac_aj;
    cur_aa   = Ac_aa;
    cur_n    = nC;
    owns_cur = PETSC_TRUE;
  }

  /* store coarsest level matrix -- if nlevels==0 the pointers still reference
     the caller's (possibly stack-allocated) arrays, so we must copy them. */
  hier->nrows_coarsest  = cur_n;
  hier->Ac_coarsest_nnz = cur_ai[cur_n];
  if (!owns_cur) {
    PetscInt ai_len = cur_n + 1;
    PetscInt aj_len = cur_ai[cur_n];
    PetscCall(PetscMalloc1(ai_len, &hier->Ac_coarsest_ai));
    PetscCall(PetscMalloc1(aj_len, &hier->Ac_coarsest_aj));
    PetscCall(PetscArraycpy(hier->Ac_coarsest_ai, cur_ai, ai_len));
    PetscCall(PetscArraycpy(hier->Ac_coarsest_aj, cur_aj, aj_len));
  } else {
    hier->Ac_coarsest_ai = cur_ai;
    hier->Ac_coarsest_aj = cur_aj;
  }
  /* copy coarsest sparsity to device */
  using IntView1D   = Kokkos::View<PetscInt *>;
  using HostIntView = Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  HostIntView h_cai(cur_ai, cur_n + 1);
  HostIntView h_caj(cur_aj, hier->Ac_coarsest_nnz);
  hier->d_Ac_coarsest_ai = new IntView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_cai));
  hier->d_Ac_coarsest_aj = new IntView1D(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_caj));
  Kokkos::deep_copy(*hier->d_Ac_coarsest_ai, h_cai);
  Kokkos::deep_copy(*hier->d_Ac_coarsest_aj, h_caj);

  /* free coarsest values (only structure kept; values recomputed on device) */
  if (owns_cur) PetscCall(PetscFree(cur_aa));
  else {
    /* finest level: cur_aa == aa (caller owns), nothing to free */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Public entry point: PCBJKOKKOSSetupAMG
   Called from PCSetUp_BJKOKKOS when ksp_type_idx is an AMG variant.
   Builds one hierarchy per unique block size and maps each block to it.
   ----------------------------------------------------------------------- */
PetscErrorCode PCBJKOKKOSSetupAMG(PC pc, Mat Aseq)
{
  PC_PCBJKOKKOS     *jac = (PC_PCBJKOKKOS *)pc->data;
  const PetscInt    *d_ai, *d_aj;
  const PetscScalar *d_aa;
  PetscScalar       *dummy;
  PetscMemType       mtype;

  PetscFunctionBegin;
  /* defaults */
  if (jac->amg_max_levels <= 0) jac->amg_max_levels = PCBJKOKKOS_MAX_AMG_LEVELS;
  PetscCheck(jac->amg_max_levels <= PCBJKOKKOS_MAX_AMG_LEVELS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "amg_max_levels %" PetscInt_FMT " exceeds PCBJKOKKOS_MAX_AMG_LEVELS (%d)", jac->amg_max_levels, PCBJKOKKOS_MAX_AMG_LEVELS);
  if (jac->amg_strong_threshold <= 0.0) jac->amg_strong_threshold = 0.25;
  if (jac->amg_pre_sweeps <= 0) jac->amg_pre_sweeps = 1;
  if (jac->amg_post_sweeps <= 0) jac->amg_post_sweeps = 1;
  if (jac->amg_coarse_sweeps <= 0) jac->amg_coarse_sweeps = 10;

  PetscInt nBlk = jac->nBlocks;

  /* build host copy of block offsets */
  auto h_bid_off = Kokkos::create_mirror_view(*jac->d_bid_eqOffset_k);
  Kokkos::deep_copy(h_bid_off, *jac->d_bid_eqOffset_k);

  /* MatSeqAIJGetCSRAndMemType returns device pointers for Kokkos matrices on CUDA.
     We keep them as device pointers and use Kokkos parallel_for for extraction. */
  PetscCall(MatSeqAIJGetCSRAndMemType(Aseq, &d_ai, &d_aj, &dummy, &mtype));
  PetscCheck(PetscMemTypeDevice(mtype) || Kokkos::DefaultExecutionSpace().concurrency() < 1000, PETSC_COMM_SELF, PETSC_ERR_SUP, "MatSeqAIJGetCSRAndMemType returned host memory but Kokkos execution space is a device; PCBJKOKKOS AMG requires a Kokkos-aware (device) matrix");
  d_aa = dummy;

  /* Mirror d_isicol and d_isrow once (shared across all grids) */
  auto d_isicol = jac->d_isicol_k->data();
  auto d_isrow  = jac->d_isrow_k->data();

  /* Find unique block sizes (host loop over h_bid_off is fine) */
  std::vector<PetscInt> unique_sizes;
  for (PetscInt b = 0; b < nBlk; b++) {
    PetscInt sz    = h_bid_off[b + 1] - h_bid_off[b];
    bool     found = false;
    for (PetscInt g : unique_sizes) {
      if (g == sz) {
        found = true;
        break;
      }
    }
    if (!found) unique_sizes.push_back(sz);
  }
  PetscInt num_grids = (PetscInt)unique_sizes.size();

  PetscCall(PetscMalloc1(num_grids, &jac->amg_hierarchy));
  PetscCall(PetscMalloc1(nBlk, &jac->block_to_grid));
  jac->num_unique_grids = num_grids;

  /* map each block to its grid index */
  for (PetscInt b = 0; b < nBlk; b++) {
    PetscInt sz = h_bid_off[b + 1] - h_bid_off[b];
    for (PetscInt g = 0; g < num_grids; g++) {
      if (unique_sizes[g] == sz) {
        jac->block_to_grid[b] = g;
        break;
      }
    }
  }

  /* build one hierarchy per unique grid using the first block of that size */
  for (PetscInt g = 0; g < num_grids; g++) {
    /* find first block with this grid size */
    PetscInt first_blk = -1;
    for (PetscInt b = 0; b < nBlk; b++) {
      if (jac->block_to_grid[b] == g) {
        first_blk = b;
        break;
      }
    }
    PetscCheck(first_blk >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No block found for grid %" PetscInt_FMT, g);
    PetscInt start = h_bid_off[first_blk];
    PetscInt end   = h_bid_off[first_blk + 1];
    PetscInt blk_n = end - start;

    /* Extract local block CSR from device using Kokkos parallel_for.
       Pass 1: count nnz per block row into d_blk_ai[rowb-start+1] */
    Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace> d_blk_ai("d_blk_ai", blk_n + 1);
    Kokkos::deep_copy(d_blk_ai, 0);
    Kokkos::parallel_for(
      "blk_csr_count", Kokkos::RangePolicy<>(start, end), KOKKOS_LAMBDA(const PetscInt rowb) {
        PetscInt rowa = d_isicol[rowb];
        PetscInt cnt  = 0;
        for (PetscInt k = d_ai[rowa]; k < d_ai[rowa + 1]; k++) {
          PetscInt colb = d_isrow[d_aj[k]];
          if (colb >= start && colb < end) cnt++;
        }
        d_blk_ai[rowb - start + 1] = cnt;
      });
    Kokkos::fence();

    /* In-place prefix sum: ai[0]=0, ai[i] = ai[i-1] + ai[i] for i=1..blk_n.
       This converts per-row counts in ai[1..blk_n] into CSR row pointers. */
    {
      auto d_blk_ai_l = d_blk_ai; /* capture for lambda */
      Kokkos::parallel_scan(
        "blk_csr_scan", blk_n, KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
          PetscInt val = d_blk_ai_l[i + 1];
          update += val;
          if (final) d_blk_ai_l[i + 1] = update;
        });
    }
    Kokkos::fence();

    /* Read total nnz from device */
    PetscInt blk_nnz = 0;
    Kokkos::deep_copy(blk_nnz, Kokkos::subview(d_blk_ai, blk_n));

    /* Pass 2: fill d_blk_aj and d_blk_aa using row offsets.
       Each thread handles one row so no atomics are needed. */
    Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>    d_blk_aj("d_blk_aj", blk_nnz);
    Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> d_blk_aa("d_blk_aa", blk_nnz);
    Kokkos::parallel_for(
      "blk_csr_fill", Kokkos::RangePolicy<>(start, end), KOKKOS_LAMBDA(const PetscInt rowb) {
        PetscInt rowa   = d_isicol[rowb];
        PetscInt lrow   = rowb - start;
        PetscInt offset = d_blk_ai[lrow];
        PetscInt cnt    = 0;
        for (PetscInt k = d_ai[rowa]; k < d_ai[rowa + 1]; k++) {
          PetscInt colb = d_isrow[d_aj[k]];
          if (colb >= start && colb < end) {
            d_blk_aj[offset + cnt] = colb - start;
            d_blk_aa[offset + cnt] = d_aa[k];
            cnt++;
          }
        }
      });
    Kokkos::fence();

    /* Deep copy device block CSR to host vectors for BuildAMGHierarchy */
    auto h_blk_ai_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_blk_ai);
    auto h_blk_aj_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_blk_aj);
    auto h_blk_aa_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_blk_aa);

    std::vector<PetscInt>    blk_ai(h_blk_ai_v.data(), h_blk_ai_v.data() + blk_n + 1);
    std::vector<PetscInt>    blk_aj(h_blk_aj_v.data(), h_blk_aj_v.data() + blk_nnz);
    std::vector<PetscScalar> blk_aa(h_blk_aa_v.data(), h_blk_aa_v.data() + blk_nnz);

    PetscCall(PetscInfo(pc, "AMG: building hierarchy for grid %" PetscInt_FMT " (size %" PetscInt_FMT ")\n", g, blk_n));

    /* Validate fine-level diagonal: every row must have a structural diagonal entry.
       When using Jacobi smoother, the diagonal value must also be non-zero.
       When using L1-Jacobi smoother, the L1 row norm must also be non-zero. */
    for (PetscInt i = 0; i < blk_n; i++) {
      PetscBool has_diag = PETSC_FALSE;
      for (PetscInt k = blk_ai[i]; k < blk_ai[i + 1]; k++)
        if (blk_aj[k] == i) {
          has_diag = PETSC_TRUE;
          PetscCheck(jac->amg_smoother_type != BJKOKKOS_SMOOTH_JACOBI || PetscAbsScalar(blk_aa[k]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "AMG grid %" PetscInt_FMT ": fine row %" PetscInt_FMT " has zero diagonal; Jacobi smoother requires non-zero diagonal", g, i);
          break;
        }
      PetscCheck(has_diag, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "AMG grid %" PetscInt_FMT ": fine row %" PetscInt_FMT " has no structural diagonal entry", g, i);
      if (jac->amg_smoother_type == BJKOKKOS_SMOOTH_L1_JACOBI) {
        PetscReal l1 = 0.0;
        for (PetscInt k = blk_ai[i]; k < blk_ai[i + 1]; k++) l1 += PetscAbsScalar(blk_aa[k]);
        PetscCheck(l1 > 0.0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "AMG grid %" PetscInt_FMT ": fine row %" PetscInt_FMT " has zero L1 row norm; L1-Jacobi smoother requires non-zero row norm", g, i);
      }
    }

    PetscCall(BuildAMGHierarchy(blk_ai.data(), blk_aj.data(), blk_aa.data(), blk_n, jac->amg_strong_threshold, jac->amg_max_levels, jac->amg_min_coarse_size, jac->amg_pre_sweeps, jac->amg_post_sweeps, jac->amg_coarse_sweeps, jac->amg_smoother_type,
                                jac->amg_smoother_omega, &jac->amg_hierarchy[g]));

    PetscCall(PetscInfo(pc, "AMG: grid %" PetscInt_FMT " hierarchy has %" PetscInt_FMT " levels, coarsest size %" PetscInt_FMT "\n", g, jac->amg_hierarchy[g].nlevels, jac->amg_hierarchy[g].nrows_coarsest));

    /* Store fine-grid local CSR (local 0-based column indices) in the hierarchy */
    {
      AMGHierarchy *hier    = &jac->amg_hierarchy[g];
      PetscInt      fine_n  = blk_n;
      PetscInt      fine_nz = (PetscInt)blk_aj.size();
      hier->fine_nnz        = fine_nz;
      PetscCall(PetscMalloc1(fine_n + 1, &hier->fine_ai));
      PetscCall(PetscMalloc1(fine_nz, &hier->fine_aj));
      for (PetscInt i = 0; i <= fine_n; i++) hier->fine_ai[i] = blk_ai[i];
      for (PetscInt k = 0; k < fine_nz; k++) hier->fine_aj[k] = blk_aj[k];
      /* copy sparsity to device */
      const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_fai(hier->fine_ai, fine_n + 1);
      const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_faj(hier->fine_aj, fine_nz);
      hier->d_fine_ai = new Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_fai));
      hier->d_fine_aj = new Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_faj));
      Kokkos::deep_copy(*hier->d_fine_ai, h_fai);
      Kokkos::deep_copy(*hier->d_fine_aj, h_faj);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Public entry point: PCBJKOKKOSDestroyAMG
   ----------------------------------------------------------------------- */
PetscErrorCode PCBJKOKKOSDestroyAMG(PC_PCBJKOKKOS *jac)
{
  PetscFunctionBegin;
  if (!jac->amg_hierarchy) PetscFunctionReturn(PETSC_SUCCESS);

  for (PetscInt g = 0; g < jac->num_unique_grids; g++) {
    AMGHierarchy *hier = &jac->amg_hierarchy[g];
    if (hier->nlevels > 0) {
      /* Null out coarsest aliases BEFORE freeing levels, since Ac_coarsest_ai/aj
         are aliases of levels[nlevels-1].Ac_ai/aj and will be freed by AMGLevelFreeHost. */
      hier->Ac_coarsest_ai = nullptr;
      hier->Ac_coarsest_aj = nullptr;
    } else {
      /* nlevels==0: Ac_coarsest_ai/aj were independently allocated in BuildAMGHierarchy */
      PetscCall(PetscFree(hier->Ac_coarsest_ai));
      PetscCall(PetscFree(hier->Ac_coarsest_aj));
    }
    for (PetscInt lev = 0; lev < hier->nlevels; lev++) {
      PetscCall(AMGLevelFreeHost(&hier->levels[lev]));
      PetscCall(AMGLevelFreeDevice(&hier->levels[lev]));
    }
    delete hier->d_Ac_coarsest_ai;
    hier->d_Ac_coarsest_ai = nullptr;
    delete hier->d_Ac_coarsest_aj;
    hier->d_Ac_coarsest_aj = nullptr;
    /* fine-grid local CSR */
    PetscCall(PetscFree(hier->fine_ai));
    PetscCall(PetscFree(hier->fine_aj));
    delete hier->d_fine_ai;
    hier->d_fine_ai = nullptr;
    delete hier->d_fine_aj;
    hier->d_fine_aj = nullptr;
  }
  PetscCall(PetscFree(jac->amg_hierarchy));
  jac->amg_hierarchy = nullptr;
  PetscCall(PetscFree(jac->block_to_grid));
  jac->block_to_grid    = nullptr;
  jac->num_unique_grids = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
