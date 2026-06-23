#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>
#include <petscsf.h>

#define MISK_NOT_DONE -2
#define MISK_REMOVED  -3

/* Knuth-style multiplicative hash for Luby MIS weight assignment -- hashing the global id
   makes the result thread-schedule-independent so golden-output tests stay reproducible.
   KOKKOS_INLINE_FUNCTION so it is callable from device kernels. */
KOKKOS_INLINE_FUNCTION static PetscInt misk_hash(PetscInt gid)
{
  PetscInt h = gid ^ (gid >> 16);
  h *= 0x45d9f3b;
  h ^= h >> 16;
  return h & 0x7FFFFFFF;
}

/* Device CSR connectivity (row_map, entries) of a SEQAIJ or SEQAIJKOKKOS graph. For a
   SEQAIJKOKKOS matrix the device views are returned directly; for a host SEQAIJ matrix a
   mirror is created and copied. Values are unused -- strength filtering is baked into Gmat. */
static PetscErrorCode MatCoarsenMISKokkosGetDeviceCSR(Mat M, Kokkos::View<const PetscInt *, DefaultMemorySpace> &i_d, Kokkos::View<const PetscInt *, DefaultMemorySpace> &j_d)
{
  PetscInt  nrows;
  PetscBool isKok;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(M, &nrows, NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)M, MATSEQAIJKOKKOS, &isKok));
  if (isKok) {
    KokkosCsrMatrix csr;

    PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(M, &csr));
    i_d = csr.graph.row_map;
    j_d = csr.graph.entries;
  } else {
    Mat_SeqAIJ *m = (Mat_SeqAIJ *)M->data;

    Kokkos::View<const PetscInt *, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> i_h(m->i, nrows + 1);
    Kokkos::View<const PetscInt *, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> j_h(m->j, m->i[nrows]);
    PetscCallCXX(i_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), i_h));
    PetscCallCXX(j_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), j_h));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Refresh the device ghost-owner buffer from the local owner array via a device-resident
   PetscSF broadcast over the column layout (owning rank -> ghost copies). The owner array
   never leaves device memory: PetscSFBcastWithMemType moves it through the same memtype-aware
   path MPIAIJKOKKOS MatMult uses for its halo, so there is no per-round host staging. */
static PetscErrorCode MatCoarsenMISKokkosHaloExchange(PetscSF sf, PetscIntKokkosView owner_d, PetscIntKokkosView cpcol_owner_d)
{
  PetscFunctionBegin;
  PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_INT, PETSC_MEMTYPE_KOKKOS, owner_d.data(), PETSC_MEMTYPE_KOKKOS, cpcol_owner_d.data(), MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, owner_d.data(), cpcol_owner_d.data(), MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatCoarsenApply_MISKokkos_Device - device-resident parallel maximal independent set (MIS).

  Runs Jones-Plassmann/Luby rounds as Kokkos kernels over the graph's device CSR and
  produces a per-vertex owner array (owner(i) = aggregate-root gid, or MISK_REMOVED for
  singletons). A single D2H copy of that array feeds the host bridge, which builds the
  same PetscCoarsenData(strict_aggs = PETSC_TRUE) contract as MatCoarsenApply_MIS_private()
  for SEQAIJ graphs. Determinism (weight hashed on the global id, lowest-gid root wins a
  tie) makes two runs produce identical owner arrays, independent of the rank count.

  Handles SEQAIJ/SEQAIJKOKKOS (single rank) and MPIAIJ/MPIAIJKOKKOS. In the MPI case the
  diagonal block A and off-diagonal block B are coarsened together: ghost (off-rank)
  neighbors enter the root-selection and absorb comparisons through a PetscSF halo exchange
  of the owner array, mirroring MatCoarsenApply_MIS_private().
*/
static PetscErrorCode MatCoarsenApply_MISKokkos_Device(MatCoarsen coarse)
{
  Mat               Gmat   = coarse->graph;
  Mat_MPIAIJ       *mpimat = NULL;
  MPI_Comm          comm;
  PetscInt          nloc, my0, Iend, Nglobal, lid, o, nselected = 0, nremoved = 0, g_undone, round;
  PetscInt          num_fine_ghosts = 0, *cpcol_gid = NULL, *lid_gid = NULL, *owner_h;
  PetscCoarsenData *agg_lists;
  PetscBool         isMPI, isAIJ;
  PetscSF           sf = NULL;
  PetscLayout       layout;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Gmat, &comm));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPI));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATSEQAIJ, &isAIJ));
  PetscCheck(isMPI || isAIJ, comm, PETSC_ERR_PLIB, "MatCoarsenApply_MISKokkos requires a SEQAIJ- or MPIAIJ-based graph");
  PetscCheck(coarse->strict_aggs, comm, PETSC_ERR_SUP, "MatCoarsenApply_MISKokkos requires strict_aggs = PETSC_TRUE");

  PetscCall(MatGetOwnershipRange(Gmat, &my0, &Iend));
  PetscCall(MatGetSize(Gmat, &Nglobal, NULL));
  nloc = Iend - my0;

  PetscCall(PetscKokkosInitializeCheck()); /* a SEQAIJ graph may reach here before Kokkos is initialized */

  /* MPI setup: ghost gids and a PetscSF over the column layout (owning rank -> ghost copies). */
  if (isMPI) {
    mpimat = (Mat_MPIAIJ *)Gmat->data;
    PetscCall(VecGetLocalSize(mpimat->lvec, &num_fine_ghosts));
    PetscCall(PetscMalloc2(nloc, &lid_gid, num_fine_ghosts, &cpcol_gid));
    for (PetscInt kk = 0; kk < nloc; kk++) lid_gid[kk] = my0 + kk;
    PetscCall(PetscSFCreate(comm, &sf));
    PetscCall(MatGetLayouts(Gmat, &layout, NULL));
    PetscCall(PetscSFSetGraphLayout(sf, layout, num_fine_ghosts, NULL, PETSC_COPY_VALUES, mpimat->garray));
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, lid_gid, cpcol_gid, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, lid_gid, cpcol_gid, MPI_REPLACE));
  }

  {
    /* A (local diagonal block) and B (off-diagonal, ghost columns) device CSR. For the
       sequential case B is an empty stand-in so a single set of kernels covers both. */
    Kokkos::View<const PetscInt *, DefaultMemorySpace> ai_d, aj_d, bi_d, bj_d;
    PetscIntKokkosView                                 bi_owner; /* keeps the seq dummy alive */
    PetscIntKokkosView                                 owner_d("misk_owner", nloc);
    PetscIntKokkosView                                 is_root("misk_is_root", nloc);
    PetscIntKokkosView                                 cpcol_owner_d("misk_cpcol_owner", num_fine_ghosts);
    PetscIntKokkosView                                 cpcol_gid_d("misk_cpcol_gid", num_fine_ghosts);
    const PetscInt                                     my0_k = my0;
    const PetscInt                                     nfg   = num_fine_ghosts;

    if (isMPI) {
      PetscCall(MatCoarsenMISKokkosGetDeviceCSR(mpimat->A, ai_d, aj_d));
      PetscCall(MatCoarsenMISKokkosGetDeviceCSR(mpimat->B, bi_d, bj_d));
      {
        PetscIntKokkosViewHost cpcol_gid_hv(cpcol_gid, nfg);
        PetscCallCXX(Kokkos::deep_copy(cpcol_gid_d, cpcol_gid_hv));
      }
    } else {
      PetscCall(MatCoarsenMISKokkosGetDeviceCSR(Gmat, ai_d, aj_d));
      PetscCallCXX(bi_owner = PetscIntKokkosView("misk_bi_empty", nloc + 1)); /* all-zero row map => no ghost neighbors */
      bi_d = bi_owner;
      PetscCallCXX(bj_d = PetscIntKokkosView("misk_bj_empty", 0));
    }

    /* Initialize: weight via global-id hash, mark singletons (fewer than two real neighbors,
       counting both local and ghost adjacency) removed, everything else undecided. */
    PetscCallCXX(Kokkos::parallel_for(
      "misk_init", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc), KOKKOS_LAMBDA(const PetscInt i) {
        const PetscBool sing = (PetscBool)((ai_d(i + 1) - ai_d(i) < 2) && (bi_d(i + 1) - bi_d(i) == 0));
        owner_d(i)           = sing ? MISK_REMOVED : MISK_NOT_DONE;
      }));

    /* Seed ghost states so the first round sees consistent off-rank owner values. */
    if (isMPI) PetscCall(MatCoarsenMISKokkosHaloExchange(sf, owner_d, cpcol_owner_d));

    /* Luby rounds. Each round: (1) snapshot-select roots, (2) commit roots, (3) absorb
       undecided neighbors of a root (lowest-gid root wins for determinism). Ghost neighbors
       participate via cpcol_owner_d/cpcol_gid_d, refreshed by a halo exchange after commit
       and after absorb. */
    round    = 0;
    g_undone = Nglobal; /* upper bound; recomputed each round */
    while (g_undone > 0) {
      PetscInt l_undone = 0;

      /* (1) mark roots: undecided vertex maximal in (weight, gid) among undecided neighbors,
         local (A) and ghost (B). */
      PetscCallCXX(Kokkos::parallel_for(
        "misk_root", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc), KOKKOS_LAMBDA(const PetscInt i) {
          PetscInt rt = 0;
          if (owner_d(i) == MISK_NOT_DONE) {
            const PetscInt wi = misk_hash(i + my0_k), gi = i + my0_k;
            rt = 1;
            for (PetscInt k = ai_d(i); k < ai_d(i + 1); k++) {
              const PetscInt jj = aj_d(k);
              if (jj == i || owner_d(jj) != MISK_NOT_DONE) continue;
              const PetscInt wj = misk_hash(jj + my0_k);
              if (wj > wi || (wj == wi && (jj + my0_k) > gi)) {
                rt = 0;
                break;
              }
            }
            if (rt) {
              for (PetscInt k = bi_d(i); k < bi_d(i + 1); k++) {
                const PetscInt cpid = bj_d(k);
                if (cpcol_owner_d(cpid) != MISK_NOT_DONE) continue;
                const PetscInt gj = cpcol_gid_d(cpid), wj = misk_hash(gj);
                if (wj > wi || (wj == wi && gj > gi)) {
                  rt = 0;
                  break;
                }
              }
            }
          }
          is_root(i) = rt;
        }));

      /* (2) commit roots: selected state encodes the root's own global id */
      PetscCallCXX(Kokkos::parallel_for(
        "misk_commit", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc), KOKKOS_LAMBDA(const PetscInt i) {
          if (is_root(i)) owner_d(i) = i + my0_k;
        }));

      /* (3a) refresh ghosts so newly committed off-rank roots are visible to absorb */
      if (isMPI) PetscCall(MatCoarsenMISKokkosHaloExchange(sf, owner_d, cpcol_owner_d));

      /* (3b) absorb: each still-undecided vertex joins the lowest-gid root among its neighbors.
         A ghost neighbor is a root iff it owns itself (owner == its own gid). */
      PetscCallCXX(Kokkos::parallel_for(
        "misk_absorb", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc), KOKKOS_LAMBDA(const PetscInt i) {
          if (owner_d(i) != MISK_NOT_DONE) return;
          PetscInt best = -1;
          for (PetscInt k = ai_d(i); k < ai_d(i + 1); k++) {
            const PetscInt jj = aj_d(k);
            if (jj == i || !is_root(jj)) continue;
            const PetscInt gj = jj + my0_k;
            if (best < 0 || gj < best) best = gj;
          }
          for (PetscInt k = bi_d(i); k < bi_d(i + 1); k++) {
            const PetscInt cpid = bj_d(k), gj = cpcol_gid_d(cpid);
            if (cpcol_owner_d(cpid) != gj) continue; /* not a root */
            if (best < 0 || gj < best) best = gj;
          }
          if (best >= 0) owner_d(i) = best;
        }));

      /* (3c) refresh ghosts so the next round's root-mark sees updated decided state */
      if (isMPI) PetscCall(MatCoarsenMISKokkosHaloExchange(sf, owner_d, cpcol_owner_d));

      /* (4) convergence: count remaining undecided, globally */
      PetscCallCXX(Kokkos::parallel_reduce(
        "misk_count", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc),
        KOKKOS_LAMBDA(const PetscInt i, PetscInt &lsum) {
          if (owner_d(i) == MISK_NOT_DONE) lsum++;
        },
        l_undone));
      if (isMPI) PetscCallMPI(MPIU_Allreduce(&l_undone, &g_undone, 1, MPIU_INT, MPI_SUM, comm));
      else g_undone = l_undone;
      round++;
      PetscCheck(round <= Nglobal + 1, comm, PETSC_ERR_PLIB, "MatCoarsenApply_MISKokkos: Luby iteration did not converge");
    }

    /* Single D2H copy of the resolved owner array feeds the host bridge. */
    PetscCall(PetscMalloc1(nloc, &owner_h));
    {
      PetscIntKokkosViewHost owner_hv(owner_h, nloc);
      PetscCallCXX(Kokkos::deep_copy(owner_hv, owner_d));
    }
  }

  /* Host bridge: build the strict_aggs PetscCoarsenData. Root's own gid is appended first,
     then locally-owned members, matching the [root, members...] list order of
     MatCoarsenApply_MIS_private(). Members absorbed across a rank boundary are added via a
     final PetscSF exchange of each vertex's parent (root) gid. */
  PetscCall(PetscCDCreate(nloc, &agg_lists));
  coarse->agg_lists = agg_lists;
  for (lid = 0; lid < nloc; lid++) {
    o = owner_h[lid];
    if (o == MISK_REMOVED) nremoved++;
    else if (o == lid + my0) {
      PetscCall(PetscCDAppendID(agg_lists, lid, lid + my0));
      nselected++;
    }
  }
  for (lid = 0; lid < nloc; lid++) {
    o = owner_h[lid];
    if (o >= 0 && o != lid + my0 && o >= my0 && o < Iend) PetscCall(PetscCDAppendID(agg_lists, o - my0, lid + my0));
  }
  if (isMPI) {
    PetscInt *lid_parent_gid, *cpcol_sel_gid;

    PetscCall(PetscMalloc2(nloc, &lid_parent_gid, num_fine_ghosts, &cpcol_sel_gid));
    for (lid = 0; lid < nloc; lid++) lid_parent_gid[lid] = (owner_h[lid] >= 0) ? owner_h[lid] : -1;
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, lid_parent_gid, cpcol_sel_gid, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, lid_parent_gid, cpcol_sel_gid, MPI_REPLACE));
    for (PetscInt cpid = 0; cpid < num_fine_ghosts; cpid++) {
      PetscInt sgid = cpcol_sel_gid[cpid];
      if (sgid >= my0 && sgid < Iend) PetscCall(PetscCDAppendID(agg_lists, sgid - my0, cpcol_gid[cpid]));
    }
    PetscCall(PetscFree2(lid_parent_gid, cpcol_sel_gid));
  }

  if (isMPI) {
    PetscInt aa[2], bb[2];

    aa[0] = 0;
    aa[1] = nremoved;
    PetscCall(PetscCDCount(agg_lists, &aa[0]));
    PetscCallMPI(MPIU_Allreduce(aa, bb, 2, MPIU_INT, MPI_SUM, comm));
    PetscCheck(Nglobal >= bb[0], comm, PETSC_ERR_PLIB, "Sum of aggregates %" PetscInt_FMT " exceeds N = %" PetscInt_FMT, bb[0], Nglobal);
    if (Nglobal != bb[0]) PetscCall(PetscInfo(Gmat, "mis_kokkos: N = %" PetscInt_FMT ", sum of aggregates %" PetscInt_FMT ", %" PetscInt_FMT " removed total\n", Nglobal, bb[0], bb[1]));
  }

  PetscCall(PetscInfo(Gmat, "mis_kokkos: removed %" PetscInt_FMT " of %" PetscInt_FMT " local vertices, %" PetscInt_FMT " selected in %" PetscInt_FMT " Luby rounds\n", nremoved, nloc, nselected, round));
  PetscCall(PetscFree(owner_h));
  if (isMPI) {
    PetscCall(PetscSFDestroy(&sf));
    PetscCall(PetscFree2(lid_gid, cpcol_gid));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCoarsenApply_MISKokkos(MatCoarsen coarse)
{
  PetscFunctionBegin;
  PetscCall(MatCoarsenApply_MISKokkos_Device(coarse));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCoarsenView_MISKokkos(MatCoarsen coarse, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  MIS Kokkos aggregator (device-resident Luby MIS)\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATCOARSENMISKOKKOS - A coarsening object using a Kokkos device-resident parallel maximal independent set (MIS) algorithm

   Level: beginner

   Notes:
   This coarsener targets `PCGAMG` on device-resident operators via Jones-Plassmann hash
   weights for deterministic, thread-schedule-independent aggregate selection. It accepts
   `MATSEQAIJ`, `MATSEQAIJKOKKOS`, `MATMPIAIJ`, and `MATMPIAIJKOKKOS` strength graphs and
   runs the selection in Kokkos kernels. In the MPI case the off-rank boundary neighborhood
   is exchanged through a device-resident `PetscSF` halo of the owner array (the owner buffer
   stays in device memory across rounds); the result is independent of the rank count.

.seealso: `MatCoarsen`, `MatCoarsenApply()`, `MatCoarsenGetData()`, `MatCoarsenSetType()`, `MatCoarsenType`, `MATCOARSENMIS`
M*/
PETSC_EXTERN PetscErrorCode MatCoarsenCreate_MISKokkos(MatCoarsen coarse)
{
  PetscFunctionBegin;
  coarse->ops->apply = MatCoarsenApply_MISKokkos;
  coarse->ops->view  = MatCoarsenView_MISKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}
