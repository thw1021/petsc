#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

#define MISK_NOT_DONE -2
#define MISK_REMOVED  -3

/* Knuth-style multiplicative hash for Luby MIS weight assignment — hashing the global id
   makes the result thread-schedule-independent so golden-output tests stay reproducible.
   KOKKOS_INLINE_FUNCTION so it is callable from device kernels. */
KOKKOS_INLINE_FUNCTION static PetscInt misk_hash(PetscInt gid)
{
  PetscInt h = gid ^ (gid >> 16);
  h *= 0x45d9f3b;
  h ^= h >> 16;
  return h & 0x7FFFFFFF;
}

/*
  MatCoarsenApply_MISKokkos_Device - device-resident parallel maximal independent set (MIS).

  Runs Jones-Plassmann/Luby rounds as Kokkos kernels over the graph's device CSR and
  produces a per-vertex owner array (owner(i) = aggregate-root gid, or MISK_REMOVED for
  singletons). A single D2H copy of that array feeds the host bridge, which builds the
  same PetscCoarsenData(strict_aggs = PETSC_TRUE) contract as MatCoarsenApply_MIS_private()
  for SEQAIJ graphs. Determinism (weight hashed on the global id, lowest-gid root wins a
  tie) makes two runs produce identical owner arrays.

  Targets single-rank SEQAIJ / SEQAIJKOKKOS graphs; MPIAIJ errors out (use MATCOARSENMIS).
*/
static PetscErrorCode MatCoarsenApply_MISKokkos_Device(MatCoarsen coarse)
{
  Mat               Gmat = coarse->graph;
  MPI_Comm          comm;
  PetscInt          nloc, my0, Iend, lid, o, nselected = 0, nremoved = 0, n_undone, round;
  PetscInt         *owner_h;
  PetscCoarsenData *agg_lists;
  PetscBool         isAIJ, isMPI, isKok;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Gmat, &comm));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPI));
  PetscCheck(!isMPI, comm, PETSC_ERR_SUP, "MatCoarsenApply_MISKokkos: MPI parallel case not yet implemented; use MATCOARSENMIS");
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATSEQAIJ, &isAIJ));
  PetscCheck(isAIJ, comm, PETSC_ERR_PLIB, "MatCoarsenApply_MISKokkos requires a SEQAIJ-based graph");
  PetscCheck(coarse->strict_aggs, comm, PETSC_ERR_SUP, "MatCoarsenApply_MISKokkos requires strict_aggs = PETSC_TRUE");
  PetscCall(PetscObjectTypeCompare((PetscObject)Gmat, MATSEQAIJKOKKOS, &isKok));

  PetscCall(MatGetOwnershipRange(Gmat, &my0, &Iend));
  nloc = Iend - my0;

  PetscCall(PetscKokkosInitializeCheck()); /* a SEQAIJ graph may reach here before Kokkos is initialized */
  {
    /* Device CSR connectivity (values unused — strength filtering is already baked into Gmat). */
    Kokkos::View<const PetscInt *, DefaultMemorySpace> i_d, j_d;
    PetscIntKokkosView                                 owner_d("misk_owner", nloc);
    PetscIntKokkosView                                 is_root("misk_is_root", nloc);
    const PetscInt                                     my0_k = my0;

    if (isKok) {
      KokkosCsrMatrix csr;

      PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(Gmat, &csr));
      i_d = csr.graph.row_map;
      j_d = csr.graph.entries;
    } else {
      Mat_SeqAIJ *matA = (Mat_SeqAIJ *)Gmat->data;

      Kokkos::View<const PetscInt *, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> i_h(matA->i, nloc + 1);
      Kokkos::View<const PetscInt *, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> j_h(matA->j, matA->i[nloc]);
      PetscCallCXX(i_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), i_h));
      PetscCallCXX(j_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), j_h));
    }

    /* Initialize: weight via global-id hash, mark singletons (rows with < 2 nonzeros have no
       real neighbor) removed, everything else undecided. */
    PetscCallCXX(Kokkos::parallel_for("misk_init", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc), KOKKOS_LAMBDA(const PetscInt i) { owner_d(i) = (i_d(i + 1) - i_d(i) < 2) ? MISK_REMOVED : MISK_NOT_DONE; }));

    /* Luby rounds. Each round: (1) snapshot-select roots, (2) commit roots, (3) absorb
       undecided neighbors of a root (lowest-gid root wins for determinism). */
    round    = 0;
    n_undone = nloc; /* upper bound; recomputed each round */
    while (n_undone > 0) {
      /* (1) mark roots: undecided vertex maximal in (weight, gid) among undecided neighbors */
      PetscCallCXX(Kokkos::parallel_for(
        "misk_root", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc), KOKKOS_LAMBDA(const PetscInt i) {
          PetscInt rt = 0;
          if (owner_d(i) == MISK_NOT_DONE) {
            const PetscInt wi = misk_hash(i + my0_k), gi = i + my0_k;
            rt = 1;
            for (PetscInt k = i_d(i); k < i_d(i + 1); k++) {
              const PetscInt jj = j_d(k);
              if (jj == i || owner_d(jj) != MISK_NOT_DONE) continue;
              const PetscInt wj = misk_hash(jj + my0_k);
              if (wj > wi || (wj == wi && (jj + my0_k) > gi)) {
                rt = 0;
                break;
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

      /* (3) absorb: each still-undecided vertex joins the lowest-gid root among its neighbors */
      PetscCallCXX(Kokkos::parallel_for(
        "misk_absorb", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc), KOKKOS_LAMBDA(const PetscInt i) {
          if (owner_d(i) != MISK_NOT_DONE) return;
          PetscInt best = -1;
          for (PetscInt k = i_d(i); k < i_d(i + 1); k++) {
            const PetscInt jj = j_d(k);
            if (jj == i || !is_root(jj)) continue;
            const PetscInt gj = jj + my0_k;
            if (best < 0 || gj < best) best = gj;
          }
          if (best >= 0) owner_d(i) = best;
        }));

      /* (4) convergence: count remaining undecided */
      n_undone = 0;
      PetscCallCXX(Kokkos::parallel_reduce(
        "misk_count", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nloc),
        KOKKOS_LAMBDA(const PetscInt i, PetscInt &lsum) {
          if (owner_d(i) == MISK_NOT_DONE) lsum++;
        },
        n_undone));
      round++;
      PetscCheck(round <= nloc + 1, comm, PETSC_ERR_PLIB, "MatCoarsenApply_MISKokkos: Luby iteration did not converge");
    }

    /* Single D2H copy of the resolved owner array feeds the host bridge. */
    PetscCall(PetscMalloc1(nloc, &owner_h));
    {
      PetscIntKokkosViewHost owner_hv(owner_h, nloc);
      PetscCallCXX(Kokkos::deep_copy(owner_hv, owner_d));
    }
  }

  /* Host bridge: build the strict_aggs PetscCoarsenData. Root's own gid is appended first
     (pass 1), then absorbed members under their root's local index (pass 2), matching the
     [root, members...] list order of MatCoarsenApply_MIS_private(). */
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
    if (o >= 0 && o != lid + my0) PetscCall(PetscCDAppendID(agg_lists, o - my0, lid + my0));
  }

  PetscCall(PetscInfo(Gmat, "mis_kokkos: removed %" PetscInt_FMT " of %" PetscInt_FMT " vertices, %" PetscInt_FMT " selected in %" PetscInt_FMT " Luby rounds\n", nremoved, nloc, nselected, round));
  PetscCall(PetscFree(owner_h));
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
   This coarsener targets single-GPU `PCGAMG` via Jones-Plassmann hash weights for
   deterministic, thread-schedule-independent aggregate selection. It accepts `MATSEQAIJ`
   and `MATSEQAIJKOKKOS` strength graphs and runs the selection in Kokkos kernels. MPI
   parallel coarsening is not yet supported; use `MATCOARSENMIS` for multi-rank runs.

.seealso: `MatCoarsen`, `MatCoarsenApply()`, `MatCoarsenGetData()`, `MatCoarsenSetType()`, `MatCoarsenType`, `MATCOARSENMIS`
M*/
PETSC_EXTERN PetscErrorCode MatCoarsenCreate_MISKokkos(MatCoarsen coarse)
{
  PetscFunctionBegin;
  coarse->ops->apply = MatCoarsenApply_MISKokkos;
  coarse->ops->view  = MatCoarsenView_MISKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}
