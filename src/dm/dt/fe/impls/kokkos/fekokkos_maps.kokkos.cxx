/*
  fekokkos_maps.kokkos.cxx -- PetscFEKokkosMaps lifecycle functions

  Non-template host functions that manage the PetscFEKokkosMaps struct:
  build, stage, resize, preallocate, reset, destroy, set-up geometry, set-up.

  These were previously static inline in petscfekokkos.h.  Moving them here
  reduces per-TU compile time for every .kokkos.cxx file that includes the
  header, and eliminates ~800 lines of duplicated object code per translation
  unit.

  The template kernel functions (PetscFEKokkosIntegrateResidualCell, etc.)
  and the template host wrappers (DMPlexSNESComputeResidualFEM_Kokkos, etc.)
  remain in petscfekokkos.h because they must be instantiated in each TU
  that calls them with specific template arguments.
*/

#include <petsc/private/petscfeimpl.h>
#include <petsc/private/petscfekokkosimpl.h>
#include <Kokkos_Core.hpp>

/*@C
  PetscFEKokkosMapsCreate - Allocate a `PetscFEKokkosMaps` struct with proper C++ construction

  Not Collective

  Output Parameter:
. maps - pointer to the newly allocated `PetscFEKokkosMaps`

  Level: intermediate

  Notes:
  This function must be used instead of `PetscNew()` because `PetscFEKokkosMaps`
  contains `Kokkos::View` members that require C++ construction.  `PetscNew()`
  zero-fills memory, which leaves the Views' internal reference-count tracker
  in an invalid state and causes a segfault on the first assignment.

  The returned struct must be freed with `PetscFEKokkosMapsDestroy()`.

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosMapsDestroy()`, `PetscFEKokkosSetUp()`
@*/
PetscErrorCode PetscFEKokkosMapsCreate(PetscFEKokkosMaps **maps)
{
  PetscFunctionBegin;
  PetscAssertPointer(maps, 1);
  *maps = new PetscFEKokkosMaps();
  /* Zero the POD (non-View) fields that PetscNew would have zeroed */
  (*maps)->num_elements             = 0;
  (*maps)->num_dof                  = 0;
  (*maps)->local_dof                = 0;
  (*maps)->Nb                       = 0;
  (*maps)->totDim                   = 0;
  (*maps)->coo_size                 = 0;
  (*maps)->h_gIdx                   = NULL;
  (*maps)->h_lIdx                   = NULL;
  (*maps)->h_active_idx             = NULL;
  (*maps)->h_Nb_active              = NULL;
  (*maps)->h_coo_elem_offsets       = NULL;
  (*maps)->num_reduced              = 0;
  (*maps)->num_face                 = 0;
  (*maps)->h_c_maps_gid             = NULL;
  (*maps)->h_c_maps_scale           = NULL;
  (*maps)->h_coo_elem_point_offsets = NULL;
  (*maps)->h_fullNb                 = NULL;
  (*maps)->cached_Ne                = -1;
  (*maps)->cached_Nq                = -1;
  (*maps)->cached_Nc                = -1;
  (*maps)->cached_dim               = -1;
  (*maps)->cached_dE                = -1;
  (*maps)->cached_totDim            = -1;
  (*maps)->cached_numConstants      = -1;
  (*maps)->isAffine                 = PETSC_FALSE;
  (*maps)->geom_cached              = PETSC_FALSE;
  (*maps)->cached_dE_geom           = 0;
  (*maps)->cached_cStart            = 0;
  (*maps)->cached_cEnd              = 0;
  (*maps)->cached_fullGeom          = NULL;
  (*maps)->cached_chunkGeom         = NULL;
  (*maps)->cached_cellIS            = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEKokkosCreateMaps - Build host-side assembly maps (DOF indices, constraint info, COO offsets) for GPU FEM assembly

  Not Collective

  Input Parameter:
. dm - the `DM` with `PetscFE` discretization attached (after `DMCreateDS()`)

  Output Parameter:
. maps - pointer to a zero-initialized `PetscFEKokkosMaps` struct; populated with host arrays

  Level: intermediate

  Notes:
  This is the first phase of `PetscFEKokkosSetUp()`. It builds the element-to-global DOF index
  map, identifies constrained DOFs, and computes COO element offsets on the host. The host arrays
  must be staged to device with `PetscFEKokkosStageMaps()` before any GPU kernel launch.

  For most users, `PetscFEKokkosSetUp()` is the preferred entry point, which calls this function
  internally.

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosSetUp()`, `PetscFEKokkosStageMaps()`, `PetscFEKokkosPreallocateCOO()`,
          `PetscFEKokkosSetUpGeometry()`, `PetscFEKokkosMapsDestroy()`
@*/
PetscErrorCode PetscFEKokkosCreateMaps(DM dm, PetscFEKokkosMaps *maps)
{
  PetscSection section, globalSection;
  PetscInt     cStart, cEnd, Ne, Nb, Nc, totDim, num_dof;
  PetscDS      ds;
  PetscFE      fe;
  PetscInt     dim;
  DM           plex; /* adapted DMPlex view (may == dm for plain DMPlex) */
  PetscBool    isPlex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(maps, 2);
  PetscCall(DMGetDimension(dm, &dim));
  maps->local_dof = 0; /* will be set below */

  /* If dm is a DMForest (p4est), convert to the adapted DMPlex so that
   * DMPlexGetHeightStratum / DMPlexGetClosureIndices work correctly. */
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isPlex));
  if (isPlex) {
    plex = dm;
    PetscCall(PetscObjectReference((PetscObject)plex));
  } else PetscCall(DMConvert(dm, DMPLEX, &plex));

  /* Get local and global sections from the adapted plex */
  PetscCall(DMGetLocalSection(plex, &section));
  PetscCall(DMGetGlobalSection(plex, &globalSection));

  /* Cell range from the adapted plex */
  PetscCall(DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd));
  Ne = cEnd - cStart;

  /* Get Nb and totDim from the DS (on the original dm -- DS lives there) */
  PetscCall(DMGetDS(dm, &ds));
  {
    PetscInt Nf;
    PetscCall(PetscDSGetNumFields(ds, &Nf));
    PetscCheck(Nf == 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscFEKokkosCreateMaps supports single-field (Nf=1) only, got Nf=%" PetscInt_FMT, Nf);
  }
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  {
    PetscTabulation *T;
    PetscCall(PetscDSGetTabulation(ds, &T));
    Nb = T[0]->Nb;
    Nc = T[0]->Nc;
  }
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));

  /* Get local DOF count for the global Vec (used for bounds checking).
   * VecGetLocalSize returns the per-rank portion of the global Vec,
   * which is the correct bound for local row/column indices. */
  num_dof = 0;
  {
    Vec gvec;
    PetscCall(DMGetGlobalVector(dm, &gvec));
    PetscCall(VecGetLocalSize(gvec, &num_dof));
    PetscCall(DMRestoreGlobalVector(dm, &gvec));
  }

  /* Get local DOF count (including ghosts) for local Vec sizing */
  {
    Vec lvec;
    PetscCall(DMGetLocalVector(dm, &lvec));
    PetscCall(VecGetLocalSize(lvec, &maps->local_dof));
    PetscCall(DMRestoreLocalVector(dm, &lvec));
  }

  maps->num_elements        = Ne;
  maps->num_dof             = num_dof;
  maps->Nb                  = Nb;
  maps->totDim              = totDim;
  maps->num_reduced         = 0;
  maps->cached_Ne           = -1;
  maps->cached_Nq           = -1;
  maps->cached_Nc           = -1;
  maps->cached_dim          = -1;
  maps->cached_dE           = -1;
  maps->cached_totDim       = -1;
  maps->cached_numConstants = -1;
  maps->isAffine            = PETSC_FALSE;
  maps->geom_cached         = PETSC_FALSE;

  /* num_face: number of DOFs on a face edge = degree + 1 for 2D quads.
   * For tensor-product quads: Nb_scalar = (degree+1)^dim, so degree+1 = round(Nb_scalar^(1/dim)).
   * Nb is the total DOF count (= Nb_scalar * Nc for vector FE), so we divide by Nc first.
   * For simplices or dim==1 we fall back to Nb (no constraint expansion needed).
   * num_face computation assumes tensor-product elements; simplex not yet supported for constraints */
  {
    PetscInt       nf;
    const PetscInt Nb_scalar = Nb / Nc; /* scalar basis count (strip component multiplier) */
    if (dim >= 2) {
      /* degree+1 = round(Nb_scalar^(1/dim)) via integer search */
      nf = 1;
      while ((nf + 1) * (nf + 1) <= Nb_scalar) ++nf; /* works for dim==2 */
      if (dim == 3) {
        /* For dim==3: Nb_scalar = nf^3, so nf = round(Nb_scalar^(1/3)) */
        nf = (PetscInt)(PetscPowReal((PetscReal)Nb_scalar, 1.0 / 3.0) + 0.5);
      }
      /* num_face = (degree+1)^(dim-1) */
      PetscInt face_dofs = nf;
      for (PetscInt d = 1; d < dim - 1; ++d) face_dofs *= nf;
      maps->num_face = face_dofs;
    } else {
      /* dim < 2: no hanging nodes; for simplices num_reduced will be 0
       * so this value is unused in the COO scatter. */
      maps->num_face = Nb;
    }
    /* Error if num_face exceeds the compile-time stack array limit */
    PetscCheck(maps->num_face <= PETSCFE_KOKKOS_MAX_FACE, PETSC_COMM_SELF, PETSC_ERR_SUP, "num_face %" PetscInt_FMT " exceeds PETSCFE_KOKKOS_MAX_FACE %d; recompile with -DPETSCFE_KOKKOS_MAX_FACE=%d or reduce polynomial degree", maps->num_face, PETSCFE_KOKKOS_MAX_FACE,
               (int)maps->num_face);
    if (maps->num_face < 1) maps->num_face = 1;
  }

  /* Allocate host arrays (all created together -> freed together with PetscFree7) */
  PetscCall(PetscMalloc7(Ne * Nb, &maps->h_gIdx, Ne * Nb, &maps->h_lIdx, Ne * Nb, &maps->h_active_idx, Ne, &maps->h_Nb_active, Ne + 1, &maps->h_coo_elem_offsets, Ne, &maps->h_fullNb, Ne * (Nb + 1), &maps->h_coo_elem_point_offsets));

  /* Build h_lIdx: local DOF indices from the local section.
   * DMPlexGetClosureIndices(section, section) returns local indices that are
   * always non-negative (including ghost DOFs), so scatter to locF is safe.
   * Dirichlet DOFs are encoded as -(local_idx+1) in the local section;
   * we mark those as -1 to skip them in the scatter. */
  for (PetscInt e = 0; e < Ne; ++e) {
    PetscInt  numLoc = 0;
    PetscInt *locIdx = NULL;
    PetscCall(DMPlexGetClosureIndices(plex, section, section, cStart + e, PETSC_FALSE, &numLoc, &locIdx, NULL, NULL));
    /* locIdx has numLoc entries in closure order = totDim entries for no constraints.
     * Map closure position -> basis function index using the same ordering as h_gIdx. */
    for (PetscInt q = 0; q < Nb && q < numLoc; ++q) {
      const PetscInt li        = locIdx[q];
      maps->h_lIdx[e * Nb + q] = (li < 0) ? -1 : li;
    }
    /* Fill any remaining slots (shouldn't happen for conforming meshes) */
    for (PetscInt q = numLoc; q < Nb; ++q) maps->h_lIdx[e * Nb + q] = -1;
    PetscCall(DMPlexRestoreClosureIndices(plex, section, section, cStart + e, PETSC_FALSE, &numLoc, &locIdx, NULL, NULL));
  }

  /* Temporary storage for constraint maps (upper bound: Ne*Nb constraints) */
  const PetscInt max_reduced = Ne * Nb;
  const PetscInt num_face    = maps->num_face;
  PetscInt      *tmp_c_gid;
  PetscScalar   *tmp_c_scale;
  PetscCall(PetscMalloc2(max_reduced * num_face, &tmp_c_gid, max_reduced * num_face, &tmp_c_scale));
  /* Initialize to -1 / 0 */
  for (PetscInt i = 0; i < max_reduced * num_face; ++i) {
    tmp_c_gid[i]   = -1;
    tmp_c_scale[i] = 0.0;
  }

  /* Allocate probing element matrix (totDim x totDim) */
  PetscScalar *elMat;
  PetscCall(PetscMalloc1(totDim * totDim, &elMat));

  /* Probing loop: for each element e and basis function q,
   * set a unit element matrix and call DMPlexGetClosureIndices to discover
   * whether the DOF is unconstrained or constrained (hanging node). */
  for (PetscInt e = 0; e < Ne; ++e) {
    PetscInt fullNb = 0;

    for (PetscInt q = 0; q < Nb; ++q) {
      /* Set unit element matrix: elMat[q*totDim + q] = 1, rest zero */
      PetscCall(PetscArrayzero(elMat, totDim * totDim));
      elMat[q * totDim + q] = 1.0;

      PetscInt     numIndices = 0;
      PetscInt    *indices    = NULL;
      PetscScalar *valuesOrig = elMat;

      PetscCall(DMPlexGetClosureIndices(plex, section, globalSection, cStart + e, PETSC_TRUE, &numIndices, &indices, NULL, &elMat));

      /* Scan the returned elMat diagonal for this basis function.
       *
       * After DMPlexGetClosureIndices with useConstraints=TRUE, the element
       * matrix is expanded to numIndices x numIndices (numIndices >= totDim).
       * The probed basis function q produces exactly one non-zero diagonal
       * block in the expanded matrix:
       *   - Unconstrained DOF: diagonal ~ 1.0 at one position
       *   - Constrained DOF:   diagonal entries = c_i^2 at parent positions
       *
       * Classification:
       *   1. diag ~ 1.0 -> unconstrained active DOF
       *   2. 0 < diag < 1 -> hanging-node constrained DOF
       *   3. No non-zero diagonal found -> Dirichlet (fully constrained away)
       *
       * IMPORTANT: Do NOT use indices[f] < 0 to classify as Dirichlet.
       * A constrained DOF may have its first parent at a Dirichlet index,
       * but the DOF itself is a hanging node, not Dirichlet.
       */
      PetscBool found = PETSC_FALSE;
      /* NOTE: f is intentionally advanced inside the constraint branch (do-while)
       * to skip past expanded constraint entries in the returned element matrix. */
      for (PetscInt f = 0; f < numIndices && !found; ++f) {
        const PetscReal diag = PetscRealPart(elMat[f * numIndices + f]);
        if (PetscAbs(diag) > PETSC_MACHINE_EPSILON) {
          found = PETSC_TRUE;
          if (PetscAbs(diag - 1.0) < PETSC_MACHINE_EPSILON) {
            /* Unconstrained active DOF: diagonal ~ 1.0.
             * indices[f] gives the global index (may be negative for Dirichlet). */
            if (indices[f] < 0) {
              maps->h_gIdx[e * Nb + q] = (PetscFEKokkosIdx)(-1);
            } else {
              maps->h_gIdx[e * Nb + q] = (PetscFEKokkosIdx)indices[f];
              fullNb++;
            }
          } else {
            /* Hanging-node constrained DOF: 0 < diagonal < 1.
             * Extract constraint weights by summing rows of the outer product. */
            const PetscInt ff        = f;
            const PetscInt idx       = maps->num_reduced;
            maps->h_gIdx[e * Nb + q] = -(PetscFEKokkosIdx)(idx + 1);

            PetscInt jj = 0;
            do {
              /* Sum row ff of the outer product to recover the weight */
              PetscScalar sc = 0.0;
              for (PetscInt ii = 0; ii < num_face; ++ii) {
                if (ff + ii < numIndices) sc += PetscRealPart(elMat[f * numIndices + ff + ii]);
              }
              tmp_c_scale[idx * num_face + jj] = sc;
              if (PetscRealPart(sc) == 0.0 || indices[f] < 0) {
                tmp_c_gid[idx * num_face + jj] = -1;
              } else {
                tmp_c_gid[idx * num_face + jj] = indices[f];
                fullNb++;
              }
              ++jj;
              ++f;
            } while (jj < num_face && f < numIndices);
            /* Fill remaining slots */
            while (jj < num_face) {
              tmp_c_scale[idx * num_face + jj] = 0.0;
              tmp_c_gid[idx * num_face + jj]   = -1;
              ++jj;
            }

            maps->num_reduced++;
            PetscCheck(maps->num_reduced <= max_reduced, PETSC_COMM_SELF, PETSC_ERR_PLIB, "num_reduced %" PetscInt_FMT " exceeds max %" PetscInt_FMT, maps->num_reduced, max_reduced);
          }
        }
      }
      /* If no diagonal found (fully zeroed by constraint), mark as Dirichlet */
      if (!found) maps->h_gIdx[e * Nb + q] = (PetscFEKokkosIdx)(-1);

      /* Restore closure indices.
       * FRAGILE: DMPlexRestoreClosureIndices may reallocate elMat internally
       * (when expanding for constraints).  After Restore, elMat may point to
       * freed memory.  We reset it to valuesOrig (our PetscMalloc'd buffer)
       * which is guaranteed to remain valid.  This relies on the DMPlex
       * contract that RestoreClosureIndices does NOT free the user's original
       * buffer -- only its own internal work array.  If this contract ever
       * changes, this code will double-free.  A safer approach would be to
       * PetscMemcpy into a stable buffer rather than swapping pointers. */
      PetscCall(DMPlexRestoreClosureIndices(plex, section, globalSection, cStart + e, PETSC_TRUE, &numIndices, &indices, NULL, &elMat));
      elMat = valuesOrig;
    } /* basis q */

    maps->h_fullNb[e] = fullNb;

    /* Build coo_elem_point_offsets[e*(Nb+1)+b]: prefix sum of expanded rows */
    maps->h_coo_elem_point_offsets[e * (Nb + 1) + 0] = 0;
    for (PetscInt b = 0; b < Nb; ++b) {
      PetscInt               nr   = 0;
      const PetscFEKokkosIdx gidx = maps->h_gIdx[e * Nb + b];
      if (gidx >= 0) {
        nr = 1;
      } else if (gidx < -1) {
        /* Constrained: count valid parent DOFs */
        const PetscInt cidx = -(PetscInt)gidx - 1;
        for (PetscInt qq = 0; qq < num_face; ++qq) {
          if (tmp_c_gid[cidx * num_face + qq] >= 0) ++nr;
        }
      }
      /* gidx == -1 means Dirichlet: nr = 0 */
      maps->h_coo_elem_point_offsets[e * (Nb + 1) + b + 1] = maps->h_coo_elem_point_offsets[e * (Nb + 1) + b] + nr;
    }
  } /* element e */

  PetscCall(PetscFree(elMat));

  /* Copy constraint maps to final host arrays (created together -> freed together with PetscFree2) */
  PetscCall(PetscMalloc2(maps->num_reduced * num_face, &maps->h_c_maps_gid, maps->num_reduced * num_face, &maps->h_c_maps_scale));
  for (PetscInt i = 0; i < maps->num_reduced * num_face; ++i) {
    maps->h_c_maps_gid[i]   = tmp_c_gid[i];
    maps->h_c_maps_scale[i] = tmp_c_scale[i];
  }

  PetscCall(PetscFree2(tmp_c_gid, tmp_c_scale));

  /* Build active_idx, Nb_active, and coo_elem_offsets */
  maps->h_coo_elem_offsets[0] = 0;
  for (PetscInt e = 0; e < Ne; ++e) {
    PetscInt cnt = 0;
    for (PetscInt b = 0; b < Nb; ++b) {
      /* active_idx: -1 for Dirichlet (gidx == -1), sequential otherwise */
      if (maps->h_gIdx[e * Nb + b] == -1) {
        maps->h_active_idx[e * Nb + b] = -1;
      } else {
        maps->h_active_idx[e * Nb + b] = cnt++;
      }
    }
    maps->h_Nb_active[e] = cnt;
    /* COO size uses fullNb (expanded) not Nb_active */
    const PetscInt fNb              = maps->h_fullNb[e];
    maps->h_coo_elem_offsets[e + 1] = maps->h_coo_elem_offsets[e] + fNb * fNb;
  }
  maps->coo_size = (PetscCount)maps->h_coo_elem_offsets[Ne];

  /* Release the adapted plex reference */
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEKokkosStageMaps - Deep-copy host assembly maps to device Kokkos Views

  Not Collective

  Input Parameters:
+ maps - `PetscFEKokkosMaps` struct previously populated by `PetscFEKokkosCreateMaps()`
- dm   - the `DM` (used to query tabulation data)

  Level: intermediate

  Notes:
  Must be called after `PetscFEKokkosCreateMaps()` and before any GPU kernel launch.
  Also stages the basis function tables (B, D) and quadrature weights to device.

  For most users, `PetscFEKokkosSetUp()` is the preferred entry point, which calls this function
  internally.

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosSetUp()`, `PetscFEKokkosCreateMaps()`, `PetscFEKokkosPreallocateCOO()`,
          `PetscFEKokkosSetUpGeometry()`, `PetscFEKokkosMapsDestroy()`
@*/
PetscErrorCode PetscFEKokkosStageMaps(PetscFEKokkosMaps *maps, DM dm)
{
  PetscFunctionBegin;
  PetscAssertPointer(maps, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);

  const PetscInt Ne       = maps->num_elements;
  const PetscInt Nb       = maps->Nb;
  const PetscInt num_face = maps->num_face;

  /* gIdx -- h_gIdx is NULL when Ne==0; skip unmanaged View construction in that case */
  maps->d_gIdx = Kokkos::View<PetscFEKokkosIdx *>("fekokkos_coo_gIdx", Ne * Nb);
  if (maps->h_gIdx) {
    Kokkos::View<PetscFEKokkosIdx *, Kokkos::HostSpace> hv(maps->h_gIdx, Ne * Nb);
    Kokkos::deep_copy(maps->d_gIdx, hv);
  }

  /* lIdx */
  maps->d_lIdx = Kokkos::View<PetscInt *>("fekokkos_coo_lIdx", Ne * Nb);
  if (maps->h_lIdx) {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_lIdx, Ne * Nb);
    Kokkos::deep_copy(maps->d_lIdx, hv);
  }

  /* active_idx */
  maps->d_active_idx = Kokkos::View<PetscInt *>("fekokkos_coo_active_idx", Ne * Nb);
  if (maps->h_active_idx) {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_active_idx, Ne * Nb);
    Kokkos::deep_copy(maps->d_active_idx, hv);
  }

  /* Nb_active */
  maps->d_Nb_active = Kokkos::View<PetscInt *>("fekokkos_coo_Nb_active", Ne);
  if (maps->h_Nb_active) {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_Nb_active, Ne);
    Kokkos::deep_copy(maps->d_Nb_active, hv);
  }

  /* coo_elem_offsets: size Ne+1, so h_coo_elem_offsets is non-NULL even when Ne==0 */
  maps->d_coo_elem_offsets = Kokkos::View<PetscInt *>("fekokkos_coo_offsets", Ne + 1);
  {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_coo_elem_offsets, Ne + 1);
    Kokkos::deep_copy(maps->d_coo_elem_offsets, hv);
  }

  /* Stage constraint maps to device */
  const PetscInt nr = maps->num_reduced;
  if (nr > 0) {
    maps->d_c_maps_gid = Kokkos::View<PetscInt *>("fekokkos_c_maps_gid", nr * num_face);
    {
      Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_c_maps_gid, nr * num_face);
      Kokkos::deep_copy(maps->d_c_maps_gid, hv);
    }
    maps->d_c_maps_scale = Kokkos::View<PetscScalar *>("fekokkos_c_maps_scale", nr * num_face);
    {
      Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv(maps->h_c_maps_scale, nr * num_face);
      Kokkos::deep_copy(maps->d_c_maps_scale, hv);
    }
  } else {
    /* No constraints: create empty Views (size 1 to avoid zero-size allocation) */
    maps->d_c_maps_gid   = Kokkos::View<PetscInt *>("fekokkos_c_maps_gid", 1);
    maps->d_c_maps_scale = Kokkos::View<PetscScalar *>("fekokkos_c_maps_scale", 1);
  }

  /* coo_elem_point_offsets -- h_coo_elem_point_offsets is NULL when Ne==0 */
  maps->d_coo_elem_point_offsets = Kokkos::View<PetscInt *>("fekokkos_coo_pt_offsets", Ne * (Nb + 1));
  if (maps->h_coo_elem_point_offsets) {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_coo_elem_point_offsets, Ne * (Nb + 1));
    Kokkos::deep_copy(maps->d_coo_elem_point_offsets, hv);
  }

  /* fullNb */
  maps->d_fullNb = Kokkos::View<PetscInt *>("fekokkos_fullNb", Ne);
  if (maps->h_fullNb) {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_fullNb, Ne);
    Kokkos::deep_copy(maps->d_fullNb, hv);
  }

  /* Stage static FE tabulation data (B, D, w) to device.
   * These are constant for a given FE space and only need to be
   * copied once at setup time. */
  {
    PetscDS          ds;
    PetscFE          fe;
    PetscTabulation *T;
    PetscQuadrature  quad;
    const PetscReal *quadWeights;
    PetscInt         Nq_tab, Nb_tab, Nc_tab, dim_tab, qdim, qNc;

    PetscCall(DMGetDS(dm, &ds));
    {
      PetscInt Nf;
      PetscCall(PetscDSGetNumFields(ds, &Nf));
      PetscCheck(Nf == 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscFEKokkosStageMaps supports single-field (Nf=1) only, got Nf=%" PetscInt_FMT, Nf);
    }
    PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
    PetscCall(PetscDSGetTabulation(ds, &T));
    PetscCall(PetscFEGetQuadrature(fe, &quad));
    PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &Nq_tab, NULL, &quadWeights));
    PetscCall(PetscFEGetSpatialDimension(fe, &dim_tab));
    Nb_tab = T[0]->Nb;
    Nc_tab = T[0]->Nc;

    const PetscInt nB = Nq_tab * Nb_tab * Nc_tab;
    const PetscInt nD = Nq_tab * Nb_tab * Nc_tab * dim_tab;

    maps->d_B = Kokkos::View<PetscReal *>("fekokkos_cached_B", nB);
    maps->d_D = Kokkos::View<PetscReal *>("fekokkos_cached_D", nD);
    maps->d_w = Kokkos::View<PetscReal *>("fekokkos_cached_w", Nq_tab);
    {
      Kokkos::View<PetscReal *, Kokkos::HostSpace> h_B(const_cast<PetscReal *>(T[0]->T[0]), nB);
      Kokkos::View<PetscReal *, Kokkos::HostSpace> h_D(const_cast<PetscReal *>(T[0]->T[1]), nD);
      Kokkos::View<PetscReal *, Kokkos::HostSpace> h_w(const_cast<PetscReal *>(quadWeights), Nq_tab);
      Kokkos::deep_copy(maps->d_B, h_B);
      Kokkos::deep_copy(maps->d_D, h_D);
      Kokkos::deep_copy(maps->d_w, h_w);
    }
    maps->cached_Nq  = Nq_tab;
    maps->cached_Nc  = Nc_tab;
    maps->cached_dim = dim_tab;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEKokkosEnsureDynamicViews - Reallocate cached dynamic device Views only when sizes change

  Not Collective

  Input Parameters:
+ maps         - `PetscFEKokkosMaps` struct
. Ne           - number of elements
. Nq           - number of quadrature points per element
. dE           - embedding dimension
. totDim       - total DOFs per element
- numConstants - number of PDE constants

  Level: developer

  Notes:
  Called internally at the start of each residual/Jacobian evaluation. Reallocates device Views
  for invJ, detJ, coords, coeff, elemVec, elemMat, coo_vals, and constants only when the
  corresponding sizes have changed since the last call.

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosSetUp()`, `PetscFEKokkosCreateMaps()`, `PetscFEKokkosStageMaps()`
@*/
PetscErrorCode PetscFEKokkosEnsureDynamicViews(PetscFEKokkosMaps *maps, PetscInt Ne, PetscInt Nq, PetscInt dE, PetscInt totDim, PetscInt numConstants)
{
  PetscFunctionBegin;
  if (Ne != maps->cached_Ne || Nq != maps->cached_Nq || dE != maps->cached_dE || totDim != maps->cached_totDim) {
    const PetscInt nInvJ   = Ne * Nq * dE * dE;
    const PetscInt nDetJ   = Ne * Nq;
    const PetscInt nCoords = Ne * Nq * dE;
    const PetscInt nCoeff  = Ne * totDim;
    const PetscInt nEV     = Ne * totDim;
    const PetscInt nEM     = Ne * totDim * totDim;

    maps->d_invJ     = Kokkos::View<PetscReal *>("fekokkos_cached_invJ", nInvJ);
    maps->d_detJ     = Kokkos::View<PetscReal *>("fekokkos_cached_detJ", nDetJ);
    maps->d_coords   = Kokkos::View<PetscReal *>("fekokkos_cached_coords", nCoords);
    maps->d_coeff    = Kokkos::View<PetscScalar *>("fekokkos_cached_coeff", nCoeff);
    maps->d_elemVec  = Kokkos::View<PetscScalar *>("fekokkos_cached_elemVec", nEV);
    maps->d_elemMat  = Kokkos::View<PetscScalar *>("fekokkos_cached_elemMat", nEM);
    maps->d_coo_vals = Kokkos::View<PetscScalar *>("fekokkos_cached_coo_vals", maps->coo_size > 0 ? maps->coo_size : 1);

    maps->cached_Ne     = Ne;
    maps->cached_dE     = dE;
    maps->cached_totDim = totDim;
  }
  if (numConstants != maps->cached_numConstants) {
    maps->d_constants         = Kokkos::View<PetscScalar *>("fekokkos_cached_constants", numConstants > 0 ? numConstants : 1);
    maps->cached_numConstants = numConstants;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEKokkosPreallocateCOO - Build COO row/column index arrays and call `MatSetPreallocationCOO()` for GPU Jacobian assembly

  Collective

  Input Parameters:
+ maps - `PetscFEKokkosMaps` struct previously populated by `PetscFEKokkosCreateMaps()`
- J    - matrix to preallocate; must already exist (e.g., from `DMCreateMatrix()`)

  Level: intermediate

  Notes:
  Builds the COO row/column arrays from the host global index map (with constraint expansion
  for boundary DOFs) and calls `MatSetPreallocationCOO()`. The matrix must be of type
  `MATAIJKOKKOS` so that `MatSetValuesCOO()` can accept device pointers during Jacobian assembly.

  For most users, `PetscFEKokkosSetUp()` is the preferred entry point, which calls this function
  internally.

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosSetUp()`, `PetscFEKokkosCreateMaps()`, `PetscFEKokkosStageMaps()`,
          `PetscFEKokkosMapsDestroy()`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`
@*/
PetscErrorCode PetscFEKokkosPreallocateCOO(PetscFEKokkosMaps *maps, Mat J)
{
  const PetscInt   Ne       = maps->num_elements;
  const PetscInt   Nb       = maps->Nb;
  const PetscCount coo_size = maps->coo_size;
  const PetscInt   num_face = maps->num_face;
  PetscInt        *coo_i, *coo_j;

  PetscFunctionBegin;
  PetscAssertPointer(maps, 1);
  PetscValidHeaderSpecific(J, MAT_CLASSID, 2);
  {
    PetscBool isKokkosMat;
    PetscCall(PetscObjectTypeCompareAny((PetscObject)J, &isKokkosMat, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, ""));
    PetscCheck(isKokkosMat, PetscObjectComm((PetscObject)J), PETSC_ERR_SUP, "PetscFEKokkosPreallocateCOO requires MATAIJKOKKOS; use -dm_mat_type aijkokkos");
  }
  PetscCall(PetscMalloc2(coo_size, &coo_i, coo_size, &coo_j));
  /* Initialize to -1 (unused slots).
   * MatSetPreallocationCOO / MatSetValuesCOO silently ignore entries where
   * either row or column index is -1, so unused slots in the COO arrays are
   * harmless.  This avoids a separate compaction pass. */
  for (PetscCount k = 0; k < coo_size; ++k) coo_i[k] = coo_j[k] = -1;

  for (PetscInt e = 0; e < Ne; ++e) {
    const PetscInt fullNb = maps->h_fullNb[e];
    const PetscInt off    = maps->h_coo_elem_offsets[e];

    for (PetscInt b = 0; b < Nb; ++b) {
      /* Expand row b */
      PetscInt               nr = 0;
      PetscInt               rows[PETSCFE_KOKKOS_MAX_FACE];
      const PetscFEKokkosIdx gidx_b = maps->h_gIdx[e * Nb + b];
      if (gidx_b >= 0) {
        nr      = 1;
        rows[0] = (PetscInt)gidx_b;
      } else if (gidx_b < -1) {
        const PetscInt cidx = -(PetscInt)gidx_b - 1;
        for (PetscInt q = 0; q < num_face; ++q) {
          if (maps->h_c_maps_gid[cidx * num_face + q] >= 0) rows[nr++] = maps->h_c_maps_gid[cidx * num_face + q];
        }
      }
      /* gidx_b == -1: Dirichlet, nr = 0, skip */
      const PetscInt pt_off_b = maps->h_coo_elem_point_offsets[e * (Nb + 1) + b];

      for (PetscInt b2 = 0; b2 < Nb; ++b2) {
        /* Expand column b2 */
        PetscInt               nc = 0;
        PetscInt               cols[PETSCFE_KOKKOS_MAX_FACE];
        const PetscFEKokkosIdx gidx_b2 = maps->h_gIdx[e * Nb + b2];
        if (gidx_b2 >= 0) {
          nc      = 1;
          cols[0] = (PetscInt)gidx_b2;
        } else if (gidx_b2 < -1) {
          const PetscInt cidx = -(PetscInt)gidx_b2 - 1;
          for (PetscInt q = 0; q < num_face; ++q) {
            if (maps->h_c_maps_gid[cidx * num_face + q] >= 0) cols[nc++] = maps->h_c_maps_gid[cidx * num_face + q];
          }
        }
        const PetscInt pt_off_b2 = maps->h_coo_elem_point_offsets[e * (Nb + 1) + b2];

        /* COO base offset for this (b, b2) pair.
         * Formula:
         *   idx0 = off + fullNb * pt_off_b + nr * pt_off_b2
         *
         * NOTE: This formula is correct ONLY when num_reduced == 0
         * (no hanging-node constraints), because in that case every DOF
         * has nr == 1 and nc == 1, so the formula reduces to simple
         * row-major indexing into a fullNb x fullNb block.  For meshes
         * with constrained DOFs (nr or nc > 1 and varying across DOFs),
         * the column stride nr*pt_off_b2 is dimensionally inconsistent
         * and would scatter into wrong matrix entries.  Constraint
         * support is not yet implemented; the residual/Jacobian
         * dispatchers guard against it with PetscCheck(num_reduced==0).
         * When constraint support is added, this index formula must be
         * revised to use a consistent stride (e.g., fullNb * pt_off_b2
         * or a per-element prefix-sum table). */
        const PetscInt idx0 = off + fullNb * pt_off_b + nr * pt_off_b2;

        for (PetscInt p = 0; p < nr; ++p) {
          for (PetscInt d = 0; d < nc; ++d) {
            coo_i[idx0 + p * nc + d] = rows[p];
            coo_j[idx0 + p * nc + d] = cols[d];
          }
        }
      }
    }
  }

  PetscCall(MatSetPreallocationCOO(J, coo_size, coo_i, coo_j));
  PetscCall(PetscFree2(coo_i, coo_j));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEKokkosResetGeometry - Release cached element geometry and invalidate device geometry Views

  Not Collective

  Input Parameter:
. maps - `PetscFEKokkosMaps` struct

  Level: intermediate

  Notes:
  Releases cached element geometry objects (fullGeom, chunkGeom, cellIS) and clears the device
  geometry Views (d_invJ, d_detJ, d_coords). Sets `geom_cached` to `PETSC_FALSE` so that
  `PetscFEKokkosSetUpGeometry()` will rebuild on the next call.

  Safe to call when geometry is not cached (no-op). Called automatically by
  `PetscFEKokkosMapsDestroy()`. Call explicitly before `PetscFEKokkosSetUpGeometry()` when the
  mesh changes (e.g., after AMR refinement or mesh motion).

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosSetUpGeometry()`, `PetscFEKokkosSetUp()`, `PetscFEKokkosMapsDestroy()`
@*/
PetscErrorCode PetscFEKokkosResetGeometry(PetscFEKokkosMaps *maps)
{
  PetscFunctionBegin;
  PetscAssertPointer(maps, 1);
  if (maps->geom_cached) {
    PetscCall(PetscFEGeomRestoreChunk(maps->cached_fullGeom, maps->cached_cStart, maps->cached_cEnd, &maps->cached_chunkGeom));
    PetscCall(PetscFEGeomDestroy(&maps->cached_fullGeom));
    PetscCall(ISDestroy(&maps->cached_cellIS));
    maps->cached_chunkGeom = NULL;
    maps->cached_fullGeom  = NULL;
    maps->cached_cellIS    = NULL;
    maps->geom_cached      = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEKokkosMapsDestroy - Free host arrays and cached geometry in a `PetscFEKokkosMaps` struct

  Not Collective

  Input Parameter:
. maps - `PetscFEKokkosMaps` struct to destroy

  Level: beginner

  Notes:
  Frees all host-allocated arrays (global indices, constraint maps, COO offsets) and calls
  `PetscFEKokkosResetGeometry()` to release cached geometry. Device Kokkos Views are
  reference-counted and freed automatically when the struct goes out of scope.

  Must be called when the `PetscFEKokkosMaps` is no longer needed, typically after the last
  `SNESSolve()`.

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosSetUp()`, `PetscFEKokkosResetGeometry()`
@*/
PetscErrorCode PetscFEKokkosMapsDestroy(PetscFEKokkosMaps **maps)
{
  PetscFunctionBegin;
  if (!*maps) PetscFunctionReturn(PETSC_SUCCESS);
  /* Release cached geometry objects via the reset function */
  PetscCall(PetscFEKokkosResetGeometry(*maps));
  PetscCall(PetscFree7((*maps)->h_gIdx, (*maps)->h_lIdx, (*maps)->h_active_idx, (*maps)->h_Nb_active, (*maps)->h_coo_elem_offsets, (*maps)->h_fullNb, (*maps)->h_coo_elem_point_offsets));
  PetscCall(PetscFree2((*maps)->h_c_maps_gid, (*maps)->h_c_maps_scale));
  /* Reset device Views to empty (releases Kokkos reference count) */
  (*maps)->d_gIdx                   = Kokkos::View<PetscFEKokkosIdx *>();
  (*maps)->d_lIdx                   = Kokkos::View<PetscInt *>();
  (*maps)->d_active_idx             = Kokkos::View<PetscInt *>();
  (*maps)->d_Nb_active              = Kokkos::View<PetscInt *>();
  (*maps)->d_coo_elem_offsets       = Kokkos::View<PetscInt *>();
  (*maps)->d_c_maps_gid             = Kokkos::View<PetscInt *>();
  (*maps)->d_c_maps_scale           = Kokkos::View<PetscScalar *>();
  (*maps)->d_coo_elem_point_offsets = Kokkos::View<PetscInt *>();
  (*maps)->d_fullNb                 = Kokkos::View<PetscInt *>();
  /* Reset cached device Views (releases Kokkos reference counts) */
  (*maps)->d_B         = Kokkos::View<PetscReal *>();
  (*maps)->d_D         = Kokkos::View<PetscReal *>();
  (*maps)->d_w         = Kokkos::View<PetscReal *>();
  (*maps)->d_invJ      = Kokkos::View<PetscReal *>();
  (*maps)->d_detJ      = Kokkos::View<PetscReal *>();
  (*maps)->d_coords    = Kokkos::View<PetscReal *>();
  (*maps)->d_coeff     = Kokkos::View<PetscScalar *>();
  (*maps)->d_constants = Kokkos::View<PetscScalar *>();
  (*maps)->d_elemVec   = Kokkos::View<PetscScalar *>();
  (*maps)->d_elemMat   = Kokkos::View<PetscScalar *>();
  (*maps)->d_coo_vals  = Kokkos::View<PetscScalar *>();
  delete *maps;
  *maps = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEKokkosSetUpGeometry - Build element geometry on host and upload to device

  Not Collective

  Input Parameters:
+ dm   - the `DM` with `PetscFE` discretization and coordinate field attached
- maps - `PetscFEKokkosMaps` struct already initialized by `PetscFEKokkosSetUp()`

  Level: intermediate

  Notes:
  Builds element geometry on the host via `DMFieldCreateFEGeom()`, uploads invJ, detJ, and
  physical coordinates to device Kokkos Views, and caches the result. Idempotent: a second
  call is a no-op unless `PetscFEKokkosResetGeometry()` has been called first.

  Called automatically by `PetscFEKokkosSetUp()` for the common static-mesh case. Call
  explicitly after `PetscFEKokkosResetGeometry()` when the mesh changes (e.g., after AMR
  refinement or mesh motion).

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosSetUp()`, `PetscFEKokkosResetGeometry()`, `PetscFEKokkosMapsDestroy()`
@*/
PetscErrorCode PetscFEKokkosSetUpGeometry(DM dm, PetscFEKokkosMaps *maps)
{
  PetscDS            ds;
  PetscFE            fe;
  PetscTabulation   *T;
  PetscQuadrature    quad;
  DMField            coordField;
  IS                 cellIS;
  PetscFEGeom       *fullGeom  = NULL;
  PetscFEGeom       *chunkGeom = NULL;
  PetscInt           depth, cStart, cEnd, totDim;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           Nq, Nc, dim, dE, qdim, qNc;
  PetscInt           numConstants;
  const PetscScalar *constants;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(maps, 2);
  /* Idempotent: skip if geometry already on device */
  if (maps->geom_cached) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetConstants(ds, &numConstants, &constants));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(PetscDSGetTabulation(ds, &T));
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));

  Nc = T[0]->Nc;

  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetStratumIS(dm, "depth", depth, &cellIS));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  const PetscInt Ne = cEnd - cStart;

  /* Build element geometry on host */
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, PETSC_FEGEOM_BASIC, &fullGeom));
  PetscCall(PetscFEGeomGetChunk(fullGeom, cStart, cEnd, &chunkGeom));
  dE = chunkGeom->dimEmbed;

  /* Runtime bounds check: Nc and dE must fit in the kernel's stack arrays */
  PetscCheck(Nc <= 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Nc %" PetscInt_FMT " exceeds PETSCFE_KOKKOS_MAX_NC 3", (PetscInt)Nc);
  PetscCheck(dE <= 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "dE %" PetscInt_FMT " exceeds PETSCFE_KOKKOS_MAX_DE 3", (PetscInt)dE);

  /* Ensure dynamic Views are allocated (invJ, detJ, coords, coeff, elemVec, elemMat, coo_vals) */
  PetscCall(PetscFEKokkosEnsureDynamicViews(maps, Ne, Nq, dE, totDim, numConstants));

  /* Sizes needed for geometry staging */
  const PetscInt nInvJ_g   = Ne * Nq * dE * dE;
  const PetscInt nDetJ_g   = Ne * Nq;
  const PetscInt nCoords_g = Ne * Nq * dE;

  /* Stage geometry to device: affine-optimized path reduces H->D transfer by Nq */
  maps->isAffine              = chunkGeom->isAffine;
  const PetscBool isAffine_su = maps->isAffine;
  if (isAffine_su) {
    /* Affine: copy compact geometry [Ne * dE * dE] invJ + [Ne] detJ to device,
     * then expand on-device via a small Kokkos kernel (GPU replication is
     * much faster than host replication + larger H->D transfer). */
    const PetscInt nInvJ_compact = Ne * dE * dE;
    const PetscInt nDetJ_compact = Ne;

    PetscReal *h_invJ_compact, *h_detJ_compact, *h_coords_buf;
    PetscCall(PetscMalloc3(nInvJ_compact, &h_invJ_compact, nDetJ_compact, &h_detJ_compact, nCoords_g, &h_coords_buf));

    /* Fill compact buffers: one invJ and one detJ per element */
    const PetscInt Np_su = chunkGeom->numPoints;
    for (PetscInt e = 0; e < Ne; ++e) {
      for (PetscInt i = 0; i < dE * dE; ++i) h_invJ_compact[e * dE * dE + i] = chunkGeom->invJ[e * Np_su * dE * dE + i];
      h_detJ_compact[e] = chunkGeom->detJ[e * Np_su];
    }

    /* Compute physical coords for affine elements */
    for (PetscInt e = 0; e < Ne; ++e) {
      const PetscReal *v0_e = &chunkGeom->v[e * Np_su * dE];
      const PetscReal *J_e  = &chunkGeom->J[e * Np_su * dE * dE];
      for (PetscInt q = 0; q < Nq; ++q) PetscFEKokkosCoordRefToReal(dE, dim, chunkGeom->xi, v0_e, J_e, &quadPoints[q * dim], &h_coords_buf[(e * Nq + q) * dE]);
    }

    /* Copy compact invJ/detJ to temporary device Views */
    Kokkos::View<PetscReal *> d_invJ_compact("su_invJ_compact", nInvJ_compact);
    Kokkos::View<PetscReal *> d_detJ_compact("su_detJ_compact", nDetJ_compact);
    {
      Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_invJ(h_invJ_compact, nInvJ_compact);
      Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_detJ(h_detJ_compact, nDetJ_compact);
      Kokkos::deep_copy(d_invJ_compact, hv_invJ);
      Kokkos::deep_copy(d_detJ_compact, hv_detJ);
    }

    /* Copy coords to cached device View */
    {
      Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_coords(h_coords_buf, nCoords_g);
      Kokkos::deep_copy(maps->d_coords, hv_coords);
    }
    PetscCall(PetscFree3(h_invJ_compact, h_detJ_compact, h_coords_buf));

    /* Expand on device: replicate single invJ/detJ across all Nq slots */
    auto su_d_invJ_expand = maps->d_invJ;
    auto su_d_detJ_expand = maps->d_detJ;
    Kokkos::parallel_for(
      "PetscFEKokkos_expand_affine_geom_su", Kokkos::RangePolicy<>(0, Ne), KOKKOS_LAMBDA(const PetscInt e) {
        for (PetscInt q = 0; q < Nq; ++q) {
          for (PetscInt i = 0; i < dE * dE; ++i) su_d_invJ_expand[(e * Nq + q) * dE * dE + i] = d_invJ_compact[e * dE * dE + i];
          su_d_detJ_expand[e * Nq + q] = d_detJ_compact[e];
        }
      });
    Kokkos::fence();
  } else {
    /* Non-affine: use PetscFEKokkosExpandGeometry as before */
    PetscReal *h_invJ_buf, *h_detJ_buf, *h_coords_buf;
    PetscCall(PetscMalloc3(nInvJ_g, &h_invJ_buf, nDetJ_g, &h_detJ_buf, nCoords_g, &h_coords_buf));
    PetscCall(PetscFEKokkosExpandGeometry(Ne, Nq, dim, dE, chunkGeom, quadPoints, h_invJ_buf, h_detJ_buf, h_coords_buf));
    {
      Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_invJ(h_invJ_buf, nInvJ_g);
      Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_detJ(h_detJ_buf, nDetJ_g);
      Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_coords(h_coords_buf, nCoords_g);
      Kokkos::deep_copy(maps->d_invJ, hv_invJ);
      Kokkos::deep_copy(maps->d_detJ, hv_detJ);
      Kokkos::deep_copy(maps->d_coords, hv_coords);
    }
    PetscCall(PetscFree3(h_invJ_buf, h_detJ_buf, h_coords_buf));
  }

  /* Cache geometry objects, range, and dE for subsequent calls.
   * cached_cStart / cached_cEnd are passed back symmetrically to
   * PetscFEGeomRestoreChunk in PetscFEKokkosResetGeometry. */
  maps->cached_dE_geom   = dE;
  maps->cached_cStart    = cStart;
  maps->cached_cEnd      = cEnd;
  maps->cached_fullGeom  = fullGeom;
  maps->cached_chunkGeom = chunkGeom;
  maps->cached_cellIS    = cellIS;
  maps->geom_cached      = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEKokkosSetUp - Build assembly maps, stage to device, preallocate COO matrix, and build element geometry for GPU FEM assembly

  Collective

  Input Parameters:
+ dm   - the `DM` with `PetscFE` discretization attached (after `DMCreateDS()`)
. maps - pointer to a zero-initialized `PetscFEKokkosMaps` struct (stack or heap)
- J    - matrix to preallocate via `MatSetPreallocationCOO()`; must already exist (e.g., from `DMCreateMatrix()`)

  Level: beginner

  Notes:
  This is the primary entry point for setting up GPU-resident FEM assembly. Must be called after
  `SNESSetFromOptions()`, which triggers Kokkos initialization.

  Internally calls (in order)\:
  `PetscFEKokkosCreateMaps()`, `PetscFEKokkosStageMaps()`, `PetscFEKokkosPreallocateCOO()`,
  and `PetscFEKokkosSetUpGeometry()`.

  The matrix `J` must be of type `MATAIJKOKKOS` (pass `-dm_mat_type aijkokkos` on the command line).

.seealso: [](ch_fe), `PetscFE`, `PetscFEKokkosMapsDestroy()`, `PetscFEKokkosCreateMaps()`, `PetscFEKokkosStageMaps()`,
          `PetscFEKokkosPreallocateCOO()`, `PetscFEKokkosSetUpGeometry()`, `PetscFEKokkosResetGeometry()`,
          `MatSetPreallocationCOO()`, `MatSetValuesCOO()`
@*/
PetscErrorCode PetscFEKokkosSetUp(DM dm, PetscFEKokkosMaps *maps, Mat J)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(maps, 2);
  PetscValidHeaderSpecific(J, MAT_CLASSID, 3);
  PetscCall(PetscFEKokkosCreateMaps(dm, maps));
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscFEKokkosStageMaps(maps, dm));
  PetscCall(PetscFEKokkosPreallocateCOO(maps, J));
  PetscCall(PetscFEKokkosSetUpGeometry(dm, maps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DMPlexSNESComputeResidualFEM_Kokkos<f0, f1>

   GPU-resident SNES residual callback.  Drop-in replacement for
   DMPlexSNESComputeResidualFEM (the CPU path) when using COO assembly.

   Template parameters:
     f0 -- KOKKOS_INLINE_FUNCTION void(PETSC_POINT_ARGS, PetscScalar[])
          source term (zeroth-order residual)
     f1 -- KOKKOS_INLINE_FUNCTION void(PETSC_POINT_ARGS, PetscScalar[])
          flux term (first-order residual)

   ctx_ptr must point to a PetscFEKokkosMaps with maps already staged
   (i.e. PetscFEKokkosSetUp has been called).

   Algorithm:
     1. Zero F.
     2. Get local solution with BCs applied (host-sync guard for device Vecs).
     3. Get element geometry and coefficients.
     4. Stage tabulation, geometry, coefficients, and DS constants to device.
     5. Pass 1: PetscFEKokkosIntegrateResidualCell<f0,f1> -> d_elemVec.
     6. Pass 2: scatter d_elemVec -> locF via Kokkos::atomic_add (local indices).
     7. DMLocalToGlobal(locF, ADD_VALUES, F) -- MPI reduction for ghost DOFs. */
