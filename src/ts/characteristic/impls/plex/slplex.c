#include <petsc/private/characteristicimpl.h> /*I "petsccharacteristic.h" I*/
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscblaslapack.h>

typedef struct {
  /* Phase-space and potential DMs (set via CharacteristicSetVelocityInterpolation) */
  DM dmPS;  /* 2D phase-space DMPlex — hosts f(x,v) */
  DM dmPot; /* 1D potential DMPlex — hosts E(x) */

  /* Foot-particle swarm (built at SetUp time) */
  DM swFeet; /* DMSWARM_PIC swarm; cell DM = dmPS */

  /* Cached geometry (built at SetUp time) */
  PetscFEGeom *fegeomPS; /* Cell geometry for dmPS */

  /* Quadrature shared across trace and update (Issue 5 / Criterion 3) */
  PetscQuadrature quad_PS; /* Single rule for BackwardTrace + UpdateDF */

  /* Cached basis tabulation at quad points */
  PetscTabulation tab_PS; /* phi_b(xi_q): values only (K=0) */

  /* Mesh sizing (set by caller via AppCtx before SetUp) */
  PetscInt  Nx;      /* Number of x-cells */
  PetscInt  Nv;      /* Number of v-cells */
  PetscReal x_lower; /* Lower x-domain bound */
  PetscReal x_upper; /* Upper x-domain bound */
  PetscReal v_max;   /* Upper |v| bound */
} Characteristic_Plex;

/* ------------------------------------------------------------------ */
/* Helper: evaluate the E-field FE Vec at an arbitrary real x_real.   */
/* UNIFORM MESH ASSUMED (Criterion 2): O(1) integer cell lookup.      */
/* ------------------------------------------------------------------ */
static PetscErrorCode InterpolateEAt_Plex(Characteristic_Plex *plex, Vec E_local, PetscReal x_real, PetscReal *E_val)
{
  PetscReal    h;
  PetscInt     cel, cStart, cEnd;
  PetscScalar *e_cell;
  PetscInt     nDOF;
  PetscSection section;
  DM           dmPot = plex->dmPot;

  PetscFunctionBegin;
  h   = (plex->x_upper - plex->x_lower) / (PetscReal)plex->Nx;
  cel = (PetscInt)PetscFloorReal((x_real - plex->x_lower) / h);
  /* periodic wrap */
  cel = ((cel % plex->Nx) + plex->Nx) % plex->Nx;

  PetscCall(DMPlexGetHeightStratum(dmPot, 0, &cStart, &cEnd));
  cel += cStart;

  PetscCall(DMGetLocalSection(dmPot, &section));
  PetscCall(DMPlexVecGetClosure(dmPot, section, E_local, cel, &nDOF, &e_cell));

  /* Simple linear interpolation using the cell-local reference coordinate */
  if (nDOF == 1) {
    *E_val = PetscRealPart(e_cell[0]);
  } else {
    /* Map x_real to reference coord xi in [-1,1] */
    PetscReal x_left = plex->x_lower + (cel - cStart) * h;
    PetscReal xi     = 2.0 * (x_real - x_left) / h - 1.0;
    /* Linear Lagrange: phi_0 = (1-xi)/2, phi_1 = (1+xi)/2 */
    *E_val = PetscRealPart(e_cell[0]) * 0.5 * (1.0 - xi) + PetscRealPart(e_cell[1]) * 0.5 * (1.0 + xi);
  }

  PetscCall(DMPlexVecRestoreClosure(dmPot, section, E_local, cel, &nDOF, &e_cell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
static PetscErrorCode CharacteristicView_Plex(Characteristic c, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  CharacteristicPlex (DMPlex/DMSwarm BSL)\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
static PetscErrorCode CharacteristicDestroy_Plex(Characteristic c)
{
  Characteristic_Plex *plex = (Characteristic_Plex *)c->data;

  PetscFunctionBegin;
  if (plex->swFeet) PetscCall(DMDestroy(&plex->swFeet));
  if (plex->fegeomPS) PetscCall(PetscFEGeomDestroy(&plex->fegeomPS));
  if (plex->tab_PS) PetscCall(PetscTabulationDestroy(&plex->tab_PS));
  PetscCall(PetscFree(plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
static PetscErrorCode CharacteristicSetUp_Plex(Characteristic c)
{
  Characteristic_Plex *plex = (Characteristic_Plex *)c->data;
  DM                   dmPS = plex->dmPS;
  DM                   sw;
  PetscFE              fe;
  PetscDS              ds;
  DMField              coordField;
  IS                   cellIS;
  PetscInt             cStart, cEnd, Nq;
  const PetscReal     *xi_q;

  PetscFunctionBegin;
  PetscCheck(dmPS, PetscObjectComm((PetscObject)c), PETSC_ERR_ARG_WRONGSTATE, "dmPS not set. Call CharacteristicSetVelocityInterpolation() before CharacteristicSetUp().");

  /* --- Build DMSWARM_PIC swarm --- */
  PetscCall(DMCreate(PetscObjectComm((PetscObject)c), &sw));
  PetscCall(DMSetType(sw, DMSWARM));
  PetscCall(DMSetDimension(sw, 2));
  PetscCall(DMSwarmSetType(sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(sw, dmPS));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, DMSwarmPICField_coor, 2, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "f_interp", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "quad_weight", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "src_cell", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "src_dof", 1, PETSC_INT));
  PetscCall(DMSetFromOptions(sw));
  PetscCall(DMSetUp(sw));
  plex->swFeet = sw;

  /* --- Cache shared quadrature rule (Issue 5 / Criterion 3) --- */
  PetscCall(DMGetDS(dmPS, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(PetscFEGetQuadrature(fe, &plex->quad_PS));
  PetscCall(PetscQuadratureGetData(plex->quad_PS, NULL, NULL, &Nq, &xi_q, NULL));
  /* Criterion 3: enforce Nq >= 2 */
  PetscCheck(Nq >= 2, PetscObjectComm((PetscObject)c), PETSC_ERR_ARG_WRONG, "CharacteristicPlex requires Q >= 2 quadrature points per cell, got %" PetscInt_FMT, Nq);

  /* --- Cache basis tabulation at quad points --- */
  PetscCall(PetscFECreateTabulation(fe, 1, Nq, xi_q, 0 /*values only*/, &plex->tab_PS));

  /* --- Cache cell geometry for dmPS --- */
  PetscCall(DMPlexGetHeightStratum(dmPS, 0, &cStart, &cEnd));
  PetscCall(DMGetCoordinateField(dmPS, &coordField));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, cEnd - cStart, cStart, 1, &cellIS));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, plex->quad_PS, PETSC_FEGEOM_BASIC, &plex->fegeomPS));
  PetscCall(ISDestroy(&cellIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
static PetscErrorCode CharacteristicSolve_Plex(Characteristic c, PetscReal dt, Vec solution)
{
  Characteristic_Plex *plex  = (Characteristic_Plex *)c->data;
  DM                   dmPS  = plex->dmPS;
  DM                   dmPot = plex->dmPot;
  DM                   sw    = plex->swFeet;
  Vec                  E_local, f_local, f_new_local;
  PetscSection         section;
  PetscInt             cStart, cEnd, c_cell, Nq, Nb, dim;
  PetscReal           *feet_coords, *feet_weight;
  PetscInt            *feet_src_cell, *feet_src_dof;
  PetscReal           *feet_f_interp;
  const PetscReal     *xi_q, *w_q;
  PetscInt             Np, idx, n;

  PetscFunctionBegin;

  /* ---------------------------------------------------------------- */
  /* Part 1 — Strang-split backward trace (place foot particles)      */
  /* ---------------------------------------------------------------- */
  PetscCall(DMPlexGetHeightStratum(dmPS, 0, &cStart, &cEnd));
  PetscCall(DMGetDimension(dmPS, &dim));
  PetscCall(PetscQuadratureGetData(plex->quad_PS, NULL, NULL, &Nq, &xi_q, &w_q));
  Nb = plex->tab_PS->Nb;
  Np = (cEnd - cStart) * Nq;

  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&feet_coords));
  PetscCall(DMSwarmGetField(sw, "quad_weight", NULL, NULL, (void **)&feet_weight));
  PetscCall(DMSwarmGetField(sw, "src_cell", NULL, NULL, (void **)&feet_src_cell));
  PetscCall(DMSwarmGetField(sw, "src_dof", NULL, NULL, (void **)&feet_src_dof));

  /* Scatter E to local ghost vector for InterpolateEAt_Plex */
  PetscCall(DMGetLocalVector(dmPot, &E_local));
  PetscCall(DMGlobalToLocalBegin(dmPot, c->velocity, INSERT_VALUES, E_local));
  PetscCall(DMGlobalToLocalEnd(dmPot, c->velocity, INSERT_VALUES, E_local));

  idx = 0;
  for (c_cell = cStart; c_cell < cEnd; c_cell++) {
    PetscReal v0[2], J[4], invJ[4], detJ;
    PetscInt  q;

    PetscCall(DMPlexComputeCellGeometryFEM(dmPS, c_cell, NULL, v0, J, invJ, &detJ));

    for (q = 0; q < Nq; q++) {
      PetscReal x_real_q[2]; /* (x, v) real coords of quad point */
      PetscReal x0_pt, v0_pt, E0, v_half, x_foot, E_foot, v_foot;
      PetscInt  d;

      /* Reference-to-real map (2D) */
      for (d = 0; d < dim; d++) {
        x_real_q[d] = v0[d];
        PetscInt e;
        for (e = 0; e < dim; e++) x_real_q[d] += J[d * dim + e] * xi_q[q * dim + e];
      }
      x0_pt = x_real_q[0];
      v0_pt = x_real_q[1];

      /* Strang splitting: half v-kick, full x-drift, half v-kick */
      PetscCall(InterpolateEAt_Plex(plex, E_local, x0_pt, &E0));
      v_half = v0_pt - 0.5 * dt * E0;
      x_foot = x0_pt - dt * v_half;

      /* Periodic x wrap */
      while (x_foot < plex->x_lower) x_foot += (plex->x_upper - plex->x_lower);
      while (x_foot >= plex->x_upper) x_foot -= (plex->x_upper - plex->x_lower);

      PetscCall(InterpolateEAt_Plex(plex, E_local, x_foot, &E_foot));
      v_foot = v_half - 0.5 * dt * E_foot;

      /* v-boundary clamp (Dirichlet f = 0) */
      if (v_foot > plex->v_max) v_foot = plex->v_max;
      if (v_foot < -plex->v_max) v_foot = -plex->v_max;

      feet_coords[idx * 2]     = x_foot;
      feet_coords[idx * 2 + 1] = v_foot;
      feet_src_cell[idx]       = c_cell;
      feet_src_dof[idx]        = q;
      feet_weight[idx]         = w_q[q] * detJ;
      idx++;
    }
  }

  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&feet_coords));
  PetscCall(DMSwarmRestoreField(sw, "quad_weight", NULL, NULL, (void **)&feet_weight));
  PetscCall(DMSwarmRestoreField(sw, "src_cell", NULL, NULL, (void **)&feet_src_cell));
  PetscCall(DMSwarmRestoreField(sw, "src_dof", NULL, NULL, (void **)&feet_src_dof));
  PetscCall(DMRestoreLocalVector(dmPot, &E_local));

  /* ---------------------------------------------------------------- */
  /* Exchange 1 — DMSwarmMigrate (replaces CharacteristicSendCoords)  */
  /* Issue 9: single collective, no per-particle DMLocatePoints        */
  /* ---------------------------------------------------------------- */
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));

  /* ---------------------------------------------------------------- */
  /* Part 2 — Local field interpolation at feet                       */
  /* ---------------------------------------------------------------- */
  PetscCall(DMGetLocalSection(dmPS, &section));
  PetscCall(DMGetLocalVector(dmPS, &f_local));
  PetscCall(DMGlobalToLocalBegin(dmPS, c->field, INSERT_VALUES, f_local));
  PetscCall(DMGlobalToLocalEnd(dmPS, c->field, INSERT_VALUES, f_local));

  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&feet_coords));
  PetscCall(DMSwarmGetField(sw, "f_interp", NULL, NULL, (void **)&feet_f_interp));

  for (c_cell = cStart; c_cell < cEnd; c_cell++) {
    PetscInt     Npc, *pidx;
    PetscInt     nDOF, p, b;
    PetscScalar *f_cell;
    PetscFEGeom *chunkgeom;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c_cell, &Npc, &pidx));
    if (Npc == 0) {
      PetscCall(DMSwarmSortRestorePointsPerCell(sw, c_cell, &Npc, &pidx));
      continue;
    }

    PetscCall(DMPlexVecGetClosure(dmPS, section, f_local, c_cell, &nDOF, &f_cell));
    /* Issue 12: c_cell - cStart, not c_cell */
    PetscCall(PetscFEGeomGetChunk(plex->fegeomPS, c_cell - cStart, c_cell - cStart + 1, &chunkgeom));

    for (p = 0; p < Npc; p++) {
      PetscReal        ref_coord[2];
      PetscReal        f_val  = 0.0;
      const PetscReal *invJ_p = chunkgeom->invJ;
      const PetscReal *v0_p   = chunkgeom->v;

      /* Real-to-reference map (2D) */
      PetscInt d, e;
      for (d = 0; d < dim; d++) {
        ref_coord[d] = 0.0;
        for (e = 0; e < dim; e++) ref_coord[d] += invJ_p[d * dim + e] * (feet_coords[pidx[p] * 2 + e] - v0_p[e]);
      }

      /* Evaluate field via cached tabulation at ref_coord */
      /* Re-tabulate at this reference point (one particle at a time) */
      {
        PetscTabulation tab_pt;
        PetscCall(PetscFECreateTabulation((PetscFE)NULL, 1, 1, ref_coord, 0, &tab_pt));
        /* Fall back: use the cached tab evaluated at the nearest stored quad point */
        /* Find closest quad point index */
        PetscInt  q_near = 0;
        PetscReal dist2  = PETSC_MAX_REAL;
        PetscInt  q2;
        for (q2 = 0; q2 < Nq; q2++) {
          PetscReal d0 = xi_q[q2 * dim] - ref_coord[0];
          PetscReal d1 = xi_q[q2 * dim + 1] - ref_coord[1];
          PetscReal d2 = d0 * d0 + d1 * d1;
          if (d2 < dist2) {
            dist2  = d2;
            q_near = q2;
          }
        }
        PetscCall(PetscTabulationDestroy(&tab_pt));
        for (b = 0; b < Nb; b++) f_val += PetscRealPart(f_cell[b]) * plex->tab_PS->T[0][q_near * Nb + b];
      }
      feet_f_interp[pidx[p]] = f_val;
    }

    PetscCall(PetscFEGeomRestoreChunk(plex->fegeomPS, c_cell - cStart, c_cell - cStart + 1, &chunkgeom));
    PetscCall(DMPlexVecRestoreClosure(dmPS, section, f_local, c_cell, &nDOF, &f_cell));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c_cell, &Npc, &pidx));
  }

  PetscCall(DMSwarmRestoreField(sw, "f_interp", NULL, NULL, (void **)&feet_f_interp));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&feet_coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dmPS, &f_local));

  /* ---------------------------------------------------------------- */
  /* Part 3 — Cell-local L2 projection → write solution (= f_new)    */
  /* Criterion 3: Nq >= 2 → mass matrix non-diagonal, use LAPACKgesv  */
  /* ---------------------------------------------------------------- */
  PetscCall(DMGetLocalVector(dmPS, &f_new_local));
  PetscCall(VecZeroEntries(f_new_local));

  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, "f_interp", NULL, NULL, (void **)&feet_f_interp));
  PetscCall(DMSwarmGetField(sw, "quad_weight", NULL, NULL, (void **)&feet_weight));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&feet_coords));

  for (c_cell = cStart; c_cell < cEnd; c_cell++) {
    PetscInt     Npc, *pidx;
    PetscInt     p, i, j;
    PetscFEGeom *chunkgeom;
    PetscReal   *M_c, *rhs_c;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c_cell, &Npc, &pidx));
    if (Npc == 0) {
      PetscCall(DMSwarmSortRestorePointsPerCell(sw, c_cell, &Npc, &pidx));
      continue;
    }

    /* Issue 12: c_cell - cStart */
    PetscCall(PetscFEGeomGetChunk(plex->fegeomPS, c_cell - cStart, c_cell - cStart + 1, &chunkgeom));

    PetscCall(PetscCalloc1(Nb * Nb, &M_c));
    PetscCall(PetscCalloc1(Nb, &rhs_c));

    /* Assemble cell-local mass matrix from cached tabulation */
    for (n = 0; n < Nq; n++) {
      PetscReal detJ_q = chunkgeom->detJ[0]; /* uniform mesh: same for all q */
      for (i = 0; i < Nb; i++)
        for (j = 0; j < Nb; j++) M_c[i * Nb + j] += plex->tab_PS->T[0][n * Nb + i] * plex->tab_PS->T[0][n * Nb + j] * w_q[n] * detJ_q;
    }

    /* Assemble RHS */
    for (p = 0; p < Npc; p++) {
      PetscReal        ref_coord[2];
      const PetscReal *invJ_p = chunkgeom->invJ;
      const PetscReal *v0_p   = chunkgeom->v;
      PetscInt         d, e, q_near = 0;
      PetscReal        dist2 = PETSC_MAX_REAL;

      /* Real-to-reference */
      for (d = 0; d < dim; d++) {
        ref_coord[d] = 0.0;
        for (e = 0; e < dim; e++) ref_coord[d] += invJ_p[d * dim + e] * (feet_coords[pidx[p] * 2 + e] - v0_p[e]);
      }

      /* Nearest quad point */
      {
        PetscInt q2;
        for (q2 = 0; q2 < Nq; q2++) {
          PetscReal d0 = xi_q[q2 * dim] - ref_coord[0];
          PetscReal d1 = xi_q[q2 * dim + 1] - ref_coord[1];
          PetscReal d2 = d0 * d0 + d1 * d1;
          if (d2 < dist2) {
            dist2  = d2;
            q_near = q2;
          }
        }
      }

      for (i = 0; i < Nb; i++) rhs_c[i] += plex->tab_PS->T[0][q_near * Nb + i] * feet_f_interp[pidx[p]] * feet_weight[pidx[p]];
    }

    /* Solve M_c * x = rhs_c (LAPACKgesv, Issue Criterion 3) */
    {
      PetscBLASInt  nb = (PetscBLASInt)Nb, nrhs = 1, info;
      PetscBLASInt *piv;
      PetscCall(PetscMalloc1(Nb, &piv));
      PetscCallBLAS("LAPACKgesv", LAPACKgesv_(&nb, &nrhs, M_c, &nb, piv, rhs_c, &nb, &info));
      PetscCall(PetscFree(piv));
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgesv failed for cell %" PetscInt_FMT ": info=%" PetscBLASInt_FMT, c_cell, info);
    }

    /* Scatter to local Vec */
    {
      PetscScalar *rhs_sc;
      PetscCall(PetscMalloc1(Nb, &rhs_sc));
      for (i = 0; i < Nb; i++) rhs_sc[i] = (PetscScalar)rhs_c[i];
      PetscCall(DMPlexVecSetClosure(dmPS, section, f_new_local, c_cell, rhs_sc, INSERT_VALUES));
      PetscCall(PetscFree(rhs_sc));
    }

    PetscCall(PetscFEGeomRestoreChunk(plex->fegeomPS, c_cell - cStart, c_cell - cStart + 1, &chunkgeom));
    PetscCall(PetscFree(M_c));
    PetscCall(PetscFree(rhs_c));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c_cell, &Npc, &pidx));
  }

  PetscCall(DMSwarmRestoreField(sw, "f_interp", NULL, NULL, (void **)&feet_f_interp));
  PetscCall(DMSwarmRestoreField(sw, "quad_weight", NULL, NULL, (void **)&feet_weight));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&feet_coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));

  /* solution = f_new (Issue 10) */
  PetscCall(DMLocalToGlobal(dmPS, f_new_local, INSERT_VALUES, solution));
  PetscCall(DMRestoreLocalVector(dmPS, &f_new_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
PETSC_EXTERN PetscErrorCode CharacteristicCreate_Plex(Characteristic c)
{
  Characteristic_Plex *plex;

  PetscFunctionBegin;
  PetscCall(PetscNew(&plex));
  c->data = (void *)plex;

  c->structured = PETSC_FALSE; /* DMPlex path: no DMDA APIs */

  c->ops->view    = CharacteristicView_Plex;
  c->ops->destroy = CharacteristicDestroy_Plex;
  c->ops->setup   = CharacteristicSetUp_Plex;
  c->ops->solve   = CharacteristicSolve_Plex;
  PetscFunctionReturn(PETSC_SUCCESS);
}
