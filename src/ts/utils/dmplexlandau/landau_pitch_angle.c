/*
  landau_pitch_angle.c - Excess pitch-angle scattering operator for partial screening.

  Implements Option D from plans/petsc_landau_extensions_plan.md:

    Delta C_pitch[f] = nu_D^excess(v) * L^2[f]

  where L^2[f] = div_v [(v^2 I - v x v)/v^3 * grad_v f] is the Lorentz pitch-angle
  operator in Cartesian velocity coordinates, and

    nu_D^excess(v) = (lnL2_factor - 1) * nu_D_standard(v)

  with nu_D_standard the standard deflection frequency using the Chandrasekhar
  function G(x) = erf(x) - (2/sqrt(pi)) x exp(-x^2).

  This operator is added as a TSSetRHSFunction() callback on top of the base
  Landau operator (DMPlexLandauIFunction / DMPlexLandauIJacobian), which handles
  the standard slowing-down and pitch-angle scattering at the unscreened rate.
  The total collision operator is:

    C_total[f] = C_Landau[f]  +  Delta C_pitch[f]

  The excess pitch-angle operator conserves particle number exactly (the
  divergence theorem guarantees integral Delta C_pitch[f] dv = 0 since P_ij is traceless
  in the pitch-angle direction).

  Reference: Hesslow, L. et al. (2018). Nucl. Fusion 58, 106028.
             doi:10.1088/1741-4326/aac33e

  Usage:
    PitchAngleCtx pa_ctx = {
        .lnL2_factor  = 150.0,   // Hesslow Ar2+ pitch-angle enhancement
        .lnL_standard = 17.0,    // standard Coulomb logarithm
        .T_e_eV       = 5.0,     // bulk electron temperature [eV]
        .n_e          = 2e20,    // electron density [m^-3]
        .Z_eff        = 2.0,     // effective charge
    };
    PetscCall(LandauPitchAngleCreateMatrix(pack, &pa_ctx));
    PetscCall(TSSetRHSFunction(ts, NULL, LandauPitchAngleRHS, &pa_ctx));
    // ... TSSolve ...
    PetscCall(LandauPitchAngleDestroy(&pa_ctx));
*/

#include <petscdmplex.h>
#include <petscdmcomposite.h>
#include <petsclandau.h> /* includes PitchAngleCtx typedef */
#include <petscds.h>
#include <petscts.h>
#include <petscfe.h>

/* ------------------------------------------------------------------ */
/* Chandrasekhar function G(x) = erf(x) - (2/sqrt(pi)) x exp(-x^2)  */
/* ------------------------------------------------------------------ */
static PetscReal ChandrasekharG(PetscReal x)
{
  if (x < 1e-8) return (2.0 / (3.0 * PETSC_PI)) * x; /* Taylor: G ~ 2x/(3*sqrt(pi)) for x->0 */
  return PetscErfReal(x) - (2.0 / PetscSqrtReal(PETSC_PI)) * x * PetscExpReal(-x * x);
}

/* ------------------------------------------------------------------ */
/* PetscDS g3 callback: excess pitch-angle diffusion tensor           */
/*                                                                    */
/* Implements the weak-form stiffness integrand:                      */
/*   g3[i*dim+j] = nu_D^excess(v) * P_ij(v)                         */
/* where P_ij = (v^2 delta_ij - v_i v_j) / v^3                      */
/*                                                                    */
/* constants[0] = nu_D^excess prefactor (SI units, non-dimensionalized)*/
/* constants[1] = v_th (thermal velocity for Chandrasekhar argument)  */
/* ------------------------------------------------------------------ */
static void g3_pitch_angle_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  /* x[] = (vx, vy, vz) in non-dimensional units (v/v_0) */
  PetscReal v2   = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  PetscReal v    = PetscSqrtReal(v2 + 1e-30); /* avoid v=0 singularity */
  PetscReal v3   = v2 * v;
  PetscReal v_th = PetscRealPart(constants[1]); /* thermal velocity (non-dim) */
  PetscReal xarg = v / (v_th + 1e-30);
  PetscReal G    = ChandrasekharG(xarg);
  /* nu_D^standard ~ G(v/v_th)/v^3 -- the prefactor is absorbed into constants[0] */
  PetscReal nu_excess = PetscRealPart(constants[0]) * G / (v3 + 1e-30);

  /* P_ij = (v^2 delta_ij - v_i v_j) / v^3, scaled by nu_excess */
  /* g3 layout: g3[field_i * dim*dim + i*dim + j] for (test_i, basis_j) */
  for (PetscInt i = 0; i < dim; i++) {
    for (PetscInt j = 0; j < dim; j++) {
      PetscReal delta_ij = (i == j) ? 1.0 : 0.0;
      PetscReal Pij      = (v2 * delta_ij - x[i] * x[j]) / (v3 + 1e-30);
      g3[i * dim + j]    = nu_excess * Pij;
    }
  }
}

/* 2D cylindrical version: x[] = (v_perp, v_par), P_ij in (r,z) */
static void g3_pitch_angle_2d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  /* x[] = (v_perp, v_par); cylindrical: v^2 = v_perp^2 + v_par^2 */
  PetscReal v2   = x[0] * x[0] + x[1] * x[1];
  PetscReal v    = PetscSqrtReal(v2 + 1e-30);
  PetscReal v3   = v2 * v;
  PetscReal v_th = PetscRealPart(constants[1]);
  PetscReal xarg = v / (v_th + 1e-30);
  PetscReal G    = ChandrasekharG(xarg);
  /* cylindrical Jacobian factor 2*pi*v_perp is handled by PetscFE */
  PetscReal nu_excess = PetscRealPart(constants[0]) * G / (v3 + 1e-30);

  for (PetscInt i = 0; i < dim; i++) {
    for (PetscInt j = 0; j < dim; j++) {
      PetscReal delta_ij = (i == j) ? 1.0 : 0.0;
      PetscReal Pij      = (v2 * delta_ij - x[i] * x[j]) / (v3 + 1e-30);
      g3[i * dim + j]    = nu_excess * Pij;
    }
  }
}

/* ------------------------------------------------------------------ */
/* LandauPitchAngleCreateMatrix                                       */
/*                                                                    */
/* Pre-assemble the excess pitch-angle scattering matrix Apitch.      */
/* This matrix only changes when T_e changes (not every time step).   */
/*                                                                    */
/* Input:                                                             */
/*   pack    - the Landau DM (from DMPlexLandauCreateVelocitySpace)   */
/*   pa_ctx  - PitchAngleCtx with physical parameters                 */
/*                                                                    */
/* Output:                                                            */
/*   pa_ctx->Apitch is set                                            */
/* ------------------------------------------------------------------ */
PetscErrorCode LandauPitchAngleCreateMatrix(DM pack, PitchAngleCtx *pa_ctx)
{
  LandauCtx *ctx;
  PetscInt   dim;
  PetscReal  e       = 1.602176634e-19; /* elementary charge [C] */
  PetscReal  m_e     = 9.10938e-31;     /* electron mass [kg] */
  PetscReal  eps0    = 8.854188e-12;    /* permittivity [F/m] */
  PetscReal  T_e_J   = pa_ctx->T_e_eV * e;
  PetscReal  v_th_SI = PetscSqrtReal(2.0 * T_e_J / m_e); /* thermal speed [m/s] */
  /* Standard deflection frequency prefactor (SI):
     nu_D0 = n_e Z_eff^2 e^4 lnLambda / (4*pi eps0^2 m_e^2 v_th^3)
     The Chandrasekhar G(v/v_th)/v^3 factor is applied pointwise in g3. */
  PetscReal   nu_D0_SI      = pa_ctx->n_e * pa_ctx->Z_eff * pa_ctx->Z_eff * PetscPowReal(e, 4) * pa_ctx->lnL_standard / (4.0 * PETSC_PI * PetscSqr(eps0) * PetscSqr(m_e) * PetscPowReal(v_th_SI, 3));
  PetscReal   excess_factor = pa_ctx->lnL2_factor - 1.0; /* (lnLambda2/lnLambda - 1) */
  PetscReal   nu_excess_SI  = excess_factor * nu_D0_SI;
  DM          pitchDM[LANDAU_MAX_GRIDS], pitch_pack;
  Mat         subA[LANDAU_MAX_GRIDS], packA;
  PetscDS     prob;
  PetscScalar constants[2];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pack, DM_CLASSID, 1);
  PetscAssertPointer(pa_ctx, 2);
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_PLIB, "no LandauCtx on DM");
  PetscCall(DMGetDimension(pack, &dim));

  /* Non-dimensionalize: v_th_nondim = v_th_SI / v_0, nu_nondim = nu_SI * t_0 */
  PetscReal v_th_nondim  = v_th_SI / ctx->v_0;
  PetscReal nu_excess_nd = nu_excess_SI * ctx->t_0;

  constants[0] = nu_excess_nd; /* nu_D^excess prefactor (non-dim) */
  constants[1] = v_th_nondim;  /* v_th (non-dim) for Chandrasekhar argument */

  /* Build a composite DM mirroring the mass matrix structure */
  PetscCall(DMCompositeCreate(PetscObjectComm((PetscObject)pack), &pitch_pack));
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
    PetscCall(DMClone(ctx->plex[grid], &pitchDM[grid]));
    PetscCall(DMCopyFields(ctx->plex[grid], PETSC_DETERMINE, PETSC_DETERMINE, pitchDM[grid]));
    PetscCall(DMCreateDS(pitchDM[grid]));
    PetscCall(DMGetDS(pitchDM[grid], &prob));
    /* Register the pitch-angle g3 callback for each species on this grid */
    PetscInt nspec = ctx->species_offset[grid + 1] - ctx->species_offset[grid];
    for (PetscInt ix = 0; ix < nspec; ix++) {
      PetscCall(PetscDSSetConstants(prob, 2, constants));
      if (dim == 3) PetscCall(PetscDSSetJacobian(prob, ix, ix, NULL, NULL, NULL, g3_pitch_angle_3d));
      else PetscCall(PetscDSSetJacobian(prob, ix, ix, NULL, NULL, NULL, g3_pitch_angle_2d));
    }
    for (PetscInt b_id = 0; b_id < ctx->batch_sz; b_id++) PetscCall(DMCompositeAddDM(pitch_pack, pitchDM[grid]));
    PetscCall(DMCreateMatrix(pitchDM[grid], &subA[grid]));
  }

  PetscCall(PetscOptionsInsertString(NULL, "-dm_preallocate_only"));
  PetscCall(DMCreateMatrix(pitch_pack, &packA));
  PetscCall(PetscOptionsInsertString(NULL, "-dm_preallocate_only false"));
  PetscCall(MatSetOption(packA, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(packA, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(DMDestroy(&pitch_pack));

  /* Assemble each grid's sub-matrix */
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
    Vec locX;
    PetscCall(DMGetLocalVector(pitchDM[grid], &locX));
    PetscCall(VecZeroEntries(locX));
    PetscCall(DMPlexSNESComputeJacobianFEM(pitchDM[grid], locX, subA[grid], subA[grid], ctx));
    PetscCall(DMRestoreLocalVector(pitchDM[grid], &locX));
    PetscCall(DMDestroy(&pitchDM[grid]));
  }

  /* Scatter sub-matrices into the packed matrix (same offset logic as mass matrix) */
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
    Mat      B = subA[grid];
    PetscInt nloc, nzl, *colbuf, COL_BF_SIZE = 1024, row;
    PetscCall(PetscMalloc(sizeof(*colbuf) * COL_BF_SIZE, &colbuf));
    PetscCall(MatGetSize(B, &nloc, NULL));
    for (PetscInt b_id = 0; b_id < ctx->batch_sz; b_id++) {
      const PetscInt     moffset = LAND_MOFFSET(b_id, grid, ctx->batch_sz, ctx->num_grids, ctx->mat_offset);
      const PetscInt    *cols;
      const PetscScalar *vals;
      for (PetscInt i = 0; i < nloc; i++) {
        PetscCall(MatGetRow(B, i, &nzl, NULL, NULL));
        if (nzl > COL_BF_SIZE) {
          PetscCall(PetscFree(colbuf));
          COL_BF_SIZE = nzl;
          PetscCall(PetscMalloc(sizeof(*colbuf) * COL_BF_SIZE, &colbuf));
        }
        PetscCall(MatGetRow(B, i, &nzl, &cols, &vals));
        for (PetscInt j = 0; j < nzl; j++) colbuf[j] = cols[j] + moffset;
        row = i + moffset;
        PetscCall(MatSetValues(packA, 1, &row, nzl, colbuf, vals, INSERT_VALUES));
        PetscCall(MatRestoreRow(B, i, &nzl, &cols, &vals));
      }
    }
    PetscCall(PetscFree(colbuf));
    PetscCall(MatDestroy(&subA[grid]));
  }
  PetscCall(MatAssemblyBegin(packA, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(packA, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)packA, "pitch_angle_excess"));

  pa_ctx->Apitch = packA;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
/* LandauPitchAngleRHS                                                */
/*                                                                    */
/* TSSetRHSFunction() callback. Computes F += Apitch * X.            */
/*                                                                    */
/* Register with:                                                     */
/*   TSSetRHSFunction(ts, NULL, LandauPitchAngleRHS, &pa_ctx);       */
/* ------------------------------------------------------------------ */
PetscErrorCode LandauPitchAngleRHS(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  PitchAngleCtx *pa = (PitchAngleCtx *)ctx;

  PetscFunctionBegin;
  PetscAssertPointer(pa, 5);
  PetscCheck(pa->Apitch, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Call LandauPitchAngleCreateMatrix() before LandauPitchAngleRHS()");
  /* F += Apitch * X  (excess pitch-angle scattering term) */
  PetscCall(MatMultAdd(pa->Apitch, X, F, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
/* LandauPitchAngleDestroy                                            */
/* ------------------------------------------------------------------ */
PetscErrorCode LandauPitchAngleDestroy(PitchAngleCtx *pa_ctx)
{
  PetscFunctionBegin;
  if (!pa_ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDestroy(&pa_ctx->Apitch));
  PetscFunctionReturn(PETSC_SUCCESS);
}
