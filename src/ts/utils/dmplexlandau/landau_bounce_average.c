/*
  landau_bounce_average.c -- Bounce-average Jacobian for the Landau collision operator.

  Implements Option BA-4 from plans/petsc_landau_extensions_plan.md:

  The bounce-averaged kinetic equation is:

    H(p, xi) df/dt = C[f]   =>   df/dt = (1/H) C[f]

  where H(p, xi) is the bounce-average Jacobian:

    Passing electrons (|xi| > xi0):  H = (4/pi) K(k_pass) p^2
    Trapped electrons (|xi| < xi0):  H = (2/pi) K(k_trap) p^2

  with K(k) the complete elliptic integral of the first kind and:

    k_pass^2 = 1 - xi0^2/xi^2,   k_trap^2 = 1 - xi^2/xi0^2

  The trapping boundary is xi0 = sqrt(1 - r/R).

  In Cartesian velocity coordinates (v_x, v_y, v_z) with v_z = v_par:

    p = |v| = sqrt(v_x^2 + v_y^2 + v_z^2)
    xi = v_z / |v|

  The bounce-averaged mass matrix is:

    M_bounce[i,j] = integral phi_i * (H/p^2) * phi_j dv

  This is implemented by replacing the standard g0=1 mass callback with
  g0 = H(v)/p^2 = H(v)/|v|^2 at each quadrature point.

  Usage:
    BounceAverageCtx ba_ctx = {
        .xi0       = 0.316,   // trapping boundary sqrt(1 - r/R), e.g. r=0.2m, R=1.4m
        .delta_xi  = 0.05,    // tanh smoothing width at xi0
    };
    // Replace the standard mass matrix with the bounce-averaged one:
    PetscCall(LandauBounceAverageCreateMassMatrix(pack, &ba_ctx));
    // Use ba_ctx.M_bounce in place of ctx->M in DMPlexLandauIFunction/IJacobian.

  Reference: Liu et al. (2020) arXiv:2009.11801v3, Eq. 4-6.
             Adams et al. (2025) SIAM J. Sci. Comput. 47(2), B360-B381.
*/

#include <petscdmplex.h>
#include <petscdmcomposite.h>
#include <petsclandau.h> /* includes BounceAverageCtx typedef */
#include <petscds.h>
#include <petscts.h>
#include <petscfe.h>

/* ------------------------------------------------------------------ */
/* Complete elliptic integral K(k) via arithmetic-geometric mean      */
/* ------------------------------------------------------------------ */
static PetscReal EllipticK(PetscReal k)
{
  PetscReal a = 1.0, b = PetscSqrtReal(1.0 - k * k + 1e-30);
  PetscReal c;
  /* AGM iteration: converges quadratically */
  for (PetscInt iter = 0; iter < 20; iter++) {
    c = 0.5 * (a + b);
    b = PetscSqrtReal(a * b);
    a = c;
    if (PetscAbsReal(a - b) < 1e-15 * a) break;
  }
  return PETSC_PI / (2.0 * a);
}

/* ------------------------------------------------------------------ */
/* Bounce-average weight H(xi, xi0) / p^2                            */
/*                                                                    */
/* Returns the bounce-average Jacobian divided by p^2 = |v|^2:       */
/*   Passing (|xi| > xi0): H/p^2 = (4/pi) K(k_pass)                 */
/*   Trapped (|xi| < xi0): H/p^2 = (2/pi) K(k_trap)                 */
/* Smoothed with tanh transition of width delta_xi at xi0.           */
/* ------------------------------------------------------------------ */
static PetscReal BounceWeight(PetscReal xi, PetscReal xi0, PetscReal delta_xi)
{
  PetscReal xi2  = xi * xi;
  PetscReal xi02 = xi0 * xi0;
  PetscReal H_pass, H_trap, w;

  /* Passing weight: k_pass^2 = 1 - xi0^2/xi^2 */
  if (xi2 > 1e-14) {
    PetscReal k2_pass = 1.0 - xi02 / (xi2 + 1e-30);
    k2_pass           = PetscMax(0.0, PetscMin(1.0 - 1e-12, k2_pass));
    H_pass            = (4.0 / PETSC_PI) * EllipticK(PetscSqrtReal(k2_pass));
  } else {
    H_pass = 2.0; /* K(0) = pi/2, so (4/pi)(pi/2) = 2 */
  }

  /* Trapped weight: k_trap^2 = 1 - xi^2/xi0^2 */
  if (xi02 > 1e-14) {
    PetscReal k2_trap = 1.0 - xi2 / (xi02 + 1e-30);
    k2_trap           = PetscMax(0.0, PetscMin(1.0 - 1e-12, k2_trap));
    H_trap            = (2.0 / PETSC_PI) * EllipticK(PetscSqrtReal(k2_trap));
  } else {
    H_trap = 1.0; /* deeply trapped limit */
  }

  /* Smooth transition: w=1 -> passing, w=0 -> trapped */
  w = 0.5 * (1.0 + PetscTanhReal((PetscAbsReal(xi) - xi0) / (delta_xi + 1e-30)));
  return w * H_pass + (1.0 - w) * H_trap;
}

/* ------------------------------------------------------------------ */
/* PetscDS g0 callback: bounce-averaged mass integrand                */
/*                                                                    */
/* g0 = H(v) / p^2  where p = |v| (non-relativistic)                 */
/*                                                                    */
/* constants[0] = xi0    (trapping boundary)                          */
/* constants[1] = delta_xi (smoothing width)                          */
/* ------------------------------------------------------------------ */
static void g0_bounce_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  /* x[] = (vx, vy, vz) with vz = v_par */
  PetscReal v2       = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  PetscReal v        = PetscSqrtReal(v2 + 1e-30);
  PetscReal xi       = x[2] / v; /* v_par / |v| */
  PetscReal xi0      = PetscRealPart(constants[0]);
  PetscReal delta_xi = PetscRealPart(constants[1]);
  g0[0]              = BounceWeight(xi, xi0, delta_xi);
}

/* 2D cylindrical: x[] = (v_perp, v_par) */
static void g0_bounce_2d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  /* x[] = (v_perp, v_par); cylindrical Jacobian 2*pi*v_perp handled by PetscFE */
  PetscReal v2       = x[0] * x[0] + x[1] * x[1];
  PetscReal v        = PetscSqrtReal(v2 + 1e-30);
  PetscReal xi       = x[1] / v; /* v_par / |v| */
  PetscReal xi0      = PetscRealPart(constants[0]);
  PetscReal delta_xi = PetscRealPart(constants[1]);
  /* cylindrical mass: g0 = 2*pi*v_perp * H/p^2 -- the 2*pi*v_perp factor is in PetscFE */
  g0[0] = BounceWeight(xi, xi0, delta_xi);
}

/* ------------------------------------------------------------------ */
/* LandauBounceAverageCreateMassMatrix                                */
/*                                                                    */
/* Assemble the bounce-averaged mass matrix M_bounce.                 */
/* This replaces the standard mass matrix (g0=1) with g0=H(v)/p^2.   */
/*                                                                    */
/* Input:                                                             */
/*   pack    - the Landau DM (from DMPlexLandauCreateVelocitySpace)   */
/*   ba_ctx  - BounceAverageCtx with xi0 and delta_xi                 */
/*                                                                    */
/* Output:                                                            */
/*   ba_ctx->M_bounce is set                                          */
/* ------------------------------------------------------------------ */
PetscErrorCode LandauBounceAverageCreateMassMatrix(DM pack, BounceAverageCtx *ba_ctx)
{
  LandauCtx  *ctx;
  PetscInt    dim;
  DM          bounceDM[LANDAU_MAX_GRIDS], bounce_pack;
  Mat         subM[LANDAU_MAX_GRIDS], packM;
  PetscDS     prob;
  PetscScalar constants[2];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pack, DM_CLASSID, 1);
  PetscAssertPointer(ba_ctx, 2);
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_PLIB, "no LandauCtx on DM");
  PetscCall(DMGetDimension(pack, &dim));

  constants[0] = ba_ctx->xi0;      /* trapping boundary */
  constants[1] = ba_ctx->delta_xi; /* smoothing width */

  PetscCall(DMCompositeCreate(PetscObjectComm((PetscObject)pack), &bounce_pack));
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
    PetscCall(DMClone(ctx->plex[grid], &bounceDM[grid]));
    PetscCall(DMCopyFields(ctx->plex[grid], PETSC_DETERMINE, PETSC_DETERMINE, bounceDM[grid]));
    PetscCall(DMCreateDS(bounceDM[grid]));
    PetscCall(DMGetDS(bounceDM[grid], &prob));
    PetscInt nspec = ctx->species_offset[grid + 1] - ctx->species_offset[grid];
    for (PetscInt ix = 0; ix < nspec; ix++) {
      PetscCall(PetscDSSetConstants(prob, 2, constants));
      if (dim == 3) PetscCall(PetscDSSetJacobian(prob, ix, ix, g0_bounce_3d, NULL, NULL, NULL));
      else PetscCall(PetscDSSetJacobian(prob, ix, ix, g0_bounce_2d, NULL, NULL, NULL));
    }
    for (PetscInt b_id = 0; b_id < ctx->batch_sz; b_id++) PetscCall(DMCompositeAddDM(bounce_pack, bounceDM[grid]));
    PetscCall(DMCreateMatrix(bounceDM[grid], &subM[grid]));
  }

  PetscCall(PetscOptionsInsertString(NULL, "-dm_preallocate_only"));
  PetscCall(DMCreateMatrix(bounce_pack, &packM));
  PetscCall(PetscOptionsInsertString(NULL, "-dm_preallocate_only false"));
  PetscCall(MatSetOption(packM, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(packM, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(DMDestroy(&bounce_pack));

  /* Assemble each grid's sub-matrix */
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
    Vec locX;
    PetscCall(DMGetLocalVector(bounceDM[grid], &locX));
    PetscCall(VecZeroEntries(locX));
    PetscCall(DMPlexSNESComputeJacobianFEM(bounceDM[grid], locX, subM[grid], subM[grid], ctx));
    PetscCall(DMRestoreLocalVector(bounceDM[grid], &locX));
    PetscCall(DMDestroy(&bounceDM[grid]));
  }

  /* Scatter sub-matrices into the packed matrix */
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
    Mat      B = subM[grid];
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
        PetscCall(MatSetValues(packM, 1, &row, nzl, colbuf, vals, INSERT_VALUES));
        PetscCall(MatRestoreRow(B, i, &nzl, &cols, &vals));
      }
    }
    PetscCall(PetscFree(colbuf));
    PetscCall(MatDestroy(&subM[grid]));
  }
  PetscCall(MatAssemblyBegin(packM, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(packM, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)packM, "mass_bounce_averaged"));

  ba_ctx->M_bounce = packM;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------ */
/* LandauBounceAverageDestroy                                         */
/* ------------------------------------------------------------------ */
PetscErrorCode LandauBounceAverageDestroy(BounceAverageCtx *ba_ctx)
{
  PetscFunctionBegin;
  if (!ba_ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDestroy(&ba_ctx->M_bounce));
  PetscFunctionReturn(PETSC_SUCCESS);
}
