/* MANSEC = DM */
/* SUBMANSEC = FE */
/*
  petscfekokkos.h -- Header-template kernel API for PETSCFEKOKKOS (Phase 1.B)

  This header provides the compile-time template kernel for GPU-resident
  PetscFE residual and Jacobian integration.  The key design principle:

    CUDA cannot call host function pointers from device kernels.
    PetscDSSetResidual() registers host function pointers at runtime.
    Solution: user writes KOKKOS_INLINE_FUNCTION callbacks and passes them
    as C++ template parameters.  nvcc_wrapper sees both the kernel body and
    the callbacks in the same translation unit and inlines them at compile time.

  Usage pattern (in user's .kokkos.cxx file):

    #include <petscfekokkos.h>

    KOKKOS_INLINE_FUNCTION
    static void f0_poisson(PETSCFE_KOKKOS_POINT_ARGS, PetscScalar f0[]) { ... }

    KOKKOS_INLINE_FUNCTION
    static void f1_poisson(PETSCFE_KOKKOS_POINT_ARGS, PetscScalar f1[]) { ... }

    // In SetupDiscretization or solve loop:
    PetscCall(PetscFEKokkosComputeResidual<f0_poisson, f1_poisson>(ds, key, Ne, cgeom,
                coefficients, coefficients_t, t, elemVec));

  Relationship to existing PETSCFEKOKKOS:
    - fekokkos.kokkos.cxx implements the two-phase host/device path (fallback).
    - This header provides the opt-in all-device path.
    - Both coexist; the template path is selected by calling
      PetscFEKokkosComputeResidual instead of relying on the ops dispatch.

  Restrictions (Phase 1.B -- lifted in later phases):
    - Single field (Nf == 1), no auxiliary fields (dsAux == NULL).
    - Nc in {1,2,3} -- scalar and multi-component vector fields supported (Phase 1b.B2 complete).
    - Affine and non-affine elements both supported.
    - No COO assembly (Phase 1.C adds that).

  Compatibility:
    - Requires C++14 or later (template function pointers, constexpr).
    - Compatible with nvcc_wrapper (CUDA), hipcc (HIP), and host-only Kokkos.
    - No PETSc error-handling macros (PetscCall, PetscFunctionBegin) in device
      code -- those are host-only.  Host convenience functions use them normally.

  Jacobian ops (integratejacobian) fall back to the Basic (CPU) implementation;
  only integrateresidual is GPU-accelerated via the template path.
*/

#pragma once

#if defined(PETSC_HAVE_KOKKOS)

  #include <petscfe.h>
  #include <petscds.h>
  #include <petscsnes.h>
  #include <Kokkos_Core.hpp>

  /* Macro: PETSCFE_KOKKOS_POINT_ARGS
   Expands to the full PetscPointFn argument list (minus the output array).
   Use this in KOKKOS_INLINE_FUNCTION callback declarations to keep them
   consistent with the PetscDS weak-form signature.

   The signature matches PetscPointFn (petscdstypes.h):
     void fn(PetscInt dim, PetscInt Nf, PetscInt NfAux,
             const PetscInt uOff[], const PetscInt uOff_x[],
             const PetscScalar u[], const PetscScalar u_t[],
             const PetscScalar u_x[],
             const PetscInt aOff[], const PetscInt aOff_x[],
             const PetscScalar a[], const PetscScalar a_t[],
             const PetscScalar a_x[],
             PetscReal t, const PetscReal x[],
             PetscInt numConstants, const PetscScalar constants[],
             PetscScalar out[]) */
  #define PETSCFE_KOKKOS_POINT_ARGS \
    PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[]

  /* Jacobian callbacks have an extra u_tShift argument between t and x */
  #define PETSCFE_KOKKOS_JAC_POINT_ARGS \
    PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[]

/* PetscFEKokkosIntegrateResidualCell<F0, F1>

   Device-callable template function that integrates the residual for a
   single element e.  Both F0 and F1 are resolved at compile time -- no
   function pointer indirection on device.

   Template parameters:
     F0  -- KOKKOS_INLINE_FUNCTION callback for the zeroth-order term
            (volume source / reaction).  Signature: PetscPointFn.
            Pass nullptr to skip.
     F1  -- KOKKOS_INLINE_FUNCTION callback for the first-order term
            (flux / diffusion).  Signature: PetscPointFn.
            Pass nullptr to skip.

   Arguments (all flat device-accessible arrays):
     e          -- element index (0-based)
     Nq         -- number of quadrature points
     Nb         -- TOTAL number of DOFs per element = T->Nb = Nb_scalar * Nc
                  (matches PetscFEEvaluateFieldJets_Internal convention)
     Nc         -- number of field components (1 for scalar)
     dim        -- spatial / reference dimension
     dE         -- embedding dimension (== dim for non-embedded meshes)
     B[Nq*Nb*Nc]          -- basis values
     D[Nq*Nb*Nc*dim]      -- basis reference derivatives
     w[Nq]                -- quadrature weights
     invJ[Ne*Nq*dE*dE]    -- inverse Jacobian (expanded: one per (e,q))
     detJ[Ne*Nq]          -- Jacobian determinant (expanded: one per (e,q))
     coords[Ne*Nq*dE]     -- physical coordinates at each (e,q)
     coeff[Ne*totDim]     -- element coefficients (solution DOFs)
     totDim               -- total DOFs per element across all fields
     uOff                 -- field offset in coeff (scalar: 0)
     fOff                 -- field offset in elemVec
     t                    -- time
     numConstants         -- number of PDE constants
     constants[numConstants] -- PDE constants
     elemVec[Ne*totDim]   -- output element vector (accumulated, not zeroed here)

   Layout conventions (match fekokkos.kokkos.cxx / PetscFEEvaluateFieldJets_Internal):
     B[q * Nb*Nc + b*Nc + c]          -- b = 0..Nb-1 (total DOFs), c = 0..Nc-1
     D[q * Nb*Nc*dim + b*Nc*dim + c*dim + e2]
     invJ[e * Nq*dE*dE + q*dE*dE + i*dE + j]   (expanded, affine replicated)
     detJ[e * Nq + q]                            (expanded)
     coords[e * Nq*dE + q*dE + d]
     coeff[e * totDim + uOff + b]               -- b indexes ALL DOFs (0..Nb-1)
     elemVec[e * totDim + fOff + b*Nc + c]

   IMPORTANT: coeff is indexed as coeff[b] (not coeff[b*Nc+c]) because PETSc
   stores DOFs in the order imposed by PetscDualSpaceGetDimension(), where b
   already encodes both the scalar basis index and the component.  This matches
   PetscFEEvaluateFieldJets_Internal (fe.c) and fekokkos.kokkos.cxx exactly.

   Note: elemVec is accumulated (+=), not zeroed.  The caller must zero it
   before the kernel launch (or use Kokkos::deep_copy to zero the device View). */
template <PetscPointFn *F0, PetscPointFn *F1>
KOKKOS_INLINE_FUNCTION void PetscFEKokkosIntegrateResidualCell(PetscInt e, PetscInt Nq, PetscInt Nb, PetscInt Nc, PetscInt dim, PetscInt dE, const PetscReal *B,                                     /* [Nq * Nb * Nc]          */
                                                               const PetscReal   *D,                                                                                                                 /* [Nq * Nb * Nc * dim]    */
                                                               const PetscReal   *w,                                                                                                                 /* [Nq]                    */
                                                               const PetscReal   *invJ,                                                                                                              /* [Ne * Nq * dE * dE]     */
                                                               const PetscReal   *detJ,                                                                                                              /* [Ne * Nq]               */
                                                               const PetscReal   *coords,                                                                                                            /* [Ne * Nq * dE]          */
                                                               const PetscScalar *coeff,                                                                                                             /* [Ne * totDim]            */
                                                               PetscInt totDim, PetscInt uOff, PetscInt fOff, PetscReal t, PetscInt numConstants, const PetscScalar *constants, PetscScalar *elemVec /* [Ne * totDim] -- accumulated */
)
{
  /* Per-element scratch -- stack-allocated, sized for the call-site Nb/Nc/dE.
     CUDA stack is limited (~16 KB per thread) but for typical FE orders
     (P1-P3, Q1-Q3) these arrays are small:
       u_loc:  Nc      <= 4  scalars  (Nc=1 for Poisson)
       ux_loc: Nc*dE   <= 8  scalars
       f0_loc: Nc      <= 4  scalars
       f1_loc: Nc*dE   <= 8  scalars
       val:    Nb*Nc   <= 36 scalars  (P3 quad: Nb=16, Nc=1)
     Total ~ 56 PetscScalar = 448 bytes -- well within CUDA stack limits.

     For Phase 1b (Nc=dim=2) and higher orders these grow but remain safe.
     If stack pressure becomes an issue, switch to Kokkos scratch memory. */

  /* Maximum sizes for stack arrays -- sized for P4 hex 3D with Nc=dim=3.
     These are compile-time upper bounds; actual loops use runtime Nb/Nc/dE.
     NOTE: val[] was removed -- write directly to elemVec device buffer to
     eliminate 3,000 bytes/thread GPU stack pressure (cudaErrorIllegalAddress
     for P3/P4 hex 3D on A100).  elemVec is pre-allocated and zeroed by the
     caller (FormResidual_COO) before launching this kernel. */
  constexpr PetscInt PETSCFE_KOKKOS_MAX_NC = 3; /* max components (dim) */
  constexpr PetscInt PETSCFE_KOKKOS_MAX_DE = 3; /* max embedding dim */

  PetscScalar u_loc[PETSCFE_KOKKOS_MAX_NC];                          /* u at one qp */
  PetscScalar ux_loc[PETSCFE_KOKKOS_MAX_NC * PETSCFE_KOKKOS_MAX_DE]; /* gradu at one qp */
  PetscScalar f0_loc[PETSCFE_KOKKOS_MAX_NC];                         /* f0 output */
  PetscScalar f1_loc[PETSCFE_KOKKOS_MAX_NC * PETSCFE_KOKKOS_MAX_DE]; /* f1 output */

  /* Constant offset arrays for single-field, no-aux case */
  const PetscInt uOff_l[1]   = {0};
  const PetscInt uOff_x_l[1] = {0};

  /* Write directly to elemVec device buffer -- no intermediate stack array.
     Each element e is processed by exactly one thread (RangePolicy over Ne),
     so no atomics are needed here.  The caller zeroes d_elemVec before launch. */
  PetscScalar *ev_e = &elemVec[e * totDim + fOff];

  for (PetscInt q = 0; q < Nq; ++q) {
    /* Geometry at (e, q) -- invJ and detJ are pre-expanded by the host
       (affine elements: single invJ replicated across all Nq slots). */
    const PetscReal *invJ_eq = &invJ[(e * Nq + q) * dE * dE];
    const PetscReal  detJ_eq = detJ[e * Nq + q];
    const PetscReal *x_eq    = &coords[(e * Nq + q) * dE];
    const PetscReal  wq      = w[q] * detJ_eq;

    /* Basis pointers for this quadrature point */
    const PetscReal *B_q = &B[q * Nb * Nc];       /* B[q,b,c] */
    const PetscReal *D_q = &D[q * Nb * Nc * dim]; /* D[q,b,c,e2] */

    /* Coefficient pointer for this element */
    const PetscScalar *coeff_e = &coeff[e * totDim + uOff];

    /* Zero per-qp scratch */
    for (PetscInt c = 0; c < Nc; ++c) u_loc[c] = 0.0;
    for (PetscInt i = 0; i < Nc * dE; ++i) ux_loc[i] = 0.0;

    /* Interpolate u[c] = sum_b B[q,b,c] * coeff[b]
       b = 0..Nb-1 (total DOFs), coeff[b] matches PetscFEEvaluateFieldJets_Internal.
       For vector FE (Nc>1): b encodes both scalar basis index and component,
       so coeff[b] is the correct single-index DOF access. */
    for (PetscInt b = 0; b < Nb; ++b)
      for (PetscInt c = 0; c < Nc; ++c) u_loc[c] += B_q[b * Nc + c] * coeff_e[b];

    /* Interpolate ux[c,d] = sum_b D[q,b,c,e2] * invJ[e2,d] * coeff[b]
       Physical gradient: ux[c*dE+d] = sum_{b,e2} D[q,b,c,e2] * invJ[e2,d] * coeff[b] */
    for (PetscInt b = 0; b < Nb; ++b) {
      const PetscScalar coeff_b = coeff_e[b];
      for (PetscInt c = 0; c < Nc; ++c) {
        for (PetscInt d = 0; d < dE; ++d) {
          PetscReal ref_grad = 0.0;
          for (PetscInt e2 = 0; e2 < dim; ++e2) ref_grad += D_q[b * Nc * dim + c * dim + e2] * invJ_eq[e2 * dE + d];
          ux_loc[c * dE + d] += coeff_b * ref_grad;
        }
      }
    }

    /* Call F0 (zeroth-order / source term) -- resolved at compile time */
    if (F0 != nullptr) {
      for (PetscInt c = 0; c < Nc; ++c) f0_loc[c] = 0.0;
      F0(dE, 1, 0, uOff_l, uOff_x_l, u_loc, nullptr, ux_loc, nullptr, nullptr, nullptr, nullptr, nullptr, t, x_eq, numConstants, constants, f0_loc);
      /* Accumulate directly into elemVec device buffer: ev_e[b] += B[q,b,c] * f0[c] * w * detJ
         Matches fekokkos.kokkos.cxx Phase 2: val_e[b] += B_q[bc] * f0_s[q*Nc+c] */
      for (PetscInt b = 0; b < Nb; ++b)
        for (PetscInt c = 0; c < Nc; ++c) ev_e[b] += B_q[b * Nc + c] * f0_loc[c] * wq;
    }

    /* Call F1 (first-order / flux term) -- resolved at compile time */
    if (F1 != nullptr) {
      for (PetscInt i = 0; i < Nc * dE; ++i) f1_loc[i] = 0.0;
      F1(dE, 1, 0, uOff_l, uOff_x_l, u_loc, nullptr, ux_loc, nullptr, nullptr, nullptr, nullptr, nullptr, t, x_eq, numConstants, constants, f1_loc);
      /* Accumulate directly into elemVec device buffer: ev_e[b] += phys_grad(b,c,d) * f1[c,d] * w * detJ
         phys_grad(b,c,d) = sum_{e2} D[q,b,c,e2] * invJ[e2,d]
         Matches fekokkos.kokkos.cxx Phase 2: val_e[b] += phys_grad * f1_s[(q*Nc+c)*dE+d] */
      for (PetscInt b = 0; b < Nb; ++b) {
        for (PetscInt c = 0; c < Nc; ++c) {
          for (PetscInt d = 0; d < dE; ++d) {
            PetscReal phys_grad = 0.0;
            for (PetscInt e2 = 0; e2 < dim; ++e2) phys_grad += D_q[b * Nc * dim + c * dim + e2] * invJ_eq[e2 * dE + d];
            ev_e[b] += phys_grad * f1_loc[c * dE + d] * wq;
          }
        }
      }
    }
  } /* end q */
}

/* PetscFEKokkosIntegrateJacobianCell<G0, G1, G2, G3>

   Device-callable template function that integrates the Jacobian for a
   single element e.  All four Jacobian callbacks are resolved at compile time.

   Template parameters (any may be nullptr to skip that term):
     G0  -- g0[fc*Nc+gc]         = df0[fc]/du[gc]
     G1  -- g1[fc*Nc*dE+gc*dE+d] = df0[fc]/d(gradu)[gc,d]
     G2  -- g2[fc*dE*Nc+d*Nc+gc] = df1[fc,d]/du[gc]
     G3  -- g3[fc*dE*Nc*dE+d*Nc*dE+gc*dE+e2] = df1[fc,d]/d(gradu)[gc,e2]

   Arguments: same geometry/tabulation arrays as the residual kernel, plus:
     Nb                            -- TOTAL DOFs per element (= T->Nb = Nb_scalar * Nc)
     elemMat[Ne * totDim * totDim] -- output element matrix (accumulated)
     u_tShift                      -- time-derivative shift (for implicit TS)

   Layout of elemMat (matches febasic.c PetscFEUpdateElementMat_Internal):
     elemMat[e * totDim*totDim + (fOff + b) * totDim + (gOff + b2)]
     where b, b2 = 0..Nb-1 (total DOFs, not scalar basis x component pairs)

   Note: For Poisson, only G3 is non-zero (g3[d*dE+e2] = delta_{d,e2}). */
template <PetscPointJacFn *G0, PetscPointJacFn *G1, PetscPointJacFn *G2, PetscPointJacFn *G3>
KOKKOS_INLINE_FUNCTION void PetscFEKokkosIntegrateJacobianCell(PetscInt e, PetscInt Nq, PetscInt Nb, PetscInt Nc, PetscInt dim, PetscInt dE, const PetscReal *B, /* [Nq * Nb * Nc]          */
                                                               const PetscReal   *D,                                                                             /* [Nq * Nb * Nc * dim]    */
                                                               const PetscReal   *w,                                                                             /* [Nq]                    */
                                                               const PetscReal   *invJ,                                                                          /* [Ne * Nq * dE * dE]     */
                                                               const PetscReal   *detJ,                                                                          /* [Ne * Nq]               */
                                                               const PetscReal   *coords,                                                                        /* [Ne * Nq * dE]          */
                                                               const PetscScalar *coeff,                                                                         /* [Ne * totDim]            */
                                                               PetscInt totDim, PetscInt uOff, PetscInt fOff, PetscInt gOff, PetscReal t, PetscReal u_tShift, PetscInt numConstants, const PetscScalar *constants, PetscScalar *elemMat /* [Ne * totDim * totDim] -- accumulated */
)
{
  constexpr PetscInt PETSCFE_KOKKOS_MAX_NC  = 3;
  constexpr PetscInt PETSCFE_KOKKOS_MAX_DE  = 3;
  constexpr PetscInt PETSCFE_KOKKOS_MAX_NCD = PETSCFE_KOKKOS_MAX_NC * PETSCFE_KOKKOS_MAX_DE;

  PetscScalar u_loc[PETSCFE_KOKKOS_MAX_NC];
  PetscScalar ux_loc[PETSCFE_KOKKOS_MAX_NCD];
  /* Jacobian output tensors -- g0[Nc*Nc], g1[Nc*Nc*dE], g2[Nc*dE*Nc], g3[Nc*dE*Nc*dE] */
  PetscScalar g0_loc[PETSCFE_KOKKOS_MAX_NC * PETSCFE_KOKKOS_MAX_NC];
  PetscScalar g1_loc[PETSCFE_KOKKOS_MAX_NC * PETSCFE_KOKKOS_MAX_NC * PETSCFE_KOKKOS_MAX_DE];
  PetscScalar g2_loc[PETSCFE_KOKKOS_MAX_NC * PETSCFE_KOKKOS_MAX_DE * PETSCFE_KOKKOS_MAX_NC];
  PetscScalar g3_loc[PETSCFE_KOKKOS_MAX_NCD * PETSCFE_KOKKOS_MAX_NCD];

  const PetscInt uOff_l[1]   = {0};
  const PetscInt uOff_x_l[1] = {0};

  /* Pointer to this element's block in elemMat */
  PetscScalar *em_e = &elemMat[e * totDim * totDim];

  for (PetscInt q = 0; q < Nq; ++q) {
    const PetscReal *invJ_eq = &invJ[(e * Nq + q) * dE * dE];
    const PetscReal  detJ_eq = detJ[e * Nq + q];
    const PetscReal *x_eq    = &coords[(e * Nq + q) * dE];
    const PetscReal  wq      = w[q] * detJ_eq;

    const PetscReal   *B_q     = &B[q * Nb * Nc];
    const PetscReal   *D_q     = &D[q * Nb * Nc * dim];
    const PetscScalar *coeff_e = &coeff[e * totDim + uOff];

    /* Zero per-qp scratch */
    for (PetscInt c = 0; c < Nc; ++c) u_loc[c] = 0.0;
    for (PetscInt i = 0; i < Nc * dE; ++i) ux_loc[i] = 0.0;

    /* Interpolate u and gradu (same convention as residual kernel and fekokkos.kokkos.cxx):
       b = 0..Nb-1 (total DOFs), coeff[b] -- b already encodes scalar basis + component. */
    for (PetscInt b = 0; b < Nb; ++b)
      for (PetscInt c = 0; c < Nc; ++c) u_loc[c] += B_q[b * Nc + c] * coeff_e[b];

    for (PetscInt b = 0; b < Nb; ++b) {
      const PetscScalar coeff_b = coeff_e[b];
      for (PetscInt c = 0; c < Nc; ++c) {
        for (PetscInt d = 0; d < dE; ++d) {
          PetscReal ref_grad = 0.0;
          for (PetscInt e2 = 0; e2 < dim; ++e2) ref_grad += D_q[b * Nc * dim + c * dim + e2] * invJ_eq[e2 * dE + d];
          ux_loc[c * dE + d] += coeff_b * ref_grad;
        }
      }
    }

    /* Evaluate Jacobian callbacks -- each resolved at compile time */
    if (G0 != nullptr) {
      for (PetscInt i = 0; i < Nc * Nc; ++i) g0_loc[i] = 0.0;
      G0(dE, 1, 0, uOff_l, uOff_x_l, u_loc, nullptr, ux_loc, nullptr, nullptr, nullptr, nullptr, nullptr, t, u_tShift, x_eq, numConstants, constants, g0_loc);
    }
    if (G1 != nullptr) {
      for (PetscInt i = 0; i < Nc * Nc * dE; ++i) g1_loc[i] = 0.0;
      G1(dE, 1, 0, uOff_l, uOff_x_l, u_loc, nullptr, ux_loc, nullptr, nullptr, nullptr, nullptr, nullptr, t, u_tShift, x_eq, numConstants, constants, g1_loc);
    }
    if (G2 != nullptr) {
      for (PetscInt i = 0; i < Nc * dE * Nc; ++i) g2_loc[i] = 0.0;
      G2(dE, 1, 0, uOff_l, uOff_x_l, u_loc, nullptr, ux_loc, nullptr, nullptr, nullptr, nullptr, nullptr, t, u_tShift, x_eq, numConstants, constants, g2_loc);
    }
    if (G3 != nullptr) {
      for (PetscInt i = 0; i < Nc * dE * Nc * dE; ++i) g3_loc[i] = 0.0;
      G3(dE, 1, 0, uOff_l, uOff_x_l, u_loc, nullptr, ux_loc, nullptr, nullptr, nullptr, nullptr, nullptr, t, u_tShift, x_eq, numConstants, constants, g3_loc);
    }

    /* Assemble element matrix contributions.
       Nb = Nb_total (total DOFs per element).  b and b2 each run 0..Nb-1.
       row = fOff + b,  col = gOff + b2  (one index per DOF, matching febasic.c).

       For each (b, b2) pair we sum over all component pairs (fc, gc):
         Test  physical grad: phi_grad[b,fc,d]  = sum_{e2} D[q,b,fc,e2] * invJ[e2,d]
         Trial physical grad: psi_grad[b2,gc,d] = sum_{e2} D[q,b2,gc,e2] * invJ[e2,d]

         G0 term: B[q,b,fc] * g0[fc,gc] * B[q,b2,gc] * wq
         G1 term: B[q,b,fc] * g1[fc,gc,d] * psi_grad[b2,gc,d] * wq
         G2 term: phi_grad[b,fc,d] * g2[fc,d,gc] * B[q,b2,gc] * wq
         G3 term: phi_grad[b,fc,d] * g3[fc,d,gc,e2] * psi_grad[b2,gc,e2] * wq
    */
    for (PetscInt b = 0; b < Nb; ++b) {
      const PetscInt row = fOff + b;

      for (PetscInt b2 = 0; b2 < Nb; ++b2) {
        const PetscInt col = gOff + b2;

        PetscScalar entry = 0.0;

        /* Sum over all component pairs (fc, gc) */
        for (PetscInt fc = 0; fc < Nc; ++fc) {
          /* Precompute test physical gradient phi_grad[d] for (b, fc) */
          PetscReal phi_grad[PETSCFE_KOKKOS_MAX_DE];
          for (PetscInt d = 0; d < dE; ++d) {
            phi_grad[d] = 0.0;
            for (PetscInt e2 = 0; e2 < dim; ++e2) phi_grad[d] += D_q[b * Nc * dim + fc * dim + e2] * invJ_eq[e2 * dE + d];
          }

          for (PetscInt gc = 0; gc < Nc; ++gc) {
            /* Precompute trial physical gradient psi_grad[d] for (b2, gc) */
            PetscReal psi_grad[PETSCFE_KOKKOS_MAX_DE];
            for (PetscInt d = 0; d < dE; ++d) {
              psi_grad[d] = 0.0;
              for (PetscInt e2 = 0; e2 < dim; ++e2) psi_grad[d] += D_q[b2 * Nc * dim + gc * dim + e2] * invJ_eq[e2 * dE + d];
            }

            /* G0: B_test * g0[fc,gc] * B_trial */
            if (G0 != nullptr) entry += B_q[b * Nc + fc] * g0_loc[fc * Nc + gc] * B_q[b2 * Nc + gc];

            /* G1: B_test * g1[fc,gc,d] * psi_grad[d] */
            if (G1 != nullptr)
              for (PetscInt d = 0; d < dE; ++d) entry += B_q[b * Nc + fc] * g1_loc[(fc * Nc + gc) * dE + d] * psi_grad[d];

            /* G2: phi_grad[d] * g2[fc,d,gc] * B_trial */
            if (G2 != nullptr)
              for (PetscInt d = 0; d < dE; ++d) entry += phi_grad[d] * g2_loc[(fc * dE + d) * Nc + gc] * B_q[b2 * Nc + gc];

            /* G3: phi_grad[d] * g3[fc,d,gc,e2] * psi_grad[e2] */
            if (G3 != nullptr)
              for (PetscInt d = 0; d < dE; ++d)
                for (PetscInt e2 = 0; e2 < dE; ++e2) entry += phi_grad[d] * g3_loc[((fc * dE + d) * Nc + gc) * dE + e2] * psi_grad[e2];
          }
        }

        em_e[row * totDim + col] += entry * wq;
      }
    }
  } /* end q */
}

/* Host-side geometry staging helper (internal, not for user code).

   Expands cgeom->invJ and cgeom->detJ into flat [Ne*Nq*...] arrays that
   the device kernel can index uniformly regardless of affine/non-affine.
   Also fills coords[Ne*Nq*dE] from cgeom->v (non-affine) or computes
   physical coords via CoordinatesRefToReal (affine).

   Declared here so that PetscFEKokkosComputeResidual / ComputeJacobian
   can call it without duplicating the geometry logic.

   This is a host-only function -- it uses PetscCall and PetscFunctionBegin. */
/* Local helper: map reference coordinates to physical coordinates.
   Formula: x[d] = v0[d] + sum_e J[d*dimReal + e] * (xi[e] - xi0[e])
   where xi0[] is the reference origin stored in cgeom->xi.
   Device helper -- no PetscFunctionBegin for Kokkos device functions */
static inline void PetscFEKokkosCoordRefToReal(PetscInt dimReal, PetscInt dimRef, const PetscReal xi0[], const PetscReal v0[], const PetscReal J[], const PetscReal xi[], PetscReal x[])
{
  for (PetscInt d = 0; d < dimReal; ++d) {
    x[d] = v0[d];
    for (PetscInt e = 0; e < dimRef; ++e) x[d] += J[d * dimReal + e] * (xi[e] - xi0[e]);
  }
}

static inline PetscErrorCode PetscFEKokkosExpandGeometry(PetscInt Ne, PetscInt Nq, PetscInt dim, PetscInt dE, PetscFEGeom *cgeom, const PetscReal *quadPoints, /* [Nq * dim] -- reference quadrature points */
                                                         PetscReal *h_invJ,                                                                                    /* [Ne * Nq * dE * dE] -- output */
                                                         PetscReal *h_detJ,                                                                                    /* [Ne * Nq]            -- output */
                                                         PetscReal *h_coords                                                                                   /* [Ne * Nq * dE]       -- output */
)
{
  const PetscInt  Np       = cgeom->numPoints;
  const PetscBool isAffine = cgeom->isAffine;

  PetscFunctionBegin;
  for (PetscInt e = 0; e < Ne; ++e) {
    const PetscReal *invJ_e = &cgeom->invJ[e * Np * dE * dE];
    const PetscReal *detJ_e = &cgeom->detJ[e * Np];
    const PetscReal *v0_e   = &cgeom->v[e * Np * dE];
    /* cgeom->J is NULL for non-affine elements (geometry stored per quad point).
     * Guard the pointer here; the affine branch below checks it is non-NULL. */
    const PetscReal *J_e  = cgeom->J ? &cgeom->J[e * Np * dE * dE] : NULL;
    const PetscReal *xi_e = cgeom->xi;
    PetscCheck(xi_e, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "cgeom->xi is NULL for affine element");

    for (PetscInt q = 0; q < Nq; ++q) {
      const PetscInt eq = e * Nq + q;

      if (isAffine) {
        /* Replicate single affine invJ across all Nq slots */
        for (PetscInt i = 0; i < dE * dE; ++i) h_invJ[eq * dE * dE + i] = invJ_e[i];
        h_detJ[eq] = detJ_e[0];
        /* Compute physical coords from reference quadrature point.
         * J_e must be non-NULL for affine elements (cgeom->J is always set
         * when isAffine == PETSC_TRUE by DMFieldCreateFEGeom). */
        PetscCheck(J_e, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "cgeom->J is NULL for affine element %d -- DMFieldCreateFEGeom must provide J", (int)e);
        PetscFEKokkosCoordRefToReal(dE, dim, xi_e, v0_e, J_e, &quadPoints[q * dim], &h_coords[eq * dE]);
      } else {
        /* Non-affine: geometry stored per quadrature point */
        for (PetscInt i = 0; i < dE * dE; ++i) h_invJ[eq * dE * dE + i] = invJ_e[q * dE * dE + i];
        h_detJ[eq] = detJ_e[q];
        for (PetscInt d = 0; d < dE; ++d) h_coords[eq * dE + d] = cgeom->v[(e * Np + q) * dE + d];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscFEKokkosComputeResidual<F0, F1>

   Host convenience function: extracts all data from PetscDS / PetscFEGeom,
   stages to device, launches PetscFEKokkosIntegrateResidualCell<F0,F1>
   via Kokkos::parallel_for, and copies results back.

   This replaces the two-phase path in PetscFEIntegrateResidual_Kokkos for
   users who have written KOKKOS_INLINE_FUNCTION callbacks.

   Signature matches PetscFEIntegrateResidual_Kokkos (fekokkos.kokkos.cxx)
   so it can be called from the same context.

   Restrictions (Phase 1.B):
     - dsAux == NULL (no auxiliary fields)
     - Single field (Nf == 1)
     - n0 <= 1, n1 <= 1 (at most one callback per term)
     - coefficients_t == NULL (no time derivative)

   Returns PETSC_SUCCESS on success; falls back gracefully if restrictions
   are violated (caller should use the Basic path in that case). */
template <PetscPointFn *F0, PetscPointFn *F1>
static PetscErrorCode PetscFEKokkosComputeResidual(PetscDS ds, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom, const PetscScalar *coefficients, const PetscScalar *coefficients_t, PetscReal t, PetscScalar *elemVec /* [Ne * totDim] -- accumulated */
)
{
  PetscFE            fe;
  PetscTabulation   *T;
  PetscQuadrature    quad;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           Nf, totDim, fOffset, field;
  PetscInt          *uOff, *uOff_x;
  PetscInt           numConstants;
  const PetscScalar *constants;
  PetscInt           Nq, Nb, Nc, dim, dE, qdim, qNc;

  PetscFunctionBegin;
  field = key.field;
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));
  PetscCall(PetscDSGetComponentOffsets(ds, &uOff));
  PetscCall(PetscDSGetComponentDerivativeOffsets(ds, &uOff_x));
  PetscCall(PetscDSGetFieldOffset(ds, field, &fOffset));
  PetscCall(PetscDSGetConstants(ds, &numConstants, &constants));
  PetscCall(PetscDSGetDiscretization(ds, field, (PetscObject *)&fe));
  PetscCall(PetscDSGetTabulation(ds, &T));
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));

  /* T[field]->Nb is the TOTAL number of DOFs per element = Nb_scalar * Nc.
     Pass Nb_total to the kernel so that the loop "b=0..Nb-1" with coeff[b]
     matches PetscFEEvaluateFieldJets_Internal and fekokkos.kokkos.cxx exactly.
     The B/D arrays are also sized [Nq * Nb_total * Nc], so B_q = &B[q*Nb*Nc]
     is correct with Nb = Nb_total. */
  Nc = T[field]->Nc;
  Nb = T[field]->Nb; /* total DOFs per element (= Nb_scalar * Nc for vector FE) */
  dE = cgeom->dimEmbed;

  /* Flat array sizes */
  const PetscInt nB      = Nq * Nb * Nc;
  const PetscInt nD      = Nq * Nb * Nc * dim;
  const PetscInt nInvJ   = Ne * Nq * dE * dE;
  const PetscInt nDetJ   = Ne * Nq;
  const PetscInt nCoords = Ne * Nq * dE;
  const PetscInt nCoeff  = Ne * totDim;
  const PetscInt nEV     = Ne * totDim;

  /* Stage static data (B, D, w).
   * PERFORMANCE NOTE: d_B, d_D, d_w are re-allocated and copied on every call.
   * For production use, cache these Views in PetscFEKokkosMaps (or a similar
   * persistent context) and copy only when Nq or Nb changes (i.e., after
   * PetscFEStageTabulation_Kokkos).  The current per-call allocation is
   * acceptable for correctness testing but adds ~1 cudaMalloc + H->D copy
   * per residual/Jacobian evaluation. */
  Kokkos::View<PetscReal *> d_B("fekokkos_tmpl_B", nB);
  Kokkos::View<PetscReal *> d_D("fekokkos_tmpl_D", nD);
  Kokkos::View<PetscReal *> d_w("fekokkos_tmpl_w", Nq);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_B(const_cast<PetscReal *>(T[field]->T[0]), nB);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_D(const_cast<PetscReal *>(T[field]->T[1]), nD);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_w(const_cast<PetscReal *>(quadWeights), Nq);
    Kokkos::deep_copy(d_B, h_B);
    Kokkos::deep_copy(d_D, h_D);
    Kokkos::deep_copy(d_w, h_w);
  }

  /* Expand and stage geometry (invJ, detJ, coords) */
  /* Host buffers for expanded geometry */
  PetscReal *h_invJ_buf, *h_detJ_buf, *h_coords_buf;
  PetscCall(PetscMalloc3(nInvJ, &h_invJ_buf, nDetJ, &h_detJ_buf, nCoords, &h_coords_buf));
  PetscCall(PetscFEKokkosExpandGeometry(Ne, Nq, dim, dE, cgeom, quadPoints, h_invJ_buf, h_detJ_buf, h_coords_buf));

  Kokkos::View<PetscReal *> d_invJ("fekokkos_tmpl_invJ", nInvJ);
  Kokkos::View<PetscReal *> d_detJ("fekokkos_tmpl_detJ", nDetJ);
  Kokkos::View<PetscReal *> d_coords("fekokkos_tmpl_coords", nCoords);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_invJ(h_invJ_buf, nInvJ);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_detJ(h_detJ_buf, nDetJ);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_coords(h_coords_buf, nCoords);
    Kokkos::deep_copy(d_invJ, hv_invJ);
    Kokkos::deep_copy(d_detJ, hv_detJ);
    Kokkos::deep_copy(d_coords, hv_coords);
  }
  PetscCall(PetscFree3(h_invJ_buf, h_detJ_buf, h_coords_buf));

  /* Stage coefficients */
  Kokkos::View<PetscScalar *> d_coeff("fekokkos_tmpl_coeff", nCoeff);
  {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_coeff(const_cast<PetscScalar *>(coefficients), nCoeff);
    Kokkos::deep_copy(d_coeff, hv_coeff);
  }

  /* Stage constants */
  Kokkos::View<PetscScalar *> d_constants("fekokkos_tmpl_constants", numConstants > 0 ? numConstants : 1);
  if (numConstants > 0) {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_const(const_cast<PetscScalar *>(constants), numConstants);
    Kokkos::deep_copy(d_constants, hv_const);
  }

  /* Allocate and zero output element vector */
  Kokkos::View<PetscScalar *> d_elemVec("fekokkos_tmpl_elemVec", nEV);
  Kokkos::deep_copy(d_elemVec, PetscScalar(0.0));

  /* Capture scalars for lambda */
  const PetscInt  Nq_     = Nq;
  const PetscInt  Nb_     = Nb;
  const PetscInt  Nc_     = Nc;
  const PetscInt  dim_    = dim;
  const PetscInt  dE_     = dE;
  const PetscInt  totDim_ = totDim;
  const PetscInt  uOff0   = uOff[field];
  const PetscInt  fOff    = fOffset;
  const PetscReal t_      = t;
  const PetscInt  nConst_ = numConstants;

  /* Launch kernel */
  Kokkos::parallel_for(
    "PetscFEKokkosComputeResidual", Kokkos::RangePolicy<>(0, Ne), KOKKOS_LAMBDA(const PetscInt e) {
      PetscFEKokkosIntegrateResidualCell<F0, F1>(e, Nq_, Nb_, Nc_, dim_, dE_, d_B.data(), d_D.data(), d_w.data(), d_invJ.data(), d_detJ.data(), d_coords.data(), d_coeff.data(), totDim_, uOff0, fOff, t_, nConst_, d_constants.data(), d_elemVec.data());
    });

  Kokkos::fence();

  /* Log GPU flops for the Kokkos kernel (Phase 2 basis assembly).
     Formula accounts for both interpolation and contraction:
       Interpolation (per element, per quad point):
         u[c]   = sum_b B[q,b,c] * coeff[b]:  Nb*Nc*2 flops
         u_x[c,d] = sum_b D[q,b,c,d] * coeff[b] * invJ[d,e2]: Nb*Nc*dim*2 flops
       Contraction (per element, per quad point, per basis, per component):
         f0 term: 2 flops (multiply by detJ*w, add to elemVec)
         f1 term: dE * (dim*2 + 1) flops (physical grad inner product + add)
     PetscLogGpuFlops also increments the total PetscLogFlops counter.
     GPU %F in -log_view = gpu_flops / total_flops * 100. */
  PetscCall(PetscLogGpuFlops((PetscLogDouble)Ne * Nq * (Nb * Nc * 2.0 + Nb * Nc * (PetscLogDouble)dim * 2.0) + (PetscLogDouble)Ne * Nb * Nc * Nq * (2.0 + dE * (dim * 2.0 + 1.0))));

  /* Copy result back and accumulate into elemVec */
  {
    auto h_elemVec = Kokkos::create_mirror_view(d_elemVec);
    Kokkos::deep_copy(h_elemVec, d_elemVec);
    const PetscScalar *src = h_elemVec.data();
    for (PetscInt i = 0; i < nEV; ++i) elemVec[i] += src[i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscFEKokkosComputeJacobian<G0, G1, G2, G3>

   Host convenience function: stages data, launches
   PetscFEKokkosIntegrateJacobianCell<G0,G1,G2,G3>, copies back.

   Any of G0-G3 may be nullptr to skip that Jacobian term.

   Arguments:
     ds, key, Ne, cgeom, coefficients -- same as ComputeResidual
     t, u_tShift                      -- time and time-derivative shift
     elemMat[Ne * totDim * totDim]    -- output element matrix (accumulated) */
template <PetscPointJacFn *G0, PetscPointJacFn *G1, PetscPointJacFn *G2, PetscPointJacFn *G3>
static PetscErrorCode PetscFEKokkosComputeJacobian(PetscDS ds, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom, const PetscScalar *coefficients, PetscReal t, PetscReal u_tShift, PetscScalar *elemMat /* [Ne * totDim * totDim] -- accumulated */
)
{
  PetscFE            fe;
  PetscTabulation   *T;
  PetscQuadrature    quad;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           Nf, totDim, fOffset, gOffset, field;
  PetscInt          *uOff, *uOff_x;
  PetscInt           numConstants;
  const PetscScalar *constants;
  PetscInt           Nq, Nb, Nc, dim, dE, qdim, qNc;

  PetscFunctionBegin;
  field = key.field;
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));
  PetscCall(PetscDSGetComponentOffsets(ds, &uOff));
  PetscCall(PetscDSGetComponentDerivativeOffsets(ds, &uOff_x));
  PetscCall(PetscDSGetFieldOffset(ds, field, &fOffset));
  /* For the Jacobian, test and trial fields may differ; for single-field
     problems (Poisson) they are the same.  gOffset == fOffset here. */
  gOffset = fOffset;
  PetscCall(PetscDSGetConstants(ds, &numConstants, &constants));
  PetscCall(PetscDSGetDiscretization(ds, field, (PetscObject *)&fe));
  PetscCall(PetscDSGetTabulation(ds, &T));
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));

  Nb = T[field]->Nb;
  Nc = T[field]->Nc;
  dE = cgeom->dimEmbed;

  const PetscInt nB      = Nq * Nb * Nc;
  const PetscInt nD      = Nq * Nb * Nc * dim;
  const PetscInt nInvJ   = Ne * Nq * dE * dE;
  const PetscInt nDetJ   = Ne * Nq;
  const PetscInt nCoords = Ne * Nq * dE;
  const PetscInt nCoeff  = Ne * totDim;
  const PetscInt nEM     = Ne * totDim * totDim;

  /* Stage static data */
  Kokkos::View<PetscReal *> d_B("fekokkos_jac_B", nB);
  Kokkos::View<PetscReal *> d_D("fekokkos_jac_D", nD);
  Kokkos::View<PetscReal *> d_w("fekokkos_jac_w", Nq);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_B(const_cast<PetscReal *>(T[field]->T[0]), nB);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_D(const_cast<PetscReal *>(T[field]->T[1]), nD);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_w(const_cast<PetscReal *>(quadWeights), Nq);
    Kokkos::deep_copy(d_B, h_B);
    Kokkos::deep_copy(d_D, h_D);
    Kokkos::deep_copy(d_w, h_w);
  }

  /* Expand and stage geometry */
  PetscReal *h_invJ_buf, *h_detJ_buf, *h_coords_buf;
  PetscCall(PetscMalloc3(nInvJ, &h_invJ_buf, nDetJ, &h_detJ_buf, nCoords, &h_coords_buf));
  PetscCall(PetscFEKokkosExpandGeometry(Ne, Nq, dim, dE, cgeom, quadPoints, h_invJ_buf, h_detJ_buf, h_coords_buf));

  Kokkos::View<PetscReal *> d_invJ("fekokkos_jac_invJ", nInvJ);
  Kokkos::View<PetscReal *> d_detJ("fekokkos_jac_detJ", nDetJ);
  Kokkos::View<PetscReal *> d_coords("fekokkos_jac_coords", nCoords);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_invJ(h_invJ_buf, nInvJ);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_detJ(h_detJ_buf, nDetJ);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_coords(h_coords_buf, nCoords);
    Kokkos::deep_copy(d_invJ, hv_invJ);
    Kokkos::deep_copy(d_detJ, hv_detJ);
    Kokkos::deep_copy(d_coords, hv_coords);
  }
  PetscCall(PetscFree3(h_invJ_buf, h_detJ_buf, h_coords_buf));

  /* Stage coefficients */
  Kokkos::View<PetscScalar *> d_coeff("fekokkos_jac_coeff", nCoeff);
  {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_coeff(const_cast<PetscScalar *>(coefficients), nCoeff);
    Kokkos::deep_copy(d_coeff, hv_coeff);
  }

  /* Stage constants */
  Kokkos::View<PetscScalar *> d_constants("fekokkos_jac_constants", numConstants > 0 ? numConstants : 1);
  if (numConstants > 0) {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_const(const_cast<PetscScalar *>(constants), numConstants);
    Kokkos::deep_copy(d_constants, hv_const);
  }

  /* Allocate and zero output element matrix */
  Kokkos::View<PetscScalar *> d_elemMat("fekokkos_jac_elemMat", nEM);
  Kokkos::deep_copy(d_elemMat, PetscScalar(0.0));

  /* Capture scalars for lambda */
  const PetscInt  Nq_     = Nq;
  const PetscInt  Nb_     = Nb;
  const PetscInt  Nc_     = Nc;
  const PetscInt  dim_    = dim;
  const PetscInt  dE_     = dE;
  const PetscInt  totDim_ = totDim;
  const PetscInt  uOff0   = uOff[field];
  const PetscInt  fOff    = fOffset;
  const PetscInt  gOff    = gOffset;
  const PetscReal t_      = t;
  const PetscReal tShift_ = u_tShift;
  const PetscInt  nConst_ = numConstants;

  /* Launch kernel */
  Kokkos::parallel_for(
    "PetscFEKokkosComputeJacobian", Kokkos::RangePolicy<>(0, Ne), KOKKOS_LAMBDA(const PetscInt e) {
      PetscFEKokkosIntegrateJacobianCell<G0, G1, G2, G3>(e, Nq_, Nb_, Nc_, dim_, dE_, d_B.data(), d_D.data(), d_w.data(), d_invJ.data(), d_detJ.data(), d_coords.data(), d_coeff.data(), totDim_, uOff0, fOff, gOff, t_, tShift_, nConst_, d_constants.data(),
                                                         d_elemMat.data());
    });

  Kokkos::fence();

  /* Log GPU flops for the Kokkos Jacobian kernel.
     Per element, per quad point, per (test basis b, test comp fc,
     trial basis b2, trial comp gc):
       G3 term (dominant): 2*dE*dE flops (phi_grad * g3 * psi_grad)
       G0 term:            2 flops (B_test * g0 * B_trial)
       G1/G2 terms:        2*dE flops each (grad-value contractions)
     Simplified to the G3-dominant count (matches fekokkos.kokkos.cxx convention):
       Ne * Nq * Nb * Nc * Nb * Nc * 2 * dE * dE */
  PetscCall(PetscLogGpuFlops((PetscLogDouble)Ne * Nq * Nb * Nc * Nb * Nc * 2.0 * dE * dE));

  /* Copy result back and accumulate */
  {
    auto h_elemMat = Kokkos::create_mirror_view(d_elemMat);
    Kokkos::deep_copy(h_elemMat, d_elemMat);
    const PetscScalar *src = h_elemMat.data();
    for (PetscInt i = 0; i < nEM; ++i) elemMat[i] += src[i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* COO Assembly Infrastructure

   PetscFEKokkosMaps -- precomputed assembly maps for COO scatter.

   Built once at setup time from DMPlex closure indices by probing;
   staged to device so that the Jacobian and residual kernels can scatter
   element contributions directly into the global COO arrays without any
   host round-trip.

   Phase 1.C design (no AMR, no hanging nodes):
     gIdx[e * Nb + b]   = global DOF index for element e, basis function b.
                          Negative values = Dirichlet-constrained DOF (skip).
     active_idx[e*Nb+b] = sequential index among active (non-Dirichlet) bases
                          for element e; -1 if constrained.
     Nb_active[e]       = number of active bases for element e.
     coo_elem_offsets[e] = index of first COO entry for element e in the flat
                           COO arrays (prefix sum over Nb_active[e]^2).

   Phase 1.D additions (p4est AMR, hanging nodes via Landau probing):
     num_reduced        = number of constrained (hanging) DOFs across all elements.
     num_face           = max parent DOFs per constrained DOF (degree+1 for 2D edges).
     c_maps_gid[idx*num_face+q]   = global DOF of q-th parent for constraint idx.
     c_maps_scale[idx*num_face+q] = interpolation weight for that parent.
     coo_elem_point_offsets[e*(Nb+1)+b] = prefix sum of expanded COO rows per
                                          basis function within element e.
     fullNb[e]          = total expanded basis count for element e
                          (Nb for no constraints, > Nb when hanging nodes present).

   Backward compatibility: when num_reduced == 0 (uniform mesh or simplex),
   the Phase 1.C code path works unchanged. */

/* typedef matching LandauIdx for future unification */
typedef PetscInt PetscFEKokkosIdx;

/* Constraint map entry: one parent DOF with its interpolation weight.
 * Mirrors the Landau LandauIdx/scale pair in plexland.c. */
typedef struct {
  PetscInt    gid;   /* global DOF index of parent face DOF; -1 = unused */
  PetscScalar scale; /* interpolation weight */
} PetscFEKokkosConstraint;

  /* Maximum parent DOFs per constrained DOF.
 * For Q2 in 2D: 3 DOFs on a face edge.  8 matches LANDAU_MAX_Q_FACE. */
  #if !defined(PETSCFE_KOKKOS_MAX_FACE)
    #define PETSCFE_KOKKOS_MAX_FACE 8
  #endif

typedef struct {
  /* Phase 1.C fields */
  PetscInt   num_elements; /* Ne -- number of owned cells */
  PetscInt   num_dof;      /* total global DOFs (for bounds checking) */
  PetscInt   local_dof;    /* local DOFs including ghosts (local Vec size) */
  PetscInt   Nb;           /* basis functions per element (single field) */
  PetscInt   totDim;       /* total DOFs per element across all fields */
  PetscCount coo_size;     /* total COO entries */

  /* gIdx[e * Nb + b]:
   *   >= 0: unconstrained, value = global DOF index
   *   <  0: constrained (Dirichlet or hanging), idx = -(value) - 1 into c_maps */
  Kokkos::View<PetscFEKokkosIdx *> d_gIdx;
  PetscFEKokkosIdx                *h_gIdx; /* host mirror (PetscMalloc'd) */

  /* lIdx[e * Nb + b]:
   *   >= 0: local DOF index (into local Vec with ghost entries)
   *   == -1: Dirichlet (constrained away)
   * Used by FormResidual_COO to scatter into locF before DMLocalToGlobal.
   * Built from section (not globalSection) so ghost DOFs are always in-bounds. */
  Kokkos::View<PetscInt *> d_lIdx;
  PetscInt                *h_lIdx; /* host mirror (PetscMalloc'd) */

  /* active_idx[e * Nb + b] = sequential active index for (e,b); -1 if constrained.
   * Used only in Phase 1.C (no-constraint) scatter path. */
  Kokkos::View<PetscInt *> d_active_idx;
  PetscInt                *h_active_idx; /* host mirror (PetscMalloc'd) */

  /* Nb_active[e] = count of non-Dirichlet bases for element e (Phase 1.C) */
  Kokkos::View<PetscInt *> d_Nb_active;
  PetscInt                *h_Nb_active; /* host mirror (PetscMalloc'd) */

  /* coo_elem_offsets[e] = first COO index for element e; [Ne+1] prefix sum */
  Kokkos::View<PetscInt *> d_coo_elem_offsets;
  PetscInt                *h_coo_elem_offsets; /* host mirror (PetscMalloc'd) */

  /* Phase 1.D fields (constraint / hanging-node support) */

  /* Number of constrained (hanging) DOFs discovered by probing across all elements */
  PetscInt num_reduced;

  /* Max parent DOFs per constrained DOF (degree+1 for 2D edges) */
  PetscInt num_face;

  /* c_maps_gid[idx * num_face + q]   = global DOF of q-th parent for constraint idx
   * c_maps_scale[idx * num_face + q] = interpolation weight for that parent
   * Flat layout: num_reduced * num_face entries each. */
  Kokkos::View<PetscInt *>    d_c_maps_gid;
  Kokkos::View<PetscScalar *> d_c_maps_scale;
  PetscInt                   *h_c_maps_gid;   /* host mirror (PetscMalloc'd) */
  PetscScalar                *h_c_maps_scale; /* host mirror (PetscMalloc'd) */

  /* coo_elem_point_offsets[e * (Nb+1) + b]:
   * prefix sum of expanded COO rows per basis function within element e.
   * coo_elem_point_offsets[e*(Nb+1) + Nb] = fullNb[e]. */
  Kokkos::View<PetscInt *> d_coo_elem_point_offsets;
  PetscInt                *h_coo_elem_point_offsets; /* host mirror (PetscMalloc'd) */

  /* fullNb[e] = total expanded basis count for element e
   * (Nb for no constraints, > Nb when hanging nodes present) */
  Kokkos::View<PetscInt *> d_fullNb;
  PetscInt                *h_fullNb; /* host mirror (PetscMalloc'd) */
} PetscFEKokkosMaps;

/* PetscFEKokkosCreateMaps
   Build gIdx and constraint maps on the host by probing.  For each element
   and basis function, set a unit element matrix and call
   DMPlexGetClosureIndices(useConstraints=TRUE) to discover whether the DOF
   is unconstrained (diagonal ~ 1) or constrained (0 < c < 1).

   Call PetscFEKokkosStageMaps afterwards to copy to device. */
static inline PetscErrorCode PetscFEKokkosCreateMaps(DM dm, PetscFEKokkosMaps *maps)
{
  PetscSection section, globalSection;
  PetscInt     cStart, cEnd, Ne, Nb, totDim, num_dof;
  PetscDS      ds;
  PetscFE      fe;
  PetscInt     dim;
  DM           plex; /* adapted DMPlex view (may == dm for plain DMPlex) */
  PetscBool    isPlex;

  PetscFunctionBegin;
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
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  {
    PetscTabulation *T;
    PetscCall(PetscDSGetTabulation(ds, &T));
    Nb = T[0]->Nb;
  }
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));

  /* Get total global DOFs for bounds checking */
  num_dof = 0;
  {
    Vec gvec;
    PetscCall(DMGetGlobalVector(dm, &gvec));
    PetscCall(VecGetSize(gvec, &num_dof));
    PetscCall(DMRestoreGlobalVector(dm, &gvec));
  }

  /* Get local DOF count (including ghosts) for local Vec sizing */
  {
    Vec lvec;
    PetscCall(DMGetLocalVector(dm, &lvec));
    PetscCall(VecGetLocalSize(lvec, &maps->local_dof));
    PetscCall(DMRestoreLocalVector(dm, &lvec));
  }

  maps->num_elements = Ne;
  maps->num_dof      = num_dof;
  maps->Nb           = Nb;
  maps->totDim       = totDim;
  maps->num_reduced  = 0;

  /* num_face: number of DOFs on a face edge = degree + 1 for 2D quads.
   * Landau computes this as pow(num_face, dim-1) for higher dimensions.
   * For tensor-product quads: Nb = (degree+1)^dim, so degree+1 = round(Nb^(1/dim)).
   * For simplices or dim==1 we fall back to Nb (no constraint expansion needed).
   * num_face computation assumes tensor-product elements; simplex not yet supported for constraints */
  {
    PetscInt nf;
    if (dim >= 2) {
      /* degree+1 = round(Nb^(1/dim)) via integer search */
      nf = 1;
      while ((nf + 1) * (nf + 1) <= Nb) ++nf; /* works for dim==2 */
      if (dim == 3) {
        /* For dim==3: Nb = nf^3, so nf = round(Nb^(1/3)) */
        nf = (PetscInt)(PetscPowReal((PetscReal)Nb, 1.0 / 3.0) + 0.5);
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
    /* Clamp to PETSCFE_KOKKOS_MAX_FACE */
    if (maps->num_face > PETSCFE_KOKKOS_MAX_FACE) maps->num_face = PETSCFE_KOKKOS_MAX_FACE;
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

  /* Landau probing loop: for each element e and basis function q,
   * set a unit element matrix and call DMPlexGetClosureIndices to discover
   * whether the DOF is unconstrained or constrained (hanging node).
   *
   * Reference: plexland.c:1516-1601 */
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
       * Classification (matching Landau plexland.c:1531-1585):
       *   1. diag ~ 1.0 -> unconstrained active DOF
       *   2. 0 < diag < 1 -> hanging-node constrained DOF
       *   3. No non-zero diagonal found -> Dirichlet (fully constrained away)
       *
       * IMPORTANT: Do NOT use indices[f] < 0 to classify as Dirichlet.
       * A constrained DOF may have its first parent at a Dirichlet index,
       * but the DOF itself is a hanging node, not Dirichlet.
       */
      PetscBool found = PETSC_FALSE;
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
             * Extract constraint weights by summing rows of the outer product.
             * Reference: plexland.c:1541-1584 */
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

      /* Restore closure indices; check if elMat was reallocated */
      PetscInt savedNumIndices = numIndices;
      PetscCall(DMPlexRestoreClosureIndices(plex, section, globalSection, cStart + e, PETSC_TRUE, &numIndices, &indices, NULL, &elMat));
      if (elMat != valuesOrig) {
        PetscCall(DMRestoreWorkArray(plex, savedNumIndices * savedNumIndices, MPIU_SCALAR, &elMat));
        /* Re-allocate for next probe */
        PetscCall(PetscMalloc1(totDim * totDim, &elMat));
      }
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

  /* Build active_idx, Nb_active, and coo_elem_offsets.
   *
   * Phase 1.D: coo_size = sum_e fullNb[e]^2 (accounts for constraint expansion).
   * Phase 1.C: fullNb[e] == Nb_active[e] when num_reduced == 0.
   *
   * Reference: plexland.c:1619-1648 */
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

/* PetscFEKokkosStageMaps
   Deep-copy all host arrays to device Kokkos::Views.
   Must be called after PetscFEKokkosCreateMaps and before any kernel launch. */
static inline PetscErrorCode PetscFEKokkosStageMaps(PetscFEKokkosMaps *maps)
{
  const PetscInt Ne       = maps->num_elements;
  const PetscInt Nb       = maps->Nb;
  const PetscInt num_face = maps->num_face;

  PetscFunctionBegin;
  /* gIdx */
  maps->d_gIdx = Kokkos::View<PetscFEKokkosIdx *>("fekokkos_coo_gIdx", Ne * Nb);
  {
    Kokkos::View<PetscFEKokkosIdx *, Kokkos::HostSpace> hv(maps->h_gIdx, Ne * Nb);
    Kokkos::deep_copy(maps->d_gIdx, hv);
  }

  /* lIdx */
  maps->d_lIdx = Kokkos::View<PetscInt *>("fekokkos_coo_lIdx", Ne * Nb);
  {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_lIdx, Ne * Nb);
    Kokkos::deep_copy(maps->d_lIdx, hv);
  }

  /* active_idx */
  maps->d_active_idx = Kokkos::View<PetscInt *>("fekokkos_coo_active_idx", Ne * Nb);
  {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_active_idx, Ne * Nb);
    Kokkos::deep_copy(maps->d_active_idx, hv);
  }

  /* Nb_active */
  maps->d_Nb_active = Kokkos::View<PetscInt *>("fekokkos_coo_Nb_active", Ne);
  {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_Nb_active, Ne);
    Kokkos::deep_copy(maps->d_Nb_active, hv);
  }

  /* coo_elem_offsets */
  maps->d_coo_elem_offsets = Kokkos::View<PetscInt *>("fekokkos_coo_offsets", Ne + 1);
  {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_coo_elem_offsets, Ne + 1);
    Kokkos::deep_copy(maps->d_coo_elem_offsets, hv);
  }

  /* Phase 1.D: constraint maps */
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

  /* coo_elem_point_offsets */
  maps->d_coo_elem_point_offsets = Kokkos::View<PetscInt *>("fekokkos_coo_pt_offsets", Ne * (Nb + 1));
  {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_coo_elem_point_offsets, Ne * (Nb + 1));
    Kokkos::deep_copy(maps->d_coo_elem_point_offsets, hv);
  }

  /* fullNb */
  maps->d_fullNb = Kokkos::View<PetscInt *>("fekokkos_fullNb", Ne);
  {
    Kokkos::View<PetscInt *, Kokkos::HostSpace> hv(maps->h_fullNb, Ne);
    Kokkos::deep_copy(maps->d_fullNb, hv);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PetscFEKokkosPreallocateCOO
   Build COO row/col arrays from h_gIdx (with constraint expansion) and
   call MatSetPreallocationCOO.
   Must be called after PetscFEKokkosCreateMaps (host arrays must be valid).
   The matrix J must already exist (e.g. from DMCreateMatrix).

   Phase 1.D: constrained DOFs expand to num_face rows/columns.
   Phase 1.C: when num_reduced == 0, fullNb[e] == Nb_active[e] and the
              behavior is identical to the original implementation.

   Reference: plexland.c:1650-1688 */
static inline PetscErrorCode PetscFEKokkosPreallocateCOO(PetscFEKokkosMaps *maps, Mat J)
{
  const PetscInt   Ne       = maps->num_elements;
  const PetscInt   Nb       = maps->Nb;
  const PetscCount coo_size = maps->coo_size;
  const PetscInt   num_face = maps->num_face;
  PetscInt        *coo_i, *coo_j;

  PetscFunctionBegin;
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
         * Formula mirrors plexland.c:1674:
         *   idx0 = off + fullNb * pt_off_b + nr * pt_off_b2 */
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

/* PetscFEKokkosMapsDestroy
   Free all host-side PetscMalloc'd arrays.  Device Views are reference-
   counted by Kokkos and freed automatically when they go out of scope or
   when the struct is destroyed. */
static inline PetscErrorCode PetscFEKokkosMapsDestroy(PetscFEKokkosMaps *maps)
{
  PetscFunctionBegin;
  PetscCall(PetscFree7(maps->h_gIdx, maps->h_lIdx, maps->h_active_idx, maps->h_Nb_active, maps->h_coo_elem_offsets, maps->h_fullNb, maps->h_coo_elem_point_offsets));
  PetscCall(PetscFree2(maps->h_c_maps_gid, maps->h_c_maps_scale));
  /* Reset device Views to empty (releases Kokkos reference count) */
  maps->d_gIdx                   = Kokkos::View<PetscFEKokkosIdx *>();
  maps->d_lIdx                   = Kokkos::View<PetscInt *>();
  maps->d_active_idx             = Kokkos::View<PetscInt *>();
  maps->d_Nb_active              = Kokkos::View<PetscInt *>();
  maps->d_coo_elem_offsets       = Kokkos::View<PetscInt *>();
  maps->d_c_maps_gid             = Kokkos::View<PetscInt *>();
  maps->d_c_maps_scale           = Kokkos::View<PetscScalar *>();
  maps->d_coo_elem_point_offsets = Kokkos::View<PetscInt *>();
  maps->d_fullNb                 = Kokkos::View<PetscInt *>();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PETSc-level API for GPU-resident FEM assembly

   Lifecycle:
     PetscFEKokkosSetUp(dm, maps, J)   -- build maps, stage to device, preallocate COO
     DMPlexSNESComputeResidualFEM_Kokkos<f0,f1>(snes, X, F, maps)
     DMPlexSNESComputeJacobianFEM_Kokkos<G0,G1,G2,G3>(snes, X, J, Jp, maps)
     PetscFEKokkosMapsDestroy(maps)    -- free host arrays (device Views auto-freed)

   Recommended order:
     SNESSetFromOptions(snes)          -- triggers DMSetUp -> Kokkos::initialize
     PetscFEKokkosSetUp(dm, maps, J)  -- Kokkos already initialized */

/* PetscFEKokkosSetUp
   Build assembly maps, stage to device, and preallocate the COO matrix J.

   Parameters:
     dm   -- the DM with FE discretization already attached (after DMCreateDS)
     maps -- output; caller must pass a pointer to an uninitialised
             PetscFEKokkosMaps (stack or heap).
     J    -- matrix to preallocate via MatSetPreallocationCOO; must already
             exist (e.g. from DMCreateMatrix).

   Calls (in order):
     PetscFEKokkosCreateMaps(dm, maps)
     PetscKokkosInitializeCheck()
     PetscFEKokkosStageMaps(maps)
     PetscFEKokkosPreallocateCOO(maps, J) */
static inline PetscErrorCode PetscFEKokkosSetUp(DM dm, PetscFEKokkosMaps *maps, Mat J)
{
  PetscFunctionBegin;
  PetscCall(PetscFEKokkosCreateMaps(dm, maps));
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(PetscFEKokkosStageMaps(maps));
  PetscCall(PetscFEKokkosPreallocateCOO(maps, J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DMPlexSNESComputeResidualFEM_Kokkos<f0, f1>

   GPU-resident SNES residual callback.  Drop-in replacement for
   DMPlexSNESComputeResidualFEM (the CPU path) when using COO assembly.

   Template parameters:
     f0 -- KOKKOS_INLINE_FUNCTION void(PETSCFE_KOKKOS_POINT_ARGS, PetscScalar[])
          source term (zeroth-order residual)
     f1 -- KOKKOS_INLINE_FUNCTION void(PETSCFE_KOKKOS_POINT_ARGS, PetscScalar[])
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
template <PetscPointFn *f0, PetscPointFn *f1>
static PetscErrorCode DMPlexSNESComputeResidualFEM_Kokkos(SNES snes, Vec X, Vec F, void *ctx_ptr)
{
  PetscFEKokkosMaps *ctx = (PetscFEKokkosMaps *)ctx_ptr;
  DM                 dm;
  PetscDS            ds;
  PetscFE            fe;
  PetscTabulation   *T;
  PetscQuadrature    quad;
  DMField            coordField;
  IS                 cellIS;
  PetscFEGeom       *fullGeom  = NULL;
  PetscFEGeom       *chunkGeom = NULL;
  Vec                locX;
  PetscScalar       *u_arr = NULL, *u_t_arr = NULL, *a_arr = NULL;
  PetscInt           depth, cStart, cEnd, totDim;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           Nq, Nb, Nc, dim, dE, qdim, qNc;
  PetscInt           numConstants;
  const PetscScalar *constants;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetConstants(ds, &numConstants, &constants));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(PetscDSGetTabulation(ds, &T));
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));

  Nb = T[0]->Nb;
  Nc = T[0]->Nc;

  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetStratumIS(dm, "depth", depth, &cellIS));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  const PetscInt Ne = cEnd - cStart;

  /* Zero the residual */
  PetscCall(VecZeroEntries(F));

  /* Get local solution with BCs */
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, locX));
  /* Force device->host sync if locX is a device Vec: DMPlexInsertBoundaryValues
   * calls VecGetArray(locX) internally which requires host-authoritative data.
   * DMPlex/PetscFE do not support GPU Vecs. */
  {
    PetscScalar *tmp;
    PetscMemType mtype;
    PetscCall(VecGetArrayAndMemType(locX, &tmp, &mtype));
    PetscCall(VecRestoreArrayAndMemType(locX, &tmp));
    if (PetscMemTypeDevice(mtype)) {
      PetscCall(VecGetArray(locX, &tmp));     /* KokkosDualViewSyncHost: device->host */
      PetscCall(VecRestoreArray(locX, &tmp)); /* marks host modified */
    }
  }
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, 0.0, NULL, NULL, NULL));

  /* Build geometry */
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, PETSC_FEGEOM_BASIC, &fullGeom));
  PetscCall(PetscFEGeomGetChunk(fullGeom, cStart, cEnd, &chunkGeom));
  dE = chunkGeom->dimEmbed;

  /* Runtime bounds check: Nc and dE must fit in the kernel's stack arrays */
  PetscCheck(Nc <= 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Nc %" PetscInt_FMT " exceeds PETSCFE_KOKKOS_MAX_NC 3", (PetscInt)Nc);
  PetscCheck(dE <= 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "dE %" PetscInt_FMT " exceeds PETSCFE_KOKKOS_MAX_DE 3", (PetscInt)dE);

  /* Get element coefficients */
  PetscCall(DMPlexGetCellFields(dm, cellIS, locX, NULL, NULL, &u_arr, &u_t_arr, &a_arr));

  /* Stage geometry and tabulation to device */
  const PetscInt nB      = Nq * Nb * Nc;
  const PetscInt nD      = Nq * Nb * Nc * dim;
  const PetscInt nInvJ   = Ne * Nq * dE * dE;
  const PetscInt nDetJ   = Ne * Nq;
  const PetscInt nCoords = Ne * Nq * dE;
  const PetscInt nCoeff  = Ne * totDim;

  Kokkos::View<PetscReal *> d_B("res_B", nB);
  Kokkos::View<PetscReal *> d_D("res_D", nD);
  Kokkos::View<PetscReal *> d_w("res_w", Nq);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_B(const_cast<PetscReal *>(T[0]->T[0]), nB);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_D(const_cast<PetscReal *>(T[0]->T[1]), nD);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_w(const_cast<PetscReal *>(quadWeights), Nq);
    Kokkos::deep_copy(d_B, h_B);
    Kokkos::deep_copy(d_D, h_D);
    Kokkos::deep_copy(d_w, h_w);
  }

  PetscReal *h_invJ_buf, *h_detJ_buf, *h_coords_buf;
  PetscCall(PetscMalloc3(nInvJ, &h_invJ_buf, nDetJ, &h_detJ_buf, nCoords, &h_coords_buf));
  PetscCall(PetscFEKokkosExpandGeometry(Ne, Nq, dim, dE, chunkGeom, quadPoints, h_invJ_buf, h_detJ_buf, h_coords_buf));

  Kokkos::View<PetscReal *> d_invJ("res_invJ", nInvJ);
  Kokkos::View<PetscReal *> d_detJ("res_detJ", nDetJ);
  Kokkos::View<PetscReal *> d_coords("res_coords", nCoords);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_invJ(h_invJ_buf, nInvJ);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_detJ(h_detJ_buf, nDetJ);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_coords(h_coords_buf, nCoords);
    Kokkos::deep_copy(d_invJ, hv_invJ);
    Kokkos::deep_copy(d_detJ, hv_detJ);
    Kokkos::deep_copy(d_coords, hv_coords);
  }
  PetscCall(PetscFree3(h_invJ_buf, h_detJ_buf, h_coords_buf));

  Kokkos::View<PetscScalar *> d_coeff("res_coeff", nCoeff);
  {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_coeff(const_cast<PetscScalar *>(u_arr), nCoeff);
    Kokkos::deep_copy(d_coeff, hv_coeff);
  }

  /* Stage DS constants to device (numConstants==0 -> no-op) */
  Kokkos::View<PetscScalar *> d_constants("res_constants", numConstants > 0 ? numConstants : 1);
  const PetscInt              numConstants_ = numConstants;
  if (numConstants > 0) {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_constants(const_cast<PetscScalar *>(constants), numConstants);
    Kokkos::deep_copy(d_constants, hv_constants);
  }

  /* Scatter d_elemVec into a LOCAL vector (which includes ghost entries) using
   * local DOF indices (d_lIdx), then assemble to the global F via
   * DMLocalToGlobal(ADD_VALUES).  This is the correct parallel FEM pattern:
   *
   *   locF[lIdx] += elemVec[e,b]   (atomic, all DOFs in-bounds incl. ghosts)
   *   DMLocalToGlobal(locF, ADD_VALUES, F)   (MPI reduction for ghost DOFs)
   *
   * Ghost DOFs are included in the local vector (size = local_dof_with_ghosts).
   * The atomic_add is required because multiple elements may share a DOF. */
  Vec          locF;
  PetscScalar *locF_arr;
  PetscMemType locF_memtype;
  PetscCall(DMGetLocalVector(dm, &locF));
  PetscCall(VecZeroEntries(locF));
  PetscCall(VecGetArrayAndMemType(locF, &locF_arr, &locF_memtype));

  const PetscInt local_dof_with_ghosts = ctx->local_dof;

  using DeviceUnmanaged = Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using HostUnmanaged   = Kokkos::View<PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  Kokkos::View<PetscScalar *> d_F;
  DeviceUnmanaged             d_F_unmanaged;
  HostUnmanaged               h_F_unmanaged;

  if (locF_memtype == PETSC_MEMTYPE_DEVICE) {
    d_F_unmanaged = DeviceUnmanaged(locF_arr, local_dof_with_ghosts);
  } else {
    h_F_unmanaged = HostUnmanaged(locF_arr, local_dof_with_ghosts);
    d_F           = Kokkos::View<PetscScalar *>("res_F", local_dof_with_ghosts);
    Kokkos::deep_copy(d_F, h_F_unmanaged);
  }

  /* Capture maps for lambda */
  auto d_lIdx         = ctx->d_lIdx;
  auto d_c_maps_gid   = ctx->d_c_maps_gid;
  auto d_c_maps_scale = ctx->d_c_maps_scale;

  /* Capture scalars */
  const PetscInt Nq_     = Nq;
  const PetscInt Nb_     = Nb;
  const PetscInt Nc_     = Nc;
  const PetscInt dim_    = dim;
  const PetscInt dE_     = dE;
  const PetscInt totDim_ = totDim;

  /* Allocate full Ne*totDim element vector array on device */
  Kokkos::View<PetscScalar *> d_elemVec("res_elemVec", (PetscCount)Ne * totDim);
  Kokkos::deep_copy(d_elemVec, PetscScalar(0.0));

  /* Pass 1: integrate residual for all elements into d_elemVec */
  /* TODO: pass actual t for time-dependent problems; currently steady-state only */
  Kokkos::parallel_for(
    "DMPlexSNESComputeResidualFEM_Kokkos_integrate", Kokkos::RangePolicy<>(0, Ne), KOKKOS_LAMBDA(const PetscInt e) {
      PetscFEKokkosIntegrateResidualCell<f0, f1>(e, Nq_, Nb_, Nc_, dim_, dE_, d_B.data(), d_D.data(), d_w.data(), d_invJ.data(), d_detJ.data(), d_coords.data(), d_coeff.data(), totDim_, 0, 0, 0.0, numConstants_, d_constants.data(), d_elemVec.data());
    });
  Kokkos::fence();

  /* Pass 2: scatter d_elemVec to locF via atomic_add using LOCAL indices */
  PetscScalar *F_dev = (locF_memtype == PETSC_MEMTYPE_DEVICE) ? d_F_unmanaged.data() : d_F.data();

  Kokkos::parallel_for(
    "DMPlexSNESComputeResidualFEM_Kokkos_scatter", Kokkos::RangePolicy<>(0, Ne), KOKKOS_LAMBDA(const PetscInt e) {
      for (PetscInt b = 0; b < Nb_; ++b) {
        const PetscInt    lidx = d_lIdx[e * Nb_ + b];
        const PetscScalar val  = d_elemVec[e * totDim_ + b];
        if (lidx >= 0) Kokkos::atomic_add(&F_dev[lidx], val);
        /* lidx == -1: Dirichlet, skip */
      }
    });
  Kokkos::fence();

  /* CPU path only: copy d_F back to locF_arr */
  if (locF_memtype != PETSC_MEMTYPE_DEVICE) Kokkos::deep_copy(h_F_unmanaged, d_F);

  PetscCall(VecRestoreArrayAndMemType(locF, &locF_arr));

  /* Assemble ghost contributions into global F via MPI reduction */
  PetscCall(DMLocalToGlobal(dm, locF, ADD_VALUES, F));
  PetscCall(DMRestoreLocalVector(dm, &locF));

  /* Cleanup */
  PetscCall(DMPlexRestoreCellFields(dm, cellIS, locX, NULL, NULL, &u_arr, &u_t_arr, &a_arr));
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(PetscFEGeomRestoreChunk(fullGeom, cStart, cEnd, &chunkGeom));
  PetscCall(PetscFEGeomDestroy(&fullGeom));
  PetscCall(ISDestroy(&cellIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DMPlexSNESComputeJacobianFEM_Kokkos<G0, G1, G2, G3>

   GPU-resident SNES Jacobian callback.  Drop-in replacement for
   DMPlexSNESComputeJacobianFEM (the CPU path) when using COO assembly.

   Template parameters (all four Jacobian callbacks):
     G0 -- KOKKOS_INLINE_FUNCTION void(PETSCFE_KOKKOS_JAC_POINT_ARGS, PetscScalar[])
          g0 term (u*v coupling)
     G1 -- g1 term (u*gradv coupling)
     G2 -- g2 term (gradu*v coupling)
     G3 -- g3 term (gradu*gradv coupling, dominant for elliptic problems)

   Pass nullptr for unused callbacks (e.g. G0=nullptr, G1=nullptr, G2=nullptr
   for a pure Laplacian where only G3 is non-zero).

   ctx_ptr must point to a PetscFEKokkosMaps with maps already staged.

   Algorithm:
     1. Get local solution with BCs applied (host-sync guard for device Vecs).
     2. Get element geometry and coefficients.
     3. Stage tabulation, geometry, coefficients, and DS constants to device.
     4. Allocate d_coo_vals[coo_size], zero it.
     5. Pass 1: PetscFEKokkosIntegrateJacobianCell<G0,G1,G2,G3> -> d_elemMat.
     6. Pass 2: scatter d_elemMat -> d_coo_vals (constraint-aware).
     7. MatSetValuesCOO(J, d_coo_vals.data(), INSERT_VALUES). */
template <PetscPointJacFn *G0, PetscPointJacFn *G1, PetscPointJacFn *G2, PetscPointJacFn *G3>
static PetscErrorCode DMPlexSNESComputeJacobianFEM_Kokkos(SNES snes, Vec X, Mat J, Mat Jp, void *ctx_ptr)
{
  PetscFEKokkosMaps *ctx = (PetscFEKokkosMaps *)ctx_ptr;
  DM                 dm;
  PetscDS            ds;
  PetscFE            fe;
  PetscTabulation   *T;
  PetscQuadrature    quad;
  DMField            coordField;
  IS                 cellIS;
  PetscFEGeom       *fullGeom  = NULL;
  PetscFEGeom       *chunkGeom = NULL;
  Vec                locX;
  PetscScalar       *u_arr = NULL, *u_t_arr = NULL, *a_arr = NULL;
  PetscInt           depth, cStart, cEnd, totDim;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           Nq, Nb, Nc, dim, dE, qdim, qNc;
  PetscInt           numConstants;
  const PetscScalar *constants;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetConstants(ds, &numConstants, &constants));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(PetscDSGetTabulation(ds, &T));
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));

  Nb = T[0]->Nb;
  Nc = T[0]->Nc;

  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetStratumIS(dm, "depth", depth, &cellIS));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  const PetscInt Ne = cEnd - cStart;

  /* Get local solution with BCs */
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, locX));
  /* Force device->host sync if locX is a device Vec */
  {
    PetscScalar *tmp;
    PetscMemType mtype;
    PetscCall(VecGetArrayAndMemType(locX, &tmp, &mtype));
    PetscCall(VecRestoreArrayAndMemType(locX, &tmp));
    if (PetscMemTypeDevice(mtype)) {
      PetscCall(VecGetArray(locX, &tmp));
      PetscCall(VecRestoreArray(locX, &tmp));
    }
  }
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locX, 0.0, NULL, NULL, NULL));

  /* Build geometry */
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, PETSC_FEGEOM_BASIC, &fullGeom));
  PetscCall(PetscFEGeomGetChunk(fullGeom, cStart, cEnd, &chunkGeom));
  dE = chunkGeom->dimEmbed;

  /* Runtime bounds check: Nc and dE must fit in the kernel's stack arrays */
  PetscCheck(Nc <= 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Nc %" PetscInt_FMT " exceeds PETSCFE_KOKKOS_MAX_NC 3", (PetscInt)Nc);
  PetscCheck(dE <= 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "dE %" PetscInt_FMT " exceeds PETSCFE_KOKKOS_MAX_DE 3", (PetscInt)dE);

  /* Get element coefficients */
  PetscCall(DMPlexGetCellFields(dm, cellIS, locX, NULL, NULL, &u_arr, &u_t_arr, &a_arr));

  /* Stage geometry and tabulation to device */
  const PetscInt nB      = Nq * Nb * Nc;
  const PetscInt nD      = Nq * Nb * Nc * dim;
  const PetscInt nInvJ   = Ne * Nq * dE * dE;
  const PetscInt nDetJ   = Ne * Nq;
  const PetscInt nCoords = Ne * Nq * dE;
  const PetscInt nCoeff  = Ne * totDim;

  Kokkos::View<PetscReal *> d_B("jac_B", nB);
  Kokkos::View<PetscReal *> d_D("jac_D", nD);
  Kokkos::View<PetscReal *> d_w("jac_w", Nq);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_B(const_cast<PetscReal *>(T[0]->T[0]), nB);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_D(const_cast<PetscReal *>(T[0]->T[1]), nD);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_w(const_cast<PetscReal *>(quadWeights), Nq);
    Kokkos::deep_copy(d_B, h_B);
    Kokkos::deep_copy(d_D, h_D);
    Kokkos::deep_copy(d_w, h_w);
  }

  PetscReal *h_invJ_buf, *h_detJ_buf, *h_coords_buf;
  PetscCall(PetscMalloc3(nInvJ, &h_invJ_buf, nDetJ, &h_detJ_buf, nCoords, &h_coords_buf));
  PetscCall(PetscFEKokkosExpandGeometry(Ne, Nq, dim, dE, chunkGeom, quadPoints, h_invJ_buf, h_detJ_buf, h_coords_buf));

  Kokkos::View<PetscReal *> d_invJ("jac_invJ", nInvJ);
  Kokkos::View<PetscReal *> d_detJ("jac_detJ", nDetJ);
  Kokkos::View<PetscReal *> d_coords("jac_coords", nCoords);
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_invJ(h_invJ_buf, nInvJ);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_detJ(h_detJ_buf, nDetJ);
    Kokkos::View<PetscReal *, Kokkos::HostSpace> hv_coords(h_coords_buf, nCoords);
    Kokkos::deep_copy(d_invJ, hv_invJ);
    Kokkos::deep_copy(d_detJ, hv_detJ);
    Kokkos::deep_copy(d_coords, hv_coords);
  }
  PetscCall(PetscFree3(h_invJ_buf, h_detJ_buf, h_coords_buf));

  Kokkos::View<PetscScalar *> d_coeff("jac_coeff", nCoeff);
  {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_coeff(const_cast<PetscScalar *>(u_arr), nCoeff);
    Kokkos::deep_copy(d_coeff, hv_coeff);
  }

  /* Stage DS constants to device */
  Kokkos::View<PetscScalar *> d_constants("jac_constants", numConstants > 0 ? numConstants : 1);
  const PetscInt              numConstants_ = numConstants;
  if (numConstants > 0) {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_constants(const_cast<PetscScalar *>(constants), numConstants);
    Kokkos::deep_copy(d_constants, hv_constants);
  }

  /* Allocate COO values array and zero it */
  const PetscCount            coo_size = ctx->coo_size;
  Kokkos::View<PetscScalar *> d_coo_vals("jac_coo_vals", coo_size);
  Kokkos::deep_copy(d_coo_vals, PetscScalar(0.0));

  /* Allocate full Ne*totDim*totDim element matrix array on device */
  Kokkos::View<PetscScalar *> d_elemMat("jac_elemMat", (PetscCount)Ne * totDim * totDim);
  Kokkos::deep_copy(d_elemMat, PetscScalar(0.0));

  /* Capture maps for lambda */
  auto           d_gIdx                   = ctx->d_gIdx;
  auto           d_coo_elem_offsets       = ctx->d_coo_elem_offsets;
  auto           d_coo_elem_point_offsets = ctx->d_coo_elem_point_offsets;
  auto           d_fullNb                 = ctx->d_fullNb;
  auto           d_c_maps_gid             = ctx->d_c_maps_gid;
  auto           d_c_maps_scale           = ctx->d_c_maps_scale;
  const PetscInt num_face_                = ctx->num_face;

  /* Capture scalars */
  const PetscInt Nq_     = Nq;
  const PetscInt Nb_     = Nb;
  const PetscInt Nc_     = Nc;
  const PetscInt dim_    = dim;
  const PetscInt dE_     = dE;
  const PetscInt totDim_ = totDim;

  /* Pass 1: integrate Jacobian for all elements into d_elemMat */
  /* TODO: pass actual t and u_tShift for time-dependent problems; currently steady-state only */
  Kokkos::parallel_for(
    "DMPlexSNESComputeJacobianFEM_Kokkos_integrate", Kokkos::RangePolicy<>(0, Ne), KOKKOS_LAMBDA(const PetscInt e) {
      PetscFEKokkosIntegrateJacobianCell<G0, G1, G2, G3>(e, Nq_, Nb_, Nc_, dim_, dE_, d_B.data(), d_D.data(), d_w.data(), d_invJ.data(), d_detJ.data(), d_coords.data(), d_coeff.data(), totDim_, 0, 0, 0, 0.0, 0.0, numConstants_, d_constants.data(),
                                                         d_elemMat.data());
    });
  Kokkos::fence();

  /* Pass 2: scatter d_elemMat to COO values (constraint-aware) */
  Kokkos::parallel_for(
    "DMPlexSNESComputeJacobianFEM_Kokkos_scatter", Kokkos::RangePolicy<>(0, Ne), KOKKOS_LAMBDA(const PetscInt e) {
      const PetscInt off    = d_coo_elem_offsets[e];
      const PetscInt fullNb = d_fullNb[e];
      for (PetscInt b = 0; b < Nb_; ++b) {
        PetscInt               nr = 0;
        PetscScalar            row_scale[PETSCFE_KOKKOS_MAX_FACE];
        const PetscFEKokkosIdx gidx_b = d_gIdx[e * Nb_ + b];
        if (gidx_b >= 0) {
          nr           = 1;
          row_scale[0] = 1.0;
        } else if (gidx_b < -1) {
          const PetscInt cidx = -(PetscInt)gidx_b - 1;
          for (PetscInt q = 0; q < num_face_; ++q) {
            if (d_c_maps_gid[cidx * num_face_ + q] < 0) break;
            row_scale[nr++] = d_c_maps_scale[cidx * num_face_ + q];
          }
        }
        if (nr == 0) continue;
        const PetscInt pt_off_b = d_coo_elem_point_offsets[e * (Nb_ + 1) + b];

        for (PetscInt b2 = 0; b2 < Nb_; ++b2) {
          PetscInt               nc = 0;
          PetscScalar            col_scale[PETSCFE_KOKKOS_MAX_FACE];
          const PetscFEKokkosIdx gidx_b2 = d_gIdx[e * Nb_ + b2];
          if (gidx_b2 >= 0) {
            nc           = 1;
            col_scale[0] = 1.0;
          } else if (gidx_b2 < -1) {
            const PetscInt cidx = -(PetscInt)gidx_b2 - 1;
            for (PetscInt q = 0; q < num_face_; ++q) {
              if (d_c_maps_gid[cidx * num_face_ + q] < 0) break;
              col_scale[nc++] = d_c_maps_scale[cidx * num_face_ + q];
            }
          }
          if (nc == 0) continue;
          const PetscInt pt_off_b2 = d_coo_elem_point_offsets[e * (Nb_ + 1) + b2];

          const PetscInt    idx0 = off + fullNb * pt_off_b + nr * pt_off_b2;
          const PetscScalar Aij  = d_elemMat[e * totDim_ * totDim_ + b * totDim_ + b2];

          /* Assignment (not atomic_add) is safe here: PetscFEKokkosPreallocateCOO
           * assigns each (row,col) pair to exactly one COO slot per element, so
           * no two kernel threads write to the same d_coo_vals index.
           * If this invariant ever breaks (e.g., shared DOFs across elements in
           * the same Kokkos team), replace with Kokkos::atomic_add. */
          for (PetscInt p = 0; p < nr; ++p) {
            for (PetscInt d = 0; d < nc; ++d) {
              d_coo_vals[idx0 + p * nc + d] = row_scale[p] * col_scale[d] * Aij;
            }
          }
        }
      }
    });
  Kokkos::fence();

  /* Set COO values into matrix */
  /* Requires MATAIJKOKKOS -- d_coo_vals is a device view */
  PetscCall(MatSetValuesCOO(J, d_coo_vals.data(), INSERT_VALUES));

  /* Cleanup */
  PetscCall(DMPlexRestoreCellFields(dm, cellIS, locX, NULL, NULL, &u_arr, &u_t_arr, &a_arr));
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(PetscFEGeomRestoreChunk(fullGeom, cStart, cEnd, &chunkGeom));
  PetscCall(PetscFEGeomDestroy(&fullGeom));
  PetscCall(ISDestroy(&cellIS));

  /* Propagate to preconditioner if different */
  if (J != Jp) PetscCall(MatCopy(J, Jp, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_HAVE_KOKKOS */
