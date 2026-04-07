/*
  fekokkos.kokkos.cxx -- PETSCFEKOKKOS: Kokkos-parallel PetscFE residual integration

  Implements PetscFEIntegrateResidual_Kokkos(), which replaces the serial cell
  loop in PetscFEIntegrateResidual_Basic (febasic.c) with a Kokkos::parallel_for
  using TeamPolicy (one team per cell, one thread per quadrature point).

  Design: Option C from plans/phase_1k_gpu_petscfe.md
    - Plugs into fe->ops->integrateresidual dispatch (no changes to plexfem.c)
    - Reimplements interpolation/contraction in Kokkos kernels
    - Static data (B, D, w) staged to device once at setup
    - Dynamic data (invJ, detJ, v, coeff) staged per call
    - User callbacks must be KOKKOS_INLINE_FUNCTION (no PetscCall, no MPI)

  Geometry layout (from PetscFEGeom):
    invJ[c * Nq * dE * dE + q * dE * dE + i * dE + j]
    detJ[c * Nq + q]
    v   [c * Nq * dE + q * dE + d]

  Tabulation layout (from PetscTabulation T[field]):
    T->T[0][q * Nb * Nc + b * Nc + c]   (basis values)
    T->T[1][q * Nb * Nc * dim + b * Nc * dim + c * dim + d]  (basis derivatives)

  Restriction: No auxiliary fields (dsAux == NULL required). Single field only.
  This covers the baby Poisson problem and cae_eigenmode.

  Author: pedra-ai (2026-04-02)
*/

#include <petsc/private/petscfeimpl.h>
#include <Kokkos_Core.hpp>
/* PetscKokkosInitializeCheck() is declared in petscsys.h (included via petscfeimpl.h) */

/* Scratch memory level: 0 = shared (fast), 1 = global (fallback).
   Matches the convention in src/ts/utils/dmplexlandau/kokkos/landau.kokkos.cxx. */
#define KOKKOS_SHARED_LEVEL 0

/* Forward declarations of PETSC_INTERN Basic ops we reuse */
PETSC_INTERN PetscErrorCode PetscFESetUp_Basic(PetscFE);
PETSC_INTERN PetscErrorCode PetscFEComputeTabulation_Basic(PetscFE, PetscInt, const PetscReal[], PetscInt, PetscTabulation);
PETSC_INTERN PetscErrorCode PetscFEIntegrate_Basic(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateBd_Basic(PetscDS, PetscInt, PetscBdPointFn *, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateResidual_Basic(PetscDS, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscDS, PetscWeakForm, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateHybridResidual_Basic(PetscDS, PetscDS, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscDS, PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateBdJacobian_Basic(PetscDS, PetscWeakForm, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateHybridJacobian_Basic(PetscDS, PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);

/* =========================================================================
   PetscFEGetDimension_Kokkos: returns the number of basis functions.
   PetscFEGetDimension_Basic is static in febasic.c so we reimplement it
   here -- it is a one-liner that queries the dual space dimension.
   ========================================================================= */
static PetscErrorCode PetscFEGetDimension_Kokkos(PetscFE fem, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetDimension(fem->dualSpace, dim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* =========================================================================
   Private data stored on the PetscFE object for the Kokkos implementation.
   Static data (basis tabulation, quadrature weights) is staged to device
   once during PetscFESetUp and reused across all calls.
   ========================================================================= */
typedef struct {
  /* Tabulation device Views -- re-staged when Nq changes (DS quadrature may differ from FE quadrature) */
  Kokkos::View<PetscReal *> d_B; /* basis values:       [Nq * Nb * Nc]          */
  Kokkos::View<PetscReal *> d_D; /* basis derivatives:  [Nq * Nb * Nc * dim]    */
  /* Quadrature weights -- staged once at setup (from FE quadrature, same Nq as DS for matching rules) */
  Kokkos::View<PetscReal *> d_w; /* quadrature weights: [Nq]                    */
  /* Host mirrors of tabulation -- updated when Nq changes */
  Kokkos::View<PetscReal *, Kokkos::HostSpace> h_B;
  Kokkos::View<PetscReal *, Kokkos::HostSpace> h_D;
  Kokkos::View<PetscReal *, Kokkos::HostSpace> h_w;
  /* Cached per-call device Views -- reallocated only when Ne or totDim changes */
  Kokkos::View<PetscReal *>   d_invJ;    /* [Ne * Nq * dE * dE]  */
  Kokkos::View<PetscScalar *> d_elemVec; /* [Ne * totDim]         */
  Kokkos::View<PetscScalar *> d_f0_scr;  /* [Ne * Nq * Nc]        */
  Kokkos::View<PetscScalar *> d_f1_scr;  /* [Ne * Nq * Nc * dE]   */
  Kokkos::View<PetscScalar *> d_val;     /* [Ne * Nb] -- per-element accumulator (Nb = total DOFs) */
  /* Host mirror of d_elemVec -- cached to avoid per-call mirror alloc */
  Kokkos::View<PetscScalar *, Kokkos::HostSpace> h_elemVec;
  /* Host scratch for f0/f1 -- single contiguous allocation; h_f1_buf = h_f0_buf + Ne*Nq*Nc.
     Packed into one block to reduce allocator overhead and improve cache locality. */
  PetscScalar *h_f0_buf; /* points to start of block: [Ne * Nq * Nc]      */
  PetscScalar *h_f1_buf; /* points into block:        [Ne * Nq * Nc * dE] */
  /* Expanded invJ buffer [Ne * Nq * dE * dE] -- for affine elements the single
     per-element invJ is replicated across all Nq slots so Phase 2 can index
     uniformly as invJ[e * Nq * dE * dE + q * dE * dE].  Cached and reallocated
     only when Ne or Nq changes (same lifetime as h_f0_buf). */
  PetscReal *h_invJ_buf; /* [Ne * Nq * dE * dE] */
  /* Per-(e,q) interpolation scratch -- heap-allocated at setup.
     h_u_buf  [Nc]:       field values at one quadrature point (Nc components).
     h_ux_buf [Nc * dim]: field gradients at one quadrature point.
     Note: u_loc[c] and ux_loc[c*dE+d] are indexed by component c only;
     the basis-function loop (b=0..Nb-1) accumulates into these Nc-sized arrays. */
  PetscScalar *h_u_buf;      /* [Nc]       -- field values at one quadrature point  */
  PetscScalar *h_ux_buf;     /* [Nc * dim] -- field gradients at one quadrature point */
  PetscInt     Ne_alloc;     /* Ne    for which cached Views/bufs were last allocated */
  PetscInt     Nq_alloc;     /* Nq    for which d_B/d_D/d_w/d_f0_scr/d_f1_scr were last staged */
  PetscInt     totDim_alloc; /* totDim for which d_elemVec/h_elemVec were last allocated */
  /* Sizes cached at setup */
  PetscInt  Nb;  /* total number of DOFs per element = PetscDualSpaceGetDimension()
                     For scalar FE (Nc=1): Nb = number of scalar basis functions.
                     For vector FE (Nc>1): Nb = Nb_scalar * Nc (total DOFs).
                     Matches T->Nb from PetscFECreateTabulation. */
  PetscInt  Nc;  /* number of field components  */
  PetscInt  dim; /* spatial dimension           */
  PetscBool setup_done;
} PetscFE_Kokkos;

/* =========================================================================
   PetscFESetUp_Kokkos: call Basic setup (builds invV, tabulation), then
   stage static data (B, D, w) to device.
   ========================================================================= */
static PetscErrorCode PetscFESetUp_Kokkos(PetscFE fem)
{
  PetscFE_Kokkos  *kk = (PetscFE_Kokkos *)fem->data;
  PetscTabulation  T;
  PetscQuadrature  quad;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         qdim, qNc, Nq, Nb, Nc, dim;

  PetscFunctionBegin;
  if (kk->setup_done) PetscFunctionReturn(PETSC_SUCCESS);

  /* Ensure Kokkos is initialized before allocating Views */
  PetscCall(PetscKokkosInitializeCheck());

  /* PetscFECreateDefault calls PetscFESetUp (via PetscFECreateFromSpaces) which
     runs PetscFESetUp_Basic and allocates fem->invV.  When the user then calls
     PetscFESetType(fe, PETSCFEKOKKOS), PetscFESetType only invokes the
     type-specific destroy (PetscFEDestroy_Basic, which frees fem->data) but
     does NOT free fem->invV -- that is the base class's job in PetscFEDestroy.
     PetscFESetUp_Basic below would then overwrite fem->invV with a fresh
     allocation, leaking the original one.  Free it here before re-running
     Basic setup so the pointer is NULL when PetscFESetUp_Basic assigns it. */
  PetscCall(PetscFree(fem->invV));

  /* Run Basic setup to build invV and cache tabulation on fem->T */
  PetscCall(PetscFESetUp_Basic(fem));

  PetscCall(PetscFEGetSpatialDimension(fem, &dim));
  PetscCall(PetscFEGetQuadrature(fem, &quad));
  PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights));

  /* Create tabulation at FE quadrature points to get Nb and Nc.
     T->Nb = PetscDualSpaceGetDimension() = total DOFs per element.
       Scalar FE (Nc=1): T->Nb = number of scalar basis functions (e.g., 3 for P1 triangle).
       Vector FE (Nc>1): T->Nb = Nb_scalar * Nc (e.g., 6 for P1 vector on triangle with Nc=2).
     T->Nc = number of field components.
     T->T[0] has Nq * T->Nb * T->Nc entries per quadrature point.
     We store Nb = T->Nb (total DOFs) so that the interpolation loop
       b=0..Nb-1, coeff[b], B[b*Nc+c]
     matches PetscFEEvaluateFieldJets_Internal exactly. */
  PetscCall(PetscFECreateTabulation(fem, 1, Nq, quadPoints, 1, &T));
  Nb = T->Nb; /* total DOFs per element */
  Nc = T->Nc; /* field components */
  PetscCall(PetscTabulationDestroy(&T));

  kk->Nb  = Nb; /* total DOFs per element (= Nb_scalar * Nc for vector FE) */
  kk->Nc  = Nc; /* field components */
  kk->dim = dim;

  /* Allocate per-(e,q) interpolation scratch.
     u_loc  [Nc]:       field values at one quadrature point (Nc components).
     ux_loc [Nc * dim]: field gradients at one quadrature point.
     The basis-function loop (b=0..Nb-1) accumulates into these Nc-sized arrays,
     matching PetscFEEvaluateFieldJets_Internal which uses u[c] and u_x[c*dE+d]. */
  PetscCall(PetscMalloc2(Nc, &kk->h_u_buf, Nc * dim, &kk->h_ux_buf));

  kk->setup_done = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* =========================================================================
   PetscFEStageTabulation_Kokkos: stage B/D/w to device for a given Nq.
   Called at integration time when the DS tabulation Nq differs from the
   last staged Nq (e.g., P2 uses a richer quadrature than P1).
   ========================================================================= */
static PetscErrorCode PetscFEStageTabulation_Kokkos(PetscFE_Kokkos *kk, PetscTabulation Tab, const PetscReal *quadWeights, PetscInt Nq)
{
  const PetscInt Nb  = kk->Nb;
  const PetscInt Nc  = kk->Nc;
  const PetscInt dim = kk->dim;
  const PetscInt nB  = Nq * Nb * Nc;
  const PetscInt nD  = Nq * Nb * Nc * dim;

  PetscFunctionBegin;
  /* Stage basis values B[Nq * Nb * Nc] to device; cache host mirror */
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> tmp_h_B(const_cast<PetscReal *>(Tab->T[0]), nB);
    kk->d_B = Kokkos::View<PetscReal *>("fekokkos_B", nB);
    Kokkos::deep_copy(kk->d_B, tmp_h_B);
    kk->h_B = Kokkos::create_mirror_view(kk->d_B);
    Kokkos::deep_copy(kk->h_B, kk->d_B);
  }
  /* Stage basis derivatives D[Nq * Nb * Nc * dim] to device; cache host mirror */
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> tmp_h_D(const_cast<PetscReal *>(Tab->T[1]), nD);
    kk->d_D = Kokkos::View<PetscReal *>("fekokkos_D", nD);
    Kokkos::deep_copy(kk->d_D, tmp_h_D);
    kk->h_D = Kokkos::create_mirror_view(kk->d_D);
    Kokkos::deep_copy(kk->h_D, kk->d_D);
  }
  /* Stage quadrature weights w[Nq] to device; cache host mirror */
  {
    Kokkos::View<PetscReal *, Kokkos::HostSpace> tmp_h_w(const_cast<PetscReal *>(quadWeights), Nq);
    kk->d_w = Kokkos::View<PetscReal *>("fekokkos_w", Nq);
    Kokkos::deep_copy(kk->d_w, tmp_h_w);
    kk->h_w = Kokkos::create_mirror_view(kk->d_w);
    Kokkos::deep_copy(kk->h_w, kk->d_w);
  }
  kk->Nq_alloc = Nq;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* =========================================================================
   PetscFEIntegrateResidual_Kokkos

   Replaces the serial cell loop in PetscFEIntegrateResidual_Basic with a
   Kokkos::parallel_for using TeamPolicy.

   Restriction: dsAux == NULL (no auxiliary fields). Single field (Nf == 1).
   Falls back to Basic implementation if these constraints are not met.
   ========================================================================= */
static PetscErrorCode PetscFEIntegrateResidual_Kokkos(PetscDS ds, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom, const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  PetscFE            fe;
  PetscFE_Kokkos    *kk;
  PetscWeakForm      wf;
  PetscInt           n0, n1;
  PetscPointFn     **f0_func, **f1_func;
  PetscInt           Nf, totDim, fOffset;
  PetscInt          *uOff, *uOff_x;
  PetscInt           numConstants;
  const PetscScalar *constants;
  const PetscInt     field = key.field;

  PetscFunctionBegin;
  /* Fall back to Basic if auxiliary fields are present */
  if (dsAux) {
    PetscCall(PetscFEIntegrateResidual_Basic(ds, key, Ne, cgeom, coefficients, coefficients_t, dsAux, coefficientsAux, t, elemVec));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscDSGetDiscretization(ds, field, (PetscObject *)&fe));
  kk = (PetscFE_Kokkos *)fe->data;

  /* Ensure static data is staged -- inline check avoids a function call on every
     integration when setup is already done (the common case). */
  if (!kk->setup_done) PetscCall(PetscFESetUp_Kokkos(fe));

  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));
  PetscCall(PetscDSGetComponentOffsets(ds, &uOff));
  PetscCall(PetscDSGetComponentDerivativeOffsets(ds, &uOff_x));
  PetscCall(PetscDSGetFieldOffset(ds, field, &fOffset));
  PetscCall(PetscDSGetWeakForm(ds, &wf));
  PetscCall(PetscWeakFormGetResidual(wf, key.label, key.value, key.field, key.part, &n0, &f0_func, &n1, &f1_func));
  if (!n0 && !n1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDSGetConstants(ds, &numConstants, &constants));

  /* Fall back to Basic for multi-callback or time-dependent problems */
  if (n0 > 1 || n1 > 1 || coefficients_t) {
    PetscCall(PetscFEIntegrateResidual_Basic(ds, key, Ne, cgeom, coefficients, coefficients_t, dsAux, coefficientsAux, t, elemVec));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Nq = number of quadrature points from the FE quadrature rule.
     Np = cgeom->numPoints = geometry stride per element (may differ from Nq
     for affine elements where the Jacobian is constant across quadrature points).
     Always use feNq (from PetscFEGetQuadrature) as the loop count -- this matches
     the DS tabulation (T[field]->Np == feNq) and the quadrature weights array.
     cgeom->numPoints is only the geometry stride, not the quadrature count. */
  const PetscInt  Nb       = kk->Nb;
  const PetscInt  Nc       = kk->Nc;
  const PetscInt  dim      = kk->dim;
  const PetscInt  dE       = cgeom->dimEmbed;
  const PetscInt  Np       = cgeom->numPoints; /* geometry stride per element */
  const PetscBool isAffine = cgeom->isAffine;

  /* Get the FE quadrature -- Nq and quadPoints/quadWeights come from here.
     The DS tabulation T[field]->Np == Nq (they are evaluated at the same points).
     Re-stage B/D/w to device only when Nq changes (e.g., first call or degree change). */
  PetscInt         Nq;
  const PetscReal *quadPoints;
  const PetscReal *quadWeights;
  {
    PetscTabulation *T;
    PetscQuadrature  quad;
    PetscInt         qNc, qdim;
    PetscCall(PetscDSGetTabulation(ds, &T));
    PetscCall(PetscFEGetQuadrature(fe, &quad));
    PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights));
    if (Nq != kk->Nq_alloc) PetscCall(PetscFEStageTabulation_Kokkos(kk, T[field], quadWeights, Nq));
  }

  auto           f0_fn = (n0 > 0) ? f0_func[0] : nullptr;
  auto           f1_fn = (n1 > 0) ? f1_func[0] : nullptr;
  const PetscInt uOff0 = uOff[field];
  const PetscInt fOff  = fOffset;

  /* -----------------------------------------------------------------------
     Two-phase Kokkos integration (always used -- CPU Serial or GPU CUDA):

     Phase 1 (host, serial): evaluate u_loc / ux_loc per (cell, qp), call
       f0/f1 host function pointers, fill h_f0_scr / h_f1_scr.
       Host function pointers cannot be called from a CUDA device kernel,
       so this phase always runs on the host regardless of backend.

     Phase 2 (Kokkos::parallel_for): deep_copy scratch to device Views,
       then run basis-function assembly kernel.  On Kokkos::Serial this
       executes on the host; on Kokkos::Cuda it executes on the GPU.
       No function-pointer calls -- pure arithmetic.

     Geometry handling (mirrors PetscFEGeomGetPoint / febasic.c):
       Affine elements (isAffine=1): invJ/detJ constant per element, stored
         at index 0 within each element's geometry block (stride Np).
         Physical coords v computed per-q via CoordinatesRefToReal.
       Non-affine elements: invJ/detJ/v stored at each of the Np==Nq points.

     Flop accounting:
       PetscLogFlops    -- Phase 1 host work (interpolation + f0/f1 scaling)
       PetscLogGpuFlops -- Phase 2 Kokkos work (basis assembly).
         PetscLogGpuFlops also adds to the total PetscLogFlops counter, so
         CPU-only flops = total_flops - gpu_flops.
         On Kokkos::Serial the "GPU" flops are still registered as device
         flops (Kokkos convention), making the split meaningful even without
         a physical GPU.
     ----------------------------------------------------------------------- */

  /* d_invJ is sized [Ne * Nq * dE * dE] -- for affine elements we replicate
     the single per-element invJ across all Nq slots during Phase 1 so that
     Phase 2 can index uniformly as invJ[e * Nq * dE * dE + q * dE * dE]. */
  const PetscInt nInvJ = Ne * Nq * dE * dE;

  /* Use cached host mirrors of static basis/quadrature data (no per-call D->H copy).
     Use raw .data() pointers for the Phase 1 serial loop -- avoids View accessor
     overhead (bounds checks in debug builds, extra indirection in opt builds). */
  const PetscReal *h_B = kk->h_B.data();
  const PetscReal *h_D = kk->h_D.data();
  const PetscReal *h_w = kk->h_w.data();

  /* Phase 1: host loop -- evaluate physics callbacks, fill h_f0/h_f1 scratch.
     Reuse cached host scratch buffers; reallocate only when Ne or Nq changes.
     h_f0_buf and h_f1_buf share one contiguous block: [Ne*Nq*Nc + Ne*Nq*Nc*dE].
     h_f1_buf is set to h_f0_buf + Ne*Nq*Nc (pointer arithmetic into the block). */
  if (Ne != kk->Ne_alloc || Nq != kk->Nq_alloc) {
    PetscCall(PetscFree(kk->h_f0_buf));
    PetscCall(PetscMalloc1(Ne * Nq * Nc + Ne * Nq * Nc * dE, &kk->h_f0_buf));
    kk->h_f1_buf = kk->h_f0_buf + Ne * Nq * Nc;
  }
  PetscScalar *h_f0_scr = kk->h_f0_buf;
  PetscScalar *h_f1_scr = kk->h_f1_buf;
  /* Always zero the scratch buffers before calling the callbacks.
     PETSc convention: f0/f1 callbacks accumulate (+=) into a pre-zeroed output.
     Failing to zero here causes NaN/Inf when callbacks use += into uninitialized
     memory (e.g., elasticity with Nc=2).  The Poisson case worked accidentally
     because its callbacks use = (assignment) rather than +=.
     When a callback is absent, the buffer must also be zero so Phase 2 sees
     no contribution from the missing term. */
  PetscCall(PetscArrayzero(h_f0_scr, Ne * Nq * Nc));
  PetscCall(PetscArrayzero(h_f1_scr, Ne * Nq * Nc * dE));

  /* Hoist single-element offset arrays out of the (e,q) loop -- they are
     constant for the entire call (single-field, no auxiliary fields). */
  PetscInt uOff_l[1]   = {0};
  PetscInt uOff_x_l[1] = {0};

  /* Heap-allocated per-(e,q) scratch -- supports any polynomial order */
  PetscScalar *u_loc  = kk->h_u_buf;  /* [Nc]       */
  PetscScalar *ux_loc = kk->h_ux_buf; /* [Nc * dim] */

  /* h_invJ_buf: host buffer [Ne * Nq * dE * dE] for expanded invJ.
     For affine elements, the single per-element invJ is replicated across
     all Nq quadrature slots so Phase 2 can index uniformly.
     Cached in kk struct; reallocated only when Ne or Nq changes. */
  if (Ne != kk->Ne_alloc || Nq != kk->Nq_alloc) {
    PetscCall(PetscFree(kk->h_invJ_buf));
    PetscCall(PetscMalloc1(Ne * Nq * dE * dE, &kk->h_invJ_buf));
  }
  PetscReal *h_invJ_buf = kk->h_invJ_buf;

  /* Physical coordinate workspace for affine elements (one point at a time) */
  PetscReal v_affine[3] = {0.0, 0.0, 0.0}; /* max dE = 3 */

  for (PetscInt e = 0; e < Ne; ++e) {
    /* Set up per-element geometry pointers (mirrors febasic.c lines 214-220) */
    const PetscReal *invJ_e; /* invJ for this element (affine: constant; non-affine: per-q) */
    const PetscReal *detJ_e; /* detJ for this element */
    const PetscReal *v0_e;   /* reference origin v0 for affine coord mapping */
    const PetscReal *J_e;    /* J for this element (needed for affine coord mapping) */
    const PetscReal *xi_e;   /* reference origin xi0 for affine coord mapping */

    if (isAffine) {
      invJ_e = &cgeom->invJ[e * Np * dE * dE];
      detJ_e = &cgeom->detJ[e * Np];
      J_e    = &cgeom->J[e * Np * dE * dE];
      v0_e   = &cgeom->v[e * Np * dE];
      xi_e   = cgeom->xi;
      /* Replicate the single affine invJ across all Nq slots in h_invJ_buf */
      for (PetscInt q = 0; q < Nq; ++q)
        for (PetscInt i = 0; i < dE * dE; ++i) h_invJ_buf[(e * Nq + q) * dE * dE + i] = invJ_e[i];
    } else {
      /* Non-affine: geometry stored at each of the Np==Nq quadrature points */
      invJ_e = &cgeom->invJ[e * Np * dE * dE];
      detJ_e = &cgeom->detJ[e * Np];
      J_e    = nullptr;
      v0_e   = nullptr;
      xi_e   = nullptr;
      /* Copy non-affine invJ directly into h_invJ_buf */
      for (PetscInt q = 0; q < Nq; ++q)
        for (PetscInt i = 0; i < dE * dE; ++i) h_invJ_buf[(e * Nq + q) * dE * dE + i] = invJ_e[q * dE * dE + i];
    }

    for (PetscInt q = 0; q < Nq; ++q) {
      /* Get geometry at this quadrature point (mirrors PetscFEGeomGetPoint) */
      const PetscReal *invJ_eq;
      PetscReal        detJ_eq;
      const PetscReal *v_eq;

      if (isAffine) {
        invJ_eq = invJ_e; /* same for all q */
        detJ_eq = detJ_e[0];
        /* Compute physical coords from reference quadrature point */
        CoordinatesRefToReal(dE, dim, xi_e, v0_e, J_e, &quadPoints[q * dim], v_affine);
        v_eq = v_affine;
      } else {
        invJ_eq = &invJ_e[q * dE * dE];
        detJ_eq = detJ_e[q];
        v_eq    = &cgeom->v[(e * Np + q) * dE];
      }

      const PetscReal w = h_w[q];

      /* Zero per-(e,q) scratch -- heap buffers are reused across iterations.
         u_loc  [Nc]:       field values at one quadrature point.
         ux_loc [Nc * dE]:  field gradients at one quadrature point.
         The basis-function loop (b=0..Nb-1) accumulates into these Nc-sized arrays,
         matching PetscFEEvaluateFieldJets_Internal: u[c] += Bq[b*Nc+c] * coeff[b]. */
      PetscCall(PetscArrayzero(u_loc, Nc));
      PetscCall(PetscArrayzero(ux_loc, Nc * dE));

      /* u[c] = sum_b B[q,b,c] * coeff[b]
         Matches PetscFEEvaluateFieldJets_Internal line 2377:
           u[fOffset + c] += Bq[b*Ncf + c] * coefficients[dOffset + b]
         coeff[b] for b=0..Nb-1 (NOT coeff[b*Nc+c]) */
      for (PetscInt b = 0; b < Nb; ++b)
        for (PetscInt c = 0; c < Nc; ++c) u_loc[c] += h_B[q * Nb * Nc + b * Nc + c] * coefficients[e * totDim + uOff0 + b];

      /* ux[c*dE+d] = sum_b sum_{e2} D[q,b,c,e2] * invJ[e2,d] * coeff[b]
         Matches PetscFEEvaluateFieldJets_Internal + PetscFEPushforwardGradient:
           u_x_ref[c*cdim+e2] = sum_b D[b,c,e2] * coeff[b]
           u_x_phys[c*dE+d]   = sum_{e2} invJ[e2*dE+d] * u_x_ref[c*cdim+e2]
         PetscDualSpaceTransformGradient applies invJ^T unconditionally (lines 1945-1967
         of dualspace.c), BEFORE the IDENTITY_TRANSFORM break at line 1971.
         coeff[b] for b=0..Nb-1 (NOT coeff[b*Nc+c]) */
      for (PetscInt b = 0; b < Nb; ++b) {
        const PetscScalar coeff_b = coefficients[e * totDim + uOff0 + b];
        for (PetscInt c = 0; c < Nc; ++c) {
          for (PetscInt d = 0; d < dE; ++d) {
            PetscReal ref_grad = 0.0;
            for (PetscInt e2 = 0; e2 < dim; ++e2) ref_grad += h_D[q * Nb * Nc * dim + b * Nc * dim + c * dim + e2] * invJ_eq[e2 * dE + d];
            ux_loc[c * dE + d] += coeff_b * ref_grad;
          }
        }
      }

      if (f0_fn) {
        f0_fn(dE, 1, 0, uOff_l, uOff_x_l, u_loc, nullptr, ux_loc, nullptr, nullptr, nullptr, nullptr, nullptr, t, v_eq, numConstants, constants, &h_f0_scr[(e * Nq + q) * Nc]);
        for (PetscInt c = 0; c < Nc; ++c) h_f0_scr[(e * Nq + q) * Nc + c] *= detJ_eq * w;
      }
      if (f1_fn) {
        f1_fn(dE, 1, 0, uOff_l, uOff_x_l, u_loc, nullptr, ux_loc, nullptr, nullptr, nullptr, nullptr, nullptr, t, v_eq, numConstants, constants, &h_f1_scr[(e * Nq + q) * Nc * dE]);
        for (PetscInt c = 0; c < Nc; ++c)
          for (PetscInt d = 0; d < dE; ++d) h_f1_scr[((e * Nq + q) * Nc + c) * dE + d] *= detJ_eq * w;
      }
    } /* end q */
  } /* end e (Phase 1) */

  /* Log Phase 1 host flops:
     Per element per qp:
       u_loc interpolation:  Nb*Nc*2  (multiply-add)
       ux_loc interpolation: Nb*Nc*(dE*dim*2 + dE)  (ref_grad + accumulate)
       f0 scaling:           Nc  (multiply by detJ*w)
       f1 scaling:           Nc*dE  (multiply by detJ*w)
  */
  PetscCall(PetscLogFlops((PetscLogDouble)Ne * Nq * (Nb * Nc * 2.0 + Nb * (dE * dim * 2.0 + dE) * Nc + Nc + Nc * dE)));

  /* Phase 2: stage f0/f1 scratch to device, run Kokkos basis assembly kernel */

  /* Realloc cached device Views when Ne, Nq, or totDim changes.
     d_invJ/d_f0_scr/d_f1_scr depend on Nq; d_elemVec/h_elemVec depend on totDim. */
  if (Ne != kk->Ne_alloc || Nq != kk->Nq_alloc || totDim != kk->totDim_alloc) {
    kk->d_invJ       = Kokkos::View<PetscReal *>("fekokkos_invJ", nInvJ);
    kk->d_elemVec    = Kokkos::View<PetscScalar *>("fekokkos_elemVec", Ne * totDim);
    kk->h_elemVec    = Kokkos::View<PetscScalar *, Kokkos::HostSpace>("fekokkos_h_elemVec", Ne * totDim);
    kk->d_f0_scr     = Kokkos::View<PetscScalar *>("fekokkos_f0", Ne * Nq * Nc);
    kk->d_f1_scr     = Kokkos::View<PetscScalar *>("fekokkos_f1", Ne * Nq * Nc * dE);
    kk->d_val        = Kokkos::View<PetscScalar *>("fekokkos_val", Ne * Nb);
    kk->Ne_alloc     = Ne;
    kk->totDim_alloc = totDim;
    /* Note: Nq_alloc is updated by PetscFEStageTabulation_Kokkos above */
  }

  {
    /* Use h_invJ_buf (expanded, [Ne*Nq*dE*dE]) -- NOT cgeom->invJ which has
       size [Ne*Np*dE*dE] and may differ from Nq for affine elements. */
    Kokkos::View<PetscReal *, Kokkos::HostSpace> h_invJ(h_invJ_buf, nInvJ);
    Kokkos::deep_copy(kk->d_invJ, h_invJ);
  }
  Kokkos::deep_copy(kk->d_elemVec, PetscScalar(0.0));

  {
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_f0(h_f0_scr, Ne * Nq * Nc);
    Kokkos::View<PetscScalar *, Kokkos::HostSpace> hv_f1(h_f1_scr, Ne * Nq * Nc * dE);
    Kokkos::deep_copy(kk->d_f0_scr, hv_f0);
    Kokkos::deep_copy(kk->d_f1_scr, hv_f1);
  }
  /* h_f0_scr/h_f1_scr are now owned by kk->h_f0_buf/h_f1_buf -- do not free here */

  auto d_B_       = kk->d_B;
  auto d_D_       = kk->d_D;
  auto d_invJ_    = kk->d_invJ;
  auto d_elemVec_ = kk->d_elemVec;
  auto d_val_     = kk->d_val;

  auto d_f0_scr_ = kk->d_f0_scr;
  auto d_f1_scr_ = kk->d_f1_scr;

  /* Zero the per-element accumulator View before the kernel */
  Kokkos::deep_copy(d_val_, PetscScalar(0.0));

  /* Phase 2 kernel: loop order (q, b, c) for coalesced d_B_ / d_D_ access.
     d_B_ layout is [q * Nb*Nc + b*Nc + c] -- sequential in q for fixed (b,c).
     Outer q loop -> consecutive d_B_ reads per thread -> coalesced on GPU.
     d_val_ [Ne * Nb] -- one accumulator per DOF b (not per (b,c) pair).
     Matches PetscFEUpdateElementVec_Internal:
       elemVec[b] += B[q,b,c] * f0[q,c]  (b=0..Nb-1, c=0..Nc-1)
     Supporting any polynomial order (P1, P2, P3, ...) without stack overflow. */
  Kokkos::parallel_for(
    "PetscFEIntegrateResidual_Kokkos", Kokkos::RangePolicy<>(0, Ne), KOKKOS_LAMBDA(const PetscInt e) {
      const PetscScalar *f0_s  = &d_f0_scr_(e * Nq * Nc);
      const PetscScalar *f1_s  = &d_f1_scr_(e * Nq * Nc * dE);
      PetscScalar       *val_e = &d_val_(e * Nb); /* row for element e: [Nb] */

      for (PetscInt q = 0; q < Nq; ++q) {
        /* Pointer to the start of B[q, *, *] -- layout [Nb * Nc] per q-point */
        const PetscReal *B_q = &d_B_(q * Nb * Nc);
        /* Pointer to the start of D[q, *, *, *] -- layout [Nb * Nc * dim] per q-point */
        const PetscReal *D_q = &d_D_(q * Nb * Nc * dim);
        /* Pointer to invJ[e, q, *, *] */
        const PetscReal *invJ_eq = &d_invJ_(e * Nq * dE * dE + q * dE * dE);

        /* Matches PetscFEUpdateElementVec_Internal + PetscFEPushforwardGradient:
             tmpBasisDer_phys[b,c,d] = sum_{e2} D[b,c,e2] * invJ[e2,d]
             elemVec[b] += tmpBasisDer_phys[b,c,d] * f1[q,c,d]
           PetscDualSpaceTransformGradient applies invJ^T unconditionally (lines 1945-1967
           of dualspace.c), BEFORE the IDENTITY_TRANSFORM break at line 1971.
           val_e[b] accumulates over all c -- one entry per DOF b. */
        for (PetscInt b = 0; b < Nb; ++b) {
          for (PetscInt c = 0; c < Nc; ++c) {
            const PetscInt bc = b * Nc + c;
            /* f0 contribution: B[q,b,c] * f0_s[q,c] -> val_e[b] */
            val_e[b] += B_q[bc] * f0_s[q * Nc + c];
            /* f1 contribution: phys_grad[b,c,d] * f1_s[q,c,d] -> val_e[b]
               phys_grad[b,c,d] = sum_{e2} D[b,c,e2] * invJ[e2,d] */
            for (PetscInt d = 0; d < dE; ++d) {
              PetscReal phys_grad = 0.0;
              for (PetscInt e2 = 0; e2 < dim; ++e2) phys_grad += D_q[bc * dim + e2] * invJ_eq[e2 * dE + d];
              val_e[b] += phys_grad * f1_s[(q * Nc + c) * dE + d];
            }
          }
        }
      }

      /* Write accumulated values to d_elemVec_ -- one pass, no atomics.
         Matches PetscFEUpdateElementVec_Internal: elemVec[b] for b=0..Nb-1.
         elemVec layout: [totDim] per element; field f starts at fOff.
         For vector FE: fOff+b for b=0..Nb-1 covers all Nb DOFs of this field. */
      for (PetscInt b = 0; b < Nb; ++b) d_elemVec_(e * totDim + fOff + b) += val_e[b];
    }); /* end RangePolicy */

  Kokkos::fence();

  /* Log Phase 2 device flops (Kokkos basis assembly):
     Per element per (b,c): Nq * (2 + dE*(dim*2+1)) multiply-adds
       - f0 term:  Nq * 2  (multiply B[q,b,c]*f0_s + add to val)
       - f1 term:  Nq * dE * (dim*2 + 1)  (phys_grad inner product + add to val)
     PetscLogGpuFlops also increments the total PetscLogFlops counter.
     GPU %F in -log_view = gpu_flops / total_flops * 100.
  */
  PetscCall(PetscLogGpuFlops((PetscLogDouble)Ne * Nb * Nc * Nq * (2.0 + (PetscLogDouble)dE * (dim * 2.0 + 1.0))));

  /* Copy result back using cached host mirror -- no per-call mirror alloc.
     Use a raw pointer loop over the flat Ne*totDim array so the compiler
     can auto-vectorize the += accumulation. */
  Kokkos::deep_copy(kk->h_elemVec, kk->d_elemVec);
  {
    const PetscScalar *src = kk->h_elemVec.data();
    const PetscInt     n   = Ne * totDim;
    for (PetscInt i = 0; i < n; ++i) elemVec[i] += src[i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* =========================================================================
   PetscFEDestroy_Kokkos
   ========================================================================= */
static PetscErrorCode PetscFEDestroy_Kokkos(PetscFE fem)
{
  PetscFE_Kokkos *kk = (PetscFE_Kokkos *)fem->data;

  PetscFunctionBegin;
  /* Free the single contiguous f0/f1 scratch block (h_f1_buf points into it) */
  PetscCall(PetscFree(kk->h_f0_buf));
  /* Free expanded invJ buffer */
  PetscCall(PetscFree(kk->h_invJ_buf));
  /* Free per-(e,q) interpolation scratch (h_ux_buf is paired with h_u_buf) */
  PetscCall(PetscFree2(kk->h_u_buf, kk->h_ux_buf));
  /* Use C++ delete so that Kokkos::View destructors run properly.
     PetscFree/free() bypasses destructors and corrupts View ref counts. */
  delete kk;
  fem->data = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* =========================================================================
   PetscFEInitialize_Kokkos: set ops table.

   We set all ops to the Basic implementations (via PETSC_INTERN forward
   declarations) and then override setup, destroy, and integrateresidual
   with our Kokkos versions.  view and getdimension are left NULL because
   PetscFEView_Basic and PetscFEGetDimension_Basic are static in febasic.c.
   ========================================================================= */
static PetscErrorCode PetscFEInitialize_Kokkos(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = PetscFESetUp_Kokkos;
  fem->ops->view                    = NULL;
  fem->ops->destroy                 = PetscFEDestroy_Kokkos;
  fem->ops->getdimension            = PetscFEGetDimension_Kokkos;
  fem->ops->computetabulation       = PetscFEComputeTabulation_Basic;
  fem->ops->integrate               = PetscFEIntegrate_Basic;
  fem->ops->integratebd             = PetscFEIntegrateBd_Basic;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_Kokkos;
  fem->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Basic;
  fem->ops->integratehybridresidual = PetscFEIntegrateHybridResidual_Basic;
  fem->ops->integratejacobianaction = NULL;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  fem->ops->integratebdjacobian     = PetscFEIntegrateBdJacobian_Basic;
  fem->ops->integratehybridjacobian = PetscFEIntegrateHybridJacobian_Basic;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCFEKOKKOS = "kokkos" - A `PetscFE` object that integrates the residual
  using Kokkos::parallel_for over cells (TeamPolicy).

  The Jacobian integration falls back to the Basic (CPU) implementation.
  User residual callbacks (f0, f1) must be KOKKOS_INLINE_FUNCTION.
  Auxiliary fields (dsAux) are not supported; falls back to Basic if present.

  Level: intermediate

.seealso: `PetscFE`, `PetscFEType`, `PETSCFEBASIC`, `PetscFECreate()`, `PetscFESetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscFECreate_Kokkos(PetscFE fem)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  /* Use C++ new so that Kokkos::View members are default-constructed (empty,
     ref-count = null).  PetscNew uses PetscMalloc which skips constructors. */
  PetscFE_Kokkos *kk = new PetscFE_Kokkos();
  kk->setup_done     = PETSC_FALSE;
  kk->Ne_alloc       = -1;
  kk->Nq_alloc       = -1;
  kk->totDim_alloc   = -1;
  kk->h_f0_buf       = nullptr;
  kk->h_f1_buf       = nullptr;
  kk->h_invJ_buf     = nullptr;
  kk->h_u_buf        = nullptr;
  kk->h_ux_buf       = nullptr;
  fem->data          = kk;

  PetscCall(PetscFEInitialize_Kokkos(fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}
