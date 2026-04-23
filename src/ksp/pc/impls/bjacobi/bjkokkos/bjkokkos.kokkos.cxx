#define PETSC_SKIP_CXX_COMPLEX_FIX // Kokkos::complex does not need the PetscComplex fix

#include <petsc/private/pcbjkokkosimpl.h>

#include <petsc/private/kspimpl.h>
#include <petscksp.h> /*I "petscksp.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>
#include <petscsection.h>
#include <petscdmcomposite.h>

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

#include <petscdevice_cupm.h>

// -----------------------------------------------------------------------
// PetscLog event handles for PCApply_BJKOKKOS sub-phases
// Registered once in PCSetUp_BJKOKKOS (guarded by a static flag).
// -----------------------------------------------------------------------
PetscLogEvent BJKOKKOS_AMG_RAP      = 0;
PetscLogEvent BJKOKKOS_Krylov_Solve = 0;
PetscLogEvent BJKOKKOS_Post_solve   = 0;

static PetscErrorCode PCBJKOKKOSCreateKSP_BJKOKKOS(PC pc)
{
  const char    *prefix;
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  DM             dm;

  PetscFunctionBegin;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &jac->ksp));
  PetscCall(KSPSetNestLevel(jac->ksp, pc->kspnestlevel));
  PetscCall(KSPSetErrorIfNotConverged(jac->ksp, pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->ksp, (PetscObject)pc, 1));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(KSPSetOptionsPrefix(jac->ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(jac->ksp, "pc_bjkokkos_"));
  PetscCall(PCGetDM(pc, &dm));
  if (dm) {
    PetscCall(KSPSetDM(jac->ksp, dm));
    PetscCall(KSPSetDMActive(jac->ksp, KSP_DMACTIVE_ALL, PETSC_FALSE));
  }
  jac->reason       = PETSC_FALSE;
  jac->monitor      = PETSC_FALSE;
  jac->batch_target = 0;
  jac->rank_target  = 0;
  jac->nsolves_team = 1;
  jac->ksp->max_it  = 50; // this is really for GMRES w/o restarts
  PetscFunctionReturn(PETSC_SUCCESS);
}

// y <-- Ax
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMult(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, const PetscInt start, const PetscInt end, const PetscScalar *x_loc, PetscScalar *y_loc)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end), [=](const int rowb) {
    int                rowa = ic[rowb];
    int                n    = glb_Aai[rowa + 1] - glb_Aai[rowa];
    const PetscInt    *aj   = glb_Aaj + glb_Aai[rowa]; // global
    const PetscScalar *aa   = glb_Aaa + glb_Aai[rowa];
    PetscScalar        sum;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, n), [=](const int i, PetscScalar &lsum) { lsum += aa[i] * x_loc[r[aj[i]] - start]; }, sum);
    Kokkos::single(Kokkos::PerThread(team), [=]() { y_loc[rowb - start] = sum; });
  });
  team.team_barrier();
  return PETSC_SUCCESS;
}

// temp buffer per thread with reduction at end?
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMultTranspose(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, const PetscInt start, const PetscInt end, const PetscScalar *x_loc, PetscScalar *y_loc)
{
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, end - start), [=](int i) { y_loc[i] = 0; });
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end), [=](const int rowb) {
    int                rowa = ic[rowb];
    int                n    = glb_Aai[rowa + 1] - glb_Aai[rowa];
    const PetscInt    *aj   = glb_Aaj + glb_Aai[rowa]; // global
    const PetscScalar *aa   = glb_Aaa + glb_Aai[rowa];
    const PetscScalar  xx   = x_loc[rowb - start]; // rowb = ic[rowa] = ic[r[rowb]]
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, n), [=](const int &i) {
      PetscScalar val = aa[i] * xx;
      Kokkos::atomic_fetch_add(&y_loc[r[aj[i]] - start], val);
    });
  });
  team.team_barrier();
  return PETSC_SUCCESS;
}

// Apply diagonal (Jacobi) preconditioner: out[i] = diag[i] * in[i]
KOKKOS_INLINE_FUNCTION void PCApply_Diag(const team_member team, const PetscInt Nblk, const PetscScalar *diag, const PetscScalar *in, PetscScalar *out)
{
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { out[i] = diag[i] * in[i]; });
  team.team_barrier();
}

// Apply diagonal (Jacobi) preconditioner in-place: vec[i] *= diag[i]
KOKKOS_INLINE_FUNCTION void PCApplyInPlace_Diag(const team_member team, const PetscInt Nblk, const PetscScalar *diag, PetscScalar *vec)
{
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { vec[i] = diag[i] * vec[i]; });
  team.team_barrier();
}

// AMG sparse kernels (SpMV_Local, ComputeL1Norms, L1JacobiSmooth, NumericRAP, AMGVCycle)
// are in include/petsc/private/pcbjkokkosimpl.h

// AMG structs (AMGLevelInfo, AMGFineInfo, AMGCoarsestInfo) and V-cycle
// are now in include/petsc/private/pcbjkokkosimpl.h

typedef struct Batch_MetaData_TAG {
  PetscInt           flops;
  PetscInt           its;
  KSPConvergedReason reason;
} Batch_MetaData;

// Solve A(BB^-1)x = y with TFQMR and diagonal (Jacobi) PC. Right preconditioned to get un-preconditioned residual
static KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_TFQMR_Jac(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  int                Nblk = end - start, it, m, stride = stride_shared, idx = 0;
  PetscReal          dp, dpold, w, dpest, tau, psi, cm, r0;
  const PetscScalar *Diag = &glb_idiag[start];
  PetscScalar       *ptr  = work_space_shared, rho, rhoold, a, s, b, eta, etaold, psiold, cf, dpi;

  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *XX = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *R = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *RP = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *V = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *T = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Q = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *P = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *U = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *D = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *AUQ = V;

  // init: get b, zero x
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    R[rowb - start]  = glb_b[rowa];
    XX[rowb - start] = 0;
  });
  team.team_barrier();
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += R[idx] * PetscConj(R[idx]); }, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
  // diagnostics
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", 0, (double)dp); });
#endif
  if (dp < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
    it            = 0;
    goto done;
  }
  if (0 == maxit) {
    metad->reason = KSP_CONVERGED_ITS;
    it            = 0;
    goto done;
  }

  /* Make the initial Rp = R */
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { RP[idx] = R[idx]; });
  team.team_barrier();
  /* Set the initial conditions */
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;

  /* rhoold = (r,rp)     */
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += R[idx] * PetscConj(RP[idx]); }, rhoold);
  team.team_barrier();
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
    U[idx] = R[idx];
    P[idx] = R[idx];
    D[idx] = 0;
  });
  team.team_barrier();
  PCApply_Diag(team, Nblk, Diag, P, T);
  static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, V));

  it = 0;
  do {
    /* s <- (v,rp)          */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += V[idx] * PetscConj(RP[idx]); }, s);
    team.team_barrier();
    if (s == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done;
    }
    a = rhoold / s; /* a <- rho / s         */
    /* q <- u - a v    VecWAXPY(w,alpha,x,y): w = alpha x + y.     */
    /* t <- u + q           */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      Q[idx] = U[idx] - a * V[idx];
      T[idx] = U[idx] + Q[idx];
    });
    team.team_barrier();
    // KSP_PCApplyBAorAB
    PCApplyInPlace_Diag(team, Nblk, Diag, T);
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, AUQ));
    /* r <- r - a K (u + q) */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { R[idx] = R[idx] - a * AUQ[idx]; });
    team.team_barrier();
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += R[idx] * PetscConj(R[idx]); }, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
    for (m = 0; m < 2; m++) {
      if (!m) w = PetscSqrtReal(dp * dpold);
      else w = dp;
      psi = w / tau;
      cm  = 1.0 / PetscSqrtReal(1.0 + psi * psi);
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (!m) {
        /* D = U + cf D */
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { D[idx] = U[idx] + cf * D[idx]; });
      } else {
        /* D = Q + cf D */
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { D[idx] = Q[idx] + cf * D[idx]; });
      }
      team.team_barrier();
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { XX[idx] = XX[idx] + eta * D[idx]; });
      team.team_barrier();
      dpest = PetscSqrtReal(2 * it + m + 2.0) * tau;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (monitor && m == 1) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", it + 1, (double)dpest); });
#endif
      if (dpest < atol) {
        metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
        goto done;
      }
      if (dpest / r0 < rtol) {
        metad->reason = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS;
        goto done;
      }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (dpest / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: %d it, res=%e, r_0=%e\n", team.league_rank(), it, dpest, r0); });
        goto done;
      }
#else
      if (dpest / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
        goto done;
      }
#endif
      if (it + 1 == maxit) {
        metad->reason = KSP_CONVERGED_ITS;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: TFQMR %d:%d it, res=%e, r_0=%e r_res=%e\n", team.league_rank(), it, m, dpest, r0, dpest / r0); });
#endif
        goto done;
      }
      etaold = eta;
      psiold = psi;
    }

    /* rho <- (r,rp)       */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += R[idx] * PetscConj(RP[idx]); }, rho);
    team.team_barrier();
    if (rho == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done;
    }
    b = rho / rhoold; /* b <- rho / rhoold   */
    /* u <- r + b q        */
    /* p <- u + b(q + b p) */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      U[idx] = R[idx] + b * Q[idx];
      Q[idx] = Q[idx] + b * P[idx];
      P[idx] = U[idx] + b * Q[idx];
    });
    /* v <- K p  */
    team.team_barrier();
    PCApply_Diag(team, Nblk, Diag, P, T);
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, V));

    rhoold = rho;
    dpold  = dp;

    it++;
  } while (it < maxit);
done:
  // KSPUnwindPreconditioner
  PCApplyInPlace_Diag(team, Nblk, Diag, XX);
  // put x into Plex order
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = XX[rowb - start];
  });
  metad->its = it;
  {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange(team, start, end), [=](const int idx, int &lsum) { lsum += (glb_Aai[idx + 1] - glb_Aai[idx]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  }
  return PETSC_SUCCESS;
}

// Solve Ax = y with biCG and diagonal (Jacobi) PC
static KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_BICG_Jac(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  int                Nblk = end - start, it, stride = stride_shared, idx = 0; // start in shared mem
  PetscReal          dp, r0;
  const PetscScalar *Di  = &glb_idiag[start];
  PetscScalar       *ptr = work_space_shared, dpi, a = 1.0, beta, betaold = 1.0, t1, t2;

  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *XX = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Rl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Zl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Pl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Rr = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Zr = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Pr = ptr;
  ptr += stride;

  /*     r <- b (x is 0) */
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    Rl[rowb - start] = Rr[rowb - start] = glb_b[rowa];
    XX[rowb - start]                    = 0;
  });
  team.team_barrier();
  /*     z <- Br         */
  PCApply_Diag(team, Nblk, Di, Rr, Zr);
  PCApply_Diag(team, Nblk, Di, Rl, Zl);
  /*    dp <- r'*r       */
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Rr[idx] * PetscConj(Rr[idx]); }, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", 0, (double)dp); });
#endif
  if (dp < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
    it            = 0;
    goto done;
  }
  if (0 == maxit) {
    metad->reason = KSP_CONVERGED_ITS;
    it            = 0;
    goto done;
  }

  it = 0;
  do {
    /*     beta <- r'z     */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += Zr[idx] * PetscConj(Rl[idx]); }, beta);
    team.team_barrier();
#if PCBJKOKKOS_VERBOSE_LEVEL >= 6
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%7d beta = Z.R = %22.14e \n", i, (double)beta); });
  #endif
#endif
    if (beta == 0.0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done;
    }
    if (it == 0) {
      /*     p <- z          */
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
        Pr[idx] = Zr[idx];
        Pl[idx] = Zl[idx];
      });
    } else {
      t1 = beta / betaold;
      /*     p <- z + b* p   */
      t2 = PetscConj(t1);
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
        Pr[idx] = t1 * Pr[idx] + Zr[idx];
        Pl[idx] = t2 * Pl[idx] + Zl[idx];
      });
    }
    team.team_barrier();
    betaold = beta;
    /*     z <- Kp         */
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Pr, Zr));
    static_cast<void>(MatMultTranspose(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Pl, Zl));
    /*     dpi <- z'p      */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Zr[idx] * PetscConj(Pl[idx]); }, dpi);
    team.team_barrier();
    if (dpi == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done;
    }
    //
    a  = beta / dpi; /*     a = beta/p'z    */
    t1 = -a;
    t2 = PetscConj(t1);
    /*     x <- x + ap     */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      XX[idx] = XX[idx] + a * Pr[idx];
      Rr[idx] = Rr[idx] + t1 * Zr[idx];
      Rl[idx] = Rl[idx] + t2 * Zl[idx];
    });
    team.team_barrier();
    /*    dp <- r'*r       */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Rr[idx] * PetscConj(Rr[idx]); }, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", it + 1, (double)dp); });
#endif
    if (dp < atol) {
      metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
      goto done;
    }
    if (dp / r0 < rtol) {
      metad->reason = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS;
      goto done;
    }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (dp / r0 > dtol) {
      metad->reason = KSP_DIVERGED_DTOL;
      Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: %d it, res=%e, r_0=%e (BICG does this)\n", team.league_rank(), it, dp, r0); });
      goto done;
    }
#else
    if (dp / r0 > dtol) {
      metad->reason = KSP_DIVERGED_DTOL;
      goto done;
    }
#endif
    if (it + 1 == maxit) {
      metad->reason = KSP_CONVERGED_ITS; // don't worry about hitting max iterations
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: BICG %d it, res=%e, r_0=%e r_res=%e\n", team.league_rank(), it, dp, r0, dp / r0); });
#endif
      goto done;
    }
    /* z <- Br  */
    PCApply_Diag(team, Nblk, Di, Rr, Zr);
    PCApply_Diag(team, Nblk, Di, Rl, Zl);

    it++;
  } while (it < maxit);
done:
  // put x back into Plex order
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = XX[rowb - start];
  });
  metad->its = it;
  {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange(team, start, end), [=](const int idx, int &lsum) { lsum += (glb_Aai[idx + 1] - glb_Aai[idx]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  }
  return PETSC_SUCCESS;
}

// -----------------------------------------------------------------------
// BJSolve_TFQMR_AMG -- TFQMR with AMG V-cycle preconditioner
// -----------------------------------------------------------------------
static KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_TFQMR_AMG(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor, const AMGFineInfo amg_fine, const AMGLevelInfo *amg_levels, PetscInt amg_nlevels, const AMGCoarsestInfo amg_coarsest, PetscScalar *amg_work)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  int          Nblk = end - start, it, m, stride = stride_shared, idx = 0;
  PetscReal    dp, dpold, w, dpest, tau, psi, cm, r0;
  PetscScalar *ptr = work_space_shared, rho, rhoold, a, s, b, eta, etaold, psiold, cf, dpi;

  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *XX = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *R = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *RP = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *V = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *T = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Q = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *P = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *U = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *D = ptr;
  ptr += stride;
  (void)ptr;
  PetscScalar *AUQ = V;

  // init: get b, zero x
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    R[rowb - start]  = glb_b[rowa];
    XX[rowb - start] = 0;
  });
  team.team_barrier();
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &lsum) { lsum += R[i] * PetscConj(R[i]); }, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", 0, (double)dp); });
#endif
  if (dp < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
    it            = 0;
    goto done_tfqmr_amg;
  }
  if (0 == maxit) {
    metad->reason = KSP_CONVERGED_ITS;
    it            = 0;
    goto done_tfqmr_amg;
  }

  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { RP[i] = R[i]; });
  team.team_barrier();
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &dot) { dot += R[i] * PetscConj(RP[i]); }, rhoold);
  team.team_barrier();
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) {
    U[i] = R[i];
    P[i] = R[i];
    D[i] = 0;
  });
  team.team_barrier();
  // KP: AMG V-cycle applied to P -> T (preconditioner application)
  AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, P, T);
  team.team_barrier();
  static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, V));

  it = 0;
  do {
    /* s <- (v,rp) */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &dot) { dot += V[i] * PetscConj(RP[i]); }, s);
    team.team_barrier();
    if (s == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done_tfqmr_amg;
    }
    a = rhoold / s;
    /* q <- u - a v,  t <- u + q */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) {
      Q[i] = U[i] - a * V[i];
      T[i] = U[i] + Q[i];
    });
    team.team_barrier();
    /* KP: apply AMG in-place to T */
    AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, T, AUQ);
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, AUQ, T));
    /* r <- r - a K(u+q) */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { R[i] = R[i] - a * T[i]; });
    team.team_barrier();
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &lsum) { lsum += R[i] * PetscConj(R[i]); }, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
    for (m = 0; m < 2; m++) {
      if (!m) w = PetscSqrtReal(dp * dpold);
      else w = dp;
      psi = w / tau;
      cm  = 1.0 / PetscSqrtReal(1.0 + psi * psi);
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (!m) {
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { D[i] = U[i] + cf * D[i]; });
      } else {
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { D[i] = Q[i] + cf * D[i]; });
      }
      team.team_barrier();
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { XX[i] = XX[i] + eta * D[i]; });
      team.team_barrier();
      dpest = PetscSqrtReal(2 * it + m + 2.0) * tau;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (monitor && m == 1) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", it + 1, (double)dpest); });
#endif
      if (dpest < atol) {
        metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
        goto done_tfqmr_amg;
      }
      if (dpest / r0 < rtol) {
        metad->reason = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS;
        goto done_tfqmr_amg;
      }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (dpest / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: %d it, res=%e, r_0=%e\n", team.league_rank(), it, dpest, r0); });
        goto done_tfqmr_amg;
      }
#else
      if (dpest / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
        goto done_tfqmr_amg;
      }
#endif
      if (it + 1 == maxit) {
        metad->reason = KSP_CONVERGED_ITS;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: TFQMR_AMG %d:%d it, res=%e, r_0=%e r_res=%e\n", team.league_rank(), it, m, dpest, r0, dpest / r0); });
#endif
        goto done_tfqmr_amg;
      }
      etaold = eta;
      psiold = psi;
    }
    /* rho <- (r,rp) */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &dot) { dot += R[i] * PetscConj(RP[i]); }, rho);
    team.team_barrier();
    if (rho == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done_tfqmr_amg;
    }
    b = rho / rhoold;
    /* u <- r + b q,  p <- u + b(q + b p) */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) {
      U[i] = R[i] + b * Q[i];
      Q[i] = Q[i] + b * P[i];
      P[i] = U[i] + b * Q[i];
    });
    team.team_barrier();
    /* KP: AMG V-cycle applied to P -> T */
    AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, P, T);
    team.team_barrier();
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, V));
    rhoold = rho;
    dpold  = dp;
    it++;
  } while (it < maxit);
done_tfqmr_amg:
  // KSPUnwindPreconditioner: XX is in the unpreconditioned space; apply M^{-1} to get the actual solution.
  // Use T as scratch for the output of AMGVCycle.
  AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, XX, T);
  team.team_barrier();
  // put x back into Plex order
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = T[rowb - start];
  });
  metad->its = it;
  {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange(team, start, end), [=](const int i, int &lsum) { lsum += (glb_Aai[i + 1] - glb_Aai[i]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  }
  return PETSC_SUCCESS;
}

// -----------------------------------------------------------------------
// BJSolve_BICG_AMG -- BiCG with AMG V-cycle preconditioner
// -----------------------------------------------------------------------
static KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_BICG_AMG(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor, const AMGFineInfo amg_fine, const AMGLevelInfo *amg_levels, PetscInt amg_nlevels, const AMGCoarsestInfo amg_coarsest, PetscScalar *amg_work)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  int          Nblk = end - start, it, stride = stride_shared, idx = 0;
  PetscReal    dp, r0;
  PetscScalar *ptr = work_space_shared, dpi, a = 1.0, beta, betaold = 1.0, t1, t2;

  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *XX = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Rl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Zl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Pl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Rr = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Zr = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Pr = ptr;
  ptr += stride;
  (void)ptr;

  /* r <- b (x is 0) */
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    Rl[rowb - start] = Rr[rowb - start] = glb_b[rowa];
    XX[rowb - start]                    = 0;
  });
  team.team_barrier();
  /* z <- AMG(r) */
  AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, Rr, Zr);
  AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, Rl, Zl);
  /* dp <- r'*r */
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &lsum) { lsum += Rr[i] * PetscConj(Rr[i]); }, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", 0, (double)dp); });
#endif
  if (dp < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
    it            = 0;
    goto done_bicg_amg;
  }
  if (0 == maxit) {
    metad->reason = KSP_CONVERGED_ITS;
    it            = 0;
    goto done_bicg_amg;
  }

  it = 0;
  do {
    /* beta <- r'z */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &dot) { dot += Zr[i] * PetscConj(Rl[i]); }, beta);
    team.team_barrier();
    if (beta == 0.0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done_bicg_amg;
    }
    if (it == 0) {
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) {
        Pr[i] = Zr[i];
        Pl[i] = Zl[i];
      });
    } else {
      t1 = beta / betaold;
      t2 = PetscConj(t1);
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) {
        Pr[i] = t1 * Pr[i] + Zr[i];
        Pl[i] = t2 * Pl[i] + Zl[i];
      });
    }
    team.team_barrier();
    betaold = beta;
    /* z <- Kp */
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Pr, Zr));
    static_cast<void>(MatMultTranspose(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Pl, Zl));
    /* dpi <- z'p */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &lsum) { lsum += Zr[i] * PetscConj(Pl[i]); }, dpi);
    team.team_barrier();
    if (dpi == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done_bicg_amg;
    }
    a  = beta / dpi;
    t1 = -a;
    t2 = PetscConj(t1);
    /* x <- x + ap,  r <- r - a*Kp */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) {
      XX[i] = XX[i] + a * Pr[i];
      Rr[i] = Rr[i] + t1 * Zr[i];
      Rl[i] = Rl[i] + t2 * Zl[i];
    });
    team.team_barrier();
    /* dp <- r'*r */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &lsum) { lsum += Rr[i] * PetscConj(Rr[i]); }, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", it + 1, (double)dp); });
#endif
    if (dp < atol) {
      metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
      goto done_bicg_amg;
    }
    if (dp / r0 < rtol) {
      metad->reason = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS;
      goto done_bicg_amg;
    }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (dp / r0 > dtol) {
      metad->reason = KSP_DIVERGED_DTOL;
      Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: %d it, res=%e, r_0=%e (BICG_AMG)\n", team.league_rank(), it, dp, r0); });
      goto done_bicg_amg;
    }
#else
    if (dp / r0 > dtol) {
      metad->reason = KSP_DIVERGED_DTOL;
      goto done_bicg_amg;
    }
#endif
    if (it + 1 == maxit) {
      metad->reason = KSP_CONVERGED_ITS;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: BICG_AMG %d it, res=%e, r_0=%e r_res=%e\n", team.league_rank(), it, dp, r0, dp / r0); });
#endif
      goto done_bicg_amg;
    }
    /* z <- AMG(r) */
    AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, Rr, Zr);
    AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, Rl, Zl);
    it++;
  } while (it < maxit);
done_bicg_amg:
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = XX[rowb - start];
  });
  metad->its = it;
  {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange(team, start, end), [=](const int i, int &lsum) { lsum += (glb_Aai[i + 1] - glb_Aai[i]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  }
  return PETSC_SUCCESS;
}

#if !defined(PETSC_USE_COMPLEX)
// Native GMRES with diagonal (Jacobi) preconditioning.
// Right-preconditioned: solves A(D^{-1}y)=b, x = D^{-1}y.
// nwork = maxit + 3 (V[0..maxit] + Z + XX).
static KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_GMRES_Jac(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar *glb_idiag, const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor, PetscScalar *gmres_hwork)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  const int Nblk = end - start;
  // All work vectors go to global memory (GMRES needs maxit+3 vectors)
  PetscScalar *ptr = work_space_global;
  // V[0..maxit] -- Krylov basis vectors
  PetscScalar *V0 = ptr;
  ptr += stride_global;
  // Z -- preconditioned vector (D^{-1} * Vj)
  PetscScalar *Z = ptr + maxit * stride_global; // after V[1..maxit]
  // XX -- solution in preconditioned space
  PetscScalar *XX = Z + stride_global;
  // Hessenberg matrix H, Givens rotations cs/sn, and g vector stored in gmres_hwork
  // Layout: H[(maxit+1)*maxit] | cs[maxit] | sn[maxit] | g[maxit+1]
  PetscScalar *H  = gmres_hwork;
  PetscScalar *cs = H + (maxit + 1) * maxit;
  PetscScalar *sn = cs + maxit;
  PetscScalar *g  = sn + maxit;

  // init: r0 = b - A*x0, but x0 = 0, so r0 = b
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    V0[rowb - start] = glb_b[rowa];
    XX[rowb - start] = 0;
  });
  team.team_barrier();

  // beta = ||r0||
  PetscScalar dpi;
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &lsum) { lsum += V0[i] * PetscConj(V0[i]); }, dpi);
  team.team_barrier();
  PetscReal beta = PetscSqrtReal(PetscRealPart(dpi));
  PetscReal r0   = beta;

  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", 0, (double)beta); });
  #endif
  if (beta < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
    metad->its    = 0;
    goto done_gmres_jac;
  }
  if (0 == maxit) {
    metad->reason = KSP_CONVERGED_ITS;
    metad->its    = 0;
    goto done_gmres_jac;
  }

  // V[0] = r0 / beta
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { V0[i] /= beta; });
  team.team_barrier();

  // Initialize g[0] = beta, rest = 0
  Kokkos::single(Kokkos::PerTeam(team), [=]() {
    g[0] = beta;
    for (int j = 1; j <= maxit; j++) g[j] = 0;
    for (int j = 0; j < (maxit + 1) * maxit; j++) H[j] = 0;
  });
  team.team_barrier();

  {
    int j;
    for (j = 0; j < maxit; j++) {
      PetscScalar *Vj   = work_space_global + j * stride_global;
      PetscScalar *Vjp1 = work_space_global + (j + 1) * stride_global;

      // Z = D^{-1} * V[j]  (diagonal Jacobi right preconditioning)
      parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) { Z[rowb - start] = glb_idiag[rowb] * Vj[rowb - start]; });
      team.team_barrier();

      // W = A * Z  (using global permuted CSR)
      static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Z, Vjp1));

      // Modified Gram-Schmidt orthogonalization
      for (int i = 0; i <= j; i++) {
        PetscScalar *Vi = work_space_global + i * stride_global;
        PetscScalar  hij;
        parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int k, PetscScalar &dot) { dot += Vjp1[k] * PetscConj(Vi[k]); }, hij);
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team), [=]() { H[i + j * (maxit + 1)] = hij; });
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int k) { Vjp1[k] -= hij * Vi[k]; });
        team.team_barrier();
      }

      // h_{j+1,j} = ||V[j+1]||
      PetscScalar hjp1j_sq;
      parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int k, PetscScalar &lsum) { lsum += Vjp1[k] * PetscConj(Vjp1[k]); }, hjp1j_sq);
      team.team_barrier();
      PetscReal hjp1j = PetscSqrtReal(PetscRealPart(hjp1j_sq));
      Kokkos::single(Kokkos::PerTeam(team), [=]() { H[(j + 1) + j * (maxit + 1)] = hjp1j; });

      // Normalize V[j+1]
      if (hjp1j > 0) parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int k) { Vjp1[k] /= hjp1j; });
      team.team_barrier();

      // Apply previous Givens rotations to column j of H
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (int i = 0; i < j; i++) {
          PetscScalar temp             = cs[i] * H[i + j * (maxit + 1)] + sn[i] * H[(i + 1) + j * (maxit + 1)];
          H[(i + 1) + j * (maxit + 1)] = -sn[i] * H[i + j * (maxit + 1)] + cs[i] * H[(i + 1) + j * (maxit + 1)];
          H[i + j * (maxit + 1)]       = temp;
        }
        // Compute new Givens rotation for row j
        PetscReal a_val = PetscRealPart(H[j + j * (maxit + 1)]);
        PetscReal b_val = PetscRealPart(H[(j + 1) + j * (maxit + 1)]);
        PetscReal r_val = PetscSqrtReal(a_val * a_val + b_val * b_val);
        if (r_val > 0) {
          cs[j] = a_val / r_val;
          sn[j] = b_val / r_val;
        } else {
          cs[j] = 1.0;
          sn[j] = 0.0;
        }
        H[j + j * (maxit + 1)]       = r_val;
        H[(j + 1) + j * (maxit + 1)] = 0;
        // Apply to g
        PetscScalar temp = cs[j] * g[j] + sn[j] * g[j + 1];
        g[j + 1]         = -sn[j] * g[j] + cs[j] * g[j + 1];
        g[j]             = temp;
      });
      team.team_barrier();

      PetscReal res = PetscAbsScalar(g[j + 1]);
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", j + 1, (double)res); });
  #endif
      if (res < atol) {
        metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
        metad->its    = j + 1;
        j++;
        goto solve_gmres_jac;
      }
      if (res / r0 < rtol) {
        metad->reason = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS;
        metad->its    = j + 1;
        j++;
        goto solve_gmres_jac;
      }
      if (res / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: GMRES_JAC %d it, res=%e, r_0=%e\n", team.league_rank(), j + 1, res, r0); });
  #endif
        metad->its = j + 1;
        j++;
        goto solve_gmres_jac;
      }
    }
    metad->reason = KSP_CONVERGED_ITS;
    metad->its    = maxit;
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d: GMRES_JAC max iterations %d reached\n", team.league_rank(), (int)maxit); });
  #endif

  solve_gmres_jac:
    // Back-substitution: solve H * y = g (upper triangular, j equations)
    {
      int m = j;
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (int i = m - 1; i >= 0; i--) {
          PetscScalar sum = g[i];
          for (int k = i + 1; k < m; k++) sum -= H[i + k * (maxit + 1)] * g[k]; // reuse g for y
          g[i] = sum / H[i + i * (maxit + 1)];
        }
      });
      team.team_barrier();

      // x = D^{-1} * V * y = sum_j y[j] * D^{-1} V[j]
      // First accumulate V*y into XX, then apply D^{-1}
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int k) { XX[k] = 0; });
      team.team_barrier();
      for (int i = 0; i < m; i++) {
        PetscScalar *Vi = work_space_global + i * stride_global;
        PetscScalar  yi = g[i];
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int k) { XX[k] += yi * Vi[k]; });
        team.team_barrier();
      }
      // Apply D^{-1} to get the actual solution (right preconditioning)
      parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
        int rowa    = ic[rowb];
        glb_x[rowa] = glb_idiag[rowb] * XX[rowb - start];
      });
    }
  }
  {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange(team, start, end), [=](const int i, int &lsum) { lsum += (glb_Aai[i + 1] - glb_Aai[i]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  }
  return PETSC_SUCCESS;

done_gmres_jac:
  // Zero solution (converged at iteration 0)
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = 0;
  });
  metad->flops = 0;
  return PETSC_SUCCESS;
}

// -----------------------------------------------------------------------
static KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_GMRES_AMG(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor, const AMGFineInfo amg_fine, const AMGLevelInfo *amg_levels, PetscInt amg_nlevels, const AMGCoarsestInfo amg_coarsest, PetscScalar *amg_work, PetscScalar *gmres_hwork)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  const int Nblk = end - start;
  // All work vectors go to global memory (GMRES needs maxit+3 vectors)
  PetscScalar *ptr = work_space_global;
  // V[0..maxit] -- Krylov basis vectors
  // We store pointers to V[j] as offsets: V_j = work_space_global + j * stride_global
  // V[0] is also the initial residual
  PetscScalar *V0 = ptr;
  ptr += stride_global;
  // Z -- preconditioned vector
  PetscScalar *Z = ptr + maxit * stride_global; // after V[1..maxit]
  // XX -- solution in preconditioned space
  PetscScalar *XX = Z + stride_global;
  // Hessenberg matrix H, Givens rotations cs/sn, and g vector stored in gmres_hwork
  // Layout: H[(maxit+1)*maxit] | cs[maxit] | sn[maxit] | g[maxit+1]
  PetscScalar *H  = gmres_hwork;
  PetscScalar *cs = H + (maxit + 1) * maxit;
  PetscScalar *sn = cs + maxit;
  PetscScalar *g  = sn + maxit;

  // init: r0 = b - A*x0, but x0 = 0, so r0 = b
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    V0[rowb - start] = glb_b[rowa];
    XX[rowb - start] = 0;
  });
  team.team_barrier();

  // beta = ||r0||
  PetscScalar dpi;
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int i, PetscScalar &lsum) { lsum += V0[i] * PetscConj(V0[i]); }, dpi);
  team.team_barrier();
  PetscReal beta = PetscSqrtReal(PetscRealPart(dpi));
  PetscReal r0   = beta;

  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", 0, (double)beta); });
  #endif
  if (beta < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
    metad->its    = 0;
    goto done_gmres_amg;
  }
  if (0 == maxit) {
    metad->reason = KSP_CONVERGED_ITS;
    metad->its    = 0;
    goto done_gmres_amg;
  }

  // V[0] = r0 / beta
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int i) { V0[i] /= beta; });
  team.team_barrier();

  // Initialize g[0] = beta, rest = 0
  Kokkos::single(Kokkos::PerTeam(team), [=]() {
    g[0] = beta;
    for (int j = 1; j <= maxit; j++) g[j] = 0;
    for (int j = 0; j < (maxit + 1) * maxit; j++) H[j] = 0;
  });
  team.team_barrier();

  {
    int j;
    for (j = 0; j < maxit; j++) {
      PetscScalar *Vj   = work_space_global + j * stride_global;
      PetscScalar *Vjp1 = work_space_global + (j + 1) * stride_global;

      // Z = M^{-1} V[j]  (right preconditioning)
      AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, Vj, Z);
      team.team_barrier();

      // W = A * Z  (using global permuted CSR)
      static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Z, Vjp1));

      // Modified Gram-Schmidt orthogonalization
      for (int i = 0; i <= j; i++) {
        PetscScalar *Vi = work_space_global + i * stride_global;
        PetscScalar  hij;
        parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int k, PetscScalar &dot) { dot += Vjp1[k] * PetscConj(Vi[k]); }, hij);
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team), [=]() { H[i + j * (maxit + 1)] = hij; });
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int k) { Vjp1[k] -= hij * Vi[k]; });
        team.team_barrier();
      }

      // h_{j+1,j} = ||V[j+1]||
      PetscScalar hjp1j_sq;
      parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int k, PetscScalar &lsum) { lsum += Vjp1[k] * PetscConj(Vjp1[k]); }, hjp1j_sq);
      team.team_barrier();
      PetscReal hjp1j = PetscSqrtReal(PetscRealPart(hjp1j_sq));
      Kokkos::single(Kokkos::PerTeam(team), [=]() { H[(j + 1) + j * (maxit + 1)] = hjp1j; });

      // Normalize V[j+1]
      if (hjp1j > 0) parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int k) { Vjp1[k] /= hjp1j; });
      team.team_barrier();

      // Apply previous Givens rotations to column j of H
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (int i = 0; i < j; i++) {
          PetscScalar temp             = cs[i] * H[i + j * (maxit + 1)] + sn[i] * H[(i + 1) + j * (maxit + 1)];
          H[(i + 1) + j * (maxit + 1)] = -sn[i] * H[i + j * (maxit + 1)] + cs[i] * H[(i + 1) + j * (maxit + 1)];
          H[i + j * (maxit + 1)]       = temp;
        }
        // Compute new Givens rotation for row j
        PetscReal a_val = PetscRealPart(H[j + j * (maxit + 1)]);
        PetscReal b_val = PetscRealPart(H[(j + 1) + j * (maxit + 1)]);
        PetscReal r_val = PetscSqrtReal(a_val * a_val + b_val * b_val);
        if (r_val > 0) {
          cs[j] = a_val / r_val;
          sn[j] = b_val / r_val;
        } else {
          cs[j] = 1.0;
          sn[j] = 0.0;
        }
        H[j + j * (maxit + 1)]       = r_val;
        H[(j + 1) + j * (maxit + 1)] = 0;
        // Apply to g
        PetscScalar temp = cs[j] * g[j] + sn[j] * g[j + 1];
        g[j + 1]         = -sn[j] * g[j] + cs[j] * g[j + 1];
        g[j]             = temp;
      });
      team.team_barrier();

      PetscReal res = PetscAbsScalar(g[j + 1]);
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", j + 1, (double)res); });
  #endif
      if (res < atol) {
        metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
        metad->its    = j + 1;
        j++;
        goto solve_gmres_amg;
      }
      if (res / r0 < rtol) {
        metad->reason = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS;
        metad->its    = j + 1;
        j++;
        goto solve_gmres_amg;
      }
      if (res / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d diverged: GMRES_AMG %d it, res=%e, r_0=%e\n", team.league_rank(), j + 1, res, r0); });
  #endif
        metad->its = j + 1;
        j++;
        goto solve_gmres_amg;
      }
    }
    metad->reason = KSP_CONVERGED_ITS;
    metad->its    = maxit;
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("WARNING block %d: GMRES_AMG max iterations %d reached\n", team.league_rank(), (int)maxit); });
  #endif

  solve_gmres_amg:
    // Back-substitution: solve H * y = g (upper triangular, j equations)
    {
      int m = j; // number of GMRES iterations completed
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (int i = m - 1; i >= 0; i--) {
          PetscScalar sum = g[i];
          for (int k = i + 1; k < m; k++) sum -= H[i + k * (maxit + 1)] * g[k]; // reuse g for y
          g[i] = sum / H[i + i * (maxit + 1)];
        }
      });
      team.team_barrier();

      // x = x0 + M^{-1} * V * y = sum_j y[j] * M^{-1} V[j]
      // First compute V*y into XX, then apply M^{-1}
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int k) { XX[k] = 0; });
      team.team_barrier();
      for (int i = 0; i < m; i++) {
        PetscScalar *Vi = work_space_global + i * stride_global;
        PetscScalar  yi = g[i];
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int k) { XX[k] += yi * Vi[k]; });
        team.team_barrier();
      }
      // Apply M^{-1} to get the actual solution (right preconditioning)
      AMGVCycle(team, amg_fine, amg_levels, amg_nlevels, amg_coarsest, amg_work, XX, Z);
      team.team_barrier();
      // Copy to output
      parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
        int rowa    = ic[rowb];
        glb_x[rowa] = Z[rowb - start];
      });
    }
  }
  {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange(team, start, end), [=](const int i, int &lsum) { lsum += (glb_Aai[i + 1] - glb_Aai[i]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  }
  return PETSC_SUCCESS;

done_gmres_amg:
  // Zero solution (converged at iteration 0)
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = 0;
  });
  metad->flops = 0;
  return PETSC_SUCCESS;
}
#endif /* !defined(PETSC_USE_COMPLEX) */

// KSP solver solve Ax = b; xout is output, bin is input
static PetscErrorCode PCApply_BJKOKKOS(PC pc, Vec bin, Vec xout)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  Mat            A = pc->pmat, Aseq = A;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  if (!A->spptr) Aseq = ((Mat_MPIAIJ *)A->data)->A; // MPI
  PetscCall(MatSeqAIJKokkosSyncDevice(Aseq));
  {
    PetscInt       maxit = jac->ksp->max_it;
    const PetscInt conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp == 0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
    PetscCheck(team_size <= PCBJKOKKOS_TEAM_SIZE, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "team_size %" PetscInt_FMT " exceeds PCBJKOKKOS_TEAM_SIZE %d (AMG spa buffer overflow)", team_size, PCBJKOKKOS_TEAM_SIZE);
    const PetscInt     nwork = jac->nwork, nBlk = jac->nBlocks;
    PetscScalar       *glb_xdata = NULL, *dummy;
    PetscReal          rtol = jac->ksp->rtol, atol = jac->ksp->abstol, dtol = jac->ksp->divtol;
    const PetscScalar *glb_idiag = jac->d_idiag_k->data(), *glb_bdata = NULL;
    const PetscInt    *glb_Aai, *glb_Aaj, *d_bid_eqOffset = jac->d_bid_eqOffset_k->data();
    const PetscScalar *glb_Aaa;
    const PetscInt    *d_isicol = jac->d_isicol_k->data(), *d_isrow = jac->d_isrow_k->data();
    PCFailedReason     pcreason;
    KSPIndex           ksp_type_idx = jac->ksp_type_idx;
    PetscMemType       mtype;
    PetscContainer     container;
    PetscInt           batch_sz;                // the number of repeated DMs, [DM_e_1, DM_e_2, DM_e_batch_sz, DM_i_1, ...]
    VecScatter         plex_batch = NULL;       // not used
    Vec                bvec;                    // a copy of b for scatter (just alias to bin now)
    PetscBool          monitor  = jac->monitor; // captured
    PetscInt           view_bid = jac->batch_target;
    MatInfo            info;

    PetscCall(MatSeqAIJGetCSRAndMemType(Aseq, &glb_Aai, &glb_Aaj, &dummy, &mtype));
    PetscCheck(PetscMemTypeDevice(mtype) || Kokkos::DefaultExecutionSpace().concurrency() < 1000, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "MatSeqAIJGetCSRAndMemType returned host memory but Kokkos execution space is a device; PCBJKOKKOS requires a Kokkos-aware (device) matrix");
    jac->max_nits = 0;
    glb_Aaa       = dummy;
    if (jac->rank_target != rank) view_bid = -1; // turn off all but one process
    PetscCall(MatGetInfo(A, MAT_LOCAL, &info));
    // get field major is to map plex IO to/from block/field major
    PetscCall(PetscObjectQuery((PetscObject)A, "plex_batch_is", (PetscObject *)&container));
    if (container) {
      PetscCall(VecDuplicate(bin, &bvec));
      PetscCall(PetscContainerGetPointer(container, &plex_batch));
      PetscCall(VecScatterBegin(plex_batch, bin, bvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(plex_batch, bin, bvec, INSERT_VALUES, SCATTER_FORWARD));
      SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "No plex_batch_is -- require NO field major ordering for now");
    } else {
      bvec = bin;
    }
    // get x
    PetscCall(VecGetArrayAndMemType(xout, &glb_xdata, &mtype));
#if defined(PETSC_HAVE_CUDA)
    PetscCheck(PetscMemTypeDevice(mtype), PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "No GPU data for x %d != %d", static_cast<int>(mtype), static_cast<int>(PETSC_MEMTYPE_DEVICE));
#endif
    PetscCall(VecGetArrayReadAndMemType(bvec, &glb_bdata, &mtype));
#if defined(PETSC_HAVE_CUDA)
    PetscCheck(PetscMemTypeDevice(mtype), PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "No GPU data for b");
#endif
    // get batch size
    PetscCall(PetscObjectQuery((PetscObject)A, "batch size", (PetscObject *)&container));
    if (container) {
      PetscInt *pNf = NULL;
      PetscCall(PetscContainerGetPointer(container, &pNf));
      batch_sz = *pNf; // number of times to repeat the DMs
    } else batch_sz = 1;
    PetscCheck(nBlk % batch_sz == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "batch_sz = %" PetscInt_FMT ", nBlk = %" PetscInt_FMT, batch_sz, nBlk);
    if (ksp_type_idx == BATCH_KSP_GMRESKK_JAC_IDX || ksp_type_idx == BATCH_KSP_GMRESKK_AMG_IDX) {
      // KK solver - move PETSc data into Kokkos Views, setup solver, solve, move data out of Kokkos, process metadata (convergence tests, etc.)
#if defined(PETSC_HAVE_KOKKOS_KERNELS_BATCH)
      if (ksp_type_idx == BATCH_KSP_GMRESKK_JAC_IDX) PetscCall(PCApply_BJKOKKOSKERNELS(pc, glb_bdata, glb_xdata, glb_Aai, glb_Aaj, glb_Aaa, team_size, info, batch_sz, &pcreason));
      else {
        // GMRES+AMG: use cached per-grid AMG device views (built once in PCSetUp)
        PetscInt     amg_work_stride_l = jac->amg_work_stride;
        PetscScalar *d_amg_work_ptr_l  = jac->d_amg_work->data();

        // Use cached device views from PCSetUp
        const AMGFineInfo     *d_fine_arr_l     = jac->d_amg_fine_arr_k->data();
        const AMGLevelInfo    *d_levels_flat_l  = jac->d_amg_levels_flat_k->data();
        const AMGCoarsestInfo *d_coarsest_arr_l = jac->d_amg_coarsest_arr_k->data();
        const PetscInt        *d_nlevels_arr_l  = jac->d_amg_nlevels_k->data();
        const PetscInt        *d_level_offs_l   = jac->d_amg_level_offsets_k->data();
        const PetscInt        *d_b2g_l          = jac->d_block_to_grid_k->data();
        // --- Phase: BJKOKKOS_AMG_RAP (KK-Kernels path) ---
        // Recompute Galerkin product every PCApply: the fine-grid matrix values (glb_Aaa)
        // change on each Newton step, so the coarse-grid operators must be rebuilt.
        // Capture precomputed gidx pointer and stride for direct indexed read.
        const PetscInt *d_fine_aa_gidx_ptr_l  = jac->d_fine_aa_gidx ? jac->d_fine_aa_gidx->data() : nullptr;
        const PetscInt  fine_aa_gidx_stride_l = jac->fine_aa_gidx_stride;
        {
          PetscCall(PetscLogEventBegin(BJKOKKOS_AMG_RAP, pc, 0, 0, 0));
          Kokkos::Profiling::pushRegion("BJKOKKOS_AMG_RAP");
          Kokkos::parallel_for(
            "AMG_RAP_GMRES", Kokkos::TeamPolicy<>(nBlk, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
              const int              blkID        = team.league_rank();
              const int              gridID       = d_b2g_l[blkID];
              const AMGFineInfo     &fine         = d_fine_arr_l[gridID];
              const AMGLevelInfo    *levels       = d_levels_flat_l + d_level_offs_l[gridID];
              const PetscInt         nlev         = d_nlevels_arr_l[gridID];
              const AMGCoarsestInfo &coarsest     = d_coarsest_arr_l[gridID];
              PetscScalar           *blk_amg_work = d_amg_work_ptr_l + (PetscInt)blkID * amg_work_stride_l;
              {
                PetscScalar *f_aa = blk_amg_work + fine.off_fine_aa;
                // Direct indexed read -- no linear search.
                // d_fine_aa_gidx_ptr_l[blkID * stride + lk] = position in glb_Aaa[] for entry lk.
                const PetscInt *blk_gidx = d_fine_aa_gidx_ptr_l + (PetscInt)blkID * fine_aa_gidx_stride_l;
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team, fine.fine_nnz), [=](const int lk) { f_aa[lk] = glb_Aaa[blk_gidx[lk]]; });
                team.team_barrier();
                PetscScalar *l1_fine = blk_amg_work + fine.off_l1;
                ComputeSmootherNorms(team, fine.fine_ai, fine.fine_aj, f_aa, fine.nrows, l1_fine, fine.smoother_type);
              }
              for (PetscInt lev = 0; lev < nlev; lev++) {
                const AMGLevelInfo &L   = levels[lev];
                PetscScalar        *Ac  = blk_amg_work + L.off_Ac_aa;
                PetscScalar        *l1  = blk_amg_work + L.off_l1;
                PetscScalar        *spa = blk_amg_work + L.off_spa;
                if (lev == 0) {
                  const PetscScalar *f_aa_loc = blk_amg_work + fine.off_fine_aa;
                  NumericRAP(team, fine.fine_ai, fine.fine_aj, f_aa_loc, fine.nrows, L.P_ai, L.P_aj, L.P_aa, L.nrows_fine, L.R_ai, L.R_aj, L.R_aa, L.nrows_coarse, L.Ac_ai, L.Ac_aj, Ac, spa);
                } else {
                  const AMGLevelInfo &Lprev   = levels[lev - 1];
                  const PetscScalar  *Ac_prev = blk_amg_work + Lprev.off_Ac_aa;
                  NumericRAP(team, Lprev.Ac_ai, Lprev.Ac_aj, Ac_prev, Lprev.nrows_coarse, L.P_ai, L.P_aj, L.P_aa, L.nrows_fine, L.R_ai, L.R_aj, L.R_aa, L.nrows_coarse, L.Ac_ai, L.Ac_aj, Ac, spa);
                }
                ComputeSmootherNorms(team, L.Ac_ai, L.Ac_aj, Ac, L.nrows_coarse, l1, L.smoother_type);
              }
              {
                const PetscScalar *Ac_c = blk_amg_work + coarsest.off_Ac_aa;
                PetscScalar       *l1_c = blk_amg_work + coarsest.off_l1;
                ComputeSmootherNorms(team, coarsest.Ac_ai, coarsest.Ac_aj, Ac_c, coarsest.nrows, l1_c, coarsest.smoother_type);
              }
            });
          Kokkos::fence();
          Kokkos::Profiling::popRegion(); // BJKOKKOS_AMG_RAP (KK path)
          PetscCall(PetscLogEventEnd(BJKOKKOS_AMG_RAP, pc, 0, 0, 0));
        }

        PetscCall(PCApply_BJKOKKOSKERNELS_AMG(pc, glb_bdata, glb_xdata, glb_Aai, glb_Aaj, glb_Aaa, team_size, info, batch_sz, &pcreason, d_fine_arr_l, d_levels_flat_l, d_level_offs_l, d_nlevels_arr_l, d_coarsest_arr_l, d_b2g_l, d_amg_work_ptr_l, amg_work_stride_l));
      }
#else
      PetscCheck(ksp_type_idx != BATCH_KSP_GMRESKK_JAC_IDX && ksp_type_idx != BATCH_KSP_GMRESKK_AMG_IDX, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Type: BATCH_KSP_GMRES not supported for complex");
#endif
    } else { // Kokkos Krylov
      using scr_mem_t    = Kokkos::DefaultExecutionSpace::scratch_memory_space;
      using vect2D_scr_t = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, scr_mem_t>;
      Kokkos::View<Batch_MetaData *, Kokkos::DefaultExecutionSpace> d_metadata("solver meta data", nBlk);
      int                                                           stride_shared, stride_global;
      d_bid_eqOffset = jac->d_bid_eqOffset_k->data();
      // solve each block independently
      int scr_bytes_team_shared = 0, nShareVec = 0, nGlobBVec = 0;
      if (jac->const_block_size) { // use shared memory for work vectors only if constant block size - TODO: test efficiency loss
        if (ksp_type_idx == BATCH_KSP_GMRES_AMG_IDX || ksp_type_idx == BATCH_KSP_GMRES_JAC_IDX) {
          // GMRES needs all work vectors in global memory (random access to V[j] during orthogonalization)
          stride_shared         = 0;
          nShareVec             = 0;
          nGlobBVec             = nwork;
          scr_bytes_team_shared = 0;
        } else {
          size_t      maximum_shared_mem_size = 64000;
          PetscDevice device;
          PetscCall(PetscDeviceGetDefault_Internal(&device));
          PetscCall(PetscDeviceGetAttribute(device, PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK, &maximum_shared_mem_size));
          stride_shared = jac->const_block_size;                                                   // captured
          nShareVec     = maximum_shared_mem_size / (jac->const_block_size * sizeof(PetscScalar)); // integer floor, number of vectors that fit in shared
          if (nShareVec > nwork) nShareVec = nwork;
          else nGlobBVec = nwork - nShareVec;
          scr_bytes_team_shared = jac->const_block_size * nShareVec * sizeof(PetscScalar);
        }
      } else {
        scr_bytes_team_shared = 0;
        stride_shared         = 0;
        nGlobBVec             = nwork; // not needed == fix
      }
      (void)nGlobBVec;        // suppress unused-variable warning in complex builds where verbose logging is disabled
      stride_global = jac->n; // captured
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePushA("batch-kokkos-solve");
#endif
      // Use the pre-allocated persistent work buffer (allocated once in PCSetUp to avoid GPU malloc fragmentation).
      auto &d_work_vecs_k = *jac->d_work_vecs_k;
#if PCBJKOKKOS_VERBOSE_LEVEL > 1
      PetscCall(PetscInfo(pc, "\tn = %d. %d shared bytes/team, global_buff_words=%" PetscInt_FMT ", rtol=%e, num blocks %d, team_size=%d, %d vector threads, %d shared vectors, %d global vectors\n", (int)jac->n, scr_bytes_team_shared,
                          (PetscInt)jac->d_work_vecs_k->extent(0), rtol, (int)nBlk, (int)team_size, PCBJKOKKOS_VEC_SIZE, nShareVec, nGlobBVec));
#endif
      PetscScalar *d_work_vecs = d_work_vecs_k.data();

      // For AMG types, use cached per-grid AMG device views (built once in PCSetUp),
      // run Galerkin recomputation kernel, then pass them into the Krylov solver.
      PetscScalar *d_amg_work_ptr  = NULL;
      PetscInt     amg_work_stride = 0;
      // Device pointers for per-grid AMG data -- declared at outer scope so the Solve lambda can capture them
      const AMGFineInfo     *d_fine_arr     = NULL;
      const AMGLevelInfo    *d_levels_flat  = NULL;
      const AMGCoarsestInfo *d_coarsest_arr = NULL;
      const PetscInt        *d_nlevels_arr  = NULL;
      const PetscInt        *d_level_offs   = NULL;
      const PetscInt        *d_b2g          = NULL;
      if (ksp_type_idx == BATCH_KSP_BICG_AMG_IDX || ksp_type_idx == BATCH_KSP_TFQMR_AMG_IDX || ksp_type_idx == BATCH_KSP_GMRES_AMG_IDX) {
        amg_work_stride = jac->amg_work_stride;
        d_amg_work_ptr  = jac->d_amg_work->data();

        // Use cached device views from PCSetUp
        d_fine_arr     = jac->d_amg_fine_arr_k->data();
        d_levels_flat  = jac->d_amg_levels_flat_k->data();
        d_coarsest_arr = jac->d_amg_coarsest_arr_k->data();
        d_nlevels_arr  = jac->d_amg_nlevels_k->data();
        d_level_offs   = jac->d_amg_level_offsets_k->data();
        d_b2g          = jac->d_block_to_grid_k->data();
        // --- Phase: BJKOKKOS_AMG_RAP (Kokkos-Krylov path) ---
        // Recompute Galerkin product every PCApply: the fine-grid matrix values (glb_Aaa)
        // change on each Newton step, so the coarse-grid operators must be rebuilt.
        // Capture precomputed gidx pointer and stride for direct indexed read.
        const PetscInt *d_fine_aa_gidx_ptr  = jac->d_fine_aa_gidx ? jac->d_fine_aa_gidx->data() : nullptr;
        const PetscInt  fine_aa_gidx_stride = jac->fine_aa_gidx_stride;
        {
          PetscCall(PetscLogEventBegin(BJKOKKOS_AMG_RAP, pc, 0, 0, 0));
          Kokkos::Profiling::pushRegion("BJKOKKOS_AMG_RAP");
          Kokkos::parallel_for(
            "AMG_RAP", Kokkos::TeamPolicy<>(nBlk, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
              const int              blkID        = team.league_rank();
              const int              gridID       = d_b2g[blkID];
              const AMGFineInfo     &fine         = d_fine_arr[gridID];
              const AMGLevelInfo    *levels       = d_levels_flat + d_level_offs[gridID];
              const PetscInt         nlev         = d_nlevels_arr[gridID];
              const AMGCoarsestInfo &coarsest     = d_coarsest_arr[gridID];
              PetscScalar           *blk_amg_work = d_amg_work_ptr + (PetscInt)blkID * amg_work_stride;
              // Extract fine-grid Aa values from global permuted matrix into local work buffer.
              {
                PetscScalar *f_aa = blk_amg_work + fine.off_fine_aa;
                // Direct indexed read -- no linear search.
                // d_fine_aa_gidx_ptr[blkID * stride + lk] = position in glb_Aaa[] for entry lk.
                const PetscInt *blk_gidx = d_fine_aa_gidx_ptr + (PetscInt)blkID * fine_aa_gidx_stride;
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team, fine.fine_nnz), [=](const int lk) { f_aa[lk] = glb_Aaa[blk_gidx[lk]]; });
                team.team_barrier();
                PetscScalar *l1_fine = blk_amg_work + fine.off_l1;
                ComputeSmootherNorms(team, fine.fine_ai, fine.fine_aj, f_aa, fine.nrows, l1_fine, fine.smoother_type);
              }
              for (PetscInt lev = 0; lev < nlev; lev++) {
                const AMGLevelInfo &L   = levels[lev];
                PetscScalar        *Ac  = blk_amg_work + L.off_Ac_aa;
                PetscScalar        *l1  = blk_amg_work + L.off_l1;
                PetscScalar        *spa = blk_amg_work + L.off_spa;
                if (lev == 0) {
                  const PetscScalar *f_aa_loc = blk_amg_work + fine.off_fine_aa;
                  NumericRAP(team, fine.fine_ai, fine.fine_aj, f_aa_loc, fine.nrows, L.P_ai, L.P_aj, L.P_aa, L.nrows_fine, L.R_ai, L.R_aj, L.R_aa, L.nrows_coarse, L.Ac_ai, L.Ac_aj, Ac, spa);
                } else {
                  const AMGLevelInfo &Lprev   = levels[lev - 1];
                  const PetscScalar  *Ac_prev = blk_amg_work + Lprev.off_Ac_aa;
                  NumericRAP(team, Lprev.Ac_ai, Lprev.Ac_aj, Ac_prev, Lprev.nrows_coarse, L.P_ai, L.P_aj, L.P_aa, L.nrows_fine, L.R_ai, L.R_aj, L.R_aa, L.nrows_coarse, L.Ac_ai, L.Ac_aj, Ac, spa);
                }
                ComputeSmootherNorms(team, L.Ac_ai, L.Ac_aj, Ac, L.nrows_coarse, l1, L.smoother_type);
              }
              {
                const PetscScalar *Ac_c = blk_amg_work + coarsest.off_Ac_aa;
                PetscScalar       *l1_c = blk_amg_work + coarsest.off_l1;
                ComputeSmootherNorms(team, coarsest.Ac_ai, coarsest.Ac_aj, Ac_c, coarsest.nrows, l1_c, coarsest.smoother_type);
              }
            });
          Kokkos::fence();
          Kokkos::Profiling::popRegion(); // BJKOKKOS_AMG_RAP (Kokkos-Krylov path)
          PetscCall(PetscLogEventEnd(BJKOKKOS_AMG_RAP, pc, 0, 0, 0));
        }
      }

#if !defined(PETSC_USE_COMPLEX)
      // Allocate per-block Hessenberg work buffer for GMRES_AMG
      // Layout per block: H[(maxit+1)*maxit] | cs[maxit] | sn[maxit] | g[maxit+1]
      PetscInt gmres_hwork_per_blk = (maxit + 1) * maxit + 2 * maxit + (maxit + 1);
      // Use the pre-allocated persistent GMRES Hessenberg buffer (allocated once in PCSetUp).
      auto        &d_gmres_hwork_k = *jac->d_gmres_hwork_k;
      PetscScalar *d_gmres_hwork   = d_gmres_hwork_k.data();
#endif

      // --- Phase: BJKOKKOS_Krylov_Solve ---
      PetscCall(PetscLogEventBegin(BJKOKKOS_Krylov_Solve, pc, 0, 0, 0));
      Kokkos::Profiling::pushRegion("BJKOKKOS_Krylov_Solve");
      Kokkos::parallel_for(
        "Solve", Kokkos::TeamPolicy<Kokkos::LaunchBounds<256, 4>>(nBlk, team_size, PCBJKOKKOS_VEC_SIZE).set_scratch_size(PCBJKOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes_team_shared)), KOKKOS_LAMBDA(const team_member team) {
          const int    blkID = team.league_rank(), start = d_bid_eqOffset[blkID], end = d_bid_eqOffset[blkID + 1];
          vect2D_scr_t work_vecs_shared(team.team_scratch(PCBJKOKKOS_SHARED_LEVEL), end - start, nShareVec);
          PetscScalar *work_buff_shared = work_vecs_shared.data();
          PetscScalar *work_buff_global = &d_work_vecs[start]; // start inc'ed in
          bool         print            = monitor && (blkID == view_bid);
          switch (ksp_type_idx) {
          case BATCH_KSP_BICG_JAC_IDX:
            static_cast<void>(BJSolve_BICG_Jac(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print));
            break;
          case BATCH_KSP_TFQMR_JAC_IDX:
            static_cast<void>(BJSolve_TFQMR_Jac(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print));
            break;
          case BATCH_KSP_BICG_AMG_IDX: {
            const int           gid = d_b2g[blkID];
            const AMGLevelInfo *lev = d_levels_flat + d_level_offs[gid];
            static_cast<void>(BJSolve_BICG_AMG(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_bdata, glb_xdata, print, d_fine_arr[gid], lev, d_nlevels_arr[gid], d_coarsest_arr[gid], d_amg_work_ptr + (PetscInt)blkID * amg_work_stride));
          } break;
          case BATCH_KSP_TFQMR_AMG_IDX: {
            const int           gid = d_b2g[blkID];
            const AMGLevelInfo *lev = d_levels_flat + d_level_offs[gid];
            static_cast<void>(BJSolve_TFQMR_AMG(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_bdata, glb_xdata, print, d_fine_arr[gid], lev, d_nlevels_arr[gid], d_coarsest_arr[gid], d_amg_work_ptr + (PetscInt)blkID * amg_work_stride));
          } break;
#if !defined(PETSC_USE_COMPLEX)
          case BATCH_KSP_GMRES_AMG_IDX: {
            const int           gid = d_b2g[blkID];
            const AMGLevelInfo *lev = d_levels_flat + d_level_offs[gid];
            static_cast<void>(BJSolve_GMRES_AMG(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_bdata, glb_xdata, print, d_fine_arr[gid], lev, d_nlevels_arr[gid], d_coarsest_arr[gid], d_amg_work_ptr + (PetscInt)blkID * amg_work_stride, d_gmres_hwork + (PetscInt)blkID * gmres_hwork_per_blk));
          } break;
          case BATCH_KSP_GMRES_JAC_IDX:
            static_cast<void>(BJSolve_GMRES_Jac(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print, d_gmres_hwork + (PetscInt)blkID * gmres_hwork_per_blk));
            break;
#endif
          default:
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
            printf("Unknown KSP type %d\n", ksp_type_idx);
#else
            /* void */;
#endif
          }
        });
      Kokkos::fence();
      Kokkos::Profiling::popRegion(); // BJKOKKOS_Krylov_Solve
      PetscCall(PetscLogEventEnd(BJKOKKOS_Krylov_Solve, pc, 0, 0, 0));
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePop();
      nvtxRangePushA("Post-solve-metadata");
#endif
      // --- Phase: BJKOKKOS_Post_solve ---
      PetscCall(PetscLogEventBegin(BJKOKKOS_Post_solve, pc, 0, 0, 0));
      Kokkos::Profiling::pushRegion("BJKOKKOS_Post_solve");
      auto h_metadata = Kokkos::create_mirror(Kokkos::HostSpace::memory_space(), d_metadata);
      Kokkos::deep_copy(h_metadata, d_metadata);
      PetscInt max_nnit = -1;
      PetscInt mbid     = 0;
      int      in[2], out[2];
      // Determine KSP/PC type names for diagnostics
      const char *ksp_name = (ksp_type_idx == BATCH_KSP_BICG_JAC_IDX || ksp_type_idx == BATCH_KSP_BICG_AMG_IDX) ? "bicg" : (ksp_type_idx == BATCH_KSP_GMRES_AMG_IDX || ksp_type_idx == BATCH_KSP_GMRES_JAC_IDX) ? "gmres" : "tfqmr";
      const char *pc_name  = (ksp_type_idx == BATCH_KSP_BICG_AMG_IDX || ksp_type_idx == BATCH_KSP_TFQMR_AMG_IDX || ksp_type_idx == BATCH_KSP_GMRES_AMG_IDX) ? "amg" : "jacobi";
      if (jac->reason) {                            // -pc_bjkokkos_ksp_converged_reason
        const PetscInt btarget = jac->batch_target; // -1 = all batches, >= 0 = specific batch
        // Print per-species iteration counts
        if (batch_sz != 1) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    %s+%s iterations per species:", ksp_name, pc_name));
        else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    %s+%s iterations:", ksp_name, pc_name));
        for (PetscInt dmIdx = 0, head = 0, s = 0; dmIdx < jac->num_dms; dmIdx += batch_sz) {
          for (PetscInt f = 0, idx = head; f < jac->dm_Nf[dmIdx]; f++, idx++, s++) {
            if (btarget == -1) {
              // Print all batch elements for this species
              PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), " s%" PetscInt_FMT ":[", s));
              for (int bid = 0; bid < batch_sz; bid++) {
                PetscInt its = h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its;
                PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%s%" PetscInt_FMT, bid ? "," : "", its));
                if (its > max_nnit) {
                  max_nnit = its;
                  mbid     = idx + bid * jac->dm_Nf[dmIdx];
                }
              }
              PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "]"));
            } else {
              // Print only the target batch element (or max across batches)
              PetscInt max_its = 0;
              for (int bid = 0; bid < batch_sz; bid++) {
                PetscInt its = h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its;
                if (its > max_its) max_its = its;
                if (its > max_nnit) {
                  max_nnit = its;
                  mbid     = idx + bid * jac->dm_Nf[dmIdx];
                }
              }
              PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%3" PetscInt_FMT " ", max_its));
            }
          }
          head += batch_sz * jac->dm_Nf[dmIdx];
        }
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\n"));
        jac->max_nits = max_nnit;
        in[0]         = max_nnit;
        in[1]         = rank;
        PetscCallMPI(MPIU_Allreduce(in, out, 1, MPI_2INT, MPI_MAXLOC, PetscObjectComm((PetscObject)A)));
        if (0 == rank) {
          if (batch_sz != 1)
            PetscCall(
              PetscPrintf(PETSC_COMM_SELF, "    [%d] %s+%s max iterations %d, species %" PetscInt_FMT ", batch %" PetscInt_FMT " (%s)\n", out[1], ksp_name, pc_name, out[0], mbid / batch_sz, mbid % batch_sz, KSPConvergedReasons[h_metadata[mbid].reason]));
          else PetscCall(PetscPrintf(PETSC_COMM_SELF, "    [%d] %s+%s max iterations %d, block %" PetscInt_FMT " (%s)\n", out[1], ksp_name, pc_name, out[0], mbid, KSPConvergedReasons[h_metadata[mbid].reason]));
        }
      }
      for (int blkID = 0; blkID < nBlk; blkID++) {
        PetscCall(PetscLogGpuFlops((PetscLogDouble)h_metadata[blkID].flops));
        PetscCheck(h_metadata[blkID].reason >= 0 || !jac->ksp->errorifnotconverged, PetscObjectComm((PetscObject)pc), PETSC_ERR_CONV_FAILED, "ERROR reason=%s, its=%" PetscInt_FMT ". species %" PetscInt_FMT ", batch %" PetscInt_FMT,
                   KSPConvergedReasons[h_metadata[blkID].reason], h_metadata[blkID].its, blkID / batch_sz, blkID % batch_sz);
      }
      {
        int errsum;
        Kokkos::parallel_reduce(
          nBlk,
          KOKKOS_LAMBDA(const int idx, int &lsum) {
            if (d_metadata[idx].reason < 0) ++lsum;
          },
          errsum);
        pcreason = errsum ? PC_SUBPC_ERROR : PC_NOERROR;
        if (!errsum && !jac->max_nits) { // set max its to give back to top KSP
          for (int blkID = 0; blkID < nBlk; blkID++) {
            if (h_metadata[blkID].its > jac->max_nits) jac->max_nits = h_metadata[blkID].its;
          }
        } else if (errsum) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] ERROR Kokkos batch solver did not converge in all solves\n", (int)rank));
        }
      }
      Kokkos::Profiling::popRegion(); // BJKOKKOS_Post_solve
      PetscCall(PetscLogEventEnd(BJKOKKOS_Post_solve, pc, 0, 0, 0));
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePop();
#endif
    } // end of Kokkos (not Kernels) solvers block
    PetscCall(VecRestoreArrayAndMemType(xout, &glb_xdata));
    PetscCall(VecRestoreArrayReadAndMemType(bvec, &glb_bdata));
    PetscCall(PCSetFailedReason(pc, pcreason));
    // map back to Plex space - not used
    if (plex_batch) {
      PetscCall(VecCopy(xout, bvec));
      PetscCall(VecScatterBegin(plex_batch, bvec, xout, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterEnd(plex_batch, bvec, xout, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecDestroy(&bvec));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  Mat            A = pc->pmat, Aseq = A; // use filtered block matrix, really "P"
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCheck(A, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "No matrix - A is used above");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "must use '-[dm_]mat_type aijkokkos -[dm_]vec_type kokkos' for -pc_type bjkokkos");
  if (!A->spptr) Aseq = ((Mat_MPIAIJ *)A->data)->A; // MPI
  PetscCall(MatSeqAIJKokkosSyncDevice(Aseq));
  {
    PetscInt    Istart, Iend;
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    if (!jac->vec_diag) {
      Vec     *subX = NULL;
      DM       pack, *subDM = NULL;
      PetscInt nDMs, n, *block_sizes = NULL;
      IS       isrow, isicol;
      { // Permute the matrix to get a block diagonal system: d_isrow_k, d_isicol_k
        MatOrderingType rtype;
        const PetscInt *rowindices, *icolindices;
        rtype = MATORDERINGRCM;
        // get permutation. And invert. should we convert to local indices?
        PetscCall(MatGetOrdering(Aseq, rtype, &isrow, &isicol)); // only seems to work for seq matrix
        PetscCall(ISDestroy(&isrow));
        PetscCall(ISInvertPermutation(isicol, PETSC_DECIDE, &isrow)); // THIS IS BACKWARD -- isrow is inverse
        // if (rank==1) PetscCall(ISView(isicol, PETSC_VIEWER_STDOUT_SELF));
        if (0) {
          Mat mat_block_order; // debug
          PetscCall(ISShift(isicol, Istart, isicol));
          PetscCall(MatCreateSubMatrix(A, isicol, isicol, MAT_INITIAL_MATRIX, &mat_block_order));
          PetscCall(ISShift(isicol, -Istart, isicol));
          PetscCall(MatViewFromOptions(mat_block_order, NULL, "-ksp_batch_reorder_view"));
          PetscCall(MatDestroy(&mat_block_order));
        }
        PetscCall(ISGetIndices(isrow, &rowindices)); // local idx
        PetscCall(ISGetIndices(isicol, &icolindices));
        const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_isrow_k((PetscInt *)rowindices, A->rmap->n);
        const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_isicol_k((PetscInt *)icolindices, A->rmap->n);
        jac->d_isrow_k  = new Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_isrow_k));
        jac->d_isicol_k = new Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_isicol_k));
        Kokkos::deep_copy(*jac->d_isrow_k, h_isrow_k);
        Kokkos::deep_copy(*jac->d_isicol_k, h_isicol_k);
        PetscCall(ISRestoreIndices(isrow, &rowindices));
        PetscCall(ISRestoreIndices(isicol, &icolindices));
        // if (rank==1) PetscCall(ISView(isicol, PETSC_VIEWER_STDOUT_SELF));
      }
      // get block sizes & allocate vec_diag
      PetscCall(PCGetDM(pc, &pack));
      if (pack) {
        PetscCall(PetscObjectTypeCompare((PetscObject)pack, DMCOMPOSITE, &flg));
        if (flg) {
          PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
          PetscCall(DMCreateGlobalVector(pack, &jac->vec_diag));
        } else pack = NULL; // flag for no DM
      }
      if (!jac->vec_diag) { // get 'nDMs' and sizes 'block_sizes' w/o DMComposite. TODO: User could provide ISs
        PetscInt        bsrt, bend, ncols, ntot = 0;
        const PetscInt *colsA, nloc = Iend - Istart;
        const PetscInt *rowindices, *icolindices;
        PetscCall(PetscMalloc1(nloc, &block_sizes)); // very inefficient, to big
        PetscCall(ISGetIndices(isrow, &rowindices));
        PetscCall(ISGetIndices(isicol, &icolindices));
        nDMs = 0;
        bsrt = 0;
        bend = 1;
        for (PetscInt row_B = 0; row_B < nloc; row_B++) { // for all rows in block diagonal space
          PetscInt rowA = icolindices[row_B], minj = PETSC_INT_MAX, maxj = 0;
          //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] rowA = %d\n",rank,rowA));
          PetscCall(MatGetRow(Aseq, rowA, &ncols, &colsA, NULL)); // not sorted in permutation
          PetscCheck(ncols, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Empty row not supported: %" PetscInt_FMT, row_B);
          for (PetscInt colj = 0; colj < ncols; colj++) {
            PetscInt colB = rowindices[colsA[colj]]; // use local idx
            //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t[%d] colB = %d\n",rank,colB));
            PetscCheck(colB >= 0 && colB < nloc, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "colB < 0: %" PetscInt_FMT, colB);
            if (colB > maxj) maxj = colB;
            if (colB < minj) minj = colB;
          }
          PetscCall(MatRestoreRow(Aseq, rowA, &ncols, &colsA, NULL));
          if (minj >= bend) { // first column is > max of last block -- new block or last block
            //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\t\t finish block %d, N loc = %d (%d,%d)\n", nDMs+1, bend - bsrt,bsrt,bend));
            block_sizes[nDMs] = bend - bsrt;
            ntot += block_sizes[nDMs];
            PetscCheck(minj == bend, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "minj != bend: %" PetscInt_FMT " != %" PetscInt_FMT, minj, bend);
            bsrt = bend;
            bend++; // start with size 1 in new block
            nDMs++;
          }
          if (maxj + 1 > bend) bend = maxj + 1;
          PetscCheck(minj >= bsrt || row_B == Iend - 1, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "%" PetscInt_FMT ") minj < bsrt: %" PetscInt_FMT " != %" PetscInt_FMT, rowA, minj, bsrt);
          //PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] %d) row %d.%d) cols %d : %d ; bsrt = %d, bend = %d\n",rank,row_B,nDMs,rowA,minj,maxj,bsrt,bend));
        }
        // do last block
        //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t\t [%d] finish block %d, N loc = %d (%d,%d)\n", rank, nDMs+1, bend - bsrt,bsrt,bend));
        block_sizes[nDMs] = bend - bsrt;
        ntot += block_sizes[nDMs];
        nDMs++;
        // cleanup
        PetscCheck(ntot == nloc, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "n total != n local: %" PetscInt_FMT " != %" PetscInt_FMT, ntot, nloc);
        PetscCall(ISRestoreIndices(isrow, &rowindices));
        PetscCall(ISRestoreIndices(isicol, &icolindices));
        PetscCall(PetscRealloc(sizeof(PetscInt) * nDMs, &block_sizes));
        PetscCall(MatCreateVecs(A, &jac->vec_diag, NULL));
        PetscCall(PetscInfo(pc, "Setup Matrix based meta data (not DMComposite not attached to PC) %" PetscInt_FMT " sub domains\n", nDMs));
      }
      PetscCall(ISDestroy(&isrow));
      PetscCall(ISDestroy(&isicol));
      jac->num_dms = nDMs;
      PetscCall(VecGetLocalSize(jac->vec_diag, &n));
      jac->n         = n;
      jac->d_idiag_k = new Kokkos::View<PetscScalar *, Kokkos::LayoutRight>("idiag", n);
      // options
      PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
      /* Check if user requested -pc_bjkokkos_pc_type amg.
         "amg" is not a registered PETSc PC type, so we intercept it here and
         clear it from the options database before KSPSetFromOptions runs. */
      PetscBool use_amg = PETSC_FALSE;
      {
        char      pc_type_str[64] = "";
        PetscBool pc_type_set     = PETSC_FALSE;
        PetscOptionsBegin(PetscObjectComm((PetscObject)jac->ksp), ((PetscObject)jac->ksp)->prefix, "BJKOKKOS batch PC type", "PC");
        PetscCall(PetscOptionsString("-pc_type", "Batch preconditioner type (jacobi or amg)", "PCSetType", "", pc_type_str, sizeof(pc_type_str), &pc_type_set));
        PetscOptionsEnd();
        if (pc_type_set == PETSC_TRUE) {
          if (!strcmp(pc_type_str, "amg")) use_amg = PETSC_TRUE;
          else PetscCheck(!strcmp(pc_type_str, "jacobi") || !strcmp(pc_type_str, ""), PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "BJKOKKOS supports only -pc_type jacobi or amg, not \"%s\"", pc_type_str);
        }
        if (use_amg == PETSC_TRUE) {
          // Clear the option so KSPSetFromOptions does not try to register "amg" as a PC type.
          const char *prefix = ((PetscObject)jac->ksp)->prefix;
          char        opt_name[256];
          PetscCall(PetscSNPrintf(opt_name, sizeof(opt_name), "-%spc_type", prefix ? prefix : ""));
          PetscCall(PetscOptionsClearValue(NULL, opt_name));
        }
      }
      PetscCall(KSPSetFromOptions(jac->ksp));
      // Set the inner PC type locally -- do not write to the global options database
      {
        PC innerpc;
        PetscCall(KSPGetPC(jac->ksp, &innerpc));
        PetscCall(PCSetType(innerpc, PCJACOBI));
      }
      PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPBICG, ""));
      if (flg) {
        jac->ksp_type_idx = use_amg ? BATCH_KSP_BICG_AMG_IDX : BATCH_KSP_BICG_JAC_IDX;
        jac->nwork        = 7;
      } else {
        PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPTFQMR, ""));
        if (flg) {
          jac->ksp_type_idx = use_amg ? BATCH_KSP_TFQMR_AMG_IDX : BATCH_KSP_TFQMR_JAC_IDX;
          jac->nwork        = 10;
        } else {
          PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPGMRES, ""));
          PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Unsupported batch ksp type");
#if defined(PETSC_USE_COMPLEX)
          SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Batch GMRES solver does not support complex scalars (Givens rotation is real-only)");
#else
          if (use_amg) {
            jac->ksp_type_idx = BATCH_KSP_GMRES_AMG_IDX;
            jac->nwork        = jac->ksp->max_it + 3; // V[0..maxit] + Z + XX
          } else {
            jac->ksp_type_idx = BATCH_KSP_GMRES_JAC_IDX;
            jac->nwork        = jac->ksp->max_it + 3; // V[0..maxit] + Z + XX
          }
#endif
        }
      }
      /* Apply looser default rtol for AMG variants (reduces inner iteration count
         while preserving outer SNES convergence and energy conservation).
         Only override if the user did not explicitly set -ksp_rtol via the
         prefixed option or the unprefixed global option. */
      if (use_amg == PETSC_TRUE) {
        PetscBool rtol_set = PETSC_FALSE;
        PetscCall(PetscOptionsHasName(NULL, ((PetscObject)jac->ksp)->prefix, "-ksp_rtol", &rtol_set));
        if (rtol_set == PETSC_FALSE) {
          // Also check the unprefixed global option in case the user set -ksp_rtol globally
          PetscCall(PetscOptionsHasName(NULL, NULL, "-ksp_rtol", &rtol_set));
        }
        if (rtol_set == PETSC_FALSE) {
          PetscCall(KSPSetTolerances(jac->ksp, 5e-3, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
          PetscCall(PetscInfo(pc, "BJKOKKOS AMG: overriding default rtol to 5e-3 (user did not set -ksp_rtol)\n"));
        }
      }
      PetscOptionsBegin(PetscObjectComm((PetscObject)jac->ksp), ((PetscObject)jac->ksp)->prefix, "Options for Kokkos batch solver", "none");
      PetscCall(PetscOptionsBool("-ksp_converged_reason", "", "bjkokkos.kokkos.cxx.c", jac->reason, &jac->reason, NULL));
      PetscCall(PetscOptionsBool("-ksp_monitor", "", "bjkokkos.kokkos.cxx.c", jac->monitor, &jac->monitor, NULL));
      PetscCall(PetscOptionsInt("-ksp_batch_target", "", "bjkokkos.kokkos.cxx.c", jac->batch_target, &jac->batch_target, NULL));
      PetscCall(PetscOptionsInt("-ksp_rank_target", "", "bjkokkos.kokkos.cxx.c", jac->rank_target, &jac->rank_target, NULL));
      PetscCall(PetscOptionsInt("-ksp_batch_nsolves_team", "", "bjkokkos.kokkos.cxx.c", jac->nsolves_team, &jac->nsolves_team, NULL));
      PetscCheck(jac->batch_target == -1 || jac->batch_target < jac->num_dms, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "-ksp_batch_target (%" PetscInt_FMT ") >= number of DMs (%" PetscInt_FMT "); use -1 for all batches", jac->batch_target, jac->num_dms);
      PetscOptionsEnd();
      // AMG tuning options -- defaults
      jac->amg_strong_threshold = 0.25;
      jac->amg_max_levels       = PCBJKOKKOS_MAX_AMG_LEVELS;
      jac->amg_min_coarse_size  = 10; // stop coarsening when block size <= this
      jac->amg_pre_sweeps       = 1;
      jac->amg_post_sweeps      = 1;
      jac->amg_coarse_sweeps    = 4;
      jac->amg_smoother_type    = BJKOKKOS_SMOOTH_L1_JACOBI;
      jac->amg_smoother_omega   = 1.0;
      // AMG tuning options -- read from options database via standard PETSc API
      PetscOptionsBegin(PetscObjectComm((PetscObject)pc), ((PetscObject)pc)->prefix, "Options for BJKOKKOS AMG preconditioner", "PC");
      PetscCall(PetscOptionsReal("-pc_bjkokkos_amg_strong_threshold", "Strength-of-connection threshold for AMG coarsening", "PCBJKOKKOSSetupAMG", jac->amg_strong_threshold, &jac->amg_strong_threshold, NULL));
      PetscCall(PetscOptionsInt("-pc_bjkokkos_amg_max_levels", "Maximum number of AMG levels", "PCBJKOKKOSSetupAMG", jac->amg_max_levels, &jac->amg_max_levels, NULL));
      PetscCall(PetscOptionsInt("-pc_bjkokkos_amg_min_coarse_size", "Minimum coarse grid size to stop AMG coarsening", "PCBJKOKKOSSetupAMG", jac->amg_min_coarse_size, &jac->amg_min_coarse_size, NULL));
      PetscCall(PetscOptionsInt("-pc_bjkokkos_amg_pre_sweeps", "Number of pre-smoothing sweeps per AMG level", "PCBJKOKKOSSetupAMG", jac->amg_pre_sweeps, &jac->amg_pre_sweeps, NULL));
      PetscCall(PetscOptionsInt("-pc_bjkokkos_amg_post_sweeps", "Number of post-smoothing sweeps per AMG level", "PCBJKOKKOSSetupAMG", jac->amg_post_sweeps, &jac->amg_post_sweeps, NULL));
      PetscCall(PetscOptionsInt("-pc_bjkokkos_amg_coarse_sweeps", "Number of smoother sweeps on the coarsest AMG level", "PCBJKOKKOSSetupAMG", jac->amg_coarse_sweeps, &jac->amg_coarse_sweeps, NULL));
      // PCMG-style aliases (override flat options if present)
      {
        char      jac_type_str[64] = "";
        PetscBool set;
        PetscInt  mg_levels_max_it = -1, mg_coarse_max_it = -1;
        PetscReal mg_omega = -1.0;

        // -pc_bjkokkos_mg_levels_ksp_max_it (overrides pre_sweeps AND post_sweeps)
        PetscCall(PetscOptionsInt("-pc_bjkokkos_mg_levels_ksp_max_it", "Sweeps per level (pre and post) -- PCMG-style alias", "PCBJKOKKOSSetupAMG", mg_levels_max_it, &mg_levels_max_it, &set));
        if (set) {
          jac->amg_pre_sweeps  = mg_levels_max_it;
          jac->amg_post_sweeps = mg_levels_max_it;
        }
        // -pc_bjkokkos_mg_coarse_ksp_max_it (overrides coarse_sweeps)
        PetscCall(PetscOptionsInt("-pc_bjkokkos_mg_coarse_ksp_max_it", "Coarse-level sweeps -- PCMG-style alias", "PCBJKOKKOSSetupAMG", mg_coarse_max_it, &mg_coarse_max_it, &set));
        if (set) jac->amg_coarse_sweeps = mg_coarse_max_it;
        // -pc_bjkokkos_mg_levels_pc_jacobi_type rowl1|diagonal
        PetscCall(PetscOptionsString("-pc_bjkokkos_mg_levels_pc_jacobi_type", "Smoother type: rowl1 or diagonal -- PCMG-style alias", "PCBJKOKKOSSetupAMG", "", jac_type_str, sizeof(jac_type_str), &set));
        if (set && !strcmp(jac_type_str, "rowl1")) jac->amg_smoother_type = BJKOKKOS_SMOOTH_L1_JACOBI;
        else if (set && (!strcmp(jac_type_str, "diagonal") || !strcmp(jac_type_str, "diag"))) jac->amg_smoother_type = BJKOKKOS_SMOOTH_JACOBI;
        // -pc_bjkokkos_mg_levels_pc_jacobi_rowl1_scale (omega/damping)
        PetscCall(PetscOptionsReal("-pc_bjkokkos_mg_levels_pc_jacobi_rowl1_scale", "Smoother damping factor omega -- PCMG-style alias", "PCBJKOKKOSSetupAMG", mg_omega, &mg_omega, &set));
        if (set) jac->amg_smoother_omega = mg_omega;
      }
      PetscOptionsEnd();
      // get blocks - jac->d_bid_eqOffset_k
      if (pack) {
        PetscCall(PetscMalloc(sizeof(*subX) * nDMs, &subX));
        PetscCall(PetscMalloc(sizeof(*subDM) * nDMs, &subDM));
      }
      PetscCall(PetscMalloc(sizeof(*jac->dm_Nf) * nDMs, &jac->dm_Nf));
      PetscCall(PetscInfo(pc, "Have %" PetscInt_FMT " blocks, n=%" PetscInt_FMT " rtol=%g type = %s\n", nDMs, n, (double)jac->ksp->rtol, ((PetscObject)jac->ksp)->type_name));
      if (pack) PetscCall(DMCompositeGetEntriesArray(pack, subDM));
      jac->nBlocks = 0;
      for (PetscInt ii = 0; ii < nDMs; ii++) {
        PetscInt Nf;
        if (subDM) {
          DM           dm = subDM[ii];
          PetscSection section;
          PetscCall(DMGetLocalSection(dm, &section));
          PetscCall(PetscSectionGetNumFields(section, &Nf));
        } else Nf = 1;
        jac->nBlocks += Nf;
#if PCBJKOKKOS_VERBOSE_LEVEL <= 2
        if (ii == 0) PetscCall(PetscInfo(pc, "%" PetscInt_FMT ") %" PetscInt_FMT " blocks (%" PetscInt_FMT " total)\n", ii, Nf, jac->nBlocks));
#else
        PetscCall(PetscInfo(pc, "%" PetscInt_FMT ") %" PetscInt_FMT " blocks (%" PetscInt_FMT " total)\n", ii, Nf, jac->nBlocks));
#endif
        jac->dm_Nf[ii] = Nf;
      }
      { // d_bid_eqOffset_k
        Kokkos::View<PetscInt *, Kokkos::LayoutRight, Kokkos::HostSpace> h_block_offsets("block_offsets", jac->nBlocks + 1);
        if (pack) PetscCall(DMCompositeGetAccessArray(pack, jac->vec_diag, nDMs, NULL, subX));
        h_block_offsets[0]    = 0;
        jac->const_block_size = -1;
        for (PetscInt ii = 0, idx = 0; ii < nDMs; ii++) {
          PetscInt nloc, nblk;
          if (pack) PetscCall(VecGetSize(subX[ii], &nloc));
          else nloc = block_sizes[ii];
          nblk = nloc / jac->dm_Nf[ii];
          PetscCheck(nloc % jac->dm_Nf[ii] == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "nloc%%jac->dm_Nf[ii] (%" PetscInt_FMT ") != 0 DMs", nloc % jac->dm_Nf[ii]);
          for (PetscInt jj = 0; jj < jac->dm_Nf[ii]; jj++, idx++) {
            h_block_offsets[idx + 1] = h_block_offsets[idx] + nblk;
#if PCBJKOKKOS_VERBOSE_LEVEL <= 2
            if (idx == 0) PetscCall(PetscInfo(pc, "Add first of %" PetscInt_FMT " blocks with %" PetscInt_FMT " equations\n", jac->nBlocks, nblk));
#else
            PetscCall(PetscInfo(pc, "\t%" PetscInt_FMT ") Add block with %" PetscInt_FMT " equations of %" PetscInt_FMT "\n", idx + 1, nblk, jac->nBlocks));
#endif
            if (jac->const_block_size == -1) jac->const_block_size = nblk;
            else if (jac->const_block_size > 0 && jac->const_block_size != nblk) jac->const_block_size = 0;
          }
        }
        if (pack) {
          PetscCall(DMCompositeRestoreAccessArray(pack, jac->vec_diag, jac->nBlocks, NULL, subX));
          PetscCall(PetscFree(subX));
          PetscCall(PetscFree(subDM));
        }
        jac->d_bid_eqOffset_k = new Kokkos::View<PetscInt *, Kokkos::LayoutRight>(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_block_offsets));
        Kokkos::deep_copy(*jac->d_bid_eqOffset_k, h_block_offsets);
      }
      if (!pack) PetscCall(PetscFree(block_sizes));
    }
    // If AMG type, build hierarchy and allocate per-block device work buffer.
    // Must be called AFTER d_bid_eqOffset_k and nBlocks are set (above).
    //
    // Optimization: skip the expensive symbolic AMG rebuild when only matrix
    // values changed (pc->flag == SAME_NONZERO_PATTERN).  The AMG hierarchy (P, R, Ac
    // sparsity) is fixed for the lifetime of the nonzero pattern -- only Ac values change
    // each Newton step, and those are recomputed by NumericRAP in PCApply (guarded by
    // jac->need_rap).  Rebuild only on first call or when the sparsity pattern changes.
    if (jac->ksp_type_idx == BATCH_KSP_BICG_AMG_IDX || jac->ksp_type_idx == BATCH_KSP_TFQMR_AMG_IDX || jac->ksp_type_idx == BATCH_KSP_GMRES_AMG_IDX) {
      if (pc->flag != SAME_NONZERO_PATTERN) {
        // First call or sparsity changed: rebuild full AMG hierarchy (symbolic + numeric).
        PetscCall(PCBJKOKKOSSetupAMG(pc, Aseq));
        // Compute amg_work_stride: fine-grid vectors first, then per-level slots, then coarsest.
        // Layout per block:
        //   [fine_aa | l1_fine | x_fine | b_fine | r_fine]  (fine_nnz + 4*nrows_fine)
        //   for lev in 0..nlevels-1: [Ac_aa(Ac_nnz) | l1 | x | b | r] (nrows_coarse each)
        //   coarsest: [l1 | x | b | r] (nrows_coarsest each); Ac_aa shared with last inter-level
        // Take the max over all unique grids for variable-size support.
        PetscInt max_stride = 0;
        for (PetscInt g = 0; g < jac->num_unique_grids; g++) {
          const AMGHierarchy *hier       = &jac->amg_hierarchy[g];
          PetscInt            nrows_fine = (hier->nlevels > 0) ? hier->levels[0].nrows_fine : hier->nrows_coarsest;
          PetscInt            stride     = 0;
          stride += hier->fine_nnz; // fine_aa values (per-block, updated each solve)
          stride += nrows_fine;     // l1_fine
          stride += nrows_fine;     // x_fine
          stride += nrows_fine;     // b_fine
          stride += nrows_fine;     // r_fine
          for (PetscInt lev = 0; lev < hier->nlevels; lev++) {
            const AMGLevel *L = &hier->levels[lev];
            stride += L->Ac_nnz;                              // Ac_aa values (also used as coarsest Ac_aa for last level)
            stride += L->nrows_coarse;                        // l1 norms
            stride += L->nrows_coarse;                        // x
            stride += L->nrows_coarse;                        // b
            stride += L->nrows_coarse;                        // r (scratch for smoother)
            stride += PCBJKOKKOS_TEAM_SIZE * L->nrows_coarse; // spa (sparse accumulator for NumericRAP); must match team_size used in PCApply (guarded by PetscCheck)
          }
          // coarsest solve vectors only (Ac_aa is shared with last inter-level slot above)
          stride += hier->nrows_coarsest; // l1
          stride += hier->nrows_coarsest; // x
          stride += hier->nrows_coarsest; // b
          stride += hier->nrows_coarsest; // r
          if (stride > max_stride) max_stride = stride;
        }
        jac->amg_work_stride = max_stride;
        // Only (re-)allocate if the size has changed -- avoids GPU malloc fragmentation when
        // PCReset+PCSetUp is called repeatedly by the adaptive TS (e.g. arkimex/1bee step rejection).
        const size_t amg_work_needed = (size_t)jac->nBlocks * (size_t)max_stride;
        if (!jac->d_amg_work || jac->d_amg_work->extent(0) != amg_work_needed) {
          if (jac->d_amg_work) delete jac->d_amg_work;
          jac->d_amg_work = new Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace>("amg_work", amg_work_needed);
        }
        PetscCall(PetscInfo(pc, "AMG: rebuilt hierarchy (DIFFERENT_NONZERO_PATTERN), amg_work_stride=%" PetscInt_FMT ", total device words=%" PetscInt_FMT "\n", max_stride, jac->nBlocks * max_stride));
        // Precompute per-block fine_aa_gidx[] to eliminate the O(nnz/row) linear
        // search in the PCApply RAP kernel.  For each block and each local CSR entry lk,
        // store the position in glb_Aaa[] where that entry lives.  Binary search is valid
        // because the global CSR column indices are sorted within each row.
        // Only grid 0 is used (all blocks share the same fine-grid sparsity pattern).
        {
          const PetscInt fine_nnz  = jac->amg_hierarchy[0].fine_nnz;
          jac->fine_aa_gidx_stride = fine_nnz;
          const size_t gidx_needed = (size_t)jac->nBlocks * (size_t)fine_nnz;
          if (!jac->d_fine_aa_gidx || jac->d_fine_aa_gidx->extent(0) != gidx_needed) {
            if (jac->d_fine_aa_gidx) delete jac->d_fine_aa_gidx;
            jac->d_fine_aa_gidx = new Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>("fine_aa_gidx", gidx_needed);
          }
          // Get global CSR row/col arrays from Aseq (values not needed here).
          const PetscInt *setup_Aai, *setup_Aaj;
          {
            PetscScalar *dummy_aa;
            PetscMemType dummy_mtype;
            PetscCall(MatSeqAIJGetCSRAndMemType(Aseq, &setup_Aai, &setup_Aaj, &dummy_aa, &dummy_mtype));
          }
          // team_size for this kernel (same logic as the Diag kernel below).
          const PetscInt setup_conc      = Kokkos::DefaultExecutionSpace().concurrency();
          const PetscInt setup_openmp    = !!(setup_conc < 1000);
          const PetscInt setup_team_size = (setup_openmp == 0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
          // Capture raw pointers for the kernel (no jac-> inside KOKKOS_LAMBDA).
          const PetscInt *d_isicol_p       = jac->d_isicol_k->data();
          const PetscInt *d_bid_eqOffset_p = jac->d_bid_eqOffset_k->data();
          PetscInt       *d_gidx_p         = jac->d_fine_aa_gidx->data();
          const PetscInt  fine_nnz_l       = fine_nnz;
          // fine_ai / fine_aj come from hierarchy grid 0 (same for all blocks).
          const PetscInt *fine_ai_p  = jac->amg_hierarchy[0].d_fine_ai->data();
          const PetscInt *fine_aj_p  = jac->amg_hierarchy[0].d_fine_aj->data();
          const PetscInt  nrows_fine = (jac->amg_hierarchy[0].nlevels > 0) ? jac->amg_hierarchy[0].levels[0].nrows_fine : jac->amg_hierarchy[0].nrows_coarsest;
          Kokkos::parallel_for(
            "fine_aa_gidx", Kokkos::TeamPolicy<>(jac->nBlocks, setup_team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
              const int      blkID    = team.league_rank();
              const PetscInt start    = d_bid_eqOffset_p[blkID];
              PetscInt      *blk_gidx = d_gidx_p + (PetscInt)blkID * fine_nnz_l;
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nrows_fine), [=](const int rowl) {
                const PetscInt grow = d_isicol_p[start + rowl];
                for (PetscInt lk = fine_ai_p[rowl]; lk < fine_ai_p[rowl + 1]; lk++) {
                  const PetscInt gcol = d_isicol_p[start + fine_aj_p[lk]];
                  // Binary search: global CSR column indices are sorted within each row.
                  PetscInt lo = setup_Aai[grow], hi = setup_Aai[grow + 1] - 1, mid_idx = -1;
                  while (lo <= hi) {
                    PetscInt m = (lo + hi) / 2;
                    if (setup_Aaj[m] == gcol) {
                      mid_idx = m;
                      break;
                    } else if (setup_Aaj[m] < gcol) lo = m + 1;
                    else hi = m - 1;
                  }
                  if (mid_idx < 0) Kokkos::abort("PCBJKOKKOS: fine_aa_gidx binary search miss -- gcol not found in global CSR row");
                  blk_gidx[lk] = mid_idx;
                }
              });
            });
          Kokkos::fence();
          PetscCall(PetscInfo(pc, "AMG: computed fine_aa_gidx, fine_nnz=%" PetscInt_FMT ", total entries=%" PetscInt_FMT "\n", fine_nnz, (PetscInt)gidx_needed));
        }
        // Cache AMG device views: build host arrays for AMGFineInfo/AMGLevelInfo/AMGCoarsestInfo
        // and deep-copy to device once here in PCSetUp.  These encode the AMG hierarchy topology
        // which is constant between PCSetUp calls -- avoids re-creating and deep-copying on every
        // PCApply call.
        {
          const PetscInt num_grids    = jac->num_unique_grids;
          PetscInt       total_levels = 0;
          for (PetscInt g = 0; g < num_grids; g++) total_levels += jac->amg_hierarchy[g].nlevels;

          Kokkos::View<AMGFineInfo *, Kokkos::HostSpace>     h_amg_fine_arr("h_amg_fine_arr", num_grids);
          Kokkos::View<AMGLevelInfo *, Kokkos::HostSpace>    h_amg_levels_flat("h_amg_levels_flat", total_levels);
          Kokkos::View<AMGCoarsestInfo *, Kokkos::HostSpace> h_amg_coarsest_arr("h_amg_coarsest_arr", num_grids);
          Kokkos::View<PetscInt *, Kokkos::HostSpace>        h_amg_nlevels("h_amg_nlevels", num_grids);
          Kokkos::View<PetscInt *, Kokkos::HostSpace>        h_amg_level_offsets("h_amg_level_offsets", num_grids + 1);

          PetscInt flat_idx = 0;
          for (PetscInt g = 0; g < num_grids; g++) {
            const AMGHierarchy *hier       = &jac->amg_hierarchy[g];
            PetscInt            gnlevels   = hier->nlevels;
            PetscInt            nrows_fine = (gnlevels > 0) ? hier->levels[0].nrows_fine : hier->nrows_coarsest;
            h_amg_nlevels[g]               = gnlevels;
            h_amg_level_offsets[g]         = flat_idx;

            AMGFineInfo &FI  = h_amg_fine_arr[g];
            PetscInt     off = 0;
            FI.nrows         = nrows_fine;
            FI.fine_nnz      = hier->fine_nnz;
            FI.fine_ai       = hier->d_fine_ai->data();
            FI.fine_aj       = hier->d_fine_aj->data();
            FI.off_fine_aa   = off;
            off += hier->fine_nnz;
            FI.off_l1 = off;
            off += nrows_fine;
            FI.off_x = off;
            off += nrows_fine;
            FI.off_b = off;
            off += nrows_fine;
            FI.off_r = off;
            off += nrows_fine;
            FI.pre_sweeps    = hier->pre_sweeps;
            FI.post_sweeps   = hier->post_sweeps;
            FI.omega         = hier->smoother_omega;
            FI.smoother_type = hier->smoother_type;

            for (PetscInt lev = 0; lev < gnlevels; lev++) {
              const AMGLevel *L  = &hier->levels[lev];
              AMGLevelInfo   &LI = h_amg_levels_flat[flat_idx + lev];
              LI.P_ai            = L->d_P_ai->data();
              LI.P_aj            = L->d_P_aj->data();
              LI.P_aa            = L->d_P_aa->data();
              LI.R_ai            = L->d_R_ai->data();
              LI.R_aj            = L->d_R_aj->data();
              LI.R_aa            = L->d_R_aa->data();
              LI.Ac_ai           = L->d_Ac_ai->data();
              LI.Ac_aj           = L->d_Ac_aj->data();
              LI.nrows_fine      = L->nrows_fine;
              LI.nrows_coarse    = L->nrows_coarse;
              LI.Ac_nnz          = L->Ac_nnz;
              LI.off_Ac_aa       = off;
              off += L->Ac_nnz;
              LI.off_l1 = off;
              off += L->nrows_coarse;
              LI.off_x = off;
              off += L->nrows_coarse;
              LI.off_b = off;
              off += L->nrows_coarse;
              LI.off_r = off;
              off += L->nrows_coarse;
              LI.off_spa = off;
              off += PCBJKOKKOS_TEAM_SIZE * L->nrows_coarse;
              LI.pre_sweeps    = hier->pre_sweeps;
              LI.post_sweeps   = hier->post_sweeps;
              LI.omega         = hier->smoother_omega;
              LI.smoother_type = hier->smoother_type;
            }

            AMGCoarsestInfo &CI = h_amg_coarsest_arr[g];
            CI.Ac_ai            = hier->d_Ac_coarsest_ai->data();
            CI.Ac_aj            = hier->d_Ac_coarsest_aj->data();
            CI.nrows            = hier->nrows_coarsest;
            CI.Ac_nnz           = hier->Ac_coarsest_nnz;
            CI.off_Ac_aa        = (gnlevels > 0) ? h_amg_levels_flat[flat_idx + gnlevels - 1].off_Ac_aa : off;
            CI.off_l1           = off;
            off += hier->nrows_coarsest;
            CI.off_x = off;
            off += hier->nrows_coarsest;
            CI.off_b = off;
            off += hier->nrows_coarsest;
            CI.off_r = off;
            off += hier->nrows_coarsest;
            CI.coarse_sweeps = hier->coarse_sweeps;
            CI.omega         = hier->smoother_omega;
            CI.smoother_type = hier->smoother_type;
            flat_idx += gnlevels;
          }
          h_amg_level_offsets[num_grids] = flat_idx;

          // Allocate device views and deep-copy
          if (jac->d_amg_fine_arr_k) delete jac->d_amg_fine_arr_k;
          if (jac->d_amg_levels_flat_k) delete jac->d_amg_levels_flat_k;
          if (jac->d_amg_coarsest_arr_k) delete jac->d_amg_coarsest_arr_k;
          if (jac->d_amg_nlevels_k) delete jac->d_amg_nlevels_k;
          if (jac->d_amg_level_offsets_k) delete jac->d_amg_level_offsets_k;
          if (jac->d_block_to_grid_k) delete jac->d_block_to_grid_k;
          jac->d_amg_fine_arr_k      = new Kokkos::View<AMGFineInfo *, Kokkos::DefaultExecutionSpace>("amg_fine_arr", num_grids);
          jac->d_amg_levels_flat_k   = new Kokkos::View<AMGLevelInfo *, Kokkos::DefaultExecutionSpace>("amg_levels_flat", total_levels);
          jac->d_amg_coarsest_arr_k  = new Kokkos::View<AMGCoarsestInfo *, Kokkos::DefaultExecutionSpace>("amg_coarsest_arr", num_grids);
          jac->d_amg_nlevels_k       = new Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>("amg_nlevels", num_grids);
          jac->d_amg_level_offsets_k = new Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>("amg_level_offsets", num_grids + 1);
          jac->d_block_to_grid_k     = new Kokkos::View<PetscInt *, Kokkos::DefaultExecutionSpace>("block_to_grid", jac->nBlocks);
          Kokkos::deep_copy(*jac->d_amg_fine_arr_k, h_amg_fine_arr);
          Kokkos::deep_copy(*jac->d_amg_levels_flat_k, h_amg_levels_flat);
          Kokkos::deep_copy(*jac->d_amg_coarsest_arr_k, h_amg_coarsest_arr);
          Kokkos::deep_copy(*jac->d_amg_nlevels_k, h_amg_nlevels);
          Kokkos::deep_copy(*jac->d_amg_level_offsets_k, h_amg_level_offsets);
          {
            Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_b2g(jac->block_to_grid, jac->nBlocks);
            Kokkos::deep_copy(*jac->d_block_to_grid_k, h_b2g);
          }
          PetscCall(PetscInfo(pc, "AMG: cached %" PetscInt_FMT " device views (%" PetscInt_FMT " grids, %" PetscInt_FMT " total levels)\n", (PetscInt)6, num_grids, total_levels));
        }
      } else {
        PetscCall(PetscInfo(pc, "AMG: skipping hierarchy rebuild (SAME_NONZERO_PATTERN) -- NumericRAP will run in PCApply\n"));
      }
      // Note: no deep_copy zeroing needed -- every region of d_amg_work is overwritten
      // before it is read.  The RAP kernel overwrites fine_aa, Ac_aa, and l1 (via
      // NumericRAP which zeros Ac_aa internally, and ComputeSmootherNorms which assigns
      // norms[row]).  The V-cycle zeros x and overwrites b and r at each level.
    }
  }
  // Update inverse diagonal from current matrix values -- must run every time PCSetUp is called
  // (i.e., whenever the matrix changes), not just on first setup.  The diagonal is used by the
  // Jacobi smoother in the non-AMG path and as a fallback; keeping it stale across Newton steps
  // would silently degrade convergence.
  {
    const PetscInt    *d_ai, *d_aj;
    const PetscScalar *d_aa;
    const PetscInt     conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp == 0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
    const PetscInt    *d_bid_eqOffset = jac->d_bid_eqOffset_k->data(), *r = jac->d_isrow_k->data(), *ic = jac->d_isicol_k->data();
    PetscScalar       *d_idiag = jac->d_idiag_k->data(), *dummy;
    PetscMemType       mtype;
    PetscCall(MatSeqAIJGetCSRAndMemType(Aseq, &d_ai, &d_aj, &dummy, &mtype));
    PetscCheck(PetscMemTypeDevice(mtype) || Kokkos::DefaultExecutionSpace().concurrency() < 1000, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "MatSeqAIJGetCSRAndMemType returned host memory but Kokkos execution space is a device; PCBJKOKKOS requires a Kokkos-aware (device) matrix");
    d_aa = dummy;
    Kokkos::parallel_for(
      "Diag", Kokkos::TeamPolicy<>(jac->nBlocks, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
        const PetscInt blkID = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, d_bid_eqOffset[blkID], d_bid_eqOffset[blkID + 1]), [=](const int rowb) {
          const PetscInt     rowa = ic[rowb], ai = d_ai[rowa], *aj = d_aj + ai; // grab original data
          const PetscScalar *aa   = d_aa + ai;
          const PetscInt     nrow = d_ai[rowa + 1] - ai;
          int                found;
          Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, nrow),
            [=](const int &j, int &count) {
              const PetscInt colb = r[aj[j]];
              if (colb == rowb) {
                d_idiag[rowb] = 1. / aa[j];
                count++;
              }
            },
            found);
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
          if (found != 1) Kokkos::single(Kokkos::PerThread(team), [=]() { printf("ERRORrow %d) found = %d\n", rowb, found); });
#endif
        });
      });
  }
  // Pre-allocate global Krylov work vectors and GMRES Hessenberg buffer once here in PCSetUp
  // so that PCApply does not repeatedly malloc/free GPU memory (which fragments the pool and
  // eventually causes OOM on long runs).  Use size guards so that repeated PCReset+PCSetUp
  // calls (e.g. from arkimex/1bee step rejection) do not free and re-allocate these buffers.
  {
    // Compute global_buff_words the same way PCApply does (GMRES path: all in global memory).
    const PetscInt global_buff_words = jac->n * jac->nwork;
    if (!jac->d_work_vecs_k || jac->d_work_vecs_k->extent(0) != (size_t)global_buff_words) {
      if (jac->d_work_vecs_k) delete jac->d_work_vecs_k;
      jac->d_work_vecs_k = new Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace>("workvectors", global_buff_words);
    }
    // GMRES Hessenberg: (maxit+1)*maxit + 2*maxit + (maxit+1) words per block.
    const PetscInt gmres_hwork_per_blk = (jac->ksp->max_it + 1) * jac->ksp->max_it + 2 * jac->ksp->max_it + (jac->ksp->max_it + 1);
    const PetscInt gmres_hwork_total   = (jac->ksp_type_idx == BATCH_KSP_GMRES_AMG_IDX || jac->ksp_type_idx == BATCH_KSP_GMRES_JAC_IDX) ? jac->nBlocks * gmres_hwork_per_blk : 0;
    if (!jac->d_gmres_hwork_k || jac->d_gmres_hwork_k->extent(0) != (size_t)gmres_hwork_total) {
      if (jac->d_gmres_hwork_k) delete jac->d_gmres_hwork_k;
      jac->d_gmres_hwork_k = new Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace>("gmres_hwork", gmres_hwork_total);
    }
    PetscCall(PetscInfo(pc, "PCSetUp_BJKOKKOS: pre-allocated workvectors (%" PetscInt_FMT " words) and gmres_hwork (%" PetscInt_FMT " words)\n", global_buff_words, gmres_hwork_total));
  }
  // Register PCApply_BJKOKKOS sub-phase events once.
  {
    static PetscBool events_registered = PETSC_FALSE;
    if (!events_registered) {
      PetscCall(PetscLogEventRegister("BJKOKKOS_AMG_RAP", PC_CLASSID, &BJKOKKOS_AMG_RAP));
      PetscCall(PetscLogEventRegister("BJKOKKOS_Krylov_Solve", PC_CLASSID, &BJKOKKOS_Krylov_Solve));
      PetscCall(PetscLogEventRegister("BJKOKKOS_Post_solve", PC_CLASSID, &BJKOKKOS_Post_solve));
      events_registered = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCReset_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&jac->ksp));
  PetscCall(VecDestroy(&jac->vec_diag));
  if (jac->d_bid_eqOffset_k) delete jac->d_bid_eqOffset_k;
  if (jac->d_idiag_k) delete jac->d_idiag_k;
  if (jac->d_isrow_k) delete jac->d_isrow_k;
  if (jac->d_isicol_k) delete jac->d_isicol_k;
  jac->d_bid_eqOffset_k = NULL;
  jac->d_idiag_k        = NULL;
  jac->d_isrow_k        = NULL;
  jac->d_isicol_k       = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSGetKSP_C", NULL)); // not published now (causes configure errors)
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSSetKSP_C", NULL));
  PetscCall(PetscFree(jac->dm_Nf));
  jac->dm_Nf = NULL;
  if (jac->rowOffsets) delete jac->rowOffsets;
  if (jac->colIndices) delete jac->colIndices;
  if (jac->batch_b) delete jac->batch_b;
  if (jac->batch_x) delete jac->batch_x;
  if (jac->batch_values) delete jac->batch_values;
  jac->rowOffsets   = NULL;
  jac->colIndices   = NULL;
  jac->batch_b      = NULL;
  jac->batch_x      = NULL;
  jac->batch_values = NULL;
  // Destroy AMG hierarchy.
  // NOTE: d_amg_work, d_work_vecs_k, and d_gmres_hwork_k are intentionally NOT freed here.
  // PCReset is called by the adaptive TS (e.g. arkimex/1bee) on every rejected step, and
  // freeing+reallocating these large GPU buffers fragments the Kokkos memory pool causing OOM.
  // They are freed only in PCDestroy_BJKOKKOS (final teardown).
  PetscCall(PCBJKOKKOSDestroyAMG(jac));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  // Free the large GPU buffers that survive PCReset (to avoid fragmentation on step rejection).
  // These must be freed before Kokkos finalizes, so do it here in PCDestroy.
  if (jac->d_amg_work) {
    delete jac->d_amg_work;
    jac->d_amg_work = NULL;
  }
  if (jac->d_work_vecs_k) {
    delete jac->d_work_vecs_k;
    jac->d_work_vecs_k = NULL;
  }
  if (jac->d_gmres_hwork_k) {
    delete jac->d_gmres_hwork_k;
    jac->d_gmres_hwork_k = NULL;
  }
  // Free precomputed fine_aa_gidx index array.
  if (jac->d_fine_aa_gidx) {
    delete jac->d_fine_aa_gidx;
    jac->d_fine_aa_gidx = NULL;
  }
  // Free cached AMG device views (built once in PCSetUp, reused across PCApply calls).
  if (jac->d_amg_fine_arr_k) {
    delete jac->d_amg_fine_arr_k;
    jac->d_amg_fine_arr_k = NULL;
  }
  if (jac->d_amg_levels_flat_k) {
    delete jac->d_amg_levels_flat_k;
    jac->d_amg_levels_flat_k = NULL;
  }
  if (jac->d_amg_coarsest_arr_k) {
    delete jac->d_amg_coarsest_arr_k;
    jac->d_amg_coarsest_arr_k = NULL;
  }
  if (jac->d_amg_nlevels_k) {
    delete jac->d_amg_nlevels_k;
    jac->d_amg_nlevels_k = NULL;
  }
  if (jac->d_amg_level_offsets_k) {
    delete jac->d_amg_level_offsets_k;
    jac->d_amg_level_offsets_k = NULL;
  }
  if (jac->d_block_to_grid_k) {
    delete jac->d_block_to_grid_k;
    jac->d_block_to_grid_k = NULL;
  }
  jac->amg_work_stride     = 0;
  jac->fine_aa_gidx_stride = 0;
  PetscCall(PCReset_BJKOKKOS(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_BJKOKKOS(PC pc, PetscViewer viewer)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscBool is_amg = (PetscBool)(jac->ksp_type_idx == BATCH_KSP_BICG_AMG_IDX || jac->ksp_type_idx == BATCH_KSP_TFQMR_AMG_IDX || jac->ksp_type_idx == BATCH_KSP_GMRES_AMG_IDX);
    if (is_amg) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Batched device linear solver: %s with AMG preconditioning\n", ((PetscObject)jac->ksp)->type_name));
      if (jac->amg_hierarchy) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "    AMG hierarchy: %" PetscInt_FMT " unique grid(s)\n", jac->num_unique_grids));
        for (PetscInt g = 0; g < jac->num_unique_grids; g++) {
          AMGHierarchy *hier = &jac->amg_hierarchy[g];
          PetscCall(PetscViewerASCIIPrintf(viewer, "      Grid %" PetscInt_FMT " (%" PetscInt_FMT " fine DOFs): %" PetscInt_FMT " levels\n", g, hier->grid_size, hier->nlevels));
          PetscCall(PetscViewerASCIIPrintf(viewer, "        Level 0: %" PetscInt_FMT " DOFs, %" PetscInt_FMT " nnz (avg %.1f nnz/row)\n", hier->grid_size, hier->fine_nnz, hier->grid_size > 0 ? (double)hier->fine_nnz / (double)hier->grid_size : 0.0));
          for (PetscInt lev = 0; lev < hier->nlevels - 1; lev++) {
            AMGLevel *lvl   = &hier->levels[lev];
            PetscInt  P_nnz = lvl->P_ai[lvl->nrows_fine];
            PetscCall(PetscViewerASCIIPrintf(viewer, "        Level %" PetscInt_FMT ": %" PetscInt_FMT " DOFs, %" PetscInt_FMT " nnz (avg %.1f nnz/row)  (P: %" PetscInt_FMT "x%" PetscInt_FMT ", %" PetscInt_FMT " nnz)\n", lev + 1, lvl->nrows_coarse,
                                             lvl->Ac_nnz, lvl->nrows_coarse > 0 ? (double)lvl->Ac_nnz / (double)lvl->nrows_coarse : 0.0, lvl->nrows_fine, lvl->nrows_coarse, P_nnz));
          }
          if (hier->nlevels > 1 && hier->Ac_coarsest_nnz > 0)
            PetscCall(
              PetscViewerASCIIPrintf(viewer, "        Coarsest: %" PetscInt_FMT " DOFs, %" PetscInt_FMT " nnz (avg %.1f nnz/row)\n", hier->nrows_coarsest, hier->Ac_coarsest_nnz, hier->nrows_coarsest > 0 ? (double)hier->Ac_coarsest_nnz / (double)hier->nrows_coarsest : 0.0));
          {
            const char *stype = (hier->smoother_type == BJKOKKOS_SMOOTH_L1_JACOBI) ? "L1-Jacobi (rowl1)" : "Jacobi (diagonal)";
            PetscCall(
              PetscViewerASCIIPrintf(viewer, "    Smoother: %s, omega=%.2f, pre=%" PetscInt_FMT ", post=%" PetscInt_FMT ", coarse=%" PetscInt_FMT "\n", stype, (double)PetscRealPart(hier->smoother_omega), hier->pre_sweeps, hier->post_sweeps, hier->coarse_sweeps));
          }
          PetscCall(PetscViewerASCIIPrintf(viewer, "    Strong threshold: %g\n", (double)hier->strong_threshold));
        }
      }
    } else PetscCall(PetscViewerASCIIPrintf(viewer, "  Batched device linear solver: %s with Jacobi preconditioning\n", ((PetscObject)jac->ksp)->type_name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\t\tnwork = %" PetscInt_FMT ", rel tol = %e, abs tol = %e, div tol = %e, max it =%" PetscInt_FMT ", type = %s\n", jac->nwork, jac->ksp->rtol, jac->ksp->abstol, jac->ksp->divtol, jac->ksp->max_it,
                                     ((PetscObject)jac->ksp)->type_name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_BJKOKKOS(PC pc, PetscOptionItems PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PC BJKOKKOS options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCBJKOKKOSSetKSP_BJKOKKOS(PC pc, KSP ksp)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)ksp));
  PetscCall(KSPDestroy(&jac->ksp));
  jac->ksp = ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCBJKOKKOSSetKSP - Sets the `KSP` context for `PCBJKOKKOS`

  Collective

  Input Parameters:
+ pc  - the `PCBJKOKKOS` preconditioner context
- ksp - the `KSP` solver

  Level: advanced

  Notes:
  The `PC` and the `KSP` must have the same communicator

  If the `PC` is not `PCBJKOKKOS` this function returns without doing anything

.seealso: [](ch_ksp), `PCBJKOKKOSGetKSP()`, `PCBJKOKKOS`
@*/
PetscErrorCode PCBJKOKKOSSetKSP(PC pc, KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 2);
  PetscCheckSameComm(pc, 1, ksp, 2);
  PetscTryMethod(pc, "PCBJKOKKOSSetKSP_C", (PC, KSP), (pc, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCBJKOKKOSGetKSP_BJKOKKOS(PC pc, KSP *ksp)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
  *ksp = jac->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCBJKOKKOSGetKSP - Gets the `KSP` context for the `PCBJKOKKOS` preconditioner

  Not Collective but `KSP` returned is parallel if `PC` was parallel

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. ksp - the `KSP` solver

  Level: advanced

  Notes:
  You must call `KSPSetUp()` before calling `PCBJKOKKOSGetKSP()`.

  If the `PC` is not a `PCBJKOKKOS` object it raises an error

.seealso: [](ch_ksp), `PCBJKOKKOS`, `PCBJKOKKOSSetKSP()`
@*/
PetscErrorCode PCBJKOKKOSGetKSP(PC pc, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(ksp, 2);
  PetscUseMethod(pc, "PCBJKOKKOSGetKSP_C", (PC, KSP *), (pc, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPostSolve_BJKOKKOS(PC pc, KSP ksp, Vec b, Vec x)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ksp->its = jac->max_nits;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolve_BJKOKKOS(PC pc, KSP ksp, Vec b, Vec x)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  jac->ksp->errorifnotconverged = ksp->errorifnotconverged;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCBJKOKKOS - A batched Krylov/block Jacobi solver that runs a solve of each diagaonl block of a block diagonal `MATSEQAIJ` in a Kokkos thread group

   Options Database Key:
.  -pc_bjkokkos_ - options prefix for its `KSP` options

   Level: intermediate

   Note:
   For use with `-ksp_type preonly` to bypass any computation on the CPU

   Developer Notes:
   The entire Krylov (TFQMR or BICG) with diagonal preconditioning for each block of a block diagnaol matrix runs in a Kokkos thread group (eg, one block per SM on NVIDIA). It supports taking a non-block diagonal matrix but this is not tested. One should create an explicit block diagonal matrix and use that as the matrix for constructing the preconditioner in the outer `KSP` solver. Variable block size are supported and tested in src/ts/utils/dmplexlandau/tutorials/ex[1|2].c

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCBJACOBI`,
          `PCSHELL`, `PCCOMPOSITE`, `PCSetUseAmat()`, `PCBJKOKKOSGetKSP()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac;

  PetscFunctionBegin;
  PetscCall(PetscNew(&jac));
  pc->data = (void *)jac;

  jac->ksp              = NULL;
  jac->vec_diag         = NULL;
  jac->d_bid_eqOffset_k = NULL;
  jac->d_idiag_k        = NULL;
  jac->d_isrow_k        = NULL;
  jac->d_isicol_k       = NULL;
  jac->nBlocks          = 1;
  jac->max_nits         = 0;

  PetscCall(PetscMemzero(pc->ops, sizeof(struct _PCOps)));
  pc->ops->apply          = PCApply_BJKOKKOS;
  pc->ops->applytranspose = NULL;
  pc->ops->setup          = PCSetUp_BJKOKKOS;
  pc->ops->reset          = PCReset_BJKOKKOS;
  pc->ops->destroy        = PCDestroy_BJKOKKOS;
  pc->ops->setfromoptions = PCSetFromOptions_BJKOKKOS;
  pc->ops->view           = PCView_BJKOKKOS;
  pc->ops->postsolve      = PCPostSolve_BJKOKKOS;
  pc->ops->presolve       = PCPreSolve_BJKOKKOS;

  jac->rowOffsets            = NULL;
  jac->colIndices            = NULL;
  jac->batch_b               = NULL;
  jac->batch_x               = NULL;
  jac->batch_values          = NULL;
  jac->d_amg_fine_arr_k      = NULL;
  jac->d_amg_levels_flat_k   = NULL;
  jac->d_amg_coarsest_arr_k  = NULL;
  jac->d_amg_nlevels_k       = NULL;
  jac->d_amg_level_offsets_k = NULL;
  jac->d_block_to_grid_k     = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSGetKSP_C", PCBJKOKKOSGetKSP_BJKOKKOS));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSSetKSP_C", PCBJKOKKOSSetKSP_BJKOKKOS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
