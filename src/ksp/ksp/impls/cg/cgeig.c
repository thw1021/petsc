/*
   Code for calculating extreme eigenvalues via the Lanczos method
   running with CG. Note this only works for symmetric real and Hermitian
   matrices (not complex matrices that are symmetric).
*/
#include <../src/ksp/ksp/impls/cg/cgimpl.h>
#include <../include/petscblaslapack.h>

PetscErrorCode KSPComputeEigenvalues_CG(KSP ksp, PetscInt nmax, PetscReal *r, PetscReal *c, PetscInt *neig)
{
  KSP_CG      *cgP = (KSP_CG *)ksp->data;
  PetscScalar *d, *e;
  PetscReal   *ee;
  PetscInt     n = ksp->its;
  PetscBLASInt bn, lierr = 0, ldz = 1;

  PetscFunctionBegin;
  PetscCheck(nmax >= n, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_SIZ, "Not enough room in work space r and c for eigenvalues");
  *neig = n;

  PetscCall(PetscArrayzero(c, nmax));
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  d  = cgP->d;
  e  = cgP->e;
  ee = cgP->ee;

  /* Copy the tridiagonal matrix to the work space, truncating at the first
     non-finite Lanczos coefficient (see KSPComputeExtremeSingularValues_CG()
     for why Inf/NaN entries can appear) so LAPACKstev does not fail with the
     "xSTEV error". The leading finite block is a valid result from the
     converged Krylov subspace. */
  PetscInt nv = 0;
  for (PetscInt j = 0; j < n; j++) {
    PetscReal dj = PetscRealPart(d[j]);
    PetscReal ej = PetscRealPart(e[j + 1]);
    if (PetscIsInfOrNanReal(dj) || PetscIsInfOrNanReal(ej)) break;
    r[j]  = dj;
    ee[j] = ej;
    nv++;
  }
  *neig = nv;
  if (!nv) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscBLASIntCast(nv, &bn));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKREALstev", LAPACKREALstev_("N", &bn, r, ee, NULL, &ldz, NULL, &lierr));
  PetscCall(PetscFPTrapPop());
  /* lierr > 0: algorithm did not converge in 30*N iterations; the eigenvalues
     are only estimates used for Chebyshev smoothing, so skip rather than crash */
  if (lierr > 0) {
    PetscCall(PetscInfo(ksp, "xSTEV: %" PetscBLASInt_FMT " off-diagonal elements did not converge; returning zero eigenvalues\n", lierr));
    *neig = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(!lierr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "xSTEV error (invalid argument %" PetscBLASInt_FMT ")", lierr);
  PetscCall(PetscSortReal(nv, r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP ksp, PetscReal *emax, PetscReal *emin)
{
  KSP_CG      *cgP = (KSP_CG *)ksp->data;
  PetscScalar *d, *e;
  PetscReal   *dd, *ee;
  PetscInt     n = ksp->its;
  PetscBLASInt bn, lierr = 0, ldz = 1;

  PetscFunctionBegin;
  if (!n) {
    *emax = *emin = 1.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  d  = cgP->d;
  e  = cgP->e;
  dd = cgP->dd;
  ee = cgP->ee;

  /* Copy the tridiagonal matrix to the work space, truncating at the first
     non-finite Lanczos coefficient. With KSP_NORM_NONE and a fixed iteration
     count (as used by the GAMG/Chebyshev eigen-estimate with a random
     right-hand side) CG can (nearly) converge before the last iteration; the
     subsequent Lanczos coefficients become 0/0 or overflow, producing Inf/NaN
     entries that would otherwise make LAPACKstev fail with "xSTEV error". The
     leading finite block is a valid estimate from the converged Krylov
     subspace. */
  PetscInt nv = 0;
  for (PetscInt j = 0; j < n; j++) {
    PetscReal dj = PetscRealPart(d[j]);
    PetscReal ej = PetscRealPart(e[j + 1]);
    if (PetscIsInfOrNanReal(dj) || PetscIsInfOrNanReal(ej)) break;
    dd[j] = dj;
    ee[j] = ej;
    nv++;
  }
  if (!nv) {
    *emax = *emin = 1.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscBLASIntCast(nv, &bn));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKREALstev", LAPACKREALstev_("N", &bn, dd, ee, NULL, &ldz, NULL, &lierr));
  PetscCall(PetscFPTrapPop());
  /* lierr > 0: algorithm did not converge in 30*N iterations; the extreme
     singular values are only estimates used for Chebyshev smoothing, so use a
     safe fallback rather than crash */
  if (lierr > 0) {
    PetscCall(PetscInfo(ksp, "xSTEV: %" PetscBLASInt_FMT " off-diagonal elements did not converge; returning emin=emax=1\n", lierr));
    *emax = *emin = 1.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(!lierr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "xSTEV error (invalid argument %" PetscBLASInt_FMT ")", lierr);
  *emin = dd[0];
  *emax = dd[nv - 1];
  PetscFunctionReturn(PETSC_SUCCESS);
}
