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
  PetscInt     n  = cgP->ned; /* only rows whose Lanczos diagonal was actually written are valid (see KSPSolve_CG()) */
  PetscInt     nv = 0;
  PetscBLASInt bn, lierr = 0, ldz = 1;

  PetscFunctionBegin;
  PetscCheck(nmax >= n, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_SIZ, "Not enough room in work space r and c for eigenvalues");
  *neig = n;

  PetscCall(PetscArrayzero(c, nmax));
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  d  = cgP->d;
  e  = cgP->e;
  ee = cgP->ee;

  /* Copy the tridiagonal matrix to the work space. cgP->ned already bounds n to
     the Lanczos rows whose diagonal was written (see KSPSolve_CG()); guard
     additionally against a non-finite entry from an overflowing Lanczos
     coefficient so LAPACKstev does not fail with the "xSTEV error". Only the
     off-diagonals e[1..n-1] that LAPACKstev reads are validated -- e[n] is one
     past the last Lanczos step and is never written, so it must not be touched. */
  for (PetscInt j = 0; j < n; j++) {
    PetscReal dj = PetscRealPart(d[j]);
    if (PetscIsInfOrNanReal(dj)) break;
    r[j] = dj;
    nv++;
    if (j < n - 1) {
      PetscReal ej = PetscRealPart(e[j + 1]);
      if (PetscIsInfOrNanReal(ej)) break;
      ee[j] = ej;
    }
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
  PetscInt     n  = cgP->ned; /* only rows whose Lanczos diagonal was actually written are valid (see KSPSolve_CG()) */
  PetscInt     nv = 0;
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

  /* Copy the tridiagonal matrix to the work space. cgP->ned already bounds n to
     the Lanczos rows whose diagonal was written (see KSPSolve_CG()); guard
     additionally against a non-finite entry from an overflowing Lanczos
     coefficient (which can arise with KSP_NORM_NONE and a fixed iteration count,
     as in the GAMG/Chebyshev eigen-estimate) so LAPACKstev does not fail with
     "xSTEV error". Only the off-diagonals e[1..n-1] that LAPACKstev reads are
     validated -- e[n] is one past the last Lanczos step and is never written,
     so it must not be touched. */
  for (PetscInt j = 0; j < n; j++) {
    PetscReal dj = PetscRealPart(d[j]);
    if (PetscIsInfOrNanReal(dj)) break;
    dd[j] = dj;
    nv++;
    if (j < n - 1) {
      PetscReal ej = PetscRealPart(e[j + 1]);
      if (PetscIsInfOrNanReal(ej)) break;
      ee[j] = ej;
    }
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
